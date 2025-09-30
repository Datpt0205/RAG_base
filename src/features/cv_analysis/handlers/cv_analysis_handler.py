# server/src/features/cv_analysis/handlers/cv_analysis_handler.py

import re, json, base64, os, uuid, tempfile
from typing import Any, Optional, List, Dict
from openai import OpenAI

from src.core.utils.logger.custom_logging import LoggerMixin
from src.core.schemas.cv import (
    CvAnalysis, CvExtract, RadarItem, RecommendedRole
)
from src.core.providers.provider_factory import ModelProviderFactory, ProviderType
from src.core.schemas.cv_ui import UiCvAnalysis, UiStrength, UiWeakness, UiIndustry, UiRole

# ====== Regex ======
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\-\s]{7,}\d)")

# ====== Prompts ======
PROMPT_JSON = """You analyze CV/resume text.

Return STRICT JSON with:
- skills: {hard: string[], soft: string[]}
- strengths: string[] (max 6)
- weaknesses: string[] (max 6)
- recommended_roles: {title: string, confidence: number between 0 and 1}[] (2-4 items)
- radar: {axis: string, score: integer 0..100}[] for axes: Backend, Data, DevOps, Frontend, AI/ML

Rules:
- Normalize skill names (e.g., 'js' -> 'JavaScript', 'postgres' -> 'PostgreSQL').
- Confidence reflects suitability given experience and skills.
- Radar must have exactly 5 axes listed above.
Return ONLY JSON object.
Text:
```{text}```"""

EXPLAIN_PROMPT = """Bạn là chuyên gia tuyển dụng. Dựa vào dữ liệu phân tích CV dưới đây, hãy giải thích ngắn gọn vì sao các gợi ý là hợp lý.

DỮ LIỆU:
- Hard skills: {hard}
- Soft skills: {soft}
- Radar (0-100): {radar}
- Recommended roles: {roles}
- Weaknesses (nếu có): {weaknesses}

YÊU CẦU:
- Viết 2–5 gạch đầu dòng, ngắn gọn (tối đa 25 từ/mục), ưu tiên tiếng Việt.
- Dẫn chiếu rõ ràng tới kỹ năng, điểm mạnh/yếu, và vai trò gợi ý (ví dụ: "SQL + ETL mạnh → phù hợp Data Engineer").
- Không bịa đặt dữ kiện không có trong dữ liệu.
- Trả về JSON object dạng:
{
  "explanations": ["...", "..."]
}"""

VISION_PROMPT = """Read this resume/CV image carefully and transcribe all visible text.
Return ONLY the plain text (no explanations, no JSON)."""

_UI_PROMPT = """Bạn là chuyên gia tuyển dụng.
Từ dữ liệu phân tích CV (JSON) bên dưới, hãy tổng hợp sang cấu trúc UI sau và trả về JSON THUẦN:

{
  "strengths": [{"skill": "string", "score": 0-100}],
  "weaknesses": [{"skill": "string", "gap": 0-100, "tip": "string (optional)"}],
  "industries": [{"name": "string", "score": 0-100, "rationale": "string (optional)"}],
  "roles": [{"name": "string", "score": 0-100, "rationale": "string (optional)"}],
  "explanations": ["string", "..."]
}

QUY TẮC:
- strengths: chọn 3-6 kỹ năng nổi trội. Nếu trùng trục radar thì tham chiếu điểm radar; nếu không có, ước lượng dựa trên ngữ cảnh kỹ năng/kinh nghiệm, nhưng phải hợp lý.
- weaknesses: 2-4 mục. 'gap' ≈ khoảng cách tới 100 (ưu tiên những trục radar/thành phần yếu). Thêm tip ngắn, thực tế.
- industries: tối đa 3 ngành. Điểm 0-100 dựa vào mức phù hợp tổng thể (skills + radar + kinh nghiệm).
- roles: dựa vào recommended_roles trong dữ liệu. 'score' ≈ confidence*100 có tinh chỉnh nhẹ theo radar/skills. Thêm rationale ngắn.
- explanations: 2-5 gạch đầu dòng, nêu rõ vì sao các gợi ý hợp lý. Không bịa dữ kiện.

Trả JSON duy nhất, không thêm chữ nào khác.
DỮ LIỆU:
"""

class CvAnalysisHandler(LoggerMixin):
    def __init__(self):
        super().__init__()

    # ---------- OpenAI helpers ----------
    def _openai_client(self, api_key: str) -> OpenAI:
        return OpenAI(api_key=api_key)

    # ---------- Explanations by OpenAI ----------
    def _generate_explanations_openai(self, core: CvAnalysis, model_name: str, api_key: str) -> List[str]:
        client = self._openai_client(api_key)
        hard = core.skills.get("hard", [])
        soft = core.skills.get("soft", [])
        radar_pairs = [(r.axis, r.score) for r in (core.radar or [])]
        roles_pairs = [(r.title, r.confidence) for r in (core.recommended_roles or [])]
        weaknesses = getattr(core, "weaknesses", [])

        prompt = EXPLAIN_PROMPT.format(
            hard=hard, soft=soft, radar=radar_pairs, roles=roles_pairs, weaknesses=weaknesses
        )
        resp = client.chat.completions.create(
            model=model_name,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            data = json.loads(resp.choices[0].message.content)
            exps = data.get("explanations", [])
            return [str(x) for x in exps if isinstance(x, (str, int, float))]
        except Exception:
            return []

    # ---------- UI synthesis by OpenAI ----------
    def _generate_ui_with_openai(self, core: CvAnalysis, model_name: str, api_key: str) -> UiCvAnalysis:
        client = self._openai_client(api_key)
        core_json = core.model_dump()

        resp = client.chat.completions.create(
            model=model_name,                # gpt-4o-mini / gpt-4o
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": _UI_PROMPT + json.dumps(core_json, ensure_ascii=False)
            }]
        )
        data: Dict[str, Any] = json.loads(resp.choices[0].message.content)

        strengths = [UiStrength(**s) for s in data.get("strengths", [])]
        weaknesses = [UiWeakness(**w) for w in data.get("weaknesses", [])]
        industries = [UiIndustry(**i) for i in data.get("industries", [])]
        roles = [UiRole(**r) for r in data.get("roles", [])]
        explanations = [str(x) for x in data.get("explanations", [])] or None

        return UiCvAnalysis(
            strengths=strengths,
            weaknesses=weaknesses,
            industries=industries,
            roles=roles,
            explanations=explanations
        )

    # ---------- local helpers ----------
    def _extract_contacts(self, text: str) -> CvExtract:
        emails = list(dict.fromkeys(EMAIL_RE.findall(text or "")))
        phones = list(dict.fromkeys([re.sub(r"\s+", " ", p).strip() for p in PHONE_RE.findall(text or "")]))
        return CvExtract(full_text=text[:50000], emails=emails, phones=phones, years_experience=None)

    def _b64(self, content: bytes) -> str:
        return base64.b64encode(content).decode("utf-8")

    def _analyze_text_openai(self, text: str, model_name: str, api_key: str) -> dict:
        client = self._openai_client(api_key)
        resp = client.chat.completions.create(
            model=model_name,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": PROMPT_JSON.format(text=text)}],
        )
        return json.loads(resp.choices[0].message.content)

    def _ocr_with_openai(self, image_bytes: bytes, model_name: str, api_key: str) -> str:
        client = self._openai_client(api_key)
        b64 = self._b64(image_bytes)
        resp = client.chat.completions.create(
            model=model_name,  # gpt-4o / gpt-4o-mini
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]
            }],
        )
        return resp.choices[0].message.content.strip()

    def _pdf_to_text(self, pdf_bytes: bytes) -> str:
        try:
            import fitz  # from PyMuPDF
        except Exception:
            return ""
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as f:
            f.write(pdf_bytes); f.flush()
            doc = fitz.open(f.name)
            text = "\n".join([page.get_text() for page in doc])
            doc.close()
            return text

    # ---------- public: convert core -> UI (prefer OpenAI; fallback heuristic) ----------
    def to_ui_payload(
        self,
        core: CvAnalysis,
        provider_type: str = "openai",
        model_name: str = "gpt-4o-mini"
    ) -> UiCvAnalysis:

        if provider_type == ProviderType.OPENAI:
            api_key = ModelProviderFactory._get_api_key(provider_type)
            if api_key:
                try:
                    return self._generate_ui_with_openai(core, model_name, api_key)
                except Exception as e:
                    self.logger.warning(f"OpenAI UI synthesis failed, fallback heuristics. Error: {e}")

        # ===== Fallback Heuristic (giữ để endpoint luôn có dữ liệu) =====
        radar_map = {r.axis: r.score for r in (core.radar or [])}

        strengths: List[UiStrength] = []
        if radar_map:
            for axis, score in radar_map.items():
                strengths.append(UiStrength(skill=axis, score=int(score)))
        elif core.skills.get("hard"):
            for s in core.skills["hard"][:5]:
                strengths.append(UiStrength(skill=s, score=80))

        weaknesses: List[UiWeakness] = []
        if radar_map:
            for axis, score in sorted(radar_map.items(), key=lambda x: x[1])[:3]:
                weaknesses.append(UiWeakness(
                    skill=axis, gap=max(0, 100 - int(score)),
                    tip=f"Tăng {axis} qua dự án nhỏ & luyện 30–60 phút/ngày"
                ))
        elif core.weaknesses:
            for w in core.weaknesses[:3]:
                weaknesses.append(UiWeakness(skill=w, gap=25, tip="Bổ sung kiến thức nền + thực hành mini-project"))

        industries: List[UiIndustry] = []
        def add_ind(name, score, why): industries.append(UiIndustry(name=name, score=int(score), rationale=why))
        hs = [s.lower() for s in core.skills.get("hard", [])]
        if any(k in hs for k in ["sql","python","etl","airflow","spark"]): add_ind("Fintech", 86, "Phù hợp kỹ năng xử lý dữ liệu/ETL")
        if any(k in hs for k in ["react","next","node","fastapi"]): add_ind("E-commerce", 79, "Kinh nghiệm web + phân tích hành vi")
        if any(k in hs for k in ["ml","pytorch","nlp","recsys"]): add_ind("AI/ML", 72, "Nền tảng AI/NLP/RecSys")

        roles: List[UiRole] = []
        for r in core.recommended_roles[:4]:
            roles.append(UiRole(name=r.title, score=int(round(r.confidence*100)), rationale=None))

        explanations: List[str] = []
        if provider_type == ProviderType.OPENAI:
            api_key = ModelProviderFactory._get_api_key(provider_type)
            if api_key:
                explanations = self._generate_explanations_openai(core, model_name, api_key)

        return UiCvAnalysis(
            strengths=strengths[:5],
            weaknesses=weaknesses[:5],
            industries=industries[:3] or [UiIndustry(name="General Software", score=70)],
            roles=roles or [UiRole(name="Backend Engineer", score=75)],
            explanations=explanations or None
        )

    # ---------- public: main analyze ----------
    def analyze(
        self, *,
        provider_type: str,
        model_name: str,
        raw_text: Optional[str],
        file_bytes: Optional[bytes],
        file_content_type: Optional[str]
    ) -> CvAnalysis:
        # Hiện chỉ support OpenAI
        api_key = ModelProviderFactory._get_api_key(provider_type)
        if provider_type != ProviderType.OPENAI:
            raise ValueError("Currently only 'openai' is supported for CV analysis")

        # 1) Lấy text
        text = raw_text or ""
        if not text and file_bytes:
            if file_content_type and "pdf" in file_content_type:
                text = self._pdf_to_text(file_bytes)
            if not text:
                text = self._ocr_with_openai(file_bytes, model_name=model_name, api_key=api_key)

        if not text or len(text.strip()) < 20:
            raise ValueError("CV text is empty or too short")

        # 2) Gọi LLM phân tích cốt lõi
        data = self._analyze_text_openai(text, model_name=model_name, api_key=api_key)

        # 3) Build core response
        extract = self._extract_contacts(text)
        cv_id = "cva_" + uuid.uuid4().hex[:10]

        radar = [RadarItem(**r) for r in data.get("radar", [])]
        roles = [RecommendedRole(**r) for r in data.get("recommended_roles", [])]

        return CvAnalysis(
            cv_id=cv_id,
            extract=extract,
            skills=data.get("skills", {"hard": [], "soft": []}),
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            recommended_roles=roles,
            radar=radar,
        )
