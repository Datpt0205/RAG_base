# server/src/features/cv_analysis/handlers/cv_analysis_handler.py

import re, json, base64, os, uuid, tempfile
from typing import Any, Optional, List, Dict
from openai import OpenAI

from src.core.utils.logger.custom_logging import LoggerMixin
from src.core.schemas.cv import (
    CvAnalysis, CvExtract, EducationItem, Entities, RadarItem, RecommendedRole, WorkItem
)
from src.core.providers.provider_factory import ModelProviderFactory, ProviderType
from src.core.schemas.cv_ui import UiCvAnalysis, UiStrength, UiWeakness, UiIndustry, UiRole
from src.features.cv_analysis.cache import cv_hash, load_cache, save_cache
from src.features.cv_analysis.taxonomy import canonicalize, score_domains, learning_link

# ====== Regex ======
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\-\s]{7,}\d)")

# ====== Prompts ======
PROMPT_JSON = """You analyze CV/resume text.

Return STRICT JSON with:
- skills: {{hard: string[], soft: string[]}}
- strengths: string[] (max 6)
- weaknesses: string[] (max 6)
- recommended_roles: {{title: string, confidence: number between 0 and 1}}[] (2-4 items)

- radar_fixed: {{axis: "Backend"|"Data"|"DevOps"|"Frontend"|"AI/ML", score: integer 0..100}}[] 
  # Optional legacy 5-axis radar for compatibility. Include if relevant.

- domain_radar: 
  # REQUIRED. A dynamic radar by job domains (5–7 axes). Choose axes that best match THIS CV.
  # Examples of domains you MAY use (you can add others if strongly justified):
  # ["Helpdesk","IT Support","Networking","SysAdmin","Security","Cloud","Backend","Frontend",
  #  "Mobile","Data","DevOps","ML/AI","QA/Testing","Product","UX/UI","Project Mgmt","Content/Marketing"]
  # Output format: {{axis: string, score: integer 0..100}}[]

- education: [{{school: string, degree?: string, field?: string, start_year?: number, end_year?: number}}]
- experience: [{{company: string, title?: string, start_year?: number, end_year?: number, projects?: string[]}}]
- entities: {{companies: string[], schools: string[], projects: string[]}}

Rules:
- Normalize years to 4-digit integers where possible. If "2019–2021", map start_year=2019, end_year=2021.
- Deduplicate repeated companies, schools, projects.
- For domain_radar, pick the most relevant domains for THIS PERSON; at least 3 axes, typically 5–7.
- Return ONLY a JSON object.
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
{{
  "explanations": ["...", "..."]
}}"""

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

    # ---------- tiny utils to normalize ----------
    def _as_list(self, x):
        if x is None: return []
        if isinstance(x, list): return [str(i) for i in x]
        if isinstance(x, str): return [x]
        if isinstance(x, dict): return [str(v) for v in x.values()]
        return []

    def _sanitize_openai_output(self, data: dict) -> dict:
        """Normalize possibly-bad JSON from LLM to the schema we expect."""
        if not isinstance(data, dict):
            return {
                "skills": {"hard": [], "soft": []},
                "strengths": [], "weaknesses": [],
                "recommended_roles": [],
                "radar_fixed": [], "domain_radar": [],
                "education": [], "experience": [],
                "entities": {"companies": [], "schools": [], "projects": []},
            }

        # ---- skills
        skills = data.get("skills")
        if isinstance(skills, dict):
            skills.setdefault("hard", [])
            skills.setdefault("soft", [])
            skills["hard"] = self._as_list(skills.get("hard"))
            skills["soft"] = self._as_list(skills.get("soft"))
        else:
            skills = {"hard": self._as_list(skills), "soft": []}
        data["skills"] = skills

        # ---- strengths / weaknesses
        data["strengths"] = self._as_list(data.get("strengths"))
        data["weaknesses"] = self._as_list(data.get("weaknesses"))

        # ---- recommended_roles
        rec = data.get("recommended_roles") or []
        if isinstance(rec, dict): rec = [rec]
        fixed_rec = []
        for r in rec:
            if not isinstance(r, dict):
                continue
            title = str(r.get("title") or r.get("role") or "Engineer")
            conf = r.get("confidence")
            try:
                conf = float(conf)
            except Exception:
                conf = 0.7
            fixed_rec.append({"title": title, "confidence": max(0.0, min(1.0, conf))})
        data["recommended_roles"] = fixed_rec

        # ---- radar_fixed (legacy 5 trục)
        radar_fixed = data.get("radar_fixed") or data.get("radar") or []
        if isinstance(radar_fixed, dict):
            radar_fixed = [radar_fixed]
        fixed_5 = []
        for it in radar_fixed:
            if not isinstance(it, dict):
                continue
            axis = str(it.get("axis") or it.get("name") or "General")
            try:
                score = int(it.get("score", 0))
            except Exception:
                score = 0
            fixed_5.append({"axis": axis, "score": max(0, min(100, score))})
        data["radar_fixed"] = fixed_5

        # ---- domain_radar (linh động)
        domain_radar = data.get("domain_radar") or []
        if isinstance(domain_radar, dict):
            domain_radar = [domain_radar]
        dyn = []
        for it in domain_radar:
            if not isinstance(it, dict):
                continue
            axis = str(it.get("axis") or it.get("name") or "").strip()
            if not axis:
                continue
            try:
                score = int(it.get("score", 0))
            except Exception:
                score = 0
            dyn.append({"axis": axis, "score": max(0, min(100, score))})
        data["domain_radar"] = dyn

        # ---- education
        edu = data.get("education") or []
        if isinstance(edu, dict): edu = [edu]
        data["education"] = []
        for e in edu:
            if isinstance(e, dict):
                data["education"].append({
                    "school": (str(e.get("school","")).strip() or None),
                    "degree": (e.get("degree") or None),
                    "field": (e.get("field") or None),
                    "start_year": int(e["start_year"]) if str(e.get("start_year","")).isdigit() else None,
                    "end_year": int(e["end_year"]) if str(e.get("end_year","")).isdigit() else None,
                })

        # ---- experience
        exp = data.get("experience") or []
        if isinstance(exp, dict): exp = [exp]
        data["experience"] = []
        for w in exp:
            if isinstance(w, dict):
                sy = w.get("start_year"); ey = w.get("end_year")
                sy = int(sy) if str(sy).isdigit() else None
                ey = int(ey) if str(ey).isdigit() else None
                pr = w.get("projects") or []
                if isinstance(pr, str): pr = [pr]
                data["experience"].append({
                    "company": (str(w.get("company","")).strip() or None),
                    "title": (w.get("title") or None),
                    "start_year": sy, "end_year": ey,
                    "projects": [str(p) for p in pr if p],
                })

        # ---- entities
        ents = data.get("entities") or {}
        if not isinstance(ents, dict): ents = {}
        for k in ["companies","schools","projects"]:
            v = ents.get(k) or []
            if isinstance(v, str): v = [v]
            ents[k] = [str(s) for s in v if s]
        data["entities"] = ents

        # ---- unify final 'radar' for downstream:
        # ưu tiên domain_radar (linh động). Nếu rỗng => dùng radar_fixed (5 trục). Nếu vẫn rỗng => []
        data["radar"] = data["domain_radar"] if data["domain_radar"] else data["radar_fixed"]

        return data

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
        raw = resp.choices[0].message.content
        # Guard JSON
        try:
            data: Dict[str, Any] = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("OpenAI UI JSON is not an object")
        except Exception as e:
            self.logger.warning(f"[CV UI] Parse failed: {e}; raw={raw[:400]}")
            raise

        strengths = [UiStrength(**s) for s in data.get("strengths", []) if isinstance(s, dict)]
        weaknesses = [UiWeakness(**w) for w in data.get("weaknesses", []) if isinstance(w, dict)]
        industries = [UiIndustry(**i) for i in data.get("industries", []) if isinstance(i, dict)]
        roles = [UiRole(**r) for r in data.get("roles", []) if isinstance(r, dict)]
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
        raw = resp.choices[0].message.content
        # LOG RAW 500 CHỮ ĐẦU
        self.logger.info(f"[CV] OpenAI RAW (first 500): {raw[:500]}")
        try:
            data = json.loads(raw)
        except Exception as e:
            self.logger.exception(f"[CV] OpenAI JSON parse failed: {e}")
            raise
        return data

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

        # 1) Thử dùng OpenAI để dựng payload UI
        if provider_type == ProviderType.OPENAI or str(provider_type).lower() == "openai":
            api_key = ModelProviderFactory._get_api_key(ProviderType.OPENAI)
            if api_key:
                try:
                    return self._generate_ui_with_openai(core, model_name, api_key)
                except Exception as e:
                    self.logger.warning(f"OpenAI UI synthesis failed, fallback heuristics. Error: {e}")

        # 2) Fallback Heuristic (không gọi OpenAI)
        # 2.1 domain scores + radar động theo taxonomy
        hard_sk: List[str] = core.skills.get("hard", []) or []
        domain_scores: List[tuple[str, int]] = score_domains(hard_sk)   # <-- tạo trước, tránh UnboundLocalError
        radar_dynamic: List[RadarItem] = [RadarItem(axis=d, score=s) for d, s in domain_scores]

        # 2.2 strengths: ưu tiên domain_scores; nếu không có thì dùng radar tĩnh của core; cuối cùng là từ hard skills
        strengths: List[UiStrength] = []
        if domain_scores:
            strengths = [UiStrength(skill=d, score=s) for d, s in sorted(domain_scores, key=lambda x: -x[1])[:5]]
        elif core.radar:
            for r in core.radar:
                strengths.append(UiStrength(skill=r.axis, score=int(r.score)))
        elif hard_sk:
            for s in hard_sk[:5]:
                strengths.append(UiStrength(skill=s, score=80))

        # 2.3 weaknesses: từ domain_scores điểm thấp + link học
        weaknesses: List[UiWeakness] = []
        for axis, score in sorted(domain_scores, key=lambda x: x[1])[:3]:
            link = learning_link(axis)           # link theo domain
            if not link:                         # fallback theo 1 skill phổ biến trong domain
                for s in hard_sk:
                    link = learning_link(s)
                    if link:
                        break
            weaknesses.append(
                UiWeakness(
                    skill=axis,
                    gap=max(0, 100 - int(score)),
                    tip=f"Tăng {axis} qua dự án nhỏ & luyện 30–60 phút/ngày",
                    url=link
                )
            )

        # 2.4 industries (demo rules)
        industries: List[UiIndustry] = []
        def add_ind(name, score, why): industries.append(UiIndustry(name=name, score=int(score), rationale=why))
        hs = [s.lower() for s in hard_sk]
        if any(k in hs for k in ["sql","python","etl","airflow","spark"]): add_ind("Fintech", 86, "Phù hợp kỹ năng xử lý dữ liệu/ETL")
        if any(k in hs for k in ["react","next","node","fastapi"]):       add_ind("E-commerce", 79, "Kinh nghiệm web + phân tích hành vi")
        if any(k in hs for k in ["ml","pytorch","nlp","recsys"]):         add_ind("AI/ML", 72, "Nền tảng AI/NLP/RecSys")

        # 2.5 roles từ recommended_roles
        roles: List[UiRole] = []
        for r in core.recommended_roles[:4]:
            roles.append(UiRole(name=r.title, score=int(round(r.confidence * 100)), rationale=None))

        # 2.6 explanations (không gọi OpenAI ở heuristic; để rỗng)
        explanations: List[str] = []

        return UiCvAnalysis(
            strengths=strengths[:5],
            weaknesses=weaknesses[:5],
            industries=industries[:3] or [UiIndustry(name="General Software", score=70)],
            roles=roles or [UiRole(name="Backend Engineer", score=75)],
            explanations=explanations or None,
            radar=radar_dynamic,                    # FE sẽ vẽ radar động
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

        # 2) CACHE theo hash text
        h = cv_hash(text)
        cached = load_cache(h)
        if cached:
            self.logger.info(f"[CV] cache hit {h}")
            data = cached
        else:
            data = self._analyze_text_openai(text, model_name=model_name, api_key=api_key)
            data = self._sanitize_openai_output(data)
            save_cache(h, data)
            self.logger.info(f"[CV] cache saved {h}")

        # 3) Chuẩn hoá skill theo taxonomy
        skills = data.get("skills", {"hard": [], "soft": []})
        skills["hard"] = canonicalize(skills.get("hard", []))
        skills["soft"] = canonicalize(skills.get("soft", []))

        # 4) Build core response (bao gồm fields mới)
        extract = self._extract_contacts(text)
        cv_id = "cva_" + h

        radar = [RadarItem(**r) for r in data.get("radar", [])]
        roles = [RecommendedRole(**r) for r in data.get("recommended_roles", [])]

        education = [EducationItem(**e) for e in data.get("education", [])]
        experience = [WorkItem(**w) for w in data.get("experience", [])]
        entities = Entities(**data.get("entities", {}))

        return CvAnalysis(
            cv_id=cv_id,
            extract=extract,
            skills=skills,
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            recommended_roles=roles,
            radar=radar,
            education=education,
            experience=experience,
            entities=entities,
        )
