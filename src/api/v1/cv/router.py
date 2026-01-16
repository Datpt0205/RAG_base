import io
import os
import re
import json
import csv
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

import numpy as np
from PIL import Image
import cv2
import fitz  # PyMuPDF
import httpx

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from rapidocr_onnxruntime import RapidOCR
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

from src.core.utils.logger.custom_logging import LoggerMixin

logger = LoggerMixin().logger
router = APIRouter()

# =========================
# CONFIG
# =========================
# OCR tuning
OCR_PDF_ZOOM = float(os.getenv("OCR_PDF_ZOOM", "1.3"))           # 2.0 -> 1.2~1.5 reduces time a lot
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "2"))             # limit pages for CV
OCR_MAX_WIDTH = int(os.getenv("OCR_MAX_WIDTH", "1600"))          # resize before OCR
OCR_TEXT_MIN_CHARS = int(os.getenv("OCR_TEXT_MIN_CHARS", "400")) # if PDF has text layer -> skip OCR
OCR_FORCE_OCR = os.getenv("OCR_FORCE_OCR", "0") == "1"
OCR_PREPROCESS_GRAY = os.getenv("OCR_PREPROCESS_GRAY", "1") == "1"

JOBFIT_MODEL_DIR = os.getenv("JOBFIT_MODEL_DIR", "./merged_model")
SKILLS_CSV = os.getenv("JOBFIT_SKILLS_CSV", "./data/mappings/skills.csv")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# =========================
# UTIL: TIMER
# =========================
class StageTimer:
    def __init__(self):
        self._t0 = time.perf_counter()
        self.ms: Dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = round((time.perf_counter() - t) * 1000, 2)

    def total_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 2)

# =========================
# OCR
# =========================
_rapid_ocr = RapidOCR()

def _join_spaced_letters(s: str) -> str:
    """
    Fix kiểu: 'P y t h o n' -> 'Python'
    Chỉ join khi có chuỗi >= 4 chữ cái rời nhau.
    """
    if not s:
        return s

    def repl(m: re.Match) -> str:
        return m.group(0).replace(" ", "")

    # match: "p y t h o n" / "d j a n g o" / "r e a c t"
    return re.sub(r"(?:(?<=\b)[A-Za-z]\s){3,}[A-Za-z](?=\b)", repl, s)

def _parse_jobfit_output(text: str) -> Dict[str, str]:
    m = re.search(r"\*\*Chức danh:\*\*\s*(.*?)\s*\*\*Mô tả:\*\*\s*(.*)", text, flags=re.DOTALL)
    if not m:
        return {"title": "", "description": "", "raw": text.strip()}

    title = (m.group(1) or "").strip()
    desc = (m.group(2) or "").strip()
    desc = re.split(r"<\|end\|>", desc, maxsplit=1)[0].strip()
    return {"title": title, "description": desc, "raw": text.strip()}

def _is_pdf(filename: str, content_type: Optional[str]) -> bool:
    if content_type and "pdf" in content_type.lower():
        return True
    return filename.lower().endswith(".pdf")

def _extract_text_pdf_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts = []
    for page in doc:
        t = (page.get_text("text") or "").strip()
        if t:
            parts.append(t)
    return "\n\n".join(parts).strip()

def _bytes_to_images_pdf(pdf_bytes: bytes, zoom: float, max_pages: int) -> List[np.ndarray]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: List[np.ndarray] = []
    mat = fitz.Matrix(zoom, zoom)

    for idx, page in enumerate(doc, start=1):
        if max_pages and idx > max_pages:
            break
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(np.array(img))
    return images

def _bytes_to_image(img_bytes: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

def _preprocess_for_ocr(rgb_img: np.ndarray) -> np.ndarray:
    h, w = rgb_img.shape[:2]
    if OCR_MAX_WIDTH and w > OCR_MAX_WIDTH:
        scale = OCR_MAX_WIDTH / float(w)
        rgb_img = cv2.resize(
            rgb_img, (OCR_MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA
        )

    if OCR_PREPROCESS_GRAY:
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        rgb_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    return rgb_img

def _sort_ocr_lines(ocr_result: List[Tuple[Any, str, float]], y_tol: int = 12) -> List[str]:
    items = []
    for dt_box, text, score in ocr_result:
        pts = np.array(dt_box)
        x_min = int(np.min(pts[:, 0]))
        y_min = int(np.min(pts[:, 1]))
        t = (text or "").strip()
        if t:
            items.append((y_min, x_min, t, float(score)))

    items.sort(key=lambda t: (t[0], t[1]))

    lines: List[List[Tuple[int, int, str, float]]] = []
    for y, x, text, score in items:
        if not lines:
            lines.append([(y, x, text, score)])
            continue
        last_y = lines[-1][0][0]
        if abs(y - last_y) <= y_tol:
            lines[-1].append((y, x, text, score))
        else:
            lines.append([(y, x, text, score)])

    out = []
    for line in lines:
        line.sort(key=lambda t: t[1])
        out.append(" ".join(t[2] for t in line))
    return out

def _rapidocr_on_rgb(rgb_img: np.ndarray) -> Tuple[str, float]:
    rgb_img = _preprocess_for_ocr(rgb_img)
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    ocr_result, _ = _rapid_ocr(bgr)
    if not ocr_result:
        return "", 0.0

    lines = _sort_ocr_lines(ocr_result)
    scores = [float(x[2]) for x in ocr_result]
    avg = float(sum(scores) / max(len(scores), 1))
    return "\n".join(lines).strip(), avg

def _warmup_ocr():
    try:
        dummy = np.zeros((512, 512, 3), dtype=np.uint8)
        _rapidocr_on_rgb(dummy)
        logger.info("[init] OCR warmup done")
    except Exception as e:
        logger.warning(f"[init] OCR warmup failed: {e}")

# =========================
# JOBFIT (LoRA merged model)
# =========================
_jobfit_tokenizer = None
_jobfit_model = None

class StopOnSubstrings(StoppingCriteria):
    def __init__(self, tokenizer, stop_strs: List[str]):
        self.tokenizer = tokenizer
        self.stop_strs = stop_strs

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        return any(s in text for s in self.stop_strs)

def _lazy_load_jobfit():
    global _jobfit_tokenizer, _jobfit_model
    if _jobfit_tokenizer is not None and _jobfit_model is not None:
        return

    if not os.path.isdir(JOBFIT_MODEL_DIR):
        raise RuntimeError(f"JOBFIT_MODEL_DIR not found: {JOBFIT_MODEL_DIR}")

    _jobfit_tokenizer = AutoTokenizer.from_pretrained(JOBFIT_MODEL_DIR, trust_remote_code=True)
    _jobfit_model = AutoModelForCausalLM.from_pretrained(
        JOBFIT_MODEL_DIR,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    _jobfit_model.eval()

    if _jobfit_tokenizer.pad_token_id is None:
        _jobfit_tokenizer.pad_token = _jobfit_tokenizer.eos_token

def _build_jobfit_prompt(
    skills: str,
    salary: str = "Không rõ",
    benefits: str = "Không rõ",
    industries: str = "Không rõ",
) -> str:
    return (
        f"<|user|> Dựa trên các kỹ năng sau đây, hãy gợi ý một công việc phù hợp:\n"
        f"- Kỹ năng: {skills}\n"
        f"- Lương: {salary}\n"
        f"- Phúc lợi: {benefits}\n"
        f"- Ngành nghề: {industries}<|end|>\n"
        f"<|assistant|>"
    )

def _jobfit_predict_title_fast(prompt: str, max_new_tokens: int = 48) -> Dict[str, Any]:
    _lazy_load_jobfit()
    import torch

    tm = StageTimer()

    with tm.stage("tokenize"):
        inputs = _jobfit_tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(_jobfit_model.device) for k, v in inputs.items()}

    stops = StoppingCriteriaList([
        StopOnSubstrings(_jobfit_tokenizer, stop_strs=["\n", "<|end|>"])
    ])

    with tm.stage("generate_title_only"):
        with torch.inference_mode():
            out = _jobfit_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy for speed/stability
                pad_token_id=_jobfit_tokenizer.eos_token_id,
                stopping_criteria=stops,
            )

    with tm.stage("decode"):
        decoded = _jobfit_tokenizer.decode(out[0], skip_special_tokens=False)

    tail = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()

    # Title parsing (stop at markers/newline)
    title = tail
    title = title.split("**Description:**")[0]
    title = title.split("<|end|>")[0]
    title = title.split("\n")[0]
    title = title.strip(" -*:").strip()

    return {
        "title": title,
        "raw": tail.strip(),
        "device": str(_jobfit_model.device),
        "timing_ms": {**tm.ms, "total": tm.total_ms()},
        "prompt_preview": prompt[:400],
    }

def _warmup_jobfit():
    try:
        _lazy_load_jobfit()
        prompt = _build_jobfit_prompt(skills="python, sql")
        _ = _jobfit_predict_title_fast(prompt, max_new_tokens=8)
        logger.info("[init] JobFit warmup done")
    except Exception as e:
        logger.warning(f"[init] JobFit warmup failed: {e}")

# =========================
# SKILL EXTRACTION
# =========================
_skill_vocab: List[str] = []

def _load_skill_vocab():
    global _skill_vocab
    if _skill_vocab:
        return

    if not os.path.isfile(SKILLS_CSV):
        _skill_vocab = ["python", "sql", "fastapi", "docker", "aws", "react", "next.js", "postgresql"]
        return

    skills = set()
    with open(SKILLS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("skill_name") or "").strip()
            if name:
                skills.add(name)

    _skill_vocab = sorted(skills, key=lambda s: len(s), reverse=True)

def _normalize_text(s: str) -> str:
    s = (s or "").lower().replace("\n", " ")
    s = re.sub(r"[^\w\s\+\#\.\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _extract_skills_from_raw_text(raw_text: str, limit: int = 30) -> List[str]:
    _load_skill_vocab()
    txt = _normalize_text(raw_text)

    found: List[str] = []
    used = set()

    for skill in _skill_vocab:
        if len(found) >= limit:
            break
        sk = (skill or "").strip()
        if not sk:
            continue

        sk_norm = _normalize_text(sk)
        if not sk_norm or sk_norm in used:
            continue

        if " " in sk_norm:
            if sk_norm in txt:
                found.append(sk)
                used.add(sk_norm)
        else:
            if re.search(rf"(?<!\w){re.escape(sk_norm)}(?!\w)", txt):
                found.append(sk)
                used.add(sk_norm)

    return found

# =========================
# OPENAI: DESCRIPTION + ANALYZE
# =========================
async def _openai_generate_rationale(raw_text: str, job_title: str, skills: List[str]) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY")

    raw_text = (raw_text or "").strip()
    # giữ đủ dài để đúng ý bạn, nhưng vẫn có guard token
    if len(raw_text) > 12000:
        raw_text = raw_text[:9000] + "\n...\n" + raw_text[-2500:]

    skills_str = ", ".join(skills[:25]) if skills else "Unknown"

    instructions = (
        "You are a career advisor.\n"
        "Task: Explain briefly why the recommended job title fits the candidate's CV.\n"
        "OUTPUT RULES:\n"
        "- English only\n"
        "- 2 to 4 short sentences (no long bullets)\n"
        "- Do NOT write a job description\n"
        "- Ground your reasoning in the CV text; do not invent experiences\n"
        "- Plain text only"
    )

    user_input = (
        f"Recommended job title (from internal model): {job_title}\n"
        f"Detected skills: {skills_str}\n\n"
        f"CV raw text:\n\"\"\"\n{raw_text}\n\"\"\""
    )

    payload = {
        "model": OPENAI_MODEL,
        "instructions": instructions,
        "input": user_input,
        "max_output_tokens": 180,
    }

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post("https://api.openai.com/v1/responses", headers=headers, json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")

        data = r.json()
        out_text = ""
        for item in data.get("output", []) or []:
            for c in item.get("content", []) or []:
                if c.get("type") in ("output_text", "text"):
                    out_text += c.get("text", "")
        return (out_text or "").strip()
# async def _openai_generate_rationale(raw_text: str, job_title: str, skills: List[str]) -> str:
#     if not OPENAI_API_KEY:
#         raise RuntimeError("Missing OPENAI_API_KEY")

#     raw_text = (raw_text or "").strip()[:4500]
#     skills_str = ", ".join(skills[:20]) if skills else "Không rõ"

#     instructions = (
#         "Bạn là chuyên gia tư vấn nghề nghiệp.\n"
#         "Nhiệm vụ: giải thích NGẮN GỌN vì sao 'Job Title' phù hợp với CV.\n"
#         "CẤM TUYỆT ĐỐI:\n"
#         "- Không viết Job Description (JD)\n"
#         "- Không viết headings kiểu 'Job Summary/Responsibilities/Requirements'\n"
#         "- Không bullet dài, không liệt kê nhiệm vụ công việc\n\n"
#         "OUTPUT:\n"
#         "- 2 đến 4 câu tiếng Việt\n"
#         "- Nêu 2-3 điểm match cụ thể dựa trên kỹ năng/kinh nghiệm trong CV\n"
#         "- Plain text בלבד (không JSON)."
#     )

#     user_input = (
#         f"Job Title (đã recommend): {job_title}\n"
#         f"Kỹ năng phát hiện: {skills_str}\n\n"
#         f"CV (raw text):\n\"\"\"\n{raw_text}\n\"\"\"\n"
#     )

#     payload = {
#         "model": OPENAI_MODEL,
#         "instructions": instructions,
#         "input": user_input,
#         "max_output_tokens": 220,
#     }

#     headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

#     async with httpx.AsyncClient(timeout=45) as client:
#         r = await client.post("https://api.openai.com/v1/responses", headers=headers, json=payload)
#         if r.status_code >= 400:
#             raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")

#         data = r.json()
#         out_text = ""
#         for item in data.get("output", []) or []:
#             for c in item.get("content", []) or []:
#                 if c.get("type") in ("output_text", "text"):
#                     out_text += c.get("text", "")
#         return (out_text or "").strip()

def _extract_json_object(s: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in OpenAI output")
    return json.loads(m.group(0))

async def _openai_analyze_cv(raw_text: str, rec_title: str, rec_desc: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY")

    instructions = (
        "You are an expert CV analyst.\n"
        "Return ONLY a valid JSON object with this schema:\n"
        "{\n"
        '  "strengths": [{"skill":"...","score":0-100,"note":"..."}],\n'
        '  "weaknesses": [{"skill":"...","gap":0-100,"tip":"...","url":"..."}],\n'
        '  "industries": [{"name":"...","score":0-100,"rationale":"..."}],\n'
        '  "roles": [{"name":"...","score":0-100,"rationale":"..."}],\n'
        '  "explanations": ["..."],\n'
        '  "radar": [{"axis":"Backend|Data|DevOps|Frontend|AI/ML","score":0-100}]\n'
        "}\n"
        "Do not include any extra text outside the JSON."
    )

    user_input = (
        "Here is the CV raw text (OCR):\n"
        f"\"\"\"\n{raw_text}\n\"\"\"\n\n"
        "Internal job recommendation (to tailor analysis):\n"
        f"- Title: {rec_title}\n"
        f"- Description: {rec_desc}\n"
    )

    payload = {
        "model": OPENAI_MODEL,
        "instructions": instructions,
        "input": user_input,
        "max_output_tokens": 900,
    }

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post("https://api.openai.com/v1/responses", headers=headers, json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")

        data = r.json()
        out_text = ""
        for item in data.get("output", []) or []:
            for c in item.get("content", []) or []:
                if c.get("type") in ("output_text", "text"):
                    out_text += c.get("text", "")
        out_text = out_text.strip()

        return _extract_json_object(out_text)

# =========================
# INIT (call on FastAPI startup)
# =========================
def init_resources():
    _load_skill_vocab()
    _lazy_load_jobfit()
    _warmup_ocr()
    _warmup_jobfit()
    logger.info("[init] resources initialized")

# =========================
# ENDPOINTS
# =========================
@router.post("/ocr")
async def ocr_cv(
    file: UploadFile = File(...),
    provider_type: str = Form("openai"),        # kept for FE compatibility
    model_name: str = Form("gpt-4o-mini"),      # kept for FE compatibility
):
    tm = StageTimer()
    req_id = uuid.uuid4().hex[:8]

    with tm.stage("read_upload"):
        content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        is_pdf = _is_pdf(file.filename, file.content_type)

        # FAST PATH: PDF has text layer
        if is_pdf and not OCR_FORCE_OCR:
            with tm.stage("pdf_extract_text"):
                extracted = _extract_text_pdf_bytes(content)
                extracted = _join_spaced_letters(extracted)
                logger.info("[ocr][%s] pdf text layer chars=%s", req_id, extracted)

            if extracted and len(extracted) >= OCR_TEXT_MIN_CHARS:
                total = tm.total_ms()
                logger.info("[ocr][%s] mode=pdf_text_extract bytes=%d timing_ms=%s total_ms=%.2f",
                            req_id, len(content), tm.ms, total)
                return {
                    "text": extracted,
                    "skills": [],
                    "confidence": 1.0,
                    "_timing_ms": {**tm.ms, "total": total},
                    "_req_id": req_id,
                    "_mode": "pdf_text_extract",
                }

        # OCR PATH
        if is_pdf:
            with tm.stage("pdf_to_images"):
                pages = _bytes_to_images_pdf(content, zoom=OCR_PDF_ZOOM, max_pages=OCR_MAX_PAGES)

            texts, scores = [], []
            with tm.stage("ocr_pages_total"):
                for i, img in enumerate(pages, start=1):
                    t, s = _rapidocr_on_rgb(img)
                    if t:
                        texts.append(f"--- Page {i} ---\n{t}")
                        scores.append(s)

            with tm.stage("postprocess_join"):
                full_text = "\n\n".join(texts).strip()
                full_text = _join_spaced_letters(full_text)
                avg_conf = float(sum(scores) / max(len(scores), 1)) if scores else 0.0
        else:
            with tm.stage("bytes_to_image"):
                img = _bytes_to_image(content)

            with tm.stage("ocr_image"):
                full_text, avg_conf = _rapidocr_on_rgb(img)
                full_text = _join_spaced_letters(full_text)

        total = tm.total_ms()
        logger.info("[ocr][%s] mode=ocr file=%s pdf=%s bytes=%d timing_ms=%s total_ms=%.2f",
                    req_id, file.filename, is_pdf, len(content), tm.ms, total)

        return {
            "text": full_text,
            "skills": [],
            "confidence": avg_conf,
            "_timing_ms": {**tm.ms, "total": total},
            "_req_id": req_id,
            "_mode": "ocr",
        }

    except Exception as e:
        total = tm.total_ms()
        logger.exception("[ocr][%s] failed total_ms=%.2f timing_ms=%s err=%s", req_id, total, tm.ms, e)
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

@router.post("/recommend")
async def recommend_jobfit(raw_text: str = Form("")):
    tm = StageTimer()
    req_id = uuid.uuid4().hex[:8]

    raw_text = (raw_text or "").strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="raw_text is required")

    try:
        _lazy_load_jobfit()

        with tm.stage("extract_skills"):
            skills = _extract_skills_from_raw_text(raw_text, limit=30)
            logger.info("[recommend][%s] skills_found=%d skills=%s", req_id, len(skills), skills[:30])
            skills_str = ", ".join(skills) if skills else "Không rõ"

        with tm.stage("build_prompt"):
            prompt = _build_jobfit_prompt(
                skills=skills_str,
                salary="Không rõ",
                benefits="Không rõ",
                industries="Không rõ",
            )

        with tm.stage("tokenize"):
            inputs = _jobfit_tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(_jobfit_model.device) for k, v in inputs.items()}

        import torch
        with tm.stage("jobfit_generate_full"):
            with torch.inference_mode():
                out = _jobfit_model.generate(
                    **inputs,
                    max_new_tokens=220,
                    do_sample=True,          # giữ như bạn muốn (có thể đổi False để ổn định)
                    temperature=0.2,
                    top_p=0.9,
                    pad_token_id=_jobfit_tokenizer.eos_token_id,
                )

        with tm.stage("decode"):
            decoded = _jobfit_tokenizer.decode(out[0], skip_special_tokens=False)

        with tm.stage("parse"):
            tail = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()
            parsed = _parse_jobfit_output(tail)

            job_title = (parsed.get("title") or "").strip() or "Suggested Role"
            jobfit_desc = (parsed.get("description") or "").strip()
            jobfit_raw = parsed.get("raw", tail)
            logger.info("[recommend][%s] jobfit_title=%s jobfit_desc=%s", req_id, job_title, jobfit_desc)

        # OpenAI: chỉ giải thích lý do (không JD)
        with tm.stage("openai_rationale"):
            rationale = await _openai_generate_rationale(raw_text, job_title, skills)
            logger.info("[recommend][%s] openai_rationale_chars=%s", req_id, rationale)

        total = tm.total_ms()
        logger.info(
            "[recommend][%s] timing_ms=%s total_ms=%.2f device=%s title=%s",
            req_id, tm.ms, total, str(_jobfit_model.device), job_title
        )

        return {
            # UI sẽ hiển thị 2 field này
            "title": job_title,                 # LoRA recommend
            "description": rationale,           # OpenAI rationale (NOT JD)
            "skills_used": skills,
            "raw": jobfit_raw,                  # raw LoRA output
            "jobfit_description": jobfit_desc,  # mô tả job (nếu model có) để bạn đánh giá, nhưng UI không dùng
            "_timing_ms": {**tm.ms, "total": total},
            "_req_id": req_id,
            "_source": {"title": "jobfit_lora", "description": "openai_rationale"},
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommend failed: {e}")

@router.post("/analyze-ui")
async def analyze_ui(
    raw_text: str = Form(""),
    recommend_title: str = Form(""),
    recommend_description: str = Form(""),
    provider_type: str = Form("openai"),
    model_name: str = Form("gpt-4o-mini"),
):
    raw_text = (raw_text or "").strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="raw_text is required")

    if provider_type != "openai":
        raise HTTPException(status_code=400, detail="Only provider_type=openai is supported for analyze-ui")

    try:
        global OPENAI_MODEL
        if model_name:
            OPENAI_MODEL = model_name

        data = await _openai_analyze_cv(raw_text, recommend_title, recommend_description)
        return {"status": "success", "message": "ok", "data": data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analyze failed: {e}")
