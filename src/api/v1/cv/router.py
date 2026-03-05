# jobfit_v2_router.py
import io
import os
import re
import json
import csv
import time
import uuid
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

import numpy as np
from PIL import Image
import cv2
import fitz  # PyMuPDF
import httpx

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from rapidocr_onnxruntime import RapidOCR

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Optional Qdrant filter models (version-dependent)
try:
    from qdrant_client.models import Filter, FieldCondition, MatchValue, IsEmptyCondition, IsNullCondition
    _HAS_QDRANT_FILTER_MODELS = True
except Exception:
    Filter = FieldCondition = MatchValue = IsEmptyCondition = IsNullCondition = None  # type: ignore
    _HAS_QDRANT_FILTER_MODELS = False

# ===========
# Logger (giữ kiểu bạn)
# ===========
from src.core.utils.logger.custom_logging import LoggerMixin
logger = LoggerMixin().logger

router = APIRouter()

# =========================
# CONFIG
# =========================
OCR_PDF_ZOOM = float(os.getenv("OCR_PDF_ZOOM", "1.3"))
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "2"))
OCR_MAX_WIDTH = int(os.getenv("OCR_MAX_WIDTH", "1600"))
OCR_TEXT_MIN_CHARS = int(os.getenv("OCR_TEXT_MIN_CHARS", "400"))
OCR_FORCE_OCR = os.getenv("OCR_FORCE_OCR", "0") == "1"
OCR_PREPROCESS_GRAY = os.getenv("OCR_PREPROCESS_GRAY", "1") == "1"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6334")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "jobs")
QDRANT_EMBED_MODEL = os.getenv("QDRANT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- JobFit rubric v2 (aligns with your new training pipeline) ---
JOBFIT_SENIORITY_FIELD = os.getenv("JOBFIT_SENIORITY_FIELD", "formatted_experience_level")
JOBFIT_REQUIRE_SENIORITY = os.getenv("JOBFIT_REQUIRE_SENIORITY", "0") == "1"
JOBFIT_USE_SENIORITY_FILTER = os.getenv("JOBFIT_USE_SENIORITY_FILTER", "0") == "1"

RERANK_MODEL_DIR = os.getenv(
    "RERANK_MODEL_DIR",
    "./merged_qwen2p5_3b_jobfit"
)

RERANK_DEFAULT_FIELDS_MODE = os.getenv("RERANK_FIELDS_MODE", "full") 

RERANK_BATCH = int(os.getenv("RERANK_BATCH", "8"))
RERANK_MAX_NEW_TOKENS = int(os.getenv("RERANK_MAX_NEW_TOKENS", "48"))

# Local cache (single-user) – no DB needed
JOBFIT_CACHE_FILE = os.getenv("JOBFIT_CACHE_FILE", "./.jobfit_recommend_cache.json")

# =========================
# UTIL: TIMER
# =========================

def init_resources():
    """
    Optional preload for heavy resources.
    Call this at startup if you want to warm up models.
    """
    try:
        logger.info("Initializing reranker model...")
        _lazy_load_reranker()

        logger.info("Initializing Qdrant client + encoder...")
        _lazy_load_qdrant()

        logger.info("Resources initialized successfully")
    except Exception as e:
        logger.exception("init_resources failed: %s", e)
        raise
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
        rgb_img = cv2.resize(rgb_img, (OCR_MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)

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

# =========================
# OpenAI: Extract Candidate Profile from CV (reduce noise)
# =========================
def _extract_json_object(s: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in OpenAI output")
    return json.loads(m.group(0))

async def _openai_extract_candidate_profile(raw_text: str) -> Dict[str, Any]:
    """
    Extract: skills, industry background, maybe years exp (optional), plus a short normalized skill list.
    Salary/benefit/employee preference will be provided by user from UI, not extracted.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY")

    raw_text = (raw_text or "").strip()
    if len(raw_text) > 14000:
        raw_text = raw_text[:10000] + "\n...\n" + raw_text[-3000:]

    instructions = (
        "You are an expert CV parser.\n"
        "Return ONLY a valid JSON object with schema:\n"
        "{\n"
        '  "name": "candidate full name or Unknown",\n'
        '  "email": "email address or Unknown",\n'
        '  "phone": "phone number or Unknown",\n'
        '  "desired_title": "the most fitting job title for this candidate based on their experience and skills, e.g. Backend Engineer, Data Scientist, Product Manager. If unclear, return Unknown",\n'
        '  "skills_text": "comma separated skills",\n'
        '  "skills_list": ["..."],\n'
        '  "industry_background": "comma separated industries/domains the candidate has worked in",\n'
        '  "seniority_hint": "intern|junior|mid|senior|lead",\n'
        '  "notes": "very short"\n'
        "}\n"
        "Rules:\n"
        "- Extract name, email, phone from CV header/contact section\n"
        "- NAME EXTRACTION — CRITICAL RULES:\n"
        "  * The candidate name is typically displayed prominently (large font) at the top of the CV, often as a heading or title\n"
        "  * DO NOT confuse street names, road names, or addresses with the candidate's name\n"
        "  * Vietnamese street names to watch out for: 'Ben Van Don', 'Nguyen Hue', 'Le Loi', 'Tran Hung Dao', 'Vo Van Tan', 'Pham Ngu Lao', 'Hai Ba Trung', 'Le Duan', 'Nguyen Trai', etc.\n"
        "  * If you see a number prefix (e.g. '254/33/111 ...'), followed by a Vietnamese-looking name, that is an ADDRESS, not a person's name\n"
        "  * Text containing 'Ward', 'District', 'City', 'Street', 'Road', 'Quan', 'Phuong', 'TP' near a name indicates it is an address\n"
        "  * The person's name usually appears near their job title (e.g. '.NET Developer', 'Software Engineer') and NOT near house numbers or district info\n"
        "  * If OCR text interleaves address and name on the same line, carefully separate them\n"
        "- desired_title: Infer from job history, skills, and overall experience. Pick the SINGLE most suitable title. INCLUDE the seniority prefix if present (e.g. 'Senior Software Engineer', 'Junior Designer').\n"
        "- Do not invent skills not supported by the CV\n"
        "- skills_list max 30 items\n"
        "- Prefer concrete skills/tools/tech\n"
        "- seniority_hint STRICT RULES:\n"
        "  * If the job title or CV explicitly says 'Senior' or 'Sr.' → MUST be 'senior'\n"
        "  * If the job title says 'Lead', 'Principal', 'Staff', 'Manager', 'Director', 'VP', 'Head of' → use 'lead'\n"
        "  * If the job title says 'Junior', 'Jr.' or experience < 2 years → use 'junior'\n"
        "  * If the job title says 'Intern', 'Trainee', 'Fresher' or experience < 1 year → use 'intern'\n"
        "  * Otherwise (2-5 years, no seniority prefix) → use 'mid'\n"
        "  * The title keyword ALWAYS takes priority over years of experience\n"
        "- English only\n"
        "- No extra text outside JSON\n"
    )

    user_input = f'CV raw text:\n"""\n{raw_text}\n"""'

    payload = {
        "model": OPENAI_MODEL,
        "instructions": instructions,
        "input": user_input,
        "max_output_tokens": 800,
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
        out_text = (out_text or "").strip()
        obj = _extract_json_object(out_text)

    # sanitize
    name = str(obj.get("name", "Unknown")).strip() or "Unknown"
    email = str(obj.get("email", "Unknown")).strip() or "Unknown"
    phone = str(obj.get("phone", "Unknown")).strip() or "Unknown"
    
    skills_list = obj.get("skills_list") or []
    if not isinstance(skills_list, list):
        skills_list = []
    skills_list = [str(x).strip() for x in skills_list if str(x).strip()]
    skills_list = skills_list[:30]

    skills_text = obj.get("skills_text")
    if not skills_text or not isinstance(skills_text, str):
        skills_text = ", ".join(skills_list) if skills_list else "Unknown"

    industry_background = obj.get("industry_background")
    if not industry_background or not isinstance(industry_background, str):
        industry_background = "Unknown"

    seniority_hint = str(obj.get("seniority_hint", "unknown")).strip().lower()
    # Normalize expanded values to 5-level system
    seniority_map = {
        "intern": "intern", "fresher": "intern", "trainee": "intern",
        "junior": "junior", "jr": "junior",
        "mid": "mid", "middle": "mid", "intermediate": "mid",
        "senior": "senior", "sr": "senior",
        "lead": "lead", "principal": "lead", "staff": "lead",
        "manager": "lead", "director": "lead", "executive": "lead",
        "vp": "lead", "head": "lead",
    }
    seniority_hint = seniority_map.get(seniority_hint, "mid")

    desired_title = str(obj.get("desired_title", "Unknown")).strip() or "Unknown"
    
    # Post-hoc fix: if desired_title contains Senior/Lead/Junior, override seniority
    title_lower = desired_title.lower()
    if any(kw in title_lower for kw in ["senior", "sr.", "sr "]):
        seniority_hint = "senior"
    elif any(kw in title_lower for kw in ["lead", "principal", "staff", "head of", "director", "manager", "vp"]):
        seniority_hint = "lead"
    elif any(kw in title_lower for kw in ["junior", "jr.", "jr "]):
        seniority_hint = "junior"
    elif any(kw in title_lower for kw in ["intern", "trainee", "fresher"]):
        seniority_hint = "intern"

    notes = str(obj.get("notes", "")).strip()[:200]

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "desired_title": desired_title,
        "skills_text": skills_text,
        "skills_list": skills_list,
        "industry_background": industry_background,
        "seniority_hint": seniority_hint,
        "notes": notes,
    }

# =========================
# SKILL CATEGORY MAPPING
# =========================
_SKILL_CATEGORIES_CACHE = None
_SKILL_MAPPING_CACHE = {}  # Simple in-memory cache: hash(skills) -> categories

def load_skill_categories() -> List[Dict[str, str]]:
    """Load skill categories from CSV for mapping"""
    global _SKILL_CATEGORIES_CACHE
    if _SKILL_CATEGORIES_CACHE is not None:
        return _SKILL_CATEGORIES_CACHE
    
    try:
        # Try /app/data first (actual skill data)
        csv_path = "/app/data/mappings/skills.csv"
        if not os.path.exists(csv_path):
            # Fallback to menu_data
            csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "menu_data", "mappings", "skills.csv")
        
        if not os.path.exists(csv_path):
            logger.warning(f"[load_skill_categories] skills.csv not found at {csv_path}")
            return []
        
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            categories = [
                {"abbr": row["skill_abr"].strip(), "name": row["skill_name"].strip()}
                for row in reader
                if row.get("skill_abr") and row.get("skill_name")
            ]
        
        _SKILL_CATEGORIES_CACHE = categories
        logger.info(f"[load_skill_categories] Loaded {len(categories)} skill categories")
        return categories
    except Exception as e:
        logger.error(f"[load_skill_categories] Error: {e}")
        return []

async def map_skills_to_categories(detailed_skills: List[str]) -> Dict[str, Any]:
    """
    Use OpenAI to map detailed technical skills to broad job categories.
    Returns: {
        "detailed_skills": [...],
        "skill_categories": ["IT", "DSGN", ...],
        "mapping": {"Python": "IT", ...}
    }
    """
    if not detailed_skills:
        return {"detailed_skills": [], "skill_categories": [], "mapping": {}}
    
    # Check cache
    cache_key = hash(tuple(sorted(detailed_skills)))
    if cache_key in _SKILL_MAPPING_CACHE:
        logger.info(f"[map_skills_to_categories] Cache hit for {len(detailed_skills)} skills")
        return _SKILL_MAPPING_CACHE[cache_key]
    
    # Load available categories
    categories = load_skill_categories()
    if not categories:
        logger.warning("[map_skills_to_categories] No categories loaded, returning empty")
        return {"detailed_skills": detailed_skills, "skill_categories": [], "mapping": {}}
    
    # Build category description for AI (all categories — only 36 entries, safe for tokens)
    category_desc = "\n".join([f"- {cat['abbr']}: {cat['name']}" for cat in categories])
    
    instructions = (
        "You are an expert skill classifier.\n"
        "Map the following technical/professional skills to appropriate job categories.\n\n"
        f"Available Categories:\n{category_desc}\n\n"
        f"Skills to map: {', '.join(detailed_skills)}\n\n"
        "Return ONLY a valid JSON object with schema:\n"
        "{\n"
        '  "mapping": {"Python": "IT", "Django": "IT", "TensorFlow": "IT", "Figma": "DSGN", "AutoCAD": "ENG", ...},\n'
        '  "categories": ["IT", "DSGN", "ENG"]\n'
        "}\n"
        "Rules:\n"
        "- Use only category abbreviations from the list above\n"
        "- Programming languages, frameworks, ML/AI tools, databases, DevOps tools → IT (Information Technology)\n"
        "- If a skill doesn't clearly fit any category, use the closest match rather than omitting\n"
        "- Categories array should be unique\n"
        "- No extra text outside JSON\n"
    )
    
    try:
        if not OPENAI_API_KEY:
            raise RuntimeError("Missing OPENAI_API_KEY")
        
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that returns only valid JSON."},
                {"role": "user", "content": instructions}
            ],
            "temperature": 0.1,
            "max_tokens": 800,
        }
        
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        
        out_text = ""
        for choice in data.get("choices", []):
            msg = choice.get("message", {})
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for c in content:
                        out_text += c.get("text", "")
                else:
                    out_text += content
        
        obj = _extract_json_object(out_text.strip())
        
        mapping = obj.get("mapping", {})
        categories_list = obj.get("categories", [])
        
        # Sanitize
        if not isinstance(mapping, dict):
            mapping = {}
        if not isinstance(categories_list, list):
            categories_list = []
        
        result = {
            "detailed_skills": detailed_skills,
            "skill_categories": list(set(categories_list)),  # Unique
            "mapping": mapping
        }
        
        # Cache result
        _SKILL_MAPPING_CACHE[cache_key] = result
        logger.info(f"[map_skills_to_categories] Mapped {len(detailed_skills)} skills -> {len(result['skill_categories'])} categories")
        
        return result
        
    except Exception as e:
        logger.error(f"[map_skills_to_categories] Error: {e}")
        return {"detailed_skills": detailed_skills, "skill_categories": [], "mapping": {}}

# =========================
# INDUSTRY MAPPING
# =========================
_INDUSTRY_LIST_CACHE = None
_INDUSTRY_MAPPING_CACHE = {}  # hash(industry_background) -> industry_ids

def load_industry_list() -> List[Dict[str, Any]]:
    """Load industry list from CSV for mapping"""
    global _INDUSTRY_LIST_CACHE
    if _INDUSTRY_LIST_CACHE is not None:
        return _INDUSTRY_LIST_CACHE
    
    try:
        csv_path = "/app/data/mappings/industries.csv"
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "menu_data", "mappings", "industries.csv")
        
        if not os.path.exists(csv_path):
            logger.warning(f"[load_industry_list] industries.csv not found at {csv_path}")
            return []
        
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            industries = [
                {"id": int(row["industry_id"]), "name": row["industry_name"].strip()}
                for row in reader
                if row.get("industry_id") and row.get("industry_name")
            ]
        
        _INDUSTRY_LIST_CACHE = industries
        logger.info(f"[load_industry_list] Loaded {len(industries)} industries")
        return industries
    except Exception as e:
        logger.error(f"[load_industry_list] Error: {e}")
        return []


async def map_industries_to_ids(industry_background: str) -> Dict[str, Any]:
    """
    Use OpenAI to map free-text industry background to system industry IDs.
    Returns: {
        "industry_ids": [4, 6, ...],
        "industry_names": ["Software Development", ...],
        "mapping": {"fintech": {"id": 4, "name": "Software Development"}, ...}
    }
    """
    if not industry_background or industry_background == "Unknown":
        return {"industry_ids": [], "industry_names": [], "mapping": {}}
    
    # Check cache
    cache_key = hash(industry_background.lower().strip())
    if cache_key in _INDUSTRY_MAPPING_CACHE:
        logger.info(f"[map_industries_to_ids] Cache hit")
        return _INDUSTRY_MAPPING_CACHE[cache_key]
    
    industries = load_industry_list()
    if not industries:
        logger.warning("[map_industries_to_ids] No industries loaded, returning empty")
        return {"industry_ids": [], "industry_names": [], "mapping": {}}
    
    # Build a compact industry list for prompt (sample to fit token budget)
    industry_desc = "\n".join([f"- {ind['id']}: {ind['name']}" for ind in industries])
    
    instructions = (
        "You are an expert industry classifier.\n"
        "Given a candidate's industry background, map it to the most relevant industries from the list below.\n\n"
        f"Available Industries:\n{industry_desc}\n\n"
        f"Candidate's industry background: {industry_background}\n\n"
        "Return ONLY a valid JSON object with schema:\n"
        "{\n"
        '  "selected": [{"id": 4, "name": "Software Development"}, ...]\n'
        "}\n"
        "Rules:\n"
        "- Select 3-8 most relevant industries\n"
        "- Use only IDs and names from the list above\n"
        "- Order by relevance (most relevant first)\n"
        "- No extra text outside JSON\n"
    )
    
    try:
        if not OPENAI_API_KEY:
            raise RuntimeError("Missing OPENAI_API_KEY")
        
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that returns only valid JSON."},
                {"role": "user", "content": instructions}
            ],
            "temperature": 0.1,
            "max_tokens": 500,
        }
        
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        
        out_text = ""
        for choice in data.get("choices", []):
            msg = choice.get("message", {})
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for c in content:
                        out_text += c.get("text", "")
                else:
                    out_text += content
        
        obj = _extract_json_object(out_text.strip())
        
        selected = obj.get("selected", [])
        if not isinstance(selected, list):
            selected = []
        
        # Validate against known industry IDs
        valid_ids = {ind["id"] for ind in industries}
        validated = []
        for item in selected:
            if isinstance(item, dict) and item.get("id") in valid_ids:
                validated.append(item)
        
        result = {
            "industry_ids": [item["id"] for item in validated],
            "industry_names": [item.get("name", "") for item in validated],
            "mapping": {item.get("name", ""): item["id"] for item in validated}
        }
        
        _INDUSTRY_MAPPING_CACHE[cache_key] = result
        logger.info(f"[map_industries_to_ids] Mapped '{industry_background}' -> {len(result['industry_ids'])} industries")
        
        return result
        
    except Exception as e:
        logger.error(f"[map_industries_to_ids] Error: {e}")
        return {"industry_ids": [], "industry_names": [], "mapping": {}}

async def _openai_extract_detailed_analysis(raw_text: str) -> Dict[str, Any]:
    """
    Extract comprehensive CV analysis for Dashboard.
    Returns basic profile + detailed strengths/weaknesses/industries/roles/radar.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY")
    
    raw_text = (raw_text or "").strip()
    if len(raw_text) > 14000:
        raw_text = raw_text[:10000] + "\n...\n" + raw_text[-3000:]
    
    instructions = """
You are an expert CV analyzer.
Return ONLY a valid JSON object with this schema:

{
  "skills_text": "comma separated skills",
  "skills_list": ["skill1", "skill2", ...],
  "industry_background": "comma separated industries",
  "seniority_hint": "junior|mid|senior|unknown",
  "notes": "brief summary",
  
  "strengths": [
    {"skill": "Python", "score": 85, "note": "5 years experience"}
  ],
  "weaknesses": [
    {"skill": "Frontend", "gap": 80, "tip": "Learn React basics", "url": "https://react.dev"}
  ],
  "industries": [
    {"name": "Fintech", "score": 86, "rationale": "Strong data skills"}
  ],
  "roles": [
    {"name": "Backend Engineer", "score": 88, "rationale": "API experience"}
  ],
  "radar": [
    {"axis": "Backend", "score": 85},
    {"axis": "Data", "score": 70},
    {"axis": "DevOps", "score": 50},
    {"axis": "Frontend", "score": 30},
    {"axis": "AI/ML", "score": 60}
  ],
  "explanations": [
    "Reason 1...",
    "Reason 2..."
  ]
}

Rules:
- strengths: Top 4-6 skills with scores 0-100
- weaknesses: Top 3-4 gaps with gap score + learning tips + URLs
- industries: Top 3 with scores + rationale
- roles: Top 2-3 with scores + rationale
- radar: Exactly 5 axes (Backend, Data, DevOps, Frontend, AI/ML) scores 0-100
- explanations: 2-3 bullet points
- All in English
- No extra text outside JSON
"""
    
    user_input = f'CV raw text:\n"""\n{raw_text}\n"""'
    
    payload = {
        "model": OPENAI_MODEL,
        "instructions": instructions,
        "input": user_input,
        "max_output_tokens": 1500,
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
        out_text = (out_text or "").strip()
        obj = _extract_json_object(out_text)
    
    # Sanitize
    skills_list = obj.get("skills_list") or []
    if not isinstance(skills_list, list):
        skills_list = []
    skills_list = [str(x).strip() for x in skills_list if str(x).strip()][:30]
    
    return {
        "skills_text": str(obj.get("skills_text", "Unknown")),
        "skills_list": skills_list,
        "industry_background": str(obj.get("industry_background", "Unknown")),
        "seniority_hint": str(obj.get("seniority_hint", "unknown")),
        "notes": str(obj.get("notes", ""))[:200],
        "strengths": (obj.get("strengths") or [])[:6],
        "weaknesses": (obj.get("weaknesses") or [])[:4],
        "industries": (obj.get("industries") or [])[:3],
        "roles": (obj.get("roles") or [])[:3],
        "radar": (obj.get("radar") or [])[:5],
        "explanations": (obj.get("explanations") or [])[:5],
    }

# =========================
# Qdrant Retrieval (Stage 1)
# =========================
_qdrant_client = None
_qdrant_encoder = None

def _lazy_load_qdrant():
    global _qdrant_client, _qdrant_encoder
    if _qdrant_client is not None and _qdrant_encoder is not None:
        return
    _qdrant_client = QdrantClient(url=QDRANT_URL)
    _qdrant_encoder = SentenceTransformer(QDRANT_EMBED_MODEL)

def _qdrant_search(
    query_text: str,
    topk: int,
    seniority_exact: Optional[str] = None,
    require_seniority: bool = False,
    seniority_field: str = JOBFIT_SENIORITY_FIELD,
) -> List[Dict[str, Any]]:
    _lazy_load_qdrant()
    # Over-fetch to compensate for duplicates removed during dedup (~42% dup rate)
    fetch_limit = topk * 3
    qv = _qdrant_encoder.encode(query_text, normalize_embeddings=True).tolist()
    client = _qdrant_client

    qfilter = None
    if (seniority_exact or require_seniority) and _HAS_QDRANT_FILTER_MODELS:
        must = []
        must_not = []
        if seniority_exact:
            must.append(FieldCondition(key=seniority_field, match=MatchValue(value=str(seniority_exact))))
        if require_seniority:
            # Exclude empty string / missing
            must_not.append(IsEmptyCondition(is_empty={"key": seniority_field}))
            must_not.append(IsNullCondition(is_null={"key": seniority_field}))
        qfilter = Filter(must=must or None, must_not=must_not or None)

    # compat layer
    if hasattr(client, "search"):
        hits = client.search(collection_name=QDRANT_COLLECTION, query_vector=qv, limit=fetch_limit, query_filter=qfilter, with_payload=True)
    elif hasattr(client, "query_points"):
        hits = client.query_points(collection_name=QDRANT_COLLECTION, query=qv, limit=fetch_limit, query_filter=qfilter, with_payload=True).points
    else:
        raise RuntimeError("qdrant-client too old. Please upgrade qdrant-client.")

    out = []
    for h in hits:
        payload = getattr(h, "payload", None) or {}
        score = float(getattr(h, "score", 0.0))
        pid = int(payload.get("job_id", getattr(h, "id", 0)))
        out.append({"job_id": pid, "score_retrieval": score, "payload": payload})

    # Deduplicate by (title, company_name) — Qdrant has duplicate entries
    # with different job_ids but identical content
    seen: Dict[tuple, Dict] = {}
    for item in out:
        p = item.get("payload") or {}
        dedup_key = (str(p.get("title", "")).strip().lower(), str(p.get("company_name", "")).strip().lower())
        if dedup_key not in seen or item["score_retrieval"] > seen[dedup_key]["score_retrieval"]:
            seen[dedup_key] = item
    deduped = list(seen.values())
    if len(deduped) < len(out):
        logger.info("[qdrant] deduped %d → %d unique jobs (by title+company)", len(out), len(deduped))
    # Trim back to requested topk
    return deduped[:topk]

# =========================
# Reranker (Stage 2) - uses your NEW trained rubric JSON
# Output keys: skill_fit, industry_fit, seniority_fit, salary_fit, benefit_fit, final_score
# =========================
import threading

_rerank_tokenizer = None
_rerank_model = None
_rerank_lock = None

def _lazy_load_reranker():
    global _rerank_tokenizer, _rerank_model, _rerank_lock
    if _rerank_lock is None:
        _rerank_lock = threading.Lock()
        
    if _rerank_model is not None:
        return

    model_path = RERANK_MODEL_DIR
    logger.info("[rerank] loading model from %s ...", model_path)
    logger.info("[rerank] CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("[rerank] GPU: %s, VRAM: %.1f GB",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    _rerank_tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    _rerank_tokenizer.padding_side = "left"  # fix right-padding warning for decoder-only

    _rerank_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    _rerank_model.eval()
    logger.info("[rerank] model loaded on device: %s", _rerank_model.device)

def _safe_parse_json_obj(txt: str) -> Dict[str, Any]:
    """Best-effort JSON object parse from model output."""
    txt = (txt or "").strip()
    try:
        obj = json.loads(txt)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _clamp_0_10_int(v: Any) -> int:
    try:
        x = int(v)
    except Exception:
        x = 0
    return max(0, min(10, x))


def _parse_rubric_scores(txt: str) -> Tuple[Dict[str, int], int]:
    obj = _safe_parse_json_obj(txt)
    scores = {
        "skill_fit": _clamp_0_10_int(obj.get("skill_fit", 0)),
        "industry_fit": _clamp_0_10_int(obj.get("industry_fit", 0)),
        "seniority_fit": _clamp_0_10_int(obj.get("seniority_fit", 0)),
        "salary_fit": _clamp_0_10_int(obj.get("salary_fit", 0)),
        "benefit_fit": _clamp_0_10_int(obj.get("benefit_fit", 0)),
        "final_score": _clamp_0_10_int(obj.get("final_score", 0)),
    }
    return scores, scores["final_score"]

def _build_rerank_prompt(candidate: Dict[str, Any], job: Dict[str, Any], fields_mode: str) -> str:
    """Must match the prompt used during label/train (build_prompt_rubric in jobfit_allinone.py)."""

    prefs = candidate.get("prefs", {}) or {}

    cand_block = [
        f"- Technical Skills: {candidate.get('skills','Unknown')}",
        f"- Industry Background: {candidate.get('industry','Unknown')}",
        f"- Seniority / Experience Level: {candidate.get('seniority','Unknown')}",
    ]
    job_block = [
        f"- Job Title: {job.get('title','Unknown')}",
        f"- Required Competencies: {job.get('skills','Unknown')}",
        f"- Industry Context: {job.get('industries','Unknown')}",
        f"- Seniority / Experience Level: {job.get('seniority','Unknown')}",
    ]

    if fields_mode in ("skillssalary", "full"):
        cand_block.append(f"- Salary Expectation: {candidate.get('target_salary','Unknown')}")
        job_block.append(f"- Compensation: {job.get('salary','Unknown')}")

    if fields_mode == "full":
        prefs = candidate.get("prefs", {}) or {}
        cand_block.extend([
            f"- Benefit Preferences: {prefs.get('benefits','Unknown')}",
            f"- Preferred Company Size: {prefs.get('employee_bucket','Unknown')}",
            f"- Preferred Company Industries: {prefs.get('company_industries','Unknown')}",
            f"- Preferred Company Specialities: {prefs.get('company_specialities','Unknown')}",
        ])
        job_block.extend([
            f"- Benefits: {job.get('benefits','Unknown')}",
            f"- Company Size: {job.get('employee_bucket','Unknown')}",
            f"- Company Industries: {job.get('company_industries','Unknown')}",
            f"- Company Specialities: {job.get('company_specialities','Unknown')}",
        ])
        jd = job.get("job_description", "Unknown")
        job_block.append(f"- Job Description: {str(jd)[:400]}")

    user_msg = (
        "You are grading Candidate vs Job fit using a rubric.\n"
        "Return ONLY strict JSON with keys: skill_fit, industry_fit, seniority_fit, salary_fit, benefit_fit, final_score.\n"
        "All values are integers 0..10.\n\n"
        "Rubric definitions:\n"
        "- skill_fit: semantic overlap of skills/competencies.\n"
        "- industry_fit: domain alignment.\n"
        "- seniority_fit: experience level alignment.\n"
        "- salary_fit: salary expectation vs compensation alignment.\n"
        "- benefit_fit: benefit preference vs offered benefits alignment.\n"
        "- final_score: overall fit; industry mismatch should pull final down.\n\n"
        "CANDIDATE:\n" + "\n".join(cand_block) + "\n\n"
        "JOB:\n" + "\n".join(job_block) + "\n\n"
        "OUTPUT JSON ONLY."
    )

    # Qwen2.* chat template style (matches your previous router)
    return (
        "<|im_start|>system\nYou are a strict JSON grader. Output ONLY valid JSON.\n"
        "Keys: skill_fit, industry_fit, seniority_fit, salary_fit, benefit_fit, final_score.\n"
        "All values are integers 0..10.\n"
        "Do not include any extra text.<|im_end|>\n"
        "<|im_start|>user\n" + user_msg + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def _rerank_jobs(candidate_profile: Dict[str, Any], retrieved: List[Dict[str, Any]], fields_mode: str) -> List[Dict[str, Any]]:
    _lazy_load_reranker()

    prompts: List[str] = []
    jobs_norm: List[Dict[str, Any]] = []

    for item in retrieved:
        p = item.get("payload") or {}
        job = {
            "job_id": int(item.get("job_id")),
            "title": p.get("title", "Unknown"),
            "company_name": p.get("company_name", "Unknown"),
            "skills": p.get("skills", "Unknown"),
            "industries": p.get("industries", "Unknown"),
            "seniority": p.get(JOBFIT_SENIORITY_FIELD, p.get("seniority", p.get("formatted_experience_level", "Unknown"))),
            "salary": p.get("salary_bucket", "Unknown"),
            "benefits": p.get("benefits", "Unknown"),
            "employee_bucket": p.get("employee_bucket", "Unknown"),
            "company_industries": p.get("company_industries", "Unknown"),
            "company_specialities": p.get("company_specialities", "Unknown"),
            "job_description": p.get("job_description", "Unknown"),
        }
        jobs_norm.append(job)
        prompts.append(_build_rerank_prompt(candidate_profile, job, fields_mode))

    total_batches = (len(prompts) + RERANK_BATCH - 1) // RERANK_BATCH
    logger.info("[rerank] starting: %d jobs, batch_size=%d, total_batches=%d, fields_mode=%s",
                len(prompts), RERANK_BATCH, total_batches, fields_mode)

    results: List[Dict[str, Any]] = []
    for i in range(0, len(prompts), RERANK_BATCH):
        batch_idx = i // RERANK_BATCH + 1
        batch = prompts[i:i + RERANK_BATCH]
        batch_start = time.perf_counter()
        logger.info("[rerank] batch %d/%d (%d items) — tokenizing...", batch_idx, total_batches, len(batch))

        inputs = _rerank_tokenizer(batch, return_tensors="pt", padding=True).to(_rerank_model.device)
        logger.info("[rerank] batch %d/%d — input_ids shape=%s, generating...",
                    batch_idx, total_batches, list(inputs["input_ids"].shape))

        with _rerank_lock:
            with torch.inference_mode():
                out = _rerank_model.generate(
                    **inputs,
                    max_new_tokens=RERANK_MAX_NEW_TOKENS,
                    do_sample=False,
                )

        batch_ms = (time.perf_counter() - batch_start) * 1000
        logger.info("[rerank] batch %d/%d — done in %.0fms, decoding...", batch_idx, total_batches, batch_ms)

        texts = _rerank_tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        for j, txt in enumerate(texts):
            rubric, score = _parse_rubric_scores(txt)
            job = jobs_norm[i + j]
            retrieval_score = float(retrieved[i + j].get("score_retrieval", 0.0))

            results.append({
                "job_id": job["job_id"],
                "score": score,
                "rubric": rubric,
                "retrieval_score": retrieval_score,
                "job": job,
                "raw_model_json": (txt or "").strip(),
            })

    logger.info("[rerank] all %d batches done, sorting...", total_batches)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# =========================
# Simple single-user cache (memory + optional file)
# =========================
_jobfit_cache: Dict[str, Any] = {"profile_hash": None, "updated_at": None, "recommendations": None}


def _profile_hash(profile: Dict[str, Any]) -> str:
    try:
        s = json.dumps(profile, ensure_ascii=False, sort_keys=True)
    except Exception:
        s = str(profile)
    return uuid.uuid5(uuid.NAMESPACE_DNS, s).hex


def _cache_load() -> None:
    global _jobfit_cache
    try:
        if os.path.exists(JOBFIT_CACHE_FILE):
            sz = os.path.getsize(JOBFIT_CACHE_FILE)
            if sz < 3:  # empty or just "{}"
                logger.info("[cache] file exists but too small (%d bytes), skipping load", sz)
                return
            with open(JOBFIT_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            if isinstance(data, dict) and data.get("recommendations"):
                _jobfit_cache.update(data)
                logger.info("[cache] loaded %d recommendations from file", len(data.get("recommendations", [])))
            else:
                logger.info("[cache] file has no recommendations, keeping in-memory cache")
    except Exception as e:
        logger.warning("[cache] load failed (keeping in-memory): %s", e)


def _cache_save() -> None:
    try:
        tmp = JOBFIT_CACHE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_jobfit_cache, f, ensure_ascii=False)
        # Atomic rename (avoids corruption if interrupted)
        os.replace(tmp, JOBFIT_CACHE_FILE)
        recs = _jobfit_cache.get("recommendations", [])
        logger.info("[cache] saved %d recommendations to file (%s)", len(recs), JOBFIT_CACHE_FILE)
    except Exception as e:
        logger.error("[cache] SAVE FAILED: %s", e)
        # Clean up temp file if it exists
        try:
            os.remove(JOBFIT_CACHE_FILE + ".tmp")
        except OSError:
            pass


class CandidatePrefs(BaseModel):
    benefits: str = "Unknown"
    employee_bucket: str = "Unknown"
    company_industries: str = "Unknown"
    company_specialities: str = "Unknown"


class CandidateProfileIn(BaseModel):
    skills: str = Field(default="Unknown")
    industry: str = Field(default="Unknown")
    seniority: str = Field(default="Unknown")
    target_salary: str = Field(default="Unknown")
    prefs: CandidatePrefs = Field(default_factory=CandidatePrefs)


class RecommendRequest(BaseModel):
    profile: CandidateProfileIn
    topk: int = Field(default=50, ge=1, le=500)
    fields_mode: str = Field(default="")  # skills|skillssalary|full
    force_refresh: bool = Field(default=False)


class ExtractProfileRequest(BaseModel):
    raw_text: str
    detailed: bool = False


# =========================
# ENDPOINTS
# =========================
@router.post("/ocr")
async def ocr_cv(file: UploadFile = File(...)):
    tm = StageTimer()
    req_id = uuid.uuid4().hex[:8]

    with tm.stage("read_upload"):
        content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        is_pdf = _is_pdf(file.filename, file.content_type)

        if is_pdf and not OCR_FORCE_OCR:
            with tm.stage("pdf_extract_text"):
                extracted = _extract_text_pdf_bytes(content)
            if extracted and len(extracted) >= OCR_TEXT_MIN_CHARS:
                total = tm.total_ms()
                logger.info("[ocr][%s] mode=pdf_text_extract chars=%d total_ms=%.2f", req_id, len(extracted), total)
                return {
                    "text": extracted,
                    "confidence": 1.0,
                    "_timing_ms": {**tm.ms, "total": total},
                    "_req_id": req_id,
                    "_mode": "pdf_text_extract",
                }

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

            full_text = "\n\n".join(texts).strip()
            avg_conf = float(sum(scores) / max(len(scores), 1)) if scores else 0.0
        else:
            with tm.stage("bytes_to_image"):
                img = _bytes_to_image(content)
            with tm.stage("ocr_image"):
                full_text, avg_conf = _rapidocr_on_rgb(img)

        total = tm.total_ms()
        logger.info("[ocr][%s] mode=ocr pdf=%s bytes=%d total_ms=%.2f", req_id, is_pdf, len(content), total)

        return {
            "text": full_text,
            "confidence": avg_conf,
            "_timing_ms": {**tm.ms, "total": total},
            "_req_id": req_id,
            "_mode": "ocr",
        }

    except Exception as e:
        logger.exception("[ocr][%s] failed err=%s", req_id, e)
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

@router.post("/extract-profile")
async def extract_profile(req: ExtractProfileRequest):
    """
    Extract candidate profile from CV text.
    - detailed=False (default): Simple profile for Profile page
    - detailed=True: Full analysis for Dashboard
    """
    raw_text = (req.raw_text or "").strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="raw_text is required")

    try:
        if req.detailed:
            # Full analysis for Dashboard
            prof = await _openai_extract_detailed_analysis(raw_text)
        else:
            # Simple extraction for Profile
            prof = await _openai_extract_candidate_profile(raw_text)
        
        # Map skills to categories if skills_list exists
        if prof.get("skills_list"):
            mapping_result = await map_skills_to_categories(prof["skills_list"])
            prof["skill_categories"] = mapping_result["skill_categories"]
            prof["skill_mapping"] = mapping_result["mapping"]
            logger.info(f"[extract-profile] Mapped {len(prof['skills_list'])} skills to {len(prof['skill_categories'])} categories")
        else:
            prof["skill_categories"] = []
            prof["skill_mapping"] = {}
        
        # Map industries to IDs if industry_background exists
        if prof.get("industry_background") and prof["industry_background"] != "Unknown":
            industry_result = await map_industries_to_ids(prof["industry_background"])
            prof["industry_ids"] = industry_result["industry_ids"]
            prof["industry_names"] = industry_result["industry_names"]
            logger.info(f"[extract-profile] Mapped industries -> {len(prof['industry_ids'])} IDs")
        else:
            prof["industry_ids"] = []
            prof["industry_names"] = []
        
        return {"status": "success", "data": prof}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extract profile failed: {e}")

@router.post("/jobfit/recommend")
def jobfit_recommend(req: RecommendRequest):
    """
    Main endpoint for FE:
      - FE sends the USER PROFILE (already stored on FE)
      - BE does: embed -> Qdrant retrieval TopK -> rubric rerank by your LoRA model
      - Return TopK jobs (default 50)

    Notes:
      - No DB persistence: we cache 1 user's latest recommendation to a local JSON file.
      - If cache matches profile (hash), we return immediately.
      - This is a sync def so FastAPI runs it in thread pool, freeing the event loop
        for concurrent OCR/extract requests while keeping CUDA on the same thread.
    """
    tm = StageTimer()
    req_id = uuid.uuid4().hex[:8]

    # load cache lazily (best-effort)
    with tm.stage("cache_load"):
        _cache_load()

    profile = req.profile.model_dump()
    fm = (req.fields_mode or "").strip() or RERANK_DEFAULT_FIELDS_MODE
    if fm not in ("skills", "skillssalary", "full"):
        fm = RERANK_DEFAULT_FIELDS_MODE

    # sanitize profile (avoid None)
    candidate_profile = {
        "skills": (profile.get("skills") or "Unknown").strip() or "Unknown",
        "industry": (profile.get("industry") or "Unknown").strip() or "Unknown",
        "seniority": (profile.get("seniority") or "Unknown").strip() or "Unknown",
        "target_salary": (profile.get("target_salary") or "Unknown").strip() or "Unknown",
        "prefs": profile.get("prefs") or {},
    }

    ph = _profile_hash(candidate_profile)
    if (not req.force_refresh) and _jobfit_cache.get("profile_hash") == ph and _jobfit_cache.get("recommendations"):
        total = tm.total_ms()
        logger.info("[jobfit][%s] cache_hit topk=%s total_ms=%.2f", req_id, req.topk, total)
        return {
            "cached": True,
            "profile_hash": ph,
            "fields_mode": fm,
            "recommendations": _jobfit_cache.get("recommendations"),
            "_timing_ms": {**tm.ms, "total": total},
            "_req_id": req_id,
        }

    try:
        # Stage 1: Qdrant retrieval
        with tm.stage("qdrant_retrieve"):
            query_text = (
                f"Skills: {candidate_profile['skills']}\n"
                f"Industries: {candidate_profile['industry']}\n"
                f"Seniority: {candidate_profile['seniority']}"
            )
            logger.info("[jobfit][%s] Stage 1: Qdrant search — query: %s...", req_id, query_text[:100])
            # Over-fetch to compensate for potential duplicates in Qdrant
            fetch_topk = int(req.topk) * 2
            seniority_exact = candidate_profile["seniority"] if (JOBFIT_USE_SENIORITY_FILTER and candidate_profile["seniority"].lower() != "unknown") else None
            raw_retrieved = _qdrant_search(
                query_text=query_text,
                topk=fetch_topk,
                seniority_exact=seniority_exact,
                require_seniority=JOBFIT_REQUIRE_SENIORITY,
                seniority_field=JOBFIT_SENIORITY_FIELD,
            )
            
            # Dedup at retrieval level by (title, company_name) before expensive rerank
            seen_retrieval: set = set()
            retrieved = []
            for item in raw_retrieved:
                p = item.get("payload") or {}
                dedup_key = (str(p.get("title", "")).strip().lower(), str(p.get("company_name", "")).strip().lower())
                if dedup_key not in seen_retrieval:
                    seen_retrieval.add(dedup_key)
                    retrieved.append(item)
            
            if len(retrieved) < len(raw_retrieved):
                logger.info("[jobfit][%s] Retrieval dedup: %d -> %d unique jobs (by title+company)", req_id, len(raw_retrieved), len(retrieved))
            
            # Truncate to topk after dedup (don't rerank more than needed)
            retrieved = retrieved[:int(req.topk)]

        if not retrieved:
            raise HTTPException(status_code=404, detail="No jobs retrieved from Qdrant")

        logger.info("[jobfit][%s] Stage 1 done: retrieved %d jobs (%.0fms). Starting Stage 2 rerank...",
                    req_id, len(retrieved), tm.ms.get("qdrant_retrieve", 0))

        # Stage 2: Rerank with LoRA
        with tm.stage("rerank"):
            reranked = _rerank_jobs(candidate_profile, retrieved, fields_mode=fm)

        # Deduplicate by (title, company_name) post-rerank — keep highest score
        seen_titles: set = set()
        deduped = []
        for r in reranked:
            job = r["job"]
            dedup_key = (str(job.get("title", "")).strip().lower(), str(job.get("company_name", "")).strip().lower())
            if dedup_key not in seen_titles:
                seen_titles.add(dedup_key)
                deduped.append(r)
        
        if len(deduped) < len(reranked):
            logger.info("[jobfit][%s] Deduped %d -> %d jobs (by title+company)", req_id, len(reranked), len(deduped))

        # return TopK
        recs = deduped[: int(req.topk)]

        out_recs = []
        for r in recs:
            job = r["job"]
            out_recs.append({
                "job_id": r["job_id"],
                "final_score": r["score"],
                "rubric": r.get("rubric", {}),
                "retrieval_score": r["retrieval_score"],

                "title": job.get("title", "Unknown"),
                "company_name": job.get("company_name", "Unknown"),
                "industries": job.get("industries", "Unknown"),
                "seniority": job.get("seniority", "Unknown"),
                "skills": job.get("skills", "Unknown"),
                "salary_bucket": job.get("salary", "Unknown"),
                "benefits": job.get("benefits", "Unknown"),
                "employee_bucket": job.get("employee_bucket", "Unknown"),
                "company_industries": job.get("company_industries", "Unknown"),
                "company_specialities": job.get("company_specialities", "Unknown"),
                "job_description": job.get("job_description", "Unknown"),
            })

        # save cache
        with tm.stage("cache_save"):
            _jobfit_cache.update({
                "profile_hash": ph,
                "updated_at": time.time(),
                "recommendations": out_recs,
            })
            _cache_save()

        total = tm.total_ms()
        logger.info("[jobfit][%s] topk=%s fields_mode=%s total_ms=%.2f", req_id, req.topk, fm, total)

        return {
            "cached": False,
            "profile_hash": ph,
            "fields_mode": fm,
            "recommendations": out_recs,
            "_timing_ms": {**tm.ms, "total": total},
            "_req_id": req_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[jobfit][%s] failed err=%s", req_id, e)
        raise HTTPException(status_code=500, detail=f"JobFit recommend failed: {e}")


@router.get("/jobfit/cache")
async def jobfit_cache_get():
    """FE can call this to see whether backend already has recommendations cached."""
    # If in-memory cache has data, return it directly (always up-to-date)
    if _jobfit_cache.get("recommendations"):
        return _jobfit_cache
    # Otherwise try loading from file (e.g. after container restart)
    _cache_load()
    return _jobfit_cache
