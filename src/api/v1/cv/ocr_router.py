# server/src/api/v1/cv/ocr_router.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from src.core.schemas.cv_ocr import OcrResponse
from src.features.cv_analysis.handlers.cv_analysis_handler import CvAnalysisHandler
from src.core.providers.provider_factory import ProviderType, ModelProviderFactory

router = APIRouter(prefix="/cv")
handler = CvAnalysisHandler()

@router.post("/ocr", response_model=OcrResponse)
async def ocr_cv(
    file: UploadFile = File(...),
    provider_type: str = Form("openai"),
    model_name: str = Form("gpt-4o-mini"),
):
    try:
        api_key = ModelProviderFactory._get_api_key(provider_type)
        if provider_type != ProviderType.OPENAI:
            raise HTTPException(400, "Only 'openai' is supported for OCR now")

        file_bytes = await file.read()
        content_type = file.content_type or ""

        # 1) Lấy text từ PDF/Ảnh (dùng logic có sẵn trong handler)
        text = ""
        if "pdf" in content_type:
            text = handler._pdf_to_text(file_bytes) or ""
        if not text:
            text = handler._ocr_with_openai(file_bytes, model_name=model_name, api_key=api_key)

        if not text or len(text.strip()) < 20:
            raise HTTPException(400, "OCR failed or text too short")

        # 2) Rút skills nhanh bằng OpenAI (nhẹ)
        prompt = f"""Trích xuất danh sách kỹ năng (hard skills) từ CV sau, trả về mảng JSON chữ thuần:
CV:
{text[:12000]}"""
        client = handler._openai_client(api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        skills_raw = resp.choices[0].message.content.strip()

        # Chuẩn hoá: nếu LLM trả chuỗi thường, convert thô sang list
        import json
        skills = []
        try:
            skills = json.loads(skills_raw)
            if not isinstance(skills, list): skills = []
        except Exception:
            # fallback: tách theo dấu phẩy
            skills = [s.strip() for s in skills_raw.split(",") if s.strip()]

        # một số normalize đơn giản
        norm = {
            "js":"JavaScript","postgres":"PostgreSQL","node":"Node.js",
            "py":"Python","ts":"TypeScript"
        }
        skills = list(dict.fromkeys([norm.get(s.lower(), s).strip() for s in skills]))[:30]

        return OcrResponse(text=text[:50000], skills=skills)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"OCR error: {e}")
