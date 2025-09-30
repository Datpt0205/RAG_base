from fastapi import APIRouter, UploadFile, File, Form, Response, status, HTTPException
from typing import Optional
from src.core.schemas.cv import AnalyzeCvRequest, AnalyzeCvResponse
from src.core.schemas.cv_ui import UiAnalyzeResponse
from src.features.cv_analysis.handlers.cv_analysis_handler import CvAnalysisHandler

router = APIRouter(prefix="/cv")
handler = CvAnalysisHandler()

@router.post("/analyze-ui", response_model=UiAnalyzeResponse)
async def analyze_cv_ui(
    response: Response,
    provider_type: str = Form("openai"),
    model_name: str = Form("gpt-4o-mini"),
    raw_text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    try:
        file_bytes = None; content_type = None
        if file:
            content_type = file.content_type
            file_bytes = await file.read()

        core = handler.analyze(
            provider_type=provider_type,
            model_name=model_name,
            raw_text=raw_text,
            file_bytes=file_bytes,
            file_content_type=content_type,
        )
        ui = handler.to_ui_payload(core, provider_type=provider_type, model_name=model_name)
        response.status_code = 200
        return UiAnalyzeResponse(status="success", message="OK", data=ui)
    except Exception as e:
        response.status_code = 400
        return UiAnalyzeResponse(status="error", message=str(e), data=None)

@router.post("/analyze", response_model=AnalyzeCvResponse)
async def analyze_cv(
    response: Response,
    provider_type: str = Form("openai"),
    model_name: str = Form("gpt-4o-mini"),
    raw_text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    try:
        file_bytes = None
        content_type = None
        if file:
            content_type = file.content_type
            file_bytes = await file.read()

        result = handler.analyze(
            provider_type=provider_type,
            model_name=model_name,
            raw_text=raw_text,
            file_bytes=file_bytes,
            file_content_type=content_type,
        )
        response.status_code = status.HTTP_200_OK
        return AnalyzeCvResponse(status="success", message="OK", data=result)
    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return AnalyzeCvResponse(status="error", message=str(e), data=None)
