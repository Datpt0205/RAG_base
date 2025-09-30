from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from enum import Enum

class RadarItem(BaseModel):
    axis: str
    score: int  # 0..100

class RecommendedRole(BaseModel):
    title: str
    confidence: float  # 0..1

class CvExtract(BaseModel):
    full_text: Optional[str] = None
    emails: List[str] = []
    phones: List[str] = []
    years_experience: Optional[int] = None

class CvAnalysis(BaseModel):
    cv_id: str
    extract: CvExtract
    skills: Dict[str, List[str]] = Field(default_factory=lambda: {"hard": [], "soft": []})
    strengths: List[str] = []
    weaknesses: List[str] = []
    recommended_roles: List[RecommendedRole] = []
    radar: List[RadarItem] = []

# requests
class AnalyzeCvRequest(BaseModel):
    # FE có thể gửi raw_text (nhanh) hoặc upload file ở form-data (xem router)
    provider_type: str = "openai"
    model_name: str = "gpt-4o-mini"
    raw_text: Optional[str] = None  # nếu gửi JSON

class AnalyzeCvResponse(BaseModel):
    status: str
    message: str
    data: Optional[CvAnalysis] = None
