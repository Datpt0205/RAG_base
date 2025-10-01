from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from enum import Enum

class RadarItem(BaseModel):
    axis: str
    score: int = Field(ge=0, le=100)

class RecommendedRole(BaseModel):
    title: str
    confidence: float = Field(ge=0.0, le=1.0)

class CvExtract(BaseModel):
    full_text: str
    emails: List[str] = []
    phones: List[str] = []
    years_experience: Optional[float] = None

# --- NEW: timeline + entities ---
class EducationItem(BaseModel):
    school: str
    degree: Optional[str] = None
    field: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None

class WorkItem(BaseModel):
    company: str
    title: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    projects: List[str] = []

class Entities(BaseModel):
    companies: List[str] = []
    schools: List[str] = []
    projects: List[str] = []

class CvAnalysis(BaseModel):
    cv_id: str
    extract: CvExtract
    skills: Dict[str, List[str]]  # {"hard":[], "soft":[]}
    strengths: List[str] = []
    weaknesses: List[str] = []
    recommended_roles: List[RecommendedRole] = []
    radar: List[RadarItem] = []

    # --- NEW fields ---
    education: List[EducationItem] = []
    experience: List[WorkItem] = []
    entities: Entities = Entities()

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
