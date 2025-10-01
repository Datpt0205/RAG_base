from typing import List, Optional, Union, Dict
from pydantic import BaseModel
from src.core.schemas.cv import RadarItem

class UiStrength(BaseModel):
    skill: str
    score: int

class UiWeakness(BaseModel):
    skill: str
    gap: int
    tip: Optional[str] = None
    url: Optional[str] = None  # NEW: link học

class UiIndustry(BaseModel):
    name: str
    score: int
    rationale: Optional[str] = None

class UiRole(BaseModel):
    name: str
    score: int
    rationale: Optional[str] = None

class UiCvAnalysis(BaseModel):
    strengths: List[UiStrength]
    weaknesses: List[UiWeakness]
    industries: List[UiIndustry]
    roles: List[UiRole]
    explanations: Optional[List[str]] = None
    # NEW: radar động
    radar: List[RadarItem] = []

class UiAnalyzeResponse(BaseModel):
    status: str
    message: str
    data: Optional[UiCvAnalysis] = None
