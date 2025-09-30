from typing import List, Optional, Union, Dict
from pydantic import BaseModel

class UiStrength(BaseModel):
    skill: str
    score: int

class UiWeakness(BaseModel):
    skill: str
    gap: int
    tip: Optional[str] = None

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
    explanations: Optional[List[Union[str, Dict]]] = None

class UiAnalyzeResponse(BaseModel):
    status: str
    message: str
    data: Optional[UiCvAnalysis] = None
