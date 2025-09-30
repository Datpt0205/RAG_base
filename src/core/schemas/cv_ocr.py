# server/src/core/schemas/cv_ocr.py
from typing import List, Optional
from pydantic import BaseModel

class OcrResponse(BaseModel):
    text: str
    skills: List[str] = []
