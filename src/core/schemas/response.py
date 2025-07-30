from typing import Optional
from pydantic import BaseModel


class BasicResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict | list | str] = None

class BasicResponseDelete(BaseModel):
    Status: str
    Message: str
    Data: Optional[dict | list | str] = None

class ChatResponse(BaseModel):
    id: str
    role: str = "assistant"
    content: str