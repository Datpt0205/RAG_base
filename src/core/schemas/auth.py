from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


class APIKeyCreate(BaseModel):
    user_id: str
    organization_id: Optional[str] = None
    name: Optional[str] = Field(None, description="Descriptive name for API key")
    expires_in_days: int = Field(365, description="Number of days before API key expires")

class OrganizationInfo(BaseModel):
    organization_id: str
    name: str
    role: str

class APIKeyResponse(BaseModel):
    id: str
    api_key: str
    user_id: str
    organization_id: Optional[str] = None
    name: Optional[str] = None
    role: Optional[str] = None
    expiry_date: datetime
    is_active: bool
    created_at: Optional[datetime] = None

class APIKeyInfo(BaseModel):
    id: str
    name: Optional[str] = None
    user_id: str
    organization_id: Optional[str] = None
    role: Optional[str] = None
    expiry_date: datetime
    is_active: bool
    last_used: Optional[datetime] = None
    created_at: datetime
    usage_count: int

class APIKeyValidationRequest(BaseModel):
    api_key: str