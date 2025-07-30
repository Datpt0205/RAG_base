from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from src.core.providers.provider_factory import ProviderType
from src.core.utils.constants import OpenAIModelName
from enum import Enum

class MenuItem(BaseModel):
    """Schema for a single menu item"""
    name: str = Field(..., description="Name of the dish or item")
    price: Optional[float] = Field(None, description="Price of the item")
    currency: Optional[str] = Field(None, description="Currency (VND, USD, EUR, ...)")
    description: Optional[str] = Field(None, description="Description or ingredients")
    category: Optional[str] = Field(None, description="Category of the item (e.g., appetizer, main course)")


class MenuExtractionRequest(BaseModel):
    """Request schema for menu extraction"""
    image_path: str = Field(..., description="Path to the menu image file")
    provider_type: str = Field(ProviderType.OPENAI, description="LLM provider: openai or ollama")
    model_name: Optional[str] = Field(OpenAIModelName.GPT41Nano, description="Model name (e.g., gpt-4o-mini for OpenAI, llava:latest for Ollama)")


class MenuExtractionResponse(BaseModel):
    """Response schema for menu extraction"""
    status: str
    message: str
    data: Optional[List[MenuItem]] = None


# Schama generate description for dish
class CulturalStyle(str, Enum):
    ASIAN = "Asian cuisines"
    EUROPEAN = "European cuisines"
    AMERICAS = "Americas"
    MIDDLE_EASTERN_AFRICAN = "Middle Eastern/African"

class DishDescriptionRequest(BaseModel):
    name: str = Field(..., description="Name of the dish/beverage")
    price: Optional[float] = Field(None, description="Price of the item")
    currency: Optional[str] = Field(None, description="Currency code (VND, USD, etc.)")
    description: Optional[str] = Field(None, description="Original description from menu")
    category: Optional[str] = Field(None, description="Category of the dish")
    cultural_style: Optional[CulturalStyle] = Field(None, description="Cultural cuisine style")
    country: Optional[str] = Field(None, description="Country of origin or style")
    provider_type: str = Field(ProviderType.OPENAI, description="AI provider")
    model_name: str = Field(OpenAIModelName.GPT41Nano, description="Model to use")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Dish name cannot be empty")
        return v.strip()

class DishDescriptionResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None 