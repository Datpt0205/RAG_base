from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union, Any


class ProductInfo(BaseModel):
    name: str = Field(description="Name of the product")
    price: str = Field(description="Price of the product")
    category: str = Field(description="Category of the product")
    description: str = Field(description="Description of the product")


class ProductStats(BaseModel):
    average_rating: float = Field(description="Average rating of the product", ge=0, le=5)
    total_reviews: int = Field(description="Total number of reviews")


class Review(BaseModel):
    rating: int = Field(description="Rating given by the reviewer", ge=1, le=5)
    content: str = Field(description="Content of the review")


class ReviewGenerationRequest(BaseModel):
    product_info: ProductInfo = Field(description="Information about the product")
    stats: ProductStats = Field(description="Statistics about the product reviews")
    reviews: List[Review] = Field(description="List of existing reviews for context")
    tone: Optional[str] = Field(default="neutral", description="Tone of the generated review (positive, negative, neutral)")
    rating: Optional[int] = Field(default=None, description="Rating for the generated review (1-5)", ge=1, le=5)
    length: Optional[str] = Field(default="medium", description="Length of the generated review (short, medium, long)")


class ProductItem(BaseModel):
    name: Optional[str] = Field(default="", description="Name of the product")
    price: Optional[str] = Field(default="", description="Price of the product")
    description: Optional[str] = Field(default="", description="Description of the product")
    stats: Optional[ProductStats] = Field(default=None, description="Statistics about the product")
    reviews: Optional[List[Review]] = Field(default=None, description="Reviews of the product")

    # default values for stats and reviews
    def __init__(self, **data):
        super().__init__(**data)
        if self.stats is None:
            self.stats = ProductStats()
        if self.reviews is None:
            self.reviews = []


class StoreInfo(BaseModel):
    name: Optional[str] = Field(default="", description="Name of the store")
    location: Optional[str] = Field(default="", description="Location of the store")
    average_rating: Optional[float] = Field(default=4.0, description="Average rating of the store")

class StoreCategory(str, Enum):
    RESTAURANT = "restaurant"
    RETAIL_CLOTHING = "retail_clothing"
    GROCERY = "grocery"
    ELECTRONICS = "electronics"
    BEAUTY_COSMETICS = "beauty_cosmetics"
    BOOKS_STATIONERY = "books_stationery"
    PHARMACY = "pharmacy"
    AUTOMOTIVE = "automotive"
    HOME_GARDEN = "home_garden"
    SPORTS_OUTDOORS = "sports_outdoors"
    JEWELRY = "jewelry"
    PET_SUPPLIES = "pet_supplies"
    SERVICES = "services"
    ENTERTAINMENT = "entertainment"
    OTHER = ""

class StoreReviewRequestProvider(BaseModel):
    store: Optional[StoreInfo] = Field(default=None, description="Information about the store")
    store_category: Optional[StoreCategory] = Field(description="Primary business category of the store")
    product_items: Optional[List[ProductItem]] = Field(default=None, description="List of products to include in review")
    rating: Optional[int] = Field(default=None, description="Overall rating for the store (1-5)", ge=1, le=5)
    length: Optional[str] = Field(default="medium", description="Length of the generated review (short, medium, long)")
    user_note: Optional[str] = Field(default="", description="Additional notes or specific points to include")
    system_language: Optional[str] = Field(default="auto", description="Language to generate the review in")
    provider_type: Optional[str] = Field(default="openai", description="Provider type: ollama, openai, gemini")

    def __init__(self, **data):
        super().__init__(**data)
        if self.store is None:
            self.store = StoreInfo()
        if self.product_items is None:
            self.product_items = []
        if self.rating is None:
            if self.store and hasattr(self.store, 'average_rating') and self.store.average_rating is not None:
                self.rating = int(round(self.store.average_rating))
            else:
                self.rating = 4