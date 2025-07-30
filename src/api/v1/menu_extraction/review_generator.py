from typing import Dict, Any, Optional
from src.core.schemas.response import BasicResponse
from src.core.schemas.review_generator import StoreReviewRequestProvider
from src.core.utils.constants import OpenAIModelName
from fastapi import APIRouter, Response, status, Request, Depends, Body, Header

from src.features.menu_extraction.handlers.store_review_generator_handler import StoreReviewGeneratorHandler
from src.core.utils.config import settings
from src.features.rag.handlers.api_key_auth_handler import APIKeyAuth

router = APIRouter(prefix="/review-generator")

# Initialize handler
store_review_generator = StoreReviewGeneratorHandler()
api_key_auth = APIKeyAuth()

@router.post("/generate-store-review-provider", 
             response_description="Generate comprehensive store review", 
             response_model=BasicResponse)
async def generate_store_review_with_provider(
    request: Request,
    response: Response,
    model_name: str = OpenAIModelName.GPT41Nano,
    review_request: StoreReviewRequestProvider = Body(...),
    # api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Generate a comprehensive review based on multiple product experiences at a store or restaurant.
    
    This enhanced endpoint supports multiple model providers (Ollama, OpenAI, Gemini) through
    provider_settings in the request.
    
    Args:
        request: Request object with auth info
        response: Response object
        review_request: Store review generation request with store info, product list, and provider settings
        api_key_data: API key authentication data
        
    Returns:
        BasicResponse: Generated review with standard response format
    """
    
    try:
        # Extract provider settings
        provider_type = review_request.provider_type
        
        # Generate review with specified provider
        result = await store_review_generator.generate_store_review_provider(
            request=review_request,
            model_name=model_name,
            provider_type=provider_type
        )
        
        product_names = []
        if review_request.product_items:
            product_names = [item.name for item in review_request.product_items if item.name]

        response.status_code = status.HTTP_200_OK
        return BasicResponse(
            status="Success",
            message="Store review generated successfully",
            data={
                "review": result["review"],
                "rating": result["rating"],
                "store_name": review_request.store.name if review_request.store else "",
                "location": review_request.store.location if review_request.store else "",
                "products_reviewed": product_names,
                "provider_used": provider_type
            }
        )
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return BasicResponse(
            status="Failed",
            message=f"Error generating store review: {str(e)}",
            data=None
        )