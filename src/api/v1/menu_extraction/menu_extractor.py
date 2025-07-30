from typing import Dict, Any
from src.core.providers.provider_factory import ModelProviderFactory, ProviderType
from src.core.schemas.menu import DishDescriptionRequest, DishDescriptionResponse, MenuExtractionRequest, MenuExtractionResponse
from src.core.utils.logger.custom_logging import LoggerMixin
from fastapi import APIRouter, Response, status, Depends, Request, HTTPException
import time

from src.features.rag.handlers.api_key_auth_handler import APIKeyAuth
from src.features.menu_extraction.handlers.menu_extraction_handler import MenuExtractionHandler
from src.core.utils.config import settings

router = APIRouter(prefix="/menu")
menu_handler = MenuExtractionHandler()
logger = LoggerMixin().logger

api_key_auth = APIKeyAuth()

@router.post("/menu-extraction/", response_description="Extract menu", response_model=MenuExtractionResponse)
async def extract_menu_with_provider(
    response: Response,
    request: Request,
    extraction_request: MenuExtractionRequest,
    # api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Extract menu items using provider settings pattern (similar to chat endpoints).
    """
    start_time = time.time()
    
    try:

        api_key = ModelProviderFactory._get_api_key(extraction_request.provider_type)

        # Validate provider
        if extraction_request.provider_type not in ProviderType.list():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid provider. Must be one of: {', '.join(ProviderType.list())}"
            )
        
        menu_items = await menu_handler.extract_menu(
            image_path=extraction_request.image_path,
            provider_type=extraction_request.provider_type,
            model_name=extraction_request.model_name,
            api_key=api_key
        )

        if not menu_items:
            logger.warning("No menu items extracted")
            response.status_code = status.HTTP_200_OK
            return MenuExtractionResponse(
                status="success",
                message="No menu items found in the image",
                data=[]
            )
        
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Thời gian thực hiện: {elapsed_time:.4f} giây") # TODO: 41.1937 giây cần tối ưu. 
        
        response.status_code = status.HTTP_200_OK

        return MenuExtractionResponse(
            status="success",
            message=f"Successfully extracted {len(menu_items)} menu items",
            data=menu_items
        )
        
    except Exception as e:
        logger.error(f"Menu extraction error: {str(e)}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return MenuExtractionResponse(
            status="error",
            message=f"Failed to extract menu: {str(e)}",
            data=None
        )


@router.post("/generate-description/", response_model=DishDescriptionResponse)
async def generate_dish_description(
    response: Response,
    request: DishDescriptionRequest, 
    # api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Generate detailed dish description for image generation
    
    Args:
        request: Dish information
        provider_type: AI provider (currently only openai)
        model_name: Model to use
        api_key_header: API key for authentication
    """
    
    try:
        # Validate provider
        if request.provider_type not in ProviderType.list():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid provider. Must be one of: {', '.join(ProviderType.list())}"
            )
        
        # Get API key for provider
        api_key = ModelProviderFactory._get_api_key(request.provider_type)
        
        result = await menu_handler.generate_dish_description(
            request=request,
            provider_type=request.provider_type,
            model_name=request.model_name,
            api_key=api_key
        )
        
        response.status_code = status.HTTP_200_OK
        return DishDescriptionResponse(
            status="success",
            message="Successfully generated dish description",
            data=result
        )
        
    except Exception as e:
        logger.error(f"Error generating description: {str(e)}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return DishDescriptionResponse(
            status="error",
            message=f"Failed to generate description: {str(e)}",
            data=None
        )