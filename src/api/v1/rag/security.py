from src.core.schemas.auth import APIKeyCreate, APIKeyValidationRequest
from src.core.schemas.response import BasicResponse
from fastapi.routing import APIRouter
from fastapi import status, Response, Request, Depends, HTTPException, Body
from typing import Dict, Any

from src.features.rag.handlers.api_key_auth_handler import APIKeyAuth

# from src.core.schemas.auth import *
# from src.core.schemas.response import BasicResponse
# from src.features.rag.handlers.api_key_auth_handler import APIKeyAuth
# from src.core.utils.logger.custom_logging import LoggerMixin


router = APIRouter()

# API key authentication instance
api_key_auth = APIKeyAuth()


@router.post('/api-keys/create', response_description='Create new API key', response_model=BasicResponse)
async def create_api_key(response: Response, api_key_data: APIKeyCreate = Body(...)) -> Dict:
    
    try:
        # user_id & organization_id: get input from FE
        api_key_info = api_key_auth.create_api_key(
            user_id=api_key_data.user_id, 
            organization_id=api_key_data.organization_id,
            name=api_key_data.name,
            expires_in_days=api_key_data.expires_in_days
        )
        
        resp = BasicResponse(
            status='success',
            message='API key has been created successfully.',
            data=api_key_info
        )
        response.status_code = status.HTTP_201_CREATED

    except ValueError as ve:
        resp = BasicResponse(
            status='failed',
            message=f'Unable to generate API key: {str(ve)}'
        )
        response.status_code = status.HTTP_400_BAD_REQUEST

    except Exception as e:
        resp = BasicResponse(
            status='failed',
            message=f'Unable to generate API key: {str(e)}'
        )
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    return resp


@router.post('/api-keys/validate', response_description="Validate an API key", response_model=BasicResponse)
async def validate_api_key(
    payload: APIKeyValidationRequest,
    request: Request,
    response: Response
) -> BasicResponse:
    try:
        organization_id = request.headers.get("X-Organization-Id")

        api_key_data = await api_key_auth.author_with_api_key(
            api_key=payload.api_key,
            organization_id=organization_id,
            request=request
        )

        response.status_code = status.HTTP_200_OK
        return BasicResponse(
            status="success",
            message="API key is valid.",
            data={
                "apiKeyValue": payload.api_key,
                "createdDate": api_key_data["created_at"].isoformat(),
                "expiryDate": api_key_data["expiry_date"].isoformat(),
            }
        )
    except Exception as e:
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return BasicResponse(
            status="failed",
            message="API Key is invalid or expired.",
            data=str(e)  
        )


@router.get("/api-keys/{user_id}", response_description="Get User API Keys")
async def get_user_api_keys(user_id: str, request: Request, current_api_key: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)):
    if current_api_key["user_id"] != user_id:
        # Check if current user has admin rights
        if "role" in current_api_key and current_api_key["role"] == "ADMIN":
            # Admin can view other users' API keys
            pass
        else:
            # Users can usually only view their own API keys
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only view your own API keys"
            )
    
    user_api_keys = api_key_auth.get_user_api_keys(user_id)
    
    # Mask API key values for security
    for key in user_api_keys:
        if "api_key" in key:
            key["api_key"] = f"{key['api_key'][:10]}..." if key.get("api_key") else None
    
    return BasicResponse(
        status="success",
        message="User API keys retrieved successfully",
        data=user_api_keys
    )


@router.post('/api-keys/{api_key_id}/revoke', response_description='Revoke API key', response_model=BasicResponse)
async def revoke_api_key(response: Response, api_key_id: str, current_api_key: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)) -> Dict:
    """
    Revoke API key (disable but not delete)

    Args:
        api_key_id: ID of API key to revoke (Note: fill api_key_id not api_key)
        current_api_key: API key in use (from authentication)

    Returns:
        BasicResponse: Result of revocation
    """
    try:
        success = api_key_auth.revoke_api_key(api_key_id, current_api_key["user_id"])
        
        if success:
            resp = BasicResponse(
                status='success',
                message='API key has been successfully revoked'
            )
            response.status_code = status.HTTP_200_OK
        else:
            resp = BasicResponse(
                status='failed',
                message='API key not found or no revocation permission'
            )
            response.status_code = status.HTTP_404_NOT_FOUND
    except Exception as e:
        resp = BasicResponse(
            status='failed',
            message=f'Unable to revoke API key: {str(e)}'
        )
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    return resp


@router.delete('/api-keys/{api_key_id}', response_description='Delete API key', response_model=BasicResponse)
async def delete_api_key(response: Response, api_key_id: str, current_api_key: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)) -> Dict:
    """
    Delete the API key completely from the system

    Args:
        api_key_id: ID of the API key to delete
        current_api_key: API key currently in use (from authentication)

    Returns:
        BasicResponse: Result of deletion
    """
    try:
        success = api_key_auth.delete_api_key(api_key_id, current_api_key["user_id"])
        
        if success:
            resp = BasicResponse(
                status='success',
                message='API key deleted successfully'
            )
            response.status_code = status.HTTP_200_OK
        else:
            resp = BasicResponse(
                status='failed',
                message='API key not found or no permission to delete'
            )
            response.status_code = status.HTTP_404_NOT_FOUND
    except Exception as e:
        resp = BasicResponse(
            status='failed',
            message=f'Unable to delete API key: {str(e)}'
        )
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    return resp


@router.get('/api-keys/{user_id}/organizations', response_description='Get a list of user organizations', response_model=BasicResponse)
async def get_user_organizations(response: Response, user_id: str, current_api_key: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)) -> Dict:
    """
    Get the list of user's organizations and roles

    Args:
        user_id: User ID
        current_api_key: API key in use (from authentication)

    Returns:
        BasicResponse: List of organizations and roles
    """
    if current_api_key["user_id"] != user_id:
        resp = BasicResponse(
            status='failed',
            message="No access to other users' organization information"
        )
        response.status_code = status.HTTP_403_FORBIDDEN
        return resp
    
    try:
        organizations = api_key_auth.get_user_organizations(user_id)
        
        resp = BasicResponse(
            status='success',
            message=f'Found {len(organizations)} organizations',
            data=organizations
        )
        response.status_code = status.HTTP_200_OK
    except Exception as e:
        resp = BasicResponse(
            status='failed',
            message=f'Unable to get organization list: {str(e)}'
        )
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    return resp


# @router.get('/user/{user_id}/clear_cache', response_description='Clear cache', response_model=BasicResponse)
# async def clear_cache(response: Response, user_id: str, organization_id: str, current_api_key: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)) -> Dict:
#     if current_api_key["user_id"] != user_id:
#         resp = BasicResponse(
#             status='failed',
#             message="No access to other users' organization information"
#         )
#         response.status_code = status.HTTP_403_FORBIDDEN
#         return resp
    
#     try:
#         result = api_key_auth.cache_manager(user_id, organization_id)
        
#         resp = BasicResponse(
#             status='success',
#             message=f'Clear cache successful',
#             data=result
#         )
#         response.status_code = status.HTTP_200_OK
#     except Exception as e:
#         resp = BasicResponse(
#             status='failed',
#             message=f'Unable to clear cache: {str(e)}'
#         )
#         response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
#     return resp