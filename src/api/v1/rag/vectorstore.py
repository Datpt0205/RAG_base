from src.core.database.repository.file_repository import FileProcessingRepository
from src.core.database.services.collection_management_service import CollectionManagementService
from src.core.schemas.response import BasicResponse
from fastapi.routing import APIRouter
from fastapi import status, Response, Depends, Request, HTTPException, Query, Header
from typing import Dict, Any, List, Optional
from src.features.rag.handlers.api_key_auth_handler import APIKeyAuth
from src.features.rag.handlers.vector_store_handler import VectorStoreQdrant
from src.core.utils.constants import TypeDatabase

from src.core.utils.config import settings

router = APIRouter(prefix="/vectorstore")

# API key authentication instance
api_key_auth = APIKeyAuth()
file_management_dal = FileProcessingRepository()


@router.post('/create_collection', response_description='Create collection in Qdrant')
async def create_collection(
    request: Request,
    response: Response,
    collection_name: str,
    is_personal: bool = Query(False, description="Whether this is a personal collection"),
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    try:
        # Get user_id and organization_id information from request state
        user_id = getattr(request.state, "user_id", None)
        organization_id = getattr(request.state, "organization_id", None)
        
        if not user_id:
            response.status_code = status.HTTP_401_UNAUTHORIZED
            return BasicResponse(
                status="Failed",
                message="User authentication required",
                data=None
            )
        
        # If this is an organizational collection, check ADMIN rights
        if not is_personal and organization_id:
            user_role = getattr(request.state, "role", None)
            if user_role != "ADMIN":
                response.status_code = status.HTTP_403_FORBIDDEN
                return BasicResponse(
                    status="Failed",
                    message="Only administrators can create organizational collections",
                    data=None
                )
        
        # Create user object from authenticated information
        user = {
            "id": user_id,
            "role": getattr(request.state, "role", None)
        }
        
        resp = VectorStoreQdrant().create_qdrant_collection(
            collection_name=collection_name, 
            user=user,
            organization_id=organization_id if not is_personal else None,
            is_personal=is_personal
        )
        
        if resp.data:
            response.status_code = status.HTTP_200_OK
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
        
        return resp
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return BasicResponse(
            status="Failed",
            message=f"Error creating collection: {str(e)}",
            data=None
        )


@router.delete("/collection/{collection_name}", response_description="Delete entire collection and all related documents")
async def delete_collection_with_documents(
    collection_name: str,
    response: Response,
    request: Request,
    type_db: str = Query(
        default=TypeDatabase.Qdrant.value,
        enum=TypeDatabase.list(),
        description="Select vector database type"
    ),
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Delete a collection along with all its documents in both PostgreSQL and Qdrant.
    This is useful for when a chat session is deleted and all its related documents should be removed.
    """
    organization_id = getattr(request.state, "organization_id", None)
    user_role = getattr(request.state, "role", None)
    user_id = getattr(request.state, "user_id", None)
    
    try:
        # Check access/delete collection
        collection_service = CollectionManagementService()
        has_permission = collection_service.check_collection_permission(
            user_id=user_id,
            collection_name=collection_name,
            organization_id=organization_id,
            required_permission="delete"
        )
        
        if not has_permission and user_role != "ADMIN":
            response.status_code = status.HTTP_403_FORBIDDEN
            return BasicResponse(
                status="failed",
                message="You don't have permission to delete this collection",
                data=None
            )
        
        # 1. Get a list of all documents in a collection
        documents = file_management_dal.get_files_by_collection(collection_name, organization_id)
        document_count = len(documents)
        
        # 2. Delete all documents in PostgreSQL
        file_management_dal.delete_record_by_collection(collection_name, organization_id)
        
        # 3. Delete collection in Qdrant
        vector_store = VectorStoreQdrant()
        result = vector_store.delete_qdrant_collection(
            collection_name=collection_name,
            user={"id": user_id, "role": user_role},
            organization_id=organization_id,
            is_personal=(user_role != "ADMIN")  # Assume collection is personal if user is not admin
        )
        
        response.status_code = status.HTTP_200_OK
        return BasicResponse(
            status="success",
            message=f"Successfully deleted collection '{collection_name}' with {document_count} documents",
            data={"collection_name": collection_name, "document_count": document_count}
        )
        
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return BasicResponse(
            status="error",
            message=f"Error deleting collection: {str(e)}",
            data=None
        )


@router.get('/list_collections', response_description='List all collections in Qdrant')
async def list_collections(
    request: Request,
    response: Response,
    include_personal: bool = Query(True, description="Include personal collections"),
    include_organizational: bool = Query(True, description="Include organizational collections"),
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    try:
        # Get user_id and organization_id information from request state
        user_id = getattr(request.state, "user_id", None)
        organization_id = getattr(request.state, "organization_id", None)
        user_role = getattr(request.state, "role", None)
        
        if not user_id:
            response.status_code = status.HTTP_401_UNAUTHORIZED
            return {"status": "Failed", "message": "User authentication required", "data": None}
        
        # Create user object from authenticated information
        user = {
            "id": user_id,
            "role": user_role,
            "is_admin": user_role == "ADMIN"
        }
        
        # Get collection list with filtering by organization_id
        try:
            collections = VectorStoreQdrant().list_qdrant_collections(
                user=user, 
                organization_id=organization_id,
                include_personal=include_personal,
                include_organizational=include_organizational
            )
            
            personal_collections = []
            org_collections = []
            
            for collection in collections:
                if collection.get("is_personal"):
                    personal_collections.append(collection)
                else:
                    org_collections.append(collection)
            
            result = {
                "personal_collections": personal_collections if include_personal else [],
                "organizational_collections": org_collections if include_organizational else []
            }
            
            response.status_code = status.HTTP_200_OK
            return {"status": "Success", "message": "List collections success", "data": result}
        except Exception as e:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"status": "Failed", "message": str(e), "data": None}
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"status": "Failed", "message": f"Error listing collections: {str(e)}", "data": None}


# ======================= Functions for RankMate project =======================
@router.post('/create_collection_store', response_description='Create collection in Qdrant for RankMate')
async def create_collection_simple(
    request: Request,
    response: Response,
    collection_name: str = Query(..., description="Name of the collection to create"),
):
    """
    Create a collection in Qdrant without user/organization management
    No metadata is saved to PostgreSQL - Qdrant only
    """
    try:
        
        # Create collection using simplified method
        resp = VectorStoreQdrant().create_qdrant_collection_store(
            collection_name=collection_name
        )
        
        if resp.status == "Success":
            response.status_code = status.HTTP_200_OK
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
        
        return resp
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return BasicResponse(
            status="Failed",
            message=f"Error creating collection: {str(e)}",
            data=None
        )


@router.delete("/collection_store/{collection_name}", response_description="Delete collection from Qdrant for RankMate")
async def delete_collection_store(
    collection_name: str,
    response: Response,
    request: Request,
):
    """
    Delete a collection from Qdrant only
    No PostgreSQL cleanup - Qdrant only
    """
    try:
        
        # Delete collection using simplified method
        vector_store = VectorStoreQdrant()
        result = vector_store.delete_qdrant_collection_store(
            collection_name=collection_name
        )
        
        if result.status == "Success":
            response.status_code = status.HTTP_200_OK
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
        
        return result
        
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return BasicResponse(
            status="Failed",
            message=f"Error deleting collection: {str(e)}",
            data=None
        )


@router.get('/list_collections_store', response_description='List all collections in Qdrant for RankMate')
async def list_collections_simple(
    request: Request,
    response: Response,
):
    """
    List all collections in Qdrant without metadata filtering
    No PostgreSQL involvement - Qdrant only
    """
    try:
        collections = VectorStoreQdrant().list_qdrant_collections_store()
        
        response.status_code = status.HTTP_200_OK
        return BasicResponse(
            status="Success", 
            message=f"Found {len(collections)} collections in Qdrant", 
            data={
                "collections": collections,
                "total_count": len(collections),
                "source": "qdrant_only"
            }
        )
    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return BasicResponse(
            status="Failed", 
            message=str(e), 
            data=None
        )