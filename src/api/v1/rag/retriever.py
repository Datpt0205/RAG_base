from typing import Annotated, Dict, Any
from fastapi import APIRouter, Response, Query, status, Request, Depends, HTTPException
from src.core.database.services.collection_management_service import CollectionManagementService
from src.core.schemas.response import BasicResponse
from src.features.rag.handlers.api_key_auth_handler import APIKeyAuth
from src.core.utils.config import settings
from src.features.rag.handlers.retrieval_handler import default_search_retrieval



api_key_auth = APIKeyAuth()
router = APIRouter(dependencies=[Depends(api_key_auth.author_with_api_key)])

collection_service = CollectionManagementService()

@router.post("/retrieval/retriever", response_description="Retriever")
async def retriever(
    request: Request,
    response: Response,
    query: Annotated[str, Query()],
    top_k: Annotated[int, Query()] = 5,
    collection_name: Annotated[str, Query()] = settings.QDRANT_COLLECTION_NAME,
    api_key_data: Dict[str, Any] = Annotated[Dict[str, Any], Depends(api_key_auth.author_with_api_key)]
):
    """
    Retrieve and rerank documents from the vector database.
    Uses singleton instance to avoid reloading models.
    
    Args:
        request: The request object with state info from API key auth
        query: Query string for retrieval
        top_k: Number of top results to return
        collection_name: Name of the collection to search
        
    Returns:
        BasicResponse: Response with retrieved documents
    """
    # Get info user_id and organization_id from request state
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    has_access = await default_search_retrieval.check_collection_access(
        user_id=user_id,
        collection_name=collection_name,
        organization_id=organization_id,
        required_permission="read"
    )
    
    if not has_access:
        response.status_code = status.HTTP_403_FORBIDDEN
        return BasicResponse(
            status="Failed",
            message=f"You don't have permission to access collection {collection_name}",
            data=None
        )
    
    # Use the singleton instance
    resp = await default_search_retrieval.qdrant_retrieval(
        query=query, 
        top_k=top_k, 
        collection_name=collection_name
    )
    
    if resp:
        response.status_code = status.HTTP_200_OK
        data = [docs.json() for docs in resp]
        
        return BasicResponse(
            status="Success",
            message="Success retriever data from vector database",
            data=data
        )
    else:
        return BasicResponse(
            status="Failed",
            message="Failed retriever data from vector database",
            data=resp
        )