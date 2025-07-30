import uuid
from fastapi import APIRouter, Response, Query, status, Depends, Body, Request
from typing import Annotated, List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from src.features.rag.handlers.api_key_auth_handler import APIKeyAuth
from src.core.schemas.response import BasicResponse
from src.features.rag.handlers.rerank_handler import default_reranker



router = APIRouter()
api_key_auth = APIKeyAuth()

class Candidate(BaseModel):
    content: str
    doc_id: Optional[str] = None
    organization_id: Optional[str] = None

    @validator("doc_id", pre=True, always=True)
    def generate_doc_id(cls, value):
        if value is None or value == "":
            return str(uuid.uuid4())
        return value

class RerankRequest(BaseModel):
    candidates: List[Candidate]

@router.post("/reranking/rerank", response_description="Rerank")
async def rerank_endpoint(
    request: Request,
    response: Response,
    query: Annotated[str, Query()] = None,
    threshold: Annotated[float, Query()] = 0.4,
    request_body: RerankRequest = Body(...),
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Rerank candidates based on their relevance to the query.
    Uses the singleton reranker instance to avoid reloading models.
    
    Args:
        request: Request object with user authentication info
        query: Query string for reranking
        threshold: Score threshold for filtering results
        request_body: Request body with candidates
        
    Returns:
        BasicResponse: Response with reranked results
    """
    # organization_id = getattr(request.state, "organization_id", None)
    
    # Log input parameters
    print(f"Query reranking: {query}")
    # print(f"Threshold: {threshold}")
    # print(f"Organization ID: {organization_id}")
    # print(f"Number of candidates: {len(request_body.candidates)}")
    
    # Convert candidates to objects for easier debugging
    candidates = request_body.candidates
    
    # for i, candidate in enumerate(candidates[:3]):
    #     print(f"Candidate {i}: doc_id={candidate.doc_id}, org_id={candidate.organization_id}, content preview={candidate.content[:50]}...")
    
    try:
        # Call reranker but DO NOT filter by organization_id
        # Because organization_id in candidates is only for storing information, not for filtering
        result = default_reranker.process_candidates(candidates, query, threshold)
        
        # Logs
        # print(f"Reranking result: {len(result)} items found with threshold {threshold}")
        if len(result) == 0 and len(candidates) > 0:
            # Retry with lower threshold if no results
            # print("No results with current threshold, trying with lower threshold")
            result = default_reranker.process_candidates(candidates, query, 0.1)
            # print(f"Reranking with lower threshold: {len(result)} items found")

        result_response = BasicResponse(
            status="success",
            message="Reranking is successful!",
            data=result
        )
        response.status_code = status.HTTP_200_OK
    except Exception as e:
        print(f"Reranking error: {str(e)}")
        # Create a failure response in case of any issues
        result_response = BasicResponse(
            status="fail",
            message=f"Reranking failed: {str(e)}",
            data=[]
        )
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    return result_response