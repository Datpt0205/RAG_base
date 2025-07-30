import json

from src.core.utils.config import settings
from src.core.providers.provider_factory import ProviderType
from src.core.schemas.response import BasicResponse, ChatResponse
from src.core.utils.logger.custom_logging import LoggerMixin
from src.features.rag.handlers.api_key_auth_handler import APIKeyAuth
from src.features.rag.handlers.llm_chat_handler import ChatHandler, ChatMessageHistory
from pydantic import BaseModel, Field
from collections.abc import AsyncGenerator
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, Response, Query, status, Depends, Request, Body, HTTPException
from typing import Dict, Any, Optional


router = APIRouter()
api_key_auth = APIKeyAuth()
chat_handler = ChatHandler()

logger = LoggerMixin().logger


# API /chat
class ChatRequest(BaseModel):
    session_id: str
    question_input: str
    model_name: str = 'qwen3:14b'
    collection_name: str = settings.QDRANT_COLLECTION_NAME
    use_multi_collection: bool = False


# API /chat/provider
class ChatRequestWithProvider(BaseModel):
    session_id: str
    question_input: str
    model_name: str = 'gpt-4.1-nano-2025-04-14'
    collection_name: str = 'collection'
    use_multi_collection: bool = False
    enable_thinking: bool = True
    provider_type: str = Field(ProviderType.OPENAI, description="Provider type: ollama, openai, gemini")

class SmartChatRequest(BaseModel):
    session_id: str
    question_input: str
    model_name: str = 'gpt-4.1-nano-2025-04-14'
    collection_name: str = settings.QDRANT_COLLECTION_NAME
    use_multi_collection: bool = False
    enable_thinking: bool = True
    enable_reasoning: bool = False
    provider_type: str = Field(ProviderType.OPENAI, description="Provider type: ollama, openai, gemini")


# Helper function for formatting SSE
async def format_sse(generator) -> AsyncGenerator[str, None]:
    """Format an async generator into Server-Sent Events format"""
    async for chunk in generator:
        if chunk:
            # Format according to SSE standard
            yield f"data: {json.dumps({'content': chunk})}\n\n"
    yield "data: [DONE]\n\n"


# Helper function to format SSE response
async def format_sse(generator, session_id) -> AsyncGenerator[str, None]:
    async for chunk in generator:
        if chunk:
            # Format according to SSE standard with correct ChatResponse structure
            response = {
                "id": session_id,
                "role": "assistant", 
                "content": chunk
            }
            yield f"{json.dumps(response)}\n\n"
    yield "[DONE]\n\n"


# ======================= DEFINE API ENDPOINTS =======================
@router.post("/chat/smart", response_description="Smart router chat with adaptive retrieval", response_model=ChatResponse)
async def smart_chat(
    request: Request,
    response: Response,
    chat_request: SmartChatRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Smart chat endpoint that can decide whether to use retrieval or answer directly.
    
    This endpoint analyzes the query to determine if it needs knowledge retrieval
    or can be answered directly from the model's capabilities.
    
    Args:
        request: Request object
        response: Response object
        chat_request: Smart chat request parameters
        api_key_data: API key authentication data
        
    Returns:
        ChatResponse: Response with generated text
    """
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)

    # Extract provider type
    provider_type = chat_request.provider_type
    
    # Call handler method
    chat_handler = ChatHandler()
    resp = await chat_handler.handle_smart_chat(
        session_id=chat_request.session_id,
        question_input=chat_request.question_input,
        model_name=chat_request.model_name,
        collection_name=chat_request.collection_name,
        provider_type=provider_type,
        user_id=user_id,
        organization_id=organization_id,
        use_multi_collection=chat_request.use_multi_collection,
        enable_thinking=chat_request.enable_thinking,
        enable_reasoning=chat_request.enable_reasoning
    )
                                       
    if resp.status == "Success" and resp.data:
        response.status_code = status.HTTP_200_OK
        content = resp.data.get("content", "")
        return ChatResponse(
            id=chat_request.session_id,
            role="assistant",
            content=content
        )
    else:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return ChatResponse(
            id=chat_request.session_id,
            role="assistant",
            content=f"Error: {resp.message}"
        )


@router.post("/chat/smart/stream", response_description="Stream smart router chat with adaptive retrieval")
async def smart_chat_stream(
    request: Request,
    chat_request: SmartChatRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Streaming version of the smart chat endpoint.
    
    Returns a Server-Sent Events (SSE) stream with response chunks.
    """
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    # Extract provider type
    provider_type = chat_request.provider_type
    
    # Return streaming response
    chat_handler = ChatHandler()
    return StreamingResponse(
        format_sse(
            chat_handler.handle_smart_chat_stream(
                session_id=chat_request.session_id,
                question_input=chat_request.question_input,
                model_name=chat_request.model_name,
                collection_name=chat_request.collection_name,
                provider_type=provider_type,
                user_id=user_id,
                organization_id=organization_id,
                use_multi_collection=chat_request.use_multi_collection,
                enable_thinking=chat_request.enable_thinking,
                enable_reasoning=chat_request.enable_reasoning
            ),
            session_id=chat_request.session_id
        ),
        media_type="text/event-stream"
    )


@router.post("/chat/provider/chat-agent", response_description="Chat with ReAct and Chain of Thought", response_model=ChatResponse)
async def chat_provider_reasoning(
    request: Request,
    response: Response,
    chat_request: ChatRequestWithProvider,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Advanced reasoning chat endpoint combining ReAct (Reasoning and Acting) 
    with CoT (Chain of Thought) for higher accuracy responses.
    
    This approach makes the model:
    1. Break down the question into clear steps
    2. Think explicitly through possible approaches
    3. Reason about information from context
    4. Arrive at a more reliable answer
    
    Ideal for complex questions requiring deeper reasoning.
    """
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    # Extract provider type
    provider_type = chat_request.provider_type
    
    # Call handler method with ReAct+CoT
    chat_handler = ChatHandler()

    resp = await chat_handler.handle_chat_agent_provider_reasoning(
        session_id=chat_request.session_id,
        question_input=chat_request.question_input,
        model_name=chat_request.model_name,
        collection_name=chat_request.collection_name,
        provider_type=provider_type,
        user_id=user_id,
        organization_id=organization_id,
        use_multi_collection=chat_request.use_multi_collection,
        enable_thinking=chat_request.enable_thinking
    )
                                           
    if resp.status == "Success" and resp.data:
        response.status_code = status.HTTP_200_OK
        content = resp.data if isinstance(resp.data, str) else str(resp.data)
        return ChatResponse(
            id=chat_request.session_id,
            role="assistant",
            content=content
        )
    else:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return ChatResponse(
            id=chat_request.session_id,
            role="assistant",
            content=f"Error: {resp.message}"
        )



@router.post("/chat/provider/reasoning", response_description="Chat with ReAct and Chain of Thought", response_model=ChatResponse)
async def chat_provider_reasoning(
    request: Request,
    response: Response,
    chat_request: ChatRequestWithProvider,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Advanced reasoning chat endpoint combining ReAct (Reasoning and Acting) 
    with CoT (Chain of Thought) for higher accuracy responses.
    
    This approach makes the model:
    1. Break down the question into clear steps
    2. Think explicitly through possible approaches
    3. Reason about information from context
    4. Arrive at a more reliable answer
    
    Ideal for complex questions requiring deeper reasoning.
    """
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    # Extract provider type
    provider_type = chat_request.provider_type
    
    # Call handler method with ReAct+CoT
    chat_handler = ChatHandler()

    resp = await chat_handler.handle_chat_provider_reasoning(
        session_id=chat_request.session_id,
        question_input=chat_request.question_input,
        model_name=chat_request.model_name,
        collection_name=chat_request.collection_name,
        provider_type=provider_type,
        user_id=user_id,
        organization_id=organization_id,
        use_multi_collection=chat_request.use_multi_collection,
        enable_thinking=chat_request.enable_thinking
    )
                                           
    if resp.status == "Success" and resp.data:
        response.status_code = status.HTTP_200_OK
        content = resp.data if isinstance(resp.data, str) else str(resp.data)
        return ChatResponse(
            id=chat_request.session_id,
            role="assistant",
            content=content
        )
    else:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return ChatResponse(
            id=chat_request.session_id,
            role="assistant",
            content=f"Error: {resp.message}"
        )


@router.post("/chat/provider/reasoning/stream", response_description="Stream chat with ReAct and Chain of Thought")
async def chat_provider_reasoning_stream(
    request: Request,
    chat_request: ChatRequestWithProvider,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Streaming version of the ReAct+CoT chat endpoint.
    
    Returns a Server-Sent Events (SSE) stream with response chunks,
    allowing real-time viewing of the model's reasoning process.
    """
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    # Extract provider type
    provider_type = chat_request.provider_type
    
    # Clean thinking is the opposite of enable_thinking
    clean_thinking = not chat_request.enable_thinking
    
    # Return streaming response
    chat_handler = ChatHandler()

    return StreamingResponse(
        format_sse(
            chat_handler.handle_chat_provider_reasoning_stream(
                session_id=chat_request.session_id,
                question_input=chat_request.question_input,
                model_name=chat_request.model_name,
                collection_name=chat_request.collection_name,
                provider_type=provider_type,
                user_id=user_id,
                organization_id=organization_id,
                use_multi_collection=chat_request.use_multi_collection,
                clean_thinking=clean_thinking,
                enable_thinking=chat_request.enable_thinking
            ),
            session_id=chat_request.session_id
        ),
        media_type="text/event-stream"
    )


@router.post("/chat/provider", response_description="Chat with specific provider", response_model=ChatResponse)
async def chat_provider(
    request: Request,
    response: Response,
    chat_request: ChatRequestWithProvider,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Chat endpoint with specific provider and prompt management.
    
    Allows using different LLM providers (Ollama, OpenAI, Gemini)
    with intelligent prompt management.
    """
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    # Extract provider type
    provider_type = chat_request.provider_type
    
    # Call handler method with provider details
    chat_handler = ChatHandler()
    resp = await chat_handler.handle_request_chat_with_provider(
        session_id=chat_request.session_id,
        question_input=chat_request.question_input,
        model_name=chat_request.model_name,
        collection_name=chat_request.collection_name,
        provider_type=provider_type,
        user_id=user_id,
        organization_id=organization_id,
        use_multi_collection=chat_request.use_multi_collection,
        enable_thinking=chat_request.enable_thinking
    )
                                           
    if resp.status == "Success" and resp.data:
        response.status_code = status.HTTP_200_OK
        content = resp.data if isinstance(resp.data, str) else str(resp.data)
        return ChatResponse(
            id=chat_request.session_id,
            role="assistant",
            content=content
        )
    else:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return ChatResponse(
            id=chat_request.session_id,
            role="assistant",
            content=f"Error: {resp.message}"
        )


@router.post("/chat/provider/stream", response_description="Stream chat with specific provider")
async def chat_provider_stream(
    request: Request,
    chat_request: ChatRequestWithProvider,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Streaming chat endpoint with specific provider and prompt management.
    
    Returns a Server-Sent Events (SSE) stream with response chunks.
    """
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    # Extract provider type
    provider_type = chat_request.provider_type
    
    # Return streaming response
    chat_handler = ChatHandler()
    return StreamingResponse(
        format_sse(
            chat_handler.handle_streaming_chat_with_provider(
                session_id=chat_request.session_id,
                question_input=chat_request.question_input,
                model_name=chat_request.model_name,
                collection_name=chat_request.collection_name,
                provider_type=provider_type,
                user_id=user_id,
                organization_id=organization_id,
                use_multi_collection=chat_request.use_multi_collection,
                enable_thinking=chat_request.enable_thinking
            ),
            session_id=chat_request.session_id
        ),
        media_type="text/event-stream"
    )


# Chat with LLM system using SSE format
@router.post("/chat/completions", response_description="Chat with LLM system (SSE format)")
async def chat_completions(
    request: Request,
    chat_request: ChatRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key_or_bearer)
):
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    return StreamingResponse(
        format_sse(
            ChatHandler().handle_streaming_chat(
                session_id=chat_request.session_id,
                question_input=chat_request.question_input,
                model_name=chat_request.model_name,
                collection_name=chat_request.collection_name,
                user_id=user_id,
                organization_id=organization_id,
                use_multi_collection=chat_request.use_multi_collection
            ),
            session_id=chat_request.session_id
        ),
        media_type="text/event-stream"
    )


# Create session chat
@router.post("/sessions/create-session", response_description="Create session")
async def create_session(
    request: Request,
    response: Response,
    user_id: str,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    organization_id = getattr(request.state, "organization_id", None)
    
    request_user_id = getattr(request.state, "user_id", None)
    if request_user_id != user_id:
        user_role = getattr(request.state, "role", None)
        if user_role != "ADMIN":
            response.status_code = status.HTTP_403_FORBIDDEN
            return BasicResponse(
                status="Failed",
                message="You can only create sessions for yourself",
                data=None
            )
    
    resp = ChatHandler().create_session_id(
        user_id=user_id,
        organization_id=organization_id
    )
    
    if resp.data:
        response.status_code = status.HTTP_200_OK
    else:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return resp


# Endpoints delete chat session
@router.delete("/sessions/{session_id}", response_description="Delete chat session completely")
async def delete_chat_session(
    request: Request,
    response: Response,
    session_id: str,
    delete_documents: bool = Query(False, description="Delete related documents"),
    delete_collections: bool = Query(False, description="Delete related collections"),
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Delete a chat session completely, with options to delete related documents and collections
    
    Args:
        request: Request object with user authentication info
        session_id: The ID of the chat session to delete
        delete_documents: Whether to delete documents referenced in the session
        delete_collections: Whether to delete collections containing the documents
        
    Returns:
        BasicResponse: Response indicating success or failure
    """
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    # Call the handler to perform the deletion
    resp = await ChatHandler().delete_session_completely(
        session_id=session_id,
        user_id=user_id,
        organization_id=organization_id,
        delete_documents=delete_documents,
        delete_collections=delete_collections
    )
    
    if resp.Status == "Success":
        response.status_code = status.HTTP_200_OK
    else:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    return resp


@router.post("/sessions/{session_id}/delete-history", response_description="Delete history of session id")
async def delete_chat_history(
    request: Request,
    response: Response,
    session_id: str,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Delete the chat history for a session
    
    Args:
        request: Request object with user authentication info
        session_id: The ID of the chat session
        
    Returns:
        JSON response indicating success or failure
    """
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    resp = ChatMessageHistory().delete_message_history(
        session_id=session_id,
        user_id=user_id,
        organization_id=organization_id
    )
    
    if resp.status == "Success":
        response.status_code = status.HTTP_200_OK
    else:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return resp


@router.post("/sessions/{session_id}/get-chat-history", response_description="Chat history of session id")
async def get_chat_history(
    request: Request,
    response: Response,
    session_id: str,
    limit: int = 10,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Get the chat history for a session
    
    Args:
        request: Request object with user authentication info
        session_id: The ID of the chat session
        limit: Maximum number of messages to retrieve (default: 10)
        
    Returns:
        JSON response with the chat history
    """
    # Get user_id and organization_id information from request state
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    # Call the get_list_message_history method with the appropriate parameters
    resp = ChatMessageHistory().get_list_message_history(
        session_id=session_id,
        limit=limit,
        user_id=user_id,
        organization_id=organization_id
    )
    
    if resp.status == "Success":
        response.status_code = status.HTTP_200_OK
    else:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return resp