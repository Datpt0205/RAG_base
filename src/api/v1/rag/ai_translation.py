from src.core.schemas.response import BasicResponse
from fastapi import APIRouter, Response, status, Depends, Request, Body
from typing import Dict, Any, List, Optional
from src.features.rag.handlers.translation_handler import TranslationHandler
from src.features.rag.handlers.api_key_auth_handler import APIKeyAuth
from pydantic import BaseModel, Field

api_key_auth = APIKeyAuth()
router = APIRouter()

class SimpleTranslationRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    source_lang: str = Field(default="auto", description="Source language (default: auto detect)")
    target_lang: str = Field(..., description="Target language")
    model: Optional[str] = Field(default="aya:latest", description="Translation model (if not provided, default model will be used)")
    enable_thinking: bool = Field(default=True, description="Enable thinking mode to improve translation quality and prevent Chinese characters")
    
class SessionTranslationRequest(SimpleTranslationRequest):
    session_id: str = Field(
        ..., 
        description="Chat session ID to retrieve conversation context from"
    )
    max_history_messages: Optional[int] = Field(
        default=5, 
        description="Maximum number of history messages to consider"
    )

class ProviderTranslationRequest(SimpleTranslationRequest):
    provider_type: str = Field("ollama", description="Provider type: ollama, openai, gemini")
    
class ProviderSessionTranslationRequest(SessionTranslationRequest):
    provider_type: str = Field("ollama", description="Provider type: ollama, openai, gemini")

# @router.post("/translate", response_description="Translate text with AI")
# async def translate_text(
#     request: Request,
#     response: Response,
#     translation_request: SimpleTranslationRequest  = Body(...)
# ):

#     try:
#         translation_handler = TranslationHandler()
#         print(f"Translation request: {translation_request.model}")
#         translated_text = await translation_handler.translate_text(
#             text=translation_request.text,
#             source_lang=translation_request.source_lang,
#             target_lang=translation_request.target_lang,
#             model=translation_request.model,
#             enable_thinking=translation_request.enable_thinking 
#         )

#         response.status_code = status.HTTP_200_OK
#         return BasicResponse(
#             status="Success",
#             message="Translation complete",
#             data={"translated_text": translated_text}
#         )
#     except Exception as e:
#         response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
#         return BasicResponse(
#             status="Failed",
#             message=f"Error when translating text: {str(e)}",
#             data=None
#         )
    

# @router.post("/translate-text-with-context", response_description="Translate text using chat session history")
# async def translate_text_with_context(
#     request: Request,
#     response: Response,
#     translation_request: SessionTranslationRequest = Body(...)
# ):
#     """
#     Translate text using chat history from an existing chat session
#     """
#     try:
#         translation_handler = TranslationHandler()

#         translated_text = await translation_handler.translate_text_with_session(
#             text=translation_request.text,
#             session_id=translation_request.session_id,
#             source_lang=translation_request.source_lang,
#             target_lang=translation_request.target_lang,
#             max_history_messages=translation_request.max_history_messages,
#             model=translation_request.model
#         )

#         response.status_code = status.HTTP_200_OK
#         return BasicResponse(
#             status="Success",
#             message="Session-based translation complete",
#             data={"translated_text": translated_text}
#         )
#     except Exception as e:
#         response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
#         return BasicResponse(
#             status="Failed",
#             message=f"Error when translating text with session history: {str(e)}",
#             data=None
#         )


@router.post("/translate/provider", response_description="Translate text with specific provider")
async def translate_text_with_provider(
    request: Request,
    response: Response,
    translation_request: ProviderTranslationRequest = Body(...)
):
    try:
        translation_handler = TranslationHandler()
        
        # Extract provider settings
        provider_type = translation_request.provider_type
        
        source_language_name = None
        if not translation_request.source_lang or translation_request.source_lang.lower() == "auto":
            # Sử dụng phương thức mới detect_language với lingua
            detected_lang = translation_handler.detect_language(translation_request.text)
            
            # Log kết quả phát hiện
            translation_handler.logger.info(f"Detected language code: {detected_lang}")
            
            # Language mapping
            language_map = {
                "en": "ENGLISH",
                "vi": "VIETNAMESE",
                "fr": "FRENCH",
                "es": "SPANISH",
                "de": "GERMAN",
                "ja": "JAPANESE",
                "ko": "KOREAN",
                "zh": "CHINESE",
                "zh-cn": "CHINESE",
                "ru": "RUSSIAN"
            }
            
            # Chuyển mã ISO sang tên đầy đủ, hoặc giữ "automatic language detection" nếu không phát hiện được
            source_language_name = language_map.get(detected_lang, "automatic language detection").upper()
            translation_handler.logger.info(f"Source language name: {source_language_name}")
        else:
            source_language_name = translation_request.source_lang
        
        print(f"Automatic language detection {source_language_name}")
        
        translated_text = await translation_handler.translate_text(
            text=translation_request.text,
            source_lang=source_language_name,
            target_lang=translation_request.target_lang,
            model=translation_request.model,
            enable_thinking=translation_request.enable_thinking,
            provider_type=provider_type
        )

        response.status_code = status.HTTP_200_OK
        return BasicResponse(
            status="Success",
            message="Translation complete",
            data={"translated_text": translated_text}
        )
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return BasicResponse(
            status="Failed",
            message=f"Error when translating text: {str(e)}",
            data=None
        )


@router.post("/translate-with-context/provider", response_description="Translate text using chat session history with specific provider")
async def translate_text_with_session_and_provider(
    request: Request,
    response: Response,
    translation_request: ProviderSessionTranslationRequest = Body(...)
):
    """
    Translate text using chat history from an existing chat session with specific provider
    """
    try:
        translation_handler = TranslationHandler()
        
        # Extract provider settings
        provider_type = translation_request.provider_type

        translated_text = await translation_handler.translate_text_with_session(
            text=translation_request.text,
            session_id=translation_request.session_id,
            source_lang=translation_request.source_lang,
            target_lang=translation_request.target_lang,
            max_history_messages=translation_request.max_history_messages,
            model=translation_request.model,
            enable_thinking=translation_request.enable_thinking,
            provider_type=provider_type
        )

        response.status_code = status.HTTP_200_OK
        return BasicResponse(
            status="Success",
            message="Provider-based session translation complete",
            data={"translated_text": translated_text}
        )
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return BasicResponse(
            status="Failed",
            message=f"Error when translating text with session history and provider: {str(e)}",
            data=None
        )