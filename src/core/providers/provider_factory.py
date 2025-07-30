from enum import Enum
from typing import Optional

from src.core.providers.base_provider import ModelProvider
from src.core.providers.openai_provider import OpenAIModelProvider
from src.core.providers.gemini_provider import GeminiModelProvider
from src.core.providers.ollama_provider import OllamaModelProvider
from src.core.utils.config import settings
from src.core.utils.logger.custom_logging import LoggerMixin

class ProviderType(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    GEMINI = "gemini"
    
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

class ModelProviderFactory(LoggerMixin):
    """Factory for creating model providers"""
    
    @staticmethod
    def create_provider(provider_type: str, model_name: str, api_key: Optional[str] = None) -> ModelProvider:
        """Create appropriate provider based on type"""
        
        if provider_type == ProviderType.OLLAMA:
            return OllamaModelProvider(model_name=model_name)
        elif provider_type == ProviderType.OPENAI:
            if not api_key:
                raise ValueError("API key is required for OpenAI provider")
            return OpenAIModelProvider(api_key=api_key, model_name=model_name)
        elif provider_type == ProviderType.GEMINI:
            if not api_key:
                raise ValueError("API key is required for Gemini provider")
            return GeminiModelProvider(api_key=api_key, model_name=model_name)
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
        
    @staticmethod
    def _get_api_key(provider_type: str) -> Optional[str]:
        """
        Get API key for a provider from environment variables.
        
        Args:
            provider_type: Provider type (ollama, openai, gemini)
            
        Returns:
            Optional[str]: API key for the provider
        """
        if provider_type == ProviderType.OPENAI:
            return settings.OPENAI_API_KEY
        elif provider_type == ProviderType.GEMINI:
            return settings.GEMINI_API_KEY
        elif provider_type == ProviderType.OLLAMA:
            return settings.OLLAMA_ENDPOINT
        
        return None