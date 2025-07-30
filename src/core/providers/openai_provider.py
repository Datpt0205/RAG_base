from openai import AsyncOpenAI
from typing import Dict, Any, List, Optional, AsyncGenerator

from src.core.providers.base_provider import ModelProvider
from src.core.utils.logger.custom_logging import LoggerMixin

class OpenAIModelProvider(ModelProvider, LoggerMixin):
    """Provider for OpenAI models using official SDK"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4.1-nano-2025-04-14"):
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
        
    async def initialize(self) -> None:
        """Initialize the OpenAI client"""
        try:
            self.client = AsyncOpenAI(api_key=self.api_key)
            self.logger.info(f"Initialized OpenAI provider with model {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI provider: {str(e)}")
            raise
            
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate text completion"""
        if not self.client:
            await self.initialize()
            
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            return self._format_response(response)
        except Exception as e:
            self.logger.error(f"Error generating OpenAI completion: {str(e)}")
            raise
            
    async def stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Stream response chunks"""
        if not self.client:
            await self.initialize()
            
        try:
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self.logger.error(f"Error streaming OpenAI completion: {str(e)}")
            raise
    
    def supports_feature(self, feature_name: str) -> bool:
        """Check if provider supports a specific feature"""
        feature_support = {
            "thinking_mode": False,
            "vision": self.model_name in ["gpt-4-vision", "gpt-4-vision-preview"],
            "function_calling": True,
            "json_mode": True,
            "translation": True
        }
        return feature_support.get(feature_name, False)
        
    def _format_response(self, response) -> Dict[str, Any]:
        """Format OpenAI response to common structure"""
        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "id": response.id,
            "finish_reason": response.choices[0].finish_reason,
            "raw_response": response
        }