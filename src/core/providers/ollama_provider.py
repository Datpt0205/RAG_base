import aiohttp
import json
from typing import Dict, Any, List, Optional, AsyncGenerator

from src.core.providers.base_provider import ModelProvider
from src.core.utils.config import settings
from src.core.utils.logger.custom_logging import LoggerMixin
from src.features.rag.helpers.model_manager import model_manager

class OllamaModelProvider(ModelProvider, LoggerMixin):
    """Provider for Ollama models"""
    
    def __init__(self, model_name: str, base_url: str = settings.OLLAMA_ENDPOINT):
        super().__init__()
        self.model_name = model_name
        self.base_url = base_url
        
    async def initialize(self) -> None:
        """Initialize model by ensuring it's loaded"""
        try:
            if self.model_name not in model_manager.loaded_models:
                self.logger.info(f"Model {self.model_name} is not loaded yet, loading...")
                await model_manager.load_model(self.model_name)
                
            self.logger.info(f"Initialized Ollama provider with model {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama provider: {str(e)}")
            raise
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate text completion"""
        await self.initialize()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model_name,
                        "messages": messages,
                        "stream": False,
                        **self._convert_params(kwargs)
                    }
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API returned {response.status}: {error_text}")
                    
                    result = await response.json()
                    return self._format_response(result)
                    
        except Exception as e:
            self.logger.error(f"Error generating Ollama completion: {str(e)}")
            raise
    
    async def ainvoke(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Invoke the model asynchronously - compatibility layer for LangChain interface."""
        result = await self.generate(messages, **kwargs)
        return result

    async def stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Stream response chunks"""
        await self.initialize()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model_name,
                        "messages": messages,
                        "stream": True,
                        **self._convert_params(kwargs)
                    },
                    headers={"Accept": "application/x-ndjson"}
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API returned {response.status}: {error_text}")
                    
                    # Stream JSON lines
                    async for line in response.content:
                        if line:
                            try:
                                chunk = json.loads(line)
                                if "message" in chunk and "content" in chunk["message"]:
                                    yield chunk["message"]["content"]
                            except json.JSONDecodeError:
                                self.logger.warning(f"Failed to parse chunk: {line}")
                    
        except Exception as e:
            self.logger.error(f"Error streaming Ollama completion: {str(e)}")
            raise
    
    def supports_feature(self, feature_name: str) -> bool:
        """Check if provider supports a specific feature"""
        feature_support = {
            "thinking_mode": True,
            "vision": False,
            "function_calling": False,
            "json_mode": False,
            "translation": True
        }
        return feature_support.get(feature_name, False)
    
    def _convert_params(self, openai_params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI parameters to Ollama parameters"""
        ollama_params = {}
        
        # Map parameters
        if "temperature" in openai_params:
            ollama_params["temperature"] = openai_params["temperature"]
        if "top_p" in openai_params:
            ollama_params["top_p"] = openai_params["top_p"]
        if "max_tokens" in openai_params:
            ollama_params["num_predict"] = openai_params["max_tokens"]
        
        return ollama_params
        
    def _format_response(self, response) -> Dict[str, Any]:
        """Format Ollama response to common structure"""
        return {
            "content": response.get("message", {}).get("content", ""),
            "model": response.get("model", self.model_name),
            "id": None,  # Ollama doesn't provide response IDs
            "finish_reason": "stop",  # Ollama doesn't have finish_reason
            "raw_response": response
        }