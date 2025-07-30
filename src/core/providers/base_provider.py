from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator

class ModelProvider(ABC):
    """Base interface for model providers"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider"""
        pass
        
    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate text completion"""
        pass
        
    @abstractmethod
    async def stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Stream response chunks"""
        pass
    
    @abstractmethod
    def supports_feature(self, feature_name: str) -> bool:
        """
        Check if provider supports a specific feature.
        
        Args:
            feature_name: Name of the feature to check
                Supported features:
                - thinking_mode: Whether provider supports thinking mode
                - vision: Whether provider supports vision models
                - function_calling: Whether provider supports function calling
                - json_mode: Whether provider supports JSON mode
                - translation: Whether provider supports translation
                
        Returns:
            bool: Whether the feature is supported
        """
        pass