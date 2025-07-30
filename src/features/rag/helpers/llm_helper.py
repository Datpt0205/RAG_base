import re
from typing import AsyncGenerator
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage

from src.core.utils.config import settings
from src.core.utils.logger.custom_logging import LoggerMixin
from src.features.rag.helpers.model_manager import model_manager

# # COUNT TOKENS
# def count_tokens(text: str) -> int:
#     """Count the number of tokens in a text string."""
#     try:
#         import tiktoken
#         encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
#         tokens = encoding.encode(text)
#         return len(tokens)
#     except ImportError:
#         return len(text.split())

class LLMGenerator(LoggerMixin):
    def __init__(self):
        super().__init__()


    def clean_thinking(self, content: str) -> str:
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    async def get_llm(self, model: str, base_url: str = settings.OLLAMA_ENDPOINT):
        try:
            if model not in model_manager.loaded_models:
                self.logger.info(f"Model {model} is not loaded yet, loading...")
                await model_manager.load_model(model)
                
            llm = ChatOllama(base_url=base_url,
                            model=model,
                            temperature=0,
                            top_k=10,
                            top_p=0.5,
                            # num_ctx=8000, 
                            streaming=True)
     
        except Exception as e:
            self.logger.error(f"Error: {str(e)}")
        return llm

    async def get_streaming_chain(self, model: str, base_url: str = settings.OLLAMA_ENDPOINT):
        """
        Get a configured LLM instance optimized for streaming responses
        
        This method is specifically designed for streaming use cases where
        chunks of text are returned incrementally rather than waiting for
        the complete response.
        
        Args:
            model: The name of the LLM model to use
            base_url: The base URL of the Ollama API
            
        Returns:
            ChatOllama: Configured LLM instance with streaming enabled
        """
        try:
            if model not in model_manager.loaded_models:
                self.logger.info(f"Model {model} is not loaded yet, loading...")
                await model_manager.load_model(model)
                
            # For streaming, we use the same configuration as regular LLM but ensure streaming is explicitly enabled
            llm = ChatOllama(base_url=base_url,
                            model=model,
                            temperature=0,
                            top_k=10,
                            top_p=0.5,
                            streaming=True)
            
            return llm
        except Exception as e:
            self.logger.error(f"Error configuring streaming LLM: {str(e)}")
            raise
    
    async def stream_response(self, 
                              llm,
                              messages,
                              clean_thinking: bool = True) -> AsyncGenerator[str, None]:
        """
        Stream response chunks from the LLM
        
        Args:
            llm: The LLM instance to use
            messages: The messages to send to the LLM
            clean_thinking: Whether to clean thinking sections from chunks
            
        Yields:
            str: Chunks of the response
        """
        async for chunk in llm.astream(messages):
            if isinstance(chunk, AIMessage) and chunk.content:
                content = chunk.content
                if clean_thinking:
                    content = self.clean_thinking(content)
                if content:
                    yield content
            elif hasattr(chunk, 'content') and chunk.content:
                content = chunk.content
                if clean_thinking:
                    content = self.clean_thinking(content)
                if content:
                    yield content
    

import re
from typing import AsyncGenerator, Optional, Dict, Any, List
from src.core.utils.logger.custom_logging import LoggerMixin
from src.core.providers.provider_factory import ModelProviderFactory, ProviderType
from src.features.menu_extraction.prompts.prompt_manager import prompt_manager


class LLMGeneratorProvider(LoggerMixin):
    def __init__(self):
        super().__init__()
        self._provider_instances = {}  # Cache for provider instances

    def clean_thinking(self, content: str) -> str:
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    async def get_llm(self, 
                      model_name: str, 
                      provider_type: str = ProviderType.OLLAMA, 
                      api_key: Optional[str] = None) -> Any:
        """
        Get a model provider instance
        
        Args:
            model_name: Name of the model to use
            provider_type: Type of provider (ollama, openai, gemini)
            api_key: API key for paid providers
            
        Returns:
            Any: Provider instance
        """
        try:
            # Create unique key for caching
            cache_key = f"{provider_type}:{model_name}:{api_key}"
            
            if cache_key not in self._provider_instances:
                # Create provider
                provider = ModelProviderFactory.create_provider(
                    provider_type=provider_type,
                    model_name=model_name,
                    api_key=api_key
                )
                
                # Initialize model
                await provider.initialize()
                
                # Cache provider
                self._provider_instances[cache_key] = provider
            
            return self._provider_instances[cache_key]
            
        except Exception as e:
            self.logger.error(f"Error getting LLM: {str(e)}")
            raise

    async def generate_response(self, 
                                model_name: str,
                                messages: List[Dict[str, str]],
                                provider_type: str = ProviderType.OLLAMA,
                                api_key: Optional[str] = None,
                                enable_thinking: bool = True,
                                **kwargs) -> Dict[str, Any]:
        """
        Generate a response from the model
        
        Args:
            model_name: Name of the model to use
            messages: Messages to send to the model
            provider_type: Type of provider (ollama, openai, gemini)
            api_key: API key for paid providers
            enable_thinking: Whether to enable thinking mode (only works with models that support it)
            **kwargs: Additional parameters for the model
            
        Returns:
            Dict[str, Any]: Response from the model
        """
        provider = await self.get_llm(model_name, provider_type, api_key)
        
        # Add chat template parameters for Qwen models
        if "qwen" in model_name.lower() and provider_type == ProviderType.OLLAMA:
            # Pass enable_thinking to Qwen model via chat_template_kwargs
            kwargs["chat_template_kwargs"] = kwargs.get("chat_template_kwargs", {})
            kwargs["chat_template_kwargs"]["enable_thinking"] = enable_thinking
        
        return await provider.generate(messages, **kwargs)
        

    async def generate_with_template(self,
                                    model_name: str,
                                    system_template: str,
                                    user_content: str,
                                    history_messages: Optional[List[Dict[str, str]]] = None,
                                    provider_type: str = ProviderType.OLLAMA,
                                    api_key: Optional[str] = None,
                                    enable_thinking: bool = True,
                                    template_vars: Optional[Dict[str, Any]] = None,
                                    **kwargs) -> Dict[str, Any]:
        """
        Generate a response using a template.
        
        Args:
            model_name: Name of the model to use
            system_template: Name of the system template
            user_content: User message content
            history_messages: Previous messages in the conversation
            provider_type: Type of provider (ollama, openai, gemini)
            api_key: API key for paid providers
            template_vars: Variables for template formatting
            **kwargs: Additional parameters for the model
            
        Returns:
            Dict[str, Any]: Response from the model
        """
        template_vars = template_vars or {}
        
        # Format messages using prompt manager
        messages = prompt_manager.format_messages(
            system_template=system_template,
            user_content=user_content,
            history_messages=history_messages,
            **template_vars
        )
        
        # Generate response
        return await self.generate_response(
            model_name=model_name,
            messages=messages,
            provider_type=provider_type,
            api_key=api_key,
            enable_thinking=enable_thinking,
            **kwargs
        )
    
    async def generate_rag_response(self,
                                model_name: str,
                                query: str,
                                context: str,
                                history_messages: Optional[List[Dict[str, str]]] = None,
                                provider_type: str = ProviderType.OLLAMA,
                                api_key: Optional[str] = None,
                                enable_thinking: bool = True,
                                **kwargs) -> Dict[str, Any]:
        """
        Generate a RAG response.
        
        Args:
            model_name: Name of the model to use
            query: User query
            context: Retrieved context
            history_messages: Previous messages in the conversation
            provider_type: Type of provider (ollama, openai, gemini)
            api_key: API key for paid providers
            detailed: Whether to use detailed RAG template
            enable_thinking: Whether to enable thinking mode
            **kwargs: Additional parameters for the model
            
        Returns:
            Dict[str, Any]: Response from the model
        """
        # Format messages specifically for RAG
        messages = prompt_manager.format_rag_messages(
            query=query,
            context=context,
            history_messages=history_messages,
        )
        
        # Generate response
        return await self.generate_response(
            model_name=model_name,
            messages=messages,
            provider_type=provider_type,
            api_key=api_key,
            enable_thinking=enable_thinking,
            **kwargs
        )
    
    async def stream_response(self, 
                            model_name: str,
                            messages: List[Dict[str, str]],
                            provider_type: str = ProviderType.OLLAMA,
                            api_key: Optional[str] = None,
                            clean_thinking: bool = True,
                            enable_thinking: bool = True,
                            **kwargs) -> AsyncGenerator[str, None]:
        """
        Stream response chunks from the model
        
        Args:
            model_name: Name of the model to use
            messages: Messages to send to the model
            provider_type: Type of provider (ollama, openai, gemini)
            api_key: API key for paid providers
            clean_thinking: Whether to clean thinking sections from chunks
            enable_thinking: Whether to enable thinking mode
            **kwargs: Additional parameters for the model
            
        Yields:
            str: Chunks of the response
        """
        provider = await self.get_llm(model_name, provider_type, api_key)
        
        # Add chat template parameters for Qwen models
        if "qwen" in model_name.lower() and provider_type == ProviderType.OLLAMA:
            # Pass enable_thinking to Qwen model via chat_template_kwargs
            kwargs["chat_template_kwargs"] = kwargs.get("chat_template_kwargs", {})
            kwargs["chat_template_kwargs"]["enable_thinking"] = enable_thinking
        
        # If thinking is not enabled, we should clean thinking tags from outputs
        # If thinking is enabled, only clean if explicitly requested
        should_clean_thinking = not enable_thinking or clean_thinking
        
        async for chunk in provider.stream(messages, **kwargs):
            if should_clean_thinking:
                chunk = self.clean_thinking(chunk)
            if chunk:
                yield chunk

    
    async def stream_with_template(self,
                                  model_name: str,
                                  system_template: str,
                                  user_content: str,
                                  history_messages: Optional[List[Dict[str, str]]] = None,
                                  provider_type: str = ProviderType.OLLAMA,
                                  api_key: Optional[str] = None,
                                  template_vars: Optional[Dict[str, Any]] = None,
                                  enable_thinking: bool = True,
                                  clean_thinking: bool = True,
                                  **kwargs) -> AsyncGenerator[str, None]:
        """
        Stream a response using a template.
        
        Args:
            model_name: Name of the model to use
            system_template: Name of the system template
            user_content: User message content
            history_messages: Previous messages in the conversation
            provider_type: Type of provider (ollama, openai, gemini)
            api_key: API key for paid providers
            template_vars: Variables for template formatting
            clean_thinking: Whether to clean thinking sections from chunks
            **kwargs: Additional parameters for the model
            
        Yields:
            str: Chunks of the response
        """
        template_vars = template_vars or {}
        
        # Format messages using prompt manager
        messages = prompt_manager.format_messages(
            system_template=system_template,
            user_content=user_content,
            history_messages=history_messages,
            **template_vars
        )
        
        # Stream response
        async for chunk in self.stream_response(
            model_name=model_name,
            messages=messages,
            provider_type=provider_type,
            api_key=api_key,
            clean_thinking=clean_thinking,
            enable_thinking=enable_thinking,
            **kwargs
        ):
            yield chunk

    
    async def generate_react_cot_response(self,
                                        model_name: str,
                                        query: str,
                                        context: str = "",
                                        history_messages: Optional[List[Dict[str, str]]] = None,
                                        provider_type: str = ProviderType.OLLAMA,
                                        api_key: Optional[str] = None,
                                        enable_thinking: bool = True,
                                        **kwargs) -> Dict[str, Any]:
        """
        Generate a response using ReAct+CoT (Reasoning and Acting with Chain of Thought).
        
        Args:
            model_name: Name of the model to use
            query: User query
            context: Retrieved context (optional)
            history_messages: Previous messages in the conversation
            provider_type: Type of provider (ollama, openai, gemini)
            api_key: API key for paid providers
            **kwargs: Additional parameters for the model
            
        Returns:
            Dict[str, Any]: Response from the model
        """
        # Format messages specifically for ReAct+CoT
        messages = prompt_manager.format_react_cot_messages(
            query=query,
            context=context,
            history_messages=history_messages,
            enable_thinking=enable_thinking
        )
        
        # Use higher temperature for better reasoning diversity
        kwargs.setdefault("temperature", 0.2)
        
        # For OpenAI, make sure to set a high max_tokens to allow full reasoning
        if provider_type == ProviderType.OPENAI:
            kwargs.setdefault("max_tokens", 4000)
        
        # Generate response
        return await self.generate_response(
            model_name=model_name,
            messages=messages,
            provider_type=provider_type,
            api_key=api_key,
            enable_thinking=enable_thinking,
            **kwargs
        )

    async def stream_react_cot_response(self,
                                    model_name: str,
                                    query: str,
                                    context: str = "",
                                    history_messages: Optional[List[Dict[str, str]]] = None,
                                    provider_type: str = ProviderType.OLLAMA,
                                    api_key: Optional[str] = None,
                                    clean_thinking: bool = True,
                                    enable_thinking: bool = True,
                                    **kwargs) -> AsyncGenerator[str, None]:
        """
        Stream a response using ReAct+CoT.
        
        Args:
            model_name: Name of the model to use
            query: User query
            context: Retrieved context (optional)
            history_messages: Previous messages in the conversation
            provider_type: Type of provider (ollama, openai, gemini)
            api_key: API key for paid providers
            clean_thinking: Whether to clean thinking sections
            **kwargs: Additional parameters for the model
            
        Yields:
            str: Chunks of the response
        """
        # Format messages specifically for ReAct+CoT
        messages = prompt_manager.format_react_cot_messages(
            query=query,
            context=context,
            history_messages=history_messages,
            enable_thinking=enable_thinking
        )
        
        # Use higher temperature for better reasoning diversity
        kwargs.setdefault("temperature", 0.2)
        
        # For OpenAI, make sure to set a high max_tokens to allow full reasoning
        if provider_type == ProviderType.OPENAI:
            kwargs.setdefault("max_tokens", 4000)
        
        # Stream response
        async for chunk in self.stream_response(
            model_name=model_name,
            messages=messages,
            provider_type=provider_type,
            api_key=api_key,
            clean_thinking=clean_thinking,
            enable_thinking=enable_thinking,
            **kwargs
        ):
            yield chunk