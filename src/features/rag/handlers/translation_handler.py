from openai import AsyncOpenAI
from typing import List, Dict, Tuple, Optional
from src.core.utils.logger.custom_logging import LoggerMixin
from src.core.utils.config import settings
from src.features.rag.helpers.model_manager import model_manager
from src.features.rag.handlers.llm_chat_handler import chat_service
from src.core.providers.provider_factory import ModelProviderFactory, ProviderType
from src.core.providers.base_provider import ModelProvider
from lingua import Language, LanguageDetectorBuilder

# # COUNT TOKENS
# from src.features.rag.helpers.llm_helper import count_tokens

class TranslationHandler(LoggerMixin):
    DEFAULT_TRANSLATION_MODEL = "aya:latest"
    
    def __init__(self):
        super().__init__()
        self._provider_instances = {}  # Cache for provider instances

        self._language_detector = self._init_language_detector()
    
    def _init_language_detector(self):
        """Khởi tạo language detector với các ngôn ngữ phổ biến"""
        try:
            languages = [
                Language.ENGLISH,
                Language.VIETNAMESE,
                Language.FRENCH,
                Language.SPANISH,
                Language.GERMAN,
                Language.JAPANESE,
                Language.KOREAN,
                Language.CHINESE,
                Language.RUSSIAN
            ]
            return LanguageDetectorBuilder.from_languages(*languages).build()
        except Exception as e:
            self.logger.error(f"Error initializing language detector: {str(e)}")
            return None
    
    def detect_language(self, text: str) -> Optional[str]:
        """Phát hiện ngôn ngữ của văn bản sử dụng lingua language detector"""
        if not text or text.strip() == "":
            return None
            
        try:
            if not self._language_detector:
                self.logger.warning("Language detector not initialized, falling back to default 'en'")
                return "en"
                
            # Phát hiện ngôn ngữ
            detected_lang = self._language_detector.detect_language_of(text)
            if detected_lang:
                # Lấy mã ISO code
                iso_code = detected_lang.iso_code_639_1.name.lower()
                
                # Mapping một số trường hợp đặc biệt
                mapping = {
                    "zh": "zh",  # Chinese
                    "cmn": "zh-cn"  # Mandarin Chinese
                }
                
                self.logger.info(f"Detected language: {iso_code}")
                return mapping.get(iso_code, iso_code)
            
            self.logger.warning(f"Could not detect language, falling back to default 'en'")
            return "en"  # Default to English if detection fails
        except Exception as e:
            self.logger.error(f"Error detecting language: {str(e)}")
            return "en"  # Default to English if there's an error

    async def get_provider(self,
                            model_name: str, 
                            provider_type: str = ProviderType.OLLAMA, 
                            api_key: Optional[str] = None) -> ModelProvider:
            """
            Get a model provider instance for translation.
            
            Args:
                model_name: Name of the model to use
                provider_type: Type of provider (ollama, openai, gemini)
                api_key: API key for paid providers
                
            Returns:
                ModelProvider: Provider instance
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
                self.logger.error(f"Error getting provider for translation: {str(e)}")
                raise

#     async def translate_text(self,
#         text: str,
#         source_lang: str = "auto",
#         target_lang: str = "English",
#         model: str = None,
#         enable_thinking: bool = True,
#     ) -> str:
#         # # COUNT TOKENS
#         # output_tokens = ""
#         try:
#             model_to_use = model if model else self.DEFAULT_TRANSLATION_MODEL
            
#             if model_to_use not in model_manager.loaded_models:
#                 self.logger.info(f"Model {model_to_use} not loaded yet, loading...")
#                 await model_manager.load_model(model_to_use)

#             self.logger.info(f"Translate text with model {model_to_use}, translation from {source_lang} to {target_lang}")
            
#             client = AsyncOpenAI(
#                 api_key="EMPTY",
#                 base_url=f"{settings.OLLAMA_ENDPOINT}/v1"
#             )
            
#             # # COUNT TOKENS
#             # system_message = f"You are a professional translator. Translate the text from {source_lang} to {target_lang} accurately, maintaining the original meaning and tone."
#             # full_prompt = f"{system_message}\n\n{text}"
#             # input_tokens_with_context = count_tokens(full_prompt)
#             # self.logger.info(f"Translation input tokens (with system message): {input_tokens_with_context}")
            
# #             system_message = f"""
# # You are a translation engine.

# # Instruction:
# # - Translate ONLY the following input text from {source_lang} to {target_lang}.
# # - Keep the original meaning and tone.
# # - Do NOT add explanation.
# # - Output MUST be only the translated sentence.
# # - NOTE: using no_think mode
# # """
#             self.logger.info(f"Thinking: {enable_thinking}")
#             system_message = f"""
# You are a precise translation engine with expertise in cultural language nuances.

# RULES:
# - Translate the input text from {source_lang} to {target_lang}
# - IMPORTANT: Ensure ALL words are in {target_lang} only
# - DO NOT include ANY Chinese characters or words
# - Keep the original meaning and tone
# - Output MUST contain only {target_lang} words
# - Return ONLY the translated text without explanations

# SPECIAL ATTENTION FOR IDIOMS:
# - PAY SPECIAL ATTENTION to idioms, expressions, and colloquialisms in the source text
# - When you encounter idioms or cultural expressions, find the equivalent idiom in {target_lang} that conveys the same meaning
# - If no exact idiom equivalent exists, translate the intended meaning while preserving the figurative tone
# - NEVER translate idioms literally word-for-word - interpret their contextual meaning

# NOTES: Using {'NO-THINKING mode - DO NOT think.' if not enable_thinking else 'THINKING mode - First THINK carefully about how to translate accurately before providing your answer. When you detect idioms or expressions, think extra carefully about finding equivalent expressions in the target language.'}"""
#             messages = [
#                 {
#                     "role": "system", 
#                     "content": system_message #f"You are a professional translator. Translate the text from {source_lang} to {target_lang} accurately, maintaining the original meaning and tone."
#                 },
#                 {
#                     "role": "user",
#                     "content": f"Input text:\n{text}" # text
#                 }
#             ]
            
#             print(f"Messages translate: {messages}")
#             completion = await client.chat.completions.create(
#                 model=model_to_use,
#                 messages=messages,
#                 extra_body={
#                     "translation_options": {
#                         "source_lang": source_lang,
#                         "target_lang": target_lang
#                     },
#                     "chat_template_kwargs": {
#                         "enable_thinking": enable_thinking
#                     }
#                 }
#             )
            
#             translated_text = completion.choices[0].message.content
#             self.logger.info(f"Translation completed successfully")

#             # # COUNT TOKENS
#             # output_tokens = count_tokens(translated_text)
#             # self.logger.info(f"Translation output tokens: {output_tokens}")
#             # total_tokens = input_tokens_with_context + output_tokens
#             # self.logger.info(f"Translation total tokens: {total_tokens} (input: {input_tokens_with_context}, output: {output_tokens})")
            
# #             has_chinese = any('\u4e00' <= char <= '\u9fff' for char in translated_text)
            
# #             if has_chinese:
# #                 self.logger.warning("Detected Chinese characters in translation result, retrying...")
                
# #                 retry_system_message = f"""
# # TRANSLATION CORRECTION TASK:
# # - The following text contains some Chinese characters
# # - Replace ALL Chinese characters with appropriate {target_lang} words
# # - Return ONLY the corrected text
# # - ENSURE NO Chinese characters remain in your output

# # NOTE: using no_think mode
# # """
# #                 retry_messages = [
# #                     {
# #                         "role": "system", 
# #                         "content": retry_system_message
# #                     },
# #                     {
# #                         "role": "user",
# #                         "content": f"Text with Chinese characters to fix:\n{translated_text}"
# #                     }
# #                 ]
                
# #                 retry_completion = await client.chat.completions.create(
# #                     model=model_to_use,
# #                     messages=retry_messages
# #                 )
                
# #                 translated_text = retry_completion.choices[0].message.content      

#             print(f"Translated text: {translated_text}")
#             return translated_text
#         except Exception as e:
#             self.logger.error(f"Error when translating text: {str(e)}")
#             raise


#     async def translate_text_with_session(self,
#         text: str,
#         session_id: str,
#         source_lang: str = "auto",
#         target_lang: str = "English",
#         max_history_messages: int = 5,
#         model: str = None,
#         enable_thinking: bool = False,
#     ) -> str:
#         try:
#             model_to_use = model if model else self.DEFAULT_TRANSLATION_MODEL
            
#             if model_to_use not in model_manager.loaded_models:
#                 self.logger.info(f"Model {model_to_use} not loaded yet, loading...")
#                 await model_manager.load_model(model_to_use)
#             try:
#                 chat_history_tuples = chat_service.get_chat_history(
#                     session_id=session_id, 
#                     limit=max_history_messages
#                 )
#                 self.logger.info(f"Retrieved {len(chat_history_tuples)} messages from session {session_id}")
#             except Exception as e:
#                 self.logger.warning(f"Failed to retrieve chat history: {str(e)}. Using basic translation.")
#                 return await self.translate_text(text, source_lang, target_lang, model)
               
            
#             # hardcoded_chat_history = [
#             #     ("Hi, I need to see a dermatologist.", "user"),
#             #     ("Sure. Do you prefer an online consultation or visiting the clinic?", "assistant"),
#             #     ("Online would be great.", "user")
#             # ]
            
#             client = AsyncOpenAI(
#                 api_key="EMPTY",
#                 base_url=f"{settings.OLLAMA_ENDPOINT}/v1"
#             )
            
#             system_message = {
#                 "role": "system",
#                 "content": f"""You are a professional translator. Your task is to translate from {source_lang} to {target_lang} with a two-phase approach:

# Phase 1: Translate the input text using the conversation context for accurate meaning.
# Phase 2: Verify and refine your translation to ensure it sounds natural and matches the context.

# Return ONLY the final translated text without explanations or additional content.
# NOTE: using no_think mode
# """
#             }
            
#             user_message = {
#                 "role": "user",
#                 "content": f"""
# Conversation history:
# {chat_history_tuples}

# Original text: "{text}"

# Please translate this to {target_lang}, considering the context. First translate it, then refine your translation if needed.
# """
#             }
            
#             messages = [system_message, user_message]
#             self.logger.info(f"Messages translate: {messages}")
            
#             completion = await client.chat.completions.create(
#                 model=model_to_use,
#                 messages=messages,
#                 temperature=0.1,
#                 extra_body={
#                     "translation_options": {
#                         "source_lang": source_lang,
#                         "target_lang": target_lang
#                     },
#                     "chat_template_kwargs": {
#                         "enable_thinking": enable_thinking
#                     }
#                 }
#             )
            
#             translation = completion.choices[0].message.content.strip().strip('"\'')
#             return translation
            
#         except Exception as e:
#             self.logger.error(f"Error in combined translation: {str(e)}")
#             return await self.translate_text(text, source_lang, target_lang, model)

    def _get_api_key(self, provider_type: str) -> Optional[str]:
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

    async def translate_text(self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "English",
        model: str = None,
        enable_thinking: bool = True,
        provider_type: str = ProviderType.OLLAMA
    ) -> str:
        try:
            api_key = self._get_api_key(provider_type)

            model_to_use = model if model else self.DEFAULT_TRANSLATION_MODEL
            
            # Check if using local Ollama model
            if provider_type == ProviderType.OLLAMA:
                if model_to_use not in model_manager.loaded_models:
                    self.logger.info(f"Model {model_to_use} not loaded yet, loading...")
                    await model_manager.load_model(model_to_use)
            
            self.logger.info(f"Translate text with provider {provider_type}, model {model_to_use}, translation from {source_lang} to {target_lang}")
            
            # Create system message for translation
            system_message = f"""
You are a precise translation engine with expertise in cultural language nuances.

RULES:
- Translate the input text from {source_lang} to {target_lang}
- IMPORTANT: Ensure ALL words are in {target_lang} only
- DO NOT include ANY Chinese characters or words
- Keep the original meaning and tone
- Output MUST contain only {target_lang} words
- Return ONLY the translated text without explanations

SPECIAL ATTENTION FOR IDIOMS:
- PAY SPECIAL ATTENTION to idioms, expressions, and colloquialisms in the source text
- When you encounter idioms or cultural expressions, find the equivalent idiom in {target_lang} that conveys the same meaning
- If no exact idiom equivalent exists, translate the intended meaning while preserving the figurative tone
- NEVER translate idioms literally word-for-word - interpret their contextual meaning

NOTES: Using {'NO-THINKING mode - DO NOT think.' if not enable_thinking else 'THINKING mode - First THINK carefully about how to translate accurately before providing your answer. When you detect idioms or expressions, think extra carefully about finding equivalent expressions in the target language.'}"""

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Input text:\n{text}"}
            ]
            
            if provider_type == ProviderType.OLLAMA:
                # Use direct Ollama client for legacy compatibility
                client = AsyncOpenAI(
                    api_key="EMPTY",
                    base_url=f"{settings.OLLAMA_ENDPOINT}/v1"
                )
                
                # Ollama supports translation_options
                completion = await client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    extra_body={
                        "translation_options": {
                            "source_lang": source_lang,
                            "target_lang": target_lang
                        },
                        "chat_template_kwargs": {
                            "enable_thinking": enable_thinking
                        }
                    }
                )
                
                translated_text = completion.choices[0].message.content
            else:
                # Use provider model
                provider = await self.get_provider(
                    model_name=model_to_use,
                    provider_type=provider_type,
                    api_key=api_key
                )
                
                # For other providers, just pass the messages without translation_options
                response = await provider.generate(messages)
                translated_text = response.get("content", "")
            
            self.logger.info(f"Translation completed successfully")
            return translated_text
            
        except Exception as e:
            self.logger.error(f"Error when translating text: {str(e)}")
            raise

    async def translate_text_with_session(self,
        text: str,
        session_id: str,
        source_lang: str = "auto",
        target_lang: str = "English",
        max_history_messages: int = 5,
        model: str = None,
        enable_thinking: bool = False,
        provider_type: str = ProviderType.OLLAMA
    ) -> str:
        try:
            api_key = self._get_api_key(provider_type)
            
            model_to_use = model if model else self.DEFAULT_TRANSLATION_MODEL
            
            # Check if using local Ollama model
            if provider_type == ProviderType.OLLAMA:
                if model_to_use not in model_manager.loaded_models:
                    self.logger.info(f"Model {model_to_use} not loaded yet, loading...")
                    await model_manager.load_model(model_to_use)
            
            try:
                chat_history_tuples = chat_service.get_chat_history(
                    session_id=session_id, 
                    limit=max_history_messages
                )
                self.logger.info(f"Retrieved {len(chat_history_tuples)} messages from session {session_id}")
            except Exception as e:
                self.logger.warning(f"Failed to retrieve chat history: {str(e)}. Using basic translation.")
                return await self.translate_text(
                    text, 
                    source_lang, 
                    target_lang, 
                    model, 
                    enable_thinking,
                    provider_type, 
                    api_key
                )
            
            system_message = {
                "role": "system",
                "content": f"""You are a professional translator. Your task is to translate from {source_lang} to {target_lang} with a two-phase approach:

Phase 1: Translate the input text using the conversation context for accurate meaning.
Phase 2: Verify and refine your translation to ensure it sounds natural and matches the context.

Return ONLY the final translated text without explanations or additional content.
NOTE: using {'no_think mode' if not enable_thinking else 'thinking mode'}
"""
            }
            
            user_message = {
                "role": "user",
                "content": f"""
Conversation history:
{chat_history_tuples}

Original text: "{text}"

Please translate this to {target_lang}, considering the context. First translate it, then refine your translation if needed.
"""
            }
            
            messages = [system_message, user_message]
            
            if provider_type == ProviderType.OLLAMA:
                # Use direct Ollama client for legacy compatibility
                client = AsyncOpenAI(
                    api_key="EMPTY",
                    base_url=f"{settings.OLLAMA_ENDPOINT}/v1"
                )
                
                # Ollama supports translation_options
                completion = await client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    temperature=0.1,
                    extra_body={
                        "translation_options": {
                            "source_lang": source_lang,
                            "target_lang": target_lang
                        },
                        "chat_template_kwargs": {
                            "enable_thinking": enable_thinking
                        }
                    }
                )
                
                translation = completion.choices[0].message.content.strip().strip('"\'')
            else:
                # Use provider model
                provider = await self.get_provider(
                    model_name=model_to_use,
                    provider_type=provider_type,
                    api_key=api_key
                )
                
                # For other providers, just pass the messages with temperature
                response = await provider.generate(messages, temperature=0.1)
                translation = response.get("content", "").strip().strip('"\'')
            
            return translation
            
        except Exception as e:
            self.logger.error(f"Error in combined translation: {str(e)}")
            return await self.translate_text(
                text, 
                source_lang, 
                target_lang, 
                model,
                enable_thinking,
                provider_type, 
                api_key
            )