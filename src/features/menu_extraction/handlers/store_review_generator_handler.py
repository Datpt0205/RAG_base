from typing import Dict, Any, Optional

from src.core.utils.config import settings
from src.core.utils.constants import OpenAIModelName
from src.core.utils.logger.custom_logging import LoggerMixin
from src.features.rag.helpers.llm_helper import LLMGeneratorProvider
from src.core.schemas.review_generator import StoreReviewRequestProvider
from src.features.menu_extraction.prompts.prompt_manager import prompt_manager
from src.core.providers.provider_factory import ProviderType


class StoreReviewGeneratorHandler(LoggerMixin):
    """
    Handler for generating store reviews based on multiple product experiences.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.llm_generator_provider = LLMGeneratorProvider()
        

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
           

    async def generate_store_review_provider(self, 
                                             request: StoreReviewRequestProvider,
                                             model_name: str = OpenAIModelName.GPT41Nano,
                                             provider_type: str = ProviderType.OPENAI
                                             ) -> Dict[str, Any]:
        """
        Generate a comprehensive store review based on multiple product experiences.
        
        Args:
            request: The request containing store information and products
            model_name: The name of the LLM model to use
            provider_type: The provider type to use
            api_key: API key for paid providers
            
        Returns:
            Dict[str, Any]: Generated review and rating
        """
        try:
            # Extract provider settings from request if available
            api_key = self._get_api_key(provider_type)
            
            # Determine overall rating if not specified
            rating = request.rating
            if rating is None:
                # Generate a rating based on store average and product ratings
                store_avg = request.store.average_rating
                product_avgs = [item.stats.average_rating for item in request.product_items]
                combined_avg = (store_avg + sum(product_avgs)) / (len(product_avgs) + 1)
                rating = round(combined_avg)
                # Ensure it's between 1-5
                rating = max(1, min(5, rating))
            
            # Convert StoreReviewRequest to format needed for format_store_review_messages
            store_info = {
                'name': request.store.name,
                'location': request.store.location,
                'average_rating': request.store.average_rating,
                'store_category': request.store_category,
            }
            
            # Convert product items
            product_items_dict = []
            for item in request.product_items:
                product_dict = {
                    'name': item.name,
                    'description': item.description,
                    'price': item.price,
                    'stats': {
                        'average_rating': item.stats.average_rating,
                        'total_reviews': item.stats.total_reviews
                    },
                    'reviews': [{'rating': r.rating, 'content': r.content} for r in item.reviews]
                }
                product_items_dict.append(product_dict)
                
            # Use format_store_review_messages from prompt_manager
            messages = prompt_manager.format_store_review_messages(
                store_info=store_info,
                product_items=product_items_dict,
                rating=rating,
                user_note=request.user_note or "",
                length=request.length or "medium",
                system_language=request.system_language or "User input language"
            )
            
            self.logger.info(f"Using provider {provider_type} to generate store review for {request.store.name}")
            
            # Generate response using provider
            response = await self.llm_generator_provider.generate_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key
            )
            
            # Extract content from response
            review_text = response["content"]
            
            # Clean up response if needed
            review_text = self._clean_response(review_text)
            
            # Log success
            self.logger.info(f"Successfully generated review for store: {request.store.name} using {provider_type}")
            
            return {
                "review": review_text,
                "rating": rating
            }
            
        except Exception as e:
            self.logger.error(f"Error generating store review: {str(e)}")
            raise
    

    def _clean_response(self, text: str) -> str:
        # Remove any leading/trailing quotes
        text = text.strip('"\'')
        # Remove any thinking sections if present
        text = self.llm_generator_provider.clean_thinking(text)
        return text.strip()
    

store_review_generator = StoreReviewGeneratorHandler()