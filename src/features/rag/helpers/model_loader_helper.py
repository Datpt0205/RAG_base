from functools import lru_cache
from typing import Dict, Any, Optional
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
import torch

from src.core.utils.config import settings
from src.core.utils.config_loader import ConfigReaderInstance
from src.core.utils.logger.custom_logging import LoggerMixin


# Load configuration
config = ConfigReaderInstance.yaml.read_config_from_file(settings.MODEL_CONFIG_FILENAME)
rerank_config = config.get('RERANKING_MODEL', {})

# Define cache directory for models
CACHE_DIR = "/app/cache"

class ModelLoader(LoggerMixin):
    """
    Singleton class to load and cache models across the application.
    This ensures models are loaded only once and reused throughout the app.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    @lru_cache(maxsize=5)
    def get_flag_reranker(model_key: Optional[str] = None) -> FlagReranker:
        """
        Load and cache a FlagReranker model.
        
        Args:
            model_key (Optional[str]): Key of model in config or direct model name
            
        Returns:
            FlagReranker: Loaded model
        """
        logger = ModelLoader().logger
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine model name based on key or default
        model_name = ModelLoader._resolve_model_name(model_key, "BAAI_COLLECTION_RERANK")
        
        logger.info(f"Loading FlagReranker model: {model_name} on device: {device}")
        
        return FlagReranker(model_name, use_fp16=(device.type == "cuda"))
    
    @staticmethod
    @lru_cache(maxsize=5)
    def get_sentence_transformer(model_key: Optional[str] = None) -> SentenceTransformer:
        """
        Load and cache a SentenceTransformer model.
        
        Args:
            model_key (Optional[str]): Key of model in config or direct model name
            
        Returns:
            SentenceTransformer: Loaded model
        """
        logger = ModelLoader().logger
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine model name based on key or default
        model_name = ModelLoader._resolve_model_name(model_key, "CROSS_ENCODER_MS_MARCO_RERANK")
        
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        return SentenceTransformer(model_name, cache_folder=CACHE_DIR, device=str(device))
    
    @staticmethod
    @lru_cache(maxsize=5)
    def get_tokenizer(model_name: str = "BAAI/bge-small-en-v1.5") -> AutoTokenizer:
        """
        Load and cache a tokenizer.
        
        Args:
            model_name (str): Name of the tokenizer model
            
        Returns:
            AutoTokenizer: Loaded tokenizer
        """
        logger = ModelLoader().logger
        logger.info(f"Loading tokenizer: {model_name}")
        return AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    
    @staticmethod
    def _resolve_model_name(model_key: Optional[str], default_key: str) -> str:
        """
        Resolve model name from config based on key or use default.
        
        Args:
            model_key (Optional[str]): Key in config or direct model name
            default_key (str): Default key to use if model_key is None
            
        Returns:
            str: Resolved model name
        """
        if model_key is None:
            # Use default key if available
            if default_key in rerank_config:
                return rerank_config[default_key]
            else:
                # Fallback to first model in config
                first_key = list(rerank_config.keys())[0] if rerank_config else default_key
                return rerank_config.get(first_key, "cross-encoder/ms-marco-MiniLM-L-2-v2")
        elif model_key in rerank_config:
            # Get model name from config using provided key
            return rerank_config[model_key]
        else:
            # Use key directly as model name if not found in config
            return model_key

# Create singleton instances for direct import
flag_reranker = ModelLoader.get_flag_reranker()
sentence_transformer = ModelLoader.get_sentence_transformer()
default_tokenizer = ModelLoader.get_tokenizer()