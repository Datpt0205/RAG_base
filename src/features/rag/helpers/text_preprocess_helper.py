import os
import torch
from functools import lru_cache
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from fastembed.text import TextEmbedding
from fastembed.sparse.bm25 import Bm25
from fastembed.late_interaction import LateInteractionTextEmbedding

from src.core.utils.config import settings
from src.core.utils.config_loader import ConfigReaderInstance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FASTEMBED_CACHE_DIR = os.environ.get('FASTEMBED_CACHE_DIR', '/app/cache')
model_config = ConfigReaderInstance.yaml.read_config_from_file(settings.MODEL_CONFIG_FILENAME)
EMBEDDING_MODEL = model_config.get('EMBEDDING_MODEL', {}).get('SENTENCE_TRANSFORMER', {})

TEXT_EMBEDDING_MODEL = model_config.get('EMBEDDING_MODEL', {}).get('TEXT_EMBEDDING_MODEL', {})
LATE_INTERACTION_TEXT_EMBEDDING_MODEL = model_config.get('EMBEDDING_MODEL', {}).get('LATE_INTERACTION_TEXT_EMBEDDING_MODEL', {})
BM25_EMBEDDING_MODEL = model_config.get('EMBEDDING_MODEL', {}).get('BM25_EMBEDDING_MODEL', {})

@lru_cache()
def get_embedding_model():
    _embedding = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL, 
                                               model_kwargs={
                                                                'token': settings.HUGGINGFACE_ACCESS_TOKEN,
                                                                'device': str(device)
                                                            }
                                              )
    return _embedding

embedding_function = get_embedding_model()


@lru_cache()
def get_text_embedding_model() -> TextEmbedding:
    _text_embedding = TextEmbedding(model_name=TEXT_EMBEDDING_MODEL, cache_dir=FASTEMBED_CACHE_DIR, gpu=(device.type == "cuda"))
    return _text_embedding

@lru_cache()
def get_late_interaction_text_embedding_model() -> LateInteractionTextEmbedding:
    _late_interaction_text_embedding = LateInteractionTextEmbedding(model_name=LATE_INTERACTION_TEXT_EMBEDDING_MODEL, cache_dir=FASTEMBED_CACHE_DIR, gpu=(device.type == "cuda"))
    return _late_interaction_text_embedding

@lru_cache()
def get_bm25_embedding_model() -> Bm25:
    _bm25_embedding = Bm25(model_name=BM25_EMBEDDING_MODEL, cache_dir=FASTEMBED_CACHE_DIR)
    return _bm25_embedding

text_embedding_model = get_text_embedding_model()
late_interaction_text_embedding_model = get_late_interaction_text_embedding_model()
bm25_embedding_model = get_bm25_embedding_model()





