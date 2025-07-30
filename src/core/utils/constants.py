from enum import Enum

class MessageType(int, Enum):
    QUESTION = 0,
    ANSWER = 1

class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

class DocumentExtractionBackend(ExtendedEnum):
    Pymupdf = 'pymupdf'
    Docling = 'docling'

class TypeDocument(ExtendedEnum):
    Pdf = 'pdf'
    Word = 'word'
    Pptx = 'pptx'
    Image = 'image'

class TypeDatabase(ExtendedEnum):
    Qdrant = 'qdrant'

class TypeSearch(ExtendedEnum):
    Key_word = 'key_word'
    Semantic = 'semantic'
    Hybrid = 'hybrid'
    Similarity = 'similarity'
    MMR = "mmr"
    SimilarityWithScore = 'similarity_score_threshold'

# LLM Providers
class ProviderType(ExtendedEnum):
    Ollama = 'ollama'
    OpenAI = 'openai'
    Gemini = 'gemini'

class OpenAIModelName(ExtendedEnum):
    GPT35Turbo = "gpt-3.5-turbo"
    GPT41Nano = "gpt-4.1-nano-2025-04-14"
    GPT41Mini = "gpt-4.1-mini-2025-04-14"

class GeminiModelName(ExtendedEnum):
    GeminiPro = "gemini-pro"
    GeminiProVision = "gemini-pro-vision"
    Gemini15Pro = "gemini-1.5-pro"


SCHEMA_DB = [
    {"name": "document_name", "type": "text_general", "indexed": "true", "stored": "true", "multiValued": "false"},
    {"name": "page", "type": "text_general", "indexed": "true", "stored": "true", "multiValued": "false"},
    {"name": "embedding_vector", "type": "knn_vector", "indexed": "true", "stored": "true"},
    {"name": "page_content", "type": "text_general", "indexed": "true", "stored": "true", "multiValued": "false"},
    {"name": "document_id", "type": "text_general", "indexed": "true", "stored": "true", "multiValued": "false"},
    {"name": "is_parent", "type": "boolean", "indexed": "true", "stored": "true", "multiValued": "false"}
]

DPI = 150

CODEMIND_LLM = r""" 
  ____   ___   ____  _____ __  __ ___ _   _ ____  
 / ___| / _ \ |  _ \| ____|  \/  |_ _| \ | |  _ \ 
| |    | | | || | | |  _| | |\/| || ||  \| | | | |
| |___ | |_| || |_| | |___| |  | ||_|| |\  | |_| |
 \____| \___/ |____/|_____|_|  |_|___|_| \_|____/ 

"""