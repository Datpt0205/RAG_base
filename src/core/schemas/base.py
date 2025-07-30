from pydantic import BaseModel
from typing_extensions import Literal

from src.core.utils.constants import TypeDatabase, TypeSearch


class RequestUserBase(BaseModel):
    question_input: str
    created_by: str = ''


class RequestWebsocketBase(BaseModel):
    session_id: str
    question: str
    created_time: str
    created_by: str = ''
    llm_model_name: str
    type_db: str = TypeDatabase.Qdrant.value


class RequestRetrievalBase(BaseModel):
    collection_name: str
    query: str
    type_db: str = TypeDatabase.Qdrant.value
    type_search: str = TypeSearch.Hybrid.value
    top_k: int = 3
    is_rerank: bool = True

class RequestRetrievalDocument(BaseModel):
    collection_name: str
    document_id: str
    type_db: str = TypeDatabase.Qdrant.value
    limit: int = 3
    is_sort: bool = True

class DocumentIds(BaseModel):
    document_ids: list[str]= []