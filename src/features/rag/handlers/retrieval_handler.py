import numpy as np
from typing import List, Optional
from langchain_core.documents import Document

from src.core.utils.config import settings
from src.core.utils.logger.custom_logging import LoggerMixin
from src.features.rag.helpers.qdrant_connection_helper import QdrantConnection
from src.features.rag.helpers.model_loader_helper import ModelLoader, flag_reranker
from src.core.database.services.collection_management_service import CollectionManagementService


class SearchRetrieval(LoggerMixin):
    """
    Handler for retrieving and reranking documents from the vector database.
    Uses singleton pattern to ensure models are loaded only once.
    """
    
    def __init__(self, model_key: Optional[str] = None):
        """
        Initialize the search retrieval handler.
        
        Args:
            model_key (Optional[str]): Key of reranking model in config.
                                      If None, uses the default model.
        """
        super().__init__()
        self.qdrant_client = QdrantConnection()
        self.collection_service = CollectionManagementService()
        
        if model_key is None:
            # Use singleton instance for better performance
            self.reranker = flag_reranker
            self.model_name = "default"  # Just for logging
        else:
            # Load specific model if requested
            self.reranker = ModelLoader.get_flag_reranker(model_key)
            self.model_name = model_key
            
        # self.logger.info(f"Using FlagReranker model: {self.model_name}")
    
    def _query_retrieval_reranking(self, candidates: List[Document], query: str, threshold=0.06) -> List[Document]:
        """
        Rerank the candidate documents based on their relevance to the query.
        
        Args:
            candidates (List[Document]): List of retrieved documents
            query (str): Query string
            threshold (float): Minimum score threshold
            
        Returns:
            List[Document]: Reranked and filtered documents
        """
        if not candidates:
            return []
            
        query_docs_pair = []
        for candidate in candidates:
            query_docs_pair.append([query, candidate.page_content.strip()])

        scores = self.reranker.compute_score(query_docs_pair, normalize=True)
        if len(query_docs_pair) == 1:
            scores = np.array([scores])
        else:
            scores = np.array(scores)
        
        # Sorted scores and indices
        sorted_indices = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
        sorted_scores = [scores[i] for i in sorted_indices]
        # self.logger.info(f"[RERANKING] - Reranking Sorted Scores: {sorted_scores}")

        filtered_indices = [index for index, score in zip(sorted_indices, sorted_scores) if score >= threshold]
        rerank_content = [query_docs_pair[index][1] for index in filtered_indices]
        
        content_to_results_map = {candidate.page_content.strip(): candidate for candidate in candidates}
        
        final_results = [content_to_results_map[content] for content in rerank_content if content in content_to_results_map]
        for idx, doc in enumerate(final_results):
            content_preview = doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content
            self.logger.info(f"[RERANKING] Document #{idx+1}: '{content_preview}' | doc_name={doc.metadata.get('document_name', 'N/A')}")
            
        # self.logger.info(f"[RERANKING] - Final Results: {final_results}")
        
        return final_results

    async def qdrant_retrieval(
            self, 
            query: str | dict,
            top_k: int = 5,
            collection_name: str = settings.QDRANT_COLLECTION_NAME
        ) -> Optional[List[Document]]:
        """
        Retrieve documents from Qdrant, rerank them, and return the top results.
        
        Args:
            query (str | dict): Query string or dict with query field
            top_k (int): Number of top results to return
            collection_name (str): Name of the collection to search
            
        Returns:
            Optional[List[Document]]: Retrieved and reranked documents
        """
        if type(query) == dict:
            query = query.get('query')

        try:            
            # Check if collection exists
            if not self.qdrant_client.client.collection_exists(collection_name=collection_name):
                self.logger.warning(f"[RETRIEVE] Collection {collection_name} does not exist")
                return []
            
            # Retrieve documents
            self.logger.info(f"[RETRIEVE] - STARTING RETRIEVAL: '{query}' in collection: {collection_name}")
            docs = await self.qdrant_client.hybrid_search(query=query, collection_name=collection_name)
            for idx, doc in enumerate(docs[:5]):
                self.logger.info(f"[RETRIEVE] Top {idx+1}: doc_name={doc.metadata.get('document_name', 'N/A')},"
                                 f"headers={doc.metadata.get('headers', 'N/A')[:500]}...")
            
            # Rerank documents
            self.logger.info(f"[RERANK] - STARTING RERANKING OF {len(docs)} DOCUMENTS - threshold 0.4")
            docs = self._query_retrieval_reranking(docs, query, 0.4)
            for idx, doc in enumerate(docs[:5]): 
                self.logger.info(f"[RERANK] Top {idx+1}: doc_name={doc.metadata.get('document_name', 'N/A')}, "
                                 f"headers={doc.metadata.get('headers', 'N/A')[:500]}...")
                
            # Get full document content
            self.logger.info(f"[HEADERS] STARTING FULL CONTENT RETRIEVAL for {len(docs)} DOCUMENTS")
            extended_docs = await self.qdrant_client.query_headers(docs, collection_name)
            for idx, doc in enumerate(extended_docs[:5]):
                self.logger.info(f"[HEADERS] Final Top {idx+1}: doc_name={doc.metadata.get('document_name', 'N/A')}, "
                                 f"headers={doc.metadata.get('headers', 'N/A')[:500]}..., content_length={len(doc.page_content)}")
                
            self.logger.debug(f"Retrieved {len(extended_docs)} documents from {collection_name}")
            
            return extended_docs[:top_k] 
        except Exception as e:
            self.logger.error('event=query-relevant-context-in-database '
                             'message="Failed to retrieve relevant context from database"'
                             f'error={e}')
            return []


    async def check_collection_access(
            self,
            user_id: str,
            collection_name: str,
            organization_id: Optional[str] = None,
            required_permission: str = "read"
        ) -> bool:
        """
        Check if user has access to a collection
        
        Args:
            user_id: User ID
            collection_name: Collection name
            organization_id: Organization ID
            required_permission: Required permission level
            
        Returns:
            bool: True if user has access, False otherwise
        """
        # Delegate to collection service
        return self.collection_service.check_collection_permission(
            user_id=user_id,
            collection_name=collection_name,
            organization_id=organization_id,
            required_permission=required_permission
        )

# Create a singleton instance for default usage
default_search_retrieval = SearchRetrieval()