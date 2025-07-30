import asyncio
from typing import List, Optional
from langchain_core.documents import Document
from src.core.utils.logger.custom_logging import LoggerMixin
from src.features.rag.handlers.retrieval_handler import default_search_retrieval
from src.core.database.services.collection_management_service import CollectionManagementService

class MultiCollectionRetrieval(LoggerMixin):
    """
    Retriever that can search across multiple collections (personal and organizational)
    """
    
    def __init__(self):
        """Initialize the multi-collection retriever"""
        super().__init__()
        self.search_retrieval = default_search_retrieval
        self.collection_service = CollectionManagementService()
    
    async def retrieve_from_collections(
        self,
        query: str,
        user_id: str,
        organization_id: Optional[str] = None,
        top_k: int = 5,
        include_personal: bool = True,
        include_organizational: bool = True
    ) -> List[Document]:
        """
        Retrieve documents from both personal and organizational collections
        
        Args:
            query: User's query
            user_id: User's ID for filtering collections
            organization_id: Optional organization ID for filtering collections
            top_k: Number of top results to return per collection
            include_personal: Whether to search in personal collections
            include_organizational: Whether to search in organizational collections
            
        Returns:
            List[Document]: Combined and reranked results
        """
        try:
            # 1. Get collections that the user has access to
            user_collections = self.collection_service.get_user_collections(
                user_id=user_id,
                organization_id=organization_id,
                include_personal=include_personal,
                include_organizational=include_organizational
            )
            
            # Extract collection names
            collection_names = [c.get("collection_name") for c in user_collections]
            
            # 2. Search in all collections in parallel
            all_results = []
            tasks = []
            
            # Add tasks for each collection
            for collection_name in collection_names:
                tasks.append(self.search_retrieval.qdrant_retrieval(
                    query=query,
                    collection_name=collection_name,
                    top_k=top_k
                ))
            
            # Execute all retrieval tasks in parallel
            if tasks:
                results_list = await asyncio.gather(*tasks)
                
                # Flatten results and add metadata about source collection
                for i, results in enumerate(results_list):
                    collection_info = user_collections[i]
                    collection_name = collection_info.get("collection_name")
                    is_personal = collection_info.get("is_personal", False)
                    
                    for doc in results:
                        # Add collection source metadata
                        if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                            doc.metadata["source_collection"] = collection_name
                            doc.metadata["is_personal"] = is_personal
                        
                        all_results.append(doc)
            
            # Sort the results by relevance (assuming results have score)
            all_results = sorted(
                all_results,
                key=lambda doc: getattr(doc.metadata, 'score', 0) if hasattr(doc, 'metadata') else 0,
                reverse=True
            )
            
            return all_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error in multi-collection retrieval: {str(e)}")
            return []

# Create singleton instance
multi_collection_retriever = MultiCollectionRetrieval()