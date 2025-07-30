import torch
import numpy as np
from typing import List, Dict, Any, Optional

from src.core.utils.logger.custom_logging import LoggerMixin
from src.features.rag.helpers.model_loader_helper import ModelLoader, sentence_transformer, default_tokenizer


class RerankHandler(LoggerMixin):
    """Handler for reranking retrieved documents using local models instead of Triton Server.
    
    This class provides methods to rerank documents using embeddings from sentence transformers
    models while maintaining compatibility with the original codebase.
    """
    
    def __init__(self, model_key: Optional[str] = None):
        """Initialize the reranking handler with a specified model from config.
        
        Args:
            model_key (Optional[str]): Key of the reranking model in config file.
                                      If None, uses the default model.
        """
        super().__init__()
        
        if model_key is None:
            # Use the singleton instance for better performance
            self.model = sentence_transformer
            self.model_name = "default"  # Just for logging
        else:
            # Load a specific model if requested
            self.model = ModelLoader.get_sentence_transformer(model_key)
            self.model_name = model_key
            
        self.logger.info(f"Using reranker model: {self.model_name}")
        
        # Use the singleton tokenizer
        self.tokenizer = default_tokenizer
    
    def tokenize_input(self, text: str) -> Dict[str, Any]:
        """Tokenizes the input text and prepares it for the model.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            Dict[str, Any]: The request body compatible with the original code structure.
        """
        # Keep the same interface as the original code
        encoding = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        
        input_ids = encoding['input_ids'].numpy().tolist()  
        attention_mask = encoding['attention_mask'].numpy().tolist()  
        token_type_ids = encoding.get('token_type_ids', None)

        # Format the same way as original code for compatibility
        request_body = {
            "inputs": [
                {
                    "name": "input_ids",
                    "shape": [1, len(input_ids[0])],
                    "datatype": "INT64",
                    "data": input_ids
                },
                {
                    "name": "attention_mask",
                    "shape": [1, len(attention_mask[0])],
                    "datatype": "INT64",
                    "data": attention_mask
                }
            ]
        }

        if token_type_ids is not None:
            request_body["inputs"].append({
                "name": "token_type_ids",
                "shape": [1, len(token_type_ids[0])],
                "datatype": "INT64",
                "data": token_type_ids.numpy().tolist()
            })

        return request_body

    def request_ranking_triton_kserve(self, text_input: str) -> Dict[str, Any]:
        """Gets embeddings for a text input using the local model.
        
        Replaces the original method that called Triton server.

        Args:
            text_input (str): The input text.

        Returns:
            Dict[str, Any]: A dictionary with embeddings in the same format as the original Triton response.
        """
        # Encode the text using the loaded model
        with torch.no_grad():
            embedding = self.model.encode(text_input, convert_to_numpy=True)
        
        # Return in the same format as the original Triton response
        return {
            'outputs': [
                {
                    'name': 'output',
                    'data': embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                }
            ]
        }

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculates the cosine similarity between two vectors.

        Args:
            vec_a (np.ndarray): First vector.
            vec_b (np.ndarray): Second vector.

        Returns:
            float: The cosine similarity score.
        """
        if len(vec_a) == 0 or len(vec_b) == 0:
            return 0.0
            
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)

    def pad_or_truncate(self, embedding: List[float], target_size: int) -> List[float]:
        """Pads or truncates the embedding to a specified target size.

        Args:
            embedding (List[float]): The embedding to be padded or truncated.
            target_size (int): The desired size for the embedding.

        Returns:
            List[float]: The modified embedding of target size.
        """
        if len(embedding) > target_size:
            return embedding[:target_size]  # Truncate
        elif len(embedding) < target_size:
            return embedding + [0] * (target_size - len(embedding))  # Pad
        return embedding

    def rerank_embeddings(self, embeddings: List[np.ndarray], query_embedding: np.ndarray, candidates: List) -> List[tuple]:
        """Reranks embeddings based on their similarity to the query embedding.

        Args:
            embeddings (List[np.ndarray]): The candidate embeddings.
            query_embedding (np.ndarray): The embedding of the query.
            candidates (List): The candidate documents with doc_id attribute.

        Returns:
            List[tuple]: A list of ranked documents with their scores.
        """
        # Calculate similarity scores between query and candidate embeddings
        scores = {}
        for candidate, embedding in zip(candidates, embeddings):
            if len(embedding) > 0 and len(query_embedding) > 0:
                scores[candidate.doc_id] = self.cosine_similarity(np.array(embedding), np.array(query_embedding))
            else:
                scores[candidate.doc_id] = 0.0
       
        # Sort candidates based on scores in descending order
        ranked_embeddings = sorted(scores.items(), key=lambda item: item[1], reverse=True)
     
        return ranked_embeddings

    def process_candidates(self, candidates: List, query: str, threshold: float) -> List[Dict[str, Any]]:
        """Processes candidate documents against a query and filters based on similarity scores.

        Args:
            candidates (List): List of candidate documents with doc_id and content attributes.
            query (str): The query string.
            threshold (float): The minimum score for a candidate to be considered.

        Returns:
            List[Dict[str, Any]]: Filtered candidates with their scores.
        """
        if not query or not candidates:
            self.logger.warning("Empty query or candidates list")
            return []
            
        try:
            embeddings = []

            # Get embeddings for each candidate
            for candidate in candidates:
                resp = self.request_ranking_triton_kserve(candidate.content)
                embedding = resp.get('outputs', [{}])[0].get("data")
                if embedding is not None:
                    embeddings.append(np.array(embedding))
                else:
                    embeddings.append(np.array([]))

            # Get the query embedding
            query_resp = self.request_ranking_triton_kserve(query)
            query_embedding = np.array(query_resp.get('outputs', [{}])[0].get("data"))

            # Determine the target size for padding
            valid_embeddings = [e for e in embeddings if len(e) > 0]
            target_size = max(len(embedding) for embedding in valid_embeddings) if valid_embeddings else 0
            if target_size == 0:
                self.logger.warning("No valid embeddings found")
                return []

            # Pad or truncate embeddings to the target size
            embeddings = [self.pad_or_truncate(embedding.tolist(), target_size) for embedding in embeddings]
            query_embedding = self.pad_or_truncate(query_embedding.tolist(), target_size)

            # Rerank the embeddings based on the query
            ranked_results = self.rerank_embeddings(embeddings, query_embedding, candidates)

            # Map the results to the desired output format
            mapped_results = []
            for doc_id, score in ranked_results:
                for candidate in candidates:
                    if candidate.doc_id == doc_id:
                        result = {
                            'doc_id': doc_id,
                            'score': score,
                            'content': candidate.content
                        }
                        # Preserve organization_id if present
                        if hasattr(candidate, 'organization_id') and candidate.organization_id:
                            result['organization_id'] = candidate.organization_id
                        mapped_results.append(result)
                        break

            # Filter results based on the similarity threshold
            filtered_results = [item for item in mapped_results if item['score'] >= threshold]
            
            # Log results
            self.logger.info(f"Reranking results: {len(filtered_results)} items passed threshold {threshold} out of {len(candidates)} candidates")
            
            return filtered_results
        except Exception as e:
            self.logger.error(f"Error during reranking: {str(e)}")
            # Return empty list rather than raising exception
            return []

# Create a singleton instance for default usage
default_reranker = RerankHandler()