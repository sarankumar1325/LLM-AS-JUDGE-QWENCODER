"""
Base vector store interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

class BaseVectorStore(ABC):
    """Base class for vector stores."""
    
    @abstractmethod
    def add_documents(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Add document chunks to the vector store."""
        pass
    
    @abstractmethod
    def similarity_search(self, 
                         query_embedding: List[float], 
                         k: int = 5,
                         filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform similarity search."""
        pass
    
    @abstractmethod
    def similarity_search_with_scores(self, 
                                    query_embedding: List[float], 
                                    k: int = 5,
                                    filter_dict: Optional[Dict[str, Any]] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Perform similarity search with scores."""
        pass
    
    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the vector store."""
        pass
    
    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        pass
    
    @abstractmethod
    def save_store(self, path: str) -> bool:
        """Save the vector store to disk."""
        pass
    
    @abstractmethod
    def load_store(self, path: str) -> bool:
        """Load the vector store from disk."""
        pass
