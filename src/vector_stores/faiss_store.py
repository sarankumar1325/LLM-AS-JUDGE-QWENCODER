"""
FAISS vector store implementation.
"""

import pickle
import uuid
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from src.vector_stores.base_store import BaseVectorStore
from config.model_configs import get_vector_store_config

class FAISSVectorStore(BaseVectorStore):
    """FAISS implementation of vector store."""
    
    def __init__(self, 
                 index_path: str = None,
                 dimension: int = 1024,
                 index_type: str = "IndexFlatIP"):
        try:
            import faiss
            
            config = get_vector_store_config("faiss")
            self.index_path = Path(index_path or config["index_path"])
            self.dimension = dimension
            self.index_type = index_type
            
            # Create directory
            self.index_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize FAISS index
            if index_type == "IndexFlatIP":
                self.index = faiss.IndexFlatIP(dimension)  # Inner Product (for cosine similarity)
            elif index_type == "IndexFlatL2":
                self.index = faiss.IndexFlatL2(dimension)  # L2 distance
            elif index_type == "IndexIVFFlat":
                nlist = config.get("nlist", 100)
                quantizer = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
            
            # Storage for document metadata
            self.documents = {}  # id -> document data
            self.id_to_index = {}  # document_id -> index position
            self.index_to_id = {}  # index position -> document_id
            
            # Try to load existing index
            self._load_if_exists()
            
        except ImportError:
            raise ImportError("faiss-cpu or faiss-gpu is required for FAISSVectorStore")
    
    def _load_if_exists(self):
        """Load existing index and metadata if they exist."""
        index_file = self.index_path / "index.faiss"
        metadata_file = self.index_path / "metadata.pkl"
        
        if index_file.exists() and metadata_file.exists():
            try:
                import faiss
                self.index = faiss.read_index(str(index_file))
                
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                    self.documents = metadata.get("documents", {})
                    self.id_to_index = metadata.get("id_to_index", {})
                    self.index_to_id = metadata.get("index_to_id", {})
                    
            except Exception as e:
                print(f"Error loading existing FAISS index: {e}")
                # Reset to empty state
                self.documents = {}
                self.id_to_index = {}
                self.index_to_id = {}
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Add document chunks to FAISS."""
        if not chunks:
            return []
        
        import faiss
        
        # Prepare data
        ids = []
        embeddings = []
        
        for chunk in chunks:
            # Generate unique ID
            chunk_id = chunk.get("id", str(uuid.uuid4()))
            ids.append(chunk_id)
            
            # Extract embedding
            if "embedding" not in chunk:
                raise ValueError("Chunks must have embeddings")
            
            embedding = np.array(chunk["embedding"], dtype=np.float32)
            
            # Normalize for cosine similarity (if using IndexFlatIP)
            if self.index_type == "IndexFlatIP":
                embedding = embedding / np.linalg.norm(embedding)
            
            embeddings.append(embedding)
        
        # Convert to numpy array
        embeddings_array = np.vstack(embeddings)
        
        # Get current index size
        current_size = self.index.ntotal
        
        # Add to FAISS index
        try:
            # Train index if needed (for IVF indices)
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                self.index.train(embeddings_array)
            
            self.index.add(embeddings_array)
            
            # Update metadata
            for i, (chunk_id, chunk) in enumerate(zip(ids, chunks)):
                index_pos = current_size + i
                
                self.documents[chunk_id] = {
                    "id": chunk_id,
                    "content": chunk["content"],
                    "metadata": chunk.get("metadata", {})
                }
                
                self.id_to_index[chunk_id] = index_pos
                self.index_to_id[index_pos] = chunk_id
            
            return ids
            
        except Exception as e:
            raise Exception(f"Error adding documents to FAISS: {e}")
    
    def similarity_search(self, 
                         query_embedding: List[float], 
                         k: int = 5,
                         filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform similarity search without scores."""
        results = self.similarity_search_with_scores(query_embedding, k, filter_dict)
        return [doc for doc, _ in results]
    
    def similarity_search_with_scores(self, 
                                    query_embedding: List[float], 
                                    k: int = 5,
                                    filter_dict: Optional[Dict[str, Any]] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Perform similarity search with scores."""
        if self.index.ntotal == 0:
            return []
        
        try:
            # Prepare query embedding
            query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            
            # Normalize for cosine similarity (if using IndexFlatIP)
            if self.index_type == "IndexFlatIP":
                query_vector = query_vector / np.linalg.norm(query_vector)
            
            # Search
            scores, indices = self.index.search(query_vector, min(k, self.index.ntotal))
            
            # Process results
            documents_with_scores = []
            
            for score, idx in zip(scores[0], indices[0]):
                # Skip invalid indices
                if idx == -1:
                    continue
                
                # Get document ID
                doc_id = self.index_to_id.get(idx)
                if not doc_id or doc_id not in self.documents:
                    continue
                
                doc = self.documents[doc_id].copy()
                
                # Apply filtering if specified
                if filter_dict and not self._matches_filter(doc.get("metadata", {}), filter_dict):
                    continue
                
                # Convert score based on index type
                if self.index_type == "IndexFlatIP":
                    # For inner product (cosine similarity), score is already similarity
                    similarity_score = float(score)
                elif self.index_type == "IndexFlatL2":
                    # For L2 distance, convert to similarity (1 / (1 + distance))
                    similarity_score = 1.0 / (1.0 + float(score))
                else:
                    similarity_score = float(score)
                
                documents_with_scores.append((doc, similarity_score))
            
            # Sort by similarity score (descending)
            documents_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            return documents_with_scores[:k]
            
        except Exception as e:
            raise Exception(f"Error querying FAISS: {e}")
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata or str(metadata[key]) != str(value):
                return False
        return True
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from FAISS (not directly supported, requires rebuild)."""
        try:
            # Remove from metadata
            for doc_id in document_ids:
                if doc_id in self.documents:
                    index_pos = self.id_to_index.get(doc_id)
                    
                    del self.documents[doc_id]
                    if index_pos is not None:
                        del self.id_to_index[doc_id]
                        del self.index_to_id[index_pos]
            
            # Note: FAISS doesn't support direct deletion
            # Would need to rebuild index to actually remove vectors
            print("Warning: FAISS doesn't support direct deletion. Metadata removed, but vectors remain in index.")
            return True
            
        except Exception as e:
            print(f"Error deleting documents from FAISS: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the FAISS index."""
        return {
            "total_documents": len(self.documents),
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "index_path": str(self.index_path),
            "is_trained": getattr(self.index, 'is_trained', True)
        }
    
    def save_store(self, path: str = None) -> bool:
        """Save the FAISS index and metadata to disk."""
        try:
            import faiss
            
            save_path = Path(path) if path else self.index_path
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            index_file = save_path / "index.faiss"
            faiss.write_index(self.index, str(index_file))
            
            # Save metadata
            metadata_file = save_path / "metadata.pkl"
            metadata = {
                "documents": self.documents,
                "id_to_index": self.id_to_index,
                "index_to_id": self.index_to_id,
                "dimension": self.dimension,
                "index_type": self.index_type
            }
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            return True
            
        except Exception as e:
            print(f"Error saving FAISS store: {e}")
            return False
    
    def load_store(self, path: str) -> bool:
        """Load the FAISS index and metadata from disk."""
        try:
            import faiss
            
            load_path = Path(path)
            index_file = load_path / "index.faiss"
            metadata_file = load_path / "metadata.pkl"
            
            if not (index_file.exists() and metadata_file.exists()):
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_file))
            
            # Load metadata
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                self.documents = metadata.get("documents", {})
                self.id_to_index = metadata.get("id_to_index", {})
                self.index_to_id = metadata.get("index_to_id", {})
                self.dimension = metadata.get("dimension", self.dimension)
                self.index_type = metadata.get("index_type", self.index_type)
            
            return True
            
        except Exception as e:
            print(f"Error loading FAISS store: {e}")
            return False
    
    def rebuild_index(self) -> bool:
        """Rebuild the FAISS index (useful after deletions)."""
        try:
            import faiss
            
            if not self.documents:
                return True
            
            # Extract embeddings from documents (if stored)
            embeddings = []
            doc_ids = []
            
            for doc_id, doc_data in self.documents.items():
                if "embedding" in doc_data:
                    embedding = np.array(doc_data["embedding"], dtype=np.float32)
                    if self.index_type == "IndexFlatIP":
                        embedding = embedding / np.linalg.norm(embedding)
                    embeddings.append(embedding)
                    doc_ids.append(doc_id)
            
            if not embeddings:
                print("No embeddings found in documents for rebuild")
                return False
            
            # Create new index
            if self.index_type == "IndexFlatIP":
                new_index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "IndexFlatL2":
                new_index = faiss.IndexFlatL2(self.dimension)
            else:
                print(f"Rebuild not supported for index type: {self.index_type}")
                return False
            
            # Add embeddings
            embeddings_array = np.vstack(embeddings)
            new_index.add(embeddings_array)
            
            # Update mappings
            self.index = new_index
            self.id_to_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}
            self.index_to_id = {i: doc_id for i, doc_id in enumerate(doc_ids)}
            
            return True
            
        except Exception as e:
            print(f"Error rebuilding FAISS index: {e}")
            return False
