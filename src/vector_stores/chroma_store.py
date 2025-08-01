"""
ChromaDB vector store implementation.
"""

import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from src.vector_stores.base_store import BaseVectorStore
from config.model_configs import get_vector_store_config

class ChromaVectorStore(BaseVectorStore):
    """ChromaDB implementation of vector store."""
    
    def __init__(self, 
                 persist_directory: str = None,
                 collection_name: str = None):
        try:
            import chromadb
            from chromadb.config import Settings
            
            config = get_vector_store_config("chroma")
            self.persist_directory = persist_directory or config["persist_directory"]
            self.collection_name = collection_name or config["collection_name"]
            
            # Create persist directory
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
        except ImportError:
            raise ImportError("chromadb is required for ChromaVectorStore")
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Add document chunks to ChromaDB."""
        if not chunks:
            return []
        
        # Generate IDs if not present
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            # Generate unique ID
            chunk_id = chunk.get("id", str(uuid.uuid4()))
            ids.append(chunk_id)
            
            # Extract embedding
            if "embedding" not in chunk:
                raise ValueError("Chunks must have embeddings")
            embeddings.append(chunk["embedding"])
            
            # Extract text content
            documents.append(chunk["content"])
            
            # Prepare metadata (ChromaDB doesn't support nested objects)
            metadata = chunk.get("metadata", {}).copy()
            # Flatten nested metadata and convert to strings
            flat_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (dict, list)):
                    flat_metadata[key] = str(value)
                else:
                    flat_metadata[key] = str(value) if value is not None else ""
            
            metadatas.append(flat_metadata)
        
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            return ids
        except Exception as e:
            raise Exception(f"Error adding documents to ChromaDB: {e}")
    
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
        try:
            # Prepare where clause for filtering
            where_clause = {}
            if filter_dict:
                # Convert filter to ChromaDB format
                for key, value in filter_dict.items():
                    where_clause[key] = {"$eq": str(value)}
            
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            documents_with_scores = []
            
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    doc = {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] or {}
                    }
                    
                    # Convert distance to similarity score (1 - distance for cosine)
                    distance = results["distances"][0][i]
                    similarity_score = 1.0 - distance
                    
                    documents_with_scores.append((doc, similarity_score))
            
            return documents_with_scores
            
        except Exception as e:
            raise Exception(f"Error querying ChromaDB: {e}")
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from ChromaDB."""
        try:
            self.collection.delete(ids=document_ids)
            return True
        except Exception as e:
            print(f"Error deleting documents from ChromaDB: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB collection."""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            return {"error": str(e)}
    
    def save_store(self, path: str) -> bool:
        """ChromaDB persists automatically, so this is a no-op."""
        return True
    
    def load_store(self, path: str) -> bool:
        """ChromaDB loads automatically from persist directory."""
        return True
    
    def reset_collection(self) -> bool:
        """Reset the collection (delete all documents)."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception as e:
            print(f"Error resetting ChromaDB collection: {e}")
            return False
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID."""
        try:
            results = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"]
            )
            
            if results["ids"] and results["ids"][0]:
                return {
                    "id": results["ids"][0],
                    "content": results["documents"][0],
                    "metadata": results["metadatas"][0] or {}
                }
            return None
            
        except Exception as e:
            print(f"Error getting document from ChromaDB: {e}")
            return None
    
    def update_document(self, doc_id: str, 
                       content: str = None, 
                       embedding: List[float] = None, 
                       metadata: Dict[str, Any] = None) -> bool:
        """Update a document in ChromaDB."""
        try:
            update_data = {"ids": [doc_id]}
            
            if content is not None:
                update_data["documents"] = [content]
            
            if embedding is not None:
                update_data["embeddings"] = [embedding]
            
            if metadata is not None:
                # Flatten metadata
                flat_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (dict, list)):
                        flat_metadata[key] = str(value)
                    else:
                        flat_metadata[key] = str(value) if value is not None else ""
                update_data["metadatas"] = [flat_metadata]
            
            self.collection.update(**update_data)
            return True
            
        except Exception as e:
            print(f"Error updating document in ChromaDB: {e}")
            return False
