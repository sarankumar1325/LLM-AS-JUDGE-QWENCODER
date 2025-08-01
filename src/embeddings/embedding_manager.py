"""
Embedding management utilities for storing and retrieving embeddings.
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from src.embeddings.nvidia_embedder import BaseEmbedder, EmbedderFactory
from config.settings import settings

class EmbeddingManager:
    """Manages embedding generation, storage, and retrieval."""
    
    def __init__(self, 
                 embedder: Optional[BaseEmbedder] = None,
                 cache_dir: str = None):
        self.embedder = embedder or EmbedderFactory.create_default_embedder()
        self.cache_dir = Path(cache_dir or settings.DATA_DIR / "processed" / "embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file to track embeddings
        self.metadata_file = self.cache_dir / "embeddings_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load embeddings metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save embeddings metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def _get_embedding_key(self, text: str, model_name: str = None) -> str:
        """Generate a unique key for an embedding."""
        import hashlib
        if model_name is None:
            model_name = getattr(self.embedder, 'model_name', 'default')
        
        content_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{model_name}_{content_hash}"
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a given key."""
        return self.cache_dir / f"{key}.pkl"
    
    def embed_chunks(self, 
                    chunks: List[Dict[str, Any]], 
                    use_cache: bool = True,
                    batch_size: int = None) -> List[Dict[str, Any]]:
        """Generate embeddings for text chunks."""
        if batch_size is None:
            config = self.embedder.config if hasattr(self.embedder, 'config') else {}
            batch_size = config.get('batch_size', 32)
        
        embedded_chunks = []
        
        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = [chunk["content"] for chunk in batch_chunks]
            batch_keys = [self._get_embedding_key(text) for text in batch_texts]
            
            # Check cache for existing embeddings
            cached_embeddings = {}
            uncached_indices = []
            uncached_texts = []
            
            if use_cache:
                for idx, key in enumerate(batch_keys):
                    cache_path = self._get_cache_path(key)
                    if cache_path.exists():
                        try:
                            with open(cache_path, 'rb') as f:
                                cached_embeddings[idx] = pickle.load(f)
                        except Exception as e:
                            print(f"Error loading cached embedding {key}: {e}")
                            uncached_indices.append(idx)
                            uncached_texts.append(batch_texts[idx])
                    else:
                        uncached_indices.append(idx)
                        uncached_texts.append(batch_texts[idx])
            else:
                uncached_indices = list(range(len(batch_texts)))
                uncached_texts = batch_texts
            
            # Generate embeddings for uncached texts
            new_embeddings = {}
            if uncached_texts:
                try:
                    embeddings = self.embedder.embed_batch(uncached_texts)
                    
                    for i, embedding in enumerate(embeddings):
                        idx = uncached_indices[i]
                        key = batch_keys[idx]
                        new_embeddings[idx] = embedding
                        
                        # Cache the embedding
                        if use_cache:
                            cache_path = self._get_cache_path(key)
                            try:
                                with open(cache_path, 'wb') as f:
                                    pickle.dump(embedding, f)
                                
                                # Update metadata
                                self.metadata[key] = {
                                    'model_name': getattr(self.embedder, 'model_name', 'default'),
                                    'dimension': len(embedding),
                                    'created_at': datetime.now().isoformat(),
                                    'text_length': len(batch_texts[idx])
                                }
                            except Exception as e:
                                print(f"Error caching embedding {key}: {e}")
                
                except Exception as e:
                    print(f"Error generating embeddings: {e}")
                    # Use zero embeddings as fallback
                    dimension = getattr(self.embedder, 'get_dimension', lambda: 1024)()
                    for idx in uncached_indices:
                        new_embeddings[idx] = [0.0] * dimension
            
            # Combine cached and new embeddings
            for idx, chunk in enumerate(batch_chunks):
                embedding = cached_embeddings.get(idx) or new_embeddings.get(idx)
                
                if embedding:
                    chunk_with_embedding = chunk.copy()
                    chunk_with_embedding["embedding"] = embedding
                    chunk_with_embedding["metadata"]["embedding_model"] = getattr(
                        self.embedder, 'model_name', 'default'
                    )
                    chunk_with_embedding["metadata"]["embedding_dimension"] = len(embedding)
                    embedded_chunks.append(chunk_with_embedding)
        
        # Save metadata
        if use_cache:
            self._save_metadata()
        
        return embedded_chunks
    
    def embed_queries(self, 
                     queries: List[str], 
                     use_cache: bool = True) -> Dict[str, List[float]]:
        """Generate embeddings for query texts."""
        query_embeddings = {}
        
        for query in queries:
            key = self._get_embedding_key(query)
            
            # Check cache
            embedding = None
            if use_cache:
                cache_path = self._get_cache_path(key)
                if cache_path.exists():
                    try:
                        with open(cache_path, 'rb') as f:
                            embedding = pickle.load(f)
                    except Exception as e:
                        print(f"Error loading cached query embedding: {e}")
            
            # Generate if not cached
            if embedding is None:
                try:
                    embedding = self.embedder.embed_text(query)
                    
                    # Cache the embedding
                    if use_cache:
                        cache_path = self._get_cache_path(key)
                        try:
                            with open(cache_path, 'wb') as f:
                                pickle.dump(embedding, f)
                            
                            self.metadata[key] = {
                                'model_name': getattr(self.embedder, 'model_name', 'default'),
                                'dimension': len(embedding),
                                'created_at': datetime.now().isoformat(),
                                'text_length': len(query),
                                'type': 'query'
                            }
                        except Exception as e:
                            print(f"Error caching query embedding: {e}")
                
                except Exception as e:
                    print(f"Error generating query embedding: {e}")
                    # Use zero embedding as fallback
                    dimension = getattr(self.embedder, 'get_dimension', lambda: 1024)()
                    embedding = [0.0] * dimension
            
            query_embeddings[query] = embedding
        
        # Save metadata
        if use_cache:
            self._save_metadata()
        
        return query_embeddings
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings."""
        total_embeddings = len(self.metadata)
        models = {}
        dimensions = {}
        
        for key, meta in self.metadata.items():
            model_name = meta.get('model_name', 'unknown')
            dimension = meta.get('dimension', 0)
            
            models[model_name] = models.get(model_name, 0) + 1
            dimensions[dimension] = dimensions.get(dimension, 0) + 1
        
        return {
            'total_embeddings': total_embeddings,
            'models': models,
            'dimensions': dimensions,
            'cache_size_mb': sum(
                f.stat().st_size for f in self.cache_dir.glob("*.pkl")
            ) / (1024 * 1024)
        }
    
    def clear_cache(self, model_name: str = None):
        """Clear embedding cache."""
        if model_name:
            # Clear cache for specific model
            keys_to_remove = [
                key for key, meta in self.metadata.items()
                if meta.get('model_name') == model_name
            ]
            
            for key in keys_to_remove:
                cache_path = self._get_cache_path(key)
                if cache_path.exists():
                    cache_path.unlink()
                del self.metadata[key]
        else:
            # Clear all cache
            for f in self.cache_dir.glob("*.pkl"):
                f.unlink()
            self.metadata = {}
        
        self._save_metadata()
    
    def export_embeddings(self, 
                         output_path: str, 
                         format: str = "numpy") -> str:
        """Export embeddings to file."""
        output_path = Path(output_path)
        
        if format == "numpy":
            embeddings_data = []
            metadata_list = []
            
            for key, meta in self.metadata.items():
                cache_path = self._get_cache_path(key)
                if cache_path.exists():
                    try:
                        with open(cache_path, 'rb') as f:
                            embedding = pickle.load(f)
                        embeddings_data.append(embedding)
                        metadata_list.append({**meta, 'key': key})
                    except Exception as e:
                        print(f"Error loading embedding {key}: {e}")
            
            if embeddings_data:
                np.save(output_path / "embeddings.npy", np.array(embeddings_data))
                with open(output_path / "metadata.json", 'w') as f:
                    json.dump(metadata_list, f, indent=2, default=str)
        
        return str(output_path)
