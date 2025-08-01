"""
NVIDIA embedding client for Llama-3.2 NemoRetriever embeddings.
Enhanced with caching, batch processing, and error handling.
"""

import os
import time
import json
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import pickle
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from abc import ABC, abstractmethod
import sys

# Add parent directories to path for imports
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
sys.path.insert(0, str(root_dir))

from config.settings import settings
from config.model_configs import get_embedding_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class EmbeddingRequest:
    """Data class for embedding requests."""
    text: str
    input_type: str = "passage"  # or "query"
    model: str = "nvidia/nv-embedqa-e5-v5"

@dataclass
class EmbeddingResponse:
    """Data class for embedding responses."""
    embedding: List[float]
    text: str
    token_count: int
    processing_time: float

class BaseEmbedder(ABC):
    """Base class for embedding models."""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        pass

class NVIDIAEmbedder(BaseEmbedder):
    """Enhanced NVIDIA API client for generating embeddings using NemoRetriever."""
    
    def __init__(self, api_key: str = None, model: str = None):
        """Initialize the NVIDIA embedder.
        
        Args:
            api_key: NVIDIA API key
            model: Model name to use for embeddings
        """
        self.api_key = api_key or settings.NVIDIA_API_KEY
        self.model = model or settings.EMBEDDING_MODEL
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.embeddings_url = f"{self.base_url}/embeddings"
        self.config = get_embedding_config(self.model)
        
        # Request configuration
        self.max_retries = 3
        self.timeout = 30
        self.rate_limit_delay = 1.0  # seconds between requests
        
        # Setup session with retry strategy
        self.session = self._setup_session()
        
        # Cache configuration
        self.cache_enabled = True
        self.cache_dir = Path(settings.DATA_DIR) / "cache" / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized NVIDIA embedder with model: {self.model}")
    
    def _setup_session(self) -> requests.Session:
        """Setup requests session with retry strategy."""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set headers
        session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "RAG-Evaluation-System/1.0"
        })
        
        return session
    
    def _get_cache_key(self, text: str, input_type: str = "passage") -> str:
        """Generate cache key for text."""
        content = f"{text}_{input_type}_{self.model}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_embedding(self, cache_key: str) -> Optional[EmbeddingResponse]:
        """Get embedding from cache if available."""
        if not self.cache_enabled:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, response: EmbeddingResponse):
        """Save embedding to cache."""
        if not self.cache_enabled:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(response, f)
        except Exception as e:
            logger.warning(f"Failed to save to cache {cache_file}: {e}")
    
    def embed_text(self, text: str, input_type: str = "passage") -> List[float]:
        """Generate embedding for a single text (BaseEmbedder interface)."""
        response = self.embed_text_detailed(text, input_type)
        return response.embedding
    
    def embed_text_detailed(self, text: str, input_type: str = "passage") -> EmbeddingResponse:
        """Generate embedding for a single text with detailed response.
        
        Args:
            text: Text to embed
            input_type: Type of input ("passage" or "query")
            
        Returns:
            EmbeddingResponse with embedding and metadata
        """
        # Check cache first
        cache_key = self._get_cache_key(text, input_type)
        cached_response = self._get_cached_embedding(cache_key)
        if cached_response:
            logger.debug(f"Using cached embedding for text: {text[:50]}...")
            return cached_response
        
        # Prepare request
        start_time = time.time()
        payload = {
            "input": text,
            "model": self.model,
            "input_type": input_type
        }
        
        try:
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            # Make API request
            response = self.session.post(
                self.embeddings_url,
                json=payload,
                timeout=self.timeout
            )
            
            # Check response
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            processing_time = time.time() - start_time
            
            # Extract embedding
            if "data" in result and len(result["data"]) > 0:
                embedding = result["data"][0]["embedding"]
                token_count = result.get("usage", {}).get("total_tokens", 0)
            else:
                raise ValueError("No embedding data in response")
            
            # Create response object
            embedding_response = EmbeddingResponse(
                embedding=embedding,
                text=text,
                token_count=token_count,
                processing_time=processing_time
            )
            
            # Cache the result
            self._save_to_cache(cache_key, embedding_response)
            
            logger.debug(f"Generated embedding for text: {text[:50]}... "
                        f"(tokens: {token_count}, time: {processing_time:.2f}s)")
            
            return embedding_response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
        except ValueError as e:
            logger.error(f"Response parsing failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def embed_batch(self, texts: List[str], input_type: str = "passage", 
                   max_workers: int = 4) -> List[List[float]]:
        """Generate embeddings for multiple texts (BaseEmbedder interface)."""
        responses = self.embed_batch_detailed(texts, input_type, max_workers)
        return [response.embedding for response in responses]
    
    def embed_batch_detailed(self, texts: List[str], input_type: str = "passage", 
                           max_workers: int = 4) -> List[EmbeddingResponse]:
        """Generate embeddings for multiple texts using parallel processing.
        
        Args:
            texts: List of texts to embed
            input_type: Type of input ("passage" or "query")
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of EmbeddingResponse objects
        """
        logger.info(f"Starting batch embedding for {len(texts)} texts")
        
        results = []
        failed_texts = []
        
        # Process in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_text = {
                executor.submit(self.embed_text_detailed, text, input_type): (i, text)
                for i, text in enumerate(texts)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_text):
                index, text = future_to_text[future]
                
                try:
                    result = future.result()
                    results.append((index, result))
                except Exception as e:
                    logger.error(f"Failed to embed text {index}: {e}")
                    failed_texts.append((index, text, str(e)))
        
        # Sort results by original index
        results.sort(key=lambda x: x[0])
        embedding_responses = [result[1] for result in results]
        
        logger.info(f"Batch embedding completed: {len(embedding_responses)} successful, "
                  f"{len(failed_texts)} failed")
        
        if failed_texts:
            logger.warning(f"Failed texts: {[f'Index {i}: {text[:50]}...' for i, text, _ in failed_texts]}")
        
        return embedding_responses
    
    def embed_documents(self, documents: List[Dict[str, Any]], 
                       text_field: str = "content") -> List[Dict[str, Any]]:
        """Embed documents and add embeddings to document objects.
        
        Args:
            documents: List of document dictionaries
            text_field: Field name containing text to embed
            
        Returns:
            List of documents with added embeddings
        """
        logger.info(f"Embedding {len(documents)} documents")
        
        # Extract texts
        texts = []
        for doc in documents:
            if text_field in doc:
                texts.append(doc[text_field])
            else:
                logger.warning(f"Document missing {text_field} field: {doc.get('id', 'unknown')}")
                texts.append("")  # Empty text for missing content
        
        # Generate embeddings
        embeddings = self.embed_batch_detailed(texts, input_type="passage")
        
        # Add embeddings to documents
        embedded_documents = []
        for doc, embedding_response in zip(documents, embeddings):
            embedded_doc = doc.copy()
            embedded_doc.update({
                "embedding": embedding_response.embedding,
                "embedding_metadata": {
                    "model": self.model,
                    "token_count": embedding_response.token_count,
                    "processing_time": embedding_response.processing_time,
                    "embedding_dimension": len(embedding_response.embedding)
                }
            })
            embedded_documents.append(embedded_doc)
        
        return embedded_documents
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.config.get("dimension", 1024)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache_enabled:
            return {"cache_enabled": False}
        
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "cache_enabled": True,
            "cache_directory": str(self.cache_dir),
            "cached_embeddings": len(cache_files),
            "total_cache_size_mb": total_size / (1024 * 1024),
        }
    
    def clear_cache(self):
        """Clear embedding cache."""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Embedding cache cleared")
    
    def test_connection(self) -> bool:
        """Test connection to NVIDIA API."""
        try:
            test_response = self.embed_text_detailed("This is a test.", input_type="query")
            if test_response and test_response.embedding:
                logger.info("NVIDIA API connection test successful")
                return True
        except Exception as e:
            logger.error(f"NVIDIA API connection test failed: {e}")
        
        return False

class HuggingFaceEmbedder(BaseEmbedder):
    """HuggingFace Sentence Transformers embedder."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.config = get_embedding_config(model_name)
            logger.info(f"Initialized HuggingFace embedder with model: {model_name}")
        except ImportError:
            raise ImportError("sentence-transformers is required for HuggingFace embedder")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]
    
    def embed_documents(self, documents: List[Dict[str, Any]], 
                       text_field: str = "content") -> List[Dict[str, Any]]:
        """Embed documents and add embeddings to document objects.
        
        Args:
            documents: List of document dictionaries
            text_field: Field name containing text to embed
            
        Returns:
            List of documents with added embeddings
        """
        logger.info(f"Embedding {len(documents)} documents with HuggingFace model")
        
        # Extract texts
        texts = []
        for doc in documents:
            if text_field in doc:
                texts.append(doc[text_field])
            else:
                logger.warning(f"Document missing {text_field} field: {doc.get('id', 'unknown')}")
                texts.append("")  # Empty text for missing content
        
        # Process in batches for better progress tracking
        batch_size = 100  # Process 100 documents at a time
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        logger.info(f"Processing {len(texts)} texts in {total_batches} batches of {batch_size}")
        
        start_time = time.time()
        
        for batch_idx in range(0, len(texts), batch_size):
            batch_texts = texts[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)...")
            
            try:
                batch_embeddings = self.embed_batch(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                # Progress update
                processed = min(batch_idx + batch_size, len(texts))
                percent = (processed / len(texts)) * 100
                elapsed = time.time() - start_time
                
                if processed < len(texts):
                    eta = (elapsed / processed) * (len(texts) - processed)
                    logger.info(f"Progress: {processed}/{len(texts)} ({percent:.1f}%) - ETA: {eta:.0f}s")
                else:
                    logger.info(f"Progress: {processed}/{len(texts)} (100%) - Completed in {elapsed:.1f}s")
                    
            except Exception as e:
                logger.error(f"Failed to process batch {batch_num}: {e}")
                # Add empty embeddings for failed batch
                empty_embedding = [0.0] * self.get_dimension()
                all_embeddings.extend([empty_embedding] * len(batch_texts))
        
        processing_time = time.time() - start_time
        
        # Add embeddings to documents
        embedded_documents = []
        for doc, embedding in zip(documents, all_embeddings):
            embedded_doc = doc.copy()
            embedded_doc.update({
                "embedding": embedding,
                "embedding_metadata": {
                    "model": self.model_name,
                    "token_count": len(doc.get(text_field, "").split()),  # Approximate token count
                    "processing_time": processing_time / len(documents),  # Average time per document
                    "embedding_dimension": len(embedding)
                }
            })
            embedded_documents.append(embedded_doc)
        
        logger.info(f"Completed embedding {len(embedded_documents)} documents in {processing_time:.1f}s")
        return embedded_documents
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

class EmbedderFactory:
    """Factory for creating embedding models."""
    
    @staticmethod
    def create_embedder(provider: str, model_name: str = None, **kwargs) -> BaseEmbedder:
        """Create an embedder instance."""
        if provider == "nvidia":
            return NVIDIAEmbedder(model=model_name, **kwargs)
        elif provider == "huggingface":
            return HuggingFaceEmbedder(model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2")
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    
    @staticmethod
    def create_default_embedder() -> BaseEmbedder:
        """Create the default embedder from settings."""
        config = get_embedding_config()
        provider = config.get("provider", "nvidia")
        model_name = config.get("model_name")
        
        return EmbedderFactory.create_embedder(provider, model_name)

class EmbeddingPipeline:
    """Pipeline for processing documents and generating embeddings."""
    
    def __init__(self, embedder: BaseEmbedder):
        """Initialize the embedding pipeline.
        
        Args:
            embedder: Any embedder that implements BaseEmbedder interface
        """
        self.embedder = embedder
        self.logger = get_logger(__name__)
    
    def process_document_chunks(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process document chunks and generate embeddings.
        
        Args:
            documents: List of processed documents with chunks
            
        Returns:
            List of embedded chunks
        """
        self.logger.info(f"Processing {len(documents)} documents for embedding")
        
        all_chunks = []
        
        for doc in documents:
            doc_id = doc.get("document_id", "unknown")
            chunks = doc.get("chunks", [])
            
            if not chunks:
                self.logger.warning(f"No chunks found for document: {doc_id}")
                continue
            
            # Prepare chunks for embedding
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    "chunk_id": f"{doc_id}_chunk_{i}",
                    "document_id": doc_id,
                    "chunk_index": i,
                    "content": chunk.get("content", ""),
                    "metadata": {
                        **doc.get("metadata", {}),
                        **chunk.get("metadata", {}),
                        "chunk_start": chunk.get("start", 0),
                        "chunk_end": chunk.get("end", 0),
                    }
                }
                all_chunks.append(chunk_data)
        
        self.logger.info(f"Total chunks to embed: {len(all_chunks)}")
        
        # Generate embeddings for all chunks
        embedded_chunks = self.embedder.embed_documents(all_chunks, text_field="content")
        
        return embedded_chunks
    
    def save_embeddings(self, embedded_chunks: List[Dict[str, Any]], 
                       output_path: str) -> str:
        """Save embedded chunks to file.
        
        Args:
            embedded_chunks: List of chunks with embeddings
            output_path: Path to save embeddings
            
        Returns:
            Path to saved file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON (embeddings as lists)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(embedded_chunks, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(embedded_chunks)} embedded chunks to: {output_file}")
        
        return str(output_file)
    
    def get_embedding_stats(self, embedded_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about generated embeddings.
        
        Args:
            embedded_chunks: List of embedded chunks
            
        Returns:
            Statistics dictionary
        """
        if not embedded_chunks:
            return {"error": "No embedded chunks provided"}
        
        total_chunks = len(embedded_chunks)
        total_tokens = sum(chunk.get("embedding_metadata", {}).get("token_count", 0) 
                          for chunk in embedded_chunks)
        total_time = sum(chunk.get("embedding_metadata", {}).get("processing_time", 0) 
                        for chunk in embedded_chunks)
        
        # Get embedding dimensions
        embedding_dims = []
        for chunk in embedded_chunks:
            if "embedding" in chunk:
                embedding_dims.append(len(chunk["embedding"]))
        
        # Get unique documents and companies
        unique_docs = set(chunk.get("document_id", "") for chunk in embedded_chunks)
        unique_companies = set(chunk.get("metadata", {}).get("company_symbol", "") 
                             for chunk in embedded_chunks)
        
        return {
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "total_processing_time": total_time,
            "avg_processing_time": total_time / total_chunks if total_chunks > 0 else 0,
            "avg_tokens_per_chunk": total_tokens / total_chunks if total_chunks > 0 else 0,
            "embedding_dimension": embedding_dims[0] if embedding_dims else 0,
            "unique_documents": len(unique_docs),
            "unique_companies": len(unique_companies),
            "companies": sorted(list(unique_companies)),
        }
