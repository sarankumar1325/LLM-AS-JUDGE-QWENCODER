"""
Gemini 2.5 Pro API client for LLM integration.
"""

import os
import time
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Add parent directories for imports
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
sys.path.insert(0, str(root_dir))

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Google Generative AI not available. Run: pip install google-generativeai")

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class GeminiClient:
    """Gemini 2.5 Pro API client for RAG evaluation system."""
    
    def __init__(self, 
                 api_key: str = None,
                 model_name: str = "gemini-1.5-flash",
                 temperature: float = 0.1,
                 max_tokens: int = 4096):
        """Initialize Gemini client.
        
        Args:
            api_key: Gemini API key
            model_name: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai is required. Install with: pip install google-generativeai")
        
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=0.95,
            top_k=40
        )
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config
        )
        
        logger.info(f"Initialized Gemini client with model: {model_name}")
    
    def generate_response(self, 
                         prompt: str,
                         system_instruction: str = None,
                         max_retries: int = 3) -> Dict[str, Any]:
        """Generate response from Gemini model.
        
        Args:
            prompt: User prompt
            system_instruction: Optional system instruction
            max_retries: Maximum number of retries
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                # Prepare full prompt
                if system_instruction:
                    full_prompt = f"{system_instruction}\n\nUser: {prompt}\n\nAssistant:"
                else:
                    full_prompt = prompt
                
                # Generate response
                response = self.model.generate_content(full_prompt)
                
                # Check if response was blocked
                if response.candidates[0].finish_reason.name in ["SAFETY", "RECITATION"]:
                    logger.warning(f"Response blocked: {response.candidates[0].finish_reason.name}")
                    return {
                        "response": "",
                        "error": f"Response blocked: {response.candidates[0].finish_reason.name}",
                        "processing_time": time.time() - start_time,
                        "model": self.model_name,
                        "attempt": attempt + 1
                    }
                
                # Extract response text
                response_text = response.text if response.text else ""
                
                # Calculate token usage (approximate)
                prompt_tokens = len(full_prompt.split()) * 1.3  # Rough approximation
                response_tokens = len(response_text.split()) * 1.3
                
                result = {
                    "response": response_text,
                    "processing_time": time.time() - start_time,
                    "model": self.model_name,
                    "prompt_tokens": int(prompt_tokens),
                    "response_tokens": int(response_tokens),
                    "total_tokens": int(prompt_tokens + response_tokens),
                    "attempt": attempt + 1,
                    "success": True
                }
                
                logger.debug(f"Generated response ({len(response_text)} chars) in {result['processing_time']:.2f}s")
                return result
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return {
                        "response": "",
                        "error": str(e),
                        "processing_time": time.time() - start_time,
                        "model": self.model_name,
                        "attempt": attempt + 1,
                        "success": False
                    }
                
                # Wait before retry
                time.sleep(2 ** attempt)
        
        return {
            "response": "",
            "error": "Max retries exceeded",
            "processing_time": time.time() - start_time,
            "model": self.model_name,
            "success": False
        }
    
    def test_connection(self) -> bool:
        """Test connection to Gemini API.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            test_response = self.generate_response(
                "Hello! This is a test message. Please respond with 'Test successful'."
            )
            
            if test_response.get("success") and "test successful" in test_response.get("response", "").lower():
                logger.info("✓ Gemini API connection test successful")
                return True
            else:
                logger.error(f"✗ Gemini API connection test failed: {test_response.get('error', 'Unexpected response')}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Gemini API connection test error: {e}")
            return False

class RAGModel:
    """RAG (Retrieval-Augmented Generation) model implementation."""
    
    def __init__(self, 
                 gemini_client: GeminiClient,
                 vector_store,
                 max_context_chunks: int = 5,
                 max_context_tokens: int = 2000):
        """Initialize RAG model.
        
        Args:
            gemini_client: Gemini API client
            vector_store: Vector store for retrieval
            max_context_chunks: Maximum number of chunks to retrieve
            max_context_tokens: Maximum tokens in context
        """
        self.gemini_client = gemini_client
        self.vector_store = vector_store
        self.max_context_chunks = max_context_chunks
        self.max_context_tokens = max_context_tokens
        
        logger.info(f"Initialized RAG model with max {max_context_chunks} chunks, {max_context_tokens} tokens")
    
    def retrieve_context(self, 
                        query: str,
                        query_embedding: List[float] = None,
                        filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant context for the query.
        
        Args:
            query: User query
            query_embedding: Pre-computed query embedding
            filter_dict: Optional metadata filters
            
        Returns:
            List of relevant documents
        """
        try:
            # If no embedding provided, would need to generate one
            # For now, assume embedding is provided or vector store handles text queries
            if query_embedding:
                results = self.vector_store.similarity_search(
                    query_embedding=query_embedding,
                    k=self.max_context_chunks,
                    filter_dict=filter_dict
                )
            else:
                # This would require text-to-embedding conversion
                logger.warning("Query embedding not provided - retrieval may be limited")
                results = []
            
            # Filter by token count if needed
            context_docs = []
            total_tokens = 0
            
            for doc in results:
                content = doc.get("content", "")
                doc_tokens = len(content.split()) * 1.3  # Rough approximation
                
                if total_tokens + doc_tokens <= self.max_context_tokens:
                    context_docs.append(doc)
                    total_tokens += doc_tokens
                else:
                    break
            
            logger.info(f"Retrieved {len(context_docs)} context documents ({total_tokens:.0f} tokens)")
            return context_docs
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return []
    
    def format_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string.
        
        Args:
            context_docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not context_docs:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Add document header with metadata
            header = f"[Document {i}"
            if "company_symbol" in metadata:
                header += f" - {metadata['company_symbol']}"
            if "document_id" in doc:
                header += f" - {doc['document_id']}"
            header += "]"
            
            context_parts.append(f"{header}\n{content}")
        
        return "\n\n".join(context_parts)
    
    def generate_response(self, 
                         query: str,
                         query_embedding: List[float] = None,
                         filter_dict: Dict[str, Any] = None,
                         system_instruction: str = None) -> Dict[str, Any]:
        """Generate RAG response to query.
        
        Args:
            query: User query
            query_embedding: Pre-computed query embedding
            filter_dict: Optional metadata filters
            system_instruction: Optional system instruction
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        
        # Retrieve context
        context_docs = self.retrieve_context(query, query_embedding, filter_dict)
        context_text = self.format_context(context_docs)
        
        # Create RAG prompt
        rag_prompt = f"""Context Information:
{context_text}

Query: {query}

Please answer the query based on the provided context. If the context doesn't contain relevant information, please state that clearly."""
        
        # Default system instruction for RAG
        if not system_instruction:
            system_instruction = """You are a helpful assistant that answers questions based on provided context. 
Always base your answers on the given context. If the context doesn't contain enough information to answer the question, 
say so clearly. Be accurate and cite relevant parts of the context when possible."""
        
        # Generate response
        response_data = self.gemini_client.generate_response(
            prompt=rag_prompt,
            system_instruction=system_instruction
        )
        
        # Add RAG-specific metadata
        response_data.update({
            "rag_enabled": True,
            "context_chunks": len(context_docs),
            "context_sources": [doc.get("id", "unknown") for doc in context_docs],
            "retrieval_time": time.time() - start_time - response_data.get("processing_time", 0),
            "total_rag_time": time.time() - start_time
        })
        
        logger.info(f"Generated RAG response with {len(context_docs)} context chunks in {response_data['total_rag_time']:.2f}s")
        return response_data

class NonRAGModel:
    """Non-RAG model for baseline comparison."""
    
    def __init__(self, gemini_client: GeminiClient):
        """Initialize Non-RAG model.
        
        Args:
            gemini_client: Gemini API client
        """
        self.gemini_client = gemini_client
        logger.info("Initialized Non-RAG model")
    
    def generate_response(self, 
                         query: str,
                         system_instruction: str = None) -> Dict[str, Any]:
        """Generate non-RAG response to query.
        
        Args:
            query: User query
            system_instruction: Optional system instruction
            
        Returns:
            Dictionary with response and metadata
        """
        # Default system instruction for non-RAG
        if not system_instruction:
            system_instruction = """You are a helpful assistant that answers questions based on your training knowledge. 
Provide accurate and helpful responses to the best of your ability."""
        
        # Generate response
        response_data = self.gemini_client.generate_response(
            prompt=query,
            system_instruction=system_instruction
        )
        
        # Add non-RAG specific metadata
        response_data.update({
            "rag_enabled": False,
            "context_chunks": 0,
            "context_sources": [],
            "retrieval_time": 0
        })
        
        logger.info(f"Generated Non-RAG response in {response_data.get('processing_time', 0):.2f}s")
        return response_data

def main():
    """Test Gemini integration."""
    try:
        # Test Gemini connection
        gemini_client = GeminiClient()
        
        if gemini_client.test_connection():
            print("✓ Gemini API connection successful")
            
            # Test basic response
            response = gemini_client.generate_response("What is artificial intelligence?")
            print(f"\nTest Response ({response.get('total_tokens', 0)} tokens):")
            print(response.get("response", "No response"))
            
        else:
            print("✗ Gemini API connection failed")
            
    except Exception as e:
        print(f"Error testing Gemini integration: {e}")

if __name__ == "__main__":
    main()
