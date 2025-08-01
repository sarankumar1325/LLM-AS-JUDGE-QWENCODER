"""
Groq client for Qwen model integration.
"""

import os
import time
from typing import Dict, Any, Optional
from groq import Groq
from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class GroqClient:
    """Groq API client for Qwen models."""
    
    def __init__(self, 
                 api_key: str = None,
                 model_name: str = "qwen/qwen3-32b",
                 max_tokens: int = 4096,
                 temperature: float = 0.6,
                 top_p: float = 0.95):
        """Initialize Groq client.
        
        Args:
            api_key: Groq API key
            model_name: Model name to use
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            top_p: Top-p for generation
        """
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        if not self.api_key:
            raise ValueError("Groq API key not found. Please set GROQ_API_KEY environment variable.")
        
        # Initialize Groq client
        self.client = Groq(api_key=self.api_key)
        
        logger.info(f"Initialized Groq client with model: {self.model_name}")
    
    def generate_response(self, 
                         prompt: str,
                         system_instruction: str = None,
                         max_retries: int = 3) -> Dict[str, Any]:
        """Generate response using Groq/Qwen.
        
        Args:
            prompt: Input prompt
            system_instruction: System instruction (optional)
            max_retries: Maximum number of retries
            
        Returns:
            Dict containing response and metadata
        """
        messages = []
        
        # Add system message if provided
        if system_instruction:
            messages.append({
                "role": "system",
                "content": system_instruction
            })
        
        # Add user message
        messages.append({
            "role": "user", 
            "content": prompt
        })
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                # Create completion
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                    top_p=self.top_p,
                    reasoning_effort="default",
                    stream=False,  # Use non-streaming for simplicity
                    stop=None,
                )
                
                processing_time = time.time() - start_time
                
                # Extract response
                response_text = completion.choices[0].message.content
                
                result = {
                    'response': response_text,
                    'processing_time': processing_time,
                    'model': self.model_name,
                    'usage': {
                        'prompt_tokens': completion.usage.prompt_tokens if completion.usage else 0,
                        'completion_tokens': completion.usage.completion_tokens if completion.usage else 0,
                        'total_tokens': completion.usage.total_tokens if completion.usage else 0
                    }
                }
                
                logger.info(f"Groq response generated in {processing_time:.2f}s")
                return result
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed: {e}")
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def test_connection(self) -> bool:
        """Test the Groq API connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Testing Groq API connection...")
            
            response = self.generate_response(
                prompt="Say 'Hello' in exactly one word.",
                system_instruction="You are a helpful assistant."
            )
            
            if response and response.get('response'):
                logger.info("✓ Groq API connection test successful")
                return True
            else:
                logger.error("✗ Groq API connection test failed - no response")
                return False
                
        except Exception as e:
            logger.error(f"✗ Groq API connection test failed: {e}")
            return False

class QwenRAGModel:
    """RAG model using Qwen through Groq."""
    
    def __init__(self, vector_store, embedder, groq_client: GroqClient = None):
        """Initialize RAG model.
        
        Args:
            vector_store: Vector store for document retrieval
            embedder: Embedding model
            groq_client: Groq client instance
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.groq_client = groq_client or GroqClient()
        
        logger.info("Initialized Qwen RAG model")
    
    def generate_response(self, 
                         query: str,
                         query_embedding: list = None,
                         filter_dict: Dict = None,
                         k: int = 5) -> Dict[str, Any]:
        """Generate RAG response using Qwen.
        
        Args:
            query: User query
            query_embedding: Pre-computed query embedding
            filter_dict: Filter for document retrieval
            k: Number of context chunks to retrieve
            
        Returns:
            Dict containing response and context chunks
        """
        try:
            # Generate query embedding if not provided
            if query_embedding is None:
                query_embedding = self.embedder.embed_text(query)
            
            # Retrieve relevant documents
            results = self.vector_store.similarity_search_with_scores(
                query_embedding=query_embedding,
                k=k,
                filter_dict=filter_dict
            )
            
            # Format context
            context_chunks = []
            context_text = ""
            
            for i, (doc, score) in enumerate(results):
                chunk = {
                    'content': doc['content'],
                    'score': score,
                    'metadata': doc['metadata']
                }
                context_chunks.append(chunk)
                context_text += f"Document {i+1}:\n{doc['content']}\n\n"
            
            # Create RAG prompt
            rag_prompt = f"""Based on the following financial documents, please answer the user's question. Use the provided context to give a comprehensive and accurate response.

Context Documents:
{context_text}

User Question: {query}

Please provide a detailed answer based on the context provided. If the context doesn't contain enough information to fully answer the question, please state what information is available and what might be missing."""
            
            # Generate response
            response = self.groq_client.generate_response(
                prompt=rag_prompt,
                system_instruction="You are a financial analyst with expertise in interpreting financial documents and reports. Provide accurate, detailed responses based on the provided context."
            )
            
            return {
                'response': response['response'],
                'context_chunks': context_chunks,
                'processing_time': response['processing_time'],
                'model': response['model'],
                'usage': response['usage']
            }
            
        except Exception as e:
            logger.error(f"RAG response generation failed: {e}")
            return None

class QwenNonRAGModel:
    """Non-RAG model using Qwen through Groq."""
    
    def __init__(self, groq_client: GroqClient = None):
        """Initialize Non-RAG model.
        
        Args:
            groq_client: Groq client instance
        """
        self.groq_client = groq_client or GroqClient()
        logger.info("Initialized Qwen Non-RAG model")
    
    def generate_response(self, 
                         query: str,
                         system_instruction: str = None) -> Dict[str, Any]:
        """Generate non-RAG response using Qwen.
        
        Args:
            query: User query
            system_instruction: Optional system instruction
            
        Returns:
            Dict containing response and metadata
        """
        try:
            default_instruction = "You are a knowledgeable financial assistant. Answer questions about finance, business, and markets using your training data knowledge."
            
            response = self.groq_client.generate_response(
                prompt=query,
                system_instruction=system_instruction or default_instruction
            )
            
            return {
                'response': response['response'],
                'processing_time': response['processing_time'],
                'model': response['model'],
                'usage': response['usage']
            }
            
        except Exception as e:
            logger.error(f"Non-RAG response generation failed: {e}")
            return None
