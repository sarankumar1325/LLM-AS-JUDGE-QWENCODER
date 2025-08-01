"""
RAG model implementation.
"""

from typing import List, Dict, Any, Optional
from src.models.llm_interface import BaseLLM, LLMFactory
from src.vector_stores.base_store import BaseVectorStore
from src.embeddings.nvidia_embedder import BaseEmbedder, EmbedderFactory

class RAGModel:
    """Retrieval-Augmented Generation model."""
    
    def __init__(self,
                 llm: Optional[BaseLLM] = None,
                 vector_store: Optional[BaseVectorStore] = None,
                 embedder: Optional[BaseEmbedder] = None,
                 top_k: int = 5,
                 context_template: str = None):
        
        self.llm = llm or LLMFactory.create_default_llm()
        self.vector_store = vector_store
        self.embedder = embedder or EmbedderFactory.create_default_embedder()
        self.top_k = top_k
        
        self.context_template = context_template or """
Context information:
{context}

Question: {question}

Based on the context provided above, please answer the question. If the context doesn't contain enough information to answer the question, please say so.

Answer:"""
    
    def generate_answer(self, 
                       question: str, 
                       use_context: bool = True,
                       **kwargs) -> Dict[str, Any]:
        """Generate answer using RAG approach."""
        
        context_docs = []
        context_text = ""
        
        if use_context and self.vector_store:
            # Get query embedding
            query_embedding = self.embedder.embed_text(question)
            
            # Retrieve relevant documents
            context_docs = self.vector_store.similarity_search_with_scores(
                query_embedding, 
                k=self.top_k
            )
            
            # Format context
            context_parts = []
            for doc, score in context_docs:
                context_parts.append(f"[Score: {score:.3f}] {doc['content']}")
            
            context_text = "\n\n".join(context_parts)
        
        # Format prompt
        if context_text:
            prompt = self.context_template.format(
                context=context_text,
                question=question
            )
        else:
            prompt = f"Question: {question}\n\nAnswer:"
        
        # Generate response
        answer = self.llm.generate(prompt, **kwargs)
        
        return {
            "question": question,
            "answer": answer,
            "context_docs": [doc for doc, _ in context_docs],
            "context_scores": [score for _, score in context_docs],
            "context_text": context_text,
            "prompt": prompt,
            "model_type": "rag"
        }
