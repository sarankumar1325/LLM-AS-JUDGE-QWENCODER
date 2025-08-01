"""
Non-RAG model implementation for baseline comparison.
"""

from typing import Dict, Any, Optional
from src.models.llm_interface import BaseLLM, LLMFactory

class NonRAGModel:
    """Standard LLM without retrieval augmentation."""
    
    def __init__(self,
                 llm: Optional[BaseLLM] = None,
                 prompt_template: str = None):
        
        self.llm = llm or LLMFactory.create_default_llm()
        
        self.prompt_template = prompt_template or """
Question: {question}

Please answer the question based on your knowledge. Be as accurate and helpful as possible.

Answer:"""
    
    def generate_answer(self, 
                       question: str,
                       **kwargs) -> Dict[str, Any]:
        """Generate answer without retrieval."""
        
        # Format prompt
        prompt = self.prompt_template.format(question=question)
        
        # Generate response
        answer = self.llm.generate(prompt, **kwargs)
        
        return {
            "question": question,
            "answer": answer,
            "context_docs": [],
            "context_scores": [],
            "context_text": "",
            "prompt": prompt,
            "model_type": "non_rag"
        }
