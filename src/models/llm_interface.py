"""
LLM interface for various language models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from config.settings import settings
from config.model_configs import get_llm_config

class BaseLLM(ABC):
    """Base class for language models."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        pass

class GeminiLLM(BaseLLM):
    """Google Gemini LLM implementation."""
    
    def __init__(self, model_name: str = None, api_key: str = None):
        try:
            import google.generativeai as genai
            
            self.model_name = model_name or settings.DEFAULT_LLM_MODEL
            self.api_key = api_key or settings.GEMINI_API_KEY
            self.config = get_llm_config(self.model_name)
            
            if not self.api_key:
                raise ValueError("Gemini API key is required")
            
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            
        except ImportError:
            raise ImportError("google-generativeai is required for GeminiLLM")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Gemini."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": kwargs.get("temperature", self.config.get("temperature", 0.1)),
                    "max_output_tokens": kwargs.get("max_tokens", self.config.get("max_tokens", 4096)),
                    "top_p": kwargs.get("top_p", self.config.get("top_p", 0.95)),
                }
            )
            return response.text
        except Exception as e:
            raise Exception(f"Gemini generation failed: {e}")
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        return [self.generate(prompt, **kwargs) for prompt in prompts]

class LLMFactory:
    """Factory for creating LLM instances."""
    
    @staticmethod
    def create_llm(provider: str, model_name: str = None, **kwargs) -> BaseLLM:
        """Create LLM instance."""
        if provider == "google" or provider == "gemini":
            return GeminiLLM(model_name=model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def create_default_llm() -> BaseLLM:
        """Create default LLM from settings."""
        config = get_llm_config()
        provider = config.get("provider", "google")
        model_name = config.get("model_name")
        
        return LLMFactory.create_llm(provider, model_name)
