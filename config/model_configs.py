"""
Model configurations for different components of the RAG evaluation system.
"""

from typing import Dict, Any, List
from config.settings import settings

# Embedding Model Configurations
EMBEDDING_CONFIGS = {
    "nvidia/nv-embedqa-e5-v5": {
        "model_name": "nvidia/nv-embedqa-e5-v5",
        "dimension": 1024,
        "max_seq_length": 512,
        "provider": "nvidia",
        "batch_size": 32,
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "max_seq_length": 256,
        "provider": "huggingface",
        "batch_size": 64,
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "dimension": 768,
        "max_seq_length": 384,
        "provider": "huggingface",
        "batch_size": 32,
    },
    "sentence-transformers/all-distilroberta-v1": {
        "model_name": "sentence-transformers/all-distilroberta-v1",
        "dimension": 768,
        "max_seq_length": 512,
        "provider": "huggingface",
        "batch_size": 32,
    },
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dimension": 384,
        "max_seq_length": 128,
        "provider": "huggingface",
        "batch_size": 64,
    },
}

# LLM Model Configurations
LLM_CONFIGS = {
    "qwen/qwen3-32b": {
        "model_name": "qwen/qwen3-32b",
        "provider": "groq",
        "max_tokens": 4096,
        "temperature": 0.6,
        "top_p": 0.95,
        "context_window": 32768,  # Qwen3-32B context window
        "reasoning_effort": "default",
    },
    "gemini-1.5-flash": {
        "model_name": "gemini-1.5-flash",
        "provider": "google",
        "max_tokens": 8192,
        "temperature": 0.1,
        "top_p": 0.95,
        "context_window": 1048576,  # 1M tokens
    },
    "gemini-1.5-flash-002": {
        "model_name": "gemini-1.5-flash-002",
        "provider": "google",
        "max_tokens": 8192,
        "temperature": 0.1,
        "top_p": 0.95,
        "context_window": 1048576,  # 1M tokens
    },
    "gemini-2.0-flash-exp": {
        "model_name": "gemini-2.0-flash-exp",
        "provider": "google",
        "max_tokens": 8192,
        "temperature": 0.1,
        "top_p": 0.95,
        "context_window": 1048576,  # 1M tokens
    },
    "gemini-2.0-flash": {
        "model_name": "gemini-2.0-flash",
        "provider": "google",
        "max_tokens": 8192,
        "temperature": 0.1,
        "top_p": 0.95,
        "context_window": 1048576,  # 1M tokens
    },
    "gemini-2.5-pro": {
        "model_name": "gemini-2.5-pro",
        "provider": "google",
        "max_tokens": 4096,
        "temperature": 0.1,
        "top_p": 0.95,
        "context_window": 1000000,
    },
    "gemini-1.5-pro": {
        "model_name": "gemini-1.5-pro",
        "provider": "google",
        "max_tokens": 4096,
        "temperature": 0.1,
        "top_p": 0.95,
        "context_window": 1000000,
    },
    "claude-3-opus": {
        "model_name": "claude-3-opus-20240229",
        "provider": "anthropic",
        "max_tokens": 4096,
        "temperature": 0.1,
        "top_p": 0.95,
        "context_window": 200000,
    },
}

# Vector Store Configurations
VECTOR_STORE_CONFIGS = {
    "chroma": {
        "persist_directory": settings.CHROMA_PERSIST_DIRECTORY,
        "collection_name": "rag_documents",
        "distance_function": "cosine",
        "batch_size": 100,
    },
    "faiss": {
        "index_path": settings.FAISS_INDEX_PATH,
        "index_type": "IndexFlatIP",  # Inner Product for cosine similarity
        "nlist": 100,  # for IVF index types
        "batch_size": 1000,
    },
}

# Chunking Configurations
CHUNKING_CONFIGS = {
    "recursive": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "separators": ["\n\n", "\n", " ", ""],
        "length_function": len,
    },
    "semantic": {
        "buffer_size": 1,
        "breakpoint_threshold_type": "percentile",
        "breakpoint_threshold_amount": 95,
    },
    "fixed": {
        "chunk_size": 512,
        "chunk_overlap": 50,
    },
}

# Evaluation Prompt Templates
EVALUATION_PROMPTS = {
    "relevance": """
    You are an expert evaluator. Please evaluate the relevance of the provided answer to the given question.
    
    Question: {question}
    Answer: {answer}
    Context (if applicable): {context}
    
    Rate the relevance on a scale of 1-5:
    1 - Completely irrelevant
    2 - Mostly irrelevant with minimal connection
    3 - Somewhat relevant but missing key information
    4 - Mostly relevant with minor gaps
    5 - Completely relevant and comprehensive
    
    Provide your rating and a brief explanation.
    """,
    
    "accuracy": """
    You are an expert fact-checker. Please evaluate the factual accuracy of the provided answer.
    
    Question: {question}
    Answer: {answer}
    Context (if applicable): {context}
    
    Rate the accuracy on a scale of 1-5:
    1 - Mostly inaccurate information
    2 - Some accurate information but significant errors
    3 - Generally accurate with minor errors
    4 - Mostly accurate with very minor issues
    5 - Completely accurate
    
    Provide your rating and explanation of any inaccuracies found.
    """,
    
    "completeness": """
    You are an expert evaluator. Please evaluate how complete the provided answer is.
    
    Question: {question}
    Answer: {answer}
    Context (if applicable): {context}
    
    Rate the completeness on a scale of 1-5:
    1 - Very incomplete, missing most important information
    2 - Incomplete, missing significant information
    3 - Somewhat complete but missing some key details
    4 - Mostly complete with minor omissions
    5 - Complete and comprehensive
    
    Provide your rating and identify any missing important information.
    """,
    
    "helpfulness": """
    You are an expert evaluator. Please evaluate how helpful the provided answer is to someone asking the question.
    
    Question: {question}
    Answer: {answer}
    Context (if applicable): {context}
    
    Rate the helpfulness on a scale of 1-5:
    1 - Not helpful at all
    2 - Minimally helpful
    3 - Somewhat helpful
    4 - Very helpful
    5 - Extremely helpful and actionable
    
    Provide your rating and explain what makes it helpful or unhelpful.
    """,
}

def get_embedding_config(model_name: str = None) -> Dict[str, Any]:
    """Get embedding model configuration."""
    if model_name is None:
        model_name = settings.EMBEDDING_MODEL
    
    return EMBEDDING_CONFIGS.get(model_name, EMBEDDING_CONFIGS[settings.EMBEDDING_MODEL])

def get_llm_config(model_name: str = None) -> Dict[str, Any]:
    """Get LLM configuration."""
    if model_name is None:
        model_name = settings.DEFAULT_LLM_MODEL
    
    return LLM_CONFIGS.get(model_name, LLM_CONFIGS[settings.DEFAULT_LLM_MODEL])

def get_vector_store_config(store_type: str = None) -> Dict[str, Any]:
    """Get vector store configuration."""
    if store_type is None:
        store_type = settings.VECTOR_STORE_TYPE
    
    return VECTOR_STORE_CONFIGS.get(store_type, VECTOR_STORE_CONFIGS[settings.VECTOR_STORE_TYPE])
