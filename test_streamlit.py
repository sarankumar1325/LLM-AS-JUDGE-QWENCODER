#!/usr/bin/env python3
"""
Test script to verify Streamlit app components can be imported.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_imports():
    """Test all required imports for the Streamlit app."""
    print("🧪 Testing Streamlit App Component Imports...")
    
    # Test core Streamlit imports
    try:
        import streamlit as st
        import pandas as pd
        import plotly.graph_objects as go
        import plotly.express as px
        print("✅ Streamlit and visualization libraries imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import visualization libraries: {e}")
        return False
    
    # Test RAG system component imports
    try:
        from src.models.gemini_client import GeminiClient, RAGModel, NonRAGModel
        print("✅ Gemini client models imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Gemini models: {e}")
        return False
    
    try:
        from src.vector_stores.chroma_store import ChromaVectorStore
        print("✅ ChromaDB vector store imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ChromaDB store: {e}")
        return False
    
    try:
        from src.embeddings.nvidia_embedder import EmbedderFactory
        print("✅ Embedder factory imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import embedder: {e}")
        return False
    
    try:
        from src.utils.logger import get_logger
        print("✅ Logger utility imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import logger: {e}")
        return False
    
    print("🎉 All imports successful! Streamlit app is ready to run.")
    return True

def test_components():
    """Test basic component initialization."""
    print("\n🔧 Testing Component Initialization...")
    
    try:
        # Test vector store
        from src.vector_stores.chroma_store import ChromaVectorStore
        vector_store = ChromaVectorStore()
        stats = vector_store.get_collection_stats()
        print(f"✅ ChromaDB initialized: {stats['total_documents']} documents")
    except Exception as e:
        print(f"❌ ChromaDB initialization failed: {e}")
        return False
    
    try:
        # Test embedder
        from src.embeddings.nvidia_embedder import EmbedderFactory
        embedder = EmbedderFactory.create_default_embedder()
        print(f"✅ Embedder initialized: {embedder.model_name}")
    except Exception as e:
        print(f"❌ Embedder initialization failed: {e}")
        return False
    
    try:
        # Test Gemini client (without API call)
        from src.models.gemini_client import GeminiClient
        gemini_client = GeminiClient()
        print("✅ Gemini client initialized")
    except Exception as e:
        print(f"❌ Gemini client initialization failed: {e}")
        return False
    
    print("🎉 All components initialized successfully!")
    return True

def main():
    """Main test function."""
    print("=" * 60)
    print("🚀 RAG Evaluation Streamlit App - Component Test")
    print("=" * 60)
    
    if not test_imports():
        print("\n❌ Import tests failed. Please check dependencies.")
        return False
    
    if not test_components():
        print("\n❌ Component tests failed. Please check configuration.")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("🌐 Ready to launch Streamlit app with: python run_streamlit.py")
    print("=" * 60)
    return True

if __name__ == "__main__":
    main()
