#!/usr/bin/env python3
"""
Quick test of the Qwen Streamlit integration.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_streamlit_qwen_integration():
    """Test that all components load correctly for Streamlit."""
    
    print("🧪 Testing Qwen Streamlit Integration...")
    
    # Test imports
    try:
        from src.models.groq_client import GroqClient, QwenRAGModel, QwenNonRAGModel
        from src.vector_stores.chroma_store import ChromaVectorStore
        from src.embeddings.nvidia_embedder import EmbedderFactory
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test component initialization
    try:
        print("🔄 Initializing components...")
        
        # Initialize Groq client
        groq_client = GroqClient()
        print("✅ Groq client initialized")
        
        # Test connection
        if groq_client.test_connection():
            print("✅ Groq API connection successful")
        else:
            print("❌ Groq API connection failed")
            return False
        
        # Initialize vector store
        vector_store = ChromaVectorStore()
        print("✅ Vector store initialized")
        
        # Initialize embedder
        embedder = EmbedderFactory.create_default_embedder()
        print("✅ Embedder initialized")
        
        # Initialize models
        rag_model = QwenRAGModel(
            vector_store=vector_store,
            embedder=embedder,
            groq_client=groq_client
        )
        print("✅ Qwen RAG model initialized")
        
        non_rag_model = QwenNonRAGModel(groq_client=groq_client)
        print("✅ Qwen Non-RAG model initialized")
        
        print("🎉 All components ready for Streamlit!")
        return True
        
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        return False

if __name__ == "__main__":
    success = test_streamlit_qwen_integration()
    if success:
        print("\n🚀 Streamlit app ready to run with Qwen!")
        print("📱 Run: streamlit run streamlit_app.py")
    else:
        print("\n💥 Integration test failed. Check the errors above.")
