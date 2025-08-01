#!/usr/bin/env python3
"""
Test script to verify Groq/Qwen API connection.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_groq_connection():
    """Test Groq/Qwen API connection."""
    print("🧪 Testing Groq/Qwen API Connection...")
    
    try:
        from src.models.groq_client import GroqClient
        
        # Initialize client
        print("🔧 Initializing Groq client...")
        groq_client = GroqClient()
        print(f"✅ Client initialized with model: {groq_client.model_name}")
        print(f"✅ API key configured: {groq_client.api_key[:20]}...")
        
        # Test connection
        print("🌐 Testing API connection...")
        connection_result = groq_client.test_connection()
        
        if connection_result:
            print("✅ Connection test successful!")
        else:
            print("❌ Connection test failed!")
            return False
        
        # Test a simple generation
        print("🤖 Testing response generation...")
        test_prompt = "What is artificial intelligence? (Keep response under 50 words)"
        response = groq_client.generate_response(
            prompt=test_prompt,
            system_instruction="You are a helpful AI assistant. Be concise."
        )
        
        if response and response.get('response'):
            print("✅ Response generation successful!")
            print(f"📝 Generated response: {response['response'][:100]}...")
            print(f"⏱️ Processing time: {response.get('processing_time', 0):.2f}s")
            print(f"📊 Tokens used: {response.get('usage', {}).get('total_tokens', 0)}")
        else:
            print("❌ Response generation failed!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Groq API test failed: {e}")
        return False

def test_qwen_models():
    """Test Qwen RAG and Non-RAG models."""
    print("\n🧪 Testing Qwen Models...")
    
    try:
        from src.models.groq_client import GroqClient, QwenRAGModel, QwenNonRAGModel
        from src.vector_stores.chroma_store import ChromaVectorStore
        from src.embeddings.nvidia_embedder import EmbedderFactory
        
        # Initialize components
        print("🔧 Initializing components...")
        groq_client = GroqClient()
        vector_store = ChromaVectorStore()
        embedder = EmbedderFactory.create_default_embedder()
        
        # Test Non-RAG model
        print("🤖 Testing Qwen Non-RAG model...")
        non_rag_model = QwenNonRAGModel(groq_client=groq_client)
        non_rag_response = non_rag_model.generate_response("What is Apple's revenue growth?")
        
        if non_rag_response and non_rag_response.get('response'):
            print("✅ Non-RAG model working!")
            print(f"📝 Response: {non_rag_response['response'][:100]}...")
        else:
            print("❌ Non-RAG model failed!")
            return False
        
        # Test RAG model
        print("🔍 Testing Qwen RAG model...")
        rag_model = QwenRAGModel(
            vector_store=vector_store, 
            embedder=embedder,
            groq_client=groq_client
        )
        
        # Generate query embedding
        query = "What was Apple's revenue growth in Q4 2019?"
        query_embedding = embedder.embed_text(query)
        
        # Test RAG response
        rag_response = rag_model.generate_response(
            query=query,
            query_embedding=query_embedding,
            filter_dict={"company_symbol": "AAPL"}
        )
        
        if rag_response and rag_response.get('response'):
            print("✅ RAG model working!")
            print(f"📝 Response: {rag_response['response'][:100]}...")
            print(f"📊 Context chunks used: {len(rag_response.get('context_chunks', []))}")
        else:
            print("❌ RAG model failed!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Qwen models test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 70)
    print("🚀 Groq/Qwen API Connection Test")
    print("=" * 70)
    
    success = True
    
    success &= test_groq_connection()
    success &= test_qwen_models()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ ALL TESTS PASSED!")
        print("🎉 Qwen through Groq is working perfectly!")
        print("🌐 Ready to update Streamlit app!")
    else:
        print("❌ Some tests failed. Please check the API key and configuration.")
    print("=" * 70)
    
    return success

if __name__ == "__main__":
    main()
