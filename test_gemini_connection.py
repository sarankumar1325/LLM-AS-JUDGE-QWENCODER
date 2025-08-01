#!/usr/bin/env python3
"""
Test script to verify Gemini API connection with the new key.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_gemini_connection():
    """Test Gemini API connection thoroughly."""
    print("🧪 Testing Gemini API Connection...")
    
    try:
        from src.models.gemini_client import GeminiClient
        
        # Initialize client
        print("🔧 Initializing Gemini client...")
        gemini_client = GeminiClient()
        print(f"✅ Client initialized with model: {gemini_client.model_name}")
        print(f"✅ API key configured: {gemini_client.api_key[:20]}...")
        
        # Test connection
        print("🌐 Testing API connection...")
        connection_result = gemini_client.test_connection()
        
        if connection_result:
            print("✅ Connection test successful!")
            if isinstance(connection_result, dict):
                print(f"📊 Response: {connection_result.get('response', 'N/A')}")
                print(f"⏱️ Processing time: {connection_result.get('processing_time', 0):.2f}s")
            else:
                print("📊 Connection verified")
        else:
            print("❌ Connection test failed!")
            return False
        
        # Test a simple generation
        print("🤖 Testing response generation...")
        test_prompt = "What is artificial intelligence? (Keep response under 50 words)"
        response = gemini_client.generate_response(
            prompt=test_prompt,
            system_instruction="You are a helpful AI assistant. Be concise."
        )
        
        if response and response.get('response'):
            print("✅ Response generation successful!")
            print(f"📝 Generated response: {response['response'][:100]}...")
            print(f"⏱️ Processing time: {response.get('processing_time', 0):.2f}s")
        else:
            print("❌ Response generation failed!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Gemini API test failed: {e}")
        return False

def test_rag_models():
    """Test RAG and Non-RAG models."""
    print("\n🧪 Testing RAG Models...")
    
    try:
        from src.models.gemini_client import GeminiClient, RAGModel, NonRAGModel
        from src.vector_stores.chroma_store import ChromaVectorStore
        from src.embeddings.nvidia_embedder import EmbedderFactory
        
        # Initialize components
        print("🔧 Initializing components...")
        vector_store = ChromaVectorStore()
        embedder = EmbedderFactory.create_default_embedder()
        gemini_client = GeminiClient()  # Create gemini client for NonRAGModel
        
        # Test Non-RAG model
        print("🤖 Testing Non-RAG model...")
        non_rag_model = NonRAGModel(gemini_client=gemini_client)
        non_rag_response = non_rag_model.generate_response("What is Apple's revenue growth?")
        
        if non_rag_response and non_rag_response.get('response'):
            print("✅ Non-RAG model working!")
            print(f"📝 Response: {non_rag_response['response'][:100]}...")
        else:
            print("❌ Non-RAG model failed!")
            return False
        
        # Test RAG model
        print("🔍 Testing RAG model...")
        rag_model = RAGModel(
            gemini_client=gemini_client,
            vector_store=vector_store
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
        print(f"❌ RAG models test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 70)
    print("🚀 Gemini API Connection Test - New API Key")
    print("=" * 70)
    
    success = True
    
    success &= test_gemini_connection()
    success &= test_rag_models()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ ALL TESTS PASSED!")
        print("🎉 New Gemini API key is working perfectly!")
        print("🌐 Streamlit app is ready to use!")
    else:
        print("❌ Some tests failed. Please check the API key and configuration.")
    print("=" * 70)
    
    return success

if __name__ == "__main__":
    main()
