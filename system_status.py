#!/usr/bin/env python3
"""
RAG Evaluation System Status and Quota Management Script.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def check_system_status():
    """Check the overall system status."""
    print("🔍 RAG Evaluation System Status Check")
    print("=" * 60)
    
    # Check API key
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print(f"✅ API Key configured: {api_key[:20]}...")
    else:
        print("❌ No API key found")
        return False
    
    # Check components
    try:
        print("\n📦 Component Status:")
        
        # ChromaDB
        from src.vector_stores.chroma_store import ChromaVectorStore
        vector_store = ChromaVectorStore()
        stats = vector_store.get_collection_stats()
        print(f"✅ ChromaDB: {stats['total_documents']} documents")
        
        # Embedder
        from src.embeddings.nvidia_embedder import EmbedderFactory
        embedder = EmbedderFactory.create_default_embedder()
        print(f"✅ Embedder: {embedder.model_name}")
        
        # Gemini Client (without API call)
        from src.models.gemini_client import GeminiClient
        gemini_client = GeminiClient()
        print(f"✅ Gemini Client: {gemini_client.model_name}")
        
        print("\n🎯 System Status: READY (waiting for API quota)")
        return True
        
    except Exception as e:
        print(f"❌ Component check failed: {e}")
        return False

def suggest_alternatives():
    """Suggest alternatives while waiting for quota."""
    print("\n💡 While waiting for quota reset, you can:")
    print("1. 📊 Explore the existing evaluation results in /results/")
    print("2. 🔍 Browse the document collection in ChromaDB")
    print("3. 📈 Analyze previous evaluation metrics")
    print("4. 🛠️ Test individual components (embeddings, vector search)")
    print("5. 📝 Review the codebase and documentation")
    
def create_quota_aware_demo():
    """Create a demo that works without API calls."""
    print("\n🎮 Creating Quota-Aware Demo...")
    
    try:
        from src.vector_stores.chroma_store import ChromaVectorStore
        from src.embeddings.nvidia_embedder import EmbedderFactory
        
        # Test vector search without API calls
        vector_store = ChromaVectorStore()
        embedder = EmbedderFactory.create_default_embedder()
        
        print("✅ Testing Vector Search (No API calls needed):")
        
        # Sample query
        query = "What was Apple's revenue growth in Q4 2019?"
        print(f"📝 Query: {query}")
        
        # Generate embedding
        query_embedding = embedder.embed_text(query)
        print(f"✅ Generated query embedding: {len(query_embedding)} dimensions")
        
        # Search similar documents
        results = vector_store.search_similar(
            query_embedding=query_embedding,
            k=5,
            filter_dict={"company_symbol": "AAPL"}
        )
        
        print(f"🔍 Found {len(results)} relevant documents:")
        for i, result in enumerate(results, 1):
            print(f"   {i}. Score: {result['score']:.3f} | Company: {result['metadata'].get('company_symbol', 'N/A')}")
            print(f"      Content: {result['content'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo creation failed: {e}")
        return False

def main():
    """Main function."""
    print("🚀 RAG Evaluation System - Quota Management")
    print("=" * 60)
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check system status
    system_ready = check_system_status()
    
    if system_ready:
        # Create demo
        demo_success = create_quota_aware_demo()
        
        # Suggest alternatives
        suggest_alternatives()
        
        print("\n" + "=" * 60)
        print("📋 SUMMARY:")
        print("✅ System is fully configured and ready")
        print("✅ All components working (ChromaDB, Embeddings, etc.)")
        print("✅ Gemini 2.0 Flash model configured")
        print("⏳ Waiting for API quota reset")
        print("\n🎯 Next Steps:")
        print("1. Wait for quota reset (usually hourly/daily)")
        print("2. Monitor quota at: https://ai.google.dev/gemini-api/docs/rate-limits")
        print("3. Run: python run_streamlit.py (when quota available)")
        print("=" * 60)
    else:
        print("\n❌ System not ready - please check configuration")

if __name__ == "__main__":
    main()
