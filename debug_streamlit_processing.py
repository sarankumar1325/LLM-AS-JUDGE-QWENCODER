#!/usr/bin/env python3
"""
Debug script to identify the Streamlit processing issue.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def debug_streamlit_processing():
    """Debug the specific issue causing infinite processing."""
    
    print("🔍 Debugging Streamlit processing issue...")
    
    try:
        from src.models.groq_client import GroqClient, QwenRAGModel, QwenNonRAGModel
        from src.vector_stores.chroma_store import ChromaVectorStore
        from src.embeddings.nvidia_embedder import EmbedderFactory
        
        print("✅ Imports successful")
        
        # Initialize components
        print("🔄 Initializing components...")
        groq_client = GroqClient()
        vector_store = ChromaVectorStore()
        embedder = EmbedderFactory.create_default_embedder()
        
        # Test RAG model initialization
        rag_model = QwenRAGModel(
            vector_store=vector_store,
            embedder=embedder,
            groq_client=groq_client
        )
        
        non_rag_model = QwenNonRAGModel(groq_client=groq_client)
        print("✅ Models initialized")
        
        # Test simple query
        test_query = "What was Apple's revenue in 2019?"
        print(f"🧪 Testing query: {test_query}")
        
        # Generate embedding
        print("🔄 Generating embedding...")
        query_embedding = embedder.embed_text(test_query)
        print(f"✅ Embedding generated: shape {len(query_embedding)}")
        
        # Test RAG response (this might be where it hangs)
        print("🔄 Testing RAG response generation...")
        rag_response = rag_model.generate_response(
            query=test_query,
            query_embedding=query_embedding,
            filter_dict={"company_symbol": "AAPL"}
        )
        print(f"✅ RAG response generated: {len(rag_response['response'])} chars")
        
        # Test Non-RAG response
        print("🔄 Testing Non-RAG response generation...")
        non_rag_response = non_rag_model.generate_response(test_query)
        print(f"✅ Non-RAG response generated: {len(non_rag_response['response'])} chars")
        
        # Test evaluation
        print("🔄 Testing evaluation...")
        evaluation_prompt = f"""You are an expert AI judge. Rate these responses on a scale of 1-10.

Query: {test_query}
Response A: {rag_response['response'][:200]}...
Response B: {non_rag_response['response'][:200]}...

Return only this JSON:
{{"response_a_scores": {{"overall": 8}}, "response_b_scores": {{"overall": 7}}, "winner": "A", "reasoning": "Test"}}"""
        
        eval_response = groq_client.generate_response(
            prompt=evaluation_prompt,
            system_instruction="Return only valid JSON."
        )
        print(f"✅ Evaluation completed: {eval_response['response'][:100]}...")
        
        print("🎉 All components working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error at step: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_streamlit_processing()
