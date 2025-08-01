#!/usr/bin/env python3
"""Debug script to test RAG retrieval step by step."""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.embeddings.nvidia_embedder import EmbedderFactory
from src.vector_stores.chroma_store import ChromaVectorStore
from src.models.gemini_client import GeminiClient, RAGModel

def main():
    print("Debug: Testing RAG retrieval step by step")
    
    # Initialize components
    print("\n1. Initializing embedder...")
    embedder = EmbedderFactory.create_embedder("huggingface")
    print(f"Embedder initialized: {embedder.model_name}, dimensions: {embedder.get_dimension()}")
    
    print("\n2. Initializing vector store...")
    vector_store = ChromaVectorStore()
    stats = vector_store.get_collection_stats()
    print(f"Vector store stats: {stats}")
    
    print("\n3. Initializing RAG model...")
    gemini_client = GeminiClient()
    rag_model = RAGModel(
        gemini_client=gemini_client,
        vector_store=vector_store,
        max_context_chunks=5,
        max_context_tokens=2000
    )
    print("RAG model initialized")
    
    # Test query
    test_query = "What was Apple's revenue growth in Q4 2019?"
    print(f"\n4. Testing query: '{test_query}'")
    
    # Generate embedding
    print("\n5. Generating query embedding...")
    try:
        query_embedding = embedder.embed_text(test_query)
        print(f"Embedding generated: {len(query_embedding)} dimensions")
        print(f"Embedding preview: {query_embedding[:5]}")
    except Exception as e:
        print(f"ERROR generating embedding: {e}")
        return
    
    # Test similarity search directly
    print("\n6. Testing direct similarity search...")
    try:
        results = vector_store.similarity_search(
            query_embedding=query_embedding,
            k=5
        )
        print(f"Direct search results: {len(results)} documents found")
        for i, doc in enumerate(results):
            print(f"  Doc {i+1}: {doc.get('metadata', {}).get('company_symbol', 'N/A')} - {doc.get('content', '')[:100]}...")
    except Exception as e:
        print(f"ERROR in similarity search: {e}")
        return
    
    # Test RAG retrieve_context
    print("\n7. Testing RAG retrieve_context...")
    try:
        context_docs = rag_model.retrieve_context(
            query=test_query,
            query_embedding=query_embedding
        )
        print(f"RAG retrieve_context results: {len(context_docs)} documents found")
        for i, doc in enumerate(context_docs):
            print(f"  Doc {i+1}: {doc.get('metadata', {}).get('company_symbol', 'N/A')} - {doc.get('content', '')[:100]}...")
    except Exception as e:
        print(f"ERROR in RAG retrieve_context: {e}")
        return
    
    # Test with company filter
    print("\n8. Testing with company filter...")
    company_filter = {"company_symbol": "AAPL"}  # Fixed: use direct equality
    try:
        context_docs_filtered = rag_model.retrieve_context(
            query=test_query,
            query_embedding=query_embedding,
            filter_dict=company_filter
        )
        print(f"Filtered retrieve_context results: {len(context_docs_filtered)} documents found")
        for i, doc in enumerate(context_docs_filtered):
            print(f"  Doc {i+1}: {doc.get('metadata', {}).get('company_symbol', 'N/A')} - {doc.get('content', '')[:100]}...")
    except Exception as e:
        print(f"ERROR in filtered retrieve_context: {e}")
        return
    
    print("\nDebug complete!")

if __name__ == "__main__":
    main()
