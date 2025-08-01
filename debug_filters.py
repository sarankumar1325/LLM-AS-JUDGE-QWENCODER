#!/usr/bin/env python3
"""Test ChromaDB filter syntax."""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.vector_stores.chroma_store import ChromaVectorStore
from src.embeddings.nvidia_embedder import EmbedderFactory

def main():
    print("Debug: Testing ChromaDB filter syntax")
    
    vector_store = ChromaVectorStore()
    embedder = EmbedderFactory.create_embedder("huggingface")
    
    # Test query
    test_query = "What was Apple's revenue growth in Q4 2019?"
    query_embedding = embedder.embed_text(test_query)
    
    print(f"\nTesting query: '{test_query}'")
    
    # Test different filter syntaxes
    test_filters = [
        {"company_symbol": "AAPL"},  # Direct equality
        {"company_symbol": {"$eq": "AAPL"}},  # MongoDB style equality
        {"company_symbol": {"$in": ["AAPL"]}},  # MongoDB style in
    ]
    
    for i, filter_dict in enumerate(test_filters):
        print(f"\n{i+1}. Testing filter: {filter_dict}")
        try:
            results = vector_store.similarity_search(
                query_embedding=query_embedding,
                k=3,
                filter_dict=filter_dict
            )
            print(f"   Results: {len(results)} documents found")
            if results:
                print(f"   First result: {results[0].get('metadata', {}).get('company_symbol', 'N/A')}")
        except Exception as e:
            print(f"   ERROR: {e}")
    
    # Test multiple companies with OR logic
    print(f"\n4. Testing multiple companies...")
    try:
        # ChromaDB uses different syntax for OR queries
        # Let's try Chroma's way
        results = vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            where={"$or": [
                {"company_symbol": {"$eq": "AAPL"}},
                {"company_symbol": {"$eq": "MSFT"}}
            ]},
            include=["documents", "metadatas", "distances"]
        )
        
        documents = []
        for i in range(len(results['ids'][0])):
            documents.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        print(f"   OR query results: {len(documents)} documents found")
        for doc in documents:
            print(f"   Company: {doc['metadata'].get('company_symbol', 'N/A')}")
    except Exception as e:
        print(f"   OR query ERROR: {e}")

if __name__ == "__main__":
    main()
