#!/usr/bin/env python3
"""Debug metadata to see company field name."""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.vector_stores.chroma_store import ChromaVectorStore

def main():
    print("Debug: Checking document metadata")
    
    vector_store = ChromaVectorStore()
    
    # Get a few documents to check metadata
    print("\n1. Getting sample documents...")
    collection = vector_store.collection
    
    # Get first 5 documents
    results = collection.get(limit=5, include=["metadatas", "documents"])
    
    print(f"Found {len(results['ids'])} documents")
    
    for i, (doc_id, metadata, document) in enumerate(zip(results['ids'], results['metadatas'], results['documents'])):
        print(f"\nDocument {i+1}:")
        print(f"  ID: {doc_id}")
        print(f"  Metadata: {metadata}")
        print(f"  Content preview: {document[:100]}...")
    
    # Check if any documents have Apple company data
    print("\n2. Searching for Apple documents...")
    apple_results = collection.get(
        where={"company": "AAPL"},  # Try different field name
        limit=3,
        include=["metadatas", "documents"]
    )
    
    print(f"Found {len(apple_results['ids'])} Apple documents with 'company' field")
    
    # Try other possible field names
    for field_name in ["company_symbol", "symbol", "ticker", "company_name"]:
        try:
            test_results = collection.get(
                where={field_name: "AAPL"},
                limit=1,
                include=["metadatas"]
            )
            print(f"Field '{field_name}': {len(test_results['ids'])} documents found")
        except Exception as e:
            print(f"Field '{field_name}': Error - {e}")

if __name__ == "__main__":
    main()
