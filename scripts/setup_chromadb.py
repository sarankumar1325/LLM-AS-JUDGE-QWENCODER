"""
Script to load embeddings into ChromaDB for vector storage and retrieval.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Add project paths to Python path
current_dir = Path(__file__).parent
root_dir = current_dir.parent
src_dir = root_dir / "src"
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(src_dir))

from src.vector_stores.chroma_store import ChromaVectorStore
from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

class ChromaDBSetup:
    """Setup and populate ChromaDB with embeddings."""
    
    def __init__(self, embeddings_file: str = None):
        """Initialize ChromaDB setup.
        
        Args:
            embeddings_file: Path to embeddings JSON file
        """
        self.root_dir = Path(__file__).parent.parent
        self.embeddings_file = embeddings_file or str(self.root_dir / "processed" / "embeddings" / "all_embeddings.json")
        
        # Initialize ChromaDB store
        self.vector_store = ChromaVectorStore()
        
        logger.info(f"Initialized ChromaDB setup")
        logger.info(f"Embeddings file: {self.embeddings_file}")
        logger.info(f"ChromaDB persist directory: {self.vector_store.persist_directory}")
    
    def load_embeddings(self) -> List[Dict[str, Any]]:
        """Load embeddings from JSON file."""
        logger.info(f"Loading embeddings from: {self.embeddings_file}")
        
        if not Path(self.embeddings_file).exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_file}")
        
        with open(self.embeddings_file, 'r', encoding='utf-8') as f:
            embeddings = json.load(f)
        
        logger.info(f"Loaded {len(embeddings)} embedded chunks")
        return embeddings
    
    def prepare_chunks_for_chromadb(self, embeddings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare chunks for ChromaDB storage."""
        logger.info("Preparing chunks for ChromaDB storage...")
        
        prepared_chunks = []
        
        for chunk in embeddings:
            # Create a clean chunk for ChromaDB
            prepared_chunk = {
                "id": chunk.get("chunk_id", ""),
                "content": chunk.get("content", ""),
                "embedding": chunk.get("embedding", []),
                "metadata": {
                    "document_id": chunk.get("document_id", ""),
                    "chunk_index": str(chunk.get("chunk_index", 0)),
                    "company_symbol": chunk.get("metadata", {}).get("company_symbol", ""),
                    "date": chunk.get("metadata", {}).get("date", ""),
                    "file_name": chunk.get("metadata", {}).get("file_name", ""),
                    "chunk_start": str(chunk.get("metadata", {}).get("chunk_start", 0)),
                    "chunk_end": str(chunk.get("metadata", {}).get("chunk_end", 0)),
                    "embedding_model": chunk.get("embedding_metadata", {}).get("model", ""),
                    "embedding_dimension": str(len(chunk.get("embedding", []))),
                    "token_count": str(chunk.get("embedding_metadata", {}).get("token_count", 0))
                }
            }
            
            # Validate embedding
            if not prepared_chunk["embedding"] or not isinstance(prepared_chunk["embedding"], list):
                logger.warning(f"Invalid embedding for chunk: {prepared_chunk['id']}")
                continue
            
            prepared_chunks.append(prepared_chunk)
        
        logger.info(f"Prepared {len(prepared_chunks)} chunks for ChromaDB")
        return prepared_chunks
    
    def clear_existing_data(self):
        """Clear existing data from ChromaDB collection."""
        logger.info("Clearing existing data from ChromaDB collection...")
        try:
            # Get existing data count
            existing_count = self.vector_store.collection.count()
            logger.info(f"Found {existing_count} existing documents")
            
            if existing_count > 0:
                # Delete collection and recreate
                self.vector_store.client.delete_collection(self.vector_store.collection_name)
                logger.info("Deleted existing collection")
                
                # Recreate collection
                self.vector_store.collection = self.vector_store.client.get_or_create_collection(
                    name=self.vector_store.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("Recreated empty collection")
            else:
                logger.info("No existing data to clear")
                
        except Exception as e:
            logger.warning(f"Error clearing existing data: {e}")
    
    def load_into_chromadb(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """Load chunks into ChromaDB in batches."""
        logger.info(f"Loading {len(chunks)} chunks into ChromaDB...")
        
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        successfully_added = 0
        
        for batch_idx in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)...")
            
            try:
                added_ids = self.vector_store.add_documents(batch_chunks)
                successfully_added += len(added_ids)
                
                # Progress update
                percent = (successfully_added / len(chunks)) * 100
                logger.info(f"Progress: {successfully_added}/{len(chunks)} ({percent:.1f}%) chunks loaded")
                
            except Exception as e:
                logger.error(f"Failed to load batch {batch_num}: {e}")
                # Continue with next batch
                continue
        
        logger.info(f"Successfully loaded {successfully_added}/{len(chunks)} chunks into ChromaDB")
        return successfully_added
    
    def verify_chromadb_data(self):
        """Verify the data loaded into ChromaDB."""
        logger.info("Verifying ChromaDB data...")
        
        try:
            # Get collection info
            count = self.vector_store.collection.count()
            logger.info(f"Total documents in ChromaDB: {count}")
            
            # Test a simple query
            if count > 0:
                # Get a sample document
                sample_docs = self.vector_store.collection.peek(limit=1)
                if sample_docs['embeddings'] and len(sample_docs['embeddings']) > 0:
                    embedding_dim = len(sample_docs['embeddings'][0])
                    logger.info(f"Embedding dimension: {embedding_dim}")
                
                    # Test similarity search with the first embedding
                    test_results = self.vector_store.similarity_search(
                        query_embedding=sample_docs['embeddings'][0], 
                        k=3
                    )
                    logger.info(f"Test similarity search returned {len(test_results)} results")
                    
                    # Show sample metadata
                    if test_results:
                        sample_metadata = test_results[0].get('metadata', {})
                        logger.info(f"Sample metadata keys: {list(sample_metadata.keys())}")
            
            return count
            
        except Exception as e:
            logger.error(f"Error verifying ChromaDB data: {e}")
            return 0
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB collection."""
        try:
            count = self.vector_store.collection.count()
            
            if count == 0:
                return {"total_documents": 0}
            
            # Get sample data for analysis
            sample_data = self.vector_store.collection.peek(limit=min(100, count))
            
            # Extract companies and embedding info
            companies = set()
            embedding_dims = set()
            
            for metadata in sample_data.get('metadatas', []):
                if metadata and 'company_symbol' in metadata:
                    companies.add(metadata['company_symbol'])
            
            for embedding in sample_data.get('embeddings', []):
                if embedding and len(embedding) > 0:
                    embedding_dims.add(len(embedding))
            
            stats = {
                "total_documents": count,
                "unique_companies": len(companies),
                "companies": sorted(list(companies)),
                "embedding_dimensions": list(embedding_dims),
                "collection_name": self.vector_store.collection_name,
                "persist_directory": self.vector_store.persist_directory
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Setup ChromaDB with embeddings")
    parser.add_argument("--embeddings-file", help="Path to embeddings JSON file")
    parser.add_argument("--clear", action="store_true", help="Clear existing data before loading")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for loading")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing data")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics")
    
    args = parser.parse_args()
    
    # Initialize setup
    setup = ChromaDBSetup(embeddings_file=args.embeddings_file)
    
    # Show statistics if requested
    if args.stats:
        stats = setup.get_collection_stats()
        logger.info("=== ChromaDB Collection Statistics ===")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        return 0
    
    # Verify only if requested
    if args.verify_only:
        count = setup.verify_chromadb_data()
        return 0 if count > 0 else 1
    
    try:
        # Load embeddings
        embeddings = setup.load_embeddings()
        
        if not embeddings:
            logger.error("No embeddings found to load")
            return 1
        
        # Clear existing data if requested
        if args.clear:
            setup.clear_existing_data()
        
        # Prepare chunks for ChromaDB
        prepared_chunks = setup.prepare_chunks_for_chromadb(embeddings)
        
        if not prepared_chunks:
            logger.error("No valid chunks prepared for ChromaDB")
            return 1
        
        # Load into ChromaDB
        loaded_count = setup.load_into_chromadb(prepared_chunks, batch_size=args.batch_size)
        
        if loaded_count == 0:
            logger.error("Failed to load any chunks into ChromaDB")
            return 1
        
        # Verify the loaded data
        final_count = setup.verify_chromadb_data()
        
        if final_count > 0:
            logger.info(f"✓ ChromaDB setup completed successfully!")
            logger.info(f"✓ Loaded {final_count} documents into ChromaDB")
            
            # Show final statistics
            stats = setup.get_collection_stats()
            logger.info("=== Final ChromaDB Statistics ===")
            for key, value in stats.items():
                if key != "error":
                    logger.info(f"{key}: {value}")
        else:
            logger.error("✗ ChromaDB setup failed - no data verified")
            return 1
        
    except Exception as e:
        logger.error(f"✗ ChromaDB setup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
