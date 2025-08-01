"""
Script to generate embeddings for processed documents using NVIDIA NemoRetriever.
"""

import os
import sys
import json
from pathlib import Path
import argparse
from typing import List, Dict, Any

# Add both src and root to path for proper imports
current_dir = Path(__file__).parent
root_dir = current_dir.parent
src_dir = root_dir / "src"
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(src_dir))

from embeddings.nvidia_embedder import EmbedderFactory, EmbeddingPipeline

# Try to import data processing modules, make them optional for now
try:
    from data_processing.document_loader import DocumentLoader
    from data_processing.text_chunker import TextChunker  
    from data_processing.preprocessor import TextPreprocessor
    DATA_PROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Data processing modules not available: {e}")
    print("Some features will be limited. Install missing dependencies.")
    DATA_PROCESSING_AVAILABLE = False
    DocumentLoader = None
    TextChunker = None
    TextPreprocessor = None

from utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

class EmbeddingGenerator:
    """Main class for generating embeddings from processed documents."""
    
    def __init__(self, base_data_path: str):
        self.base_data_path = Path(base_data_path)
        self.real_data_path = self.base_data_path / "real data"
        self.processed_path = self.base_data_path / "processed"
        self.embeddings_path = self.processed_path / "embeddings"
        
        # Create output directory
        self.embeddings_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components using factory
        self.embedder = EmbedderFactory.create_default_embedder()
        self.embedding_pipeline = EmbeddingPipeline(self.embedder)
        
    def test_embedder_connection(self) -> bool:
        """Test connection to embedding service."""
        logger.info("Testing embedding service connection...")
        
        try:
            # Test with a simple text
            test_embedding = self.embedder.embed_text("This is a test.")
            
            if test_embedding and len(test_embedding) > 0:
                logger.info(f"✓ Embedding service connection successful")
                logger.info(f"✓ Embedding dimension: {len(test_embedding)}")
                
                # Show cache stats if available (for NVIDIA embedder)
                if hasattr(self.embedder, 'get_cache_stats'):
                    cache_stats = self.embedder.get_cache_stats()
                    logger.info(f"Cache statistics: {cache_stats}")
                
                return True
            else:
                logger.error("✗ Embedding service connection failed - no embedding returned")
                return False
                
        except Exception as e:
            logger.error(f"✗ Embedding service connection test error: {e}")
            return False
    
    def load_processed_documents(self, company: str = None) -> List[Dict[str, Any]]:
        """Load processed documents from JSON files.
        
        Args:
            company: Specific company to load, or None for all companies
            
        Returns:
            List of processed documents
        """
        documents = []
        
        if company:
            # Load specific company
            company_file = self.processed_path / company / f"{company}_processed.json"
            if company_file.exists():
                with open(company_file, 'r', encoding='utf-8') as f:
                    documents = json.load(f)
                logger.info(f"Loaded {len(documents)} processed documents for {company}")
            else:
                logger.warning(f"No processed documents found for {company}")
        else:
            # Load all processed documents
            all_docs_file = self.processed_path / "all_processed_documents.json"
            if all_docs_file.exists():
                with open(all_docs_file, 'r', encoding='utf-8') as f:
                    documents = json.load(f)
                logger.info(f"Loaded {len(documents)} processed documents from all companies")
            else:
                logger.warning("No processed documents file found. Run process_real_data.py first.")
        
        return documents
    
    def process_from_raw_data(self, company: str = None) -> List[Dict[str, Any]]:
        """Process raw data and then generate embeddings.
        
        Args:
            company: Specific company to process, or None for all companies
            
        Returns:
            List of processed documents with chunks
        """
        if not DATA_PROCESSING_AVAILABLE:
            logger.error("Data processing modules not available. Cannot process from raw data.")
            return []
            
        logger.info("Processing raw data before embedding...")
        
        # Initialize processors
        document_loader = DocumentLoader()
        text_chunker_factory = TextChunker()
        text_chunker = text_chunker_factory.get_chunker("recursive", chunk_size=1000, chunk_overlap=200)
        preprocessor = TextPreprocessor()
        
        # Load documents
        if company:
            documents = document_loader.load_company_data(str(self.real_data_path), company)
        else:
            documents = document_loader.load_company_data(str(self.real_data_path))
        
        logger.info(f"Loaded {len(documents)} raw documents")
        
        # Process documents
        processed_docs = []
        for i, doc in enumerate(documents):
            logger.info(f"Processing document {i+1}/{len(documents)}: {doc['file_name']}")
            
            try:
                # Preprocess content
                content = doc["full_content"]
                processed_content = preprocessor.preprocess(content)
                
                # Create chunks
                chunks = text_chunker.chunk_text(processed_content)
                
                # Create processed document
                processed_doc = {
                    "document_id": self._generate_document_id(doc),
                    "original_file": doc["file_path"],
                    "file_name": doc["file_name"],
                    "metadata": doc.get("metadata", {}),
                    "original_content": content,
                    "processed_content": processed_content,
                    "chunks": chunks,
                    "total_chunks": len(chunks),
                    "processing_stats": {
                        "original_length": len(content),
                        "processed_length": len(processed_content),
                        "chunk_count": len(chunks),
                        "avg_chunk_size": sum(len(chunk["content"]) for chunk in chunks) / len(chunks) if chunks else 0
                    }
                }
                
                processed_docs.append(processed_doc)
                
            except Exception as e:
                logger.error(f"Error processing document {doc['file_name']}: {e}")
        
        return processed_docs
    
    def generate_embeddings(self, company: str = None, use_processed: bool = True, limit: int = None) -> Dict[str, Any]:
        """Generate embeddings for documents.
        
        Args:
            company: Specific company to process, or None for all companies
            use_processed: Whether to use pre-processed documents or process from raw
            limit: Maximum number of documents to process (for testing)
            
        Returns:
            Dictionary with results and statistics
        """
        logger.info(f"Starting embedding generation for {'all companies' if not company else company}")
        
        # Load or process documents
        if use_processed:
            documents = self.load_processed_documents(company)
        else:
            documents = self.process_from_raw_data(company)
        
        if not documents:
            return {"error": "No documents to process"}
        
        # Apply limit if specified
        if limit and limit < len(documents):
            logger.info(f"Limiting processing to first {limit} documents (out of {len(documents)})")
            documents = documents[:limit]
        
        # Generate embeddings
        logger.info("Generating embeddings for document chunks...")
        embedded_chunks = self.embedding_pipeline.process_document_chunks(documents)
        
        if not embedded_chunks:
            return {"error": "No embeddings generated"}
        
        # Save embeddings
        if company:
            suffix = f"_limit{limit}" if limit else ""
            output_file = self.embeddings_path / f"{company}_embeddings{suffix}.json"
        else:
            suffix = f"_limit{limit}" if limit else ""
            output_file = self.embeddings_path / f"all_embeddings{suffix}.json"
        
        saved_file = self.embedding_pipeline.save_embeddings(embedded_chunks, str(output_file))
        
        # Get statistics
        stats = self.embedding_pipeline.get_embedding_stats(embedded_chunks)
        
        # Get cache statistics if available
        cache_stats = {}
        if hasattr(self.embedder, 'get_cache_stats'):
            cache_stats = self.embedder.get_cache_stats()
        
        result = {
            "success": True,
            "company": company,
            "total_documents": len(documents),
            "total_embedded_chunks": len(embedded_chunks),
            "output_file": saved_file,
            "embedding_stats": stats,
            "cache_stats": cache_stats,
        }
        
        logger.info(f"Embedding generation completed successfully: {result}")
        
        return result
    
    def _generate_document_id(self, document: Dict[str, Any]) -> str:
        """Generate a unique ID for a document."""
        metadata = document.get("metadata", {})
        
        if "company_symbol" in metadata and "date" in metadata:
            return f"{metadata['company_symbol']}_{metadata['date']}"
        else:
            # Fallback to filename without extension
            return Path(document["file_name"]).stem
    
    def get_available_companies(self) -> List[str]:
        """Get list of available companies."""
        if not DATA_PROCESSING_AVAILABLE:
            # Fallback: get companies from directory listing
            companies = []
            if self.real_data_path.exists():
                for item in self.real_data_path.iterdir():
                    if item.is_dir():
                        companies.append(item.name)
            return sorted(companies)
        else:
            loader = DocumentLoader()
            return loader.get_company_list(str(self.real_data_path))
    
    def show_embedding_stats(self):
        """Show existing embedding statistics."""
        logger.info("=== Embedding Statistics ===")
        
        # Check for existing embedding files
        embedding_files = list(self.embeddings_path.glob("*.json"))
        
        if not embedding_files:
            logger.info("No embedding files found.")
            return
        
        for file in embedding_files:
            logger.info(f"\nFile: {file.name}")
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list) and data:
                    total_chunks = len(data)
                    
                    # Get companies
                    companies = set()
                    total_tokens = 0
                    embedding_dim = 0
                    
                    for chunk in data:
                        if "metadata" in chunk and "company_symbol" in chunk["metadata"]:
                            companies.add(chunk["metadata"]["company_symbol"])
                        
                        if "embedding_metadata" in chunk:
                            total_tokens += chunk["embedding_metadata"].get("token_count", 0)
                        
                        if "embedding" in chunk and not embedding_dim:
                            embedding_dim = len(chunk["embedding"])
                    
                    logger.info(f"  Total chunks: {total_chunks}")
                    logger.info(f"  Companies: {sorted(list(companies))}")
                    logger.info(f"  Total tokens: {total_tokens:,}")
                    logger.info(f"  Embedding dimension: {embedding_dim}")
                    
            except Exception as e:
                logger.error(f"Error reading {file.name}: {e}")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Generate embeddings for RAG evaluation")
    parser.add_argument("--company", help="Specific company to process (e.g., AAPL)")
    parser.add_argument("--test-connection", action="store_true", 
                       help="Test embedding service connection only")
    parser.add_argument("--stats", action="store_true", 
                       help="Show existing embedding statistics")
    parser.add_argument("--from-raw", action="store_true",
                       help="Process from raw data instead of using processed files")
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear embedding cache before processing")
    parser.add_argument("--limit", type=int, 
                       help="Limit number of documents to process (for testing)")
    
    args = parser.parse_args()
    
    # Get the base directory (parent of scripts folder)
    base_dir = Path(__file__).parent.parent
    
    generator = EmbeddingGenerator(str(base_dir))
    
    # Test connection
    if args.test_connection:
        success = generator.test_embedder_connection()
        return 0 if success else 1
    
    # Show stats
    if args.stats:
        generator.show_embedding_stats()
        return 0
    
    # Clear cache if requested
    if args.clear_cache:
        if hasattr(generator.embedder, 'clear_cache'):
            generator.embedder.clear_cache()
            logger.info("Embedding cache cleared")
        else:
            logger.info("Cache clearing not supported for this embedder")

    # Test connection first
    if not generator.test_embedder_connection():
        logger.error("Cannot proceed without valid embedding service connection")
        return 1
    
    # Show available companies
    companies = generator.get_available_companies()
    logger.info(f"Available companies: {companies}")
    
    # Generate embeddings
    try:
        result = generator.generate_embeddings(
            company=args.company,
            use_processed=not args.from_raw,
            limit=args.limit
        )
        
        if result.get("success"):
            logger.info("✓ Embedding generation completed successfully!")
            logger.info(f"Generated embeddings for {result['total_embedded_chunks']} chunks")
            logger.info(f"Saved to: {result['output_file']}")
        else:
            logger.error(f"✗ Embedding generation failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"✗ Embedding generation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
