#!/usr/bin/env python3
"""
Process dataset for the RAG evaluation system.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_processing.document_loader import DocumentLoader
from data_processing.text_chunker import TextChunker
from data_processing.preprocessor import TextPreprocessor
from utils.logger import get_logger
from config.settings import settings

app_logger = get_logger(__name__)

def process_dataset(
    input_dir: str = None,
    output_dir: str = None,
    chunker_type: str = "recursive",
    preprocessor_type: str = "document"
):
    """Process raw documents into chunks."""
    
    # Set default paths
    if input_dir is None:
        input_dir = settings.DATA_DIR / "raw" / "dataset"
    if output_dir is None:
        output_dir = settings.DATA_DIR / "processed" / "chunks"
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    app_logger.info(f"Processing documents from: {input_path}")
    app_logger.info(f"Output directory: {output_path}")
    
    try:
        # Initialize components
        document_loader = DocumentLoader()
        text_chunker = TextChunker()
        text_preprocessor = TextPreprocessor()
        
        # Load documents
        app_logger.info("Loading documents...")
        documents = document_loader.load_directory(str(input_path))
        app_logger.info(f"Loaded {len(documents)} documents")
        
        if not documents:
            app_logger.warning("No documents found to process")
            return
        
        # Preprocess documents
        app_logger.info("Preprocessing documents...")
        processed_documents = text_preprocessor.preprocess_documents(
            documents, preprocessor_type
        )
        app_logger.info(f"Preprocessed {len(processed_documents)} documents")
        
        # Chunk documents
        app_logger.info("Chunking documents...")
        chunks = text_chunker.chunk_documents(
            processed_documents, chunker_type
        )
        app_logger.info(f"Created {len(chunks)} chunks")
        
        # Save chunks
        output_file = output_path / "processed_chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        app_logger.info(f"Saved chunks to: {output_file}")
        
        # Save processing statistics
        stats = {
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "chunker_type": chunker_type,
            "preprocessor_type": preprocessor_type,
            "input_directory": str(input_path),
            "output_file": str(output_file)
        }
        
        stats_file = output_path / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        app_logger.info("Dataset processing completed successfully")
        print(f"Processed {len(documents)} documents into {len(chunks)} chunks")
        
    except Exception as e:
        app_logger.error(f"Dataset processing failed: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process dataset for RAG evaluation")
    parser.add_argument("--input-dir", help="Input directory containing documents")
    parser.add_argument("--output-dir", help="Output directory for processed chunks")
    parser.add_argument("--chunker-type", default="recursive", 
                       choices=["recursive", "fixed", "token_based"],
                       help="Type of text chunker to use")
    parser.add_argument("--preprocessor-type", default="document",
                       choices=["basic", "document", "code"],
                       help="Type of preprocessor to use")
    
    args = parser.parse_args()
    
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunker_type=args.chunker_type,
        preprocessor_type=args.preprocessor_type
    )
