"""
Script to process the real data folder containing financial earnings call transcripts.
"""

import os
import sys
from pathlib import Path
import json
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_processing.document_loader import DocumentLoader
from data_processing.text_chunker import TextChunker
from data_processing.preprocessor import BasicTextPreprocessor
from utils.logger import get_logger

logger = get_logger(__name__)

class RealDataProcessor:
    """Processor for the real data folder structure."""
    
    def __init__(self, base_data_path: str):
        self.base_data_path = Path(base_data_path)
        self.real_data_path = self.base_data_path / "real data"
        self.processed_path = self.base_data_path / "processed"
        self.chunks_path = self.processed_path / "chunks"
        self.embeddings_path = self.processed_path / "embeddings"
        
        # Initialize processors
        self.document_loader = DocumentLoader()
        self.text_chunker_factory = TextChunker()
        self.text_chunker = self.text_chunker_factory.get_chunker("recursive", chunk_size=1000, chunk_overlap=200)
        self.preprocessor = BasicTextPreprocessor()
        
        # Create output directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary output directories."""
        for path in [self.processed_path, self.chunks_path, self.embeddings_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def process_all_companies(self) -> Dict[str, Any]:
        """Process documents for all companies."""
        logger.info("Starting processing of all companies...")
        
        # Load all documents
        documents = self.document_loader.load_company_data(str(self.real_data_path))
        logger.info(f"Loaded {len(documents)} documents")
        
        # Get statistics
        stats = self.document_loader.get_document_stats(documents)
        logger.info(f"Document statistics: {stats}")
        
        # Process documents
        processed_docs = []
        for i, doc in enumerate(documents):
            logger.info(f"Processing document {i+1}/{len(documents)}: {doc['file_name']}")
            
            try:
                processed_doc = self._process_single_document(doc)
                processed_docs.append(processed_doc)
            except Exception as e:
                logger.error(f"Error processing document {doc['file_name']}: {e}")
        
        # Save processed documents
        output_file = self.processed_path / "all_processed_documents.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_docs, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(processed_docs)} processed documents to {output_file}")
        
        return {
            "total_documents": len(documents),
            "processed_documents": len(processed_docs),
            "output_file": str(output_file),
            "statistics": stats
        }
    
    def process_single_company(self, company_symbol: str) -> Dict[str, Any]:
        """Process documents for a single company."""
        logger.info(f"Processing documents for company: {company_symbol}")
        
        # Load company documents
        documents = self.document_loader.load_company_data(
            str(self.real_data_path), 
            company_symbol
        )
        
        if not documents:
            logger.warning(f"No documents found for company: {company_symbol}")
            return {"error": f"No documents found for {company_symbol}"}
        
        logger.info(f"Loaded {len(documents)} documents for {company_symbol}")
        
        # Process documents
        processed_docs = []
        for doc in documents:
            try:
                processed_doc = self._process_single_document(doc)
                processed_docs.append(processed_doc)
            except Exception as e:
                logger.error(f"Error processing document {doc['file_name']}: {e}")
        
        # Save processed documents for this company
        company_output_dir = self.processed_path / company_symbol
        company_output_dir.mkdir(exist_ok=True)
        
        output_file = company_output_dir / f"{company_symbol}_processed.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_docs, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(processed_docs)} processed documents to {output_file}")
        
        return {
            "company": company_symbol,
            "total_documents": len(documents),
            "processed_documents": len(processed_docs),
            "output_file": str(output_file)
        }
    
    def _process_single_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single document."""
        # Extract content
        content = document["full_content"]
        
        # Preprocess content
        processed_content = self.preprocessor.preprocess(content)
        
        # Create chunks
        chunks = self.text_chunker.chunk_text(processed_content)
        
        # Create processed document
        processed_doc = {
            "document_id": self._generate_document_id(document),
            "original_file": document["file_path"],
            "file_name": document["file_name"],
            "metadata": document.get("metadata", {}),
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
        
        return processed_doc
    
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
        return self.document_loader.get_company_list(str(self.real_data_path))
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        companies = self.get_available_companies()
        total_files = 0
        
        stats = {
            "total_companies": len(companies),
            "companies": {},
            "total_files": 0
        }
        
        for company in companies:
            company_path = self.real_data_path / company
            if company_path.exists():
                files = list(company_path.glob("*.txt"))
                file_count = len(files)
                total_files += file_count
                
                stats["companies"][company] = {
                    "file_count": file_count,
                    "files": [f.name for f in files]
                }
        
        stats["total_files"] = total_files
        return stats

def main():
    """Main processing function."""
    # Get the base directory (parent of scripts folder)
    base_dir = Path(__file__).parent.parent
    
    processor = RealDataProcessor(str(base_dir))
    
    # Show available companies
    companies = processor.get_available_companies()
    print(f"Available companies: {companies}")
    
    # Show processing stats
    stats = processor.get_processing_stats()
    print(f"Processing statistics:")
    print(f"  Total companies: {stats['total_companies']}")
    print(f"  Total files: {stats['total_files']}")
    
    # Ask user what to process
    choice = input("\nWhat would you like to process?\n1. All companies\n2. Specific company\n3. Show stats only\nEnter choice (1-3): ")
    
    if choice == "1":
        print("Processing all companies...")
        result = processor.process_all_companies()
        print(f"Processing complete: {result}")
    
    elif choice == "2":
        print(f"Available companies: {', '.join(companies)}")
        company = input("Enter company symbol: ").upper()
        if company in companies:
            result = processor.process_single_company(company)
            print(f"Processing complete: {result}")
        else:
            print(f"Company {company} not found!")
    
    elif choice == "3":
        print("Statistics displayed above.")
    
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
