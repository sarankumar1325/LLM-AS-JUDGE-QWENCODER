"""
Document loader for various file formats.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

import json
from abc import ABC, abstractmethod

class BaseDocumentLoader(ABC):
    """Base class for document loaders."""
    
    @abstractmethod
    def load(self, file_path: str) -> Dict[str, Any]:
        """Load document and return structured data."""
        pass

class PDFLoader(BaseDocumentLoader):
    """PDF document loader using PyMuPDF."""
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """Load PDF document."""
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF is required for PDF loading. Install with: pip install PyMuPDF")
        
        doc = fitz.open(file_path)
        pages = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            pages.append({
                "page_number": page_num + 1,
                "content": text.strip()
            })
        
        doc.close()
        
        return {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "file_type": "pdf",
            "total_pages": len(pages),
            "pages": pages,
            "full_content": "\n\n".join([p["content"] for p in pages])
        }

class DocxLoader(BaseDocumentLoader):
    """DOCX document loader."""
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """Load DOCX document."""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx is required for DOCX loading. Install with: pip install python-docx")
        
        doc = DocxDocument(file_path)
        paragraphs = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text.strip())
        
        return {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "file_type": "docx",
            "total_paragraphs": len(paragraphs),
            "paragraphs": paragraphs,
            "full_content": "\n\n".join(paragraphs)
        }

class TextLoader(BaseDocumentLoader):
    """Plain text document loader with enhanced metadata extraction."""
    
    def load(self, file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """Load text document."""
        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()
        
        lines = content.split("\n")
        file_path_obj = Path(file_path)
        
        # Extract metadata from financial earnings call files
        metadata = self._extract_metadata(file_path_obj, content)
        
        return {
            "file_path": file_path,
            "file_name": file_path_obj.name,
            "file_type": "txt",
            "total_lines": len(lines),
            "lines": lines,
            "full_content": content,
            "metadata": metadata
        }
    
    def _extract_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Extract metadata from financial earnings call files."""
        metadata = {}
        
        # Extract company symbol from folder name
        if len(file_path.parts) >= 2:
            metadata["company_symbol"] = file_path.parent.name
        
        # Extract date and company from filename
        filename = file_path.stem
        parts = filename.split("-")
        if len(parts) >= 4:
            try:
                metadata["year"] = parts[0]
                metadata["month"] = parts[1]
                metadata["day"] = parts[2]
                metadata["company"] = parts[3]
                metadata["date"] = f"{parts[0]}-{parts[1]}-{parts[2]}"
            except:
                pass
        
        # Extract document type and event info from content
        if "Thomson Reuters StreetEvents" in content:
            metadata["source"] = "Thomson Reuters StreetEvents"
            metadata["document_type"] = "earnings_call"
        
        # Extract quarter and year from content
        lines = content.split("\n")[:20]  # Check first 20 lines
        for line in lines:
            if "Q" in line and "202" in line and "Earnings Call" in line:
                metadata["event_type"] = "earnings_call"
                # Try to extract quarter info
                import re
                quarter_match = re.search(r'Q(\d) (\d{4})', line)
                if quarter_match:
                    metadata["quarter"] = f"Q{quarter_match.group(1)}"
                    metadata["fiscal_year"] = quarter_match.group(2)
        
        return metadata

class JSONLoader(BaseDocumentLoader):
    """JSON document loader."""
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """Load JSON document."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Convert JSON to text format for processing
        if isinstance(data, list):
            content = "\n\n".join([json.dumps(item, indent=2) for item in data])
        else:
            content = json.dumps(data, indent=2)
        
        return {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "file_type": "json",
            "data": data,
            "full_content": content
        }

class DocumentLoader:
    """Main document loader that handles multiple file formats."""
    
    def __init__(self):
        self.loaders = {
            ".txt": TextLoader(),
            ".json": JSONLoader(),
        }
        
        # Add optional loaders if dependencies are available
        if PYMUPDF_AVAILABLE:
            self.loaders[".pdf"] = PDFLoader()
        
        if DOCX_AVAILABLE:
            self.loaders[".docx"] = DocxLoader()
    
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """Load a single document."""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension not in self.loaders:
            raise ValueError(f"Unsupported file format: {extension}")
        
        loader = self.loaders[extension]
        return loader.load(str(file_path))
    
    def load_directory(self, directory_path: str, 
                      supported_extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Load all documents from a directory."""
        if supported_extensions is None:
            supported_extensions = list(self.loaders.keys())
        
        directory = Path(directory_path)
        documents = []
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    doc = self.load_document(str(file_path))
                    documents.append(doc)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def load_company_data(self, base_directory: str, company_symbol: str = None) -> List[Dict[str, Any]]:
        """Load data for a specific company or all companies from the real data folder."""
        base_path = Path(base_directory)
        documents = []
        
        if company_symbol:
            # Load data for specific company
            company_path = base_path / company_symbol
            if company_path.exists():
                documents.extend(self.load_directory(str(company_path), [".txt"]))
        else:
            # Load data for all companies
            for company_folder in base_path.iterdir():
                if company_folder.is_dir():
                    print(f"Loading documents for {company_folder.name}...")
                    company_docs = self.load_directory(str(company_folder), [".txt"])
                    documents.extend(company_docs)
                    print(f"Loaded {len(company_docs)} documents for {company_folder.name}")
        
        return documents
    
    def get_company_list(self, base_directory: str) -> List[str]:
        """Get list of available companies in the data directory."""
        base_path = Path(base_directory)
        companies = []
        
        for company_folder in base_path.iterdir():
            if company_folder.is_dir():
                companies.append(company_folder.name)
        
        return sorted(companies)
    
    def get_document_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about loaded documents."""
        if not documents:
            return {"total_documents": 0}
        
        stats = {
            "total_documents": len(documents),
            "companies": {},
            "years": {},
            "document_types": {},
            "total_content_length": 0
        }
        
        for doc in documents:
            # Count content length
            stats["total_content_length"] += len(doc.get("full_content", ""))
            
            # Count by metadata if available
            metadata = doc.get("metadata", {})
            
            if "company_symbol" in metadata:
                company = metadata["company_symbol"]
                stats["companies"][company] = stats["companies"].get(company, 0) + 1
            
            if "year" in metadata:
                year = metadata["year"]
                stats["years"][year] = stats["years"].get(year, 0) + 1
            
            if "document_type" in metadata:
                doc_type = metadata["document_type"]
                stats["document_types"][doc_type] = stats["document_types"].get(doc_type, 0) + 1
        
        return stats
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return list(self.loaders.keys())
