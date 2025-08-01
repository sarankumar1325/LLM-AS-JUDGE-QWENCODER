"""
Text preprocessing utilities.
"""

import re
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

class BasePreprocessor(ABC):
    """Base class for text preprocessors."""
    
    @abstractmethod
    def preprocess(self, text: str) -> str:
        """Preprocess text."""
        pass

class BasicTextPreprocessor(BasePreprocessor):
    """Basic text preprocessing operations."""
    
    def __init__(self,
                 lowercase: bool = False,
                 remove_extra_whitespace: bool = True,
                 remove_special_chars: bool = False,
                 remove_numbers: bool = False,
                 remove_urls: bool = True,
                 remove_emails: bool = True):
        self.lowercase = lowercase
        self.remove_extra_whitespace = remove_extra_whitespace
        self.remove_special_chars = remove_special_chars
        self.remove_numbers = remove_numbers
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
    
    def preprocess(self, text: str) -> str:
        """Apply preprocessing steps."""
        if not text:
            return text
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove emails
        if self.remove_emails:
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove special characters (keep basic punctuation)
        if self.remove_special_chars:
            text = re.sub(r'[^a-zA-Z0-9\s.,!?;:]', '', text)
        
        # Remove extra whitespace
        if self.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text

class DocumentPreprocessor(BasePreprocessor):
    """Preprocessor specifically for document content."""
    
    def __init__(self,
                 remove_headers_footers: bool = True,
                 remove_page_numbers: bool = True,
                 clean_line_breaks: bool = True,
                 normalize_quotes: bool = True,
                 remove_excessive_punctuation: bool = True):
        self.remove_headers_footers = remove_headers_footers
        self.remove_page_numbers = remove_page_numbers
        self.clean_line_breaks = clean_line_breaks
        self.normalize_quotes = normalize_quotes
        self.remove_excessive_punctuation = remove_excessive_punctuation
    
    def preprocess(self, text: str) -> str:
        """Preprocess document text."""
        if not text:
            return text
        
        # Remove common header/footer patterns
        if self.remove_headers_footers:
            # Remove patterns like "Page 1 of 10"
            text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
            # Remove copyright notices
            text = re.sub(r'©.*?\d{4}', '', text)
            # Remove common footer patterns
            text = re.sub(r'www\.[^\s]+', '', text)
        
        # Remove standalone page numbers
        if self.remove_page_numbers:
            text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Clean line breaks
        if self.clean_line_breaks:
            # Replace multiple newlines with double newline
            text = re.sub(r'\n{3,}', '\n\n', text)
            # Fix broken words across lines
            text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Normalize quotes
        if self.normalize_quotes:
            text = re.sub(r'[""''`]', '"', text)
        
        # Remove excessive punctuation
        if self.remove_excessive_punctuation:
            text = re.sub(r'[.]{3,}', '...', text)
            text = re.sub(r'[!]{2,}', '!', text)
            text = re.sub(r'[?]{2,}', '?', text)
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

class CodePreprocessor(BasePreprocessor):
    """Preprocessor for code content."""
    
    def __init__(self,
                 remove_comments: bool = False,
                 normalize_indentation: bool = True,
                 remove_empty_lines: bool = False):
        self.remove_comments = remove_comments
        self.normalize_indentation = normalize_indentation
        self.remove_empty_lines = remove_empty_lines
    
    def preprocess(self, text: str) -> str:
        """Preprocess code text."""
        if not text:
            return text
        
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            # Remove comments (basic patterns)
            if self.remove_comments:
                # Remove Python/Shell style comments
                line = re.sub(r'#.*$', '', line)
                # Remove C/Java style comments
                line = re.sub(r'//.*$', '', line)
            
            # Skip empty lines if requested
            if self.remove_empty_lines and not line.strip():
                continue
            
            processed_lines.append(line)
        
        text = '\n'.join(processed_lines)
        
        # Normalize indentation
        if self.normalize_indentation:
            # Convert tabs to spaces
            text = text.expandtabs(4)
        
        return text

class TextPreprocessor:
    """Main preprocessor that handles different text types."""
    
    def __init__(self):
        self.preprocessors = {
            "basic": BasicTextPreprocessor,
            "document": DocumentPreprocessor,
            "code": CodePreprocessor,
        }
    
    def get_preprocessor(self, preprocessor_type: str = "basic", **kwargs) -> BasePreprocessor:
        """Get a preprocessor instance."""
        if preprocessor_type not in self.preprocessors:
            raise ValueError(f"Unsupported preprocessor type: {preprocessor_type}")
        
        preprocessor_class = self.preprocessors[preprocessor_type]
        return preprocessor_class(**kwargs)
    
    def preprocess_chunks(self, 
                         chunks: List[Dict[str, Any]], 
                         preprocessor_type: str = "basic",
                         **kwargs) -> List[Dict[str, Any]]:
        """Preprocess text chunks."""
        preprocessor = self.get_preprocessor(preprocessor_type, **kwargs)
        
        processed_chunks = []
        for chunk in chunks:
            processed_content = preprocessor.preprocess(chunk["content"])
            
            # Only keep chunks with meaningful content
            if processed_content.strip():
                processed_chunk = chunk.copy()
                processed_chunk["content"] = processed_content
                processed_chunk["metadata"]["preprocessed"] = True
                processed_chunk["metadata"]["preprocessor_type"] = preprocessor_type
                processed_chunks.append(processed_chunk)
        
        return processed_chunks
    
    def preprocess_documents(self, 
                           documents: List[Dict[str, Any]], 
                           preprocessor_type: str = "document",
                           **kwargs) -> List[Dict[str, Any]]:
        """Preprocess full documents."""
        preprocessor = self.get_preprocessor(preprocessor_type, **kwargs)
        
        processed_documents = []
        for doc in documents:
            processed_content = preprocessor.preprocess(doc["full_content"])
            
            if processed_content.strip():
                processed_doc = doc.copy()
                processed_doc["full_content"] = processed_content
                processed_doc["preprocessed"] = True
                processed_doc["preprocessor_type"] = preprocessor_type
                processed_documents.append(processed_doc)
        
        return processed_documents
