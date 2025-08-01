"""
Text chunking utilities for document processing.
"""

from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import sys
from pathlib import Path

# Add config to path
config_path = Path(__file__).parent.parent.parent / "config"
sys.path.append(str(config_path))

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from model_configs import CHUNKING_CONFIGS
except ImportError:
    # Fallback chunking configs
    CHUNKING_CONFIGS = {
        "recursive": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "separators": ["\n\n", "\n", " ", ""],
            "length_function": len,
        },
        "semantic": {
            "buffer_size": 1,
            "breakpoint_threshold_type": "percentile",
            "breakpoint_threshold_amount": 95,
        },
        "fixed": {
            "chunk_size": 512,
            "chunk_overlap": 50,
        },
    }

class BaseChunker(ABC):
    """Base class for text chunkers."""
    
    @abstractmethod
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Chunk text into smaller pieces."""
        pass

class RecursiveCharacterTextSplitter(BaseChunker):
    """Recursively split text using different separators."""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 separators: Optional[List[str]] = None,
                 length_function: Callable[[str], int] = len):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        self.length_function = length_function
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Split text recursively using separators."""
        if metadata is None:
            metadata = {}
        
        chunks = self._split_text(text)
        
        return [
            {
                "content": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "chunk_size": len(chunk),
                    "chunk_type": "recursive"
                }
            }
            for i, chunk in enumerate(chunks)
        ]
    
    def _split_text(self, text: str) -> List[str]:
        """Split text recursively."""
        final_chunks = []
        separator = self.separators[-1]
        new_separators = []
        
        for i, _s in enumerate(self.separators):
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                new_separators = self.separators[i + 1:]
                break
        
        splits = text.split(separator)
        good_splits = []
        
        for s in splits:
            if self.length_function(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged_text = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged_text)
                    good_splits = []
                
                if new_separators:
                    other_chunks = self._split_text(s)
                    final_chunks.extend(other_chunks)
                else:
                    final_chunks.append(s)
        
        if good_splits:
            merged_text = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged_text)
        
        return final_chunks
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits while respecting chunk size and overlap."""
        docs = []
        current_doc = []
        total = 0
        
        for d in splits:
            _len = self.length_function(d)
            if total + _len + (len(current_doc) - 1) * len(separator) > self.chunk_size:
                if current_doc:
                    doc = separator.join(current_doc)
                    if doc:
                        docs.append(doc)
                
                # Start overlap
                while (total > self.chunk_overlap or 
                       (total + _len + (len(current_doc) - 1) * len(separator) > self.chunk_size and total > 0)):
                    total -= self.length_function(current_doc[0]) + (1 if len(current_doc) > 1 else 0) * len(separator)
                    current_doc = current_doc[1:]
            
            current_doc.append(d)
            total += _len + (1 if len(current_doc) > 1 else 0) * len(separator)
        
        doc = separator.join(current_doc)
        if doc:
            docs.append(doc)
        
        return docs

class FixedSizeChunker(BaseChunker):
    """Split text into fixed-size chunks."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Split text into fixed-size chunks."""
        if metadata is None:
            metadata = {}
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append({
                    "content": chunk,
                    "metadata": {
                        **metadata,
                        "chunk_index": len(chunks),
                        "chunk_size": len(chunk),
                        "chunk_type": "fixed",
                        "start_pos": start,
                        "end_pos": end
                    }
                })
            
            start = end - self.chunk_overlap
        
        return chunks

class TokenBasedChunker(BaseChunker):
    """Split text based on token count."""
    
    def __init__(self, 
                 max_tokens: int = 512,
                 overlap_tokens: int = 50,
                 encoding_name: str = "cl100k_base"):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        
        if TIKTOKEN_AVAILABLE:
            self.encoding = tiktoken.get_encoding(encoding_name)
        else:
            self.encoding = None
            print("Warning: tiktoken not available, using character-based approximation")
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Split text based on token count."""
        if metadata is None:
            metadata = {}
        
        if self.encoding is not None:
            # Use tiktoken for proper token-based chunking
            tokens = self.encoding.encode(text)
            chunks = []
            start = 0
            
            while start < len(tokens):
                end = start + self.max_tokens
                chunk_tokens = tokens[start:end]
                chunk_text = self.encoding.decode(chunk_tokens)
                
                if chunk_text.strip():
                    chunks.append({
                        "content": chunk_text,
                        "metadata": {
                            **metadata,
                            "chunk_index": len(chunks),
                            "token_count": len(chunk_tokens),
                            "chunk_type": "token_based",
                            "start_token": start,
                            "end_token": end
                        }
                    })
                
                start = end - self.overlap_tokens
        else:
            # Fallback to character-based approximation (4 chars ≈ 1 token)
            char_per_token = 4
            max_chars = self.max_tokens * char_per_token
            overlap_chars = self.overlap_tokens * char_per_token
            
            chunks = []
            start = 0
            
            while start < len(text):
                end = start + max_chars
                chunk_text = text[start:end]
                
                if chunk_text.strip():
                    chunks.append({
                        "content": chunk_text,
                        "metadata": {
                            **metadata,
                            "chunk_index": len(chunks),
                            "token_count": len(chunk_text) // char_per_token,  # Approximation
                            "chunk_type": "token_based_approx",
                            "start_char": start,
                            "end_char": end
                        }
                    })
                
                start = end - overlap_chars
        
        return chunks

class TextChunker:
    """Main text chunker that supports multiple chunking strategies."""
    
    def __init__(self):
        self.chunkers = {
            "recursive": RecursiveCharacterTextSplitter,
            "fixed": FixedSizeChunker,
            "token_based": TokenBasedChunker,
        }
    
    def get_chunker(self, chunker_type: str = "recursive", **kwargs) -> BaseChunker:
        """Get a chunker instance."""
        if chunker_type not in self.chunkers:
            raise ValueError(f"Unsupported chunker type: {chunker_type}")
        
        # Get default config
        config = CHUNKING_CONFIGS.get(chunker_type, {})
        config.update(kwargs)
        
        chunker_class = self.chunkers[chunker_type]
        return chunker_class(**config)
    
    def chunk_documents(self, 
                       documents: List[Dict[str, Any]], 
                       chunker_type: str = "recursive",
                       **kwargs) -> List[Dict[str, Any]]:
        """Chunk multiple documents."""
        chunker = self.get_chunker(chunker_type, **kwargs)
        all_chunks = []
        
        for doc in documents:
            doc_metadata = {
                "source_file": doc.get("file_name", "unknown"),
                "file_type": doc.get("file_type", "unknown"),
                "file_path": doc.get("file_path", "unknown")
            }
            
            chunks = chunker.chunk_text(doc["full_content"], doc_metadata)
            all_chunks.extend(chunks)
        
        return all_chunks
