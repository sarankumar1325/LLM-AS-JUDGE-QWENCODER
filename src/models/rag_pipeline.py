"""
Complete RAG Pipeline Implementation (Task 3.2)
Combines document retrieval with LLM generation.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path
current_dir = Path(__file__).parent
root_dir = current_dir.parent
src_dir = root_dir / "src"
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(src_dir))

from src.vector_stores.chroma_store import ChromaVectorStore
from src.embeddings.nvidia_embedder import EmbedderFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)

class RAGPipeline:
    """Complete RAG (Retrieval-Augmented Generation) Pipeline."""
    
    def __init__(self, 
                 collection_name: str = "rag_evaluation_full",
                 persist_directory: str = "./data/chroma_db",
                 top_k: int = 5):
        """Initialize RAG pipeline components."""
        self.top_k = top_k
        
        # Initialize embedder for query encoding
        logger.info("Initializing embedder...")
        self.embedder = EmbedderFactory.create_default_embedder()
        
        # Initialize vector store
        logger.info("Connecting to vector store...")
        self.vector_store = ChromaVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        logger.info("RAG Pipeline initialized successfully")
    
    def retrieve_relevant_documents(self, 
                                  query: str, 
                                  company_filter: Optional[str] = None,
                                  top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: User query string
            company_filter: Optional company filter (e.g., "AAPL")
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with metadata
        """
        if top_k is None:
            top_k = self.top_k
            
        logger.info(f"Retrieving documents for query: '{query}'")
        
        try:
            # Encode query
            query_embedding = self.embedder.embed_text(query)
            
            # Prepare filter
            filter_dict = None
            if company_filter:
                filter_dict = {"company_symbol": company_filter}
                logger.info(f"Applying company filter: {company_filter}")
            
            # Retrieve documents
            results = self.vector_store.similarity_search_with_scores(
                query_embedding=query_embedding,
                k=top_k,
                filter_dict=filter_dict
            )
            
            # Format results
            retrieved_docs = []
            for doc, score in results:
                retrieved_docs.append({
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {}),
                    "similarity_score": score,
                    "document_id": doc.get("id", ""),
                    "company": doc.get("metadata", {}).get("company_symbol", "Unknown")
                })
            
            logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []
    
    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context for LLM."""
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            company = doc.get("company", "Unknown")
            content = doc["content"]
            score = doc.get("similarity_score", 0.0)
            
            context_parts.append(f"Document {i} [{company}] (Relevance: {score:.3f}):\n{content}")
        
        return "\n\n" + "\n\n".join(context_parts)
    
    def create_rag_prompt(self, query: str, context: str) -> str:
        """Create a complete RAG prompt with context and query."""
        prompt = f"""You are a financial analyst AI assistant. Use the provided context from earnings call transcripts to answer the user's question accurately and comprehensively.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Answer based primarily on the provided context
2. If the context doesn't contain sufficient information, clearly state this
3. Include specific details, numbers, and quotes when available
4. Mention which companies the information comes from
5. Be objective and factual in your analysis

ANSWER:"""
        
        return prompt
    
    def process_query(self, 
                     query: str, 
                     company_filter: Optional[str] = None,
                     top_k: Optional[int] = None) -> Dict[str, Any]:
        """Complete RAG pipeline processing.
        
        Args:
            query: User query
            company_filter: Optional company filter
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with retrieved documents, context, and prompt
        """
        logger.info(f"Processing RAG query: '{query}'")
        
        # Step 1: Retrieve relevant documents
        documents = self.retrieve_relevant_documents(query, company_filter, top_k)
        
        # Step 2: Format context
        context = self.format_context(documents)
        
        # Step 3: Create RAG prompt
        rag_prompt = self.create_rag_prompt(query, context)
        
        # Step 4: Prepare response
        result = {
            "query": query,
            "retrieved_documents": documents,
            "context": context,
            "rag_prompt": rag_prompt,
            "company_filter": company_filter,
            "document_count": len(documents)
        }
        
        return result
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG pipeline."""
        vector_stats = self.vector_store.get_collection_stats()
        
        return {
            "vector_store_stats": vector_stats,
            "embedder_dimension": self.embedder.get_dimension(),
            "default_top_k": self.top_k
        }

def main():
    """Test the RAG pipeline."""
    logger.info("Testing Complete RAG Pipeline...")
    
    # Initialize pipeline
    rag = RAGPipeline()
    
    # Test queries
    test_queries = [
        "What was Apple's revenue in Q4 2017?",
        "How did NVIDIA perform in the data center business?",
        "What are the key growth drivers mentioned by management?"
    ]
    
    for query in test_queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing query: {query}")
        logger.info(f"{'='*60}")
        
        result = rag.process_query(query)
        
        logger.info(f"Retrieved {result['document_count']} documents")
        logger.info(f"Context length: {len(result['context'])} characters")
        
        # Show prompt preview
        prompt_preview = result['rag_prompt'][:500] + "..." if len(result['rag_prompt']) > 500 else result['rag_prompt']
        logger.info(f"RAG Prompt Preview:\n{prompt_preview}")
    
    # Get pipeline stats
    stats = rag.get_pipeline_stats()
    logger.info(f"\nPipeline Statistics: {stats}")

if __name__ == "__main__":
    main()
