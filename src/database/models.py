"""
Database models for the RAG evaluation system.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, 
    Boolean, JSON, ForeignKey, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.mysql import LONGTEXT

Base = declarative_base()


class Document(Base):
    """Document model for storing processed documents."""
    
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)  # pdf, txt, json, etc.
    file_size = Column(Integer)  # in bytes
    content_hash = Column(String(64), unique=True, nullable=False)  # SHA-256 hash
    metadata = Column(JSON)  # Additional metadata
    processed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}')>"


class DocumentChunk(Base):
    """Document chunk model for storing text chunks."""
    
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)  # Order within document
    content = Column(LONGTEXT, nullable=False)  # The actual text chunk
    start_char = Column(Integer)  # Start position in original document
    end_char = Column(Integer)    # End position in original document
    chunk_size = Column(Integer)  # Length in characters
    overlap_size = Column(Integer, default=0)  # Overlap with previous chunk
    metadata = Column(JSON)  # Chunk-specific metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    embeddings = relationship("ChunkEmbedding", back_populates="chunk", cascade="all, delete-orphan")
    
    # Index for faster queries
    __table_args__ = (
        Index('idx_document_chunk', 'document_id', 'chunk_index'),
    )
    
    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"


class ChunkEmbedding(Base):
    """Embedding vectors for document chunks."""
    
    __tablename__ = "chunk_embeddings"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(Integer, ForeignKey("document_chunks.id"), nullable=False)
    embedding_model = Column(String(100), nullable=False)  # Model used for embedding
    embedding_vector = Column(JSON, nullable=False)  # The actual embedding vector
    embedding_dimension = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    chunk = relationship("DocumentChunk", back_populates="embeddings")
    
    # Index for faster model-based queries
    __table_args__ = (
        Index('idx_chunk_model', 'chunk_id', 'embedding_model'),
    )
    
    def __repr__(self):
        return f"<ChunkEmbedding(id={self.id}, chunk_id={self.chunk_id}, model='{self.embedding_model}')>"


class Query(Base):
    """Query model for storing evaluation queries."""
    
    __tablename__ = "queries"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50))  # factual, analytical, creative, etc.
    domain = Column(String(100))     # subject domain
    difficulty_level = Column(String(20))  # easy, medium, hard
    expected_answer = Column(LONGTEXT)  # Ground truth answer if available
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    responses = relationship("QueryResponse", back_populates="query", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Query(id={self.id}, type='{self.query_type}')>"


class QueryResponse(Base):
    """Model for storing LLM responses to queries."""
    
    __tablename__ = "query_responses"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(Integer, ForeignKey("queries.id"), nullable=False)
    response_type = Column(String(20), nullable=False)  # 'rag' or 'non_rag'
    model_name = Column(String(100), nullable=False)
    response_text = Column(LONGTEXT, nullable=False)
    
    # RAG-specific fields
    retrieved_chunks = Column(JSON)  # List of chunk IDs used for RAG
    retrieval_scores = Column(JSON)  # Similarity scores for retrieved chunks
    context_used = Column(LONGTEXT)  # The actual context provided to LLM
    
    # Performance metrics
    response_time_ms = Column(Float)  # Response time in milliseconds
    token_count = Column(Integer)     # Number of tokens in response
    cost_estimate = Column(Float)     # Estimated API cost
    
    # Metadata
    temperature = Column(Float)
    max_tokens = Column(Integer)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    query = relationship("Query", back_populates="responses")
    evaluations = relationship("ResponseEvaluation", back_populates="response", cascade="all, delete-orphan")
    
    # Index for faster queries
    __table_args__ = (
        Index('idx_query_response_type', 'query_id', 'response_type'),
    )
    
    def __repr__(self):
        return f"<QueryResponse(id={self.id}, query_id={self.query_id}, type='{self.response_type}')>"


class ResponseEvaluation(Base):
    """Model for storing LLM judge evaluations of responses."""
    
    __tablename__ = "response_evaluations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    response_id = Column(Integer, ForeignKey("query_responses.id"), nullable=False)
    judge_model = Column(String(100), nullable=False)  # e.g., 'gemini-2.5-pro'
    
    # Evaluation scores (0-10 scale)
    accuracy_score = Column(Float)
    relevance_score = Column(Float)
    completeness_score = Column(Float)
    factual_correctness_score = Column(Float)
    coherence_score = Column(Float)
    overall_score = Column(Float)
    
    # Detailed feedback
    accuracy_feedback = Column(Text)
    relevance_feedback = Column(Text)
    completeness_feedback = Column(Text)
    factual_correctness_feedback = Column(Text)
    coherence_feedback = Column(Text)
    overall_feedback = Column(Text)
    
    # Evaluation metadata
    evaluation_prompt = Column(LONGTEXT)  # The prompt used for evaluation
    evaluation_time_ms = Column(Float)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    response = relationship("QueryResponse", back_populates="evaluations")
    
    def __repr__(self):
        return f"<ResponseEvaluation(id={self.id}, response_id={self.response_id}, overall_score={self.overall_score})>"


class EvaluationRun(Base):
    """Model for tracking evaluation runs/experiments."""
    
    __tablename__ = "evaluation_runs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_name = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Configuration
    rag_model_config = Column(JSON)
    non_rag_model_config = Column(JSON)
    embedding_model = Column(String(100))
    vector_store_type = Column(String(50))
    judge_model = Column(String(100))
    
    # Statistics
    total_queries = Column(Integer, default=0)
    completed_queries = Column(Integer, default=0)
    failed_queries = Column(Integer, default=0)
    
    # Status
    status = Column(String(20), default='running')  # running, completed, failed, paused
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Results summary
    results_summary = Column(JSON)  # Aggregated metrics
    
    def __repr__(self):
        return f"<EvaluationRun(id={self.id}, name='{self.run_name}', status='{self.status}')>"


class SystemMetrics(Base):
    """Model for storing system performance metrics."""
    
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50))  # ms, tokens, MB, etc.
    component = Column(String(100))   # embedding, retrieval, generation, etc.
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)
    
    # Index for time-series queries
    __table_args__ = (
        Index('idx_metric_time', 'metric_name', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<SystemMetrics(id={self.id}, name='{self.metric_name}', value={self.metric_value})>"
