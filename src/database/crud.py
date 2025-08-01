"""
CRUD operations for the RAG evaluation database.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime

from src.database.models import (
    Document, DocumentChunk, ChunkEmbedding, Query, 
    QueryResponse, ResponseEvaluation, EvaluationRun, SystemMetrics
)
from src.utils.logger import app_logger


class DocumentCRUD:
    """CRUD operations for Document model."""
    
    @staticmethod
    def create_document(
        db: Session,
        filename: str,
        file_path: str,
        file_type: str,
        content_hash: str,
        file_size: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[Document]:
        """Create a new document."""
        try:
            document = Document(
                filename=filename,
                file_path=file_path,
                file_type=file_type,
                file_size=file_size,
                content_hash=content_hash,
                metadata=metadata or {}
            )
            db.add(document)
            db.commit()
            db.refresh(document)
            return document
        except SQLAlchemyError as e:
            app_logger.error(f"Error creating document: {e}")
            db.rollback()
            return None
    
    @staticmethod
    def get_document_by_id(db: Session, document_id: int) -> Optional[Document]:
        """Get document by ID."""
        return db.query(Document).filter(Document.id == document_id).first()
    
    @staticmethod
    def get_document_by_hash(db: Session, content_hash: str) -> Optional[Document]:
        """Get document by content hash."""
        return db.query(Document).filter(Document.content_hash == content_hash).first()
    
    @staticmethod
    def get_all_documents(db: Session, skip: int = 0, limit: int = 100) -> List[Document]:
        """Get all documents with pagination."""
        return db.query(Document).offset(skip).limit(limit).all()
    
    @staticmethod
    def delete_document(db: Session, document_id: int) -> bool:
        """Delete document and all its chunks."""
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                db.delete(document)
                db.commit()
                return True
            return False
        except SQLAlchemyError as e:
            app_logger.error(f"Error deleting document: {e}")
            db.rollback()
            return False


class DocumentChunkCRUD:
    """CRUD operations for DocumentChunk model."""
    
    @staticmethod
    def create_chunk(
        db: Session,
        document_id: int,
        chunk_index: int,
        content: str,
        start_char: Optional[int] = None,
        end_char: Optional[int] = None,
        overlap_size: int = 0,
        metadata: Optional[Dict] = None
    ) -> Optional[DocumentChunk]:
        """Create a new document chunk."""
        try:
            chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=chunk_index,
                content=content,
                start_char=start_char,
                end_char=end_char,
                chunk_size=len(content),
                overlap_size=overlap_size,
                metadata=metadata or {}
            )
            db.add(chunk)
            db.commit()
            db.refresh(chunk)
            return chunk
        except SQLAlchemyError as e:
            app_logger.error(f"Error creating chunk: {e}")
            db.rollback()
            return None
    
    @staticmethod
    def get_chunks_by_document(db: Session, document_id: int) -> List[DocumentChunk]:
        """Get all chunks for a document."""
        return db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).order_by(DocumentChunk.chunk_index).all()
    
    @staticmethod
    def get_chunk_by_id(db: Session, chunk_id: int) -> Optional[DocumentChunk]:
        """Get chunk by ID."""
        return db.query(DocumentChunk).filter(DocumentChunk.id == chunk_id).first()


class ChunkEmbeddingCRUD:
    """CRUD operations for ChunkEmbedding model."""
    
    @staticmethod
    def create_embedding(
        db: Session,
        chunk_id: int,
        embedding_model: str,
        embedding_vector: List[float],
        embedding_dimension: int
    ) -> Optional[ChunkEmbedding]:
        """Create a new chunk embedding."""
        try:
            embedding = ChunkEmbedding(
                chunk_id=chunk_id,
                embedding_model=embedding_model,
                embedding_vector=embedding_vector,
                embedding_dimension=embedding_dimension
            )
            db.add(embedding)
            db.commit()
            db.refresh(embedding)
            return embedding
        except SQLAlchemyError as e:
            app_logger.error(f"Error creating embedding: {e}")
            db.rollback()
            return None
    
    @staticmethod
    def get_embeddings_by_chunk(db: Session, chunk_id: int) -> List[ChunkEmbedding]:
        """Get all embeddings for a chunk."""
        return db.query(ChunkEmbedding).filter(ChunkEmbedding.chunk_id == chunk_id).all()
    
    @staticmethod
    def get_embedding_by_model(
        db: Session, 
        chunk_id: int, 
        embedding_model: str
    ) -> Optional[ChunkEmbedding]:
        """Get embedding for a chunk by model."""
        return db.query(ChunkEmbedding).filter(
            ChunkEmbedding.chunk_id == chunk_id,
            ChunkEmbedding.embedding_model == embedding_model
        ).first()


class QueryCRUD:
    """CRUD operations for Query model."""
    
    @staticmethod
    def create_query(
        db: Session,
        query_text: str,
        query_type: Optional[str] = None,
        domain: Optional[str] = None,
        difficulty_level: Optional[str] = None,
        expected_answer: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[Query]:
        """Create a new query."""
        try:
            query = Query(
                query_text=query_text,
                query_type=query_type,
                domain=domain,
                difficulty_level=difficulty_level,
                expected_answer=expected_answer,
                metadata=metadata or {}
            )
            db.add(query)
            db.commit()
            db.refresh(query)
            return query
        except SQLAlchemyError as e:
            app_logger.error(f"Error creating query: {e}")
            db.rollback()
            return None
    
    @staticmethod
    def get_query_by_id(db: Session, query_id: int) -> Optional[Query]:
        """Get query by ID."""
        return db.query(Query).filter(Query.id == query_id).first()
    
    @staticmethod
    def get_all_queries(db: Session, skip: int = 0, limit: int = 100) -> List[Query]:
        """Get all queries with pagination."""
        return db.query(Query).offset(skip).limit(limit).all()


class QueryResponseCRUD:
    """CRUD operations for QueryResponse model."""
    
    @staticmethod
    def create_response(
        db: Session,
        query_id: int,
        response_type: str,  # 'rag' or 'non_rag'
        model_name: str,
        response_text: str,
        retrieved_chunks: Optional[List] = None,
        retrieval_scores: Optional[List] = None,
        context_used: Optional[str] = None,
        response_time_ms: Optional[float] = None,
        token_count: Optional[int] = None,
        cost_estimate: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[QueryResponse]:
        """Create a new query response."""
        try:
            response = QueryResponse(
                query_id=query_id,
                response_type=response_type,
                model_name=model_name,
                response_text=response_text,
                retrieved_chunks=retrieved_chunks,
                retrieval_scores=retrieval_scores,
                context_used=context_used,
                response_time_ms=response_time_ms,
                token_count=token_count,
                cost_estimate=cost_estimate,
                temperature=temperature,
                max_tokens=max_tokens,
                metadata=metadata or {}
            )
            db.add(response)
            db.commit()
            db.refresh(response)
            return response
        except SQLAlchemyError as e:
            app_logger.error(f"Error creating response: {e}")
            db.rollback()
            return None
    
    @staticmethod
    def get_responses_by_query(db: Session, query_id: int) -> List[QueryResponse]:
        """Get all responses for a query."""
        return db.query(QueryResponse).filter(QueryResponse.query_id == query_id).all()
    
    @staticmethod
    def get_response_by_id(db: Session, response_id: int) -> Optional[QueryResponse]:
        """Get response by ID."""
        return db.query(QueryResponse).filter(QueryResponse.id == response_id).first()


class ResponseEvaluationCRUD:
    """CRUD operations for ResponseEvaluation model."""
    
    @staticmethod
    def create_evaluation(
        db: Session,
        response_id: int,
        judge_model: str,
        accuracy_score: Optional[float] = None,
        relevance_score: Optional[float] = None,
        completeness_score: Optional[float] = None,
        factual_correctness_score: Optional[float] = None,
        coherence_score: Optional[float] = None,
        overall_score: Optional[float] = None,
        accuracy_feedback: Optional[str] = None,
        relevance_feedback: Optional[str] = None,
        completeness_feedback: Optional[str] = None,
        factual_correctness_feedback: Optional[str] = None,
        coherence_feedback: Optional[str] = None,
        overall_feedback: Optional[str] = None,
        evaluation_prompt: Optional[str] = None,
        evaluation_time_ms: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[ResponseEvaluation]:
        """Create a new response evaluation."""
        try:
            evaluation = ResponseEvaluation(
                response_id=response_id,
                judge_model=judge_model,
                accuracy_score=accuracy_score,
                relevance_score=relevance_score,
                completeness_score=completeness_score,
                factual_correctness_score=factual_correctness_score,
                coherence_score=coherence_score,
                overall_score=overall_score,
                accuracy_feedback=accuracy_feedback,
                relevance_feedback=relevance_feedback,
                completeness_feedback=completeness_feedback,
                factual_correctness_feedback=factual_correctness_feedback,
                coherence_feedback=coherence_feedback,
                overall_feedback=overall_feedback,
                evaluation_prompt=evaluation_prompt,
                evaluation_time_ms=evaluation_time_ms,
                metadata=metadata or {}
            )
            db.add(evaluation)
            db.commit()
            db.refresh(evaluation)
            return evaluation
        except SQLAlchemyError as e:
            app_logger.error(f"Error creating evaluation: {e}")
            db.rollback()
            return None
    
    @staticmethod
    def get_evaluations_by_response(db: Session, response_id: int) -> List[ResponseEvaluation]:
        """Get all evaluations for a response."""
        return db.query(ResponseEvaluation).filter(
            ResponseEvaluation.response_id == response_id
        ).all()


class EvaluationRunCRUD:
    """CRUD operations for EvaluationRun model."""
    
    @staticmethod
    def create_run(
        db: Session,
        run_name: str,
        description: Optional[str] = None,
        rag_model_config: Optional[Dict] = None,
        non_rag_model_config: Optional[Dict] = None,
        embedding_model: Optional[str] = None,
        vector_store_type: Optional[str] = None,
        judge_model: Optional[str] = None
    ) -> Optional[EvaluationRun]:
        """Create a new evaluation run."""
        try:
            run = EvaluationRun(
                run_name=run_name,
                description=description,
                rag_model_config=rag_model_config or {},
                non_rag_model_config=non_rag_model_config or {},
                embedding_model=embedding_model,
                vector_store_type=vector_store_type,
                judge_model=judge_model
            )
            db.add(run)
            db.commit()
            db.refresh(run)
            return run
        except SQLAlchemyError as e:
            app_logger.error(f"Error creating evaluation run: {e}")
            db.rollback()
            return None
    
    @staticmethod
    def update_run_status(
        db: Session,
        run_id: int,
        status: str,
        completed_queries: Optional[int] = None,
        failed_queries: Optional[int] = None,
        results_summary: Optional[Dict] = None
    ) -> bool:
        """Update evaluation run status."""
        try:
            run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
            if run:
                run.status = status
                if completed_queries is not None:
                    run.completed_queries = completed_queries
                if failed_queries is not None:
                    run.failed_queries = failed_queries
                if results_summary is not None:
                    run.results_summary = results_summary
                if status == 'completed':
                    run.completed_at = datetime.utcnow()
                
                db.commit()
                return True
            return False
        except SQLAlchemyError as e:
            app_logger.error(f"Error updating evaluation run: {e}")
            db.rollback()
            return False
    
    @staticmethod
    def get_run_by_id(db: Session, run_id: int) -> Optional[EvaluationRun]:
        """Get evaluation run by ID."""
        return db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()


class SystemMetricsCRUD:
    """CRUD operations for SystemMetrics model."""
    
    @staticmethod
    def record_metric(
        db: Session,
        metric_name: str,
        metric_value: float,
        metric_unit: Optional[str] = None,
        component: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[SystemMetrics]:
        """Record a system metric."""
        try:
            metric = SystemMetrics(
                metric_name=metric_name,
                metric_value=metric_value,
                metric_unit=metric_unit,
                component=component,
                metadata=metadata or {}
            )
            db.add(metric)
            db.commit()
            db.refresh(metric)
            return metric
        except SQLAlchemyError as e:
            app_logger.error(f"Error recording metric: {e}")
            db.rollback()
            return None
    
    @staticmethod
    def get_metrics_by_name(
        db: Session,
        metric_name: str,
        limit: int = 100
    ) -> List[SystemMetrics]:
        """Get metrics by name."""
        return db.query(SystemMetrics).filter(
            SystemMetrics.metric_name == metric_name
        ).order_by(SystemMetrics.timestamp.desc()).limit(limit).all()
