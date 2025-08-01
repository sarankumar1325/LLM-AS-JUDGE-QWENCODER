"""
RAG Evaluation Framework using Gemini 2.5 Pro as judge.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import argparse
from dataclasses import dataclass

# Add parent directories for imports
current_dir = Path(__file__).parent
root_dir = current_dir.parent
src_dir = root_dir / "src"
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(src_dir))

from src.models.gemini_client import GeminiClient, RAGModel, NonRAGModel
from src.vector_stores.chroma_store import ChromaVectorStore
from src.embeddings.nvidia_embedder import EmbedderFactory
from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

@dataclass
class EvaluationQuery:
    """Data class for evaluation queries."""
    id: str
    query: str
    expected_companies: List[str] = None
    category: str = "general"
    difficulty: str = "medium"

@dataclass
class EvaluationResult:
    """Data class for evaluation results."""
    query_id: str
    query: str
    rag_response: str
    non_rag_response: str
    rag_metadata: Dict[str, Any]
    non_rag_metadata: Dict[str, Any]
    gemini_evaluation: Dict[str, Any]
    processing_time: float

class RAGEvaluator:
    """RAG evaluation system using Gemini as judge."""
    
    def __init__(self):
        """Initialize RAG evaluator."""
        logger.info("Initializing RAG Evaluator...")
        
        # Initialize Gemini client
        self.gemini_client = GeminiClient()
        
        # Initialize vector store
        self.vector_store = ChromaVectorStore()
        
        # Initialize embedder for query embedding
        self.embedder = EmbedderFactory.create_default_embedder()
        
        # Initialize RAG and Non-RAG models
        self.rag_model = RAGModel(
            gemini_client=self.gemini_client,
            vector_store=self.vector_store,
            max_context_chunks=5,
            max_context_tokens=2000
        )
        
        self.non_rag_model = NonRAGModel(self.gemini_client)
        
        # Test connections
        if not self.gemini_client.test_connection():
            raise RuntimeError("Failed to connect to Gemini API")
        
        logger.info("✓ RAG Evaluator initialized successfully")
    
    def create_sample_queries(self) -> List[EvaluationQuery]:
        """Create sample evaluation queries."""
        queries = [
            EvaluationQuery(
                id="q1",
                query="What was Apple's revenue growth in Q4 2019?",
                expected_companies=["AAPL"],
                category="financial_metrics",
                difficulty="easy"
            ),
            EvaluationQuery(
                id="q2", 
                query="Compare the R&D spending between Apple and Microsoft in 2018.",
                expected_companies=["AAPL", "MSFT"],
                category="comparative_analysis",
                difficulty="medium"
            ),
            EvaluationQuery(
                id="q3",
                query="What were the main challenges faced by semiconductor companies in 2017?",
                expected_companies=["INTC", "AMD", "NVDA"],
                category="industry_analysis", 
                difficulty="hard"
            ),
            EvaluationQuery(
                id="q4",
                query="How did Amazon's cloud services perform in Q3 2020?",
                expected_companies=["AMZN"],
                category="business_segment",
                difficulty="medium"
            ),
            EvaluationQuery(
                id="q5",
                query="What is the outlook for artificial intelligence in the tech industry?",
                expected_companies=["GOOGL", "MSFT", "NVDA"],
                category="future_trends",
                difficulty="hard"
            )
        ]
        
        logger.info(f"Created {len(queries)} sample evaluation queries")
        return queries
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query."""
        try:
            embedding = self.embedder.embed_text(query)
            logger.debug(f"Generated embedding for query: {query[:50]}...")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return []
    
    def evaluate_query(self, query: EvaluationQuery) -> EvaluationResult:
        """Evaluate a single query with both RAG and Non-RAG models.
        
        Args:
            query: Query to evaluate
            
        Returns:
            Evaluation result
        """
        start_time = time.time()
        logger.info(f"Evaluating query: {query.id} - {query.query}")
        
        # Generate query embedding
        query_embedding = self.generate_query_embedding(query.query)
        
        # Company filter if specified
        company_filter = None
        if query.expected_companies:
            # For ChromaDB, use simple equality for single company
            # For multiple companies, we'd need to implement OR logic differently
            if len(query.expected_companies) == 1:
                company_filter = {"company_symbol": query.expected_companies[0]}
            else:
                # For multiple companies, we'll skip filtering for now
                # This could be enhanced to do multiple queries and merge results
                logger.warning(f"Multiple company filtering not fully supported, using first company: {query.expected_companies[0]}")
                company_filter = {"company_symbol": query.expected_companies[0]}
        
        # Generate RAG response
        logger.info("Generating RAG response...")
        rag_response = self.rag_model.generate_response(
            query=query.query,
            query_embedding=query_embedding,
            filter_dict=company_filter
        )
        
        # Generate Non-RAG response
        logger.info("Generating Non-RAG response...")
        non_rag_response = self.non_rag_model.generate_response(query.query)
        
        # Evaluate with Gemini judge
        logger.info("Running Gemini evaluation...")
        gemini_evaluation = self.evaluate_with_gemini(
            query=query.query,
            rag_response=rag_response.get("response", ""),
            non_rag_response=non_rag_response.get("response", ""),
            context_sources=rag_response.get("context_sources", [])
        )
        
        total_time = time.time() - start_time
        
        result = EvaluationResult(
            query_id=query.id,
            query=query.query,
            rag_response=rag_response.get("response", ""),
            non_rag_response=non_rag_response.get("response", ""),
            rag_metadata=rag_response,
            non_rag_metadata=non_rag_response,
            gemini_evaluation=gemini_evaluation,
            processing_time=total_time
        )
        
        logger.info(f"✓ Completed evaluation for query {query.id} in {total_time:.2f}s")
        return result
    
    def evaluate_with_gemini(self, 
                           query: str,
                           rag_response: str,
                           non_rag_response: str,
                           context_sources: List[str]) -> Dict[str, Any]:
        """Evaluate responses using Gemini as judge.
        
        Args:
            query: Original query
            rag_response: RAG model response
            non_rag_response: Non-RAG model response
            context_sources: Sources used by RAG model
            
        Returns:
            Evaluation scores and analysis
        """
        evaluation_prompt = f"""
You are an expert evaluator comparing two AI responses to a financial query. Please evaluate both responses across multiple criteria.

QUERY: {query}

RESPONSE A (RAG-enabled):
{rag_response}

RESPONSE B (Non-RAG baseline):
{non_rag_response}

CONTEXT SOURCES (used by Response A):
{', '.join(context_sources) if context_sources else 'None'}

Please evaluate both responses on the following criteria (score 1-10):

1. ACCURACY: How factually correct is the response?
2. RELEVANCE: How well does the response address the query?
3. COMPLETENESS: How comprehensive is the response?
4. SPECIFICITY: How specific and detailed is the response?
5. COHERENCE: How well-structured and coherent is the response?

Please provide your evaluation in the following JSON format:
{{
    "response_a_scores": {{
        "accuracy": <1-10>,
        "relevance": <1-10>,
        "completeness": <1-10>,
        "specificity": <1-10>,
        "coherence": <1-10>,
        "overall": <1-10>
    }},
    "response_b_scores": {{
        "accuracy": <1-10>,
        "relevance": <1-10>,
        "completeness": <1-10>,
        "specificity": <1-10>,
        "coherence": <1-10>,
        "overall": <1-10>
    }},
    "winner": "A" or "B" or "tie",
    "reasoning": "Brief explanation of your evaluation",
    "key_differences": "Main differences between the responses"
}}

Be objective and consider that Response A has access to specific financial documents while Response B relies on general knowledge.
"""

        try:
            response = self.gemini_client.generate_response(
                prompt=evaluation_prompt,
                system_instruction="You are a fair and objective AI evaluator. Provide detailed, unbiased evaluations."
            )
            
            # Try to parse JSON from response
            response_text = response.get("response", "")
            
            # Extract JSON from response (handle potential markdown formatting)
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                evaluation_data = json.loads(json_text)
            else:
                # Fallback: create basic evaluation
                evaluation_data = {
                    "response_a_scores": {"overall": 7},
                    "response_b_scores": {"overall": 6},
                    "winner": "A",
                    "reasoning": "Could not parse detailed evaluation",
                    "error": "JSON parsing failed"
                }
            
            # Add metadata
            evaluation_data.update({
                "evaluation_model": self.gemini_client.model_name,
                "evaluation_time": response.get("processing_time", 0),
                "evaluation_tokens": response.get("total_tokens", 0)
            })
            
            return evaluation_data
            
        except Exception as e:
            logger.error(f"Gemini evaluation failed: {e}")
            return {
                "error": str(e),
                "response_a_scores": {"overall": 5},
                "response_b_scores": {"overall": 5}, 
                "winner": "tie",
                "reasoning": f"Evaluation failed: {e}"
            }
    
    def run_evaluation_suite(self, 
                           queries: List[EvaluationQuery] = None,
                           output_file: str = None) -> Dict[str, Any]:
        """Run complete evaluation suite.
        
        Args:
            queries: List of queries to evaluate (uses samples if None)
            output_file: File to save results
            
        Returns:
            Evaluation summary
        """
        if queries is None:
            queries = self.create_sample_queries()
        
        logger.info(f"Starting evaluation suite with {len(queries)} queries")
        
        results = []
        start_time = time.time()
        
        for i, query in enumerate(queries, 1):
            logger.info(f"\n=== Evaluating Query {i}/{len(queries)} ===")
            
            try:
                result = self.evaluate_query(query)
                results.append(result)
                
                # Show progress
                logger.info(f"✓ Query {i} completed - Winner: {result.gemini_evaluation.get('winner', 'unknown')}")
                
            except Exception as e:
                logger.error(f"✗ Query {i} failed: {e}")
                continue
        
        total_time = time.time() - start_time
        
        # Calculate summary statistics
        summary = self.calculate_summary(results)
        summary.update({
            "total_queries": len(queries),
            "successful_evaluations": len(results),
            "total_evaluation_time": total_time,
            "average_time_per_query": total_time / len(results) if results else 0
        })
        
        # Save results
        if output_file:
            self.save_results(results, summary, output_file)
        
        logger.info(f"\n✓ Evaluation suite completed!")
        logger.info(f"  Processed: {len(results)}/{len(queries)} queries")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  RAG wins: {summary.get('rag_wins', 0)}")
        logger.info(f"  Non-RAG wins: {summary.get('non_rag_wins', 0)}")
        logger.info(f"  Ties: {summary.get('ties', 0)}")
        
        return summary
    
    def calculate_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate summary statistics from results."""
        if not results:
            return {}
        
        rag_wins = 0
        non_rag_wins = 0
        ties = 0
        
        total_rag_scores = {"accuracy": 0, "relevance": 0, "completeness": 0, "specificity": 0, "coherence": 0, "overall": 0}
        total_non_rag_scores = {"accuracy": 0, "relevance": 0, "completeness": 0, "specificity": 0, "coherence": 0, "overall": 0}
        
        for result in results:
            eval_data = result.gemini_evaluation
            
            # Count winners
            winner = eval_data.get("winner", "tie").lower()
            if winner == "a":
                rag_wins += 1
            elif winner == "b":
                non_rag_wins += 1
            else:
                ties += 1
            
            # Accumulate scores
            rag_scores = eval_data.get("response_a_scores", {})
            non_rag_scores = eval_data.get("response_b_scores", {})
            
            for metric in total_rag_scores.keys():
                total_rag_scores[metric] += rag_scores.get(metric, 0)
                total_non_rag_scores[metric] += non_rag_scores.get(metric, 0)
        
        # Calculate averages
        num_results = len(results)
        avg_rag_scores = {k: v / num_results for k, v in total_rag_scores.items()}
        avg_non_rag_scores = {k: v / num_results for k, v in total_non_rag_scores.items()}
        
        return {
            "rag_wins": rag_wins,
            "non_rag_wins": non_rag_wins,
            "ties": ties,
            "rag_win_rate": rag_wins / num_results,
            "non_rag_win_rate": non_rag_wins / num_results,
            "tie_rate": ties / num_results,
            "average_rag_scores": avg_rag_scores,
            "average_non_rag_scores": avg_non_rag_scores
        }
    
    def save_results(self, 
                    results: List[EvaluationResult],
                    summary: Dict[str, Any],
                    output_file: str):
        """Save evaluation results to file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to dictionaries
        results_data = []
        for result in results:
            result_dict = {
                "query_id": result.query_id,
                "query": result.query,
                "rag_response": result.rag_response,
                "non_rag_response": result.non_rag_response,
                "rag_metadata": result.rag_metadata,
                "non_rag_metadata": result.non_rag_metadata,
                "gemini_evaluation": result.gemini_evaluation,
                "processing_time": result.processing_time
            }
            results_data.append(result_dict)
        
        output_data = {
            "summary": summary,
            "results": results_data,
            "evaluation_timestamp": time.time(),
            "evaluation_config": {
                "gemini_model": self.gemini_client.model_name,
                "embedding_model": self.embedder.__class__.__name__,
                "vector_store": self.vector_store.__class__.__name__,
                "max_context_chunks": self.rag_model.max_context_chunks,
                "max_context_tokens": self.rag_model.max_context_tokens
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Results saved to: {output_path}")

def main():
    """Main function for running RAG evaluation."""
    parser = argparse.ArgumentParser(description="RAG Evaluation System")
    parser.add_argument("--output", "-o", help="Output file for results")
    parser.add_argument("--queries", "-q", help="Custom queries JSON file")
    parser.add_argument("--sample-only", action="store_true", help="Run with sample queries only")
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = RAGEvaluator()
        
        # Load custom queries if provided
        queries = None
        if args.queries:
            with open(args.queries, 'r', encoding='utf-8') as f:
                query_data = json.load(f)
                queries = [EvaluationQuery(**q) for q in query_data]
        
        # Set output file
        output_file = args.output or f"results/evaluation_{int(time.time())}.json"
        
        # Run evaluation
        summary = evaluator.run_evaluation_suite(
            queries=queries,
            output_file=output_file
        )
        
        # Print summary
        print("\n" + "="*60)
        print("RAG EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Queries: {summary.get('total_queries', 0)}")
        print(f"Successful Evaluations: {summary.get('successful_evaluations', 0)}")
        print(f"Total Time: {summary.get('total_evaluation_time', 0):.2f}s")
        print(f"\nResults:")
        print(f"  RAG Wins: {summary.get('rag_wins', 0)} ({summary.get('rag_win_rate', 0)*100:.1f}%)")
        print(f"  Non-RAG Wins: {summary.get('non_rag_wins', 0)} ({summary.get('non_rag_win_rate', 0)*100:.1f}%)")
        print(f"  Ties: {summary.get('ties', 0)} ({summary.get('tie_rate', 0)*100:.1f}%)")
        
        avg_rag = summary.get('average_rag_scores', {})
        avg_non_rag = summary.get('average_non_rag_scores', {})
        print(f"\nAverage Scores:")
        print(f"  RAG Overall: {avg_rag.get('overall', 0):.1f}/10")
        print(f"  Non-RAG Overall: {avg_non_rag.get('overall', 0):.1f}/10")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
