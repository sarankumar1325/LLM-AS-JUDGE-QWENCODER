"""
Main evaluator orchestrating the evaluation process.
"""

from typing import List, Dict, Any, Optional
from src.models.rag_model import RAGModel
from src.models.non_rag_model import NonRAGModel
from src.evaluation.gemini_judge import GeminiJudge
from src.evaluation.metrics import RAGMetrics
from src.utils.logger import app_logger

class RAGEvaluator:
    """Main evaluator for comparing RAG and non-RAG approaches."""
    
    def __init__(self,
                 rag_model: Optional[RAGModel] = None,
                 non_rag_model: Optional[NonRAGModel] = None,
                 judge: Optional[GeminiJudge] = None):
        
        self.rag_model = rag_model or RAGModel()
        self.non_rag_model = non_rag_model or NonRAGModel()
        self.judge = judge or GeminiJudge()
        self.metrics_calc = RAGMetrics()
    
    def evaluate_questions(self, 
                          questions: List[Dict[str, Any]],
                          evaluate_rag: bool = True,
                          evaluate_non_rag: bool = True) -> Dict[str, Any]:
        """Evaluate both RAG and non-RAG approaches on a set of questions."""
        
        results = {
            "rag_results": [],
            "non_rag_results": [],
            "comparisons": []
        }
        
        app_logger.info(f"Starting evaluation of {len(questions)} questions")
        
        for i, question_data in enumerate(questions):
            question = question_data.get("question", question_data.get("text", ""))
            question_id = question_data.get("id", f"q_{i}")
            
            app_logger.info(f"Evaluating question {i+1}/{len(questions)}: {question_id}")
            
            # Generate answers
            rag_answer = None
            non_rag_answer = None
            
            if evaluate_rag:
                try:
                    rag_answer = self.rag_model.generate_answer(question)
                    app_logger.debug(f"RAG answer generated for {question_id}")
                except Exception as e:
                    app_logger.error(f"Error generating RAG answer: {e}")
            
            if evaluate_non_rag:
                try:
                    non_rag_answer = self.non_rag_model.generate_answer(question)
                    app_logger.debug(f"Non-RAG answer generated for {question_id}")
                except Exception as e:
                    app_logger.error(f"Error generating non-RAG answer: {e}")
            
            # Evaluate answers
            if rag_answer:
                rag_evaluation = self.judge.evaluate_response(
                    question=question,
                    answer=rag_answer["answer"],
                    context=rag_answer.get("context_text", "")
                )
                rag_evaluation["question_id"] = question_id
                rag_evaluation["model_type"] = "rag"
                results["rag_results"].append(rag_evaluation)
            
            if non_rag_answer:
                non_rag_evaluation = self.judge.evaluate_response(
                    question=question,
                    answer=non_rag_answer["answer"]
                )
                non_rag_evaluation["question_id"] = question_id
                non_rag_evaluation["model_type"] = "non_rag"
                results["non_rag_results"].append(non_rag_evaluation)
            
            # Compare if both answers available
            if rag_answer and non_rag_answer:
                comparison = self.judge.compare_responses(
                    question=question,
                    answer1=rag_answer["answer"],
                    answer2=non_rag_answer["answer"],
                    context=rag_answer.get("context_text", ""),
                    model1_name="RAG",
                    model2_name="Non-RAG"
                )
                comparison["question_id"] = question_id
                results["comparisons"].append(comparison)
        
        # Calculate aggregate metrics
        if results["rag_results"]:
            results["rag_metrics"] = self.metrics_calc.calculate_llm_judge_metrics(
                results["rag_results"]
            )
        
        if results["non_rag_results"]:
            results["non_rag_metrics"] = self.metrics_calc.calculate_llm_judge_metrics(
                results["non_rag_results"]
            )
        
        app_logger.info("Evaluation completed successfully")
        return results
