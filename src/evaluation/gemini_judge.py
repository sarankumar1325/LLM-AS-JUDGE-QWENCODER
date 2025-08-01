"""
Gemini-based judge for evaluating model responses.
"""

from typing import Dict, Any, List, Optional
import json
import re
from src.models.llm_interface import GeminiLLM
from config.model_configs import EVALUATION_PROMPTS
from config.settings import settings

class GeminiJudge:
    """LLM-as-Judge using Google Gemini for evaluation."""
    
    def __init__(self, 
                 model_name: str = None,
                 api_key: str = None):
        self.llm = GeminiLLM(
            model_name=model_name or settings.DEFAULT_LLM_MODEL,
            api_key=api_key or settings.GEMINI_API_KEY
        )
        self.evaluation_criteria = [
            "relevance",
            "accuracy", 
            "completeness",
            "helpfulness"
        ]
    
    def evaluate_response(self,
                         question: str,
                         answer: str,
                         context: str = "",
                         criteria: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate a single response across multiple criteria."""
        
        criteria = criteria or self.evaluation_criteria
        evaluation_results = {}
        
        for criterion in criteria:
            if criterion not in EVALUATION_PROMPTS:
                print(f"Warning: No prompt template for criterion '{criterion}'")
                continue
            
            try:
                score, explanation = self._evaluate_single_criterion(
                    question, answer, context, criterion
                )
                evaluation_results[criterion] = {
                    "score": score,
                    "explanation": explanation
                }
            except Exception as e:
                print(f"Error evaluating {criterion}: {e}")
                evaluation_results[criterion] = {
                    "score": 0,
                    "explanation": f"Error during evaluation: {e}"
                }
        
        # Calculate overall score
        valid_scores = [
            result["score"] for result in evaluation_results.values()
            if isinstance(result["score"], (int, float)) and result["score"] > 0
        ]
        
        overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        
        return {
            "overall_score": overall_score,
            "criteria_scores": evaluation_results,
            "question": question,
            "answer": answer,
            "context": context
        }
    
    def _evaluate_single_criterion(self,
                                  question: str,
                                  answer: str,
                                  context: str,
                                  criterion: str) -> tuple[int, str]:
        """Evaluate response for a single criterion."""
        
        # Get prompt template
        prompt_template = EVALUATION_PROMPTS[criterion]
        
        # Format prompt
        prompt = prompt_template.format(
            question=question,
            answer=answer,
            context=context if context else "No additional context provided."
        )
        
        # Get evaluation from Gemini
        response = self.llm.generate(
            prompt,
            temperature=0.1,
            max_tokens=500
        )
        
        # Parse response to extract score and explanation
        score, explanation = self._parse_evaluation_response(response)
        
        return score, explanation
    
    def _parse_evaluation_response(self, response: str) -> tuple[int, str]:
        """Parse the LLM evaluation response to extract score and explanation."""
        
        # Look for score patterns
        score_patterns = [
            r"(?:rating|score):\s*(\d+)",
            r"(\d+)\s*out of 5",
            r"(\d+)/5",
            r"^(\d+)\s*[-:]",
            r"Score:\s*(\d+)"
        ]
        
        score = 0
        for pattern in score_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    score = int(match.group(1))
                    if 1 <= score <= 5:
                        break
                except ValueError:
                    continue
        
        # If no valid score found, try to infer from text
        if score == 0:
            response_lower = response.lower()
            if any(word in response_lower for word in ["excellent", "perfect", "outstanding"]):
                score = 5
            elif any(word in response_lower for word in ["good", "mostly", "largely"]):
                score = 4
            elif any(word in response_lower for word in ["adequate", "somewhat", "partially"]):
                score = 3
            elif any(word in response_lower for word in ["poor", "limited", "minimal"]):
                score = 2
            elif any(word in response_lower for word in ["very poor", "completely", "totally"]):
                score = 1
            else:
                score = 3  # Default to middle score
        
        # Clean up explanation
        explanation = response.strip()
        
        return score, explanation
    
    def evaluate_batch(self,
                      evaluations: List[Dict[str, Any]],
                      criteria: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Evaluate multiple responses."""
        
        results = []
        
        for i, eval_data in enumerate(evaluations):
            print(f"Evaluating response {i+1}/{len(evaluations)}")
            
            try:
                result = self.evaluate_response(
                    question=eval_data["question"],
                    answer=eval_data["answer"],
                    context=eval_data.get("context", ""),
                    criteria=criteria
                )
                
                # Add original metadata
                result.update({
                    "response_id": eval_data.get("response_id", f"response_{i}"),
                    "model_type": eval_data.get("model_type", "unknown")
                })
                
                results.append(result)
                
            except Exception as e:
                print(f"Error evaluating response {i}: {e}")
                results.append({
                    "response_id": eval_data.get("response_id", f"response_{i}"),
                    "model_type": eval_data.get("model_type", "unknown"),
                    "overall_score": 0,
                    "criteria_scores": {},
                    "error": str(e)
                })
        
        return results
    
    def compare_responses(self,
                         question: str,
                         answer1: str,
                         answer2: str,
                         context: str = "",
                         model1_name: str = "Model A",
                         model2_name: str = "Model B") -> Dict[str, Any]:
        """Compare two responses to the same question."""
        
        comparison_prompt = f"""
        You are an expert evaluator. Please compare two AI responses to the same question.
        
        Question: {question}
        
        Context (if applicable): {context if context else "No additional context provided."}
        
        {model1_name} Response:
        {answer1}
        
        {model2_name} Response:
        {answer2}
        
        Please evaluate both responses on:
        1. Relevance to the question
        2. Accuracy of information
        3. Completeness of the answer
        4. Overall helpfulness
        
        Provide:
        1. A detailed comparison
        2. Which response is better and why
        3. Scores for each response (1-5 scale) for each criterion
        4. An overall winner
        
        Format your response clearly with sections for comparison, scores, and conclusion.
        """
        
        try:
            comparison_result = self.llm.generate(
                comparison_prompt,
                temperature=0.1,
                max_tokens=1000
            )
            
            return {
                "question": question,
                "model1_name": model1_name,
                "model2_name": model2_name,
                "answer1": answer1,
                "answer2": answer2,
                "context": context,
                "comparison": comparison_result
            }
            
        except Exception as e:
            return {
                "question": question,
                "model1_name": model1_name,
                "model2_name": model2_name,
                "error": str(e)
            }
