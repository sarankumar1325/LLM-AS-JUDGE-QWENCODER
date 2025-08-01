"""
Metrics calculation for RAG evaluation.
"""

from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class RAGMetrics:
    """Calculate various metrics for RAG evaluation."""
    
    def __init__(self):
        pass
    
    def calculate_llm_judge_metrics(self, evaluations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate metrics from LLM judge evaluations."""
        if not evaluations:
            return {}
        
        # Extract scores
        overall_scores = [eval_result.get("overall_score", 0) for eval_result in evaluations]
        
        # Calculate basic statistics
        metrics = {
            "mean_score": np.mean(overall_scores),
            "median_score": np.median(overall_scores),
            "std_score": np.std(overall_scores),
            "min_score": np.min(overall_scores),
            "max_score": np.max(overall_scores)
        }
        
        # Calculate criteria-specific scores
        criteria_scores = {}
        for criterion in ["relevance", "accuracy", "completeness", "helpfulness"]:
            scores = []
            for eval_result in evaluations:
                criteria_result = eval_result.get("criteria_scores", {}).get(criterion, {})
                if isinstance(criteria_result, dict) and "score" in criteria_result:
                    scores.append(criteria_result["score"])
            
            if scores:
                criteria_scores[f"{criterion}_mean"] = np.mean(scores)
                criteria_scores[f"{criterion}_std"] = np.std(scores)
        
        metrics.update(criteria_scores)
        return metrics
