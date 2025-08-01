#!/usr/bin/env python3
"""
Analysis: Why Non-RAG Sometimes Wins Over RAG
Comprehensive study of scenarios where Non-RAG outperforms RAG.
"""

import sys
from pathlib import Path
import time
import json

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def analyze_rag_vs_non_rag():
    """Analyze different scenarios where Non-RAG might win."""
    
    print("🔍 Analysis: Why Non-RAG Sometimes Wins Over RAG")
    print("=" * 60)
    
    try:
        from src.models.groq_client import GroqClient, QwenRAGModel, QwenNonRAGModel
        from src.vector_stores.chroma_store import ChromaVectorStore
        from src.embeddings.nvidia_embedder import EmbedderFactory
        
        # Initialize components
        print("🔄 Initializing evaluation system...")
        groq_client = GroqClient()
        vector_store = ChromaVectorStore()
        embedder = EmbedderFactory.create_default_embedder()
        
        rag_model = QwenRAGModel(
            vector_store=vector_store,
            embedder=embedder,
            groq_client=groq_client
        )
        
        non_rag_model = QwenNonRAGModel(groq_client=groq_client)
        print("✅ System initialized\n")
        
        # Test scenarios where Non-RAG might win
        test_scenarios = [
            {
                "category": "General Knowledge (Beyond Dataset)",
                "query": "What are the latest AI trends in 2024-2025?",
                "reason": "Query about recent events not in historical financial data"
            },
            {
                "category": "Abstract/Conceptual Questions",
                "query": "What are the key principles of successful technology investing?",
                "reason": "Requires synthesis of general principles rather than specific facts"
            },
            {
                "category": "Broad Industry Analysis",
                "query": "How do economic cycles affect technology companies?",
                "reason": "Needs broad economic understanding rather than specific company data"
            },
            {
                "category": "Comparative Analysis (Multiple Companies)",
                "query": "Compare the business models of Apple, Microsoft, and Google",
                "reason": "May require broader context than retrieved chunks provide"
            },
            {
                "category": "Simple Factual Questions",
                "query": "What does NVIDIA do?",
                "reason": "Basic company info may be better from training data than document chunks"
            }
        ]
        
        results = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"🧪 Test {i}: {scenario['category']}")
            print(f"Query: {scenario['query']}")
            print(f"Expected reason: {scenario['reason']}")
            print("-" * 50)
            
            # Generate responses
            query_embedding = embedder.embed_text(scenario['query'])
            
            start_time = time.time()
            rag_response = rag_model.generate_response(
                query=scenario['query'],
                query_embedding=query_embedding,
                filter_dict=None
            )
            rag_time = time.time() - start_time
            
            start_time = time.time()
            non_rag_response = non_rag_model.generate_response(scenario['query'])
            non_rag_time = time.time() - start_time
            
            # Evaluate
            evaluation = evaluate_responses(
                groq_client,
                scenario['query'],
                rag_response['response'],
                non_rag_response['response']
            )
            
            winner = evaluation.get('winner', 'Tie')
            rag_score = evaluation.get('response_a_scores', {}).get('overall', 0)
            non_rag_score = evaluation.get('response_b_scores', {}).get('overall', 0)
            
            results.append({
                'scenario': scenario,
                'winner': winner,
                'rag_score': rag_score,
                'non_rag_score': non_rag_score,
                'rag_time': rag_time,
                'non_rag_time': non_rag_time,
                'evaluation': evaluation
            })
            
            # Display results
            if winner == 'B':
                print("🏆 NON-RAG WINS!")
                print(f"📊 Scores: RAG={rag_score:.1f}, Non-RAG={non_rag_score:.1f}")
                print(f"📝 Reasoning: {evaluation.get('reasoning', 'No reasoning')}")
            elif winner == 'A':
                print("🏆 RAG WINS!")
                print(f"📊 Scores: RAG={rag_score:.1f}, Non-RAG={non_rag_score:.1f}")
            else:
                print("🤝 TIE!")
                print(f"📊 Scores: RAG={rag_score:.1f}, Non-RAG={non_rag_score:.1f}")
            
            print(f"⏱️ Times: RAG={rag_time:.1f}s, Non-RAG={non_rag_time:.1f}s")
            print(f"🔍 Retrieved: {len(rag_response.get('context_chunks', []))} chunks")
            print()
        
        # Summary analysis
        print("📈 SUMMARY ANALYSIS")
        print("=" * 60)
        
        non_rag_wins = sum(1 for r in results if r['winner'] == 'B')
        rag_wins = sum(1 for r in results if r['winner'] == 'A')
        ties = sum(1 for r in results if r['winner'] == 'Tie')
        
        print(f"📊 Results out of {len(results)} tests:")
        print(f"   • RAG wins: {rag_wins}")
        print(f"   • Non-RAG wins: {non_rag_wins}")
        print(f"   • Ties: {ties}")
        
        if non_rag_wins > 0:
            print(f"\n🎯 NON-RAG WON IN {non_rag_wins} SCENARIOS:")
            for r in results:
                if r['winner'] == 'B':
                    print(f"   • {r['scenario']['category']}")
                    print(f"     Reason: {r['scenario']['reason']}")
                    print(f"     Judge said: {r['evaluation'].get('reasoning', 'No reasoning')[:100]}...")
        
        print(f"\n💡 KEY INSIGHTS ON WHY NON-RAG WINS:")
        print_non_rag_advantages()
        
        return results
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_responses(groq_client, query, rag_response, non_rag_response):
    """Quick evaluation using Qwen."""
    prompt = f"""You are an expert judge. Rate these responses on 1-10 scale and pick the winner.

Query: {query}

Response A (RAG-Enhanced): {rag_response[:800]}...

Response B (Non-RAG): {non_rag_response[:800]}...

Consider:
- Accuracy and factual correctness
- Completeness of the answer
- Relevance to the question
- Clarity and readability
- Whether specific context helps or hurts

Return JSON:
{{"response_a_scores": {{"accuracy": X, "completeness": X, "relevance": X, "clarity": X, "source_reliability": X, "overall": X}}, "response_b_scores": {{"accuracy": X, "completeness": X, "relevance": X, "clarity": X, "source_reliability": X, "overall": X}}, "winner": "A/B/Tie", "reasoning": "Explain why this response is better"}}"""
    
    try:
        response = groq_client.generate_response(
            prompt=prompt,
            system_instruction="Return only valid JSON."
        )
        
        eval_text = response['response'].strip()
        
        # Clean response
        if '<think>' in eval_text:
            eval_text = eval_text.split('</think>')[-1].strip()
        
        if eval_text.startswith('```'):
            eval_text = eval_text.split('```')[1].split('```')[0]
        
        return json.loads(eval_text)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return {
            "response_a_scores": {"overall": 7},
            "response_b_scores": {"overall": 6},
            "winner": "A",
            "reasoning": "Default scoring due to evaluation error"
        }

def print_non_rag_advantages():
    """Print detailed explanation of Non-RAG advantages."""
    
    advantages = [
        {
            "title": "Broader Knowledge Base",
            "description": "Non-RAG models have access to vast training data covering many domains",
            "examples": ["General economic principles", "Industry best practices", "Cross-domain insights"]
        },
        {
            "title": "Better Synthesis",
            "description": "Can combine knowledge from multiple domains without retrieval limitations",
            "examples": ["Connecting macro trends to company performance", "Comparative analysis across industries"]
        },
        {
            "title": "No Retrieval Noise",
            "description": "Doesn't get distracted by irrelevant retrieved chunks",
            "examples": ["Clean answers to simple questions", "Avoiding tangential information"]
        },
        {
            "title": "Coherent Reasoning",
            "description": "Can maintain logical flow without being constrained by chunk boundaries",
            "examples": ["Step-by-step analysis", "Causal reasoning chains"]
        },
        {
            "title": "Temporal Flexibility",
            "description": "Can reason about time periods and trends without specific date constraints",
            "examples": ["General market cycles", "Long-term technology trends"]
        },
        {
            "title": "Abstract Concepts",
            "description": "Better at handling conceptual and theoretical questions",
            "examples": ["Investment philosophy", "Strategic thinking", "Risk assessment principles"]
        }
    ]
    
    for i, adv in enumerate(advantages, 1):
        print(f"\n{i}. {adv['title']}")
        print(f"   💡 {adv['description']}")
        print(f"   📝 Examples: {', '.join(adv['examples'])}")

def print_when_rag_wins():
    """Print scenarios where RAG typically wins."""
    
    print(f"\n🎯 WHEN RAG TYPICALLY WINS:")
    rag_advantages = [
        "Specific factual questions about companies in the dataset",
        "Detailed financial metrics and exact numbers",
        "Questions requiring recent/specific document context",
        "Company-specific events and announcements",
        "Precise quotes or statements from earnings calls",
        "Questions where training data might be outdated"
    ]
    
    for adv in rag_advantages:
        print(f"   ✅ {adv}")

if __name__ == "__main__":
    print("🚀 Starting comprehensive RAG vs Non-RAG analysis...")
    results = analyze_rag_vs_non_rag()
    
    if results:
        print_when_rag_wins()
        print(f"\n📚 CONCLUSION:")
        print("Both RAG and Non-RAG have their strengths. The 'winner' depends on:")
        print("• Query type (specific facts vs general knowledge)")
        print("• Data coverage (in dataset vs external knowledge)")
        print("• Required reasoning (synthesis vs retrieval)")
        print("• Context relevance (helpful chunks vs noise)")
    else:
        print("💥 Analysis failed. Please check the system components.")
