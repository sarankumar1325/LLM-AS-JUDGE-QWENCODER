#!/usr/bin/env python3
"""
LLM-as-Judge Demo: RAG vs Non-RAG Query Examples
This script demonstrates the exact queries where each approach wins
"""

def demonstrate_rag_winning_queries():
    """Show queries where RAG typically wins"""
    
    rag_winners = [
        {
            "query": "What was Apple's exact revenue in Q4 2019?",
            "rag_response_sample": "According to Apple's Q4 2019 earnings report, total revenue was $64.04 billion, representing a 2% decline year-over-year from $66.0 billion in Q4 2018. iPhone revenue specifically was $33.36 billion.",
            "non_rag_response_sample": "Apple's Q4 2019 revenue was approximately $64 billion, showing a slight decline from the previous year due to declining iPhone sales in some markets.",
            "why_rag_wins": "RAG provides exact figures ($64.04B vs ~$64B), specific comparisons (2% decline), and precise iPhone revenue ($33.36B) with source attribution.",
            "judge_likely_reasoning": "RAG response provides precise financial data with exact figures that can be verified, while Non-RAG approximates. For financial queries, precision is critical.",
            "expected_scores": "RAG: 9.0, Non-RAG: 6.0"
        },
        
        {
            "query": "What did Tim Cook say about iPhone pricing strategy in Apple's Q2 2018 earnings call?",
            "rag_response_sample": "In the Q2 2018 earnings call, Tim Cook stated: 'We believe that iPhone X represents the future of the smartphone, and we're thrilled with the customer response.' He also mentioned: 'We're not in the business of making the most phones, we're in the business of making the best phones.'",
            "non_rag_response_sample": "Tim Cook generally emphasized Apple's focus on premium products and innovation over market share during earnings calls around that time. He typically discussed how iPhone X represented Apple's vision for the future.",
            "why_rag_wins": "RAG provides direct quotes with exact wording, while Non-RAG gives general paraphrasing without specific statements.",
            "judge_likely_reasoning": "RAG delivers verbatim quotes that can be verified, providing much higher accuracy and source reliability for this type of query.",
            "expected_scores": "RAG: 9.5, Non-RAG: 5.0"
        },
        
        {
            "query": "Compare AMD's and Intel's exact R&D spending percentages for 2018",
            "rag_response_sample": "Based on 2018 financial filings: AMD spent $1.43 billion on R&D (21.4% of revenue), while Intel spent $13.5 billion on R&D (19.2% of revenue). Despite Intel's higher absolute spending, AMD had a higher R&D intensity ratio.",
            "non_rag_response_sample": "In 2018, both AMD and Intel invested heavily in R&D. Intel typically spends more in absolute terms due to their larger revenue base, while AMD often has higher R&D as a percentage of revenue due to their focus on innovation to compete.",
            "why_rag_wins": "RAG provides exact percentages (21.4% vs 19.2%) and absolute figures ($1.43B vs $13.5B) with precise comparison.",
            "judge_likely_reasoning": "RAG offers specific, verifiable data that directly answers the query, while Non-RAG provides general trends without the requested exact percentages.",
            "expected_scores": "RAG: 9.0, Non-RAG: 4.0"
        }
    ]
    
    return rag_winners

def demonstrate_non_rag_winning_queries():
    """Show queries where Non-RAG typically wins"""
    
    non_rag_winners = [
        {
            "query": "How do economic cycles typically affect technology company valuations?",
            "rag_response_sample": "Looking at our dataset, Apple's valuation fluctuated during 2016-2019, with P/E ratios changing from 13.4 to 16.7. Microsoft showed similar patterns with revenue growth varying quarterly. NVIDIA experienced significant volatility...",
            "non_rag_response_sample": "Economic cycles significantly impact tech valuations through multiple mechanisms: During expansion phases, tech companies benefit from increased business investment in technology, higher consumer spending on devices, and multiple expansion as growth expectations rise. In contractions, tech stocks often see higher volatility due to their growth nature, but established companies with strong cash flows tend to be more resilient. Key factors include: interest rate sensitivity (affects discount rates), demand elasticity for tech products, and the counter-cyclical nature of digital transformation initiatives.",
            "why_non_rag_wins": "Non-RAG provides comprehensive economic theory and broader understanding, while RAG gets stuck on specific company examples without explaining underlying principles.",
            "judge_likely_reasoning": "The query asks for general economic principles. Non-RAG delivers comprehensive theoretical framework while RAG fragments into specific company examples without coherent synthesis.",
            "expected_scores": "RAG: 4.0, Non-RAG: 8.5"
        },
        
        {
            "query": "What are the key principles for evaluating technology companies as long-term investments?",
            "rag_response_sample": "Based on Apple's performance, key metrics include revenue growth (Apple grew from $215B to $260B), profit margins (Apple maintains 25%+ gross margins), and market position. Microsoft shows similar patterns with cloud growth. NVIDIA demonstrates the importance of innovation cycles...",
            "non_rag_response_sample": "Key principles for evaluating tech companies as long-term investments include: 1) Sustainable competitive advantages (network effects, switching costs, data moats), 2) Scalability and margin expansion potential, 3) Innovation pipeline and R&D effectiveness, 4) Market opportunity size and penetration, 5) Management quality and capital allocation, 6) Financial strength and cash generation, 7) Adaptability to technological shifts, 8) Regulatory and competitive risks. The focus should be on business model durability rather than short-term metrics.",
            "why_non_rag_wins": "Non-RAG provides systematic investment framework with universal principles, while RAG focuses narrowly on specific company examples without synthesizing general principles.",
            "judge_likely_reasoning": "Query seeks investment philosophy and general principles. Non-RAG delivers comprehensive framework applicable to any tech company, while RAG provides fragmented company examples.",
            "expected_scores": "RAG: 5.0, Non-RAG: 9.0"
        },
        
        {
            "query": "What are the latest AI trends affecting tech companies in 2024-2025?",
            "rag_response_sample": "Our dataset covers 2016-2019, so recent AI trends aren't well covered. However, we can see early mentions of machine learning in earnings calls, with companies like NVIDIA discussing GPU computing for AI workloads...",
            "non_rag_response_sample": "Key AI trends in 2024-2025 include: 1) Generative AI integration across enterprise applications, 2) Edge AI deployment for real-time processing, 3) AI chip specialization and custom silicon, 4) Multimodal AI capabilities combining text, image, and voice, 5) AI governance and regulatory compliance frameworks, 6) Energy-efficient AI computing, 7) AI-human collaboration interfaces, 8) Federated learning for privacy-preserving AI. Major tech companies are pivoting business models around AI-first strategies.",
            "why_non_rag_wins": "Non-RAG has current knowledge about 2024-2025 trends, while RAG is limited to historical 2016-2019 data and cannot provide recent information.",
            "judge_likely_reasoning": "Query explicitly asks for recent trends. Non-RAG provides current, relevant information while RAG acknowledges its temporal limitations with outdated data.",
            "expected_scores": "RAG: 3.0, Non-RAG: 9.0"
        }
    ]
    
    return non_rag_winners

def demonstrate_llm_as_judge_process():
    """Show how the LLM judge evaluates responses"""
    
    judge_process = {
        "evaluation_criteria": [
            "Accuracy (1-10): Factual correctness and precision",
            "Completeness (1-10): How fully the query is answered", 
            "Relevance (1-10): Direct relevance to the question",
            "Clarity (1-10): Clear, understandable presentation",
            "Source Reliability (1-10): Trustworthiness of information"
        ],
        
        "judge_prompt_template": """
You are an expert judge. Rate these responses on 1-10 scale and pick the winner.

Query: {query}

Response A (RAG-Enhanced): {rag_response}

Response B (Non-RAG): {non_rag_response}

Consider: Accuracy, Completeness, Relevance, Clarity, Source Reliability

Return JSON with scores and reasoning.
        """,
        
        "sample_judge_response": {
            "response_a_scores": {
                "accuracy": 9,
                "completeness": 8, 
                "relevance": 9,
                "clarity": 8,
                "source_reliability": 9,
                "overall": 8.6
            },
            "response_b_scores": {
                "accuracy": 6,
                "completeness": 7,
                "relevance": 8, 
                "clarity": 7,
                "source_reliability": 6,
                "overall": 6.8
            },
            "winner": "A",
            "reasoning": "Response A provides specific, verifiable financial data with exact figures that directly answer the query, while Response B offers approximations. For financial queries requiring precision, Response A's accuracy and source reliability make it superior."
        }
    }
    
    return judge_process

def main():
    """Demonstrate the LLM-as-Judge evaluation system"""
    
    print("🤖 LLM-as-Judge Demo: RAG vs Non-RAG Evaluation")
    print("=" * 60)
    
    print("\n🏆 QUERIES WHERE RAG WINS:")
    print("-" * 40)
    rag_winners = demonstrate_rag_winning_queries()
    
    for i, example in enumerate(rag_winners, 1):
        print(f"\n{i}. Query: \"{example['query']}\"")
        print(f"   Why RAG Wins: {example['why_rag_wins']}")
        print(f"   Expected Scores: {example['expected_scores']}")
        print(f"   Judge Reasoning: {example['judge_likely_reasoning']}")
    
    print("\n\n🌟 QUERIES WHERE NON-RAG WINS:")
    print("-" * 40)
    non_rag_winners = demonstrate_non_rag_winning_queries()
    
    for i, example in enumerate(non_rag_winners, 1):
        print(f"\n{i}. Query: \"{example['query']}\"")
        print(f"   Why Non-RAG Wins: {example['why_non_rag_wins']}")
        print(f"   Expected Scores: {example['expected_scores']}")
        print(f"   Judge Reasoning: {example['judge_likely_reasoning']}")
    
    print("\n\n⚖️ LLM-as-JUDGE EVALUATION PROCESS:")
    print("-" * 40)
    judge_process = demonstrate_llm_as_judge_process()
    
    print("\nEvaluation Criteria:")
    for criterion in judge_process['evaluation_criteria']:
        print(f"  • {criterion}")
    
    print(f"\nSample Judge Response:")
    sample = judge_process['sample_judge_response']
    print(f"  • Winner: Response {sample['winner']}")
    print(f"  • RAG Score: {sample['response_a_scores']['overall']}")
    print(f"  • Non-RAG Score: {sample['response_b_scores']['overall']}")
    print(f"  • Reasoning: {sample['reasoning']}")
    
    print("\n" + "=" * 60)
    print("🚀 YOUR DOCKERIZED SYSTEM PROVIDES:")
    print("   ✅ Interactive web interface for these comparisons")
    print("   ✅ Automated LLM-as-Judge evaluation") 
    print("   ✅ Detailed scoring with reasoning")
    print("   ✅ 17,664 financial documents for testing")
    print("   ✅ Easy deployment with Docker")
    print("\n🌐 Access: http://localhost:7860 after running docker-compose up -d")

if __name__ == "__main__":
    main()
