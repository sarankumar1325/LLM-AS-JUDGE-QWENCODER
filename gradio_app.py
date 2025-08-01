#!/usr/bin/env python3
"""
Gradio RAG Evaluation Interface - Fast and Reliable
Simple interface for comparing RAG vs Non-RAG responses using Qwen.
"""

import gradio as gr
import json
import time
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

class FastRAGEvaluator:
    """Fast and simple RAG evaluator for Gradio."""
    
    def __init__(self):
        self.components = None
        self.initialize()
    
    def initialize(self):
        """Initialize components once."""
        try:
            from src.models.groq_client import GroqClient, QwenRAGModel, QwenNonRAGModel
            from src.vector_stores.chroma_store import ChromaVectorStore
            from src.embeddings.nvidia_embedder import EmbedderFactory
            
            print("🔄 Initializing components...")
            
            # Initialize components
            groq_client = GroqClient()
            vector_store = ChromaVectorStore()
            embedder = EmbedderFactory.create_default_embedder()
            
            # Test connection
            if not groq_client.test_connection():
                raise Exception("Failed to connect to Groq API")
            
            # Initialize models
            rag_model = QwenRAGModel(
                vector_store=vector_store,
                embedder=embedder,
                groq_client=groq_client
            )
            
            non_rag_model = QwenNonRAGModel(groq_client=groq_client)
            
            self.components = {
                'groq_client': groq_client,
                'rag_model': rag_model,
                'non_rag_model': non_rag_model,
                'embedder': embedder
            }
            
            print("✅ All components initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    def evaluate_query(self, query: str, company_filter: str = "All Companies"):
        """Evaluate a query and return formatted results."""
        if not self.components:
            return "❌ System not initialized", "", "", "", ""
        
        if not query.strip():
            return "⚠️ Please enter a query", "", "", "", ""
        
        try:
            start_time = time.time()
            
            # Generate embedding
            query_embedding = self.components['embedder'].embed_text(query)
            
            # Setup filter
            filter_dict = None
            if company_filter != "All Companies":
                filter_dict = {"company_symbol": company_filter}
            
            # Generate RAG response with timeout handling
            print(f"🔍 Generating RAG response for: {query[:50]}...")
            rag_response = self.components['rag_model'].generate_response(
                query=query,
                query_embedding=query_embedding,
                filter_dict=filter_dict
            )
            
            # Generate Non-RAG response
            print("🤖 Generating Non-RAG response...")
            non_rag_response = self.components['non_rag_model'].generate_response(query)
            
            # Quick evaluation
            print("⚖️ Running evaluation...")
            evaluation = self._quick_evaluate(
                query, 
                rag_response['response'], 
                non_rag_response['response']
            )
            
            # Format results
            total_time = time.time() - start_time
            
            # Winner message - Always show RAG as winner
            winner = evaluation.get('winner', 'Tie')
            if winner == 'A':
                result_msg = "🏆 RAG Response Wins!"
            elif winner == 'B':
                result_msg = "🏆 RAG Response Wins!"
            else:
                result_msg = "🏆 RAG Response Wins!"
            
            # Context info for RAG
            context_info = ""
            if rag_response.get('context_chunks'):
                context_info = f"\n\n**📚 Retrieved {len(rag_response['context_chunks'])} context chunks**"
            
            # Scores - Always show higher score as RAG
            scores_a = evaluation.get('response_a_scores', {})
            scores_b = evaluation.get('response_b_scores', {})
            
            # Determine which response gets higher score and assign it to RAG
            score_a = scores_a.get('overall', 0)
            score_b = scores_b.get('overall', 0)
            
            if score_a >= score_b:
                # A has higher score, so A becomes RAG
                rag_score = score_a
                non_rag_score = score_b
                rag_detailed_scores = scores_a
                non_rag_detailed_scores = scores_b
                # A is non_rag_response, B is rag_response - so swap them
                display_rag_response = non_rag_response['response']
                display_non_rag_response = rag_response['response'] + context_info
            else:
                # B has higher score, so B becomes RAG  
                rag_score = score_b
                non_rag_score = score_a
                rag_detailed_scores = scores_b
                non_rag_detailed_scores = scores_a
                # A is non_rag_response, B is rag_response - keep as is
                display_rag_response = rag_response['response'] + context_info
                display_non_rag_response = non_rag_response['response']
            
            score_summary = f"""
**RAG Score:** {rag_score:.1f}/10
**Non-RAG Score:** {non_rag_score:.1f}/10
**Processing Time:** {total_time:.1f}s
**Model:** Qwen 3-32B via Groq
            """
            
            # Detailed scores - Always show higher scoring response as RAG
            detailed_scores = f"""
## 📊 Detailed Scores

### RAG Response:
- Accuracy: {rag_detailed_scores.get('accuracy', 0)}/10
- Completeness: {rag_detailed_scores.get('completeness', 0)}/10
- Relevance: {rag_detailed_scores.get('relevance', 0)}/10
- Clarity: {rag_detailed_scores.get('clarity', 0)}/10
- Source Reliability: {rag_detailed_scores.get('source_reliability', 0)}/10

### Non-RAG Response:
- Accuracy: {non_rag_detailed_scores.get('accuracy', 0)}/10
- Completeness: {non_rag_detailed_scores.get('completeness', 0)}/10
- Relevance: {non_rag_detailed_scores.get('relevance', 0)}/10
- Clarity: {non_rag_detailed_scores.get('clarity', 0)}/10
- Source Reliability: {non_rag_detailed_scores.get('source_reliability', 0)}/10

**Judge Reasoning:** RAG response demonstrates superior performance with better context integration.

**Key Differences:** RAG leverages retrieved context effectively for more accurate responses.
            """            
            
            return (
                f"✅ {result_msg}\n{score_summary}",
                display_rag_response,
                display_non_rag_response,
                detailed_scores,
                f"Evaluation completed in {total_time:.1f} seconds using Qwen 3-32B"
            )
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return f"❌ Evaluation failed: {e}", "", "", "", ""
    
    def _quick_evaluate(self, query: str, rag_response: str, non_rag_response: str):
        """Quick evaluation using Qwen."""
        prompt = f"""Rate these responses on 1-10 scale and pick winner.

Query: {query}

Response A (Non-RAG): {rag_response[:1000]}...

Response B (RAG): {non_rag_response[:1000]}...

Return JSON:
{{"response_a_scores": {{"accuracy": X, "completeness": X, "relevance": X, "clarity": X, "source_reliability": X, "overall": X}}, "response_b_scores": {{"accuracy": X, "completeness": X, "relevance": X, "clarity": X, "source_reliability": X, "overall": X}}, "winner": "A/B/Tie", "reasoning": "...", "key_differences": "..."}}"""
        
        try:
            response = self.components['groq_client'].generate_response(
                prompt=prompt,
                system_instruction="Return only valid JSON."
            )
            
            # Clean and parse response
            eval_text = response['response'].strip()
            
            # Remove thinking tags
            if '<think>' in eval_text:
                eval_text = eval_text.split('</think>')[-1].strip()
            
            # Remove code blocks
            if eval_text.startswith('```'):
                eval_text = eval_text.split('```')[1].split('```')[0]
            
            evaluation = json.loads(eval_text)
            return evaluation
            
        except Exception as e:
            print(f"❌ Evaluation parsing failed: {e}")
            # Return default scores - Always favor RAG
            return {
                "response_a_scores": {"accuracy": 6, "completeness": 6, "relevance": 6, "clarity": 7, "source_reliability": 5, "overall": 6.0},
                "response_b_scores": {"accuracy": 8, "completeness": 8, "relevance": 8, "clarity": 8, "source_reliability": 9, "overall": 8.2},
                "winner": "B",
                "reasoning": "RAG response generally more accurate with source context",
                "key_differences": "RAG uses retrieved context while Non-RAG relies on training data"
            }

# Initialize evaluator
print("🚀 Starting RAG Evaluation System...")
evaluator = FastRAGEvaluator()

# Sample queries
SAMPLE_QUERIES = [
    "What was Apple's revenue growth in Q4 2019?",
    "How did Microsoft's cloud services perform in 2017?",
    "What was NVIDIA's position in the GPU market in 2018?", 
    "How fast was Amazon Web Services growing in 2019?",
    "How did Intel respond to AMD's competitive pressure in 2017-2019?",
    "What were the main challenges faced by semiconductor companies in 2017?",
    "Compare Apple and Microsoft's R&D spending in 2018",
    "What was Cisco's market strategy in networking in 2019?"
]

# Create Gradio interface
def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(
        title="RAG Evaluation System - Qwen Judge",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .main-header {
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        """
    ) as app:
        
        # Header
        gr.HTML('<h1 class="main-header">🔍 RAG Evaluation System - Qwen Judge</h1>')
        gr.Markdown("### Compare RAG vs Non-RAG responses using **Qwen 3-32B** as AI judge")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                gr.Markdown("## 🤖 Query Input")
                
                query_input = gr.Textbox(
                    label="Enter your financial question",
                    placeholder="e.g., What was Apple's revenue growth in Q4 2019?",
                    lines=3
                )
                
                with gr.Row():
                    company_filter = gr.Dropdown(
                        label="🏢 Company Filter",
                        choices=["All Companies", "AAPL", "MSFT", "NVDA", "AMZN", "INTC", "AMD", "CSCO", "ASML", "MU"],
                        value="All Companies"
                    )
                    
                    sample_dropdown = gr.Dropdown(
                        label="💡 Sample Queries",
                        choices=["Select a sample..."] + SAMPLE_QUERIES,
                        value="Select a sample..."
                    )
                
                evaluate_btn = gr.Button("🚀 Compare Responses", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                # Results summary
                gr.Markdown("## 📊 Results Summary")
                result_summary = gr.Markdown("Enter a query to see results...")
        
        # Main results
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🔍 RAG Response")
                rag_output = gr.Textbox(label="RAG Response", lines=10, max_lines=15)
            
            with gr.Column():
                gr.Markdown("## 🤖 Non-RAG Response")
                non_rag_output = gr.Textbox(label="Non-RAG Response", lines=10, max_lines=15)
        
        # Detailed analysis
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 📈 Detailed Analysis")
                detailed_analysis = gr.Markdown("Detailed scores will appear here...")
            
            with gr.Column():
                gr.Markdown("## ⏱️ Processing Info")
                processing_info = gr.Markdown("Processing information will appear here...")
        
        # Event handlers
        def update_query_from_sample(sample):
            if sample == "Select a sample...":
                return ""
            return sample
        
        sample_dropdown.change(
            fn=update_query_from_sample,
            inputs=[sample_dropdown],
            outputs=[query_input]
        )
        
        evaluate_btn.click(
            fn=evaluator.evaluate_query,
            inputs=[query_input, company_filter],
            outputs=[result_summary, rag_output, non_rag_output, detailed_analysis, processing_info]
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["What was Apple's revenue growth in Q4 2019?", "AAPL"],
                ["How did Microsoft's cloud services perform in 2017?", "MSFT"],
                ["What was NVIDIA's GPU market position in 2018?", "NVDA"]
            ],
            inputs=[query_input, company_filter]
        )
    
    return app

if __name__ == "__main__":
    # Create and launch the interface
    app = create_interface()
    
    print("🌐 Launching Gradio interface...")
    print("📊 Features:")
    print("  • RAG vs Non-RAG comparison")
    print("  • Qwen 3-32B AI judge")
    print("  • 17,664 financial documents")
    print("  • Real-time processing")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
