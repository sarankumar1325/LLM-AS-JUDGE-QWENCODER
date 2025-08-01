#!/usr/bin/env python3
"""
Streamlit RAG Evaluation App - Clean Version with Qwen Judge
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import time
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Page configuration
st.set_page_config(
    page_title="RAG Evaluation - Qwen Judge",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #667eea;
    margin: 0.5rem 0;
}
.winner-rag {
    background-color: #d4edda;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #28a745;
}
.winner-non-rag {
    background-color: #f8d7da;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #dc3545;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'evaluation_history' not in st.session_state:
    st.session_state.evaluation_history = []
if 'current_evaluation' not in st.session_state:
    st.session_state.current_evaluation = None
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = None

class SimpleRAGEvaluator:
    """Simple RAG evaluation system using Qwen."""
    
    def __init__(self):
        self.components = None
        self.setup_components()
    
    def setup_components(self):
        """Setup all components."""
        if self.components is not None:
            return self.components
            
        try:
            from src.models.groq_client import GroqClient, QwenRAGModel, QwenNonRAGModel
            from src.vector_stores.chroma_store import ChromaVectorStore
            from src.embeddings.nvidia_embedder import EmbedderFactory
            
            # Initialize components
            groq_client = GroqClient()
            vector_store = ChromaVectorStore()
            embedder = EmbedderFactory.create_default_embedder()
            
            # Test connection
            if not groq_client.test_connection():
                st.error("❌ Failed to connect to Groq API")
                return None
            
            # Initialize models
            rag_model = QwenRAGModel(
                vector_store=vector_store,
                embedder=embedder,
                groq_client=groq_client
            )
            
            non_rag_model = QwenNonRAGModel(groq_client=groq_client)
            
            self.components = {
                'groq_client': groq_client,
                'vector_store': vector_store,
                'embedder': embedder,
                'rag_model': rag_model,
                'non_rag_model': non_rag_model
            }
            
            return self.components
            
        except Exception as e:
            st.error(f"❌ Failed to initialize components: {e}")
            return None
    
    def evaluate_query(self, query: str, company_filter: str = None):
        """Evaluate a query with RAG vs Non-RAG."""
        if not self.components:
            return None
        
        try:
            # Generate embedding
            query_embedding = self.components['embedder'].embed_text(query)
            
            # Setup filter
            filter_dict = None
            if company_filter and company_filter != "All Companies":
                filter_dict = {"company_symbol": company_filter}
            
            # Generate responses
            rag_response = self.components['rag_model'].generate_response(
                query=query,
                query_embedding=query_embedding,
                filter_dict=filter_dict
            )
            
            non_rag_response = self.components['non_rag_model'].generate_response(query)
            
            # Evaluate with Qwen
            evaluation = self.evaluate_with_qwen(
                query, 
                rag_response['response'], 
                non_rag_response['response']
            )
            
            return {
                'query': query,
                'rag_response': rag_response,
                'non_rag_response': non_rag_response,
                'evaluation': evaluation,
                'company_filter': company_filter,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            st.error(f"❌ Evaluation failed: {e}")
            return None
    
    def evaluate_with_qwen(self, query: str, rag_response: str, non_rag_response: str):
        """Evaluate responses using Qwen."""
        prompt = f"""You are an expert AI judge. Rate each response on a scale of 1-10 and determine the winner.

**Query:** {query}

**Response A (RAG):** {rag_response}

**Response B (Non-RAG):** {non_rag_response}

Return only this JSON format:
{{
    "response_a_scores": {{
        "accuracy": score,
        "completeness": score,
        "relevance": score,
        "clarity": score,
        "source_reliability": score,
        "overall": average_score
    }},
    "response_b_scores": {{
        "accuracy": score,
        "completeness": score,
        "relevance": score,
        "clarity": score,
        "source_reliability": score,
        "overall": average_score
    }},
    "winner": "A" or "B" or "Tie",
    "reasoning": "Brief explanation",
    "key_differences": "Main differences"
}}"""
        
        try:
            response = self.components['groq_client'].generate_response(
                prompt=prompt,
                system_instruction="Return only valid JSON."
            )
            
            # Clean response
            evaluation_json = response['response'].strip()
            
            if '<think>' in evaluation_json:
                parts = evaluation_json.split('</think>')
                if len(parts) > 1:
                    evaluation_json = parts[1].strip()
            
            if evaluation_json.startswith('```json'):
                evaluation_json = evaluation_json.split('```json')[1].split('```')[0]
            elif evaluation_json.startswith('```'):
                evaluation_json = evaluation_json.split('```')[1].split('```')[0]
            
            evaluation = json.loads(evaluation_json)
            evaluation['evaluation_model'] = 'qwen/qwen3-32b'
            
            return evaluation
            
        except Exception as e:
            st.error(f"❌ Evaluation failed: {e}")
            return None

def create_radar_chart(scores_a, scores_b):
    """Create comparison radar chart."""
    categories = ['Accuracy', 'Completeness', 'Relevance', 'Clarity', 'Source Reliability']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[scores_a.get('accuracy', 0), scores_a.get('completeness', 0), 
           scores_a.get('relevance', 0), scores_a.get('clarity', 0), 
           scores_a.get('source_reliability', 0)],
        theta=categories,
        fill='toself',
        name='RAG Response',
        line_color='#2E86C1'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[scores_b.get('accuracy', 0), scores_b.get('completeness', 0), 
           scores_b.get('relevance', 0), scores_b.get('clarity', 0), 
           scores_b.get('source_reliability', 0)],
        theta=categories,
        fill='toself',
        name='Non-RAG Response',
        line_color='#E74C3C'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True,
        title="Response Quality Comparison"
    )
    
    return fig

def main():
    """Main app."""
    st.markdown('<h1 class="main-header">🔍 RAG Evaluation System - Qwen Judge</h1>', unsafe_allow_html=True)
    st.markdown("### Compare RAG vs Non-RAG responses using **Qwen 3-32B** as AI judge")
    st.markdown("---")
    
    # Initialize evaluator
    if st.session_state.evaluator is None:
        with st.spinner("🔄 Initializing system..."):
            st.session_state.evaluator = SimpleRAGEvaluator()
    
    evaluator = st.session_state.evaluator
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        companies = ["All Companies", "AAPL", "MSFT", "NVDA", "AMZN", "INTC", "AMD", "CSCO", "ASML", "MU"]
        selected_company = st.selectbox("📊 Company Filter", companies)
        
        st.header("💡 Sample Queries")
        samples = {
            "Apple Revenue Growth": "What was Apple's revenue growth in Q4 2019?",
            "Microsoft Cloud": "How did Microsoft's cloud services perform in 2017?",
            "NVIDIA GPU Market": "What was NVIDIA's position in the GPU market in 2018?",
            "Amazon AWS": "How fast was Amazon Web Services growing in 2019?",
            "Intel vs AMD": "How did Intel respond to AMD's competitive pressure in 2017-2019?"
        }
        
        for title, query in samples.items():
            if st.button(f"💡 {title}", key=f"sample_{title}"):
                st.session_state.sample_query = query
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🤖 Query Input")
        
        query = st.text_area(
            "Enter your financial question:",
            value=st.session_state.get('sample_query', ''),
            height=100,
            placeholder="e.g., What was Apple's revenue growth in Q4 2019?"
        )
        
        if st.button("🚀 Compare Responses", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("🔄 Processing evaluation..."):
                    result = evaluator.evaluate_query(
                        query=query.strip(),
                        company_filter=selected_company if selected_company != "All Companies" else None
                    )
                
                if result:
                    st.session_state.current_evaluation = result
                    st.session_state.evaluation_history.append(result)
                    st.success("✅ Evaluation completed!")
                    st.rerun()
                else:
                    st.error("❌ Evaluation failed. Please try again.")
            else:
                st.warning("Please enter a query to evaluate.")
    
    with col2:
        st.header("📊 Quick Stats")
        
        if st.session_state.evaluation_history:
            total = len(st.session_state.evaluation_history)
            rag_wins = sum(1 for eval in st.session_state.evaluation_history 
                          if eval.get('evaluation', {}).get('winner') == 'A')
            
            st.metric("Total Evaluations", total)
            st.metric("RAG Wins", rag_wins)
            st.metric("Non-RAG Wins", total - rag_wins)
            
            if total > 0:
                win_rate = (rag_wins / total) * 100
                st.metric("RAG Win Rate", f"{win_rate:.1f}%")
    
    # Display results
    if st.session_state.current_evaluation:
        result = st.session_state.current_evaluation
        evaluation = result.get('evaluation', {})
        
        st.markdown("---")
        st.header("🎯 Evaluation Results")
        
        # Winner
        winner = evaluation.get('winner', 'Tie')
        if winner == 'A':
            st.markdown('<div class="winner-rag">🏆 <strong>RAG Response Wins!</strong></div>', unsafe_allow_html=True)
        elif winner == 'B':
            st.markdown('<div class="winner-non-rag">🏆 <strong>Non-RAG Response Wins!</strong></div>', unsafe_allow_html=True)
        else:
            st.info("🤝 **It's a Tie!**")
        
        # Scores
        scores_a = evaluation.get('response_a_scores', {})
        scores_b = evaluation.get('response_b_scores', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RAG Score", f"{scores_a.get('overall', 0):.1f}/10")
        with col2:
            st.metric("Non-RAG Score", f"{scores_b.get('overall', 0):.1f}/10")
        with col3:
            diff = abs(scores_a.get('overall', 0) - scores_b.get('overall', 0))
            st.metric("Difference", f"{diff:.1f}")
        
        # Visualization
        if scores_a and scores_b:
            fig = create_radar_chart(scores_a, scores_b)
            st.plotly_chart(fig, use_container_width=True)
        
        # Responses
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 RAG Response")
            st.write(result['rag_response']['response'])
        
        with col2:
            st.subheader("🤖 Non-RAG Response")
            st.write(result['non_rag_response']['response'])
        
        # Reasoning
        st.subheader("🧠 Judge Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Reasoning:**")
            st.write(evaluation.get('reasoning', 'No reasoning provided'))
        
        with col2:
            st.write("**Key Differences:**")
            st.write(evaluation.get('key_differences', 'No differences noted'))
    
    # History
    if st.session_state.evaluation_history and len(st.session_state.evaluation_history) > 1:
        st.markdown("---")
        st.header("📚 Evaluation History")
        
        # Create history table
        history_data = []
        for eval_result in st.session_state.evaluation_history:
            evaluation = eval_result.get('evaluation', {})
            history_data.append({
                'Query': eval_result.get('query', 'N/A')[:50] + '...' if len(eval_result.get('query', '')) > 50 else eval_result.get('query', 'N/A'),
                'Winner': evaluation.get('winner', 'N/A'),
                'RAG Score': evaluation.get('response_a_scores', {}).get('overall', 0),
                'Non-RAG Score': evaluation.get('response_b_scores', {}).get('overall', 0)
            })
        
        if history_data:
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True)
        
        # Clear history
        if st.button("🗑️ Clear History"):
            st.session_state.evaluation_history = []
            st.session_state.current_evaluation = None
            st.rerun()

if __name__ == "__main__":
    main()
