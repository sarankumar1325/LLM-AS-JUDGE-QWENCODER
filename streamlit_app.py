"""
Streamlit Web Application: LLM-as-Judge RAG Evaluation System
Interactive demo showcasing RAG vs Non-RAG response comparison using AI judge.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import our RAG system components
try:
    from src.models.groq_client import GroqClient, QwenRAGModel, QwenNonRAGModel
    from src.vector_stores.chroma_store import ChromaVectorStore
    from src.embeddings.nvidia_embedder import EmbedderFactory
    from src.utils.logger import get_logger
except ImportError as e:
    st.error(f"Failed to import RAG components: {e}")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Evaluation - Qwen Judge",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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

.response-card {
    background-color: #ffffff;
    padding: 1.5rem;
    border-radius: 0.8rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 1rem 0;
}

.rag-card {
    border-left: 4px solid #28a745;
}

.non-rag-card {
    border-left: 4px solid #17a2b8;
}

.score-high {
    color: #28a745;
    font-weight: bold;
}

.score-medium {
    color: #ffc107;
    font-weight: bold;
}

.score-low {
    color: #dc3545;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'evaluation_history' not in st.session_state:
    st.session_state.evaluation_history = []
if 'current_evaluation' not in st.session_state:
    st.session_state.current_evaluation = None

# Clean up any corrupted data in session state
if st.session_state.evaluation_history:
    clean_history = []
    for eval in st.session_state.evaluation_history:
        if (eval is not None and 
            isinstance(eval, dict) and 
            'evaluation' in eval and 
            eval['evaluation'] is not None and
            isinstance(eval['evaluation'], dict)):
            clean_history.append(eval)
    st.session_state.evaluation_history = clean_history

class StreamlitRAGEvaluator:
    """Streamlit-specific RAG evaluation system using Qwen."""
    
    def __init__(self):
        """Initialize the evaluator with caching."""
        self.components = None
        
    def setup_components(self):
        """Setup RAG components with simple caching."""
        if self.components is not None:
            return self.components
            
        try:
            # Initialize logger
            logger = get_logger(__name__)
            
            # Initialize Groq client
            groq_client = GroqClient()
            
            # Initialize vector store
            vector_store = ChromaVectorStore()
            
            # Initialize embedder
            embedder = EmbedderFactory.create_default_embedder()
            
            # Test connection
            if not groq_client.test_connection():
                st.error("❌ Failed to connect to Groq API")
                return None
            
            # Initialize RAG and Non-RAG models
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
                'non_rag_model': non_rag_model,
                'logger': logger
            }
            
            return self.components
            
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG components: {e}")
            return None
    
    def evaluate_responses(self, query: str, company_filter: str = None):
        """Evaluate RAG vs Non-RAG responses."""
        components = self.setup_components()
        if not components:
            st.error("❌ Failed to setup components")
            return None
        
        try:
            st.write("🔄 Starting evaluation process...")
            
            # Generate query embedding
            st.write("🔄 Generating query embedding...")
            query_embedding = components['embedder'].embed_text(query)
            st.write(f"✅ Embedding generated: {len(query_embedding)} dimensions")
            
            # Setup company filter
            filter_dict = None
            if company_filter and company_filter != "All Companies":
                filter_dict = {"company_symbol": company_filter}
                st.write(f"🔍 Using company filter: {company_filter}")
            
            # Generate RAG response
            with st.spinner("🔍 Generating RAG response..."):
                st.write("🔄 Calling RAG model...")
                rag_response = components['rag_model'].generate_response(
                    query=query,
                    query_embedding=query_embedding,
                    filter_dict=filter_dict
                )
                st.write(f"✅ RAG response received: {len(rag_response.get('response', ''))} chars")
            
            # Generate Non-RAG response
            with st.spinner("🤖 Generating Non-RAG response..."):
                st.write("🔄 Calling Non-RAG model...")
                non_rag_response = components['non_rag_model'].generate_response(query)
                st.write(f"✅ Non-RAG response received: {len(non_rag_response.get('response', ''))} chars")
            
            # Evaluate with Qwen judge
            with st.spinner("⚖️ Running Qwen Judge evaluation..."):
                st.write("🔄 Starting evaluation...")
                evaluation = self.evaluate_with_qwen(
                    query, 
                    rag_response['response'], 
                    non_rag_response['response'],
                    components['groq_client']
                )
                st.write("✅ Evaluation completed")
            
            result = {
                'query': query,
                'rag_response': rag_response,
                'non_rag_response': non_rag_response,
                'evaluation': evaluation,
                'company_filter': company_filter,
                'timestamp': datetime.now().isoformat()
            }
            
            st.write("🎉 Evaluation process completed successfully!")
            return result
            
        except Exception as e:
            st.error(f"❌ Evaluation failed: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def evaluate_with_qwen(self, query: str, rag_response: str, non_rag_response: str, groq_client):
        """Evaluate responses using Qwen as judge."""
        evaluation_prompt = f"""You are an expert AI judge evaluating two responses to a financial query. Rate each response on a scale of 1-10 for each criterion and determine the winner.

**Query:** {query}

**Response A (RAG-Enhanced):** {rag_response}

**Response B (Standard LLM):** {non_rag_response}

**Evaluation Criteria:**
1. **Accuracy** (1-10): How factually correct is the response?
2. **Completeness** (1-10): Does it fully address the query?
3. **Relevance** (1-10): How directly does it answer the question?
4. **Clarity** (1-10): How clear and well-structured is the response?
5. **Source Reliability** (1-10): How well-grounded is the information?

**Required JSON Response Format:**
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
    "reasoning": "Brief explanation of the decision",
    "key_differences": "Main differences between responses"
}}

Provide only the JSON response, no additional text."""
        
        try:
            response = groq_client.generate_response(
                prompt=evaluation_prompt,
                system_instruction="You are a precise evaluator. Return only valid JSON."
            )
            
            # Parse JSON response
            evaluation_json = response['response'].strip()
            
            # Clean up any thinking tags from Qwen
            if '<think>' in evaluation_json:
                # Extract content between </think> and end
                parts = evaluation_json.split('</think>')
                if len(parts) > 1:
                    evaluation_json = parts[1].strip()
            
            # Remove markdown code blocks if present
            if evaluation_json.startswith('```json'):
                evaluation_json = evaluation_json.split('```json')[1].split('```')[0]
            elif evaluation_json.startswith('```'):
                evaluation_json = evaluation_json.split('```')[1].split('```')[0]
            
            evaluation = json.loads(evaluation_json)
            evaluation['evaluation_model'] = 'qwen/qwen3-32b'
            evaluation['evaluation_time'] = response.get('processing_time', 0)
            
            return evaluation
            
        except Exception as e:
            st.error(f"❌ Qwen Judge evaluation failed: {e}")
            return None

def create_sample_queries():
    """Create sample financial queries for testing."""
    return {
        "Apple Revenue Growth": {
            "query": "What was Apple's revenue growth in Q4 2019?",
            "companies": ["AAPL"],
            "category": "Financial Performance"
        },
        "R&D Comparison": {
            "query": "Compare the R&D spending between Apple and Microsoft in 2018.",
            "companies": ["AAPL", "MSFT"],
            "category": "Research & Development"
        },
        "Semiconductor Challenges": {
            "query": "What were the main challenges faced by semiconductor companies in 2017?",
            "companies": ["INTC", "NVDA", "AMD"],
            "category": "Industry Analysis"
        },
        "Cloud Services Performance": {
            "query": "How did Amazon's cloud services perform in Q3 2020?",
            "companies": ["AMZN"],
            "category": "Technology Services"
        },
        "AI Industry Outlook": {
            "query": "What is the outlook for artificial intelligence in the tech industry?",
            "companies": ["GOOGL", "MSFT", "NVDA"],
            "category": "Future Trends"
        }
    }

def display_score_chart(scores_a, scores_b, title):
    """Create a radar chart comparing response scores."""
    categories = ['Accuracy', 'Completeness', 'Relevance', 'Clarity', 'Source Reliability']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[scores_a.get(cat.lower().replace(' ', '_'), 0) for cat in categories],
        theta=categories,
        fill='toself',
        name='RAG Response',
        line_color='rgba(40, 167, 69, 0.8)',
        fillcolor='rgba(40, 167, 69, 0.3)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[scores_b.get(cat.lower().replace(' ', '_'), 0) for cat in categories],
        theta=categories,
        fill='toself',
        name='Non-RAG Response',
        line_color='rgba(23, 162, 184, 0.8)',
        fillcolor='rgba(23, 162, 184, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=True,
        title=title,
        height=400
    )
    
    return fig

def display_score_bars(scores, title, color):
    """Create horizontal bar chart for individual scores."""
    categories = ['Accuracy', 'Completeness', 'Relevance', 'Clarity', 'Source Reliability']
    values = [scores.get(cat.lower().replace(' ', '_'), 0) for cat in categories]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=categories,
        orientation='h',
        marker_color=color,
        text=[f"{v}/10" for v in values],
        textposition='inside'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Score (1-10)",
        height=300,
        xaxis=dict(range=[0, 10])
    )
    
    return fig

def get_score_color(score):
    """Get color class based on score."""
    if score >= 7:
        return "score-high"
    elif score >= 4:
        return "score-medium"
    else:
        return "score-low"

def main():
    """Main Streamlit application."""
    
    # App header
    st.markdown('<h1 class="main-header">🔍 RAG Evaluation System - Qwen Judge</h1>', unsafe_allow_html=True)
    st.markdown("### Interactive Demo: RAG vs Non-RAG Comparison using **Qwen 3-32B** as AI Judge")
    st.markdown("---")
    
    # Initialize evaluator
    evaluator = StreamlitRAGEvaluator()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Company filter
        companies = ["All Companies", "AAPL", "AMZN", "GOOGL", "MSFT", "NVDA", "INTC", "AMD", "ASML", "CSCO", "MU"]
        selected_company = st.selectbox("🏢 Company Filter", companies)
        
        # Sample queries
        st.header("📋 Sample Queries")
        sample_queries = create_sample_queries()
        
        selected_sample = st.selectbox(
            "Choose a sample query:",
            ["Custom Query"] + list(sample_queries.keys())
        )
        
        if selected_sample != "Custom Query":
            sample_info = sample_queries[selected_sample]
            st.info(f"**Category:** {sample_info['category']}")
            st.info(f"**Companies:** {', '.join(sample_info['companies'])}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Query Input")
        
        # Query input
        if selected_sample != "Custom Query":
            default_query = sample_queries[selected_sample]['query']
        else:
            default_query = ""
        
        query = st.text_area(
            "Enter your financial query:",
            value=default_query,
            height=100,
            placeholder="e.g., What was Apple's revenue growth in Q4 2019?"
        )
        
        # Evaluate button
        if st.button("🚀 Compare Responses", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("🔄 Processing evaluation..."):
                    result = evaluator.evaluate_responses(
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
            # Debug: Show what's in evaluation_history
            st.sidebar.write("Debug: Evaluation History Content")
            for i, eval in enumerate(st.session_state.evaluation_history):
                st.sidebar.write(f"Item {i}: {type(eval)} - {eval is not None}")
                if eval is not None:
                    st.sidebar.write(f"  Keys: {list(eval.keys()) if isinstance(eval, dict) else 'Not a dict'}")
            
            # Filter out None values and ensure valid structure
            valid_evaluations = []
            for eval in st.session_state.evaluation_history:
                if (eval is not None and 
                    isinstance(eval, dict) and 
                    'evaluation' in eval and 
                    eval['evaluation'] is not None and
                    isinstance(eval['evaluation'], dict)):
                    valid_evaluations.append(eval)
            
            total_evaluations = len(valid_evaluations)
            rag_wins = sum(1 for eval in valid_evaluations 
                          if eval['evaluation'].get('winner') == 'A')
            non_rag_wins = sum(1 for eval in valid_evaluations 
                              if eval['evaluation'].get('winner') == 'B')
            
            st.metric("Total Evaluations", total_evaluations)
            st.metric("RAG Wins", rag_wins)
            st.metric("Non-RAG Wins", non_rag_wins)
            
            if total_evaluations > 0:
                rag_win_rate = (rag_wins / total_evaluations) * 100
                st.metric("RAG Win Rate", f"{rag_win_rate:.1f}%")
        else:
            st.info("No evaluations yet. Start by entering a query above!")
    
    # Display current evaluation results
    if st.session_state.current_evaluation:
        result = st.session_state.current_evaluation
        evaluation = result.get('evaluation')
        
        if evaluation:
            st.markdown("---")
            st.header("🎯 Evaluation Results")
            
            # Winner announcement
            winner = evaluation.get('winner', 'Tie')
            if winner == 'A':
                st.success("🏆 **Winner: RAG Response**")
            elif winner == 'B':
                st.info("🏆 **Winner: Non-RAG Response**")
            else:
                st.warning("🤝 **Result: Tie**")
            
            # Response comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔍 RAG Response")
                st.markdown('<div class="response-card rag-card">', unsafe_allow_html=True)
                st.write(result['rag_response']['response'])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # RAG metadata
                rag_meta = result['rag_response']
                st.markdown("**📊 RAG Metadata:**")
                st.write(f"• Context chunks: {rag_meta.get('context_chunks', 0)}")
                st.write(f"• Processing time: {rag_meta.get('processing_time', 0):.2f}s")
                st.write(f"• Total tokens: {rag_meta.get('total_tokens', 0)}")
                
                # Context sources
                context_sources = rag_meta.get('context_sources', [])
                if context_sources:
                    st.markdown("**📚 Retrieved Sources:**")
                    for i, source in enumerate(context_sources[:3], 1):
                        st.write(f"{i}. {source}")
            
            with col2:
                st.markdown("### 🤖 Non-RAG Response")
                st.markdown('<div class="response-card non-rag-card">', unsafe_allow_html=True)
                st.write(result['non_rag_response']['response'])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Non-RAG metadata
                non_rag_meta = result['non_rag_response']
                st.markdown("**📊 Non-RAG Metadata:**")
                st.write(f"• Processing time: {non_rag_meta.get('processing_time', 0):.2f}s")
                st.write(f"• Total tokens: {non_rag_meta.get('total_tokens', 0)}")
            
            # Scoring comparison
            st.markdown("---")
            st.header("📈 Detailed Scoring")
            
            scores_a = evaluation.get('response_a_scores', {})
            scores_b = evaluation.get('response_b_scores', {})
            
            # Overall scores
            col1, col2, col3 = st.columns(3)
            with col1:
                overall_a = scores_a.get('overall', 0)
                st.metric(
                    "RAG Overall Score", 
                    f"{overall_a:.1f}/10",
                    delta=f"{overall_a - scores_b.get('overall', 0):.1f}"
                )
            
            with col2:
                overall_b = scores_b.get('overall', 0)
                st.metric(
                    "Non-RAG Overall Score",
                    f"{overall_b:.1f}/10"
                )
            
            with col3:
                diff = abs(overall_a - overall_b)
                st.metric("Score Difference", f"{diff:.1f}")
            
            # Radar chart comparison
            fig_radar = display_score_chart(scores_a, scores_b, "Response Comparison (All Criteria)")
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Individual score bars
            col1, col2 = st.columns(2)
            
            with col1:
                fig_rag = display_score_bars(scores_a, "RAG Response Scores", 'rgba(40, 167, 69, 0.8)')
                st.plotly_chart(fig_rag, use_container_width=True)
            
            with col2:
                fig_non_rag = display_score_bars(scores_b, "Non-RAG Response Scores", 'rgba(23, 162, 184, 0.8)')
                st.plotly_chart(fig_non_rag, use_container_width=True)
            
            # Judge reasoning
            st.markdown("---")
            st.header("🧠 AI Judge Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🎯 Reasoning:**")
                st.write(evaluation.get('reasoning', 'No reasoning provided'))
            
            with col2:
                st.markdown("**🔍 Key Differences:**")
                st.write(evaluation.get('key_differences', 'No differences noted'))
    
    # Evaluation history
    if st.session_state.evaluation_history:
        st.markdown("---")
        st.header("📚 Evaluation History")
        
        # Summary statistics
        valid_evaluations = []
        for eval in st.session_state.evaluation_history:
            if (eval is not None and 
                isinstance(eval, dict) and 
                'evaluation' in eval and 
                eval['evaluation'] is not None and
                isinstance(eval['evaluation'], dict)):
                valid_evaluations.append(eval)
        
        if len(valid_evaluations) > 1:
            history_df = []
            for i, eval_result in enumerate(valid_evaluations):
                evaluation = eval_result['evaluation']  # We know this is valid now
                row = {
                    'Query': eval_result.get('query', 'N/A')[:50] + '...' if len(eval_result.get('query', '')) > 50 else eval_result.get('query', 'N/A'),
                    'Winner': evaluation.get('winner', 'N/A'),
                    'RAG Score': evaluation.get('response_a_scores', {}).get('overall', 0) if evaluation.get('response_a_scores') else 0,
                    'Non-RAG Score': evaluation.get('response_b_scores', {}).get('overall', 0) if evaluation.get('response_b_scores') else 0,
                    'Company Filter': eval_result.get('company_filter', 'All')
                }
                history_df.append(row)
            
            if history_df:
                df = pd.DataFrame(history_df)
                st.dataframe(df, use_container_width=True)
                
                # Download history
                if st.button("📥 Download History (JSON)"):
                    valid_history = [eval for eval in st.session_state.evaluation_history if eval is not None]
                    history_json = json.dumps(valid_history, indent=2)
                    st.download_button(
                        label="Download Evaluation History",
                        data=history_json,
                        file_name=f"rag_evaluation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        # Clear history
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.evaluation_history = []
            st.session_state.current_evaluation = None
            st.rerun()

if __name__ == "__main__":
    main()
