#!/usr/bin/env python3
"""
Streamlit RAG Evaluation App using Qwen through Groq.
Enhanced version with Qwen as the judge model.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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
    page_title="RAG Evaluation System - Qwen Judge",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ff6b6b;
}

.winner-a {
    background-color: #d4edda;
    padding: 0.5rem;
    border-radius: 0.25rem;
    border-left: 4px solid #28a745;
}

.winner-b {
    background-color: #f8d7da;
    padding: 0.5rem;
    border-radius: 0.25rem;
    border-left: 4px solid #dc3545;
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

class StreamlitQwenRAGEvaluator:
    """Streamlit-specific RAG evaluation system using Qwen."""
    
    def __init__(self):
        """Initialize the evaluator with caching."""
        self.setup_components()
        
    @st.cache_resource
    def setup_components(_self):
        """Setup and cache system components."""
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
            
            return {
                'groq_client': groq_client,
                'vector_store': vector_store,
                'embedder': embedder,
                'rag_model': rag_model,
                'non_rag_model': non_rag_model
            }
            
        except Exception as e:
            st.error(f"❌ Failed to initialize components: {e}")
            return None
    
    def generate_query_embedding(self, query: str):
        """Generate embedding for query."""
        components = self.setup_components()
        if not components:
            return None
        
        return components['embedder'].embed_text(query)
    
    def evaluate_responses(self, query: str, company_filter: str = None):
        """Evaluate RAG vs Non-RAG responses using Qwen."""
        components = self.setup_components()
        if not components:
            return None
        
        try:
            # Generate query embedding
            query_embedding = self.generate_query_embedding(query)
            
            # Setup company filter
            filter_dict = None
            if company_filter and company_filter != "All Companies":
                filter_dict = {"company_symbol": company_filter}
            
            # Generate RAG response
            with st.spinner("🔍 Generating RAG response with Qwen..."):
                rag_response = components['rag_model'].generate_response(
                    query=query,
                    query_embedding=query_embedding,
                    filter_dict=filter_dict
                )
            
            # Generate Non-RAG response
            with st.spinner("🤖 Generating Non-RAG response with Qwen..."):
                non_rag_response = components['non_rag_model'].generate_response(query)
            
            # Evaluate with Qwen judge
            with st.spinner("⚖️ Running Qwen Judge evaluation..."):
                evaluation = self.evaluate_with_qwen(
                    query, 
                    rag_response['response'], 
                    non_rag_response['response'],
                    components['groq_client']
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
        "Microsoft Cloud Services": {
            "query": "How did Microsoft's cloud services perform in 2017?",
            "companies": ["MSFT"],
            "category": "Business Segments"
        },
        "NVIDIA GPU Market": {
            "query": "What was NVIDIA's position in the GPU market in 2018?",
            "companies": ["NVDA"],
            "category": "Market Position"
        },
        "Amazon AWS Growth": {
            "query": "How fast was Amazon Web Services growing in 2019?",
            "companies": ["AMZN"],
            "category": "Cloud Computing"
        },
        "Intel vs AMD Competition": {
            "query": "How did Intel respond to AMD's competitive pressure in 2017-2019?",
            "companies": ["INTC", "AMD"],
            "category": "Competition Analysis"
        }
    }

def create_radar_chart(scores_a, scores_b):
    """Create radar chart comparing response scores."""
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
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=True,
        title="Response Quality Comparison"
    )
    
    return fig

def main():
    """Main Streamlit app."""
    st.title("🔍 RAG Evaluation System - Qwen Judge")
    st.markdown("Compare RAG-enhanced responses vs standard LLM responses using **Qwen 3-32B** as the AI judge")
    
    # Initialize evaluator
    evaluator = StreamlitQwenRAGEvaluator()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Company filter
        companies = ["All Companies", "AAPL", "MSFT", "NVDA", "AMZN", "INTC", "AMD", "CSCO", "ASML", "MU"]
        selected_company = st.selectbox("📊 Filter by Company", companies)
        
        # Sample queries
        st.header("📝 Sample Queries")
        sample_queries = create_sample_queries()
        
        for title, details in sample_queries.items():
            if st.button(f"💡 {title}", key=f"sample_{title}"):
                st.session_state.sample_query = details["query"]
                if details["companies"] and len(details["companies"]) == 1:
                    st.session_state.sample_company = details["companies"][0]
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🤖 Query Input")
        
        # Query input
        query = st.text_area(
            "Enter your financial question:",
            value=st.session_state.get('sample_query', ''),
            height=100,
            placeholder="e.g., What was Apple's revenue growth in Q4 2019?"
        )
        
        # Update company filter if sample was selected
        if hasattr(st.session_state, 'sample_company'):
            selected_company = st.session_state.sample_company
            del st.session_state.sample_company
        
        # Evaluation button
        if st.button("🚀 Compare Responses", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("🔄 Processing evaluation with Qwen..."):
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
    
    # Display current evaluation results
    if st.session_state.current_evaluation:
        result = st.session_state.current_evaluation
        evaluation = result.get('evaluation', {})
        
        st.markdown("---")
        st.header("🎯 Evaluation Results")
        
        # Winner announcement
        winner = evaluation.get('winner', 'Unknown')
        if winner == 'A':
            st.success("🏆 **RAG-Enhanced Response Wins!**")
        elif winner == 'B':
            st.error("🏆 **Standard LLM Response Wins!**")
        else:
            st.info("🤝 **It's a Tie!**")
        
        # Scores and visualization
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Detailed Scores")
            
            scores_a = evaluation.get('response_a_scores', {})
            scores_b = evaluation.get('response_b_scores', {})
            
            # Create comparison DataFrame
            comparison_data = {
                'Criterion': ['Accuracy', 'Completeness', 'Relevance', 'Clarity', 'Source Reliability', 'Overall'],
                'RAG Score': [
                    scores_a.get('accuracy', 0),
                    scores_a.get('completeness', 0),
                    scores_a.get('relevance', 0),
                    scores_a.get('clarity', 0),
                    scores_a.get('source_reliability', 0),
                    scores_a.get('overall', 0)
                ],
                'Non-RAG Score': [
                    scores_b.get('accuracy', 0),
                    scores_b.get('completeness', 0),
                    scores_b.get('relevance', 0),
                    scores_b.get('clarity', 0),
                    scores_b.get('source_reliability', 0),
                    scores_b.get('overall', 0)
                ]
            }
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Score Visualization")
            if scores_a and scores_b:
                fig = create_radar_chart(scores_a, scores_b)
                st.plotly_chart(fig, use_container_width=True)
        
        # Response comparison
        st.subheader("💬 Response Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔍 RAG-Enhanced Response:**")
            rag_response = result.get('rag_response', {})
            st.write(rag_response.get('response', 'No response available'))
            
            # Show context chunks
            if rag_response.get('context_chunks'):
                with st.expander(f"📚 Context Used ({len(rag_response['context_chunks'])} chunks)"):
                    for i, chunk in enumerate(rag_response['context_chunks'][:3]):  # Show first 3
                        st.write(f"**Chunk {i+1}** (Score: {chunk.get('score', 0):.3f})")
                        st.write(chunk.get('content', '')[:200] + "...")
                        st.write("---")
        
        with col2:
            st.markdown("**🤖 Standard LLM Response:**")
            non_rag_response = result.get('non_rag_response', {})
            st.write(non_rag_response.get('response', 'No response available'))
        
        # Judge reasoning
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**⚖️ Judge Reasoning:**")
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
                        file_name=f"qwen_rag_evaluation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        # Clear history
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.evaluation_history = []
            st.session_state.current_evaluation = None
            st.rerun()

if __name__ == "__main__":
    main()
