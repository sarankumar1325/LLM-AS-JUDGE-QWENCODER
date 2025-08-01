# 🎯 RAG Evaluation Streamlit Web Application

## 📋 Overview

An interactive web application built with Streamlit that demonstrates the RAG Evaluation System using LLM-as-Judge methodology. This app provides a user-friendly interface to compare RAG-enhanced responses against standard LLM responses using Gemini 2.5 Pro as an AI judge.

## 🚀 Features

### Core Functionality
- **💬 Interactive Query Interface**: Text input with sample financial queries
- **🔍 Dual Response Generation**: Side-by-side RAG vs Non-RAG comparison
- **⚖️ AI Judge Evaluation**: Gemini 2.5 Pro scoring on 5 criteria
- **📊 Visual Scoring**: Radar charts and bar graphs for score comparison
- **📚 Source Attribution**: Display retrieved documents for RAG responses

### User Experience
- **🎨 Clean 2-Column Layout**: Optimized for response comparison
- **⏳ Progress Indicators**: Real-time feedback during processing
- **📱 Responsive Design**: Works on desktop and tablet devices
- **🔄 Evaluation History**: Track and download previous evaluations

### Advanced Features
- **🏢 Company Filtering**: Filter by specific companies (AAPL, GOOGL, etc.)
- **📈 Performance Metrics**: Processing time, token usage, retrieval stats
- **💾 Data Export**: Download evaluation history as JSON
- **📊 Quick Stats**: Win rates and evaluation summaries

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- All RAG system components installed and configured
- ChromaDB with 17,664 financial documents loaded
- Gemini API key configured

### Quick Start
```bash
# Install Streamlit dependencies
pip install -r streamlit_requirements.txt

# Test component compatibility
python test_streamlit.py

# Launch the web app
python run_streamlit.py
```

### Alternative Launch Methods
```bash
# Direct Streamlit command
streamlit run streamlit_app.py --server.port=8501

# Windows batch file
run_streamlit.bat

# Access at: http://localhost:8501
```

## 📖 User Guide

### 1. Query Input
- **Sample Queries**: Choose from pre-configured financial questions
- **Custom Queries**: Enter your own questions in the text area
- **Company Filter**: Select specific companies or "All Companies"

### 2. Response Generation
- Click **"🚀 Compare Responses"** to start evaluation
- Watch progress indicators for each processing step:
  - 🔍 RAG response generation (with context retrieval)
  - 🤖 Non-RAG response generation
  - ⚖️ AI judge evaluation

### 3. Results Analysis
- **Winner Announcement**: Clear indication of which response won
- **Response Comparison**: Side-by-side display with metadata
- **Detailed Scoring**: 5-criteria breakdown with visualizations
- **AI Judge Reasoning**: Explanation of the decision

### 4. Data Management
- **Evaluation History**: Review all previous comparisons
- **Export Options**: Download results as JSON
- **Quick Stats**: Track RAG vs Non-RAG win rates

## 🎯 Evaluation Criteria

The AI judge scores responses (1-10) on:

1. **Accuracy** (1-10): Factual correctness and precision
2. **Completeness** (1-10): Comprehensive coverage of the query
3. **Relevance** (1-10): Direct alignment with the question asked
4. **Clarity** (1-10): Communication quality and structure
5. **Source Reliability** (1-10): Evidence-based vs speculative content

## 📊 Sample Queries

### Financial Performance
- "What was Apple's revenue growth in Q4 2019?"
- "How did Amazon's cloud services perform in Q3 2020?"

### Comparative Analysis
- "Compare the R&D spending between Apple and Microsoft in 2018."

### Industry Insights
- "What were the main challenges faced by semiconductor companies in 2017?"
- "What is the outlook for artificial intelligence in the tech industry?"

## 🔧 Configuration

### Environment Variables
Ensure these are set in your `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key
NVIDIA_API_KEY=your_nvidia_api_key  # If using NVIDIA embeddings
```

### Streamlit Configuration
The app uses these default settings:
- **Port**: 8501
- **Address**: localhost
- **Caching**: Enabled for RAG components
- **Theme**: Light mode with custom CSS styling

## 📈 Performance Metrics

### Response Generation
- **RAG Processing**: ~2-5 seconds (including retrieval)
- **Non-RAG Processing**: ~1-3 seconds
- **AI Judge Evaluation**: ~3-7 seconds
- **Total Per Query**: ~6-15 seconds

### System Resources
- **Memory Usage**: ~2-4 GB (includes embedding models)
- **ChromaDB**: 17,664 documents loaded
- **Context Retrieval**: 5 chunks per query (average)

## 🎨 UI Components

### Layout Structure
```
├── Header (App Title & Description)
├── Sidebar (Configuration & Samples)
├── Main Content
│   ├── Query Input (Col 1)
│   └── Quick Stats (Col 2)
├── Results Section
│   ├── Winner Announcement
│   ├── Response Comparison (2 columns)
│   ├── Scoring Visualizations
│   └── AI Judge Analysis
└── Evaluation History
```

### Visualizations
- **Radar Chart**: 5-criteria comparison between responses
- **Bar Charts**: Individual score breakdowns
- **Metrics Cards**: Processing time, token usage, win rates
- **Progress Bars**: Real-time processing feedback

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Run component test
python test_streamlit.py

# Install missing dependencies
pip install -r streamlit_requirements.txt
```

**ChromaDB Connection**
- Ensure ChromaDB is initialized with documents
- Check data/chroma_db directory exists
- Run: `python scripts/setup_chromadb.py`

**Gemini API Quota**
- Monitor API usage in Google AI Studio
- Implement rate limiting if needed
- Check API key configuration

**Performance Issues**
- Clear Streamlit cache: Add `?clear_cache=true` to URL
- Restart the app: `Ctrl+C` then rerun
- Check system memory usage

### Debug Mode
```bash
# Run with debug logging
streamlit run streamlit_app.py --logger.level=debug

# Check component status
python test_streamlit.py
```

## 🔮 Future Enhancements

### Planned Features
- **📊 Advanced Analytics**: Statistical significance testing
- **🎯 Batch Evaluation**: Process multiple queries simultaneously
- **📈 Performance Dashboard**: System metrics and usage analytics
- **🔍 Query Suggestions**: AI-powered query recommendations
- **💾 Database Integration**: Store evaluations in database
- **🎨 Custom Themes**: Dark mode and color customization

### Potential Integrations
- **📱 Mobile App**: React Native companion
- **🔗 API Endpoints**: RESTful API for programmatic access
- **📊 BI Tools**: Integration with Tableau/Power BI
- **🤖 Slack Bot**: Team collaboration features

## 📚 Technical Architecture

### Component Stack
```
┌─────────────────────────────────────┐
│         Streamlit Frontend          │
├─────────────────────────────────────┤
│         RAG Evaluation Logic        │
├─────────────────────────────────────┤
│    Gemini API    │    ChromaDB      │
├─────────────────────────────────────┤
│      HuggingFace Embeddings         │
├─────────────────────────────────────┤
│      Financial Document Store       │
└─────────────────────────────────────┘
```

### Data Flow
1. **User Input** → Query + Company Filter
2. **Embedding Generation** → HuggingFace Model
3. **Context Retrieval** → ChromaDB Similarity Search
4. **Response Generation** → RAG + Non-RAG Models
5. **AI Judge Evaluation** → Gemini 2.5 Pro Scoring
6. **Visualization** → Plotly Charts + Streamlit UI

## 🏆 Success Metrics

### User Experience
- **⚡ Fast Response**: < 15 seconds per evaluation
- **🎯 Accurate Results**: Consistent AI judge scoring
- **📱 Intuitive Interface**: Minimal learning curve
- **🔄 Reliable Operation**: 99%+ uptime

### Educational Value
- **💡 Clear Comparisons**: RAG vs Non-RAG differences
- **📊 Visual Learning**: Score breakdowns and explanations
- **🔍 Source Transparency**: Retrieved document attribution
- **📈 Progress Tracking**: Historical evaluation patterns

---

**🌐 Access the app at: http://localhost:8501**

**📞 Support**: Check logs and run `test_streamlit.py` for diagnostics
