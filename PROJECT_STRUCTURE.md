# 📁 RAG Evaluation System - Organized Project Structure

## 🏗️ **Root Directory**
```
rag-evaluation-system/
├── 📱 gradio_app.py              # 🚀 MAIN APPLICATION (Launch this!)
├── 📋 requirements.txt           # Python dependencies
├── 🔐 .env                      # API keys and environment variables
├── 🔐 .env.example              # Template for environment variables
└── 📋 .gitignore                # Git ignore rules
```

## 📚 **Core Directories**

### `/src/` - Source Code
```
src/
├── models/         # AI model integrations (Qwen via Groq)
├── vector_stores/  # ChromaDB vector database
├── embeddings/     # Text embedding models
├── evaluation/     # Response evaluation logic
├── data_processing/# Document processing pipelines
├── database/       # Database utilities
└── utils/          # Common utilities
```

### `/config/` - Configuration
```
config/
├── model_configs.py  # Model settings and parameters
├── settings.py       # System configuration
└── __init__.py      # Package initialization
```

### `/data/` - Data Storage
```
data/
├── chroma_db/         # ChromaDB vector database (17,664 docs)
├── cache/embeddings/  # Cached embeddings for performance
├── processed/         # Processed document chunks
├── raw/dataset/       # Original raw documents
└── sample_queries.json # Test queries for evaluation
```

### `/tools/` - Analysis & Testing Tools
```
tools/
├── 🔍 analyze_rag_vs_non_rag.py  # RAG vs Non-RAG analysis
├── ✅ test_groq_connection.py     # API connection test
├── ✅ test_final_fix.py           # System integration test
└── 🔑 test_api_key_direct.py     # API key validation
```

### `/scripts/` - Data Processing Scripts
```
scripts/
├── generate_embeddings.py      # Create document embeddings
├── process_dataset.py          # Process raw documents
├── process_real_data.py        # Process financial data
├── run_rag_evaluation.py       # Run evaluation pipeline
├── setup_chromadb.py          # Initialize vector database
└── test_chromadb_integration.py # Test database setup
```

### `/docs/` - Documentation
```
docs/
├── 📖 FINAL_SYSTEM_GUIDE.md    # Complete system guide
├── 🔧 FIX_SUMMARY.md           # Technical fixes summary
└── 📝 README.md                # Project overview
```

### `/results/` - Evaluation Results
```
results/
├── evaluation_fixed.json       # Latest evaluation results
├── evaluation_test.json        # Test results
├── logs/                       # System logs
├── metrics/                    # Performance metrics
└── visualizations/             # Result charts and graphs
```

### `/notebooks/` - Jupyter Notebooks
```
notebooks/
└── 01_data_exploration.ipynb   # Data analysis notebook
```

### `/real data/` - Financial Dataset
```
real data/
├── AAPL/   # Apple financial documents (2016-2019)
├── AMD/    # AMD financial documents
├── AMZN/   # Amazon financial documents
├── ASML/   # ASML financial documents
├── CSCO/   # Cisco financial documents
├── INTC/   # Intel financial documents
├── MSFT/   # Microsoft financial documents
├── MU/     # Micron financial documents
└── NVDA/   # NVIDIA financial documents
```

### `/docker/` - Containerization
```
docker/
├── Dockerfile          # Container definition
└── docker-compose.yml  # Multi-service setup
```

### `/tests/` - Unit Tests
```
tests/
└── __init__.py         # Test package initialization
```

---

## 🚀 **Quick Start Guide**

### 1. Launch the Main Application
```bash
python gradio_app.py
```
**URL:** http://localhost:7860

### 2. Run Analysis Tools
```bash
# Analyze RAG vs Non-RAG performance
python tools/analyze_rag_vs_non_rag.py

# Test system health
python tools/test_groq_connection.py

# Verify integration
python tools/test_final_fix.py
```

### 3. Data Processing (if needed)
```bash
# Process new documents
python scripts/process_real_data.py

# Generate embeddings
python scripts/generate_embeddings.py

# Setup vector database
python scripts/setup_chromadb.py
```

---

## 📊 **File Organization Benefits**

### ✅ **Clean Structure**
- **Main app** easily accessible at root level
- **Related tools** grouped by function
- **Documentation** centralized in `/docs/`
- **Data files** organized by type and processing stage

### ✅ **Easy Navigation**
- **Core functionality** in `/src/`
- **Analysis tools** in `/tools/`
- **Processing scripts** in `/scripts/`
- **Results** tracked in `/results/`

### ✅ **Maintainable**
- **Clear separation** of concerns
- **Logical grouping** of related files
- **Consistent naming** conventions
- **Documented structure** for new contributors

---

## 🎯 **Key Files to Know**

| File | Purpose | When to Use |
|------|---------|-------------|
| `gradio_app.py` | Main web interface | **Always** - This is your primary app |
| `tools/analyze_rag_vs_non_rag.py` | Performance analysis | When comparing RAG vs Non-RAG |
| `tools/test_groq_connection.py` | System health check | When troubleshooting API issues |
| `docs/FINAL_SYSTEM_GUIDE.md` | Complete guide | When you need comprehensive documentation |
| `config/settings.py` | System configuration | When changing model parameters |
| `src/models/groq_client.py` | AI model interface | When modifying model behavior |

---

## 🧹 **Files Removed During Cleanup**

### ❌ **Obsolete Files**
- `system_status.py` - System now stable
- `processed/` directory - Duplicate of `data/processed/`
- `scripts/setup_gemini_api.py` - Switched to Qwen/Groq
- `scripts/test_*.py` - Redundant test files
- All Streamlit apps - Switched to Gradio

### ✅ **Why Removed**
- **Reduce clutter** - Keep only essential files
- **Eliminate duplication** - Single source of truth
- **Remove obsolete tech** - No more Gemini/Streamlit
- **Improve maintainability** - Cleaner structure

---

Your RAG evaluation system is now **perfectly organized** and ready for production use! 🎉
