# RAG Evaluation System

A comprehensive system for evaluating Retrieval-Augmented Generation (RAG) models using LLM as Judge methodology.

## Overview

This project implements a robust evaluation framework that compares RAG and non-RAG approaches using various metrics and an LLM judge (Gemini) for qualitative assessment.

## Features

- Multiple vector store backends (ChromaDB, FAISS)
- NVIDIA embeddings integration
- Comprehensive evaluation metrics
- LLM-as-Judge evaluation using Gemini
- Database storage for results
- Visualization and analysis tools

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```
   cp .env.example .env
   # Edit .env with your API keys
   ```
4. Initialize the database:
   ```
   python scripts/setup_database.py
   ```

## Usage

1. Process your dataset:
   ```
   python scripts/process_dataset.py
   ```

2. Build vector store:
   ```
   python scripts/build_vector_store.py
   ```

3. Run evaluation:
   ```
   python scripts/run_evaluation.py
   ```

4. Analyze results:
   ```
   python scripts/analyze_results.py
   ```

## Project Structure

- `config/`: Configuration files and settings
- `data/`: Raw and processed data
- `src/`: Main source code
- `scripts/`: Utility scripts
- `notebooks/`: Jupyter notebooks for analysis
- `tests/`: Unit tests
- `results/`: Evaluation results and visualizations
- `docker/`: Docker configuration

## License

[Your License]
