# RAG Evaluation System - Production Docker Image
FROM python:3.11-slim

LABEL maintainer="RAG Evaluation System"
LABEL description="Dockerized RAG vs Non-RAG evaluation system with Gradio interface"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code (excluding unnecessary files)
COPY gradio_app.py .
COPY config/ ./config/
COPY src/ ./src/
COPY tools/ ./tools/
COPY scripts/ ./scripts/
COPY docs/ ./docs/
COPY .env.example .

# Create necessary directories with proper permissions
RUN mkdir -p \
    data/chroma_db \
    data/cache/embeddings \
    data/processed/chunks \
    data/processed/embeddings \
    data/raw/dataset \
    results/logs \
    results/metrics \
    results/visualizations && \
    chmod -R 755 data results

# Set environment variables
ENV PYTHONPATH=/app/src:/app
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Expose port for Gradio interface
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Default command - run the Gradio app
CMD ["python", "gradio_app.py"]
