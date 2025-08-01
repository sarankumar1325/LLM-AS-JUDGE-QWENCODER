#!/bin/bash
# RAG Evaluation System - Docker Setup Script
# This script helps users set up the dockerized RAG system on their machine

set -e

echo "🚀 Setting up RAG Evaluation System with Docker"
echo "=============================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first:"
    echo "   - Windows/Mac: https://www.docker.com/products/docker-desktop"
    echo "   - Linux: https://docs.docker.com/engine/install/"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose:"
    echo "   https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose are available"

# Create .env file from template if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  IMPORTANT: Please edit .env file and add your API keys:"
    echo "   - GROQ_API_KEY=your_groq_api_key_here"
    echo "   - Other API keys as needed"
    echo ""
    echo "🔑 Get your Groq API key from: https://console.groq.com/"
    echo ""
    read -p "Press Enter when you have updated the .env file..."
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/chroma_db
mkdir -p data/cache/embeddings
mkdir -p data/processed
mkdir -p results/logs
mkdir -p results/metrics
mkdir -p results/visualizations

echo "✅ Directories created"

# Build and start services
echo "🔨 Building Docker images (this may take a few minutes)..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
echo "🔍 Checking service status..."
docker-compose ps

# Display access information
echo ""
echo "🎉 Setup completed successfully!"
echo "================================"
echo ""
echo "🌐 Access the RAG Evaluation System at:"
echo "   http://localhost:7860"
echo ""
echo "🔧 ChromaDB (vector database) is available at:"
echo "   http://localhost:8001"
echo ""
echo "📋 Useful commands:"
echo "   docker-compose logs rag-evaluation  # View main app logs"
echo "   docker-compose logs chromadb        # View database logs"
echo "   docker-compose down                 # Stop all services"
echo "   docker-compose up -d                # Start services again"
echo ""
echo "🧪 To run analysis tools:"
echo "   docker-compose --profile tools up rag-tools"
echo ""
echo "📚 Check docs/DOCKER_GUIDE.md for detailed instructions"
