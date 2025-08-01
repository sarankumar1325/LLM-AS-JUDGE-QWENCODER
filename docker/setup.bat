@echo off
REM RAG Evaluation System - Docker Setup Script for Windows
REM This script helps users set up the dockerized RAG system on Windows

echo 🚀 Setting up RAG Evaluation System with Docker
echo ==============================================

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker Desktop:
    echo    https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed. Please install Docker Desktop:
    echo    https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo ✅ Docker and Docker Compose are available

REM Create .env file from template if it doesn't exist
if not exist .env (
    echo 📝 Creating .env file from template...
    copy .env.example .env
    echo.
    echo ⚠️  IMPORTANT: Please edit .env file and add your API keys:
    echo    - GROQ_API_KEY=your_groq_api_key_here
    echo    - Other API keys as needed
    echo.
    echo 🔑 Get your Groq API key from: https://console.groq.com/
    echo.
    pause
)

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist data\chroma_db mkdir data\chroma_db
if not exist data\cache\embeddings mkdir data\cache\embeddings
if not exist data\processed mkdir data\processed
if not exist results\logs mkdir results\logs
if not exist results\metrics mkdir results\metrics
if not exist results\visualizations mkdir results\visualizations

echo ✅ Directories created

REM Build and start services
echo 🔨 Building Docker images (this may take a few minutes)...
docker-compose build

echo 🚀 Starting services...
docker-compose up -d

REM Wait for services to be ready
echo ⏳ Waiting for services to start...
timeout /t 10 /nobreak >nul

REM Check if services are running
echo 🔍 Checking service status...
docker-compose ps

REM Display access information
echo.
echo 🎉 Setup completed successfully!
echo ================================
echo.
echo 🌐 Access the RAG Evaluation System at:
echo    http://localhost:7860
echo.
echo 🔧 ChromaDB (vector database) is available at:
echo    http://localhost:8001
echo.
echo 📋 Useful commands:
echo    docker-compose logs rag-evaluation  # View main app logs
echo    docker-compose logs chromadb        # View database logs
echo    docker-compose down                 # Stop all services
echo    docker-compose up -d                # Start services again
echo.
echo 🧪 To run analysis tools:
echo    docker-compose --profile tools up rag-tools
echo.
echo 📚 Check docs\DOCKER_GUIDE.md for detailed instructions
pause
