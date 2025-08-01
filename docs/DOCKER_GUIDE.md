# 🐳 Docker Guide - RAG Evaluation System

## 🚀 Quick Start for New Users

### Prerequisites
- **Docker Desktop** installed on your machine
  - Windows/Mac: [Download Docker Desktop](https://www.docker.com/products/docker-desktop)
  - Linux: [Install Docker Engine](https://docs.docker.com/engine/install/)
- **8GB+ RAM** recommended
- **5GB+ free disk space**

### 1-Minute Setup

1. **Download the project:**
   ```bash
   git clone <repository-url>
   cd rag-evaluation-system
   ```

2. **Run setup script:**
   ```bash
   # Windows
   docker\setup.bat
   
   # Linux/Mac
   chmod +x docker/setup.sh
   ./docker/setup.sh
   ```

3. **Add your API key:**
   - Edit `.env` file
   - Add: `GROQ_API_KEY=your_api_key_here`
   - Get key from: https://console.groq.com/

4. **Access the application:**
   - Open: http://localhost:7860
   - Start asking questions about companies!

---

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐
│   Gradio App    │    │    ChromaDB     │
│   (Port 7860)   │◄──►│   (Port 8001)   │
│                 │    │  Vector Store   │
└─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│   Groq API      │
│   (Qwen Model)  │
└─────────────────┘
```

### Services:
- **rag-evaluation**: Main Gradio web interface
- **chromadb**: Vector database for document storage
- **rag-tools**: Optional analysis tools container

---

## 📋 Detailed Commands

### Basic Operations
```bash
# Start the system
docker-compose up -d

# View logs (helpful for debugging)
docker-compose logs -f rag-evaluation

# Stop the system
docker-compose down

# Restart a specific service
docker-compose restart rag-evaluation

# Check service status
docker-compose ps
```

### Analysis Tools
```bash
# Run RAG vs Non-RAG analysis
docker-compose --profile tools up rag-tools

# Run specific analysis tool
docker-compose run --rm rag-tools python tools/test_groq_connection.py
```

### Data Management
```bash
# Reset vector database (if needed)
docker-compose down
docker volume rm rag-evaluation-system_chroma_data
docker-compose up -d

# Backup vector database
docker run --rm -v rag-evaluation-system_chroma_data:/data -v $(pwd):/backup alpine tar czf /backup/chroma_backup.tar.gz -C /data .

# Restore vector database
docker run --rm -v rag-evaluation-system_chroma_data:/data -v $(pwd):/backup alpine tar xzf /backup/chroma_backup.tar.gz -C /data
```

---

## ⚙️ Configuration

### Environment Variables (.env file)
```bash
# Required: Groq API for Qwen model
GROQ_API_KEY=your_groq_api_key_here

# Optional: Other API keys
NVIDIA_API_KEY=your_nvidia_key_here
GOOGLE_API_KEY=your_google_key_here

# System settings
PYTHONPATH=/app/src:/app
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
```

### Port Configuration
- **7860**: Gradio web interface (main app)
- **8001**: ChromaDB API (vector database)

### Volume Mounts
- `./data` → `/app/data` (vector database, cache)
- `./results` → `/app/results` (evaluation results)
- `./real data` → `/app/real data` (financial documents, read-only)

---

## 🧪 Development & Testing

### Development Mode
```bash
# Run with code changes (bind mount source)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### Testing
```bash
# Test API connectivity
docker-compose run --rm rag-evaluation python tools/test_groq_connection.py

# Test complete system
docker-compose run --rm rag-evaluation python tools/test_final_fix.py

# Run analysis
docker-compose run --rm rag-evaluation python tools/analyze_rag_vs_non_rag.py
```

### Debugging
```bash
# Get shell access to main container
docker-compose exec rag-evaluation bash

# Check container logs
docker-compose logs rag-evaluation

# Monitor resource usage
docker stats
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. **Port Already in Use**
```bash
# Find process using port 7860
lsof -i :7860  # Linux/Mac
netstat -ano | findstr :7860  # Windows

# Kill process or change port in docker-compose.yml
```

#### 2. **Out of Memory**
```bash
# Check Docker memory settings
docker system df

# Increase Docker Desktop memory allocation
# Docker Desktop → Settings → Resources → Memory → 8GB+
```

#### 3. **API Key Issues**
```bash
# Check .env file exists and has correct key
cat .env

# Test API key directly
docker-compose run --rm rag-evaluation python tools/test_api_key_direct.py
```

#### 4. **ChromaDB Connection Issues**
```bash
# Restart ChromaDB
docker-compose restart chromadb

# Check ChromaDB logs
docker-compose logs chromadb

# Reset ChromaDB (WARNING: deletes all data)
docker-compose down
docker volume rm rag-evaluation-system_chroma_data
docker-compose up -d
```

#### 5. **Gradio Interface Not Loading**
```bash
# Check if service is running
docker-compose ps

# Check application logs
docker-compose logs rag-evaluation

# Restart main application
docker-compose restart rag-evaluation
```

---

## 📊 Performance Optimization

### For Better Performance:
1. **Allocate more memory** to Docker Desktop (8GB+)
2. **Use SSD storage** for Docker volumes
3. **Close unnecessary applications** during evaluation
4. **Use local model caching** (automatically handled)

### Resource Usage:
- **Memory**: ~4-6GB total
- **CPU**: Moderate during evaluation
- **Storage**: ~2-3GB for base system + your data

---

## 🔒 Security Notes

### Production Deployment:
1. **Change default ports** if deploying publicly
2. **Use environment-specific .env files**
3. **Enable authentication** for Gradio (see Gradio docs)
4. **Use Docker secrets** for API keys
5. **Enable HTTPS** with reverse proxy (nginx/traefik)

### API Key Security:
- Never commit `.env` file to version control
- Use different API keys for different environments
- Rotate API keys regularly
- Monitor API usage for anomalies

---

## 📚 Additional Resources

### Documentation:
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [Gradio Documentation](https://gradio.app/docs/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

### Getting Help:
1. Check the logs: `docker-compose logs rag-evaluation`
2. Review this guide and troubleshooting section
3. Test components individually using tools in `/tools/`
4. Check GitHub issues (if applicable)

---

## 🎯 Usage Examples

### Basic Query:
1. Open http://localhost:7860
2. Enter: "What was Apple's revenue in Q4 2019?"
3. Compare RAG vs Non-RAG responses

### Analysis Workflow:
1. Use main interface for interactive queries
2. Run batch analysis: `docker-compose --profile tools up rag-tools`
3. Check results in `./results/` directory
4. Review logs for performance insights

### Custom Queries:
- **Financial**: "Compare Apple and Microsoft's business models"
- **Technical**: "How do economic cycles affect tech companies?"
- **Specific**: "What did NVIDIA say about AI in their earnings call?"

---

🎉 **You're all set!** Your dockerized RAG evaluation system is ready to help others explore the fascinating world of RAG vs Non-RAG AI systems.
