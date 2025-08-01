# 🐳 RAG Evaluation System - Docker Edition

**Compare RAG vs Non-RAG AI responses using a dockerized evaluation system!**

## 🚀 Quick Start (3 steps)

### 1. Prerequisites
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
- Get a [Groq API key](https://console.groq.com/) (free)

### 2. Setup
```bash
# Clone and setup
git clone <your-repo-url>
cd rag-evaluation-system

# Windows users
docker\setup.bat

# Linux/Mac users  
chmod +x docker/setup.sh && ./docker/setup.sh
```

### 3. Configure
- Edit `.env` file: add `GROQ_API_KEY=your_key_here`
- Open http://localhost:7860
- Start comparing RAG vs Non-RAG!

## 🌟 What You Get

- **Interactive Web Interface** - Compare RAG and Non-RAG responses
- **17,664 Financial Documents** - Real company data from Apple, Microsoft, NVIDIA, etc.
- **Advanced AI Models** - Qwen 3-32B via Groq API
- **Performance Analysis** - Understand when each approach wins
- **Complete Isolation** - Everything runs in Docker containers

## 📋 Basic Commands

```bash
# Start system
docker-compose up -d

# View logs
docker-compose logs -f rag-evaluation

# Stop system
docker-compose down

# Run analysis tools
docker-compose --profile tools up rag-tools
```

## 🎯 Example Queries

Try these in the web interface:

- **"What was Apple's revenue in Q4 2019?"** *(RAG typically wins)*
- **"How do economic cycles affect tech companies?"** *(Non-RAG often wins)*
- **"Compare business models of Apple vs Microsoft"** *(Interesting comparison!)*

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 7860 in use | Change port in `docker-compose.yml` |
| Out of memory | Increase Docker Desktop memory to 8GB+ |
| API errors | Check `.env` file has correct `GROQ_API_KEY` |
| Can't connect | Run `docker-compose restart rag-evaluation` |

## 📚 Documentation

- **[Complete Docker Guide](docs/DOCKER_GUIDE.md)** - Detailed setup and troubleshooting
- **[System Guide](docs/FINAL_SYSTEM_GUIDE.md)** - Understanding RAG vs Non-RAG
- **[Project Structure](PROJECT_STRUCTURE.md)** - Code organization

## 🎉 Ready to Explore!

Once running, you'll have a powerful system to:
- Compare RAG and Non-RAG AI responses
- Understand when each approach works best  
- Analyze real financial data with AI
- Learn about modern AI evaluation techniques

**Access your system:** http://localhost:7860

Happy evaluating! 🚀
