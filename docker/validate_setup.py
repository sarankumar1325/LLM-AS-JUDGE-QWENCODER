#!/usr/bin/env python3
"""
Docker Setup Validation Script
Validates that the Docker setup is ready for deployment
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(cmd, capture_output=True):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=capture_output, 
            text=True, 
            timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def check_docker():
    """Check if Docker is installed and running."""
    print("🔍 Checking Docker installation...")
    
    # Check Docker version
    success, stdout, stderr = run_command("docker --version")
    if not success:
        print("❌ Docker is not installed or not in PATH")
        return False
    
    print(f"✅ Docker found: {stdout.strip()}")
    
    # Check Docker Compose
    success, stdout, stderr = run_command("docker-compose --version")
    if not success:
        print("❌ Docker Compose is not installed")
        return False
    
    print(f"✅ Docker Compose found: {stdout.strip()}")
    
    # Check if Docker daemon is running
    success, stdout, stderr = run_command("docker ps")
    if not success:
        print("❌ Docker daemon is not running. Please start Docker Desktop.")
        return False
    
    print("✅ Docker daemon is running")
    return True

def check_files():
    """Check if all necessary files exist."""
    print("\n📁 Checking required files...")
    
    required_files = [
        "docker/Dockerfile",
        "docker/docker-compose.yml",
        "docker/setup.sh",
        "docker/setup.bat",
        "gradio_app.py",
        "requirements.txt",
        ".env.example"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ Found: {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    return True

def check_env_file():
    """Check environment file setup."""
    print("\n🔐 Checking environment configuration...")
    
    if not os.path.exists(".env"):
        print("⚠️  .env file not found. Users will need to create it from .env.example")
        if os.path.exists(".env.example"):
            print("✅ .env.example template available")
        else:
            print("❌ .env.example template missing")
            return False
    else:
        print("✅ .env file exists")
        
        # Check for required keys
        with open(".env", "r") as f:
            env_content = f.read()
            
        if "GROQ_API_KEY" in env_content:
            print("✅ GROQ_API_KEY found in .env")
        else:
            print("⚠️  GROQ_API_KEY not found in .env (users need to add this)")
    
    return True

def validate_docker_build():
    """Validate that Docker build would work."""
    print("\n🔨 Validating Docker build setup...")
    
    # Check if we can validate the Dockerfile
    dockerfile_path = "docker/Dockerfile"
    if os.path.exists(dockerfile_path):
        with open(dockerfile_path, "r") as f:
            dockerfile_content = f.read()
            
        # Basic validation
        if "FROM python:" in dockerfile_content:
            print("✅ Valid Python base image in Dockerfile")
        else:
            print("❌ Invalid or missing Python base image")
            return False
            
        if "COPY requirements.txt" in dockerfile_content:
            print("✅ Requirements.txt copy instruction found")
        else:
            print("❌ Requirements.txt copy instruction missing")
            return False
            
        if "EXPOSE 7860" in dockerfile_content:
            print("✅ Correct port exposed for Gradio")
        else:
            print("❌ Port 7860 not exposed for Gradio")
            return False
    
    # Check docker-compose.yml
    compose_path = "docker/docker-compose.yml"
    if os.path.exists(compose_path):
        with open(compose_path, "r") as f:
            compose_content = f.read()
            
        if "7860:7860" in compose_content:
            print("✅ Correct port mapping in docker-compose.yml")
        else:
            print("❌ Port mapping missing in docker-compose.yml")
            return False
    
    return True

def check_project_structure():
    """Check if project structure is ready for Docker."""
    print("\n📂 Checking project structure...")
    
    required_dirs = [
        "src",
        "config", 
        "tools",
        "scripts",
        "docs"
    ]
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory exists: {dir_name}")
        else:
            print(f"❌ Missing directory: {dir_name}")
            return False
    
    # Check key source files
    key_files = [
        "src/models/groq_client.py",
        "config/settings.py",
        "tools/analyze_rag_vs_non_rag.py"
    ]
    
    for file_path in key_files:
        if os.path.exists(file_path):
            print(f"✅ Key file exists: {file_path}")
        else:
            print(f"⚠️  Key file missing: {file_path} (may affect functionality)")
    
    return True

def generate_deployment_instructions():
    """Generate final deployment instructions."""
    print("\n📋 DEPLOYMENT INSTRUCTIONS FOR USERS")
    print("=" * 50)
    
    instructions = """
🚀 To deploy this RAG Evaluation System on any machine:

1. PREREQUISITES:
   - Install Docker Desktop: https://www.docker.com/products/docker-desktop
   - Get Groq API key: https://console.groq.com/
   - Ensure 8GB+ RAM and 5GB+ disk space

2. SETUP COMMANDS:
   git clone <your-repository>
   cd rag-evaluation-system
   
   # Windows:
   docker\\setup.bat
   
   # Linux/Mac:
   chmod +x docker/setup.sh
   ./docker/setup.sh

3. CONFIGURATION:
   - Edit .env file: add GROQ_API_KEY=your_actual_key_here
   - Wait for Docker images to build (5-10 minutes first time)

4. ACCESS:
   - Web Interface: http://localhost:7860
   - Vector Database: http://localhost:8001
   - Check logs: docker-compose logs -f rag-evaluation

5. USAGE:
   - Ask questions about companies: "What was Apple's revenue in Q4 2019?"
   - Compare RAG vs Non-RAG responses
   - Run analysis: docker-compose --profile tools up rag-tools

6. MAINTENANCE:
   - Stop: docker-compose down
   - Start: docker-compose up -d  
   - Update: git pull && docker-compose build && docker-compose up -d
"""
    
    print(instructions)

def main():
    """Main validation function."""
    print("🐳 RAG Evaluation System - Docker Deployment Validation")
    print("=" * 60)
    
    all_checks_passed = True
    
    # Run all validation checks
    checks = [
        ("Docker Installation", check_docker),
        ("Required Files", check_files),
        ("Environment Setup", check_env_file),
        ("Docker Configuration", validate_docker_build),
        ("Project Structure", check_project_structure)
    ]
    
    for check_name, check_func in checks:
        try:
            if not check_func():
                all_checks_passed = False
        except Exception as e:
            print(f"❌ {check_name} failed with error: {e}")
            all_checks_passed = False
    
    print("\n" + "=" * 60)
    
    if all_checks_passed:
        print("🎉 ALL VALIDATION CHECKS PASSED!")
        print("✅ Your system is ready for Docker deployment")
        print("✅ Users can run this on their machines using Docker")
        generate_deployment_instructions()
    else:
        print("❌ SOME VALIDATION CHECKS FAILED")
        print("🔧 Please fix the issues above before deployment")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
