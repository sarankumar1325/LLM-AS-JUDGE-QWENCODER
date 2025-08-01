#!/usr/bin/env python3
"""
Launch script for RAG Evaluation Streamlit App
Cross-platform launcher for the web application.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    try:
        import streamlit
        import plotly
        import pandas
        print("✅ Core requirements found")
        return True
    except ImportError as e:
        print(f"❌ Missing requirements: {e}")
        return False

def install_requirements():
    """Install Streamlit requirements."""
    print("📦 Installing Streamlit requirements...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "-r", "streamlit_requirements.txt"
        ], check=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def launch_streamlit():
    """Launch the Streamlit app."""
    print("🚀 Starting RAG Evaluation Streamlit App...")
    print(f"📍 Working directory: {os.getcwd()}")
    print("🌐 Access the app at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--server.headless=false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Streamlit app stopped by user")
    except FileNotFoundError:
        print("❌ Streamlit not found. Installing...")
        if install_requirements():
            launch_streamlit()
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")

def main():
    """Main launcher function."""
    print("=" * 50)
    print("🎯 RAG Evaluation System - Web Interface")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("streamlit_app.py").exists():
        print("❌ streamlit_app.py not found in current directory")
        print("📁 Please run this script from the project root directory")
        return
    
    # Check requirements
    if not check_requirements():
        print("📦 Installing missing requirements...")
        if not install_requirements():
            print("❌ Failed to install requirements. Please run manually:")
            print("   pip install -r streamlit_requirements.txt")
            return
    
    # Launch app
    launch_streamlit()

if __name__ == "__main__":
    main()
