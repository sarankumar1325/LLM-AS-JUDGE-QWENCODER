@echo off
echo Starting RAG Evaluation Streamlit App...
echo.

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Installing Streamlit requirements...
    pip install -r streamlit_requirements.txt
)

echo.
echo Launching Streamlit app...
echo Access the app at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the app
echo.

streamlit run streamlit_app.py --server.port=8501 --server.address=localhost
