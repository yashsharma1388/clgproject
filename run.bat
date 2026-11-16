@echo off
title Multimodal AI Assistant Launcher
echo ====================================================================
echo Starting Multimodal AI Assistant (Ollama + Streamlit)
echo ====================================================================

REM Start Ollama in background if not already running
start /b "" ollama serve >nul 2>&1

cd /d "%~dp0"
echo Launching Streamlit web application on http://localhost:8501 ...
.venv\Scripts\streamlit.exe run app/streamlit_app.py --server.port 8501
pause
