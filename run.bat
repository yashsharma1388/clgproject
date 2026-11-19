@echo off
title Multimodal AI Assistant Launcher
echo ====================================================================
echo Starting Multimodal AI Assistant (Ollama + Streamlit + Cloudflare)
echo ====================================================================

REM Start Ollama in background if not already running
start /b "" ollama serve >nul 2>&1

cd /d "%~dp0"

REM Start Cloudflare public tunnel in background
if exist "cloudflared.exe" (
    start /b "" "cloudflared.exe" tunnel --url http://localhost:8501 >nul 2>&1
)

echo Launching Streamlit web application on http://localhost:8501 ...
.venv\Scripts\streamlit.exe run app/streamlit_app.py --server.port 8501
pause
