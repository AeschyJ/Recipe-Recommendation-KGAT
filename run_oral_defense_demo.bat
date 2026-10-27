@echo off
REM ========================================================
REM Thesis Oral Defense Demo - Host Server Launcher
REM Port: 8501 (Port 8000 is strictly prohibited)
REM ========================================================

cd /d "%~dp0"
title KGAT Oral Defense Demo Server

echo ========================================================
echo   KGAT Recipe Recommendation and Fidelity Verification
echo   Oral Defense Interactive Demo System - Server
echo ========================================================
echo.

set PYTHON_EXE=.venv\Scripts\python.exe

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Virtual environment python was not found at %PYTHON_EXE%
    echo Please make sure .venv is created.
    pause
    exit /b 1
)

echo [1/2] Checking demo pre-extracted cache...
if not exist "src\demo\web\demo_cases.json" (
    echo Generating demo cases cache...
    "%PYTHON_EXE%" src\demo\export_demo_cases.py
) else if not exist "src\demo\web\demo_cases_data.js" (
    echo Generating demo cases cache...
    "%PYTHON_EXE%" src\demo\export_demo_cases.py
)

echo.
echo [2/2] Checking Cloudflare Tunnel utility...
where cloudflared >nul 2>nul
if %errorlevel% equ 0 (
    echo Cloudflared is detected on this system.
    echo To publish the demo via Cloudflare Tunnel, run:
    echo   cloudflared tunnel --url http://localhost:8501
    echo.
) else (
    echo Cloudflared is not installed or not in PATH.
    echo If accessing from a local network, visit:
    echo   http://localhost:8501
    echo.
)

echo ========================================================
echo Starting FastAPI Demo Server on port 8501...
echo Open your browser at: http://localhost:8501
echo Press Ctrl+C to terminate the server.
echo ========================================================
echo.

"%PYTHON_EXE%" src\demo\server.py --port 8501 --host 0.0.0.0

pause
