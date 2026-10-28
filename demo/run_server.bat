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

set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
)

if not exist "%PYTHON_EXE%" (
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        set "PYTHON_EXE=python"
    ) else (
        echo [ERROR] Python environment was not found.
        echo Please ensure .venv exists or python is in PATH.
        pause
        exit /b 1
    )
)

echo [1/2] Checking demo pre-extracted cache...
if not exist "%~dp0web\demo_cases.json" (
    echo Generating demo cases cache...
    "%PYTHON_EXE%" "%~dp0export_demo_cases.py"
) else if not exist "%~dp0web\demo_cases_data.js" (
    echo Generating demo cases cache...
    "%PYTHON_EXE%" "%~dp0export_demo_cases.py"
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

"%PYTHON_EXE%" "%~dp0server.py" --port 8501 --host 0.0.0.0

pause
