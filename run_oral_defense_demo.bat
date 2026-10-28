@echo off
REM ========================================================
REM Thesis Oral Defense Demo - Host Server Launcher
REM Forwarding to demo\run_server.bat
REM Port: 8501 (Port 8000 is strictly prohibited)
REM ========================================================

cd /d "%~dp0demo"
if not exist "run_server.bat" (
    echo [ERROR] Cannot find demo\run_server.bat
    pause
    exit /b 1
)
call run_server.bat
