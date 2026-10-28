@echo off
REM ========================================================
REM Thesis Oral Defense Demo - Host Server Launcher
REM Forwarding to demo\run_server.bat
REM Port: 8501 (Port 8000 is strictly prohibited)
REM ========================================================

cd /d "%~dp0demo"
call run_server.bat
