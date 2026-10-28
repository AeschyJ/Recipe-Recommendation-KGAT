@echo off
REM ========================================================
REM Thesis Oral Defense Demo - Offline Laptop Launcher
REM Forwarding to demo\run_offline_laptop.bat
REM ========================================================

cd /d "%~dp0demo"
if not exist "run_offline_laptop.bat" (
    echo [ERROR] Cannot find demo\run_offline_laptop.bat
    pause
    exit /b 1
)
call run_offline_laptop.bat
