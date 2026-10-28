@echo off
REM ========================================================
REM Thesis Oral Defense Demo - Offline Laptop Launcher
REM Opens the standalone presentation web interface directly.
REM ========================================================

cd /d "%~dp0"
title KGAT Oral Defense Demo - Laptop Client

set "DEMO_HTML=%~dp0web\index.html"
set "DEMO_JSON=%~dp0web\demo_cases.json"
set "DEMO_JS=%~dp0web\demo_cases_data.js"

if not exist "%DEMO_HTML%" (
    echo [ERROR] Demo HTML file was not found at %DEMO_HTML%
    echo Please make sure the demo\web directory exists.
    pause
    exit /b 1
)

if not exist "%DEMO_JSON%" goto :DO_GEN
if not exist "%DEMO_JS%" goto :DO_GEN
goto :DO_LAUNCH

:DO_GEN
echo [WARNING] Offline demo cache is missing. Attempting to generate...
set "PYTHON_EXE="
if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not defined PYTHON_EXE (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE (
    echo [ERROR] Cannot generate demo cache because Python was not found.
    pause
    exit /b 1
)

"%PYTHON_EXE%" "%~dp0export_demo_cases.py"

:DO_LAUNCH
start "" "%DEMO_HTML%"
exit
