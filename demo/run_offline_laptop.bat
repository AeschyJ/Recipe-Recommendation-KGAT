@echo off
REM ========================================================
REM Thesis Oral Defense Demo - Offline Laptop Launcher
REM Opens the standalone presentation web interface directly.
REM ========================================================

cd /d "%~dp0"
title KGAT Oral Defense Demo - Laptop Client

echo ========================================================
echo   KGAT Recipe Recommendation and Fidelity Verification
echo   Oral Defense Interactive Demo System - Laptop Client
echo ========================================================
echo.

set "DEMO_HTML=%~dp0web\index.html"
set "DEMO_JSON=%~dp0web\demo_cases.json"
set "DEMO_JS=%~dp0web\demo_cases_data.js"

if not exist "%DEMO_HTML%" (
    echo [ERROR] Demo HTML file was not found at %DEMO_HTML%
    pause
    exit /b 1
)

if not exist "%DEMO_JSON%" goto :GENERATE
if not exist "%DEMO_JS%" goto :GENERATE
goto :LAUNCH

:GENERATE
echo [WARNING] Offline demo cache is missing. Attempting to generate...
set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
)
if not exist "%PYTHON_EXE%" (
    where python >nul 2>nul
    if %errorlevel% equ 0 set "PYTHON_EXE=python"
)

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" "%~dp0export_demo_cases.py"
) else (
    echo [ERROR] Cannot generate demo cache because Python was not found.
    pause
    exit /b 1
)

:LAUNCH

echo Launching demo interface in your default browser...
start "" "%DEMO_HTML%"

echo.
echo Demo page has been opened.
echo You can present the five academic cases and 500 evaluation users offline.
echo.
timeout /t 3 >nul
exit /b 0
