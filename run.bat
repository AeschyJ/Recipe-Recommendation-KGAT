@echo off
cd /d "%~dp0"
setlocal
title KGAT Experiment Launcher
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
if "%PORT%"=="" set "PORT=8501"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py %*
) else (
    where uv >nul 2>nul
    if not errorlevel 1 (
        uv run python main.py %*
    ) else (
        python main.py %*
    )
)

set "EXIT_CODE=%errorlevel%"
if %EXIT_CODE% neq 0 (
    echo.
    echo ======================================================================
    echo [CRASH] Launcher terminated abnormally with exit code: %EXIT_CODE%
    echo Timestamp: %date% %time%
    echo Details recorded in: launcher_crash.log
    echo ======================================================================
    echo [%date% %time%] main.py %* failed with exit code: %EXIT_CODE% >> "%~dp0launcher_crash.log"
    if not defined CI pause
    exit /b %EXIT_CODE%
)
exit /b 0
