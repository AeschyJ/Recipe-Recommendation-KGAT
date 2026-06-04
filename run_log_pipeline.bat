@echo off
set PYTHON_CMD=.venv\Scripts\python.exe

echo ========================================================
echo Starting Log Processing Pipeline
echo ========================================================

echo.
echo [1/3] Running simplify_output_data.py...
%PYTHON_CMD% scripts\simplify_output_data.py
if %ERRORLEVEL% neq 0 (
    echo Error occurred during simplify_output_data.py!
    exit /b %ERRORLEVEL%
)

echo.
echo [2/3] Running validate.py...
%PYTHON_CMD% scripts\validate.py
if %ERRORLEVEL% neq 0 (
    echo Error occurred during validate.py!
    exit /b %ERRORLEVEL%
)

echo.
echo [3/3] Running analyze_logs.py...
%PYTHON_CMD% scripts\analyze_logs.py
if %ERRORLEVEL% neq 0 (
    echo Error occurred during analyze_logs.py!
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================================
echo Pipeline Finished Successfully!
echo ========================================================
pause
