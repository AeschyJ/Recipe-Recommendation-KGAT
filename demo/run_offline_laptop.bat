@echo off
cd /d "%~dp0"
setlocal
if not exist "%~dp0..\run.bat" (
    echo [ERROR] Root launcher run.bat not found.
    if not defined CI pause
    exit /b 1
)
call "%~dp0..\run.bat" demo-offline %*
exit /b %errorlevel%
