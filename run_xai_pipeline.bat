@echo off
chcp 65001 >nul
pushd "%~dp0"
echo Starting XAI Pipeline...

if not exist .venv\Scripts\python.exe (
    echo [Error] .venv\Scripts\python.exe not found!
    echo Please ensure the project was initialized in this folder.
    popd
    exit /b 1
)

set NUM_USERS=100
set MODEL_PATH=models\depth_3\kgat_checkpoint_e2.pth
set OUTPUT_DIR=output\fidelity\depth_3
set SAMPLED_USERS_FILE=%OUTPUT_DIR%\sampled_users.json
set EXPLANATION_OUT=%OUTPUT_DIR%\explanations.json
set METRICS_OUT=%OUTPUT_DIR%\metrics.json

echo [Step 1] Sampling users...
.venv\Scripts\python.exe scripts\sample_users_for_xai.py --num_users %NUM_USERS% --output %SAMPLED_USERS_FILE%

if %ERRORLEVEL% neq 0 (
    echo [Error] Failed to sample users!
    popd
    exit /b 1
)

echo.
echo [Step 2] Extracting top-1 explanations and verifying Fidelity...
.venv\Scripts\python.exe src\evaluate_fidelity.py --model_path %MODEL_PATH% --user_ids_file %SAMPLED_USERS_FILE% --output_explain %EXPLANATION_OUT% --output_metrics %METRICS_OUT%

if %ERRORLEVEL% neq 0 (
    echo [Error] Fidelity evaluation failed!
    popd
    exit /b 1
)

echo.
echo ==============================================
echo XAI Pipeline completed!
echo Results are saved in %OUTPUT_DIR%
echo ==============================================

popd
