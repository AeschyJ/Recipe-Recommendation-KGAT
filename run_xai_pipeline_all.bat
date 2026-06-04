@echo off
REM ===================================================
REM XAI 全模型 Fidelity 評估管線
REM 對 L=1 (full_kgat), L=2 (depth_2), L=3 (depth_3)
REM 三個模型執行 500 用戶的 Fidelity 評估
REM ===================================================

set VENV=.venv\Scripts\python.exe
set DATA_DIR=data/processed
set RAW_DATA_DIR=data/raw
set USERS_FILE=output/fidelity/sampled_users_500.json

echo ======================================
echo [Step 1] 採樣 500 位使用者 (seed=42)
echo ======================================
%VENV% scripts/sample_users_for_xai.py --num_users 500 --seed 42 --output %USERS_FILE%
if %errorlevel% neq 0 (
    echo 採樣失敗！
    exit /b 1
)

echo.
echo ======================================
echo [Step 2] L=3 Fidelity 評估 (depth_3)
echo   Model: 1_kgat_checkpoint_e3.pth
echo   n_hops: auto (=2)
echo ======================================
%VENV% src/evaluate_fidelity.py ^
    --model_path models/depth_3/1_kgat_checkpoint_e3.pth ^
    --data_dir %DATA_DIR% ^
    --raw_data_dir %RAW_DATA_DIR% ^
    --user_ids_file %USERS_FILE% ^
    --output_explain output/fidelity/depth_3/explanations.json ^
    --output_metrics output/fidelity/depth_3/metrics.json ^
    --top_k_paths 3

echo.
echo ======================================
echo [Step 3] L=2 Fidelity 評估 (depth_2)
echo   Model: 1_kgat_checkpoint_e1.pth
echo   n_hops: auto (=2)
echo ======================================
%VENV% src/evaluate_fidelity.py ^
    --model_path models/depth_2/1_kgat_checkpoint_e1.pth ^
    --data_dir %DATA_DIR% ^
    --raw_data_dir %RAW_DATA_DIR% ^
    --user_ids_file %USERS_FILE% ^
    --output_explain output/fidelity/depth_2/explanations.json ^
    --output_metrics output/fidelity/depth_2/metrics.json ^
    --top_k_paths 3

echo.
echo ======================================
echo [Step 4] L=1 Fidelity 評估 (full_kgat)
echo   Model: 2_kgat_checkpoint_e9.pth
echo   n_hops: 1 (match 1-layer model)
echo ======================================
%VENV% src/evaluate_fidelity.py ^
    --model_path models/full_kgat/2_kgat_checkpoint_e9.pth ^
    --data_dir %DATA_DIR% ^
    --raw_data_dir %RAW_DATA_DIR% ^
    --user_ids_file %USERS_FILE% ^
    --output_explain output/fidelity/full_kgat/explanations.json ^
    --output_metrics output/fidelity/full_kgat/metrics.json ^
    --top_k_paths 3 ^
    --n_hops 1

echo.
echo ======================================
echo [完成] 所有模型的 Fidelity 評估完成
echo ======================================
echo 結果位於：
echo   output/fidelity/depth_3/
echo   output/fidelity/depth_2/
echo   output/fidelity/full_kgat/
