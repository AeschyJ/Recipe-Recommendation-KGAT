@echo off
pushd "%~dp0"
echo Starting Baseline Model Experiments...

if not exist .venv\Scripts\python.exe (
    echo [Error] .venv\Scripts\python.exe not found!
    echo Please ensure the project was initialized in this folder.
    popd
    exit /b 1
)

@REM echo ==============================================
@REM echo [Baseline 1/6] BPR-MF (Matrix Factorization)
@REM echo ==============================================
@REM .venv\Scripts\python.exe src/train_baseline.py --model BPR-MF --epochs 10 --batch_size 1024 --use_bf16 --experiment_id 01

echo ==============================================
echo [Baseline 1/2] LightGCN (Pure GNN)
echo ==============================================
.venv\Scripts\python.exe src/train_baseline.py --model LightGCN --epochs 100 --batch_size 1024 --use_bf16 --experiment_id 02
echo ==============================================
echo [Baseline 2/2] LightGCN (Pure GNN)
echo ==============================================
.venv\Scripts\python.exe src/train_baseline.py --model LightGCN --epochs 100 --batch_size 1024 --use_bf16 --experiment_id 03

@REM echo ==============================================
@REM echo [Baseline 1/3] NFM (Neural Factorization Machine)
@REM echo ==============================================
@REM .venv\Scripts\python.exe src/train_baseline.py --model NFM --epochs 100 --batch_size 1024 --use_bf16 --experiment_id 01

@REM echo ==============================================
@REM echo [Baseline 2/3] NFM (Neural Factorization Machine)
@REM echo ==============================================
@REM .venv\Scripts\python.exe src/train_baseline.py --model NFM --epochs 100 --batch_size 1024 --use_bf16 --experiment_id 02

@REM echo ==============================================
@REM echo [Baseline 3/3] NFM (Neural Factorization Machine)
@REM echo ==============================================
@REM .venv\Scripts\python.exe src/train_baseline.py --model NFM --epochs 100 --batch_size 1024 --use_bf16 --experiment_id 03

echo ==============================================
echo All baseline experiments completed!
echo Results are saved in models/baseline and output/logs/baseline
echo ==============================================
popd
