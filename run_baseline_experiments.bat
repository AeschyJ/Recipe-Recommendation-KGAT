@echo off
pushd "%~dp0"
echo Starting Baseline Model Experiments...

if not exist .venv\Scripts\python.exe (
    echo [Error] .venv\Scripts\python.exe not found!
    echo Please ensure the project was initialized in this folder.
    popd
    exit /b 1
)

echo ==============================================
echo [Baseline 1/6] BPR-MF (Matrix Factorization)
echo ==============================================
.venv\Scripts\python.exe src/train_baseline.py --model BPR-MF --epochs 10 --batch_size 1024 --use_bf16 --experiment_id 02

echo ==============================================
echo [Baseline 2/6] LightGCN (Pure GNN)
echo ==============================================
.venv\Scripts\python.exe src/train_baseline.py --model LightGCN --epochs 10 --batch_size 1024 --use_bf16 --experiment_id 02

echo ==============================================
echo [Baseline 3/6] NFM (Neural Factorization Machine)
echo ==============================================
.venv\Scripts\python.exe src/train_baseline.py --model NFM --epochs 10 --batch_size 1024 --use_bf16 --experiment_id 02

echo ==============================================
echo [Baseline 4/6] BPR-MF (Matrix Factorization)
echo ==============================================
.venv\Scripts\python.exe src/train_baseline.py --model BPR-MF --epochs 10 --batch_size 1024 --use_bf16 --experiment_id 03

echo ==============================================
echo [Baseline 5/6] LightGCN (Pure GNN)
echo ==============================================
.venv\Scripts\python.exe src/train_baseline.py --model LightGCN --epochs 10 --batch_size 1024 --use_bf16 --experiment_id 03

echo ==============================================
echo [Baseline 6/6] NFM (Neural Factorization Machine)
echo ==============================================
.venv\Scripts\python.exe src/train_baseline.py --model NFM --epochs 10 --batch_size 1024 --use_bf16 --experiment_id 03

echo ==============================================
echo All baseline experiments completed!
echo Results are saved in models/baseline and output/logs/baseline
echo ==============================================
popd
