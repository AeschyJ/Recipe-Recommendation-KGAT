@echo off
pushd "%~dp0"
echo Starting KGAT Ablation Studies...

if not exist .venv\Scripts\python.exe (
    echo [Error] .venv\Scripts\python.exe not found!
    echo Please ensure the project was initialized in this folder.
    popd
    exit /b 1
)

@REM echo ==============================================
@REM echo [Exp 1/5] Full KGAT (L=1, Attention + Bi-Interaction)
@REM echo ==============================================
@REM .venv\Scripts\python.exe src/train.py --epochs 10 --layers 64 --model_dir models/full_kgat --log_dir output/logs/full_kgat --use_bf16 --no_compile

@REM echo ==============================================
@REM echo [Exp 1/3] w/o Attention (KGAT, Bi-Interaction Only)
@REM echo ==============================================
@REM .venv\Scripts\python.exe src/train.py --experiment_id 1 --no_attention --epochs 100 --layers 64 --model_dir models/wo_attn --log_dir output/logs/wo_attn --use_bf16 --no_compile

echo ==============================================
echo [Exp 3/5] w/o Knowledge Graph (Interaction Only)
echo ==============================================
.venv\Scripts\python.exe src/train.py --experiment_id 3 --epochs 100 --layers 64 --without_kg --model_dir models/wo_kg --log_dir output/logs/wo_kg --use_bf16 --no_compile

echo ==============================================
echo [Exp 4/5] Depth Variation L=2
echo ==============================================
.venv\Scripts\python.exe src/train.py --experiment_id 3 --epochs 100 --layers 64 64 --model_dir models/depth_2 --log_dir output/logs/depth_2 --use_bf16 --no_compile

@REM echo ==============================================
@REM echo [Exp 5/5] Depth Variation L=3
@REM echo ==============================================
@REM .venv\Scripts\python.exe src/train.py --epochs 10 --layers 64 64 64 --model_dir models/depth_3 --log_dir output/logs/depth_3 --use_bf16 --no_compile --resume models/depth_3/2_kgat_checkpoint_e3.pth

echo ==============================================
echo All experiments completed!
echo ==============================================
popd
