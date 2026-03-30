@echo off
pushd "%~dp0"
echo Starting KGAT Ablation Studies...

if not exist .venv\Scripts\python.exe (
    echo [Error] .venv\Scripts\python.exe not found!
    echo Please ensure the project was initialized in this folder.
    popd
    exit /b 1
)

echo ==============================================
echo [Exp 1/5] Full KGAT (L=1, Attention + Bi-Interaction)
echo ==============================================
.venv\Scripts\python.exe src/train_att.py --epochs 10 --layers 64 --model_dir models/full_kgat --log_dir output/logs/full_kgat --use_bf16 --no_compile

echo ==============================================
echo [Exp 2/5] w/o Attention (KGAT-a, Bi-Interaction Only)
echo ==============================================
.venv\Scripts\python.exe src/train_bi_interaction.py --epochs 10 --layers 64 --model_dir models/wo_attn --log_dir output/logs/wo_attn --use_bf16

echo ==============================================
echo [Exp 3/5] w/o Knowledge Graph (Interaction Only)
echo ==============================================
.venv\Scripts\python.exe src/train_att.py --epochs 10 --layers 64 --without_kg --model_dir models/wo_kg --log_dir output/logs/wo_kg --use_bf16 --no_compile

echo ==============================================
echo [Exp 4/5] Depth Variation L=2
echo ==============================================
.venv\Scripts\python.exe src/train_att.py --epochs 10 --layers 64 64 --model_dir models/depth_2 --log_dir output/logs/depth_2 --use_bf16 --no_compile

echo ==============================================
echo [Exp 5/5] Depth Variation L=3
echo ==============================================
.venv\Scripts\python.exe src/train_att.py --epochs 10 --layers 64 64 64 --model_dir models/depth_3 --log_dir output/logs/depth_3 --use_bf16 --no_compile

echo ==============================================
echo All experiments completed!
echo ==============================================
popd
