@echo off
echo Starting KGAT Ablation Studies...

echo ==============================================
echo [Exp 1/5] Full KGAT (L=1, Attention + Bi-Interaction)
echo ==============================================
python src/train_att.py --epochs 30 --layers 64 --model_dir models/full_kgat --log_dir output/logs/full_kgat

echo ==============================================
echo [Exp 2/5] w/o Attention (KGAT-a, Bi-Interaction Only)
echo ==============================================
python src/train_bi_interaction.py --epochs 30 --layers 64 --model_dir models/wo_attn --log_dir output/logs/wo_attn

echo ==============================================
echo [Exp 3/5] w/o Knowledge Graph (Interaction Only)
echo ==============================================
python src/train_att.py --epochs 30 --layers 64 --without_kg --model_dir models/wo_kg --log_dir output/logs/wo_kg

echo ==============================================
echo [Exp 4/5] Depth Variation L=2
echo ==============================================
python src/train_att.py --epochs 30 --layers 64 64 --model_dir models/depth_2 --log_dir output/logs/depth_2

echo ==============================================
echo [Exp 5/5] Depth Variation L=3
echo ==============================================
python src/train_att.py --epochs 30 --layers 64 64 64 --model_dir models/depth_3 --log_dir output/logs/depth_3

echo ==============================================
echo All experiments completed!
echo ==============================================
