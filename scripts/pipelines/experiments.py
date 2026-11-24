"""KGAT 主消融與 Baseline 實驗管線."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.pipelines import StepSpec, run_steps

BASELINE_DEFAULT_STEPS: List[StepSpec] = [
    (
        "Baseline 1/2: LightGCN (experiment_id=02)",
        "src/train_baseline.py",
        ["--model", "LightGCN", "--epochs", "100", "--batch_size", "1024", "--use_bf16", "--experiment_id", "02"],
    ),
    (
        "Baseline 2/2: LightGCN (experiment_id=03)",
        "src/train_baseline.py",
        ["--model", "LightGCN", "--epochs", "100", "--batch_size", "1024", "--use_bf16", "--experiment_id", "03"],
    ),
]

BASELINE_ALL_STEPS: List[StepSpec] = [
    (
        "Baseline 1/6: BPR-MF (experiment_id=01)",
        "src/train_baseline.py",
        ["--model", "BPR-MF", "--epochs", "10", "--batch_size", "1024", "--use_bf16", "--experiment_id", "01"],
    ),
    *BASELINE_DEFAULT_STEPS,
    (
        "Baseline 4/6: NFM (experiment_id=01)",
        "src/train_baseline.py",
        ["--model", "NFM", "--epochs", "100", "--batch_size", "1024", "--use_bf16", "--experiment_id", "01"],
    ),
    (
        "Baseline 5/6: NFM (experiment_id=02)",
        "src/train_baseline.py",
        ["--model", "NFM", "--epochs", "100", "--batch_size", "1024", "--use_bf16", "--experiment_id", "02"],
    ),
    (
        "Baseline 6/6: NFM (experiment_id=03)",
        "src/train_baseline.py",
        ["--model", "NFM", "--epochs", "100", "--batch_size", "1024", "--use_bf16", "--experiment_id", "03"],
    ),
]

ABLATION_DEFAULT_STEPS: List[StepSpec] = [
    (
        "Exp 3/5: w/o Knowledge Graph (Interaction Only)",
        "src/train.py",
        [
            "--experiment_id",
            "3",
            "--epochs",
            "100",
            "--layers",
            "64",
            "--without_kg",
            "--model_dir",
            "models/wo_kg",
            "--log_dir",
            "output/logs/wo_kg",
            "--use_bf16",
            "--no_compile",
        ],
    ),
    (
        "Exp 4/5: Depth Variation L=2",
        "src/train.py",
        [
            "--experiment_id",
            "3",
            "--epochs",
            "100",
            "--layers",
            "64",
            "64",
            "--model_dir",
            "models/depth_2",
            "--log_dir",
            "output/logs/depth_2",
            "--use_bf16",
            "--no_compile",
        ],
    ),
]

ABLATION_ALL_STEPS: List[StepSpec] = [
    (
        "Exp 1/5: Full KGAT (L=1, Attention + Bi-Interaction)",
        "src/train.py",
        [
            "--epochs",
            "10",
            "--layers",
            "64",
            "--model_dir",
            "models/full_kgat",
            "--log_dir",
            "output/logs/full_kgat",
            "--use_bf16",
            "--no_compile",
        ],
    ),
    (
        "Exp 2/5: w/o Attention (Bi-Interaction Only)",
        "src/train.py",
        [
            "--experiment_id",
            "1",
            "--no_attention",
            "--epochs",
            "100",
            "--layers",
            "64",
            "--model_dir",
            "models/wo_attn",
            "--log_dir",
            "output/logs/wo_attn",
            "--use_bf16",
            "--no_compile",
        ],
    ),
    *ABLATION_DEFAULT_STEPS,
    (
        "Exp 5/5: Depth Variation L=3",
        "src/train.py",
        [
            "--epochs",
            "10",
            "--layers",
            "64",
            "64",
            "64",
            "--model_dir",
            "models/depth_3",
            "--log_dir",
            "output/logs/depth_3",
            "--use_bf16",
            "--no_compile",
            "--resume",
            "models/depth_3/2_kgat_checkpoint_e3.pth",
        ],
    ),
]


def run_experiments(mode: str, *, include_all: bool = False, dry_run: bool = False) -> int:
    """執行 Baseline 或主消融實驗管線."""
    if mode == "baseline":
        steps = BASELINE_ALL_STEPS if include_all else BASELINE_DEFAULT_STEPS
    elif mode == "main":
        steps = ABLATION_ALL_STEPS if include_all else ABLATION_DEFAULT_STEPS
    else:
        raise ValueError(f"不支援的實驗模式: {mode}")

    code = run_steps(steps, dry_run=dry_run)
    if code == 0:
        print("\n[完成] 實驗管線執行結束。")
    return code


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="執行 KGAT 消融與 Baseline 實驗管線")
    parser.add_argument("--mode", choices=["baseline", "main"], required=True, help="實驗模式")
    parser.add_argument("--all", action="store_true", help="執行完整對照清單（含原註解實驗組）")
    parser.add_argument("--dry-run", action="store_true", help="僅印出指令而不實際執行訓練")
    args = parser.parse_args(argv)
    return run_experiments(args.mode, include_all=args.all, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
