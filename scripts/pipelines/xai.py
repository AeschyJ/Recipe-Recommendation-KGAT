"""XAI 解釋路徑萃取與 Fidelity 量化評估管線."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.pipelines import StepSpec, run_steps

XAI_SINGLE_STEPS: List[StepSpec] = [
    (
        "Step 1/2: 抽樣 100 位使用者",
        "scripts/sample_users_for_xai.py",
        ["--num_users", "100", "--output", "output/fidelity/depth_3/sampled_users.json"],
    ),
    (
        "Step 2/2: 萃取 Top-1 解釋與 Fidelity 驗證 (depth_3)",
        "src/evaluate_fidelity.py",
        [
            "--model_path",
            "models/depth_3/kgat_checkpoint_e2.pth",
            "--user_ids_file",
            "output/fidelity/depth_3/sampled_users.json",
            "--output_explain",
            "output/fidelity/depth_3/explanations.json",
            "--output_metrics",
            "output/fidelity/depth_3/metrics.json",
        ],
    ),
]

XAI_ALL_STEPS: List[StepSpec] = [
    (
        "Step 1/4: 抽樣 500 位使用者 (seed=42)",
        "scripts/sample_users_for_xai.py",
        ["--num_users", "500", "--seed", "42", "--output", "output/fidelity/sampled_users_500.json"],
    ),
    (
        "Step 2/4: L=3 Fidelity 評估 (depth_3)",
        "src/evaluate_fidelity.py",
        [
            "--model_path",
            "models/depth_3/1_kgat_checkpoint_e3.pth",
            "--data_dir",
            "data/processed",
            "--raw_data_dir",
            "data/raw",
            "--user_ids_file",
            "output/fidelity/sampled_users_500.json",
            "--output_explain",
            "output/fidelity/depth_3/explanations.json",
            "--output_metrics",
            "output/fidelity/depth_3/metrics.json",
            "--top_k_paths",
            "3",
        ],
    ),
    (
        "Step 3/4: L=2 Fidelity 評估 (depth_2)",
        "src/evaluate_fidelity.py",
        [
            "--model_path",
            "models/depth_2/1_kgat_checkpoint_e1.pth",
            "--data_dir",
            "data/processed",
            "--raw_data_dir",
            "data/raw",
            "--user_ids_file",
            "output/fidelity/sampled_users_500.json",
            "--output_explain",
            "output/fidelity/depth_2/explanations.json",
            "--output_metrics",
            "output/fidelity/depth_2/metrics.json",
            "--top_k_paths",
            "3",
        ],
    ),
    (
        "Step 4/4: L=1 Fidelity 評估 (full_kgat)",
        "src/evaluate_fidelity.py",
        [
            "--model_path",
            "models/full_kgat/2_kgat_checkpoint_e9.pth",
            "--data_dir",
            "data/processed",
            "--raw_data_dir",
            "data/raw",
            "--user_ids_file",
            "output/fidelity/sampled_users_500.json",
            "--output_explain",
            "output/fidelity/full_kgat/explanations.json",
            "--output_metrics",
            "output/fidelity/full_kgat/metrics.json",
            "--top_k_paths",
            "3",
            "--n_hops",
            "1",
        ],
    ),
]


def run_xai(*, run_all: bool = False, dry_run: bool = False) -> int:
    """執行單模型或跨層深全模型 (500 人) Fidelity 評估."""
    steps = XAI_ALL_STEPS if run_all else XAI_SINGLE_STEPS
    code = run_steps(steps, dry_run=dry_run)
    if code == 0:
        print("\n[完成] XAI Fidelity 評估完成，結果位於 output/fidelity/。")
    return code


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="執行 XAI 解釋與 Fidelity 評估管線")
    parser.add_argument("--all", action="store_true", help="覆蓋 L=1/L=2/L=3 500 位使用者評估")
    parser.add_argument("--dry-run", action="store_true", help="僅印出指令而不實際執行")
    args = parser.parse_args(argv)
    return run_xai(run_all=args.all, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
