"""訓練日誌簡化、資料驗證與指標彙整管線."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.pipelines import StepSpec, run_steps

LOG_PIPELINE_STEPS: List[StepSpec] = [
    ("Step 1/3: 簡化訓練日誌與評估輸出", "scripts/simplify_output_data.py", []),
    ("Step 2/3: 驗證統計與指標完整性", "scripts/validate.py", []),
    ("Step 3/3: 彙整實驗指標報表", "scripts/analyze_logs.py", []),
]


def run_log_pipeline(*, dry_run: bool = False) -> int:
    """依序執行 simplify_output_data -> validate -> analyze_logs."""
    code = run_steps(LOG_PIPELINE_STEPS, dry_run=dry_run)
    if code == 0:
        print("\n[完成] 日誌簡化與指標分析管線執行完畢。")
    return code


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="執行訓練日誌整理與分析管線")
    parser.add_argument("--dry-run", action="store_true", help="僅印出指令而不實際執行")
    args = parser.parse_args(argv)
    return run_log_pipeline(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
