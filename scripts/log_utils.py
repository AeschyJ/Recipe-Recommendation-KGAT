"""訓練日誌解析與檢查點資訊共用工具模組.

集中解析 output/logs 下各次訓練紀錄的最佳 epoch 與 HR@20 指標，
供檢查點分析、清理與備份腳本共用。
"""

from __future__ import annotations

import glob
import os
import re
from typing import Dict, Optional, Tuple

DEFAULT_MODELS_DIR = "models"
DEFAULT_LOGS_DIR = "output/logs"

_MODEL_DIR_RE = re.compile(r"model_dir='([^']+)'")
_EXP_ID_RE = re.compile(r"experiment_id='([^']+)'")
_SAVED_CKPT_RE = re.compile(r"Saved checkpoint:\s*([^\s\n]+)")
_CKPT_RUN_RE = re.compile(r"(\d+)_kgat_checkpoint")
_CKPT_FULL_RE = re.compile(r"(\d+)_kgat_checkpoint_e(\d+)\.pth")
_EVAL_LINE_RE = re.compile(
    r"Epoch (\d+) Evaluation - (?:HR|Recall)@\[10,20,50\]:\s*\[[^,]+,\s*([^,\]]+)"
)


def extract_checkpoint_info(filename: str) -> Optional[Tuple[int, int]]:
    """從檢查點檔名萃取 (run_id, epoch)；格式不符時回傳 None."""
    match = _CKPT_FULL_RE.match(filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


extract_info = extract_checkpoint_info


def _infer_run_identity(
    log_path: str,
    content: str,
    logs_dir: str,
    models_dir: str,
) -> Tuple[str, int]:
    """依序從 Args 紀錄、Saved checkpoint 路徑與日誌子目錄推斷 (model_basename, run_id)."""
    model_dir_match = _MODEL_DIR_RE.search(content)
    exp_id_match = _EXP_ID_RE.search(content)

    model_dir: Optional[str] = model_dir_match.group(1) if model_dir_match else None
    run_id: Optional[int] = None
    if exp_id_match:
        try:
            run_id = int(exp_id_match.group(1))
        except ValueError:
            run_id = None

    if not run_id or not model_dir:
        for ckpt_path in _SAVED_CKPT_RE.findall(content):
            path_parts = ckpt_path.replace("\\", "/").split("/")
            if len(path_parts) < 2:
                continue
            parsed_model, filename = path_parts[-2], path_parts[-1]
            filename_match = _CKPT_RUN_RE.match(filename)
            if filename_match:
                if not model_dir:
                    model_dir = os.path.join(models_dir, parsed_model)
                if not run_id:
                    run_id = int(filename_match.group(1))
                break

    if not model_dir:
        log_rel = os.path.relpath(log_path, logs_dir)
        parts = log_rel.replace("\\", "/").split("/")
        model_dir = os.path.join(models_dir, parts[0]) if len(parts) >= 2 else models_dir

    if not run_id:
        run_id = 1

    model_basename = os.path.basename(model_dir.replace("\\", "/"))
    return model_basename, run_id


def _extract_best_epoch_from_content(content: str) -> Optional[Tuple[int, float]]:
    """從單一日誌全文提取最高 HR@20 對應之 (best_epoch, best_hr20)."""
    best_epoch: Optional[int] = None
    best_hr20 = -1.0

    for epoch_str, hr20_str in _EVAL_LINE_RE.findall(content):
        try:
            epoch = int(epoch_str)
            hr20 = float(hr20_str)
        except ValueError:
            continue
        if hr20 > best_hr20:
            best_hr20 = hr20
            best_epoch = epoch

    if best_epoch is None:
        return None
    return best_epoch, best_hr20


def parse_logs_for_best_epochs(
    logs_dir: str = DEFAULT_LOGS_DIR,
    models_dir: str = DEFAULT_MODELS_DIR,
) -> Dict[Tuple[str, int], Tuple[int, float]]:
    """掃描日誌目錄並彙整每個 (model_dir_basename, run_id) 的最佳 (epoch, hr20)."""
    best_runs: Dict[Tuple[str, int], Tuple[int, float]] = {}
    log_files = glob.glob(os.path.join(logs_dir, "**/*.txt"), recursive=True)

    for log_path in log_files:
        if "_reformatted" in log_path:
            continue

        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        best = _extract_best_epoch_from_content(content)
        if best is None:
            continue

        key = _infer_run_identity(log_path, content, logs_dir, models_dir)
        if key not in best_runs or best[1] > best_runs[key][1]:
            best_runs[key] = best

    return best_runs
