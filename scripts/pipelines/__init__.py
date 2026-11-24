"""薄入口管線共用工具模組.

集中解析專案根目錄、虛擬環境直譯器、%PORT% 覆蓋與子步驟執行控制。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORT = 8501
FORBIDDEN_PORT = 8000

StepSpec = Tuple[str, str, list[str]]


def resolve_python_exe() -> str:
    """優先傳回專案 .venv Python 直譯器，不存在時退回當前執行環境."""
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def resolve_port(port: Optional[int] = None) -> int:
    """解析連接埠（參數 > 環境變數 PORT > 預設 8501），並嚴格攔截 8000."""
    if port is None:
        raw = os.environ.get("PORT", str(DEFAULT_PORT)).strip()
        try:
            candidate = int(raw)
        except ValueError:
            print(f"[警告] 無效的 PORT 環境變數 '{raw}'，自動改用預設連接埠 {DEFAULT_PORT}。")
            return DEFAULT_PORT
    else:
        candidate = port

    if candidate == FORBIDDEN_PORT:
        print(f"[警告] 嚴禁使用 {FORBIDDEN_PORT} 端口，自動重設為 {DEFAULT_PORT}。")
        return DEFAULT_PORT
    if candidate <= 0 or candidate > 65535:
        print(f"[警告] 連接埠 {candidate} 超出有效範圍 (1-65535)，自動重設為 {DEFAULT_PORT}。")
        return DEFAULT_PORT
    return candidate


def run_python_step(
    script_rel: str,
    args: Optional[Sequence[str]] = None,
    *,
    label: str = "",
    dry_run: bool = False,
) -> int:
    """在專案根目錄下執行單一 Python 腳本步驟並強制 UTF-8 編碼."""
    script_path = PROJECT_ROOT / script_rel
    cmd = [resolve_python_exe(), str(script_path), *(args or [])]

    if label:
        print(f"\n=== {label} ===")

    if dry_run:
        print(f"[DRY-RUN] {subprocess.list2cmdline(cmd)}")
        return 0

    if not script_path.exists():
        print(f"[錯誤] 找不到目標腳本: {script_path}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    return result.returncode


def run_steps(steps: Iterable[StepSpec], *, dry_run: bool = False) -> int:
    """循序執行多個管線步驟，任一步驟失敗即停止並回傳退出碼."""
    for label, script_rel, args in steps:
        code = run_python_step(script_rel, args, label=label, dry_run=dry_run)
        if code != 0:
            print(f"[錯誤] 步驟失敗 ({label})，退出碼: {code}", file=sys.stderr)
            return code
    return 0
