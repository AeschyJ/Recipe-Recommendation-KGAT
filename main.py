"""KGAT 實驗與展示系統統一命令路由器.

承接薄點火器 run.bat 轉發參數，支援子命令直通與無參數互動選單。
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Dict, Optional, Sequence

from scripts.pipelines.demo import run_demo_offline, run_demo_server
from scripts.pipelines.experiments import run_experiments
from scripts.pipelines.logs import run_log_pipeline
from scripts.pipelines.xai import run_xai


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.bat",
        description="KGAT 食譜推薦與忠實度評估系統統一入口",
    )
    sub = parser.add_subparsers(dest="command")

    p_demo = sub.add_parser("demo", help="啟動 FastAPI 推論與圖遮蔽展示伺服器")
    p_demo.add_argument("--port", type=int, default=None, help="指定監聽連接埠（支援 PORT 環境變數）")
    p_demo.add_argument("--host", type=str, default="0.0.0.0", help="指定監聽位址")
    p_demo.add_argument("--skip-cache-check", action="store_true", help="略過 demo_cases 快取檢查")
    p_demo.add_argument("--dry-run", action="store_true", help="僅印出實際指令")

    p_off = sub.add_parser("demo-offline", help="檢查快取並開啟完全離線展示網頁")
    p_off.add_argument("--no-open", action="store_true", help="僅驗證檔案不呼叫瀏覽器")
    p_off.add_argument("--dry-run", action="store_true", help="僅印出實際指令")

    p_base = sub.add_parser("baseline", help="執行經典 Baseline 模型實驗")
    p_base.add_argument("--all", action="store_true", help="執行含 BPR-MF / NFM 在內之完整清單")
    p_base.add_argument("--dry-run", action="store_true", help="僅印出實際指令")

    p_exp = sub.add_parser("exp", help="執行 KGAT 消融實驗管線")
    p_exp.add_argument("--all", action="store_true", help="執行 5 組完整消融配置")
    p_exp.add_argument("--dry-run", action="store_true", help="僅印出實際指令")

    p_xai = sub.add_parser("xai", help="執行 XAI 解釋萃取與 Fidelity 評估")
    p_xai.add_argument("--all", action="store_true", help="對 L=1/L=2/L=3 執行 500 位使用者評估")
    p_xai.add_argument("--dry-run", action="store_true", help="僅印出實際指令")

    p_logs = sub.add_parser("logs", help="執行日誌簡化、資料驗證與指標分析")
    p_logs.add_argument("--dry-run", action="store_true", help="僅印出實際指令")
    return parser


def dispatch(args: argparse.Namespace) -> int:
    cmd = args.command
    if cmd == "demo":
        return run_demo_server(
            port=args.port,
            host=args.host,
            skip_cache_check=args.skip_cache_check,
            dry_run=args.dry_run,
        )
    if cmd == "demo-offline":
        return run_demo_offline(open_browser=not args.no_open, dry_run=args.dry_run)
    if cmd == "baseline":
        return run_experiments("baseline", include_all=args.all, dry_run=args.dry_run)
    if cmd == "exp":
        return run_experiments("main", include_all=args.all, dry_run=args.dry_run)
    if cmd == "xai":
        return run_xai(run_all=args.all, dry_run=args.dry_run)
    if cmd == "logs":
        return run_log_pipeline(dry_run=args.dry_run)
    return 1


def run_interactive_menu() -> int:
    actions: Dict[str, tuple[str, Callable[[], int]]] = {
        "1": ("啟動展示伺服器 (demo)", lambda: run_demo_server()),
        "2": ("開啟離線展示網頁 (demo-offline)", lambda: run_demo_offline()),
        "3": ("執行 Baseline 實驗 (baseline)", lambda: run_experiments("baseline")),
        "4": ("執行 KGAT 消融實驗 (exp)", lambda: run_experiments("main")),
        "5": ("執行 XAI Fidelity 評估 - 單模型 (xai)", lambda: run_xai(run_all=False)),
        "6": ("執行 XAI Fidelity 評估 - 全模型 500 人 (xai --all)", lambda: run_xai(run_all=True)),
        "7": ("執行日誌整理與分析管線 (logs)", lambda: run_log_pipeline()),
    }
    print("\n=== KGAT 食譜推薦與可解釋性系統 控制台 ===")
    for key, (label, _) in actions.items():
        print(f"  [{key}] {label}")
    print("  [0] 離開")

    try:
        choice = input("\n請輸入執行項目編號 [0-7]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return 0

    if choice in ("0", ""):
        return 0
    if choice not in actions:
        print(f"[錯誤] 無效選項: {choice}")
        return 1
    return actions[choice][1]()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        if sys.stdin.isatty():
            return run_interactive_menu()
        parser.print_help()
        return 0

    args = parser.parse_args(raw_argv)
    return dispatch(args)


if __name__ == "__main__":
    raise SystemExit(main())
