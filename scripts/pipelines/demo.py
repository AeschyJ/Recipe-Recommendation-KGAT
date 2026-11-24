"""口試展示伺服器與離線展示介面啟動管線."""

from __future__ import annotations

import argparse
import shutil
import sys
import webbrowser
from pathlib import Path
from typing import Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.pipelines import PROJECT_ROOT, resolve_port, run_python_step


def ensure_demo_cache(*, dry_run: bool = False) -> int:
    """確保離線快取檔 demo_cases.json 與 demo_cases_data.js 已就緒."""
    web_dir = PROJECT_ROOT / "demo" / "web"
    cases_json = web_dir / "demo_cases.json"
    cases_js = web_dir / "demo_cases_data.js"
    if cases_json.exists() and cases_js.exists():
        return 0

    print("[提示] 偵測到展示快取檔缺失，正在生成預計算快取...")
    return run_python_step(
        "demo/export_demo_cases.py",
        label="生成口試展示快取資料",
        dry_run=dry_run,
    )


def run_demo_server(
    *,
    port: Optional[int] = None,
    host: str = "0.0.0.0",
    skip_cache_check: bool = False,
    dry_run: bool = False,
) -> int:
    """檢查展示快取與連線環境後，啟動 FastAPI 推論伺服器."""
    resolved_port = resolve_port(port)
    if not skip_cache_check:
        cache_code = ensure_demo_cache(dry_run=dry_run)
        if cache_code != 0:
            return cache_code

    if shutil.which("cloudflared"):
        print(
            f"[提示] 若需建立外網通道，可另行執行: cloudflared tunnel --url http://localhost:{resolved_port}"
        )
    else:
        print(f"[提示] 本機或區網訪問位址: http://localhost:{resolved_port}")

    return run_python_step(
        "demo/server.py",
        ["--port", str(resolved_port), "--host", host],
        label=f"啟動展示推論伺服器 (Port {resolved_port})",
        dry_run=dry_run,
    )


def run_demo_offline(*, open_browser: bool = True, dry_run: bool = False) -> int:
    """檢查離線快取後開啟獨立展示網頁."""
    index_html = PROJECT_ROOT / "demo" / "web" / "index.html"
    if not index_html.exists():
        print(f"[錯誤] 找不到離線展示頁面: {index_html}", file=sys.stderr)
        return 1

    cache_code = ensure_demo_cache(dry_run=dry_run)
    if cache_code != 0:
        return cache_code

    print(f"[資訊] 離線展示介面路徑: {index_html}")
    if open_browser and not dry_run:
        webbrowser.open(index_html.as_uri())
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="口試展示系統啟動輔助")
    subparsers = parser.add_subparsers(dest="action", required=True)

    srv_parser = subparsers.add_parser("server", help="啟動 FastAPI 推論伺服器")
    srv_parser.add_argument("--port", type=int, default=None, help="監聽連接埠")
    srv_parser.add_argument("--host", type=str, default="0.0.0.0", help="監聽主機位址")
    srv_parser.add_argument("--skip-cache-check", action="store_true", help="略過快取存在檢查")
    srv_parser.add_argument("--dry-run", action="store_true", help="僅印出指令而不啟動伺服器")

    off_parser = subparsers.add_parser("offline", help="開啟離線展示網頁")
    off_parser.add_argument("--no-open", action="store_true", help="僅檢查檔案與快取，不開啟瀏覽器")
    off_parser.add_argument("--dry-run", action="store_true", help="僅印出指令而不執行生成或開啟")

    args = parser.parse_args(argv)
    if args.action == "server":
        return run_demo_server(
            port=args.port,
            host=args.host,
            skip_cache_check=args.skip_cache_check,
            dry_run=args.dry_run,
        )
    return run_demo_offline(open_browser=not args.no_open, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
