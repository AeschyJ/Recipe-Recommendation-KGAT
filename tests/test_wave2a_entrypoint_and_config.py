"""Wave 2A 單一薄入口、批次衛生、依賴與伺服器配置驗證測試."""

from __future__ import annotations

import os
import re
import subprocess
import tomllib
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

import main as cli_router
from demo.server import DEFAULT_PORT, build_cors_config, get_default_port

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ACTIVE_BAT_FILES = [
    PROJECT_ROOT / "run.bat",
    PROJECT_ROOT / "demo" / "run_server.bat",
    PROJECT_ROOT / "demo" / "run_offline_laptop.bat",
]

REMOVED_LEGACY_FILES = [
    PROJECT_ROOT / "requirements.txt",
    PROJECT_ROOT / "run_baseline_experiments.bat",
    PROJECT_ROOT / "run_experiments.bat",
    PROJECT_ROOT / "run_log_pipeline.bat",
    PROJECT_ROOT / "run_xai_pipeline.bat",
    PROJECT_ROOT / "run_xai_pipeline_all.bat",
    PROJECT_ROOT / "run_offline_demo_laptop.bat",
    PROJECT_ROOT / "run_oral_defense_demo.bat",
]


@pytest.mark.parametrize("bat_path", ACTIVE_BAT_FILES, ids=lambda p: p.name)
def test_active_batch_files_hygiene(bat_path: Path):
    """確認所有保留之 .bat 皆為純 ASCII、CRLF 換行、小於 60 行且第二行錨定工作目錄."""
    assert bat_path.exists(), f"缺少批次檔: {bat_path}"
    raw = bat_path.read_bytes()

    assert all(byte < 128 for byte in raw), f"{bat_path.name} 包含非 ASCII 字元"
    assert b"\n" not in raw.replace(b"\r\n", b""), f"{bat_path.name} 包含非 CRLF 之裸 LF 換行"

    lines = raw.decode("ascii").splitlines()
    assert len(lines) <= 60, f"{bat_path.name} 超過 60 行限制 ({len(lines)})"
    assert lines[0].strip().lower() == "@echo off"
    assert lines[1].strip().lower() == 'cd /d "%~dp0"'


def test_legacy_batch_files_and_requirements_removed():
    """確認 7 支舊根目錄 .bat 與過時 requirements.txt 皆已移除."""
    for path in REMOVED_LEGACY_FILES:
        assert not path.exists(), f"舊檔案仍殘留於磁碟: {path}"


def test_main_router_line_budget():
    """確認純路由器 main.py 符合 <= 150 行規範."""
    line_count = len((PROJECT_ROOT / "main.py").read_text(encoding="utf-8").splitlines())
    assert line_count <= 150, f"main.py 行數過多: {line_count}"


@pytest.mark.parametrize(
    ("argv", "expected_snippets"),
    [
        (["demo", "--dry-run"], ["demo", "server.py", "--port", "8501"]),
        (["demo-offline", "--no-open", "--dry-run"], ["index.html"]),
        (["baseline", "--dry-run"], ["train_baseline.py", "--model", "LightGCN"]),
        (["exp", "--dry-run"], ["train.py", "--without_kg", "models/depth_2"]),
        (["xai", "--dry-run"], ["sample_users_for_xai.py", "evaluate_fidelity.py"]),
        (
            ["xai", "--all", "--dry-run"],
            ["sampled_users_500.json", "models/depth_3", "models/depth_2", "models/full_kgat"],
        ),
        (["logs", "--dry-run"], ["simplify_output_data.py", "validate.py", "analyze_logs.py"]),
    ],
)
def test_main_router_subcommands_dry_run(
    argv: list[str],
    expected_snippets: list[str],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    """驗證 main.py 全數子命令可正常分派且組裝預期目標腳本."""
    monkeypatch.delenv("PORT", raising=False)
    exit_code = cli_router.main(argv)
    captured = capsys.readouterr()

    assert exit_code == 0
    for snippet in expected_snippets:
        assert snippet in captured.out


def test_run_bat_cmd_execution_and_port_override():
    """實測透過 cmd /c run.bat 透傳參數與環境變數 PORT 覆蓋及 8000 防護."""
    base_env = os.environ.copy()
    base_env["CI"] = "1"
    base_env.pop("PORT", None)

    help_proc = subprocess.run(
        ["cmd", "/c", "run.bat", "--help"],
        cwd=PROJECT_ROOT,
        env=base_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert help_proc.returncode == 0, help_proc.stderr
    assert "demo-offline" in help_proc.stdout

    env_8502 = dict(base_env, PORT="8502")
    demo_proc = subprocess.run(
        ["cmd", "/c", "run.bat", "demo", "--dry-run"],
        cwd=PROJECT_ROOT,
        env=env_8502,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert demo_proc.returncode == 0, demo_proc.stderr
    assert "--port 8502" in demo_proc.stdout

    env_8000 = dict(base_env, PORT="8000")
    blocked_proc = subprocess.run(
        ["cmd", "/c", "run.bat", "demo", "--dry-run"],
        cwd=PROJECT_ROOT,
        env=env_8000,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert blocked_proc.returncode == 0, blocked_proc.stderr
    assert "--port 8501" in blocked_proc.stdout

    env_8505 = dict(base_env, PORT="8505")
    srv_forward_proc = subprocess.run(
        ["cmd", "/c", "demo\\run_server.bat", "--dry-run"],
        cwd=PROJECT_ROOT,
        env=env_8505,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert srv_forward_proc.returncode == 0, srv_forward_proc.stderr
    assert "--port 8505" in srv_forward_proc.stdout

    off_forward_proc = subprocess.run(
        ["cmd", "/c", "demo\\run_offline_laptop.bat", "--no-open", "--dry-run"],
        cwd=PROJECT_ROOT,
        env=base_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert off_forward_proc.returncode == 0, off_forward_proc.stderr
    assert "index.html" in off_forward_proc.stdout


def test_demo_server_port_and_cors_policy(monkeypatch: pytest.MonkeyPatch):
    """驗證 demo/server.py 之 PORT 解析規則與 CORS 白名單設定."""
    monkeypatch.delenv("PORT", raising=False)
    assert get_default_port() == DEFAULT_PORT

    monkeypatch.setenv("PORT", "8510")
    assert get_default_port() == 8510

    monkeypatch.setenv("PORT", "8000")
    assert get_default_port() == DEFAULT_PORT

    monkeypatch.setenv("PORT", "invalid")
    assert get_default_port() == DEFAULT_PORT

    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    cors_cfg = build_cors_config()
    assert cors_cfg["allow_credentials"] is False
    assert "*" not in cors_cfg["allow_origins"]
    assert cors_cfg["allow_origin_regex"]

    middleware = CORSMiddleware(app=FastAPI(), **cors_cfg)
    for allowed_origin in [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "https://aeschyj.github.io",
        "https://recipe-kgat-demo.vercel.app",
        "https://oral-demo-tunnel.trycloudflare.com",
    ]:
        assert middleware.is_allowed_origin(allowed_origin), f"應放行來源: {allowed_origin}"

    for denied_origin in [
        "https://evil.example.com",
        "http://localhost.evil.com",
        "null",
    ]:
        assert not middleware.is_allowed_origin(denied_origin), f"不應放行來源: {denied_origin}"


def test_pyproject_optional_dependencies_and_gitignore():
    """驗證 pyproject.toml 選填依賴與 .gitignore 脫敏規則完整性."""
    pyproject_data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    opt_deps = pyproject_data["project"]["optional-dependencies"]
    for pkg in ("fastapi", "uvicorn", "pydantic", "scipy"):
        assert any(pkg in item for item in opt_deps["demo"])
    assert any("pytest" in item for item in opt_deps["dev"])

    gitignore_lines = [
        line.strip()
        for line in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    ]
    for required_rule in ("Paper/Needs/", "local_config.py", "gemini4_*.md", ".ruff_cache/"):
        assert required_rule in gitignore_lines
