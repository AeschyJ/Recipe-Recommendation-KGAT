"""Wave 2B 核心拆解、日誌解析共模組與例外處理單元測試."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest
import torch

from demo.server import (
    _build_no_path_probe_response,
    _lookup_cached_probe_response,
    _normalize_path_nodes,
)
from scripts import (
    analyze_checkpoints_cleanup,
    cleanup_checkpoints,
    create_backup_zips,
    evaluate_all,
    log_utils,
    visualize_paths,
)
from src import train as train_module


def test_log_utils_extract_info_and_parse_logs(tmp_path: Path):
    """驗證 extract_info 與 parse_logs_for_best_epochs 三層身分推斷及指標挑選."""
    assert log_utils.extract_info("1_kgat_checkpoint_e3.pth") == (1, 3)
    assert log_utils.extract_info("invalid_checkpoint.pth") is None

    logs_dir = tmp_path / "logs"
    (logs_dir / "depth_2").mkdir(parents=True)
    (logs_dir / "full_kgat").mkdir(parents=True)
    (logs_dir / "wo_kg").mkdir(parents=True)

    # Case 1: 完整 Args (model_dir + experiment_id)
    (logs_dir / "depth_2" / "run3.txt").write_text(
        "Args: Namespace(model_dir='models/depth_2', experiment_id='3')\n"
        "Epoch 1 Evaluation - HR@[10,20,50]: [0.6000, 0.7100, 0.8500]\n"
        "Epoch 2 Evaluation - HR@[10,20,50]: [0.6500, 0.8250, 0.9000]\n",
        encoding="utf-8",
    )
    # Case 2: 由 Saved checkpoint 推斷 model_dir 與 run_id
    (logs_dir / "full_kgat" / "run2.txt").write_text(
        "Saved checkpoint: models/full_kgat\\2_kgat_checkpoint_e5.pth\n"
        "Epoch 5 Evaluation - Recall@[10,20,50]: [0.5500, 0.6771, 0.8000]\n",
        encoding="utf-8",
    )
    # Case 3: 僅由日誌子目錄推斷 model_dir，run_id 預設 1
    (logs_dir / "wo_kg" / "fallback.txt").write_text(
        "Epoch 4 Evaluation - HR@[10,20,50]: [0.5000, 0.7000, 0.8200]\n",
        encoding="utf-8",
    )
    # Case 4: _reformatted 應被忽略
    (logs_dir / "depth_2" / "run3_reformatted.txt").write_text(
        "Args: Namespace(model_dir='models/depth_2', experiment_id='3')\n"
        "Epoch 99 Evaluation - HR@[10,20,50]: [0.9999, 0.9999, 0.9999]\n",
        encoding="utf-8",
    )

    result = log_utils.parse_logs_for_best_epochs(
        logs_dir=str(logs_dir),
        models_dir=str(tmp_path / "models"),
    )
    assert result[("depth_2", 3)] == pytest.approx((2, 0.8250))
    assert result[("full_kgat", 2)] == pytest.approx((5, 0.6771))
    assert result[("wo_kg", 1)] == pytest.approx((4, 0.7000))


def test_checkpoint_scripts_reuse_shared_log_utils():
    """確認 3 支檢查點腳本皆已改用 scripts/log_utils.py 共用函式."""
    for mod in (analyze_checkpoints_cleanup, cleanup_checkpoints, create_backup_zips):
        assert mod.parse_logs_for_best_epochs is log_utils.parse_logs_for_best_epochs
        assert mod.extract_info is log_utils.extract_info


def test_evaluate_all_build_ckpt_to_log_map_handles_corrupt_log(tmp_path: Path, caplog):
    """確認 build_ckpt_to_log_map 可正常建立映射並在遇壞檔時記錄警告."""
    valid_log = tmp_path / "exp" / "ok.txt"
    valid_log.parent.mkdir(parents=True)
    valid_log.write_text(
        "Saved checkpoint: models\\depth_3\\1_kgat_checkpoint_e3.pth\n",
        encoding="utf-8",
    )
    reformatted_log = tmp_path / "exp" / "ok_reformatted.txt"
    reformatted_log.write_text(
        "Saved checkpoint: models\\depth_3\\9_kgat_checkpoint_e9.pth\n",
        encoding="utf-8",
    )
    corrupt_log = tmp_path / "exp" / "corrupt.txt"
    corrupt_log.write_bytes(b"\xff\xfe\xfd\x80")

    with caplog.at_level(logging.WARNING, logger=evaluate_all.logger.name):
        mapping = evaluate_all.build_ckpt_to_log_map(str(tmp_path / "**" / "*.txt"))

    assert mapping["models/depth_3/1_kgat_checkpoint_e3.pth"] == str(valid_log).replace("\\", "/")
    assert "models/depth_3/9_kgat_checkpoint_e9.pth" not in mapping
    assert any("無法讀取日誌檔" in rec.message for rec in caplog.records)


def test_train_select_device_and_save_best_checkpoint(tmp_path: Path):
    """驗證 src/train.py 裝置選取與最佳檢查點儲存／舊檔清理／早停判定."""
    assert train_module.select_device(force_cpu=True) == torch.device("cpu")

    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")
    args = SimpleNamespace(model_dir=str(tmp_path), experiment_id="9")

    # Epoch 1: 創新高 0.70
    best_hr, best_ep, patience_cnt, stop = train_module.save_best_checkpoint(
        model, optimizer, scheduler, None, args, 0, 0.5, 0.70, 0.0, 0, 0, patience=2
    )
    assert (best_hr, best_ep, patience_cnt, stop) == (0.70, 1, 0, False)
    ckpt_e1 = tmp_path / "9_kgat_checkpoint_e1.pth"
    assert ckpt_e1.exists()

    # Epoch 2: 再創新高 0.80 -> 應清除 e1
    best_hr, best_ep, patience_cnt, stop = train_module.save_best_checkpoint(
        model, optimizer, scheduler, None, args, 1, 0.4, 0.80, best_hr, best_ep, patience_cnt, patience=2
    )
    assert (best_hr, best_ep, patience_cnt, stop) == (0.80, 2, 0, False)
    assert not ckpt_e1.exists()
    assert (tmp_path / "9_kgat_checkpoint_e2.pth").exists()

    # Epoch 3 & 4: 未改善 -> 達 patience=2 觸發早停
    best_hr, best_ep, patience_cnt, stop = train_module.save_best_checkpoint(
        model, optimizer, scheduler, None, args, 2, 0.45, 0.75, best_hr, best_ep, patience_cnt, patience=2
    )
    assert (patience_cnt, stop) == (1, False)
    best_hr, best_ep, patience_cnt, stop = train_module.save_best_checkpoint(
        model, optimizer, scheduler, None, args, 3, 0.46, 0.72, best_hr, best_ep, patience_cnt, patience=2
    )
    assert (patience_cnt, stop) == (2, True)


def test_visualize_paths_helpers_and_render(tmp_path: Path):
    """驗證路徑視覺化佈局計算、交叉邊標籤錯開與 PDF 繪圖輸出."""
    # Case A: 2 節點
    g_a = nx.DiGraph()
    g_a.add_edge(0, 10, weight=0.5)
    pos_a, xlim_a, _ = visualize_paths.compute_deterministic_layout(
        g_a, 0, 10, {0: "User 0", 10: "soup"}
    )
    assert pos_a[0] == (-0.55, 0.40)
    assert pos_a[10] == (0.45, -0.35)
    assert xlim_a == (-1.1, 1.25)

    # Case B: bruschetta 案例佈局
    g_b = nx.DiGraph()
    g_b.add_edge(1, 20, weight=0.767)
    g_b.add_edge(20, 30, weight=0.4)
    labels_b = {1: "User 573230", 20: "cajun pasta", 30: "bruschetta"}
    pos_b, _, _ = visualize_paths.compute_deterministic_layout(g_b, 1, 30, labels_b)
    assert pos_b[1] == (-0.65, 0.65)
    assert pos_b[30] == (-0.75, -0.55)

    # Case C: thai salad 案例與 spring fallback
    g_c = nx.DiGraph()
    g_c.add_edge(2, 40, weight=0.6)
    g_c.add_edge(40, 50, weight=0.5)
    pos_c, _, _ = visualize_paths.compute_deterministic_layout(
        g_c, 2, 50, {2: "User 963993", 40: "chicken", 50: "thai salad"}
    )
    assert pos_c[40] == (-0.05, -0.58)
    pos_gen, xlim_gen, _ = visualize_paths.compute_deterministic_layout(
        g_c, 2, 50, {2: "User 2", 40: "tomato", 50: "curry"}
    )
    assert set(pos_gen.keys()) == {2, 40, 50}
    assert xlim_gen == (-1.2, 1.2)

    assert visualize_paths.compute_edge_label_pos("User 1", "cajun pasta") == 0.38
    assert visualize_paths.compute_edge_label_pos("chicken", "none better") == 0.68
    assert visualize_paths.compute_edge_label_pos("User 868551", "bruschetta") == 0.35
    assert visualize_paths.compute_edge_label_pos("User 1", "tomato") == 0.5

    out_pdf = tmp_path / "user_1.pdf"
    visualize_paths.render_explanation_figure(
        g_b,
        pos_b,
        (-1.25, 1.35),
        (-0.88, 0.88),
        labels_b,
        {1: "USER", 20: "RECIPE", 30: "RECIPE"},
        30,
        "User 573230",
        "bruschetta",
        str(out_pdf),
    )
    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 0


def test_demo_server_probe_helpers():
    """驗證 demo/server.py 路徑正規化、無路徑回應建構與快取探針查閱."""
    assert _normalize_path_nodes(None) is None
    assert _normalize_path_nodes([1, 2, 3]) == [[1, 2, 3]]
    assert _normalize_path_nodes([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]

    no_path = _build_no_path_probe_response(5, 10, "3", "f_plus", 0.85)
    assert no_path["is_no_path"] is True
    assert no_path["fidelity_score"] == 0.0
    assert no_path["prob"] == 0.85
    assert no_path["status_description"] == "無路徑使用者"

    cached = {
        "prob": 0.9,
        "fidelity": {
            "f_plus": 0.25,
            "prob_f_plus": 0.65,
            "f_minus": 0.02,
            "prob_f_minus": 0.88,
        },
    }
    res_norm = _lookup_cached_probe_response(5, 10, "3", "normal", cached)
    assert res_norm is not None and res_norm["prob"] == 0.9 and res_norm["fidelity_score"] == 0.0

    res_fp = _lookup_cached_probe_response(5, 10, "3", "f_plus", cached)
    assert res_fp is not None and res_fp["prob"] == 0.65 and res_fp["fidelity_score"] == 0.25

    res_fm = _lookup_cached_probe_response(5, 10, "3", "f_minus", cached)
    assert res_fm is not None and res_fm["prob"] == 0.88 and res_fm["fidelity_score"] == 0.02

    assert _lookup_cached_probe_response(5, 10, "3", "f_plus", {"prob": 0.9}) is None
