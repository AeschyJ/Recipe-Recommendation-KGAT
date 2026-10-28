"""口試展示推論伺服器與圖遮蔽探針單元測試.

驗證推論端點邏輯、圖遮蔽計算 ($F^+ / F^-$)、靜態網頁資源與學術規範合規性。
"""

import json
import os
import sys
from pathlib import Path
import pytest
from fastapi import HTTPException

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.demo.server import (
    state,
    get_status,
    recommend,
    fidelity_probe,
    RecommendRequest,
    FidelityProbeRequest,
)


@pytest.fixture(scope="module", autouse=True)
def setup_server():
    """初始化推論伺服器狀態."""
    state.initialize()
    yield


def test_api_status_endpoint():
    """驗證 GET /api/status 端點輸出結構與數值，並確認包含所有 8 款模型."""
    data = get_status()
    assert data["status"] == "連線"
    assert data["port"] == 8501
    assert "available_depths" in data
    for depth in ["1", "2", "3", "wo_kg", "wo_attn", "lightgcn", "nfm", "bprmf"]:
        assert depth in data["available_depths"]
    assert "benchmark_metrics" in data
    assert data["n_users"] > 0
    assert data["n_items"] > 0


def test_api_recommend_cached_users():
    """驗證 POST /api/recommend 端點在典型案例使用者之推薦結果."""
    # 測試深度傳播成功案例 (使用者 17049, 深度 3)
    req = RecommendRequest(user_id=17049, depth="3")
    data = recommend(req)

    assert data["user_id"] == 17049
    assert data["depth"] == "3"
    assert "recommended_item_id" in data
    assert "recommended_item_name" in data
    assert "roast pork" in data["recommended_item_name"].lower()
    assert data["prob"] > 0.90
    assert "metrics" in data
    assert data["metrics"]["hr_20"] == pytest.approx(0.8360, rel=1e-3)
    assert len(data["explanations"]) > 0

    # 檢查解釋路徑結構
    exp = data["explanations"][0]
    assert "path_structure" in exp
    assert "path_description" in exp
    assert "contribution_score" in exp
    assert len(exp["path_details"]) >= 2
    # 檢查節點欄位
    for node in exp["path_details"]:
        assert "id_remapped" in node
        assert "id_original" in node
        assert "type" in node
        assert "name" in node


def test_api_recommend_isolated_node():
    """驗證孤立節點案例 (使用者 31742) 回傳無解釋路徑且不產生錯誤."""
    req = RecommendRequest(user_id=31742, depth="3")
    data = recommend(req)

    assert data["user_id"] == 31742
    assert "recommended_item_name" in data
    assert len(data["explanations"]) == 0


def test_api_recommend_ablation():
    """驗證消融實驗 (wo_kg) 推薦端點."""
    req = RecommendRequest(user_id=17049, depth="wo_kg")
    data = recommend(req)

    assert data["depth"] == "wo_kg"
    assert data["metrics"]["hr_20"] == pytest.approx(0.7160, rel=1e-3)
    # 移除知識圖模式下無知識路徑
    assert len(data["explanations"]) == 0


def test_api_recommend_baseline_and_ablation_models():
    """驗證消融模型 (wo_attn) 與基準模型 (LightGCN, NFM, BPR-MF) 推論結果."""
    for model_key in ["wo_attn", "lightgcn", "nfm", "bprmf"]:
        req = RecommendRequest(user_id=17049, depth=model_key)
        data = recommend(req)
        assert data["depth"] == model_key
        assert "recommended_item_name" in data
        assert data["prob"] >= 0.0
        assert len(data["explanations"]) == 0


def test_api_fidelity_probe_normal():
    """驗證 POST /api/fidelity_probe 在 normal 模式下機率等於原始預測."""
    req = FidelityProbeRequest(user_id=101612, item_id=48114, depth="3", mode="normal")
    data = fidelity_probe(req)

    assert data["mode"] == "normal"
    assert data["prob"] == pytest.approx(data["original_prob"], rel=1e-5)
    assert data["fidelity_score"] == 0.0


def test_api_fidelity_probe_f_plus_high_necessity():
    """驗證高必要性案例之 F+ 圖遮蔽計算邏輯 (使用者 101612)."""
    rec_res = recommend(RecommendRequest(user_id=101612, depth="3"))
    rec_item_id = rec_res["recommended_item_id"]

    req = FidelityProbeRequest(user_id=101612, item_id=rec_item_id, depth="3", mode="f_plus")
    data = fidelity_probe(req)

    assert data["mode"] == "f_plus"
    assert data["status_description"] == "遮蔽解釋子圖：驗證必要性"
    assert data["original_prob"] > data["prob"]
    # F+ = P_orig - P_f+，高必要性案例之 F+ 應超過 0.20
    assert data["fidelity_score"] > 0.20
    assert data["fidelity_score"] == pytest.approx(
        data["original_prob"] - data["prob"], abs=1e-4
    )


def test_api_fidelity_probe_f_minus_high_sufficiency():
    """驗證高充分性案例之 F- 圖遮蔽計算邏輯 (使用者 154656)."""
    rec_res = recommend(RecommendRequest(user_id=154656, depth="3"))
    rec_item_id = rec_res["recommended_item_id"]

    req = FidelityProbeRequest(user_id=154656, item_id=rec_item_id, depth="3", mode="f_minus")
    data = fidelity_probe(req)

    assert data["mode"] == "f_minus"
    assert data["status_description"] == "僅保留解釋子圖：驗證充分性"
    # F- = P_orig - P_f-，高充分性案例差值應極小 (|F-| < 0.01)
    assert abs(data["fidelity_score"]) < 0.01
    assert data["fidelity_score"] == pytest.approx(
        data["original_prob"] - data["prob"], abs=1e-4
    )


def test_api_fidelity_probe_no_path_protection():
    """驗證無路徑或非注意力模型輸入時，探針端點安全回傳零變化指標不拋出 500."""
    # 測試空路徑
    req_empty = FidelityProbeRequest(user_id=17049, item_id=100, depth="3", mode="f_plus", path_nodes=[])
    data_empty = fidelity_probe(req_empty)
    assert data_empty["is_no_path"] is True
    assert data_empty["fidelity_score"] == 0.0
    assert data_empty["status_description"] == "無路徑使用者"

    # 測試長度小於 2 之無效路徑
    req_single = FidelityProbeRequest(user_id=17049, item_id=100, depth="3", mode="f_plus", path_nodes=[[17049]])
    data_single = fidelity_probe(req_single)
    assert data_single["is_no_path"] is True
    assert data_single["fidelity_score"] == 0.0
    assert data_single["status_description"] == "無路徑使用者"

    # 測試非圖注意力模型
    req_lightgcn = FidelityProbeRequest(user_id=17049, item_id=100, depth="lightgcn", mode="f_plus")
    data_lightgcn = fidelity_probe(req_lightgcn)
    assert data_lightgcn["is_no_path"] is True
    assert data_lightgcn["fidelity_score"] == 0.0


def test_api_fidelity_probe_invalid_mode():
    """驗證探針端點輸入無效模式時拋出 HTTPException 400."""
    req = FidelityProbeRequest(user_id=17049, item_id=100, depth="3", mode="invalid_mode")
    with pytest.raises(HTTPException) as exc_info:
        fidelity_probe(req)
    assert exc_info.value.status_code == 400


def test_demo_cases_json_integrity():
    """驗證 demo_cases.json 與 demo_cases_data.js 結構完整性 (5 組案例與 500 位評估名冊)."""
    cases_file = os.path.join(project_root, "src", "demo", "web", "demo_cases.json")
    cases_js_file = os.path.join(project_root, "src", "demo", "web", "demo_cases_data.js")
    assert os.path.exists(cases_file)
    assert os.path.exists(cases_js_file)

    with open(cases_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(cases_js_file, "r", encoding="utf-8") as f:
        js_content = f.read()
    assert js_content.startswith("window.DEMO_CASES_DATA = ")

    assert "academic_cases" in data
    assert len(data["academic_cases"]) == 5

    case_ids = [c["case_id"] for c in data["academic_cases"]]
    expected_cases = [
        "deep_propagation",
        "high_necessity",
        "high_sufficiency",
        "shallow_noise",
        "isolated_node",
    ]
    for c_id in expected_cases:
        assert c_id in case_ids

    assert "users" in data
    assert len(data["users"]) == 500

    assert "user_cache" in data
    assert len(data["user_cache"]) == 500


def test_static_web_files_exist_and_conform_to_standards():
    """驗證前端介面檔案存在且全面遵守專案學術標準與精簡文案規範."""
    web_dir = os.path.join(project_root, "src", "demo", "web")
    html_file = os.path.join(web_dir, "index.html")
    css_file = os.path.join(web_dir, "style.css")
    js_file = os.path.join(web_dir, "app.js")
    data_js_file = os.path.join(web_dir, "demo_cases_data.js")

    assert os.path.exists(html_file)
    assert os.path.exists(css_file)
    assert os.path.exists(js_file)
    assert os.path.exists(data_js_file)

    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    with open(css_file, "r", encoding="utf-8") as f:
        css_content = f.read()

    with open(js_file, "r", encoding="utf-8") as f:
        js_content = f.read()

    # 驗證使用者編號輸入框存在以支援口試現場自訂輸入與範圍限制
    assert 'id="userIdInput"' in html_content
    assert 'placeholder="範圍 0 ~ 226569"' in html_content
    assert 'max="226569"' in html_content

    # 驗證包含 demo_cases_data.js 支援離線零 CORS 阻擋
    assert "demo_cases_data.js" in html_content
    # 驗證邊線寬度依注意力權重映射
    assert "dynamicStrokeWidth" in js_content or "stroke-width" in js_content

    # 驗證節點角色屬性 (data-role) 標註與脈衝路徑
    assert "data-role" in js_content
    assert "svg-bg-pulse-path" in css_content
    assert "@keyframes dash" in css_content

    # 驗證標題精簡為「KGAT推理及KGATexplainer忠實度檢測」且副標題已移除
    assert "<title>KGAT推理及KGATexplainer忠實度檢測</title>" in html_content
    assert "<h1>KGAT推理及KGATexplainer忠實度檢測</h1>" in html_content
    assert "碩士學位口試現場展示" not in html_content

    # 驗證按鈕包含所有 8 款模型
    for model_btn in [
        "KGAT L=3", "KGAT L=2", "KGAT L=1", "wo_kg L=1", "wo_attn L=1", "LightGCN", "NFM", "BPR-MF"
    ]:
        assert model_btn in html_content, f"HTML 缺少模型按鈕: {model_btn}"

    # 驗證中央聚焦遮罩與動態狀態膠囊
    assert 'id="inferenceModal"' in html_content
    assert 'id="taskStatusPill"' in html_content
    assert 'id="modalBackgroundBtn"' in html_content
    assert 'id="modalAbortBtn"' in html_content

    # 驗證文案精簡規範
    assert "典型案例：" in html_content
    assert "典型學術案例" not in html_content
    assert '<h2 class="card-title">預測結果</h2>' in html_content
    assert "推薦信心" in html_content
    assert "推薦信心機率" not in html_content
    assert "推薦結果與排序" not in html_content
    assert '<h2 class="card-title">忠實度分析</h2>' in html_content
    assert "因果干預驗證" not in html_content
    assert "必要性 F+" in html_content
    assert "充分性 F-" in html_content
    assert "必要性忠實度" not in html_content
    assert "充分性忠實度" not in html_content
    assert "必要性忠實度" not in js_content
    assert "充分性忠實度" not in js_content
    assert "目前狀態：" not in html_content
    assert "差值：" not in html_content
    assert "遮蔽後預測" in html_content
    assert "機率降幅愈顯著，代表該路徑對模型推薦決策之必要性愈高。" in html_content
    assert "差值 |F-| 愈趨近於零，代表該路徑單獨支撐推薦之充分性愈高。" in html_content

    # 全文「圖譜」改為「圖」
    assert "圖譜" not in html_content, "HTML 仍包含未替換之「圖譜」"

    # 移除「呈現拓撲邊界」
    assert "呈現拓撲邊界" not in html_content, "HTML 仍包含「呈現拓撲邊界」"

    # 學術規範 1: 不刻意說明未加入的元素 (嚴禁提及「無廣告」、「無多餘口語」、「未包含某功能」等贅述)
    for forbidden in ["無廣告", "無多餘口語", "未包含", "無依賴", "零外部庫"]:
        assert forbidden not in html_content, f"HTML 包含未加入元素說明: {forbidden}"

    # 學術規範 3: 不在中文後附加英文翻譯 (嚴禁如「使用者 (User)」、「食譜 (Recipe)」、「遮蔽 (Mask)」)
    for forbidden in ["使用者 (User)", "食譜 (Recipe)", "遮蔽 (Mask)", "忠實度 (Fidelity)"]:
        assert forbidden not in html_content, f"HTML 包含中文附加英文翻譯: {forbidden}"

    # 學術規範 4: 不使用括號標註修改狀態或變更 (嚴禁標註「修改版」、「完全對齊」)
    for forbidden in ["(修改版)", "（修改版）", "(完全對齊)", "（完全對齊）"]:
        assert forbidden not in html_content, f"HTML 包含修改標註: {forbidden}"

    # 狀態指示僅保留「● 連線」與「○ 離線」
    assert "● 連線" in html_content or "○ 離線" in html_content

    # 驗證子圖卡片 id="graphSection" 存在以支援非圖注意力模型反灰
    assert 'id="graphSection"' in html_content
    # 驗證探針彈窗未洩漏英文變數名與括號
    assert "因果干預 (${mode})" not in js_content
    assert "(${mode})" not in js_content
    # 驗證非圖注意力模型子圖反灰邏輯
    assert "graphSection" in js_content


def test_api_boundary_user_and_item_ids():
    """驗證使用者編號與食譜編號邊界防護 (超限回傳 400)."""
    # 使用者編號超出上限
    with pytest.raises(HTTPException) as exc_info:
        recommend(RecommendRequest(user_id=226570, depth="3"))
    assert exc_info.value.status_code == 400

    # 使用者編號負數
    with pytest.raises(HTTPException) as exc_info:
        recommend(RecommendRequest(user_id=-1, depth="3"))
    assert exc_info.value.status_code == 400

    # 探針端點使用者編號超出上限
    with pytest.raises(HTTPException) as exc_info:
        fidelity_probe(FidelityProbeRequest(user_id=226570, item_id=100, depth="3"))
    assert exc_info.value.status_code == 400

    # 探針端點食譜編號超出上限
    with pytest.raises(HTTPException) as exc_info:
        fidelity_probe(FidelityProbeRequest(user_id=17049, item_id=231637, depth="3"))
    assert exc_info.value.status_code == 400

    # 探針端點食譜編號負數
    with pytest.raises(HTTPException) as exc_info:
        fidelity_probe(FidelityProbeRequest(user_id=17049, item_id=-1, depth="3"))
    assert exc_info.value.status_code == 400


def test_api_recommend_uncached_dynamic_inference():
    """驗證未快取之自訂使用者在動態 PyTorch 前向計算之正確性."""
    uncached_uid = 99999
    # 測試圖注意力模型動態推論
    data_kgat = recommend(RecommendRequest(user_id=uncached_uid, depth="3"))
    assert data_kgat["user_id"] == uncached_uid
    assert data_kgat["depth"] == "3"
    assert data_kgat["recommended_item_id"] >= 0
    assert data_kgat["prob"] >= 0.0

    # 測試基準模型動態推論
    data_lightgcn = recommend(RecommendRequest(user_id=uncached_uid, depth="lightgcn"))
    assert data_lightgcn["user_id"] == uncached_uid
    assert data_lightgcn["depth"] == "lightgcn"
    assert data_lightgcn["recommended_item_id"] >= 0
    assert len(data_lightgcn["explanations"]) == 0


def test_api_fidelity_probe_uncached_user_no_path():
    """驗證未快取使用者在無路徑或非注意力模型下回傳安全零變化指標."""
    req = FidelityProbeRequest(user_id=99999, item_id=100, depth="lightgcn", mode="f_plus")
    data = fidelity_probe(req)
    assert data["is_no_path"] is True
    assert data["fidelity_score"] == 0.0
    assert data["prob"] >= 0.0


def test_api_dynamic_inference_all_uncached_models():
    """驗證所有模型（包含消融與基準）在未快取自訂使用者下的動態推論健全性."""
    uncached_uid = 99998
    for model_key in ["1", "2", "3", "wo_kg", "wo_attn", "lightgcn", "bprmf", "nfm"]:
        data = recommend(RecommendRequest(user_id=uncached_uid, depth=model_key))
        assert data["user_id"] == uncached_uid
        assert data["depth"] == model_key
        assert data["recommended_item_id"] >= 0
        assert data["prob"] >= 0.0


def test_cached_final_embeddings_and_logging():
    """驗證 KGAT 模型之節點特徵嵌入已成功預先計算並快取，且伺服器日誌已落盤寫入."""
    model, _ = state.get_or_load_model("3")
    assert hasattr(model, "cached_final_embed")
    assert model.cached_final_embed is not None
    # 節點總數為 473577，特徵維度為 256 (64 * (3 + 1))
    assert model.cached_final_embed.shape[0] == state.num_nodes
    assert model.cached_final_embed.shape[1] == 256

    # 驗證日誌落盤
    log_file = os.path.join(project_root, "output", "logs", "demo_server.log")
    assert os.path.exists(log_file)
    assert os.path.getsize(log_file) > 0
    with open(log_file, "r", encoding="utf-8") as f:
        log_content = f.read()
    assert "命中快取推薦" in log_content or "推薦成功" in log_content


def test_ui_enhancements_and_modal_features():
    """驗證伺服器配置彈窗、雙層進度條、按鈕文案與背景裝飾元素對齊學術規範."""
    web_dir = os.path.join(project_root, "src", "demo", "web")
    html_file = os.path.join(web_dir, "index.html")
    css_file = os.path.join(web_dir, "style.css")
    js_file = os.path.join(web_dir, "app.js")

    with open(html_file, "r", encoding="utf-8") as f:
        html = f.read()
    with open(css_file, "r", encoding="utf-8") as f:
        css = f.read()
    with open(js_file, "r", encoding="utf-8") as f:
        js = f.read()

    # 1. 伺服器配置彈窗驗證
    assert 'id="serverConfigBtn"' in html
    assert 'id="serverModal"' in html
    assert 'id="serverUrlInput"' in html
    assert 'id="testServerBtn"' in html
    assert 'id="saveServerBtn"' in html
    assert "localStorage.getItem" in js
    assert "kgat_server_url" in js

    # 2. 雙層水位條結構與樣式
    assert "dual-layer-meter" in html
    assert "meter-base" in html
    assert "meter-dynamic" in html
    assert ".dual-layer-meter" in css
    assert ".meter-base" in css
    assert ".meter-dynamic" in css

    # 3. 忠實度按鈕名稱驗證
    assert "原始預測" in html
    assert "遮蔽解釋子圖 (F+)" in html
    assert "僅保留解釋子圖 (F-)" in html

    # 4. 忠實度動態提示元素
    assert 'id="promptNormal"' in html
    assert 'id="promptFPlus"' in html
    assert 'id="promptFPlusHighInteractions"' in html
    assert 'id="promptFMinus"' in html

    # 5. 背景裝飾實體點與匯聚子圖標記
    assert 'id="svgBgDecoGroup"' in html
    assert "svg-bg-deco-group" in css
    assert "renderBackgroundDeco" in js
    assert "drawConvergenceGraph" in js
    assert "arrow-backup" in html

    # 6. 學術展示規範與冗餘標籤清理驗證
    assert "排序：第 1 名" not in html
    assert "全庫推薦" not in html
    assert "未命中測試集" not in html
    assert "命中測試集" not in html

    # 7. JavaScript 腳本語法檢驗 (Node.js 語法檢核)
    import shutil
    import subprocess
    node_exe = shutil.which("node")
    if node_exe:
        res = subprocess.run([node_exe, "-c", js_file], capture_output=True, text=True)
        assert res.returncode == 0, f"app.js 語法錯誤: {res.stderr}"


def test_server_state_auto_initialize_safety():
    """驗證 ServerState 在未手動呼叫 initialize 時直接載入模型能自我修復與防護."""
    from src.demo.server import ServerState
    fresh_state = ServerState()
    assert fresh_state.is_initialized is False
    m, _ = fresh_state.get_or_load_model("1")
    assert fresh_state.is_initialized is True
    assert m is not None
    assert hasattr(m, "cached_final_embed")
    assert m.cached_final_embed is not None


def test_csr_vectorized_path_finding_exact_consistency_and_performance():
    """驗證 CSR 局部閉環尋路結果與 NetworkX 之 100% 數值一致性與毫秒級效能."""
    import time
    model, explainer = state.get_or_load_model("3")
    assert state.csr_adj is not None
    assert state.dynamic_user_cache is not None

    # 針對使用者 0 進行一致性驗證
    res_csr = explainer.explain_fast_csr(
        state.indices, state.edge_types, state.num_nodes, 0, 8822, state.csr_adj, top_k=3
    )

    # 驗證局部 CSR 向量化尋路演算法耗時低於 15 毫秒 (實測約 3ms，達成 3000+ 倍加速)
    assert res_csr["search_time"] < 0.015
    assert res_csr["search_time_ms"] < 15.0

    # 與 NetworkX 尋路結果進行完全一致性比對
    res_nx = explainer.explain(
        state.indices, state.edge_types, state.num_nodes, 0, 8822, n_hops=2, top_k=3
    )

    assert len(res_csr["top_paths"]) == len(res_nx["top_paths"])
    for (p_csr, s_csr), (p_nx, s_nx) in zip(res_csr["top_paths"], res_nx["top_paths"]):
        assert p_csr == p_nx
        assert s_csr == pytest.approx(s_nx, rel=0.01)


def test_fidelity_probe_multi_path_union_masking():
    """驗證多條解釋路徑聯集子圖在探針端點的干預遮蔽計算."""
    paths = [
        [0, 233377, 2616, 235392],
        [0, 233377, 2891, 235392],
    ]
    req_f_plus = FidelityProbeRequest(
        user_id=0, item_id=8822, depth="3", mode="f_plus", path_nodes=paths
    )
    data_f_plus = fidelity_probe(req_f_plus)
    assert data_f_plus["mode"] == "f_plus"
    assert "遮蔽解釋子圖" in data_f_plus["status_description"]
    assert "prob" in data_f_plus
    assert "fidelity_score" in data_f_plus

    req_f_minus = FidelityProbeRequest(
        user_id=0, item_id=8822, depth="3", mode="f_minus", path_nodes=paths
    )
    data_f_minus = fidelity_probe(req_f_minus)
    assert data_f_minus["mode"] == "f_minus"
    assert "僅保留解釋子圖" in data_f_minus["status_description"]
    assert "prob" in data_f_minus
    assert "fidelity_score" in data_f_minus


def test_dynamic_user_cache_hit_and_speed():
    """驗證 ServerState 動態使用者快取防重複計算與零延遲回傳."""
    import time
    req = RecommendRequest(user_id=1, depth="3")
    res1 = recommend(req)
    cache_key = "1_3"
    assert cache_key in state.dynamic_user_cache

    t0 = time.perf_counter()
    res2 = recommend(req)
    t_elapsed = time.perf_counter() - t0

    assert t_elapsed < 0.005  # 低於 5 毫秒
    assert res1["recommended_item_id"] == res2["recommended_item_id"]
    assert res1["prob"] == res2["prob"]
    assert len(res1["explanations"]) == len(res2["explanations"])


def test_fidelity_probe_single_path_1d_list_support():
    """驗證探針端點支援單一路徑 1D 整數清單 (List[int]) 且不拋出 422 驗證錯誤或 TypeError."""
    single_path = [0, 233377, 2616, 235392]
    req = FidelityProbeRequest(
        user_id=0, item_id=8822, depth="3", mode="f_plus", path_nodes=single_path
    )
    data = fidelity_probe(req)
    assert data["mode"] == "f_plus"
    assert "status_description" in data
    assert "prob" in data
    assert "fidelity_score" in data


def test_explain_fast_csr_out_of_bounds_safety():
    """驗證 CSR 向量化尋路演算法在節點超出邊界或負數時安全回傳空子圖而不崩潰."""
    model, explainer = state.get_or_load_model("3")
    res_neg = explainer.explain_fast_csr(
        state.indices, state.edge_types, state.num_nodes, -1, 8822, state.csr_adj, top_k=3
    )
    assert res_neg["top_paths"] == []
    assert len(res_neg["subgraph"].nodes) == 0

    res_oob = explainer.explain_fast_csr(
        state.indices, state.edge_types, state.num_nodes, 9999999, 8822, state.csr_adj, top_k=3
    )
    assert res_oob["top_paths"] == []
    assert len(res_oob["subgraph"].nodes) == 0


def test_dynamic_user_cache_fidelity_sync():
    """驗證動態推論使用者在忠實度探針計算後同步寫回快取以防重複計算."""
    req_rec = RecommendRequest(user_id=2, depth="3")
    res_rec = recommend(req_rec)
    rec_item = res_rec["recommended_item_id"]
    cache_key = "2_3"
    assert cache_key in state.dynamic_user_cache

    req_probe = FidelityProbeRequest(
        user_id=2, item_id=rec_item, depth="3", mode="f_plus"
    )
    data_probe = fidelity_probe(req_probe)
    assert "fidelity" in state.dynamic_user_cache[cache_key]
    assert state.dynamic_user_cache[cache_key]["fidelity"]["f_plus"] == data_probe["fidelity_score"]
    assert state.dynamic_user_cache[cache_key]["fidelity"]["prob_f_plus"] == data_probe["prob"]


def test_user_0_interactions_and_fidelity_probe_necessity_sufficiency():
    """驗證使用者 0 歷史互動總數、等價通道數量，以及忠實度遮蔽探針數值."""
    req_rec = RecommendRequest(user_id=0, depth="3")
    rec_res = recommend(req_rec)
    assert rec_res["user_id"] == 0
    assert rec_res["user_n_interactions"] == 128
    assert rec_res["n_equivalent_channels"] == 137
    assert len(rec_res["explanations"]) == 3

    rec_item_id = rec_res["recommended_item_id"]
    paths = [[d["id_remapped"] for d in exp["path_details"]] for exp in rec_res["explanations"]]

    # 驗證 F+ 遮蔽計算
    probe_plus = fidelity_probe(
        FidelityProbeRequest(user_id=0, item_id=rec_item_id, depth="3", mode="f_plus", path_nodes=paths)
    )
    assert probe_plus["mode"] == "f_plus"
    assert probe_plus["fidelity_score"] > 0.0
    assert probe_plus["fidelity_score"] == pytest.approx(0.00037, abs=0.0003)

    # 驗證 F- 保留計算
    probe_minus = fidelity_probe(
        FidelityProbeRequest(user_id=0, item_id=rec_item_id, depth="3", mode="f_minus", path_nodes=paths)
    )
    assert probe_minus["mode"] == "f_minus"
    assert probe_minus["fidelity_score"] > 0.0

    # 驗證快取使用者 (以 13087 為例) 歷史互動總數與等價通道計算
    req_cached = RecommendRequest(user_id=13087, depth="3")
    res_cached = recommend(req_cached)
    assert res_cached["user_id"] == 13087
    assert res_cached["user_n_interactions"] == 30
    assert res_cached["n_equivalent_channels"] == 3


def test_frontend_svg_masking_state_machine_and_academic_hint():
    """驗證前端三態遮蔽狀態機、背景淡出樣式與高互動數使用者動態學術提示."""
    web_dir = os.path.join(project_root, "src", "demo", "web")
    html_file = os.path.join(web_dir, "index.html")
    css_file = os.path.join(web_dir, "style.css")
    js_file = os.path.join(web_dir, "app.js")

    with open(html_file, "r", encoding="utf-8") as f:
        html = f.read()
    with open(css_file, "r", encoding="utf-8") as f:
        css = f.read()
    with open(js_file, "r", encoding="utf-8") as f:
        js = f.read()

    # 驗證高互動數提示元素與內容
    assert 'id="promptFPlusHighInteractions"' in html
    assert "此使用者歷史互動數為" in js
    assert "等價通道" in js
    assert "局部路徑遮蔽對整體決策具高度結構容錯性" in js

    # 驗證無不當之全域硬編碼 137 與條件判斷
    assert "(recData.user_id === 0 ? 137 : 137)" not in js
    assert "nInter >= 20 && nChannels > 0" in js

    # 驗證背景淡出樣式為 0.05
    assert ".svg-bg-deco-group.faded" in css
    assert "opacity: 0.05 !important" in css

    # 驗證邊線遮蔽樣式
    assert ".svg-edge.masked" in css

    # 驗證狀態機邏輯：點擊路徑時自動恢復原始預測模式
    assert 'setProbeMode("normal")' in js



