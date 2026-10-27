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
    assert data["status_description"] == "遮蔽此路徑：驗證必要性"
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
    assert data["status_description"] == "遮蔽其餘路徑：驗證充分性"
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
    assert '<h2 class="card-title">預測機率</h2>' in html_content
    assert "推薦結果與排序" not in html_content
    assert '<h2 class="card-title">忠實度分析</h2>' in html_content
    assert "因果干預驗證" not in html_content
    assert "必要性忠實度" in html_content
    assert "充分性忠實度" in html_content
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

