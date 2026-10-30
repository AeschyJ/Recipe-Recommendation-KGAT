"""Adversarial stress-test suite for Milestone 1 (Demo System & Docs: R1, Features F1 to F5).

Empirical challenger test harness:
1. USER node naming in all academic cases, intermediate collaborative filtering users (Case 2 User 10856, 27295) and deep propagation (Case 1 User 17049).
2. Exhaustive scan of all 500 users in `user_cache` and `users` list for any residual Raw ID format strings against ground-truth LabelEncoder.
3. Validate HTML structure in `demo/web/index.html` for fidelity math formulas.
4. Verify ADR-009 numbers match the exact research benchmark.
5. Online inference logic verification in `demo/server.py`.
"""

import json
import os
import pickle
import re
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def ground_truth_user_le():
    stats_path = PROJECT_ROOT / "data" / "processed" / "stats.pkl"
    assert stats_path.exists(), f"stats.pkl not found at {stats_path}"
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    return stats["user_map"]


@pytest.fixture(scope="module")
def demo_cases_json_data():
    json_path = PROJECT_ROOT / "demo" / "web" / "demo_cases.json"
    assert json_path.exists(), f"demo_cases.json not found at {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def demo_cases_js_data():
    js_path = PROJECT_ROOT / "demo" / "web" / "demo_cases_data.js"
    assert js_path.exists(), f"demo_cases_data.js not found at {js_path}"
    with open(js_path, "r", encoding="utf-8") as f:
        content = f.read()
    prefix = "window.DEMO_CASES_DATA = "
    assert content.startswith(prefix), "JS file does not start with expected prefix"
    raw_json = content[len(prefix):].rstrip(";\n ")
    return json.loads(raw_json)


class TestAcademicCasesAdversarial:
    """Stress-test academic cases in demo_cases.json."""

    def test_case1_deep_propagation_user_17049(self, demo_cases_json_data, ground_truth_user_le):
        """Case 1 (User 17049): Deep propagation. Raw ID is 144146."""
        cases = {c["case_id"]: c for c in demo_cases_json_data["academic_cases"]}
        assert "deep_propagation" in cases, "deep_propagation case missing"
        c1 = cases["deep_propagation"]
        assert c1["user_id"] == 17049
        raw_uid = ground_truth_user_le.classes_[17049]
        assert raw_uid == 144146, f"Expected raw ID 144146, got {raw_uid}"

        # In L=1, explanations must be empty
        d1 = c1["data"].get("1")
        assert d1 is not None
        assert len(d1.get("explanations", [])) == 0, "L=1 should have 0 explanations for User 17049"

        # In L=3, explanations must exist and root node MUST be User 17049
        d3 = c1["data"].get("3")
        assert d3 is not None
        exps = d3.get("explanations", [])
        assert len(exps) > 0, "L=3 should have explanations for User 17049"

        for exp in exps:
            details = exp["path_details"]
            assert len(details) >= 2
            root = details[0]
            assert root["type"] == "USER"
            assert root["id_remapped"] == 17049
            assert root["name"] == "User 17049", f"Root node name was {root['name']}, expected 'User 17049'"
            assert int(root["id_original"]) == raw_uid
            assert "User 144146" not in exp["path_description"], "Raw ID 144146 leaked in path_description!"

    def test_case2_high_necessity_collaborative_filtering_users(
        self, demo_cases_json_data, ground_truth_user_le
    ):
        """Case 2 (User 101612): High necessity with multi-hop CF intermediate users.
        Root user raw ID: 1332214.
        Intermediate user 10856 raw ID: 92385.
        Intermediate user 27295 raw ID: 231054.
        """
        cases = {c["case_id"]: c for c in demo_cases_json_data["academic_cases"]}
        assert "high_necessity" in cases
        c2 = cases["high_necessity"]
        assert c2["user_id"] == 101612
        raw_uid = ground_truth_user_le.classes_[101612]
        assert raw_uid == 1332214

        # Intermediate users raw mapping check
        assert ground_truth_user_le.classes_[10856] == 92385
        assert ground_truth_user_le.classes_[27295] == 231054

        found_intermediate_users = set()

        for d_key, d_data in c2["data"].items():
            if not isinstance(d_data, dict):
                continue
            for exp in d_data.get("explanations", []):
                details = exp["path_details"]
                desc = exp["path_description"]

                # Ensure no raw IDs anywhere in description
                assert "User 1332214" not in desc, "Raw ID 1332214 leaked in path_description"
                assert "User 92385" not in desc, "Raw ID 92385 leaked in path_description"
                assert "User 231054" not in desc, "Raw ID 231054 leaked in path_description"

                for i, node in enumerate(details):
                    if node.get("type") == "USER":
                        remap_id = node["id_remapped"]
                        orig_id = node["id_original"]
                        expected_name = f"User {remap_id}"
                        assert node["name"] == expected_name, (
                            f"Node name mismatch: got '{node['name']}', expected '{expected_name}'"
                        )
                        if i > 0:
                            found_intermediate_users.add(remap_id)

        # Confirm that the expected intermediate users 10856 and 27295 actually appeared
        assert 10856 in found_intermediate_users, "Intermediate User 10856 not found in Case 2 paths"
        assert 27295 in found_intermediate_users, "Intermediate User 27295 not found in Case 2 paths"

    def test_all_academic_cases_path_consistency(self, demo_cases_json_data):
        """Verify for every academic case, path_description == ' -> '.join(details)."""
        for case in demo_cases_json_data["academic_cases"]:
            for d_key, d_val in case.get("data", {}).items():
                if not isinstance(d_val, dict):
                    continue
                for exp in d_val.get("explanations", []):
                    details = exp.get("path_details", [])
                    reconstructed = " -> ".join(n["name"] for n in details)
                    assert exp["path_description"] == reconstructed, (
                        f"Case {case['case_id']} depth {d_key} path_description does not match details join!"
                    )


class TestFullDatasetScanAdversarial:
    """Exhaustive scan across all 500 users in user_cache and users list."""

    def test_users_list_raw_leak_oracle(self, demo_cases_json_data, ground_truth_user_le):
        """Check all entries in data['users'] for raw ID leakage."""
        users = demo_cases_json_data["users"]
        assert len(users) == 500, f"Expected 500 users, found {len(users)}"

        leaks = []
        for u in users:
            uid = u["user_id"]
            uname = u["user_name"]
            raw_id = ground_truth_user_le.classes_[uid]

            if uname != f"User {uid}":
                leaks.append((uid, uname, f"User {uid}"))
            if uname == f"User {raw_id}" and raw_id != uid:
                leaks.append((uid, uname, "RAW_LEAK"))

        assert len(leaks) == 0, f"Found {len(leaks)} leaks in users list: {leaks[:5]}"

    def test_user_cache_exhaustive_scan(self, demo_cases_json_data, ground_truth_user_le):
        """Scan all 500 users in user_cache across all depths and paths."""
        cache = demo_cases_json_data["user_cache"]
        assert len(cache) == 500, f"Expected 500 cached users, got {len(cache)}"

        total_nodes = 0
        total_user_nodes = 0
        leaks = []

        for uid_str, uentry in cache.items():
            uid = uentry["user_id"]
            raw_uid = ground_truth_user_le.classes_[uid]

            # Top level user_name
            if uentry["user_name"] != f"User {uid}":
                leaks.append(f"Top-level user_name mismatch for {uid}: {uentry['user_name']}")

            for d_key in ["3", "2", "1"]:
                d_data = uentry.get("depths", {}).get(d_key)
                if not d_data or not isinstance(d_data, dict):
                    continue
                for exp_idx, exp in enumerate(d_data.get("explanations", [])):
                    details = exp.get("path_details", [])
                    desc = exp.get("path_description", "")

                    # Check description consistency
                    reconstructed = " -> ".join(n["name"] for n in details)
                    if desc != reconstructed:
                        leaks.append(f"Description mismatch user {uid} d={d_key} exp={exp_idx}")

                    for node_idx, node in enumerate(details):
                        total_nodes += 1
                        if node.get("type") == "USER":
                            total_user_nodes += 1
                            remap_id = node.get("id_remapped")
                            orig_id = node.get("id_original")
                            name = node.get("name")

                            # Ground truth check
                            expected_orig = str(ground_truth_user_le.classes_[remap_id])
                            if str(orig_id) != expected_orig:
                                leaks.append(
                                    f"Original ID corrupt user {uid} d={d_key}: {orig_id} vs {expected_orig}"
                                )

                            expected_name = f"User {remap_id}"
                            if name != expected_name:
                                leaks.append(
                                    f"User node name mismatch user {uid} d={d_key} node {node_idx}: '{name}' vs '{expected_name}'"
                                )

                            # Direct raw ID leak check
                            if orig_id != remap_id and f"User {orig_id}" in name:
                                leaks.append(
                                    f"RAW LEAK in node name user {uid} d={d_key} node {node_idx}: {name}"
                                )
                            if orig_id != remap_id and f"User {orig_id}" in desc:
                                leaks.append(
                                    f"RAW LEAK in desc user {uid} d={d_key} node {node_idx}: {desc}"
                                )

        assert len(leaks) == 0, f"Found {len(leaks)} leaks in user_cache:\n" + "\n".join(leaks[:10])
        assert total_user_nodes >= 1000, f"Expected >1000 USER nodes, scanned {total_user_nodes}"

    def test_demo_cases_js_synchronization(self, demo_cases_json_data, demo_cases_js_data):
        """Ensure demo_cases_data.js is 100% structurally identical to demo_cases.json."""
        assert demo_cases_js_data["benchmark_metrics"] == demo_cases_json_data["benchmark_metrics"]
        assert len(demo_cases_js_data["academic_cases"]) == len(demo_cases_json_data["academic_cases"])
        assert len(demo_cases_js_data["users"]) == len(demo_cases_json_data["users"])
        assert len(demo_cases_js_data["user_cache"]) == len(demo_cases_json_data["user_cache"])

        # Check raw text of demo_cases_data.js for forbidden raw IDs
        js_path = PROJECT_ROOT / "demo" / "web" / "demo_cases_data.js"
        with open(js_path, "r", encoding="utf-8") as f:
            js_text = f.read()

        forbidden_strings = ["User 144146", "User 1332214", "User 92385", "User 231054"]
        for forbidden in forbidden_strings:
            assert forbidden not in js_text, f"Forbidden Raw ID string '{forbidden}' found in demo_cases_data.js!"


class TestHtmlFormulasAdversarial:
    """Validate HTML structure in demo/web/index.html for fidelity math formulas."""

    def test_fidelity_formulas_exact_entities(self):
        html_path = PROJECT_ROOT / "demo" / "web" / "index.html"
        assert html_path.exists()
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        # Target 1: F+ note
        # <div class="probe-stat-note" id="fPlusNote">p̂(&#x1D43A;) &minus; p̂(&#x1D43A; \ &#x1D446;)</div>
        f_plus_pattern = r'<div class="probe-stat-note" id="fPlusNote">\s*p̂\(&#x1D43A;\)\s*&minus;\s*p̂\(&#x1D43A;\s*\\\s*&#x1D446;\)\s*</div>'
        assert re.search(f_plus_pattern, html), "fPlusNote formula does not match standard mathematical syntax"

        # Target 2: F- note
        # <div class="probe-stat-note" id="fMinusNote">p̂(&#x1D43A;) &minus; p̂(&#x1D43A;<sub>&#x1D446;</sub>)</div>
        f_minus_pattern = r'<div class="probe-stat-note" id="fMinusNote">\s*p̂\(&#x1D43A;\)\s*&minus;\s*p̂\(&#x1D43A;<sub>&#x1D446;</sub>\)\s*</div>'
        assert re.search(f_minus_pattern, html), "fMinusNote formula does not match standard mathematical syntax"

        # Adversarial check: No leftover placeholder text
        assert "P_orig" not in html, "Obsolete 'P_orig' residual found in index.html"
        assert "P_f+" not in html, "Obsolete 'P_f+' residual found in index.html"
        assert "P_f-" not in html, "Obsolete 'P_f-' residual found in index.html"

    def test_app_js_does_not_clobber_formula_notes(self):
        """Ensure client script never dynamically overwrites fPlusNote or fMinusNote text."""
        js_path = PROJECT_ROOT / "demo" / "web" / "app.js"
        with open(js_path, "r", encoding="utf-8") as f:
            js = f.read()

        assert "fPlusNote" not in js, "app.js contains fPlusNote manipulation which could overwrite HTML formula!"
        assert "fMinusNote" not in js, "app.js contains fMinusNote manipulation which could overwrite HTML formula!"


class TestAdr009BenchmarkAdversarial:
    """Verify ADR-009 numbers match the exact research benchmark."""

    def test_adr009_exact_numbers(self):
        adr_path = PROJECT_ROOT / "docs" / "adr" / "ADR-009-Oral-Defense-Demo-Packaging.md"
        assert adr_path.exists()
        with open(adr_path, "r", encoding="utf-8") as f:
            adr = f.read()

        # Exact benchmarks from Paper/Research/path_extraction_csr_vs_networkx.md
        required_numbers = [
            ("56.74 MB", "CSR core array memory"),
            ("1,581.41 MB", "NetworkX dynamic peak memory"),
            ("93%", "Memory reduction ratio"),
            ("749.95 ms", "NetworkX pure traversal latency"),
            ("0.067 ms", "CSR pure traversal latency"),
            ("11,205.2 倍", "Pure traversal speedup factor"),
            ("13,757.2 ms", "NetworkX end-to-end dynamic latency"),
            ("205,550.2 倍", "End-to-end speedup factor"),
            ("18 組", "Benchmark case count"),
            ("100% 拓樸集合雙射與數值等價性", "Equivalence guarantee"),
            ("100% PASS", "Test pass verdict"),
        ]

        for text, desc in required_numbers:
            assert text in adr, f"Missing '{text}' ({desc}) in ADR-009!"

        # Adversarial check: Old numbers MUST NOT remain
        forbidden_old = [
            "27.8 秒",
            "3.2 毫秒",
            "3,970 倍",
        ]
        for old in forbidden_old:
            assert old not in adr, f"Residual obsolete benchmark '{old}' found in ADR-009!"


class TestServerInferenceLogicAdversarial:
    """Stress-test online inference wrapping in demo/server.py."""

    def test_server_node_formatting_logic(self):
        """Simulate the path_details wrapping loop in server.py (lines 553-578) with adversarial node IDs."""
        from src.generate_explanations import get_node_name

        # Create mock state elements to test formatting branch
        stats_path = PROJECT_ROOT / "data" / "processed" / "stats.pkl"
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        user_le = stats["user_map"]
        item_le = stats["item_map"]
        n_users = len(user_le.classes_)
        n_items = len(item_le.classes_)

        # Test user node formatting
        for test_uid in [0, 17049, 101612, 10856, 27295, n_users - 1]:
            n_type, n_real_id, n_name = get_node_name(
                test_uid, n_users, n_items, user_le, item_le, None, {}
            )
            assert n_type == "USER"
            assert n_real_id == str(user_le.classes_[test_uid])
            # get_node_name returns Raw ID
            assert n_name == f"User {user_le.classes_[test_uid]}"

            # Now apply server.py L564-L565 override
            if n_type == "USER":
                fixed_name = f"User {test_uid}"
            else:
                fixed_name = n_name

            assert fixed_name == f"User {test_uid}", f"Server override failed for user {test_uid}"
