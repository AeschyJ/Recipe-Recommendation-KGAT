"""Adversarial stress-test suite specifically targeting online inference packaging (`demo/server.py`),
edge-case paths, unknown nodes, boundary user IDs, and offline `demo_cases_data.js` synchronization.

Challenger: teamwork_challenger (Instance 2)
"""

import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
import pytest
from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from demo.server import (
    state,
    recommend,
    RecommendRequest,
    fidelity_probe,
    FidelityProbeRequest,
)
from src.generate_explanations import get_node_name


@pytest.fixture(scope="module", autouse=True)
def initialize_server_state():
    """Ensure server state is initialized once for the module."""
    state.initialize()
    yield


@pytest.fixture(scope="module")
def ground_truth_user_le():
    stats_path = PROJECT_ROOT / "data" / "processed" / "stats.pkl"
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    return stats["user_map"]


class TestDemoCasesDataJsOfflineSynchronization:
    """Stress-test offline demo_cases_data.js vs demo_cases.json."""

    def test_js_and_json_exact_data_equality(self):
        json_path = PROJECT_ROOT / "demo" / "web" / "demo_cases.json"
        js_path = PROJECT_ROOT / "demo" / "web" / "demo_cases_data.js"

        assert json_path.exists(), "demo_cases.json does not exist"
        assert js_path.exists(), "demo_cases_data.js does not exist"

        with open(json_path, "r", encoding="utf-8") as f:
            json_obj = json.load(f)

        with open(js_path, "r", encoding="utf-8") as f:
            js_text = f.read()

        prefix = "window.DEMO_CASES_DATA = "
        assert js_text.startswith(prefix), f"demo_cases_data.js must start with '{prefix}'"
        raw_json_str = js_text[len(prefix):].rstrip(";\n ")
        js_obj = json.loads(raw_json_str)

        # Deep equality check
        assert js_obj == json_obj, "demo_cases_data.js content does NOT deeply equal demo_cases.json!"

    def test_js_file_syntax_validity(self):
        js_path = PROJECT_ROOT / "demo" / "web" / "demo_cases_data.js"
        node_exe = shutil.which("node")
        if node_exe:
            res = subprocess.run([node_exe, "-c", str(js_path)], capture_output=True, text=True)
            assert res.returncode == 0, f"demo_cases_data.js syntax check failed: {res.stderr}"

    def test_no_raw_user_id_in_js_file(self, ground_truth_user_le):
        js_path = PROJECT_ROOT / "demo" / "web" / "demo_cases_data.js"
        with open(js_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check prominent raw IDs that existed previously
        known_raw_ids = [144146, 1332214, 92385, 231054]
        for rid in known_raw_ids:
            assert f"User {rid}" not in content, f"Raw ID leak in demo_cases_data.js: User {rid}"


class TestServerOnlineInferencePackagingAdversarial:
    """Adversarially challenge online inference execution in demo/server.py."""

    @pytest.mark.parametrize("depth", ["1", "2", "3"])
    @pytest.mark.parametrize("uid", [0, 1, 99999, 12345])
    def test_online_inference_uncached_users_depth_1_2_3(self, uid, depth, ground_truth_user_le):
        """Call recommend() for boundary and uncached users across depths 1, 2, 3.
        Verify that under dynamic PyTorch/CSR inference, NO raw USER ID is emitted.
        """
        # Evict from dynamic_user_cache if present to force fresh inference
        cache_key = f"{uid}_{depth}"
        if cache_key in state.dynamic_user_cache:
            del state.dynamic_user_cache[cache_key]

        req = RecommendRequest(user_id=uid, depth=depth)
        res = recommend(req)

        # 1. Top-level user_name check
        assert res["user_id"] == uid
        assert res["user_name"] == f"User {uid}", f"Top-level user_name was '{res['user_name']}', expected 'User {uid}'"

        raw_uid = ground_truth_user_le.classes_[uid]
        if raw_uid != uid:
            assert res["user_name"] != f"User {raw_uid}", f"Top-level user_name leaked raw ID: {res['user_name']}"

        # 2. Recommended item name check
        assert isinstance(res["recommended_item_name"], str)
        assert len(res["recommended_item_name"]) > 0

        # 3. Explanations check
        exps = res.get("explanations", [])
        for exp_idx, exp in enumerate(exps):
            details = exp.get("path_details", [])
            desc = exp.get("path_description", "")
            struct = exp.get("path_structure", "")

            # Structural consistency
            expected_struct = " -> ".join(n["type"] for n in details)
            assert struct == expected_struct, f"Struct mismatch at exp {exp_idx}: {struct} vs {expected_struct}"

            expected_desc = " -> ".join(n["name"] for n in details)
            assert desc == expected_desc, f"Desc mismatch at exp {exp_idx}: {desc} vs {expected_desc}"

            # Every USER node check (including intermediate users in CF paths)
            for n_idx, node in enumerate(details):
                n_type = node.get("type")
                if n_type == "USER":
                    remap_nid = node.get("id_remapped")
                    assert remap_nid < state.n_users, f"User node id_remapped {remap_nid} exceeds n_users"
                    orig_nid = node.get("id_original")
                    name = node.get("name")

                    expected_name = f"User {remap_nid}"
                    assert name == expected_name, f"USER node name mismatch: got '{name}', expected '{expected_name}'"

                    raw_node_uid = str(ground_truth_user_le.classes_[remap_nid])
                    assert str(orig_nid) == raw_node_uid, f"id_original mismatch: {orig_nid} vs {raw_node_uid}"

                    if str(remap_nid) != raw_node_uid:
                        assert f"User {raw_node_uid}" not in name, f"Node name leaked raw ID: {name}"
                        assert f"User {raw_node_uid}" not in desc, f"Path description leaked raw ID: {desc}"

    def test_online_inference_boundary_user_ids(self):
        """Test boundary conditions for user IDs."""
        max_valid_uid = state.n_users - 1

        # Test valid upper boundary
        req_max = RecommendRequest(user_id=max_valid_uid, depth="1")
        res_max = recommend(req_max)
        assert res_max["user_id"] == max_valid_uid
        assert res_max["user_name"] == f"User {max_valid_uid}"

        # Test invalid out-of-bound user IDs: should raise HTTPException 400
        invalid_uids = [-1, -100, state.n_users, state.n_users + 1, 9999999]
        for bad_uid in invalid_uids:
            with pytest.raises(HTTPException) as exc_info:
                recommend(RecommendRequest(user_id=bad_uid, depth="3"))
            assert exc_info.value.status_code == 400, f"Expected 400 for uid {bad_uid}, got {exc_info.value.status_code}"

    def test_online_inference_invalid_depths(self):
        """Test invalid depth arguments to recommend endpoint."""
        invalid_depths = ["0", "4", "5", "unknown", "", "None"]
        for bad_depth in invalid_depths:
            with pytest.raises(HTTPException) as exc_info:
                recommend(RecommendRequest(user_id=17049, depth=bad_depth))
            assert exc_info.value.status_code == 400


class TestPathPackagingAdversarialEdgeCases:
    """Simulate exotic, adversarial paths that an explainer might generate under extreme graph topology,
    and verify server.py's packaging loop (lines 553-578) never emits raw IDs or crashes.
    """

    def _package_mock_path(self, path):
        """Replicate server.py lines 553-578 exact packaging logic."""
        path_details = []
        path_desc_list = []
        for node_id in path:
            node_id = int(node_id)
            n_type, n_real_id, n_name = get_node_name(
                node_id,
                state.n_users,
                state.n_items,
                state.user_le,
                state.item_le,
                state.entity_maps,
                state.recipe_name_map,
            )
            if n_type == "USER":
                n_name = f"User {node_id}"
            path_details.append({
                "id_remapped": node_id,
                "id_original": n_real_id,
                "type": n_type,
                "name": n_name,
            })
            path_desc_list.append(n_name)
        return {
            "path_structure": " -> ".join([p["type"] for p in path_details]),
            "path_description": " -> ".join(path_desc_list),
            "path_details": path_details,
            "contribution_score": 0.5,
        }

    def test_cf_path_with_multiple_intermediate_users(self, ground_truth_user_le):
        """Path: USER(101612) -> RECIPE(229804) -> USER(10856) -> RECIPE(237751) -> USER(27295) -> RECIPE(241399)"""
        path = [101612, 229804, 10856, 237751, 27295, 241399]
        res = self._package_mock_path(path)

        assert res["path_structure"] == "USER -> RECIPE -> USER -> RECIPE -> USER -> RECIPE"

        # Check all three users
        expected_user_names = ["User 101612", "User 10856", "User 27295"]
        actual_user_names = [d["name"] for d in res["path_details"] if d["type"] == "USER"]
        assert actual_user_names == expected_user_names

        # Raw IDs must NOT be in path_description
        forbidden_raws = ["User 1332214", "User 92385", "User 231054"]
        for forbidden in forbidden_raws:
            assert forbidden not in res["path_description"]

    def test_boundary_node_ids_in_path(self):
        """Path with boundary user 0, boundary user n_users-1, boundary item n_users, max entity."""
        u_min = 0
        u_max = state.n_users - 1
        item_min = state.n_users
        item_max = state.n_users + state.n_items - 1
        ent_max = state.num_nodes - 1

        path = [u_min, item_min, u_max, item_max, ent_max]
        res = self._package_mock_path(path)

        # Users must have Remapped ID
        assert res["path_details"][0]["name"] == f"User {u_min}"
        assert res["path_details"][2]["name"] == f"User {u_max}"
        assert res["path_details"][0]["type"] == "USER"
        assert res["path_details"][2]["type"] == "USER"

    def test_unmapped_unknown_entity_node_safety(self):
        """What if an entity node ID is beyond normal entity maps?"""
        exotic_ent_id = state.num_nodes + 50000
        path = [0, state.n_users + 10, exotic_ent_id]
        res = self._package_mock_path(path)

        assert res["path_details"][0]["name"] == "User 0"
        assert res["path_details"][2]["type"] == "ENTITY"
        assert "Entity" in res["path_details"][2]["name"]
        assert res["path_description"].startswith("User 0 -> ")

    def test_empty_and_single_node_path(self):
        """Path with 0 nodes or 1 node."""
        res_empty = self._package_mock_path([])
        assert res_empty["path_structure"] == ""
        assert res_empty["path_description"] == ""
        assert res_empty["path_details"] == []

        res_single = self._package_mock_path([17049])
        assert res_single["path_structure"] == "USER"
        assert res_single["path_description"] == "User 17049"
        assert res_single["path_details"][0]["name"] == "User 17049"
