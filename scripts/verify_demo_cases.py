import json
import re
import sys

def verify_demo_cases():
    json_path = "demo/web/demo_cases.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=== 1. Academic Cases Verification ===")
    for case in data["academic_cases"]:
        cid = case["case_id"]
        uid = case["user_id"]
        print(f"\nCase: {cid} (User {uid})")
        for d in ["3", "2", "1"]:
            depth_data = case["data"].get(d)
            if not depth_data or not depth_data.get("explanations"):
                continue
            for i, exp in enumerate(depth_data["explanations"]):
                desc = exp["path_description"]
                print(f"  Depth {d} Path {i}: {desc}")
                for node in exp["path_details"]:
                    if node.get("type") == "USER":
                        expected_name = f"User {node['id_remapped']}"
                        actual_name = node.get("name")
                        assert actual_name == expected_name, (
                            f"Mismatch in node name! Expected {expected_name}, got {actual_name}"
                        )
                # Verify that no Raw ID appears in description
                # Check path_details names join
                expected_desc = " -> ".join(n["name"] for n in exp["path_details"])
                assert desc == expected_desc, (
                    f"Mismatch in path_description! Expected '{expected_desc}', got '{desc}'"
                )

    print("\n=== 2. Full Dataset Scan for Raw IDs in USER Nodes ===")
    raw_leak_count = 0
    total_user_nodes = 0
    
    # Check users list
    for u in data["users"]:
        uid = u["user_id"]
        uname = u["user_name"]
        expected_uname = f"User {uid}"
        if uname != expected_uname:
            print(f"ERROR: User list mismatch: uid={uid}, uname='{uname}'")
            raw_leak_count += 1

    # Check user_cache
    for uid_str, uinfo in data["user_cache"].items():
        uid = uinfo["user_id"]
        uname = uinfo["user_name"]
        expected_uname = f"User {uid}"
        if uname != expected_uname:
            print(f"ERROR: User cache mismatch: uid={uid}, uname='{uname}'")
            raw_leak_count += 1
        
        for d_key, d_val in uinfo.get("depths", {}).items():
            if isinstance(d_val, dict) and d_val.get("explanations"):
                for exp in d_val["explanations"]:
                    desc = exp.get("path_description", "")
                    details = exp.get("path_details", [])
                    reconstructed_desc = " -> ".join(n["name"] for n in details)
                    if desc != reconstructed_desc:
                        print(f"ERROR: Description mismatch for user {uid}: '{desc}' vs '{reconstructed_desc}'")
                        raw_leak_count += 1
                    for node in details:
                        if node.get("type") == "USER":
                            total_user_nodes += 1
                            expected = f"User {node['id_remapped']}"
                            if node.get("name") != expected:
                                print(f"ERROR: Node name mismatch for user {uid}: expected '{expected}', got '{node.get('name')}'")
                                raw_leak_count += 1
                            if str(node.get("id_original")) == node.get("name").replace("User ", "") and node.get("id_remapped") != node.get("id_original"):
                                print(f"ERROR: Raw ID leaked in node name for user {uid}: id_original={node.get('id_original')}, name={node.get('name')}")
                                raw_leak_count += 1

    print(f"Total USER nodes checked: {total_user_nodes}")
    print(f"Raw ID leak count: {raw_leak_count}")
    assert raw_leak_count == 0, f"Found {raw_leak_count} raw ID leaks!"
    print("ALL USER NODES VERIFIED: 100% Remapped IDs, ZERO Raw ID leaks!")

    print("\n=== 3. HTML File Entities Verification ===")
    with open("demo/web/index.html", "r", encoding="utf-8") as f:
        html = f.read()
    assert "p̂(&#x1D43A;) &minus; p̂(&#x1D43A; \\ &#x1D446;)" in html, "fPlusNote formula missing!"
    assert "p̂(&#x1D43A;) &minus; p̂(&#x1D43A;<sub>&#x1D446;</sub>)" in html, "fMinusNote formula missing!"
    assert "P_orig - P_f+" not in html, "Old P_orig - P_f+ still present in HTML!"
    assert "P_orig - P_f-" not in html, "Old P_orig - P_f- still present in HTML!"
    print("HTML Entities Verified Successfully!")

    print("\n=== 4. ADR-009 Document Benchmark Verification ===")
    with open("docs/adr/ADR-009-Oral-Defense-Demo-Packaging.md", "r", encoding="utf-8") as f:
        adr = f.read()
    assert "56.74 MB" in adr, "56.74 MB missing in ADR-009!"
    assert "0.067 ms" in adr, "0.067 ms missing in ADR-009!"
    assert "11,205.2 倍" in adr, "11,205.2 倍 missing in ADR-009!"
    assert "205,550.2 倍" in adr, "205,550.2 倍 missing in ADR-009!"
    assert "100% PASS" in adr, "100% PASS missing in ADR-009!"
    print("ADR-009 Document Verified Successfully!")

    print("\n=== 5. Offline JS Cache Verification ===")
    with open("demo/web/demo_cases_data.js", "r", encoding="utf-8") as f:
        js = f.read()
    assert js.startswith("window.DEMO_CASES_DATA = "), "demo_cases_data.js header missing!"
    assert "User 10856" in js, "User 10856 missing in demo_cases_data.js!"
    assert "User 27295" in js, "User 27295 missing in demo_cases_data.js!"
    print("Offline JS Cache Verified Successfully!")

    print("\n>>> ALL CHECKS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    verify_demo_cases()
