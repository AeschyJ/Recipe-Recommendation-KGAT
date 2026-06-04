import glob
import json
import os
import re


def simplify_explanations(input_dir, output_dir):
    # 清理不存在的原始檔案對應的簡化檔案
    if os.path.exists(output_dir):
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".json"):
                    out_path = os.path.join(root, file)
                    rel_path = os.path.relpath(out_path, output_dir)
                    in_path = os.path.join(input_dir, rel_path)
                    if not os.path.exists(in_path):
                        os.remove(out_path)
                        print(f"[清理] 刪除已失效的檔案: {out_path}")

    # Use recursive glob to match nested folders if they exist
    search_path = os.path.join(input_dir, "**", "*.json")
    for filepath in glob.glob(search_path, recursive=True):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"[警告] 無法讀取 {filepath}: {e}")
                continue

        simplified_data = []
        if isinstance(data, dict):
            simplified_data.append(data)
        for item in data:
            if isinstance(item, int):
                simplified_data.append(item)
                continue
            if not isinstance(item, dict):
                continue

            sim_item = {
                "user": item.get("user_name", str(item.get("user_id_remapped"))),
                "item": item.get(
                    "recommended_item_name",
                    str(item.get("recommended_item_id_remapped")),
                ),
                "score": round(item.get("score", 0.0), 4),
                "prob": round(item.get("original_prob", 0.0), 4),
                "explanations": [],
            }

            # 提取 Fidelity 分數 (選填)
            if item.get("fidelity"):
                fid = item.get("fidelity")
                sim_item["fidelity"] = {
                    "plus": round(fid.get("fidelity_plus", 0.0), 4),
                    "minus": round(fid.get("fidelity_minus", 0.0), 4),
                }

            for exp in item.get("explanations", []):
                sim_item["explanations"].append(
                    {
                        "path": exp.get("path_description", ""),
                        "score": round(exp.get("contribution_score", 0.0), 4),
                    }
                )
            simplified_data.append(sim_item)


        rel_path = os.path.relpath(filepath, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(simplified_data, f, indent=2, ensure_ascii=False)
        print(f"[轉換完成] 解釋數據已儲存至: {out_path}")


def simplify_logs(input_dir, output_dir):
    # 清理不存在的原始檔案對應的簡化檔案
    if os.path.exists(output_dir):
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".txt"):
                    out_path = os.path.join(root, file)
                    rel_path = os.path.relpath(out_path, output_dir)
                    in_path = os.path.join(input_dir, rel_path)
                    if not os.path.exists(in_path):
                        os.remove(out_path)
                        print(f"[清理] 刪除已失效的檔案: {out_path}")

    search_path = os.path.join(input_dir, "**", "*.txt")
    for filepath in glob.glob(search_path, recursive=True):
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        simplified_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if "100%|" in line or "s/it, loss=" in line or "s/it" in line:
                continue

            if "Saved checkpoint:" in line:
                continue

            clean_line = re.sub(
                r"^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}\s\[INFO\]\s", "", line
            )

            if clean_line:
                simplified_lines.append(clean_line)

        if not simplified_lines:
            continue

        rel_path = os.path.relpath(filepath, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(simplified_lines))
        print(f"[轉換完成] 實驗日誌已儲存至: {out_path}")


if __name__ == "__main__":
    # 取得專案根目錄 (假設此腳本位於 output 目錄下)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_in = os.path.join(base_dir, "output", "explanations")
    fid_in = os.path.join(base_dir, "output", "fidelity")
    log_in = os.path.join(base_dir, "output", "logs")
    out_dir = os.path.join(base_dir, "output", "simplified_for_llm")

    # Define specific subfolders in the simplified output
    out_exp_dir = os.path.join(out_dir, "explanations")
    out_fid_dir = os.path.join(out_dir, "fidelity")
    out_log_dir = os.path.join(out_dir, "logs")

    print("===" * 15)
    print("啟動數據簡化腳本 (LLM-Friendly Data Simplifier)")
    print("===" * 15)

    print("\n[1/2] 正在處理 Explanations (JSON)...")
    if os.path.exists(exp_in):
        simplify_explanations(exp_in, out_exp_dir)
    else:
        print(f"錯誤: 找不到目錄 {exp_in}")

    print("\n[2/3] 正在處理 Fidelity (JSON)...")
    if os.path.exists(fid_in):
        simplify_explanations(fid_in, out_fid_dir)
    else:
        print(f"提醒: 找不到目錄 {fid_in}，跳過處理。")

    print("\n[3/3] 正在處理 Logs (TXT)...")
    if os.path.exists(log_in):
        simplify_logs(log_in, out_log_dir)
    else:
        print(f"錯誤: 找不到目錄 {log_in}")

    print("\n執行完成！所有供 LLM 閱讀的簡化檔案都已存放至:")
    print(out_dir)
