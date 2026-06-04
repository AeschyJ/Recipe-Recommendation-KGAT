"""分析模型的 XAI Fidelity 數據 - 支援原始與簡化格式.

支援三種來源格式：
1. 原始格式 (evaluate_fidelity.py 直接輸出)
2. 簡化格式 (simplify_output_data.py 處理後)
3. 跨模型比較模式 (多個資料夾)
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def normalize_record(d):
    """將不同格式的紀錄正規化為統一欄位."""
    # 偵測格式
    if "fidelity" in d and isinstance(d["fidelity"], dict):
        # 原始格式或簡化格式
        fid = d["fidelity"]
        if "fidelity_plus" in fid:
            # 原始格式
            f_plus = fid.get("fidelity_plus")
            f_minus = fid.get("fidelity_minus")
        elif "plus" in fid:
            # 簡化格式
            f_plus = fid.get("plus")
            f_minus = fid.get("minus")
        else:
            f_plus = None
            f_minus = None
    else:
        f_plus = None
        f_minus = None

    # 機率
    prob = d.get("original_prob") or d.get("prob") or 0

    # 使用者名稱
    user = d.get("user_name") or d.get("user") or f"User_{d.get('user_id_remapped', '?')}"

    # 物品名稱
    item = d.get("recommended_item_name") or d.get("item") or ""

    # 解釋路徑正規化
    explanations = d.get("explanations", [])
    paths = []
    for exp in explanations:
        # 原始格式
        if "path_description" in exp:
            path_str = exp["path_description"]
            score = exp.get("contribution_score", 0)
            structure = exp.get("path_structure", "")
        # 簡化格式
        elif "path" in exp:
            path_str = exp["path"]
            score = exp.get("score", 0)
            structure = ""
        else:
            continue
        paths.append({"path": path_str, "score": score, "structure": structure})

    return {
        "user": user,
        "item": item,
        "prob": prob,
        "fidelity_plus": f_plus,
        "fidelity_minus": f_minus,
        "paths": paths,
        "has_path": len(paths) > 0,
        "user_n_interactions": d.get("user_n_interactions", None),
        "item_kg_degree": d.get("item_kg_degree", None),
    }


def classify_path(path_str):
    """分類路徑類型."""
    hops = path_str.split(" -> ")
    if len(hops) == 2:
        return "direct"
    elif len(hops) == 4:
        mid = hops[2]
        if mid.startswith("User"):
            return "cf"
        else:
            return "kg"
    elif len(hops) == 3:
        return "2hop"
    else:
        return "other"


def analyze_fidelity(input_path, output_path=None):
    """分析單個模型的 Fidelity 數據."""
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"錯誤: 找不到檔案 {input_path}")
        return None

    if output_path is None:
        source_folder = input_file.parent.name
        output_dir = input_file.parent
        output_file = output_dir / "analysis.txt"
    else:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    records = [normalize_record(d) for d in data]
    total = len(records)
    has_path = [r for r in records if r["has_path"]]
    no_path = [r for r in records if not r["has_path"]]
    has_fidelity = [r for r in records if r["fidelity_plus"] is not None]

    results = []

    def log(msg):
        print(msg)
        results.append(msg)

    log("=== 基本統計 ===")
    log(f"分析檔案: {input_path}")
    log(f"總使用者數: {total}")
    log(f"有解釋路徑: {len(has_path)} ({len(has_path) / total * 100:.1f}%)")
    log(f"無解釋路徑: {len(no_path)} ({len(no_path) / total * 100:.1f}%)")
    log(f"有 Fidelity 數據: {len(has_fidelity)} ({len(has_fidelity) / total * 100:.1f}%)")

    if has_fidelity:
        f_plus = [r["fidelity_plus"] for r in has_fidelity]
        f_minus = [r["fidelity_minus"] for r in has_fidelity]

        log("\n=== Fidelity+ 統計 ===")
        log(f"Mean: {statistics.mean(f_plus):.4f}")
        log(f"Median: {statistics.median(f_plus):.4f}")
        if len(f_plus) > 1:
            log(f"Std: {statistics.stdev(f_plus):.4f}")
        log(f"Min: {min(f_plus):.4f}")
        log(f"Max: {max(f_plus):.4f}")
        fplus_pos = sum(1 for x in f_plus if x > 0)
        log(f"F+ > 0: {fplus_pos} ({fplus_pos / len(f_plus) * 100:.1f}%)")
        fplus_05 = sum(1 for x in f_plus if x > 0.05)
        log(f"F+ > 0.05: {fplus_05} ({fplus_05 / len(f_plus) * 100:.1f}%)")
        fplus_10 = sum(1 for x in f_plus if x > 0.10)
        log(f"F+ > 0.10: {fplus_10} ({fplus_10 / len(f_plus) * 100:.1f}%)")

        log("\n=== Fidelity- 統計 ===")
        log(f"Mean: {statistics.mean(f_minus):.4f}")
        log(f"Median: {statistics.median(f_minus):.4f}")
        if len(f_minus) > 1:
            log(f"Std: {statistics.stdev(f_minus):.4f}")
        log(f"Min: {min(f_minus):.4f}")
        log(f"Max: {max(f_minus):.4f}")
        fm_low = sum(1 for x in f_minus if abs(x) <= 0.01)
        log(f"abs(F-) <= 0.01: {fm_low} ({fm_low / len(f_minus) * 100:.1f}%)")

    # 路徑類型分析
    path_types = Counter()
    all_scores = []
    scores_by_type = {"direct": [], "cf": [], "kg": [], "2hop": [], "other": []}

    for r in has_path:
        for p in r["paths"]:
            ptype = classify_path(p["path"])
            path_types[ptype] += 1
            all_scores.append(p["score"])
            scores_by_type[ptype].append(p["score"])

    total_paths = sum(path_types.values())
    if total_paths > 0:
        log("\n=== 路徑類型分析 ===")
        log(f"總路徑數: {total_paths}")
        type_names = {
            "direct": "直接連接 (User->Recipe)",
            "cf": "協同過濾 (U->R->U->R)",
            "kg": "KG 語意 (U->R->E->R)",
            "2hop": "2-hop 路徑",
            "other": "其他"
        }
        for ptype, count in path_types.most_common():
            name = type_names.get(ptype, ptype)
            log(f"  {name}: {count} ({count / total_paths * 100:.1f}%)")

        log("\n=== Attention Score 按路徑類型 ===")
        log(f"全部路徑 Avg: {statistics.mean(all_scores):.4f}")
        for ptype, scores in scores_by_type.items():
            if scores:
                name = type_names.get(ptype, ptype)
                log(f"  {name}: Avg={statistics.mean(scores):.4f} (n={len(scores)})")

    # 按路徑類型分析 Fidelity
    if has_fidelity:
        fid_by_type = {"direct": {"fp": [], "fm": []}, "cf": {"fp": [], "fm": []}, "kg": {"fp": [], "fm": []}, "2hop": {"fp": [], "fm": []}}

        for r in has_fidelity:
            if not r["paths"]:
                continue
            first_path = r["paths"][0]
            ptype = classify_path(first_path["path"])
            if ptype in fid_by_type:
                fid_by_type[ptype]["fp"].append(r["fidelity_plus"])
                fid_by_type[ptype]["fm"].append(r["fidelity_minus"])

        log("\n=== Fidelity 按路徑類型 ===")
        type_names_short = {"direct": "直接連接", "cf": "協同過濾", "kg": "KG 語意", "2hop": "2-hop"}
        for ptype, vals in fid_by_type.items():
            if vals["fp"]:
                name = type_names_short.get(ptype, ptype)
                log(f"  {name}: F+={statistics.mean(vals['fp']):.4f}, F-={statistics.mean(vals['fm']):.4f}, n={len(vals['fp'])}")

        # Negative F+ users
        neg_fplus = [r for r in has_fidelity if r["fidelity_plus"] < 0]
        log(f"\n=== F+ 為負數的用戶: {len(neg_fplus)} ({len(neg_fplus) / len(has_fidelity) * 100:.1f}%) ===")

    # 預測機率分布
    probs = [r["prob"] for r in records if r["prob"] > 0]
    if probs:
        log("\n=== 預測機率分布 ===")
        log(f"Mean: {statistics.mean(probs):.4f}")
        log(f"Min: {min(probs):.4f}")
        log(f"Max: {max(probs):.4f}")

    # KG 中介實體統計
    kg_entities = []
    for r in has_path:
        for p in r["paths"]:
            hops = p["path"].split(" -> ")
            if len(hops) == 4 and not hops[2].startswith("User"):
                kg_entities.append(hops[2])

    if kg_entities:
        log("\n=== KG 中介實體分布 (Top 15) ===")
        for entity, count in Counter(kg_entities).most_common(15):
            log(f"  {entity}: {count}")

    # 保存
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    print(f"\n分析結果已保存至: {output_file}")

    # 返回摘要數據（供跨模型比較使用）
    summary = {
        "total_users": total,
        "has_path": len(has_path),
        "no_path": len(no_path),
        "coverage": len(has_path) / total if total > 0 else 0,
        "total_paths": total_paths,
        "path_types": dict(path_types),
    }
    if has_fidelity:
        summary["avg_fidelity_plus"] = statistics.mean(f_plus)
        summary["avg_fidelity_minus"] = statistics.mean(f_minus)
        summary["std_fidelity_plus"] = statistics.stdev(f_plus) if len(f_plus) > 1 else 0
        summary["std_fidelity_minus"] = statistics.stdev(f_minus) if len(f_minus) > 1 else 0
        summary["n_fidelity"] = len(has_fidelity)

    return summary


def compare_models(model_paths, model_names, output_path=None):
    """跨模型比較分析."""
    summaries = {}
    for name, path in zip(model_names, model_paths):
        print(f"\n{'='*50}")
        print(f"分析模型: {name}")
        print(f"{'='*50}")
        summary = analyze_fidelity(path)
        if summary:
            summaries[name] = summary

    if not summaries:
        print("無可比較的數據。")
        return

    print(f"\n{'='*60}")
    print("跨模型比較總結")
    print(f"{'='*60}")

    # 比較表
    header = f"{'模型':<15} {'覆蓋率':>8} {'路徑數':>8} {'F+':>10} {'F-':>10} {'n':>6}"
    print(header)
    print("-" * len(header))

    latex_rows = []
    for name, s in summaries.items():
        coverage = f"{s['coverage']*100:.1f}%"
        paths = str(s['total_paths'])
        fp = f"{s.get('avg_fidelity_plus', 0):.4f}" if 'avg_fidelity_plus' in s else "N/A"
        fm = f"{s.get('avg_fidelity_minus', 0):.4f}" if 'avg_fidelity_minus' in s else "N/A"
        n = str(s.get('n_fidelity', 0))
        print(f"{name:<15} {coverage:>8} {paths:>8} {fp:>10} {fm:>10} {n:>6}")

        # LaTeX 行
        fp_std = f"{s.get('std_fidelity_plus', 0):.4f}" if 'std_fidelity_plus' in s else ""
        fm_std = f"{s.get('std_fidelity_minus', 0):.4f}" if 'std_fidelity_minus' in s else ""
        latex_rows.append(
            f"    {name} & {s['total_users']} & {s['has_path']} & {coverage} & "
            f"${fp} \\pm {fp_std}$ & ${fm} \\pm {fm_std}$ \\\\\\\\"
        )

    # 路徑類型比較
    print(f"\n路徑類型分布比較:")
    type_names = {"direct": "直接", "cf": "協同", "kg": "KG", "2hop": "2hop", "other": "其他"}
    all_types = set()
    for s in summaries.values():
        all_types.update(s.get("path_types", {}).keys())

    header2 = f"{'模型':<15}" + "".join(f"{type_names.get(t, t):>10}" for t in sorted(all_types))
    print(header2)
    print("-" * len(header2))
    for name, s in summaries.items():
        pt = s.get("path_types", {})
        vals = "".join(f"{pt.get(t, 0):>10}" for t in sorted(all_types))
        print(f"{name:<15}{vals}")

    # LaTeX 表格輸出
    print(f"\n=== LaTeX 跨深度比較表 ===")
    print("\\begin{table}[htbp]")
    print("  \\centering")
    print("  \\caption{不同注意力層數之可解釋性比較}")
    print("  \\label{tab:xai_depth_comparison}")
    print("  \\begin{tabular}{lccccc}")
    print("    \\toprule")
    print("    模型 & 樣本數 & 有路徑 & 覆蓋率 & Fidelity+ & Fidelity- \\\\")
    print("    \\midrule")
    for row in latex_rows:
        print(row)
    print("    \\bottomrule")
    print("  \\end{tabular}")
    print("\\end{table}")

    # 保存比較結果
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        print(f"\n比較結果已保存至: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析 XAI Fidelity 數據（支援原始/簡化格式、跨模型比較）")
    parser.add_argument(
        "--input", type=str,
        default=r"output/fidelity/depth_3/explanations.json",
        help="輸入的 explanations.json 路徑",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="跨模型比較模式"
    )
    parser.add_argument(
        "--models", nargs="+",
        default=[
            "output/fidelity/full_kgat/explanations.json",
            "output/fidelity/depth_2/explanations.json",
            "output/fidelity/depth_3/explanations.json",
        ],
        help="比較模式下的模型檔案路徑"
    )
    parser.add_argument(
        "--names", nargs="+",
        default=["KGAT-1L", "KGAT-2L", "KGAT-3L"],
        help="模型名稱"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="分析結果輸出路徑"
    )
    args = parser.parse_args()

    if args.compare:
        compare_models(args.models, args.names, args.output)
    else:
        analyze_fidelity(args.input, args.output)
