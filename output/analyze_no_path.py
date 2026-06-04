"""分析無路徑用戶 vs 有路徑用戶的特徵差異.

比較項目：
1. 用戶互動數量
2. 推薦物品 KG 度數
3. 預測信心分數
4. 輸出統計表與 LaTeX 格式
"""

import argparse
import json
import statistics
from pathlib import Path


def analyze_no_path(input_path, output_path=None):
    """分析有路徑 vs 無路徑用戶的特徵差異."""
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"錯誤: 找不到檔案 {input_path}")
        return

    if output_path is None:
        output_file = input_file.parent / "no_path_analysis.txt"
    else:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    # 分組
    has_path = [d for d in data if d.get("explanations") and len(d["explanations"]) > 0]
    no_path = [d for d in data if not d.get("explanations") or len(d["explanations"]) == 0]
    total = len(data)

    results = []

    def log(msg):
        print(msg)
        results.append(msg)

    log("=" * 60)
    log("有路徑 vs 無路徑用戶特徵比較分析")
    log(f"分析檔案: {input_path}")
    log("=" * 60)

    log(f"\n總使用者數: {total}")
    log(f"有路徑用戶: {len(has_path)} ({len(has_path) / total * 100:.1f}%)")
    log(f"無路徑用戶: {len(no_path)} ({len(no_path) / total * 100:.1f}%)")

    # 特徵對比
    features = {
        "user_n_interactions": "用戶互動數",
        "item_kg_degree": "推薦物品 KG 度數",
        "original_prob": "預測信心分數",
    }

    comparison_data = {}

    for field, label in features.items():
        # 取得有路徑群組的特徵值
        path_vals = []
        for d in has_path:
            val = d.get(field)
            if val is not None:
                path_vals.append(val)

        # 取得無路徑群組的特徵值
        no_path_vals = []
        for d in no_path:
            val = d.get(field)
            if val is not None:
                no_path_vals.append(val)

        if not path_vals or not no_path_vals:
            log(f"\n--- {label} ---")
            log(f"  資料不足，跳過分析")
            continue

        log(f"\n--- {label} ---")
        log(f"  {'指標':<15} {'有路徑':>12} {'無路徑':>12} {'差異':>12}")
        log(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")

        mean_p = statistics.mean(path_vals)
        mean_n = statistics.mean(no_path_vals)
        diff = mean_p - mean_n
        log(f"  {'平均值':<15} {mean_p:>12.4f} {mean_n:>12.4f} {diff:>12.4f}")

        med_p = statistics.median(path_vals)
        med_n = statistics.median(no_path_vals)
        log(f"  {'中位數':<15} {med_p:>12.4f} {med_n:>12.4f} {med_p - med_n:>12.4f}")

        if len(path_vals) > 1 and len(no_path_vals) > 1:
            std_p = statistics.stdev(path_vals)
            std_n = statistics.stdev(no_path_vals)
            log(f"  {'標準差':<15} {std_p:>12.4f} {std_n:>12.4f}")

        log(f"  {'最小值':<15} {min(path_vals):>12.4f} {min(no_path_vals):>12.4f}")
        log(f"  {'最大值':<15} {max(path_vals):>12.4f} {max(no_path_vals):>12.4f}")
        log(f"  {'樣本數':<15} {len(path_vals):>12} {len(no_path_vals):>12}")

        comparison_data[field] = {
            "label": label,
            "path_mean": mean_p,
            "no_path_mean": mean_n,
            "path_median": med_p,
            "no_path_median": med_n,
            "path_n": len(path_vals),
            "no_path_n": len(no_path_vals),
        }

    # 互動數區間分布
    log("\n=== 用戶互動數區間分布 ===")
    bins = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, float("inf"))]
    bin_labels = ["0-4", "5-9", "10-19", "20-49", "50-99", "100+"]

    log(f"  {'區間':<10} {'有路徑':>10} {'無路徑':>10} {'有路徑%':>10} {'無路徑%':>10}")
    log(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for (lo, hi), label in zip(bins, bin_labels):
        p_count = sum(1 for d in has_path if lo <= d.get("user_n_interactions", 0) < hi)
        n_count = sum(1 for d in no_path if lo <= d.get("user_n_interactions", 0) < hi)
        p_pct = p_count / len(has_path) * 100 if has_path else 0
        n_pct = n_count / len(no_path) * 100 if no_path else 0
        log(f"  {label:<10} {p_count:>10} {n_count:>10} {p_pct:>9.1f}% {n_pct:>9.1f}%")

    # LaTeX 表格輸出
    log("\n=== LaTeX 無路徑用戶分析表 ===")
    log("\\begin{table}[htbp]")
    log("  \\centering")
    log("  \\caption{有路徑與無路徑用戶之特徵比較}")
    log("  \\label{tab:no_path_analysis}")
    log("  \\begin{tabular}{lccc}")
    log("    \\toprule")
    log("    特徵 & 有路徑用戶 & 無路徑用戶 & 差異 \\\\")
    log("    \\midrule")
    log(f"    樣本數 & {len(has_path)} & {len(no_path)} & -- \\\\")
    for field, info in comparison_data.items():
        diff = info["path_mean"] - info["no_path_mean"]
        log(f"    {info['label']} & {info['path_mean']:.2f} & {info['no_path_mean']:.2f} & {diff:+.2f} \\\\")
    log("    \\bottomrule")
    log("  \\end{tabular}")
    log("\\end{table}")

    # 保存
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    print(f"\n分析結果已保存至: {output_file}")


def compare_no_path_across_models(model_paths, model_names):
    """跨模型的無路徑分析比較."""
    print("\n" + "=" * 60)
    print("跨模型無路徑覆蓋率比較")
    print("=" * 60)

    header = f"{'模型':<15} {'總數':>8} {'有路徑':>8} {'無路徑':>8} {'覆蓋率':>10}"
    print(header)
    print("-" * len(header))

    for name, path in zip(model_names, model_paths):
        p = Path(path)
        if not p.exists():
            print(f"{name:<15} 檔案不存在")
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        total = len(data)
        has = sum(1 for d in data if d.get("explanations") and len(d["explanations"]) > 0)
        no = total - has
        cov = has / total * 100 if total > 0 else 0
        print(f"{name:<15} {total:>8} {has:>8} {no:>8} {cov:>9.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="無路徑用戶特徵分析")
    parser.add_argument(
        "--input", type=str,
        default="output/fidelity/depth_3/explanations.json",
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
    parser.add_argument("--output", type=str, default=None, help="輸出路徑")
    args = parser.parse_args()

    if args.compare:
        compare_no_path_across_models(args.models, args.names)
    else:
        analyze_no_path(args.input, args.output)
