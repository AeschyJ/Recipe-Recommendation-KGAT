"""分析模型的 XAI Fidelity 數據."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def analyze_fidelity(input_path):
    # 確保路徑存在
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"錯誤: 找不到檔案 {input_path}")
        return

    # 決定輸出路徑: output/fidelity/來源資料夾/analysis.txt
    source_folder = input_file.parent.name
    output_dir = Path("output/simplified_for_llm/fidelity") / source_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "analysis.txt"

    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    has_explanation = [
        d for d in data if d.get("explanations") and len(d["explanations"]) > 0
    ]
    no_explanation = [
        d for d in data if not d.get("explanations") or len(d["explanations"]) == 0
    ]
    has_fidelity = [d for d in data if "fidelity" in d]

    results = []

    def log(msg):
        print(msg)
        results.append(msg)

    log("=== 基本統計 ===")
    log(f"分析檔案: {input_path}")
    log(f"總使用者數: {total}")
    log(
        f"有解釋路徑: {len(has_explanation)} ({len(has_explanation) / total * 100:.1f}%)"
    )
    log(f"無解釋路徑: {len(no_explanation)} ({len(no_explanation) / total * 100:.1f}%)")
    log(
        f"有 Fidelity 數據: {len(has_fidelity)} ({len(has_fidelity) / total * 100:.1f}%)"
    )

    if not has_fidelity:
        log("\n無 Fidelity 數據可分析。")
    else:
        # Fidelity 統計
        f_plus = [d["fidelity"]["plus"] for d in has_fidelity]
        f_minus = [d["fidelity"]["minus"] for d in has_fidelity]

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
        fm_high = sum(1 for x in f_minus if x > 0.10)
        log(f"F- > 0.10: {fm_high} ({fm_high / len(f_minus) * 100:.1f}%)")

    # 路徑類型分析
    direct = 0
    cf_path = 0
    kg_path = 0
    total_paths = 0

    # Attention Score 分析
    all_scores = []
    direct_scores = []
    cf_scores = []
    kg_scores = []

    for d in has_explanation:
        for exp in d["explanations"]:
            path = exp["path"]
            score = exp["score"]
            hops = path.split(" -> ")
            total_paths += 1
            all_scores.append(score)

            if len(hops) == 2:
                direct += 1
                direct_scores.append(score)
            elif len(hops) == 4:
                mid = hops[2]
                if mid.startswith("User"):
                    cf_path += 1
                    cf_scores.append(score)
                else:
                    kg_path += 1
                    kg_scores.append(score)

    if total_paths > 0:
        log("\n=== 路徑類型分析 ===")
        log(f"總路徑數: {total_paths}")
        log(f"直接連接 (User->Recipe): {direct} ({direct / total_paths * 100:.1f}%)")
        log(f"協同過濾 (U->R->U->R): {cf_path} ({cf_path / total_paths * 100:.1f}%)")
        log(f"KG 語意 (U->R->E->R): {kg_path} ({kg_path / total_paths * 100:.1f}%)")

        log("\n=== Attention Score 分析 ===")
        log(f"全部路徑 Avg: {statistics.mean(all_scores):.4f}")
        if direct_scores:
            log(
                f"直接連接 Avg: {statistics.mean(direct_scores):.4f} (n={len(direct_scores)})"
            )
        if cf_scores:
            log(f"協同過濾 Avg: {statistics.mean(cf_scores):.4f} (n={len(cf_scores)})")
        if kg_scores:
            log(f"KG 語意 Avg: {statistics.mean(kg_scores):.4f} (n={len(kg_scores)})")

    # 預測機率分布
    probs = [d["prob"] for d in data]
    log("\n=== 預測機率分布 ===")
    log(f"Mean: {statistics.mean(probs):.4f}")
    log(f"Min: {min(probs):.4f}")
    log(f"Max: {max(probs):.4f}")

    # 按路徑類型分析 Fidelity
    if has_fidelity:
        direct_fplus, cf_fplus, kg_fplus = [], [], []
        direct_fminus, cf_fminus, kg_fminus = [], [], []

        for d in has_fidelity:
            if not d.get("explanations"):
                continue
            paths = d["explanations"]
            first_path = paths[0]["path"]
            hops = first_path.split(" -> ")
            fp = d["fidelity"]["plus"]
            fm = d["fidelity"]["minus"]
            if len(hops) == 2:
                direct_fplus.append(fp)
                direct_fminus.append(fm)
            elif len(hops) == 4:
                mid = hops[2]
                if mid.startswith("User"):
                    cf_fplus.append(fp)
                    cf_fminus.append(fm)
                else:
                    kg_fplus.append(fp)
                    kg_fminus.append(fm)

        log("\n=== Fidelity+ 按路徑類型 ===")
        if direct_fplus:
            log(
                f"直接連接: Mean F+={statistics.mean(direct_fplus):.4f}, n={len(direct_fplus)}"
            )
        if cf_fplus:
            log(f"協同過濾: Mean F+={statistics.mean(cf_fplus):.4f}, n={len(cf_fplus)}")
        if kg_fplus:
            log(f"KG 語意: Mean F+={statistics.mean(kg_fplus):.4f}, n={len(kg_fplus)}")

        log("\n=== Fidelity- 按路徑類型 ===")
        if direct_fminus:
            log(
                f"直接連接: Mean F-={statistics.mean(direct_fminus):.4f}, n={len(direct_fminus)}"
            )
        if cf_fminus:
            log(
                f"協同過濾: Mean F-={statistics.mean(cf_fminus):.4f}, n={len(cf_fminus)}"
            )
        if kg_fminus:
            log(
                f"KG 語意: Mean F-={statistics.mean(kg_fminus):.4f}, n={len(kg_fminus)}"
            )

        # Top F+ users
        log("\n=== Top 10 Fidelity+ 最高 ===")
        sorted_by_fplus = sorted(
            has_fidelity, key=lambda x: x["fidelity"]["plus"], reverse=True
        )
        for d in sorted_by_fplus[:10]:
            p_type = "direct"
            if d.get("explanations"):
                h = d["explanations"][0]["path"].split(" -> ")
                if len(h) == 4:
                    p_type = "CF" if h[2].startswith("User") else "KG"
            user = d["user"]
            fp = d["fidelity"]["plus"]
            prob = d["prob"]
            item = d["item"][:40]
            log(f"  {user}: F+={fp:.4f}, prob={prob:.4f}, type={p_type}, item={item}")

        # Bottom F+ (negative)
        neg_fplus = [d for d in has_fidelity if d["fidelity"]["plus"] < 0]
        log(f"\n=== F+ 為負數的用戶: {len(neg_fplus)} ===")
        for d in neg_fplus:
            user = d["user"]
            fp = d["fidelity"]["plus"]
            fm = d["fidelity"]["minus"]
            log(f"  {user}: F+={fp:.4f}, F-={fm:.4f}")

    # KG entities 統計
    kg_entities = []
    for d in has_explanation:
        for exp in d["explanations"]:
            hops = exp["path"].split(" -> ")
            if len(hops) == 4 and not hops[2].startswith("User"):
                kg_entities.append(hops[2])

    if kg_entities:
        log("\n=== KG 中介實體分布 (Top 15) ===")
        for entity, count in Counter(kg_entities).most_common(15):
            log(f"  {entity}: {count}")

    # 保存至檔案
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    print(f"\n分析結果已保存至: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析 XAI Fidelity 資料夾")
    parser.add_argument(
        "--input",
        type=str,
        default=r"output/simplified_for_llm/fidelity/depth_3/explanations.json",
        help="輸入的 explanations.json 路徑",
    )
    args = parser.parse_args()

    analyze_fidelity(args.input)
