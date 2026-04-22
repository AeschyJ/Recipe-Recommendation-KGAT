import glob
import os
import re

import numpy as np


def parse_log_file(file_path):
    epochs_data = []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "Evaluation -" in line:
            # Extract NDCG
            ndcg_match = re.search(r"NDCG@\[10,20,50\]: \[(.*?),(.*?),(.*?)\]", line)
            # Extract Recall/HR
            hr_match = re.search(
                r"(?:HR|Recall)@\[10,20,50\]: \[(.*?),(.*?),(.*?)\]", line
            )
            # Extract Precision
            prec_match = re.search(
                r"Precision@\[10,20,50\]: \[(.*?),(.*?),(.*?)\]", line
            )

            # 確保提取出正確的 Epoch 數字 (因為有些實驗可能有從途中resume或是被截斷)，預防對齊錯亂
            epoch_match = re.search(r"Epoch (\d+) Evaluation", line)
            if epoch_match:
                epoch_num = int(epoch_match.group(1))
            else:
                epoch_num = len(epochs_data) + 1

            if ndcg_match and hr_match and prec_match:
                try:
                    metrics = {
                        "hr_10": float(hr_match.group(1).strip()),
                        "hr_20": float(hr_match.group(2).strip()),
                        "hr_50": float(hr_match.group(3).strip()),
                        "prec_10": float(prec_match.group(1).strip()),
                        "prec_20": float(prec_match.group(2).strip()),
                        "prec_50": float(prec_match.group(3).strip()),
                        "ndcg_10": float(ndcg_match.group(1).strip()),
                        "ndcg_20": float(ndcg_match.group(2).strip()),
                        "ndcg_50": float(ndcg_match.group(3).strip()),
                        "epoch": epoch_num,
                    }
                    epochs_data.append(metrics)
                except ValueError:
                    continue

    return epochs_data


def format_metrics(data_list, metric_key):
    values = [d[metric_key] for d in data_list]
    avg = np.mean(values)
    std = np.std(values)
    return f"{avg:.4f} ± {std:.4f}"


def get_avg_metric(data_list, metric_key):
    values = [d[metric_key] for d in data_list]
    return np.mean(values)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(base_dir, "output/simplified_for_llm/logs")
    out_file = os.path.join(base_dir, "output/simplified_for_llm/epoch_metrics_report.md")
    results = {}

    for log_path in glob.glob(os.path.join(log_dir, "**/*.txt"), recursive=True):
        rel_path = os.path.relpath(log_path, log_dir)
        dir_name = os.path.dirname(rel_path)
        file_name = os.path.basename(rel_path)

        # Determine model name
        model_group = dir_name
        if model_group == "baseline":
            model_name = file_name.split("_")[0].upper()
        else:
            model_name = model_group.replace("_", "-").upper()

        epochs_data = parse_log_file(log_path)

        if not epochs_data:
            continue

        if model_name not in results:
            results[model_name] = {}

        # 以 Epoch 為單位進行分組
        for rd in epochs_data:
            ep = rd["epoch"]
            if ep not in results[model_name]:
                results[model_name][ep] = []
            results[model_name][ep].append(rd)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# 實驗結果總結 (依 Epoch 統計)\n")
        f.write(
            "所有日誌檔案的平均值與標準差，依各個 Epoch 分別計算。若只出現過 1 次訓練則標準差會標示為 0.0000。\n\n"
        )

        for model_name, ep_data in sorted(results.items()):
            if not ep_data:
                continue

            f.write(f"## {model_name}\n\n")

            best_epoch = -1
            best_hr20 = -1

            f.write(
                "| Epoch | 日誌數 | HR@20 (Avg ± Std) | NDCG@20 (Avg ± Std) | HR@10 (Avg ± Std) | NDCG@10 (Avg ± Std) |\n"
            )
            f.write(
                "|-------|--------|-------------------|---------------------|-------------------|---------------------|\n"
            )

            sorted_epochs = sorted(list(ep_data.keys()))
            for ep in sorted_epochs:
                runs = ep_data[ep]
                hr20_str = format_metrics(runs, "hr_20")
                ndcg20_str = format_metrics(runs, "ndcg_20")
                hr10_str = format_metrics(runs, "hr_10")
                ndcg10_str = format_metrics(runs, "ndcg_10")

                avg_hr20 = get_avg_metric(runs, "hr_20")
                if avg_hr20 > best_hr20:
                    best_hr20 = avg_hr20
                    best_epoch = ep

                f.write(
                    f"| {ep} | {len(runs)} | {hr20_str} | {ndcg20_str} | {hr10_str} | {ndcg10_str} |\n"
                )

            f.write(
                f"\n**最高得分的 Epoch (基於平均 HR@20 判斷) : Epoch {best_epoch}**\n\n"
            )
            f.write("---\n\n")


if __name__ == "__main__":
    main()
