import os
import re
import glob

def parse_metrics(file_path):
    """
    從 Log 檔案中解析各 Epoch 的評估指標。
    格式支援: Epoch X Evaluation - HR@[...] | Precision@[...] | NDCG@[...]
    """
    metrics_by_epoch = {}
    # 正則表達式匹配指標行 (考慮到 [10,20,50]: [v1, v2, v3] 這種格式)
    pattern = re.compile(r"Epoch (\d+) Evaluation - HR@\[10,20,50\]: \[(.*?)\] \| Precision@\[10,20,50\]: \[(.*?)\] \| NDCG@\[10,20,50\]: \[(.*?)\]")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    epoch = int(match.group(1))
                    
                    # 處理可能存在的 None 或字串數值
                    def clean_list(s):
                        return [float(x) if (x.strip() != 'None' and x.strip() != '') else 0.0 for x in s.split(',')]
                    
                    hr = clean_list(match.group(2))
                    prec = clean_list(match.group(3))
                    ndcg = clean_list(match.group(4))
                    
                    metrics_by_epoch[epoch] = {'hr': hr, 'prec': prec, 'ndcg': ndcg}
    except Exception as e:
        print(f"解析 {file_path} 時出錯: {e}")
        
    return metrics_by_epoch

def format_diff(old, new):
    """
    格式化差異: 新值 (差異, 差異百分比)
    """
    diff = new - old
    if abs(old) < 1e-9:
        pct = 0.0
    else:
        pct = (diff / old) * 100
        
    sign = "+" if diff > 0 else ""
    return f"{new:.4f} ({sign}{diff:.4f}, {sign}{pct:.2f}%)"

def main():
    # 尋找所有重構過的檔案 (帶有 _reformatted.txt 後綴)
    reformatted_files = glob.glob("output/logs/depth_3/*_reformatted.txt", recursive=True)
    
    if not reformatted_files:
        # 如果沒找到，嘗試找原始 .txt 但假設它們已經被原地覆寫 (這比較難對比)
        # 這裡我們只針對使用 -n 產生新檔案的情況做對比
        print("未找到任何 *_reformatted.txt 檔案。請確保執行 reformat_logs.py 時有加上 -n 參數。")
        return

    report_content = "# 實驗數據重構對比報告 (Reformat Comparison Report)\n\n"
    report_content += "> 本報告對比了訓練中實時評估 (BF16 雜訊) 與 離線重新評估 (FP32 精準) 之間的差異。\n\n"
    
    for ref_path in sorted(reformatted_files):
        orig_path = ref_path.replace("_reformatted.txt", ".txt")
        if not os.path.exists(orig_path):
            continue
            
        filename = os.path.basename(orig_path)
        report_content += f"## 檔案: `{filename}`\n\n"
        
        orig_metrics = parse_metrics(orig_path)
        ref_metrics = parse_metrics(ref_path)
        
        # 取得交集的 Epoch
        epochs = sorted(list(set(orig_metrics.keys()) & set(ref_metrics.keys())))
        
        if not epochs:
            report_content += "未找到可對比的 Epoch 數據。\n\n"
            continue
            
        report_content += "| Epoch | 指標 | @10 (重構後 [差異, %]) | @20 | @50 |\n"
        report_content += "|:---:|:---:|:---|:---|:---|\n"
        
        for epoch in epochs:
            o = orig_metrics[epoch]
            r = ref_metrics[epoch]
            
            # HR Row
            hr_row = f"| {epoch} | **HR** | "
            for i in range(len(r['hr'])):
                old_val = o['hr'][i] if i < len(o['hr']) else 0.0
                hr_row += f"`{format_diff(old_val, r['hr'][i])}` | "
            report_content += hr_row + "\n"
            
            # Precision Row
            prec_row = f"| | Prec | "
            for i in range(len(r['prec'])):
                old_val = o['prec'][i] if i < len(o['prec']) else 0.0
                prec_row += f"`{format_diff(old_val, r['prec'][i])}` | "
            report_content += prec_row + "\n"
            
            # NDCG Row
            ndcg_row = f"| | **NDCG** | "
            for i in range(len(r['ndcg'])):
                old_val = o['ndcg'][i] if i < len(o['ndcg']) else 0.0
                ndcg_row += f"`{format_diff(old_val, r['ndcg'][i])}` | "
            report_content += ndcg_row + "\n"
            
            # 分隔線
            report_content += "| " + "--- | " * 5 + "\n"
            
        report_content += "\n"
        
    output_file = "scripts/depth_3_metric_comparison_report.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"成功生成對比報告: {output_file}")

if __name__ == "__main__":
    main()
