import re
import os
import argparse
import glob

def parse_footer_metrics(lines):
    """
    掃描全檔案，找出所有 'Evaluation Results' 區塊，並建立 Checkpoint -> Metrics 的映射。
    """
    ckpt_metrics = {}
    current_ckpt = None
    
    for i, line in enumerate(lines):
        # 尋找 Loading checkpoint
        ckpt_match = re.search(r"Loading checkpoint:\s*(.*)", line)
        if ckpt_match:
            current_ckpt = ckpt_match.group(1).strip().replace("\\", "/")
            continue
            
        # 尋找 Evaluation Results
        if "Evaluation Results:" in line and current_ckpt:
            metrics = {10: {}, 20: {}, 50: {}}
            # 往後看幾行找 K=10, 20, 50
            for j in range(1, 4):
                if i + j >= len(lines): break
                m_line = lines[i+j]
                # 匹配 K=10 -> HR: 0.5852, Precision: 0.0585, NDCG: 0.3819
                m_match = re.search(r"K=(\d+)\s*->\s*HR:\s*([\d.None]+),\s*Precision:\s*([\d.None]+),\s*NDCG:\s*([\d.None]+)", m_line)
                if m_match:
                    k = int(m_match.group(1))
                    metrics[k] = {
                        'hr': m_match.group(2),
                        'prec': m_match.group(3),
                        'ndcg': m_match.group(4)
                    }
            if metrics[10]: # 至少要有資料
                ckpt_metrics[current_ckpt] = metrics
                
    return ckpt_metrics

def reformat_content(lines):
    """
    根據尾部評估結果更新上半部的指標，且對所有 Eval 行補足 Precision 與 NDCG 欄位。
    """
    ckpt_metrics = parse_footer_metrics(lines)
    processed_lines = []
    
    last_eval_idx = -1
    
    for i, line in enumerate(lines):
        # --- 第一階段：初步格式化 (確保每行都有 HR, Precision, NDCG) ---
        if "Evaluation - " in line:
            # 狀況 A: 舊格式 Recall@10: ..., Recall@20: ..., Recall@50: ...
            recall_match = re.search(r"Recall@10:\s*([\d.]+),\s*Recall@20:\s*([\d.]+),\s*Recall@50:\s*([\d.]+)", line)
            if recall_match:
                v = recall_match.groups()
                hr_list = f"[{v[0]}, {v[1]}, {v[2]}]"
                prefix_match = re.match(r"(.*\[INFO\]\s*)", line)
                prefix = prefix_match.group(1) if prefix_match else ""
                epoch_match = re.search(r"Epoch \d+", line)
                epoch_info = epoch_match.group(0) + " Evaluation - " if epoch_match else ""
                line = f"{prefix}{epoch_info}HR@[10,20,50]: {hr_list} | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]\n"
            
            # 狀況 B: 已有 HR 和 NDCG 但缺少 Precision (例如: Epoch 1 Eval - HR@... | NDCG@...)
            elif "HR@[" in line and "NDCG@[" in line and "| Precision@" not in line:
                line = line.replace(" | NDCG@", " | Precision@[10,20,50]: [None, None, None] | NDCG@")

        # 暫存處理過的行
        processed_lines.append(line)
        
        # 紀錄最後一個看到的 Eval 行索引
        if " Evaluation - " in line:
            last_eval_idx = len(processed_lines) - 1
            
        # --- 第二階段：精準回填 (用末尾測出的數值複寫) ---
        save_match = re.search(r"(?:Saved checkpoint:|Checkpoint saved to)\s*(.*)", line)
        if save_match and last_eval_idx != -1:
            ckpt_path = save_match.group(1).strip().replace("\\", "/")
            
            if ckpt_path in ckpt_metrics:
                m = ckpt_metrics[ckpt_path]
                hr_list = f"[{m[10]['hr']}, {m[20]['hr']}, {m[50]['hr']}]"
                prec_list = f"[{m[10]['prec']}, {m[20]['prec']}, {m[50]['prec']}]"
                ndcg_list = f"[{m[10]['ndcg']}, {m[20]['ndcg']}, {m[50]['ndcg']}]"
                
                prefix_match = re.match(r"(.*\[INFO\]\s*)", processed_lines[last_eval_idx])
                prefix = prefix_match.group(1) if prefix_match else ""
                epoch_match = re.search(r"Epoch \d+", processed_lines[last_eval_idx])
                epoch_info = epoch_match.group(0) + " Evaluation - " if epoch_match else ""
                
                formatted_line = f"{prefix}{epoch_info}HR@[10,20,50]: {hr_list} | Precision@[10,20,50]: {prec_list} | NDCG@[10,20,50]: {ndcg_list}\n"
                processed_lines[last_eval_idx] = formatted_line
                last_eval_idx = -1 # 用過了重置
                    
    return processed_lines

def main():
    parser = argparse.ArgumentParser(description="自動將 Log 各階段指標補齊，並將尾部數值更新到訓練紀錄中。")
    parser.add_argument("files", nargs="*", help="Log 檔案路徑 (可選，若未指定則自動掃描 output/logs 內的所有 txt 檔)")
    parser.add_argument("-n", "--new", action="store_true", help="是否產生新檔案")
    args = parser.parse_args()

    files_to_process = args.files
    if not files_to_process:
        files_to_process = glob.glob("output/logs/**/*.txt", recursive=True)
        files_to_process = [f for f in files_to_process if "_reformatted" not in f]
        print(f"未手動指定檔案，自動抓取了 {len(files_to_process)} 個 Log 檔案。")
        
    if not files_to_process:
        print("沒有找到任何需要處理的 Log 檔案。")
        return

    for file_path in files_to_process:
        if not os.path.exists(file_path):
            print(f"跳過: 找不到檔案 {file_path}")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        new_lines = reformat_content(lines)
        
        output_path = file_path
        if args.new:
            base, ext = os.path.splitext(file_path)
            output_path = f"{base}_reformatted{ext}"
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
            
        print(f"成功更新指標: {os.path.basename(file_path)}")

if __name__ == "__main__":
    main()
