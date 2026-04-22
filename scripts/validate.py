import os
import glob
import re
import numpy as np

def parse_log_file(file_path):
    epochs_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        if "Evaluation -" in line:
            ndcg_match = re.search(r'NDCG@\[10,20,50\]: \[(.*?),(.*?),(.*?)\]', line)
            hr_match = re.search(r'(?:HR|Recall)@\[10,20,50\]: \[(.*?),(.*?),(.*?)\]', line)
            prec_match = re.search(r'Precision@\[10,20,50\]: \[(.*?),(.*?),(.*?)\]', line)
            epoch_match = re.search(r'Epoch (\d+) Evaluation', line)
            epoch_num = int(epoch_match.group(1)) if epoch_match else len(epochs_data) + 1
            if ndcg_match and hr_match and prec_match:
                try:
                    metrics = {
                        'hr_10': float(hr_match.group(1).strip()),
                        'hr_20': float(hr_match.group(2).strip()),
                        'hr_50': float(hr_match.group(3).strip()),
                        'prec_10': float(prec_match.group(1).strip()),
                        'prec_20': float(prec_match.group(2).strip()),
                        'prec_50': float(prec_match.group(3).strip()),
                        'ndcg_10': float(ndcg_match.group(1).strip()),
                        'ndcg_20': float(ndcg_match.group(2).strip()),
                        'ndcg_50': float(ndcg_match.group(3).strip()),
                        'epoch': epoch_num
                    }
                    epochs_data.append(metrics)
                except ValueError:
                    pass
    return epochs_data

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(base_dir, "output/simplified_for_llm/logs")
    model_bests = {}
    
    report_path = os.path.join(base_dir, "scripts/epoch_logs_output.md")
    with open(report_path, "w", encoding='utf-8') as out:
        for log_path in sorted(glob.glob(os.path.join(log_dir, "**/*.txt"), recursive=True)):
            rel_path = os.path.relpath(log_path, log_dir)
            dir_name = os.path.dirname(rel_path)
            file_name = os.path.basename(rel_path)
            
            # Determine model name
            if dir_name == "baseline":
                model_name = file_name.split('_')[0].upper()
            else:
                model_name = dir_name.replace("_", "-").upper()
                
            epochs_data = parse_log_file(log_path)
            if not epochs_data: continue
            
            out.write(f"### {model_name} / {file_name}\n\n")
            out.write("| Epoch | HR@10 | HR@20 | HR@50 | Prec@10 | Prec@20 | Prec@50 | NDCG@10 | NDCG@20 | NDCG@50 |\n")
            out.write("|---|---|---|---|---|---|---|---|---|---|\n")
            for ep in epochs_data:
                out.write(f"| {ep['epoch']} | {ep['hr_10']:.4f} | {ep['hr_20']:.4f} | {ep['hr_50']:.4f} | {ep['prec_10']:.4f} | {ep['prec_20']:.4f} | {ep['prec_50']:.4f} | {ep['ndcg_10']:.4f} | {ep['ndcg_20']:.4f} | {ep['ndcg_50']:.4f} |\n")
            out.write("\n")
            
            # Find best epoch by hr_20
            best_run = max(epochs_data, key=lambda x: x['hr_20'])
            
            if model_name not in model_bests:
                model_bests[model_name] = []
            model_bests[model_name].append({
                'file': file_name,
                'best_epoch': best_run['epoch'],
                'metrics': best_run
            })

    for model_name, runs in model_bests.items():
        print(f"=== {model_name} ===")
        for r in runs:
            m = r['metrics']
            print(f"{r['file']} - Best Epoch: {r['best_epoch']}, HR@20: {m['hr_20']:.4f}, NDCG@20: {m['ndcg_20']:.4f}, Prec@20: {m['prec_20']:.4f}, HR@10: {m['hr_10']:.4f}, NDCG@10: {m['ndcg_10']:.4f}, Prec@10: {m['prec_10']:.4f}")
            print(f"   HR@50: {m['hr_50']:.4f}, NDCG@50: {m['ndcg_50']:.4f}, Prec@50: {m['prec_50']:.4f}")
        
        if len(runs) > 0:
            avg_metrics = {}
            for k in runs[0]['metrics'].keys():
                if k == 'epoch': continue
                avg_metrics[k] = np.mean([r['metrics'][k] for r in runs])
            
            print(f"AVERAGE (n={len(runs)}):")
            print(f"HR@10: {avg_metrics['hr_10']:.4f}, HR@20: {avg_metrics['hr_20']:.4f}, HR@50: {avg_metrics['hr_50']:.4f}")
            print(f"Prec@10: {avg_metrics['prec_10']:.4f}, Prec@20: {avg_metrics['prec_20']:.4f}, Prec@50: {avg_metrics['prec_50']:.4f}")
            print(f"NDCG@10: {avg_metrics['ndcg_10']:.4f}, NDCG@20: {avg_metrics['ndcg_20']:.4f}, NDCG@50: {avg_metrics['ndcg_50']:.4f}")
        print()

if __name__ == "__main__":
    main()
