import glob
import os
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.log_utils import extract_info, parse_logs_for_best_epochs

MODELS_DIR = "models"
LOGS_DIR = "output/logs"


def analyze():
    print("Parsing logs to find log-based best epochs...")
    log_best_epochs = parse_logs_for_best_epochs(logs_dir=LOGS_DIR, models_dir=MODELS_DIR)
    print(f"Parsed {len(log_best_epochs)} runs from logs.")
    
    subdirs = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
    
    total_original_size = 0
    total_saved_size = 0
    
    report = []
    
    for subdir in subdirs:
        dir_path = os.path.join(MODELS_DIR, subdir)
        pth_files = glob.glob(os.path.join(dir_path, "*.pth"))
        if not pth_files:
            continue
            
        grouped = {}
        for fpath in pth_files:
            filename = os.path.basename(fpath)
            size = os.path.getsize(fpath)
            total_original_size += size
            
            info = extract_info(filename)
            if info:
                run_id, epoch = info
                if run_id not in grouped:
                    grouped[run_id] = []
                grouped[run_id].append((epoch, fpath, size))
        
        subdir_report = {
            "name": subdir,
            "runs": []
        }
        
        for run_id, files in sorted(grouped.items()):
            files.sort(key=lambda x: x[0], reverse=True)
            max_epoch, max_fpath, max_size = files[0]
            
            # Find best epoch from logs
            log_key = (subdir, run_id)
            best_epoch = None
            source = "log"
            
            if log_key in log_best_epochs:
                best_epoch = log_best_epochs[log_key][0]
            else:
                # Fallback to checkpoint file info
                try:
                    checkpoint = torch.load(max_fpath, map_location="cpu", weights_only=False)
                    best_epoch = checkpoint.get("best_epoch")
                    source = "checkpoint dict"
                except Exception as e:
                    print(f"Error loading {max_fpath}: {e}")
            
            if best_epoch is None:
                best_epoch = max_epoch
                source = "fallback last epoch"
                
            best_filename = f"{run_id}_kgat_checkpoint_e{best_epoch}.pth"
            best_fpath = os.path.join(dir_path, best_filename)
            
            to_keep = set()
            
            # Keep Last Epoch (max_fpath)
            to_keep.add(max_fpath)
            
            # Keep Best Epoch (best_fpath) if it exists
            if os.path.exists(best_fpath):
                to_keep.add(best_fpath)
            else:
                # If best epoch pth doesn't exist, we fallback
                best_filename = os.path.basename(max_fpath)
                best_epoch = max_epoch
                source += " (pth file missing, fallbacked to last)"
                
            to_delete = []
            for epoch, fpath, size in files:
                if fpath not in to_keep:
                    to_delete.append((fpath, size))
                else:
                    total_saved_size += size
                    
            subdir_report["runs"].append({
                "run_id": run_id,
                "best_epoch": best_epoch,
                "last_epoch": max_epoch,
                "source": source,
                "keep": [os.path.basename(p) for p in to_keep],
                "delete_count": len(to_delete),
                "delete_size_mb": sum(x[1] for x in to_delete) / (1024 * 1024)
            })
            
        report.append(subdir_report)
        
    print("\n=== Checkpoint Dry-Run Cleanup Report (LOG-based Best + Last) ===")
    for subdir_rep in report:
        print(f"\nModel Config: {subdir_rep['name']}")
        for run in subdir_rep["runs"]:
            keep_str = ", ".join(run["keep"])
            print(f"  - Run {run['run_id']}: Keep [{keep_str}] (Best: {run['best_epoch']}, Last: {run['last_epoch']}) [Source: {run['source']}] | "
                  f"Delete {run['delete_count']} files ({run['delete_size_mb']:.2f} MB)")
                  
    orig_gb = total_original_size / (1024**3)
    saved_gb = total_saved_size / (1024**3)
    freed_gb = (total_original_size - total_saved_size) / (1024**3)
    print(f"\nTotal Original Size: {orig_gb:.2f} GB")
    print(f"Total Keep Size: {saved_gb:.2f} GB")
    print(f"Estimated Freed Space: {freed_gb:.2f} GB")

if __name__ == "__main__":
    analyze()
