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
UPDATE_SCRIPT = "models/update_models_list.py"


def cleanup():
    print("Parsing logs to find log-based best epochs...")
    log_best_epochs = parse_logs_for_best_epochs(logs_dir=LOGS_DIR, models_dir=MODELS_DIR)
    print(f"Parsed {len(log_best_epochs)} runs from logs.")
    
    subdirs = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
    
    total_original_size = 0
    total_freed_size = 0
    deleted_files_count = 0
    
    to_delete_list = []
    to_keep_list = []
    
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
        
        for run_id, files in sorted(grouped.items()):
            files.sort(key=lambda x: x[0], reverse=True)
            max_epoch, max_fpath, _ = files[0]
            
            log_key = (subdir, run_id)
            best_epoch = None
            
            if log_key in log_best_epochs:
                best_epoch = log_best_epochs[log_key][0]
            else:
                try:
                    checkpoint = torch.load(max_fpath, map_location="cpu", weights_only=False)
                    best_epoch = checkpoint.get("best_epoch")
                except Exception as e:
                    print(f"Error loading {max_fpath}: {e}")
            
            if best_epoch is None:
                best_epoch = max_epoch
                
            best_filename = f"{run_id}_kgat_checkpoint_e{best_epoch}.pth"
            best_fpath = os.path.join(dir_path, best_filename)
            
            to_keep_for_this_run = set()
            
            # Keep Last Epoch
            to_keep_for_this_run.add(max_fpath)
            
            # Keep Best Epoch if exists
            if os.path.exists(best_fpath):
                to_keep_for_this_run.add(best_fpath)
            else:
                to_keep_for_this_run.add(max_fpath)
                
            to_keep_list.extend(list(to_keep_for_this_run))
            
            for epoch, fpath, size in files:
                if fpath not in to_keep_for_this_run:
                    to_delete_list.append((fpath, size))
                    
    # Execute deletion
    print("\n=== Executing Checkpoint Cleanup (LOG-based Best + Last) ===")
    for fpath, size in to_delete_list:
        try:
            os.remove(fpath)
            total_freed_size += size
            deleted_files_count += 1
            print(f"Deleted: {fpath} ({size / (1024*1024):.2f} MB)")
        except Exception as e:
            print(f"Error deleting {fpath}: {e}")
            
    print(f"\nCleanup finished.")
    print(f"Total deleted files: {deleted_files_count}")
    print(f"Total freed space: {total_freed_size / (1024**3):.2f} GB")
    
    # Update Models MD
    if os.path.exists(UPDATE_SCRIPT):
        print("\nUpdating MODELS.md...")
        try:
            import subprocess
            subprocess.run([".venv/Scripts/python", UPDATE_SCRIPT], check=True)
        except Exception as e:
            print(f"Error running {UPDATE_SCRIPT}: {e}")

if __name__ == "__main__":
    cleanup()
