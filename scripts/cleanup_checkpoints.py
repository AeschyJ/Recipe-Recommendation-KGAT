import os
import glob
import re
import torch

MODELS_DIR = "models"
LOGS_DIR = "output/logs"
UPDATE_SCRIPT = "models/update_models_list.py"

def parse_logs_for_best_epochs():
    """
    Parse all logs in output/logs/ to find the best epoch for each (model_dir, run_id).
    Returns a dict: {(model_dir_basename, run_id): (best_epoch, best_hr20)}
    """
    best_runs = {}
    
    log_files = glob.glob(os.path.join(LOGS_DIR, "**/*.txt"), recursive=True)
    
    for log_path in log_files:
        if "_reformatted" in log_path:
            continue
            
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        model_dir_match = re.search(r"model_dir='([^']+)'", content)
        exp_id_match = re.search(r"experiment_id='([^']+)'", content)
        
        model_dir = model_dir_match.group(1) if model_dir_match else None
        run_id = int(exp_id_match.group(1)) if exp_id_match else None
        
        if not run_id or not model_dir:
            ckpt_matches = re.findall(r"Saved checkpoint:\s*([^\s\n]+)", content)
            if ckpt_matches:
                for ckpt_path in ckpt_matches:
                    norm_path = ckpt_path.replace("\\", "/")
                    path_parts = norm_path.split("/")
                    if len(path_parts) >= 2:
                        parsed_model = path_parts[-2]
                        filename = path_parts[-1]
                        filename_match = re.match(r"(\d+)_kgat_checkpoint", filename)
                        if filename_match:
                            if not model_dir:
                                model_dir = os.path.join(MODELS_DIR, parsed_model)
                            if not run_id:
                                run_id = int(filename_match.group(1))
                            break
                            
        if not model_dir:
            log_rel = os.path.relpath(log_path, LOGS_DIR)
            parts = log_rel.replace("\\", "/").split("/")
            if len(parts) >= 2:
                model_dir = os.path.join(MODELS_DIR, parts[0])
            else:
                model_dir = MODELS_DIR
                
        if not run_id:
            run_id = 1
            
        model_basename = os.path.basename(model_dir.replace("\\", "/"))
        
        eval_lines = re.findall(r"Epoch (\d+) Evaluation - (?:HR|Recall)@\[10,20,50\]:\s*\[[^,]+,\s*([^,\]]+)", content)
        
        best_epoch = None
        best_hr20 = -1.0
        
        for epoch_str, hr20_str in eval_lines:
            try:
                epoch = int(epoch_str)
                hr20 = float(hr20_str)
                if hr20 > best_hr20:
                    best_hr20 = hr20
                    best_epoch = epoch
            except ValueError:
                continue
                
        if best_epoch is not None:
            key = (model_basename, run_id)
            if key not in best_runs or best_hr20 > best_runs[key][1]:
                best_runs[key] = (best_epoch, best_hr20)
                
    return best_runs

def extract_info(filename):
    match = re.match(r"(\d+)_kgat_checkpoint_e(\d+)\.pth", filename)
    if match:
        run_id = int(match.group(1))
        epoch = int(match.group(2))
        return run_id, epoch
    return None

def cleanup():
    print("Parsing logs to find log-based best epochs...")
    log_best_epochs = parse_logs_for_best_epochs()
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
