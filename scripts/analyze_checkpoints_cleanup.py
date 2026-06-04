import os
import glob
import re
import torch

MODELS_DIR = "models"
LOGS_DIR = "output/logs"

def parse_logs_for_best_epochs():
    """
    Parse all logs in output/logs/ to find the best epoch for each (model_dir, run_id).
    Returns a dict: {(model_dir_basename, run_id): (best_epoch, best_hr20)}
    """
    best_runs = {}
    
    # Find all .txt log files in logs directory recursively
    log_files = glob.glob(os.path.join(LOGS_DIR, "**/*.txt"), recursive=True)
    
    for log_path in log_files:
        if "_reformatted" in log_path:
            continue
            
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        # 1. Parse model_dir and experiment_id from Args namespace if available
        # Args namespace snippet: data_dir='data/processed', model_dir='models/depth_2', ...
        model_dir_match = re.search(r"model_dir='([^']+)'", content)
        exp_id_match = re.search(r"experiment_id='([^']+)'", content)
        
        model_dir = model_dir_match.group(1) if model_dir_match else None
        run_id = int(exp_id_match.group(1)) if exp_id_match else None
        
        # 2. If not found in args, parse from "Saved checkpoint" lines
        # snippet: Saved checkpoint: models/depth_2\2_kgat_checkpoint_e1.pth
        if not run_id or not model_dir:
            ckpt_matches = re.findall(r"Saved checkpoint:\s*([^\s\n]+)", content)
            if ckpt_matches:
                for ckpt_path in ckpt_matches:
                    norm_path = ckpt_path.replace("\\", "/")
                    # extract subdir and run_id
                    # e.g., models/depth_2/2_kgat_checkpoint_e1.pth
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
            # Fallback to subdirectory of log file
            # e.g., output/logs/depth_2/kgat_...txt -> models/depth_2
            log_rel = os.path.relpath(log_path, LOGS_DIR)
            parts = log_rel.replace("\\", "/").split("/")
            if len(parts) >= 2:
                model_dir = os.path.join(MODELS_DIR, parts[0])
            else:
                model_dir = MODELS_DIR # fallback
                
        if not run_id:
            run_id = 1 # default
            
        model_basename = os.path.basename(model_dir.replace("\\", "/"))
        
        # 3. Parse evaluation metrics and find best epoch
        # Format: Epoch 1 Evaluation - HR@[10,20,50]: [0.7542, 0.8904, 0.9860] ...
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

def analyze():
    print("Parsing logs to find log-based best epochs...")
    log_best_epochs = parse_logs_for_best_epochs()
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
