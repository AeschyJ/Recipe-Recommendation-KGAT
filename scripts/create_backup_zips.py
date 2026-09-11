import os
import zipfile
import glob
import re

MODELS_DIR = "models"
LOGS_DIR = "output/logs"
PAPER_DIR = "Paper/Main"
BACKUP_DIR = "backup_temp"

# Define target zip files
PAPER_ZIP = "paper_latex.zip"
CHECKPOINTS_ZIP = "model_checkpoints_backup.zip"

def parse_logs_for_best_epochs():
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

def find_checkpoints_to_backup():
    print("Analyzing checkpoints based on logs...")
    log_best_epochs = parse_logs_for_best_epochs()
    
    subdirs = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
    
    files_to_backup = []
    
    for subdir in subdirs:
        dir_path = os.path.join(MODELS_DIR, subdir)
        pth_files = glob.glob(os.path.join(dir_path, "*.pth"))
        if not pth_files:
            continue
            
        grouped = {}
        for fpath in pth_files:
            filename = os.path.basename(fpath)
            info = extract_info(filename)
            if info:
                run_id, epoch = info
                if run_id not in grouped:
                    grouped[run_id] = []
                grouped[run_id].append((epoch, fpath))
        
        for run_id, files in sorted(grouped.items()):
            files.sort(key=lambda x: x[0], reverse=True)
            max_epoch, max_fpath = files[0]
            
            # Get best epoch
            log_key = (subdir, run_id)
            best_epoch = None
            if log_key in log_best_epochs:
                best_epoch = log_best_epochs[log_key][0]
                
            if best_epoch is None:
                best_epoch = max_epoch
                
            best_filename = f"{run_id}_kgat_checkpoint_e{best_epoch}.pth"
            best_fpath = os.path.join(dir_path, best_filename)
            
            # Keep Last Epoch
            files_to_backup.append(max_fpath)
            
            # Keep Best Epoch
            if os.path.exists(best_fpath):
                files_to_backup.append(best_fpath)
                
    # Remove duplicates
    files_to_backup = sorted(list(set(files_to_backup)))
    return files_to_backup

def zip_paper():
    print(f"Creating {PAPER_ZIP}...")
    exclude_extensions = {".aux", ".log", ".toc", ".out", ".synctex.gz", ".blg", ".bbl", ".lof", ".lot"}
    
    with zipfile.ZipFile(PAPER_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(PAPER_DIR):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in exclude_extensions:
                    continue
                    
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, PAPER_DIR)
                zipf.write(full_path, os.path.join("Paper", rel_path))
                
    size_mb = os.path.getsize(PAPER_ZIP) / (1024 * 1024)
    print(f"Paper zip created: {PAPER_ZIP} ({size_mb:.2f} MB)")

def zip_checkpoints(files_to_backup):
    print(f"Creating {CHECKPOINTS_ZIP} with {len(files_to_backup)} files...")
    
    with zipfile.ZipFile(CHECKPOINTS_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
        for fpath in files_to_backup:
            # We want to keep the models/subdir/file structure
            rel_path = os.path.relpath(fpath, os.path.dirname(MODELS_DIR))
            zipf.write(fpath, rel_path)
            print(f"  Added: {fpath}")
            
    size_mb = os.path.getsize(CHECKPOINTS_ZIP) / (1024 * 1024)
    print(f"Checkpoints zip created: {CHECKPOINTS_ZIP} ({size_mb:.2f} MB)")

def main():
    files_to_backup = find_checkpoints_to_backup()
    zip_paper()
    zip_checkpoints(files_to_backup)

if __name__ == "__main__":
    main()
