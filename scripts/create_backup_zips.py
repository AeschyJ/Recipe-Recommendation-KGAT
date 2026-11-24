import os
import sys
import zipfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.log_utils import extract_info, parse_logs_for_best_epochs

MODELS_DIR = "models"
LOGS_DIR = "output/logs"
PAPER_DIR = "Paper/Main"
BACKUP_DIR = "backup_temp"

# Define target zip files
PAPER_ZIP = "paper_latex.zip"
CHECKPOINTS_ZIP = "model_checkpoints_backup.zip"


def find_checkpoints_to_backup():
    print("Analyzing checkpoints based on logs...")
    log_best_epochs = parse_logs_for_best_epochs(logs_dir=LOGS_DIR, models_dir=MODELS_DIR)
    
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
