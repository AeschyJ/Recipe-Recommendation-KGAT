import re
from pathlib import Path
from datetime import datetime

# Get the base directory (one level up from where the script should be run)
# We assume this script will be in models/update_models_list.py
# If it's in models/ update_models_list.py, models_dir is current dir
MODELS_DIR = Path(__file__).resolve().parent
MODELS_MD = MODELS_DIR / "MODELS.md"

def get_file_info(file_path):
    """Helper to get file size and formatted modified time."""
    stat = file_path.stat()
    size_mb = stat.st_size / (1024 * 1024)
    mod_time = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    return f"{size_mb:.2f} MB", mod_time
def extract_epoch(filename):
    """Extract numeric epoch from filename like 'kgat_checkpoint_e10.pth'."""
    match = re.search(r'e(\d+)', filename)
    return int(match.group(1)) if match else 0

def update_models_md():
    # Identify subdirectories to scan
    subdirs = {
        "1. Full KGAT (Baseline)": "full_kgat",
        "2. KGAT w/o Attention (wo_attn)": "wo_attn",
        "3. KGAT w/o KG (wo_kg)": "wo_kg",
        "4. Depth 2 (depth_2)": "depth_2",
        "5. Depth 3 (depth_3)": "depth_3"
    }

    content = [
        "# 已訓練模型列表 (Trained Models)",
        f"\n*最後更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "\n此目錄下的 `.pth` 權重檔案體積較大，已根據 `.gitignore` 設定排除於版本控制之外。",
        "以下紀錄目前實驗中產出的權重檔案資訊：\n"
    ]

    for title, dirname in subdirs.items():
        dir_path = MODELS_DIR / dirname
        content.append(f"## {title}")
        content.append(f"- **目錄**: `models/{dirname}/`")
        content.append("- **權重檔案**:")
        
        # Sort files by the numeric epoch value instead of alphabetically
        pth_files = sorted(list(dir_path.glob("*.pth")), key=lambda x: extract_epoch(x.name))
        
        if not pth_files:
            content.append("  - *(尚未產生權重檔案)*")
        else:
            for pth in pth_files:
                size, mtime = get_file_info(pth)
                content.append(f"  - `{pth.name}` ({size} | {mtime})")
        content.append("") # Empty line between sections

    content.append("## 附註")
    content.append("- 所有訓練日誌均存放於 `output/logs/` 目錄中，並已包含在版本控制內。")
    content.append("- 若需恢復訓練或進行推理，請確保本地端存在對應的 `.pth` 檔案。")

    with open(MODELS_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(content))
    
    print(f"Successfully updated {MODELS_MD.name}")

if __name__ == "__main__":
    update_models_md()
