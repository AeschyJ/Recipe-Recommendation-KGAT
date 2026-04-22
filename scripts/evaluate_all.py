import glob
import os
import subprocess
import time


def build_ckpt_to_log_map():
    ckpt_to_log = {}
    for log_file in glob.glob("output/logs/**/*.txt", recursive=True):
        # 跳過已經被重新格式化過的記錄檔，避免重複讀取或輸出
        if "_reformatted" in log_file:
            continue
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if "Saved checkpoint:" in line:
                        parts = line.split("Saved checkpoint:")
                        if len(parts) > 1:
                            ckpt = parts[1].strip()
                            ckpt_norm = ckpt.replace("\\", "/")
                            ckpt_to_log[ckpt_norm] = log_file.replace("\\", "/")
        except:
            pass
    return ckpt_to_log


CKPT_TO_LOG = build_ckpt_to_log_map()


def get_command_for_checkpoint(ckpt_path):
    # 決定如何透過檔名與路徑對應回去正確的參數
    ckpt_path_norm = ckpt_path.replace("\\", "/")

    python_exe = r".venv\Scripts\python.exe"
    if not os.path.exists(python_exe):
        python_exe = "python"

    filename = os.path.basename(ckpt_path).lower()

    if "baseline" in ckpt_path_norm:
        if "bprmf" in filename:
            cmd = [python_exe, "src/train_baseline.py", "--model", "BPR-MF"]
        elif "lightgcn" in filename:
            cmd = [python_exe, "src/train_baseline.py", "--model", "LightGCN"]
        elif "nfm" in filename:
            cmd = [python_exe, "src/train_baseline.py", "--model", "NFM"]
        else:
            return None
    elif "wo_attn" in ckpt_path_norm:
        cmd = [python_exe, "src/train_bi_interaction.py"]
    elif "wo_kg" in ckpt_path_norm:
        cmd = [python_exe, "src/train_att.py", "--without_kg", "--no_compile"]
    elif "depth_2" in ckpt_path_norm:
        cmd = [python_exe, "src/train_att.py", "--layers", "64", "64", "--no_compile"]
    elif "depth_3" in ckpt_path_norm:
        cmd = [
            python_exe,
            "src/train_att.py",
            "--layers",
            "64",
            "64",
            "64",
            "--no_compile",
        ]
    elif "full_kgat" in ckpt_path_norm:
        cmd = [python_exe, "src/train_att.py", "--no_compile"]
    else:
        return None

    # 加入共同的 evaluation flag
    cmd.extend(["--resume", ckpt_path, "--eval_only"])

    # 加入原本的 log 檔案
    if ckpt_path_norm in CKPT_TO_LOG:
        cmd.extend(["--log_file", CKPT_TO_LOG[ckpt_path_norm]])

    return cmd


def main():
    # 找尋包含子資料夾內所有的 .pth 檔案
    all_checkpoints = glob.glob("models/depth_3/*.pth", recursive=True)
    all_checkpoints = sorted(all_checkpoints)

    # 過濾掉可能是測試用的舊歸檔 'old'，並只保留檔名以 '2_' 開頭的模型
    valid_checkpoints = [
        c
        for c in all_checkpoints
        if "/old/" not in c.replace("\\", "/")
        and "/baseline/" not in c.replace("\\", "/")
        # and os.path.basename(c).startswith("1_")
    ]

    print(f"找到 {len(valid_checkpoints)} 個符合條件的評估點：")
    for ckpt in valid_checkpoints:
        print(f" - {ckpt}")
    print("=" * 70)

    if not valid_checkpoints:
        print("找不到任何以 '2_' 開頭的模型，程式結束。")
        return

    confirm = input("\n確認是否開始評估上述模型？ (y/n): ")
    if confirm.lower() != "y":
        print("已取消評估。")
        return

    print("\n開始執行評估作業...")
    time.sleep(1)

    for i, ckpt in enumerate(valid_checkpoints, 1):
        cmd = get_command_for_checkpoint(ckpt)
        if cmd:
            print(f"\n\n{'=' * 70}")
            print(f"[{i}/{len(valid_checkpoints)}] 正在評估: {ckpt}")
            print(f"執行指令: {' '.join(cmd)}")
            print(f"{'=' * 70}\n")

            # 如果有找到對應的 log 檔案，就會一起紀錄進去
            if ckpt.replace("\\", "/") in CKPT_TO_LOG:
                print(
                    f"將輸出附加到原始記錄檔: {CKPT_TO_LOG[ckpt.replace('\\', '/')]}",
                    flush=True,
                )

            # 使用 subprocess 同步呼叫並等待完成，輸出會直接打到螢幕與 Log 中
            process = subprocess.run(cmd)

            if process.returncode != 0:
                print(f"[警告] 評估 {ckpt} 過程中發生錯誤，將跳過並執行下一個。")
        else:
            print(f"\n[跳過] 無法決定 {ckpt} 的正確執行參數。")


if __name__ == "__main__":
    main()
