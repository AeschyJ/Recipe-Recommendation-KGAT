# 開發者指南與維護手冊 (Development & Maintenance Guide)

本手冊專為開發者與實驗維護人員設計，涵蓋環境建置、自動化實驗執行、腳本工具維護、NotebookLM 文檔整理及 Git 版本控制防錯指南。

---

## 1. 開發環境建置 (Environment Setup)

本專案全面採用 **`uv`** 作為包管理器與虛擬環境管理工具：

```bash
# 1. 建立虛擬環境與同步套件
uv sync

# 2. 測試 Python 環境與 PyTorch XPU
uv run python -c "import torch; print('XPU Available:', torch.xpu.is_available() if hasattr(torch, 'xpu') else False)"
```

> **注意**：專案使用 PyTorch 原生 XPU 支援（PyTorch 2.4+ / 2.9+），無需單獨安裝舊版 IPEX 套件。

---

## 2. 批次腳本工作流 (Batch Script Workflows)

專案根目錄提供 5 組 `.bat` 批次檔，分別對應不同階段的實驗任務：

1. **`run_experiments.bat`**：消融實驗全流程。依序觸發 Full KGAT ($L=1$), w/o Attention, w/o KG, $L=2$, $L=3$ 訓練。
2. **`run_baseline_experiments.bat`**：Baseline 模型訓練。觸發 BPR-MF, LightGCN, NFM 訓練。
3. **`run_xai_pipeline.bat`**：單一模型 XAI 評估。
4. **`run_xai_pipeline_all.bat`**：跨模型全自動 500 位使用者 Fidelity+ / Fidelity- 評估。
5. **`run_log_pipeline.bat`**：訓練日誌簡化、資料驗證與指標解析報告。

---

## 3. 檢查點管理與磁碟空間釋放 (Checkpoint & Disk Management)

模型訓練過程中會產生多個 Epoch 的 `.pth` 檢查點，維護人員可使用以下腳本釋放空間：

```bash
# 1. 檢視檢查點佔用情況
uv run python scripts/analyze_checkpoints_cleanup.py

# 2. 清理歷史中間 Epoch，僅留 Best Checkpoint
uv run python scripts/cleanup_checkpoints.py

# 3. 建立本地備份壓縮檔 (檔名會自動被 Git 忽略)
uv run python scripts/create_backup_zips.py
```

---

## 4. NotebookLM 文檔編譯 (NotebookLM Compilation)

為了方便將全專案技術文檔、日誌與論文資料匯入 Google NotebookLM 或 LLM 進行深度研讀，專案包含專屬編譯腳本：

```bash
# 執行 NotebookLM 資料編譯
uv run python NotebookLM/compile_for_notebooklm.py
```
*編譯後的獨立 Markdown 檔案將存放於 `NotebookLM/` 目錄下。*

---

## 5. Git 版本控制規範與排除指南 (Git Submission Guidelines)

本專案遵循 standard Conventional Commits 規範，且設定嚴格的檔案排除防錯。

### 5.1 提交訊息規範 (Conventional Commits)
* `feat(kgat)`: 新增或修改模型特徵。
* `fix(xpu)`: 修正硬體加速或計算 Bug。
* `docs(readme)`: 文檔新增或修訂。
* `chore(git)`: 版本控制與雜務維護。
* `style(code)`: 程式碼格式微調。

### 5.2 大檔案排除防錯規則
> [!CAUTION]
> **警告**：GitHub 限制單檔上傳不可超過 **100 MB**。
> 專案中的 `model_checkpoints_backup.zip` (~4 GB) 與 `paper_latex.zip` (~77 MB) 已透過 `.gitignore` 強制排除。開發者切勿使用 `git add -f` 強制加入封包。

在執行提交前，請執行以下命令確認無大型二進位檔被暫存：
```bash
git status
```
確保 `Untracked files` 中不含 `.zip`, `.pth`, `.venv/`, `output/` 等大型檔案。
