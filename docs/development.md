# 開發指南 (Development Guide)

歡迎參與本專案開發。本文件涵蓋環境設定、開發規範與常見問題排解。

## 環境設定 (Environment Setup)

本專案使用 `uv` 進行依賴管理，請確保已安裝該工具。

### 1. 初始化環境

首次 clone 專案後，請執行：

```bash
uv sync
```

此指令會讀取 `uv.lock` 並安裝所有 Python 套件 (包含 PyTorch 與相關依賴)。

### 2. GPU 支援

專案預設依賴 PyTorch。若您的環境支援 CUDA，`torch` 應能自動識別。您可以用以下指令測試：

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### 3. 開發工具

建議使用 VS Code 並安裝以下套件：
*   Python (Microsoft)
*   Ruff (Linter / Formatter)
*   Markdown All in One (文件撰寫)

---

## 程式碼規範 (Coding Standards)

*   **Python 版本**: 3.10+
*   **格式化**: 本專案使用 [PEP 8](https://peps.python.org/pep-0008/) 標準。建議設定編輯器在存檔時自動執行 `ruff format`。
*   **型別註釋**: 鼓勵在函數簽名中加入 Type Hints。
    ```python
    def process(data: pd.DataFrame, threshold: float = 0.5) -> dict:
        ...
    ```
*   **語言**: 程式碼註解、文件與 Commit Message 請使用 **繁體中文**。

---

## 常見問題 (Troubleshooting)

### Q1: DGL 安裝失敗或版本衝突？
**A**: 本專案目前已逐步移除對 `dgl` 的重度依賴，改用純 PyTorch 實作核心模型 (詳見 ADR 紀錄)。若仍需使用舊版程式碼，建議優先檢查 CUDA 版本與 DGL 預編譯包的相容性。

### Q2: 執行 `preprocess.py` 出現 Memory Error？
**A**: `RAW_interactions.csv` 檔案較大。若記憶體不足，可嘗試在讀取 CSV 時加入 `nrows=100000` 參數進行測試，或分批處理。

### Q3: 使用 VS Code 無法解析 Import？
**A**: 請確保 VS Code 的 Python Interpreter 選定為 `.venv/Scripts/python.exe` (Windows) 或 `.venv/bin/python` (Linux/Mac)。
