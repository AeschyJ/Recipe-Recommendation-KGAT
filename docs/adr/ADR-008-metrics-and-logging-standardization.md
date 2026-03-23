# ADR-008: 統一超參數、擴展召回指標與日誌系統

## Status
Accepted

## Context
隨著專案進入實驗對照階段，我們需要更嚴謹的實驗記錄與統一的評估標準。先前的訓練腳本 (`train.py` 與 `train_att.py`) 在超參數預設值、評估指標 (Recall@K) 以及輸出記錄方式上存在差異，難以直接比較 `KGAT` (Base) 與 `KGATAttention` 的效能。

## Decision
我們決定統一訓練配置並增強觀測能力：

1.  **統一超參數 (Hyperparameters Standardization)**：
    -   **Base Learning Rate**: 統一設定為 `1e-3`。
    -   **Batch Size**: 針對 Attention 模型 (`train_att.py`)，預設 Batch Size 調整為 `1024` 以匹配 Base 模型的吞吐量配置。
    -   **Epochs**: 預設 20。

2.  **擴展評估指標 (Extended Metrics)**：
    -   將原有的單一 `Loss` 擴展為 **`Recall@10`, `Recall@20`, `Recall@50`**。
    -   修改 `evaluate` 函數，統一使用 `torch.topk` 進行排名計算，並使用相同的隨機負採樣策略 (1 Positive + 100 Negatives)。

3.  **實作文件日誌系統 (File Logging)**：
    -   引入 Python `logging` 模組，取代單純的 `print` 輸出。
    -   訓練過程會同時輸出至 Console 與 **Log File** (預設路徑 `models/logs/`)。
    -   Log 檔名包含模型名稱與時間戳記，便於回溯實驗結果。

## Consequences
-   **Positive**:
    -   可以直接平行比對 Base 與 Attention 模型的實驗數據。
    -   多階層的 Recall 指標 (10/20/50) 能更全面評估模型的排序能力。
    -   實驗數據自動持久化，無需手動複製終端機輸出。
-   **Negative**:
    -   `evaluate` 函數的計算量略微增加 (計算 Top-50)，但對整體訓練時間影響微乎其微。
    -   日誌檔案會佔用少量磁碟空間，需定期清理。

## Compliance
-   修改 `src/train.py`。
-   修改 `src/train_att.py`。
