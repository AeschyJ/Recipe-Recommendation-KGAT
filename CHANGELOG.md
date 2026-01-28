# 變更日誌 (Changelog)

本文件紀錄專案的所有重要改動。依據 [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) 格式撰寫。

## [Unreleased]

### Added
- **Inference**: `src/generate_explanations.py` - 新增批量推理與解釋生成腳本，支援自動將內部 ID 轉換為原始 ID 與實際名稱（如食譜名稱、標籤名），以便於 LLM 進行解釋。
- **Inference**: 自動建立 `output/` 目錄並將解釋結果儲存為 JSON 格式。
- **Documentation**: 更新 `README.md`、`docs/architecture.md` 與 `docs/api_reference.md` 以包含新腳本的使用說明。

### Fixed
- **Environment**: 修改 `pyproject.toml` 引進 `markupsafe<3.0.0` 並鎖定 Python 3.12，解決 Windows 環境下的依賴衝突問題。
- **Inference**: 修復 JSON 序列化時 `float32` 型別不相容的問題，確保推理結果能正確輸出。

## [1.1.0] - 2025-12-23

### Added
- **Graph**: 在鄰接矩陣中引入 User-Item 互動邊，使解釋器能追蹤從使用者出發的路徑邏輯 (ADR-005)。
- **Setup**: `pyproject.toml` 新增 `torch-xpu` 專用索引，鎖定 `torch==2.9.1+xpu`。
- **Training**: `train.py` 與 `train_att.py` 新增完整的 Checkpoint 儲存與恢復功能 (`--resume`)。
- **Training**: 訓練腳本新增 `--cpu` 參數，支援在 GPU 環境下強制使用 CPU 訓練。

### Changed
- **XPU**: 遷移至 PyTorch 原生 XPU 支援，移除 IPEX 依賴 (ADR-004)。
- **Optimization**: 實作 GNN 聚合重計算策略 (Recomputation)，解決千萬級邊規模下的 VRAM 溢位與系統 Swapping 問題 (ADR-006)。
- **Training**: 優化模型 `forward` 介面，將正樣本與負樣本分數合併計算，減少 50% 的圖遍歷開銷。
- **Training**: 更新 `src/train.py` 與 `src/train_att.py` 以支援原生 XPU 偵測與大型 Batch Size 優化。


## [1.0.0] - 2025-12-19

### Added
- **Model**: `KGATAttention` (`src/model/kgat_attention.py`) - 實作基於 PyTorch `scatter_add` 的真實 Graph Attention 機制 (ADR-002)。
- **Explainer**: `KGATAttentionExplainer` (`src/model/explainer_attention.py`) - 利用 Attention Weights 進行推薦解釋 (ADR-002)。
- **Training**: `src/train_xpu.py` - 支援 Intel Arc GPU (IPEX) 加速的訓練腳本。
- **Notebook**: `notebooks/train_attention_colab.ipynb` - 針對 Google Colab 網頁環境最佳化的訓練 Notebook，支援 Google Drive 持久化與斷點續傳。
- **Optimization**: 引入 Gather-Scatter (Message Passing) 模式與主動記憶體回收機制，解決大型圖譜下的 CUDA OOM (800GB+) 與系統 RAM 洩漏問題 (ADR-003)。
- **Documentation**: 新增 ADR-002 (Attention 實作) 與 ADR-003 (Colab 訓練優化) 紀錄。

### Changed
- **架構調整**: 移除 `dgl` 依賴，將 KGAT 模型與資料流重構為純 PyTorch 實作，以解決 Windows 環境相容性問題 (Ref: ADR-001)。
- `src/model/kgat.py`: 重寫 GNN Layer，使用 native torch 運算替代 DGL message passing。
- `docs/architecture.md`: 更新架構描述以反映 DGL 的移除。

## [0.1.0] - 2025-12-17
### Added
- 專案初始化。
- `src/data/preprocess.py`：資料預處理與 KG 建構。
- `src/model/kgat.py`：KGAT 模型初始實作 (基於 DGL)。
- `notebooks/`：Colab 訓練與推論範例。
