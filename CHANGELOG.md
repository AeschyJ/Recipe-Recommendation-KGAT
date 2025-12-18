# 變更日誌 (Changelog)

本文件紀錄專案的所有重要改動。依據 [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) 格式撰寫。

## [Unreleased]

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
