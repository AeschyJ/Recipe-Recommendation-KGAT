# 專案架構說明

本文件概述了食譜推薦系統的目錄結構與模組設計理念。

## 目錄結構 (Directory Structure)

```
Experiment/
├── .agent/                 # Agent 相關設定與 Workflows
├── data/                   # 資料存放區
│   ├── raw/                # 原始資料 (需手動下載或透過腳本下載)
│   └── processed/          # 預處理後的 Pickle 檔案與中間產物
├── docs/                   # 專案文檔
│   ├── architecture.md     # 本文件
│   └── api_reference.md    # API 說明
├── notebooks/            # Jupyter Notebooks (實驗與訓練)
│   ├── inference_xai.ipynb # 推論與解釋 Demo
│   ├── train_colab.ipynb   # 訓練流程 Demo (舊版)
│   ├── train_attention_colab.ipynb # 真實注意力機制訓練 Demo (Colab 專用) [NEW]
├── output/                 # 輸出結果 (如解釋 JSON) [NEW]
├── src/                    # 原始程式碼
│    ├── data/               # 資料處理模組
    │   ├── download_data.py  # 資料下載指引
    │   └── preprocess.py     # 資料預處理與 KG 建構
    ├── model/              # 模型定義
    │   ├── explainer.py      # GNN 解釋器 (Gradient-based)
    │   ├── explainer_attention.py # GNN 解釋器 (Weight-based)
    │   ├── kgat_bi_interaction.py # KGAT 模型主體 (無 Attention 版)
    │   └── kgat_attention.py # KGAT 模型主體 (Attention)
    ├── train_bi_interaction.py # 本地訓練腳本 (Bi-Interaction 版)
    ├── train_att.py        # 注意力機制訓練腳本 (支援 XPU/CUDA/CPU)
    └── generate_explanations.py # 推論與解釋生成腳本 [NEW]
├── main.py                 # 程式進入點 (開發中)
├── pyproject.toml          # 專案設定與依賴管理
└── requirements.txt        # Python 依賴列表
```

## 模組職責說明

### 1. 資料處理 (`src/data`)

此模組負責將原始的 CSV 資料轉換為模型可讀的格式。主要邏輯位於 `preprocess.py`。

*   **輸入**: `RAW_recipes.csv` (食譜資訊), `RAW_interactions.csv` (使用者評分)。
*   **處理流程**:
    1.  **ID Remapping**: 使用 `LabelEncoder` 將 User ID 和 Recipe ID 轉換為連續整數。
    2.  **Entity Extraction**: 解析食譜中的 `ingredients` 和 `tags` 欄位，將其視為知識圖譜中的實體 (Entity)。
    3.  **Triple Construction**: 建立 `(Recipe, Relation, Entity)` 形式的三元組。
        *   Relation 0: Recipe -> Ingredient
        *   Relation 1: Recipe -> Tag
    4.  **Pruning (剪枝)**:
        *   統計 Ingredients 與 Tags 的出現頻率。
        *   過濾掉前 1% 最常出現的高頻 Ingredient 節點 (Super Nodes)、前 5% 的高頻 TAG 節點，減少雜訊並迫使模型學習更具特色的食材組合 (ADR-007)。
    5.  **Graph Construction (Collaborative Knowledge Graph)**:
        *   將 User-Item 互動與 Item-Entity 關聯整合入單一全域圖譜。
        *   節點偏移規則：Users (0~N-1), Items (N~N+M-1), Entities (N+M~End)。
*   **輸出**: 處理後的 Pickle 檔案 (`interactions.pkl`, `kg_triples.pkl`, `stats.pkl`) 存放在 `data/processed/`。

### 2. 模型核心 (`src/model`)

包含推薦模型與解釋器。

*   **KGAT_BiInteraction (`kgat_bi_interaction.py`) / KGATAttention (`kgat_attention.py`)**:
    *   實作了 Knowledge Graph Attention Network 與其注意力變體。
    *   **GNNLayer**: 定義了單層圖神經網路的聚合邏輯 (Bi-Interaction Aggregation)。
    *   **優化 (ADR-006)**: 實作自定義 `SparseAggregateFunction`，透過 **「重計算策略 (Recomputation)」** 替換原本的快取機制，大幅降低 15M 邊規模下的 VRAM 佔用。
    *   **跨平台支援**: 支援原生 Intel XPU (Arc GPU) 加速、BFloat16 混合精度訓練以及 `--cpu` 強制運算模式。
    *   **目標**: 透過傳播知識圖譜（含使用者互動）中的高階連結資訊，優化使用者與物品的 Embedding。

*   **Explainer (`explainer.py`)**:
    *   **KGATExplainer**: 使用梯度法 (Gradient-based Saliency) 提取重要路徑。
    *   **記憶體優化**: 實作了自定義 `SparseMMFunction` 以避免大型稀疏矩陣求導時產生的 dense 梯度造成 OOM。

### 3. Notebooks (`notebooks/`)

提供實驗性與互動式的開發環境，方便在 Colab 或本地環境執行。

*   **訓練流程 (`train_colab.ipynb`)**: 展示如何載入預處理資料、建構 Graph、以及訓練模型。
*   **推論與解釋 (`inference_xai.ipynb`)**: 展示如何載入訓練好的模型，對特定使用者-物品對進行推論，並透過解釋器產回路徑圖，實現可解釋性。

## 資料流 (Data Flow)

1.  **Raw Data** 📥 (`data/raw/*.csv`)
2.  ➡️ **Preprocessing** (`src/data/preprocess.py`)
3.  ➡️ **Processed Data** 💾 (`data/processed/*.pkl`)
4.  ➡️ **Model Training** (`src/train.py` / `src/train_att.py`)
    *   建構 Collaborative KG Adjacency Matrix (含 User-Item 邊)
    *   訓練模型並優化 Embeddings (支援原生 XPU 加速)
5.  ➡️ **Inference & Explanation** (`src/model/explainer.py`)
    *   產出推薦列表
    *   解釋推薦原因
6.  ➡️ **Inference Script** (`src/generate_explanations.py`)
    *   批量推理
    *   輸出 JSON 至 `output/`
