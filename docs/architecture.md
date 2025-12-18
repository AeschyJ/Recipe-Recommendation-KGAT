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
│   └── train_attention_colab.ipynb # 真實注意力機制訓練 Demo (Colab 專用) [NEW]
├── src/                    # 原始程式碼
│    ├── data/               # 資料處理模組
    │   ├── download_data.py  # 資料下載指引
    │   └── preprocess.py     # 資料預處理與 KG 建構
    ├── model/              # 模型定義
    │   ├── explainer.py      # GNN 解釋器 (Gradient-based)
    │   ├── explainer_attention.py # GNN 解釋器 (Weight-based) [NEW]
    │   ├── kgat.py           # KGAT 模型主體 (Static)
    │   └── kgat_attention.py # KGAT 模型主體 (Attention) [NEW]
    ├── train.py            # 本地訓練腳本 (Default)
    └── train_xpu.py        # 本地訓練腳本 (Intel Arc) [NEW]
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
*   **輸出**: 處理後的 Pickle 檔案 (`interactions.pkl`, `kg_triples.pkl`, `stats.pkl`) 存放在 `data/processed/`。

### 2. 模型核心 (`src/model`)

包含推薦模型與解釋器。

*   **KGAT (`kgat.py`)**:
    *   實作了 Knowledge Graph Attention Network。
    *   **GNNLayer**: 定義了單層圖神經網路的聚合邏輯 (Bi-Interaction Aggregation)。
    *   **KGAT Class**: 整合 Embedding 層與多層 GNNLayer，計算使用者與物品的匹配分數。
    *   **目標**: 透過傳播知識圖譜中的高階連結資訊，優化使用者與物品的 Embedding。

*   **Explainer (`explainer.py`)**:
    *   **KGATExplainer**: (待重構) 用於解釋推薦結果的各項權重。
    *   **目標**: 給定一個推薦 (User -> Item)，找出導致該推薦最重要的子圖 (Subgraph)，例如「因為該使用者喜歡包含『巧克力』的食譜，所以推薦了這個蛋糕」。

### 3. Notebooks (`notebooks/`)

提供實驗性與互動式的開發環境，方便在 Colab 或本地環境執行。

*   **訓練流程 (`train_colab.ipynb`)**: 展示如何載入預處理資料、建構 Graph、以及訓練靜態 KGAT 模型。
*   **注意力訓練 (`train_attention_colab.ipynb`)**: 針對新版 `KGATAttention` 最佳化的訓練腳本，包含完整的訓練迴圈實作，方便在 Colab GPU 環境直接執行。
*   **推論與解釋 (`inference_xai.ipynb`)**: 展示如何載入訓練好的模型，對特定使用者-物品對進行推論，並呼叫 Explainer 產出推薦解釋。

## 資料流 (Data Flow)

1.  **Raw Data** 📥 (`data/raw/*.csv`)
2.  ➡️ **Preprocessing** (`src/data/preprocess.py`)
3.  ➡️ **Processed Data** 💾 (`data/processed/*.pkl`)
    *   包含：Interaction Matrix, Knowledge Graph Triples, ID Maps
4.  ➡️ **Model Training** (`src/model/kgat.py`)
    *   建構 Graph Adjacency Matrix (Sparse Tensor)
    *   訓練 KGAT 模型優化 Embeddings (Pure PyTorch)
5.  ➡️ **Inference & Explanation** (`src/model/explainer.py`)
    *   產出推薦列表
    *   解釋推薦原因
