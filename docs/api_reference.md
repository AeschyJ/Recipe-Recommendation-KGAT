# API 參考文件 (API Reference)

本文件詳細說明了 `src` 目錄下各模組的類別與函數定義，以及 `notebooks` 中的實驗流程。

## `src/data` - 資料處理

### `download_data.py`

負責提供資料下載指引。

#### `download_kaggle_dataset()`
*   **功能**: 印出從 Kaggle 下載 Food.com 資料集的指引與網址。
*   **說明**: 由於資料集較大且需 Kaggle API 權限，目前主要作為提示工具。

---

### `preprocess.py`

負責資料清洗、ID 重新映射與知識圖譜建構。

#### `process_data(data_dir)`
*   **功能**: 執行完整的資料預處理流程。
*   **參數**:
    *   `data_dir` (str): 專案根目錄路徑，預期該目錄下有 `data/raw` 包含原始 CSV 檔。
*   **輸出**: 於 `data/processed` 產生以下檔案：
    *   `interactions.pkl`: 包含 `user_id_remap`, `recipe_id_remap`, `rating` 的 DataFrame。
    *   `kg_triples.pkl`: `numpy` 陣列，包含 `[head, relation, tail]` 三元組。
    *   `stats.pkl`: 包含各實體數量統計與 ID 映射表 (LabelEncoders) 的字典。

---

## `src/model` - 模型定義

### `kgat.py`

實作 KGAT (Knowledge Graph Attention Network) 模型。

#### class `KGAT(nn.Module)`
模型主體，繼承自 `torch.nn.Module`。

*   **`__init__(self, n_users, n_entities, n_relations, embed_dim=64, layers=[64, 32], ...)`**
    *   初始化 Embedding 層與 GNN 層。
    *   `n_users`: 使用者總數。
    *   `n_entities`: 知識圖譜實體總數 (含物品、成分、標籤等)。
    *   `layers`: 定義 GNN 每層的輸出維度。

*   **`forward(self, g, user_ids, item_ids)`**
    *   **參數**:
        *   `g` (DGLGraph): 協同知識圖譜 (Collaborative Knowledge Graph)。
        *   `user_ids` (Tensor): 目標使用者索引。
        *   `item_ids` (Tensor): 目標物品索引。
    *   **回傳**: `scores` (Tensor)，預測的匹配分數。
    *   **邏輯**: 執行多層 GNN 聚合後，將最後一層的 Embedding 進行內積運算。

#### class `GNNLayer(nn.Module)`
單層圖神經網路層。

*   **`forward(self, g, features)`**
    *   執行訊息傳遞 (Message Passing)。
    *   實作 Bi-Interaction Aggregation 機制 (`W1(h+h_neigh) + W2(h*h_neigh)` + Activation)。

---

### `explainer.py`

推薦結果解釋模組。

#### class `KGATExplainer`
`dgl.nn.GNNExplainer` 的包裝器。

*   **`__init__(self, model, num_hops=2)`**
    *   初始化解釋器，設定要解釋的模型與搜尋的跳數 (Hops)。

*   **`explain(self, g, user_id, item_id, top_k=10)`**
    *   **功能**: 解釋為何模型推薦了特定物品給特定使用者。
    *   **參數**:
        *   `g`: 圖結構。
        *   `user_id`, `item_id`: 目標使用者與物品。
    *   **回傳**: 包含重要節點與邊的子圖資訊字典。

*   **`visualize(self, explanation)`**
    *   **功能**: 將解釋結果視覺化 (目前為 Placeholder)。

---

## `notebooks` - 實驗與流程

### `train_colab.ipynb` (KGAT Training)

*   **功能**: 模型的訓練腳本，設計用於 Colab 或 Jupyter 環境。
*   **流程**:
    1.  **安裝依賴**: 安裝 `torch`, `dgl` 等庫。
    2.  **載入資料**: 讀取 `interactions.pkl`, `kg_triples.pkl`, `stats.pkl`。
    3.  **建構圖**: 建立 DGL Graph，包含使用者與實體節點。
    4.  **訓練迴圈**: 初始化 `KGAT` 模型，定義 BPR Loss，並在 Epochs 中執行訓練 (Sample Batch -> Forward -> Backward)。

### `inference_xai.ipynb` (Inference & Explanation)

*   **功能**: 載入模型並展示解釋性 AI 功能。
*   **流程**:
    1.  **環境設定**: 引入 `KGAT` 與 `KGATExplainer`。
    2.  **載入模型**: (範例程式碼) 初始化模型並載入權重。
    3.  **產生解釋**:
        *   初始化 `KGATExplainer(model)`.
        *   針對特定 `user_id` 與 `item_id` 呼叫 `explain()`。
        *   呼叫 `visualize()` 顯示結果。
