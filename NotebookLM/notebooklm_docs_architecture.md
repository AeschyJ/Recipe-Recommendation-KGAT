# System Architecture & Other Docs

## File: docs\api_reference.md

# API 參考文件

本文件詳細說明了 `src` 目錄下各模組的類別與函數定義，以及 `notebooks` 中的實驗流程。

## `src/data` - 資料處理

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

### `kgat_attention.py` (Full KGAT)

實作具備關係感知注意力 (Relation-Aware Attention) 的 KGAT 模型。

#### class `KGATAttention(nn.Module)`
*   **核心架構**: 包含 `RelationalGraphAttentionLayer` 堆疊，利用 `scatter_reduce` 實作 Shifted Softmax。
*   **關鍵方法**:
    *   `forward(users, pos_items, neg_items)`: 執行 BPR 損失計算。
    *   `get_final_embeddings()`: 提取所有 User/Item 的最終隱含向量。

### `bpr_mf.py` (Traditional Baseline)
實作基於 BPR 損失的傳統矩陣分解模型。
*   **關鍵方法**: `forward(users, pos_items, neg_items)` 返回 pairwise 分數。

### `lightgcn.py` (Pure GNN Baseline)
實作 LightGCN 架構，僅進行線性圖卷積聚合，無非線性轉換。
*   **參數**: `--layers` 指定卷積層數（建議 3-4 層）。

### `kgat_bi_interaction.py` (KGAT w/o Attention)

退化版的 KGAT 模型。

#### class `KGAT_BiInteraction(nn.Module)`
將 Attention 拔除，權重改為固定常數 $1/|N_h|$。
*   **優化**:
    *   GNN 聚合由傳統矩陣乘法替換為 PyTorch 原生 `index_add_`，跨越 IPEX Sparse API 不支援的效能瓶頸，速度飛升。
    *   同樣包含 `get_final_embeddings()`。

### `nfm.py` (Feature Interaction Baseline)
實作 Neural Factorization Machine，利用 Bi-Interaction 層捕捉二階特徵交互。

---

## `src` - 實驗啟動腳本 (Scripts)

### `train_att.py`
*   **功能**: 訓練 Full KGAT (支援單層至多層)、退化版對照組 (w/o KG)。
*   **參數**: `--epochs`, `--layers`, `--without_kg`, `--use_bf16`。
    *   `--epochs`: 回合數 (預設: 30，Ablation 實驗中調整為 10)。
    *   `--layers`: 自訂 GNN 層數 (例如 `--layers 64 64` 即為 L=2 模型)。
    *   `--without_kg`: 剔除知識圖譜，進入純 CF 對照模式。
    *   `--use_bf16`: 強制啟動 BFloat16 原生加速。
    *   `--no_compile`: 避開 `torch.compile` 與稀疏運算圖的不相容問題。

### `train_bi_interaction.py`

*   **功能**: 獨立訓練僅具備 Bi-Interaction 聚合但無 Attention 權重分配的退化模型。
*   **參數**:
    *   (同 `train_att.py` 大多數參數)

### `train_baseline.py`
*   **功能**: 統一訓練對照組模型 (BPR-MF, LightGCN, NFM)。

### `evaluate_fidelity.py`
*   **功能**: 對於已訓練的 KGAT 模型執行解释路徑萃取與 Fidelity 指標計算。
*   **參數**:
    *   `--model_path`: 檢查點檔案路徑。
    *   `--user_ids_file`: 存放受測使用者 ID 的 JSON 檔案。
    *   `--top_k_paths`: 每對 User-Item 萃取的解釋路徑數量 (預設: 3)。
    *   `--output_metrics`: 儲存 Fid+, Fid- 結果的 JSON 路徑。
*   **輸出指標**:
    *   `avg_fidelity_plus`: 移除路徑後的預測降幅。
    *   `avg_fidelity_minus`: 僅留路徑後的預測降幅。

---

## 實驗自動化批次檔

### `run_experiments.bat`
*   **功能**: 一鍵啟動 5 組消融實驗排程。

### `run_baseline_experiments.bat`
*   **功能**: 一鍵啟動 3 組對照組模型排程。


## File: docs\architecture.md

# 專案實驗架構與模組設計

本專案旨在重新實作與驗證知識圖譜注意力神經網路 (Knowledge Graph Attention Network, KGAT) 在食譜推薦上的效能。歷經多次重構後，目前的系統架構專為「嚴謹對照原論文」與「最大化 Intel XPU 硬體效能」而打造。

## 1. 原論文採用的部分 (Paper Alignment)

為了能在消融實驗中給出具備說服力的對比基準，我們在核心模組中嚴格對齊了 [KGAT (Wang et. al, 2019)](https://arxiv.org/abs/1909.02695) 的理論架構：

* **Relation-Aware Attention 機制**: 
  - 捨棄了一般的 GAT 節點對接。
  - 完全實作 $\pi(h,r,t) = (W_r e_t)^\top \tanh(W_r e_h + e_r)$，這使得注意力權重能夠強烈感知不同邊（如：Ingredient 關係 vs Tag 關係）的重要性。
* **Bi-Interaction 聚合公式**:
  - GNN 訊息傳遞同時包含節點與鄰居的相加 ($e_u + e_v$) 與元素級相乘 ($e_u \odot e_v$) 並經由線性轉換與 LeakyReLU 激勵函式。
* **BPR Loss 與 L2 正則化 (Weight Decay)**:
  - 以成對比較 (Pairwise) 的 Bayesian Personalized Ranking 函數指導訓練。
  - 加入 $L_2$ 正則化 (我們設定為 $10^{-5}$) 防範過擬合。
* **Message Dropout**:
  - 在 GNN 的每層訊息傳遞後與注意力權重上，皆套用 `nn.Dropout(p=0.1)` 以提升深層圖網路的抗噪能力。

---

## 2. 為個人訓練優化與修改的部分 (Training Modifications)

由於原論文的架構在有限的硬體 (如 8GB VRAM 的 Intel Arc A750) 上極易發生資源枯竭與訓練速度低落，我們實施了以下大幅度的在地化改動：

* **全 XPU、BFloat16 混合精度訓練**:
  - 放棄雲端 Colab，全面轉向本地端 Intel Extension for PyTorch (IPEX) 支援的 XPU 訓練。
  - 使用 BFloat16 將記憶體消耗減半，使訓練規模得以擴大。
* **反向傳播底層替換 (`index_add_`)**:
  - 在 XPU 上，原生的 Python Indexing 操作或部分 Sparse 乘法容易崩潰或 Fallback 至 CPU。我們全面換用基礎且效能極快的 Tensor operation `out.index_add_(0, edge_index, message)`。
* **快取推論機制 (`get_final_embeddings`)**:
  - 傳統的推薦預測需要在每個 testing batch 中走一遍龐大的 GNN Forward。我們改變策略，在 Validation/Test 階段開始前，僅呼叫**一次** GNN 得出所有 Nodes 的終極特徵 (Embeddings)，後續的 Recall 運算僅作簡單的 Index 取出與內積，測試時間因此從十分鐘銳減至不到 10 秒。
* **使用 Activation Checkpointing 挑戰深層 (L=3) 極限**:
  - 由於 Relation-Aware Attention 需要為圖上的「每一條邊」製造臨時的關聯向量矩陣，一旦疊加 3 層會輕易突破 16GB 顯存。我們引進 PyTorch `checkpoint` 技術，在 Forward 時不保留記憶範圍，強迫 Backward 時重算，最終成功在一般硬體上解鎖深層網路訓練。
* **捨棄 KGE (Knowledge Graph Embedding) Joint Training**:
  - 原論文設計模型需同時學習 TransR (圖結構任務) 與 CF (協同過濾任務)。為了讓消融實驗更為乾淨純粹、僅對比圖卷積本身的影響，我們拔除了 KGE 輔助優化，只單一依賴 BPR Loss。

---

## 3. 消融實驗架構設計 (Ablation Study Architecture)

為了科學驗證模組有效性，專案內置了 5 款對照實驗組，可由 `run_experiments.bat` 自動派發執行：

### 實驗模塊總表
1. **Full KGAT (基準, L=1)**: `train_att.py`
   - 同時具備 Attention 機制與 Bi-Interaction 的完整版。
2. **w/o Attention (KGAT-a, L=1)**: `train_bi_interaction.py`
   - 將 Attention 權重退化為平均權重 (Mean Pooling)，但保留 Bi-Interaction。
   - **目的**: 驗證「注意力分配」是否為增進推薦效能的核心。
3. **w/o Knowledge Graph (L=1)**: `train_att.py --without_kg`
   - 移除所有的 Recipe-Ingredient, Recipe-Tag 邊，模型退化為僅依賴 User-Item 互動的普通圖神經推薦。
   - **目的**: 驗證「給系統注入外部知識」的實際效益。
4. **Depth Variation (L=2)**: `train_att.py --layers 64 64`
   - 將 GNN 深度推展至 2 跳 (2-hop)。
   - **目的**: 觀察遠鄰居 (例如，與同一個 tag 相關的其他食譜) 是否帶來正面幫助。
5. **Depth Variation (L=3)**: `train_att.py --layers 64 64 64`
   - 將 GNN 深度推展至 3 跳 (3-hop)。
   - **目的**: 探索神經網路極限，測試是否發生 Oversmoothing (過度平滑導致特徵無法區分)。
6. **對照組實驗 (Baseline Matrix)**: `train_baseline.py`
   - 包含 BPR-MF, NFM, LightGCN 三款模型。
   - **目的**: 建立與經典 Matrix Factorization、Feature Interaction 模型以及純 GNN 模型的對比基準，驗證 KGAT 的綜合效能優勢。

---

## 4. 可解釋性驗證框架 (Explainability Framework)

除了推薦效能（Recall, NDCG），本專案強調對於推薦原因的量化驗證：
* **Attention 為基的解釋器**: 利用模型學習到的關係感知注意力權重，搜尋模型路徑中貢獻度最高的解釋路徑。
* **Fidelity 指標**: 
  - **Fidelity+**: 通過「遮擋 (Occlusion)」解釋路徑來觀察模型分數的下降程度，驗證路徑的 **必要性**。
  - **Fidelity-**: 通過「僅保留 (Sufficiency)」解釋路徑來觀察模型是否仍能維持預測，驗證路徑的 **充分性**。
* **評估工具**: `src/evaluate_fidelity.py` 整合了路徑搜尋與機率變化運算。

### 專案目錄分佈
```
Experiment/
├── data/
│   ├── raw/                # 原始資料 CSV
│   └── processed/          # 預處理後的圖譜檔案 (.pkl)
├── docs/                   # ADR 與架構文檔
├── models/                 # 實驗訓練好的權重模型
│   └── baseline/           # 對照組模型權重
├── output/                 # 產出的各種 Metrics Logs
│   └── logs/
│       └── baseline/       # 對照組訓練日誌
├── src/                    # 原始程式碼
│   ├── data/               # 資料預處理
│   ├── model/              # 模型定義 (kgat_bi_interaction.py, kgat_attention.py, bpr_mf.py, nfm.py, lightgcn.py)
│   ├── train_att.py        # 包含 Attention 架構的訓練腳本
│   ├── train_bi_interaction.py # 僅 Bi-Interaction 的退化訓練腳本
│   ├── train_baseline.py   # 對照組模型統一訓練腳本
│   └── evaluate_fidelity.py # 可解釋性與 Fidelity 評估腳本
├── run_experiments.bat     # 消融實驗自動化啟動腳本
└── run_baseline_experiments.bat # 對照組實驗自動化啟動腳本
```


## File: docs\data_dictionary.md

# 資料字典 (Data Dictionary)

本文件詳細說明 Food.com 資料集的欄位定義，以及經由預處理後產生的知識圖譜結構。

## 原始資料 (Raw Data)

原始資料來自 Kaggle Food.com Recipes and Interactions。請確保檔案存放於 `data/raw/`。

### `RAW_recipes.csv`

包含食譜的詳細資訊。

| 欄位名稱 | 類型 | 說明 |
| :--- | :--- | :--- |
| `id` | int | 食譜唯一識別碼 (原始 ID) |
| `name` | string | 食譜名稱 |
| `minutes` | int | 烹飪時間 (分鐘) |
| `submitted` | date | 上傳日期 |
| `tags` | list (str) | 標籤列表 (如 ['60-minutes-or-less', 'time-to-make', ...]) |
| `nutrition` | list (float) | 營養成分 (cal, fat, sugar, sodium, protein, sat. fat, carbs) |
| `n_steps` | int | 步驟數量 |
| `steps` | list (str) | 烹飪步驟描述 |
| `description` | string | 使用者提供的食譜描述 |
| `ingredients` | list (str) | 成分列表 (如 ['winter squash', 'mexican seasoning', ...]) |
| `n_ingredients`| int | 成分數量 |

### `RAW_interactions.csv`

包含使用者對食譜的評分與評論。

| 欄位名稱 | 類型 | 說明 |
| :--- | :--- | :--- |
| `user_id` | int | 使用者唯一識別碼 (原始 ID) |
| `recipe_id` | int | 食譜 ID (對應 `RAW_recipes.csv` 的 `id`) |
| `date` | date | 互動日期 |
| `rating` | int | 評分 (1-5) |
| `review` | string | 文字評論 |

---

## 預處理資料 (Processed Data)

預處理腳本 `src/data/preprocess.py` 會產生以下 Pickle 檔案，存放於 `data/processed/`。

### 1. `interactions.pkl` (DataFrame)

用於模型訓練的使用者-物品互動矩陣。

| 欄位 | 說明 |
| :--- | :--- |
| `user_id_remap` | [0, n_users) 的連續整數 ID |
| `recipe_id_remap` | [0, n_items) 的連續整數 ID |
| `rating` | 原始評分 |

### 2. `kg_triples.pkl` (Numpy Array)

知識圖譜的三元組 `(Head, Relation, Tail)`。

*   **Head**: 食譜 ID (`recipe_id_remap`)
*   **Relation**: 關係類型 ID
    *   `0`: **Has_Ingredient** (食譜包含某成分)
    *   `1`: **Has_Tag** (食譜擁有某標籤)
*   **Tail**: 實體 ID (Entity ID)
    *   實體 ID 範圍從 `0` 開始編號。
    *   成分 (Ingredients) 與標籤 (Tags) 共享同一個 ID 空間，但彼此 ID 不重疊。

### 3. `stats.pkl` (Dict)

儲存統計資訊與映射表，用於推論時還原原始資訊。

*   `n_users`: 使用者總數
*   `n_items`: 物品 (食譜) 總數
*   `n_entities`: 知識圖譜實體 (成分+標籤) 總數
*   `user_map`: `sklearn.preprocessing.LabelEncoder` 物件 (User ID 轉換)
*   `item_map`: `sklearn.preprocessing.LabelEncoder` 物件 (Recipe ID 轉換)
*   `ingredient_map`: `Dict[str, int]` (成分名稱 -> Entity ID)
*   `tag_map`: `Dict[str, int]` (標籤名稱 -> Entity ID)


## File: docs\development.md

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


