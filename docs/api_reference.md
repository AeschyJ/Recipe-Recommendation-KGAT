# API 參考文件 (API Reference)

本文件提供 `src/` 目錄下數據前處理、模型定義、訓練進入點、XAI 評估模組及 `scripts/` 工具腳本之詳細 API 規格說明。

---

## 1. 資料處理模組 (`src/data/`)

### `src/data/preprocess.py`
負責 Food.com 原始 CSV 資料清洗、極端頻率標籤與食材降噪、ID 重新編號與協同知識圖譜 (CKG) 構建。

#### `process_data(data_dir)`
* **功能**：執行完整資料前處理流程。
* **參數**：
  * `data_dir` (*str*): 包含 `data/raw/` 的專案目錄路徑。
* **產出** (存於 `data/processed/`)：
  * `interactions.pkl`: 包含 `[user_id_remap, recipe_id_remap, rating]` 的 Pandas DataFrame / NumPy 陣列。
  * `kg_triples.pkl`: `(N_triples, 3)` 之 NumPy 陣列 `[head_id, relation_id, tail_id]`。
  * `stats.pkl`: 包含 `n_users`, `n_items`, `n_entities`, `n_relations` 與 ID LabelEncoders 字典。

---

## 2. 模型架構模組 (`src/model/`)

### `src/model/kgat.py`
包含完整 KGAT 模型與消融退化版模型。

#### Class `KGATAttention(nn.Module)`
* **說明**：具備關係感知注意力 (Relation-Aware Attention) 與 Bi-Interaction 的完整 KGAT 模型。
* **主要參數**：
  * `n_users` (*int*), `n_items` (*int*), `n_entities` (*int*), `n_relations` (*int*): 圖譜節點與關係數量。
  * `emb_dim` (*int*): Embedding 向量維度 (預設: 64)。
  * `layer_dims` (*list[int]*): 各 GNN 層通道維度 (預設: `[64]`)。
  * `mess_dropout` (*float*): Message Dropout 比率 (預設: 0.1)。
* **關鍵方法**：
  * `forward(users, pos_items, neg_items)`: 前向計算正樣本與負樣本之 BPR 預測分數。
  * `get_final_embeddings()`: 執行單次 GNN 傳播，回傳全圖用戶與實體之最終拼接隱含向量 (用於極速 Validation/Testing 評估)。

#### Class `KGAT_BiInteraction(nn.Module)`
* **說明**：KGAT w/o Attention 消融退化模型。使用平均權重與 `index_add_` 原子寫入。
* **關鍵方法**：
  * `forward(users, pos_items, neg_items)`: 計算 BPR 損失。
  * `get_final_embeddings()`: 快取特徵提取。

### `src/model/bpr_mf.py`
#### Class `BPRMF(nn.Module)`
* **說明**：傳統矩陣分解 (Matrix Factorization) 協同過濾基準，無圖結構。

### `src/model/lightgcn.py`
#### Class `LightGCN(nn.Module)`
* **說明**：純圖卷積協同過濾模型，使用二分圖與對稱正規化鄰接矩陣。

### `src/model/nfm.py`
#### Class `NFM(nn.Module)`
* **說明**：Neural Factorization Machine，於二階 Bi-Interaction 池化層後接多層 MLP。

### `src/model/explainer_attention.py`
#### Class `KGATAttentionExplainer`
* **說明**：抽取 `KGATAttention` 多層注意力權重，建構 NetworkX 圖並搜尋 User $\to$ Recipe 的 Top-K 高貢獻解釋路徑。

### `src/model/explainer.py`
#### Class `KGATExplainer`
* **說明**：基於梯度顯著性 (Gradient Saliency) 的圖神經網路解釋器，計算 $\left| \frac{\partial \hat{y}}{\partial A_{i,j}} \right|$ 作為邊重要性。

---

## 3. 實驗進入點 (Training & Evaluation CLI)

### `src/train.py`
KGAT 消融實驗主訓練腳本。
* **重要 CLI 參數**：
  * `--no_attention`: 布林旗幟，啟用 Bi-Interaction 聚合器替代 Relation-Aware Attention (消融實驗)。
  * `--layers`: GNN 各層維度，如 `--layers 64` ($L=1$), `--layers 64 64` ($L=2$), `--layers 64 64 64` ($L=3$)。
  * `--without_kg`: 布林旗幟，剔除 CKG 知識三元組進入純 CF 對照模式。
  * `--use_bf16`: 啟用 BFloat16 混合精度。
  * `--epochs`: 訓練回合數 (預設: 20)。
  * `--model_dir`: 檢查點輸出目錄 (預設: `models`)。
  * `--resume`: 載入指定 `.pth` 檢查點繼續訓練。

### `src/train_baseline.py`
Baseline 對照組模型訓練腳本。
* **重要 CLI 參數**：
  * `--model`: 指定模型名稱，可選 `BPR-MF`, `LightGCN` 或 `NFM`。
  * `--layers`: 層維度。
  * `--model_dir`: 檢查點輸出目錄 (預設: `models/baseline`)。

### `src/evaluate_fidelity.py`
XAI Fidelity+ / Fidelity- 量化計算腳本。
* **重要 CLI 參數**：
  * `--model_path`: 欲評估之 `.pth` 檢查點檔案路徑。
  * `--user_ids_file`: 採樣測試使用者 JSON 檔案路徑。
  * `--output_explain`: 解釋路徑 JSON 輸出路徑 (預設: `output/fidelity/explanations.json`)。
  * `--output_metrics`: Fidelity 指標 JSON 輸出路徑 (預設: `output/fidelity/metrics.json`)。
  * `--top_k_paths`: 每對 User-Item 提取之解釋路徑數 (預設: 3)。

---

## 4. 輔助工具與維護腳本 (`scripts/`)

* **`scripts/evaluate_all.py`**：遍歷 `models/` 目錄下的檢查點，對測試集進行統一評估。
* **`scripts/sample_users_for_xai.py`**：從測試集中採樣指定數量 (如 500 位) 的 Target Users 用於 XAI 評估。
* **`scripts/analyze_logs.py`**：解析訓練日誌並產出 Best Metrics 對比報告。
* **`scripts/reformat_logs.py`**：清洗訓練日誌字元雜訊。
* **`scripts/simplify_output_data.py`**：過濾與簡化 XAI 解釋路徑數據供 LLM 閱讀或論文繪圖。
* **`scripts/visualize_paths.py`**：將提取的 CKG 解釋路徑視覺化為樹狀圖或關係文字。
* **`scripts/create_backup_zips.py`**：將權重與 LaTeX 資源封裝為本地備份壓縮檔 (已設定 Git 排除)。
* **`scripts/cleanup_checkpoints.py`**：自動清理歷史中間 Epoch 檢查點，僅保留最佳模型以釋放硬碟空間。
