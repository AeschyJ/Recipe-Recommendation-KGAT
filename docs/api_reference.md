# API 參考文件

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

### `kgat_attention.py` (Full KGAT)

實作具備關係感知注意力 (Relation-Aware Attention) 的 KGAT 模型。

#### class `KGATAttention(nn.Module)`
模型主體，繼承自 `torch.nn.Module`。
*   **參數**:
    *   `n_users`, `n_entities`, `n_relations`: 圖譜實體總數。
    *   `layers`: GNN 每層輸出維度參數 (如 `[64]`, `[64, 64]`)。
*   **`forward(...)`**: 傳入 Positive/Negative 樣本，單次傳播計算損失所需的內積預測分數。
*   **優化**:
    *   實作 $\pi(h,r,t) = (W_r e_t)^\top \tanh(W_r e_h + e_r)$ 公式。
    *   使用 PyTorch `checkpoint` 技術支援 `L=3` 深層運算不 OOM。
*   **`get_final_embeddings()`**: 在推論前將所有實體與使用者特徵提前計算快取。

### `kgat_bi_interaction.py` (KGAT w/o Attention)

退化版的 KGAT 模型。

#### class `KGAT_BiInteraction(nn.Module)`
將 Attention 拔除，權重改為固定常數 $1/|N_h|$。
*   **優化**:
    *   GNN 聚合由傳統矩陣乘法替換為 PyTorch 原生 `index_add_`，跨越 IPEX Sparse API 不支援的效能瓶頸，速度飛升。
    *   同樣包含 `get_final_embeddings()`。

---

## `src` - 實驗啟動腳本 (Scripts)

### `train_att.py`

*   **功能**: 訓練 Full KGAT (支援單層至多層)、退化版對照組 (w/o KG)。
*   **參數**:
    *   `--epochs`: 回合數 (預設: 30，Ablation 實驗中調整為 10)。
    *   `--layers`: 自訂 GNN 層數 (例如 `--layers 64 64` 即為 L=2 模型)。
    *   `--without_kg`: 剔除知識圖譜，進入純 CF 對照模式。
    *   `--use_bf16`: 強制啟動 BFloat16 原生加速。
    *   `--no_compile`: 避開 `torch.compile` 與稀疏運算圖的不相容問題。

### `train_bi_interaction.py`

*   **功能**: 獨立訓練僅具備 Bi-Interaction 聚合但無 Attention 權重分配的退化模型。
*   **參數**:
    *   (同 `train_att.py` 大多數參數)
    *   此版本架構簡潔，在 XPU 下具備最傲人的前/反向傳播效能。

### `run_experiments.bat`

*   **功能**: 一鍵消融實驗排程腳本。包含前五大對照組 (Ablation configurations)，會透過內建虛擬環境呼叫 `python` 自動走完排程、掛載 `logs`，免除手動啟動的困擾。
