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

---

## 數據整理與分析腳本 (`scripts` 與 `output`)

### `scripts/evaluate_all.py`
*   **功能**: 掃描 `models/` 與 `models/baseline/` 底下所有的 Checkpoint，統一載入並對測試集進行 HR@K, NDCG@K, Precision@K 等指標評估。

### `scripts/compare_metrics.py`
*   **功能**: 讀取實驗產生的 logs/metrics 檔案，將不同模型在相同指標下的成績進行比較，可輸出彙整後的表格以便分析。

### `scripts/reformat_logs.py`
*   **功能**: 解析並清洗原始訓練日誌中的雜訊字元，轉化為便於後續視覺化與表格轉換的乾淨結構。

### `output/analyze_xai.py`
*   **功能**: 對於產生的 `explanations.json` 進行進階統計分析，例如各類路徑（User->Item, Item->Tag 等）的佔比。

### `output/simplify_output_data.py`
*   **功能**: 簡化與過濾解釋數據，將過長或複雜的推論結果精煉成能直接放入論文附錄的格式。

### `models/update_models_list.py`
*   **功能**: 遍歷 `models/` 目錄並更新/列出可用模型檔案清單狀態。
