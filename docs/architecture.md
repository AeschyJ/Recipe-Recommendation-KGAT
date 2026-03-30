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

### 專案目錄分佈
```
Experiment/
├── data/
│   ├── raw/                # 原始資料 CSV
│   └── processed/          # 預處理後的圖譜檔案 (.pkl)
├── docs/                   # ADR 與架構文檔
├── models/                 # 實驗訓練好的權重模型
├── output/                 # 產出的各種 Metrics Logs
├── src/                    # 原始程式碼
│   ├── data/               # 資料預處理
│   ├── model/              # 模型定義 (kgat_bi_interaction.py, kgat_attention.py)
│   ├── train_att.py        # 包含 Attention 架構的訓練腳本
│   └── train_bi_interaction.py # 僅 Bi-Interaction 的退化訓練腳本
├── run_experiments.bat     # 消融實驗自動化啟動腳本
```
