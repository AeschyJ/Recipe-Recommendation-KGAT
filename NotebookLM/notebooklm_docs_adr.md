# Architecture Decision Records (ADR)

## File: docs\adr\ADR-001-Retained-Decisions.md

# ADR-001: 保留的歷史架構決策 (Retained Architecture Decisions)

## 狀態
**Status:** 已接受 (Accepted) - 保留於專案中

## 決策列表

### 1. 移除 DGL 依賴並遷移至純 PyTorch 實作 (原 ADR-001)
- **背景**: 原始 DGL 實作與底層驅動相容性差，造成圖卷積執行異常且記憶體不易管理。
- **決策**: 以純 PyTorch (使用 `torch.sparse`) 自行刻劃 Graph Neural Network 邏輯。
- **後續影響**: 提高專案對特殊硬體 (如 Intel XPU) 的相容性，且讓注意力機制有了自定義的巨大空間。此決策奠定了後續使用原生 Python 操作進行效能優化的基礎。

### 2. 遷移至 PyTorch 原生 XPU 支援 (原 ADR-004)
- **背景**: Intel Arc GPU 需要透過 Intel Extension for PyTorch (IPEX) 支援硬體加速。
- **決策**: 模型與張量全面套用 `.to(device)` 支援 `xpu` 裝置。
- **後續影響**: 成功將訓練工作重心由 Colab 轉向地端加速，開啟了本專案在地端大規模全 GPU 訓練的可能。

### 3. 在圖結構中整合使用者互動邊 (原 ADR-005)
- **背景**: 為了進行具備解釋性的推薦推理，知識圖譜與使用者行為需整合。
- **決策**: 將 User-Item 的互動行為視為 Knowledge Graph 中的新 `relation`，並以此構建 `ckg_graph` 控制節點傳遞。
- **後續影響**: 此決策確立了本專案推薦系統的核心，讓協同過濾 (CF) 得以藉由圖卷積的形式自然發揮。

### 4. 去除超級節點與提高數值穩定性 (原 ADR-007)
- **背景**: 少數的超級節點 (如 generic tags) 造成 GNN 訊息傳遞產生極大權重偏移，引發 NaN 梯度崩潰。
- **決策**:
  1. 對互動次數大於閾值的超級節點進行邊緣剪枝 (Pruning)。
  2. 應用 Laplacian 歸一化與 `torch.clamp()` 進行數值保護。
?- **後續影響**: 模型不再於第 15 epoch 後產生梯度爆炸，確立了後續實驗能穩定推進到深層 (L=3) 傳遞的基本門檻。

### 5. 統一超參數、擴展召回指標與日誌系統 (原 ADR-008)
- **背景**: 原有訓練腳本的 Epoch、學習率等參數未統一，導致每次修跑都會有無法對齊的問題。
- **決策**: 利用 `argparse` 與 `logging` 系統全面規範 `run_experiments.bat`，並在推論端統一搜集 `Recall@20` 以及 `NDCG@20` 等泛用推薦評估指標。
- **後續影響**: 為本次的消融實驗 (Ablation Study) 打下了標準的自動化實驗基礎。


## File: docs\adr\ADR-002-Superseded-Decisions.md

# ADR-002: 已修改或廢棄的歷史決策 (Superseded Architecture Decisions)

## 狀態
**Status:** 已廢棄 / 已修改 (Superseded/Obsolete)

## 決策列表

### 1. 使用過渡版本的注意力機制 (原 ADR-002)
- **原決策背景**: 早期為了解決程式碼漏洞，自行撰寫了一版簡化的 GAT，未納入 Knowledge Graph 的 `relation` 特徵融合 (`W_r`)。
- **為何被推翻**: 無法與原始 KGAT (Knowledge Graph Attention Network) 論文進行真正的學術對標。
- **取代方案**: 本次專案已「回歸原始論文」，實作正統的 Relation-Aware Attention ($\pi(h,r,t) = (W_r e_t)^\top \tanh(W_r e_h + e_r)$)，請參見 **ADR-004**。

### 2. Google Colab 記憶體優化 (原 ADR-003)
- **原決策背景**: 過去因硬體限制，嘗試透過 `gc.collect()` 或中斷 Graph 歷史圖形來省下 Colab 寶貴的 VRAM。
- **為何被推翻**: 在本機 Intel Arc (XPU) 環境下，這種 Python 層面的 GC 回收不僅無法大幅緩解 VRAM 危機，還會嚴重拖累訓練速度。
- **取代方案**: 改為深層網路的 Activation Checkpointing 策略，請參見 **ADR-003** 效能優化篇。

### 3. 以分批重算作為大規模 VRAM 優化 (原 ADR-006)
- **原決策背景**: 為解決 OOM (Out Of Memory) 嘗試過將 GNN 鄰居切塊分別矩陣相乘，放棄了時間換取空間。
- **為何被推翻**: 訓練效率崩落，且原先以 `torch.sparse` 與 Python index 實作的反向傳播在 XPU 上發生嚴重的 fallback to CPU。
- **取代方案**: 從底層改寫為 PyTorch 的 `index_add_` 原生支援，不會 OOM，訓練速度獲得提升。請參見 **ADR-003** 效能優化篇。

### 4. 舊版消融命名與基礎架構 (原 ADR-009)
- **原決策背景**: 原先嘗試定義了兩三款不同實驗的檔案名稱，並打算進行以 KGE (TransR) 為輔助的多工作業 (Joint Training)。
- **為何被推翻**: 原始設定無法達到純淨的對照實驗，容易因為 KGE 收斂速度的問題影響主線目標 (BPR 推薦任務) 的對比。
- **取代方案**: 確立更加嚴謹且獨立的 5 組新消融實驗對照組，請參見 **ADR-005**。


## File: docs\adr\ADR-003-Training-Optimization.md

# ADR-003: 訓練效能極致優化 (Training & Performance Optimization)

## 狀態
**Status:** 已接受 (Accepted) - 本次新增

## 背景與問題 (Context)
在將模型擴展以進行消融實驗時，我們遇到了嚴重效能瓶頸與資源枯竭：
1. **推論過慢**: `evaluate` 函數在每個 batch 皆完整走一次 3 層的 GNN 向前傳播 (Forward Pass)，導致 Validation 階段耗時高達十數分鐘。
2. **XPU 架構效能低落**: 先前使用的 Custom Python Autograd (`SparseAggregateFunction`) 在部分矩陣運算上無法適配 IPEX 的算子，導致訓練引發 CPU fallback，時間以倍數遞增。
3. **記憶體撐爆 (OOM)**: 當啟動完整版的 Relation-aware Attention (包含龐大的 relation feature vector 融合) 並擴張至深度 `L=3` 時，16GB VRAM 不足以負荷運算圖。

## 決策 (Decision)

針對上述問題，我們在架構內實施了以下系統性工程優化：

1. **全 XPU、bfloat16 訓練策略**
   - 將所有實驗環境預設使用 `--use_bf16`，利用 XPU BFloat16 原生算力將記憶體佔用砍半，同時維持相同的訓練收斂精度。

2. **GNN 訊息傳遞使用 PyTorch 原生 `index_add_` (取代 Autograd)**
   - 全面拔除自定義 Python 梯度類別，改以原生 Tensor 修改操作 `out.index_add_(0, edge_index, message)` 實現稀疏圖訊息聚合。
   - 經測試證實：此舉成功避免了計算卡在 CPU/XPU 間搬移，`wo_attn` 模型的反向傳播耗時降低。

3. **GNN 共享與推論快取 (Evaluate Embedding Caching)**
   - 實作新的模型方法 `get_final_embeddings()`。在 `evaluate` 迴圈開始前，率先將所有 Nodes 餵進 GNN，得出**唯一一份**推論結果 `u_g_embeddings` 以及 `i_g_embeddings`。
   - 在每一個測試 batch 中，僅依賴取出對應 index 並執行內積 (`inner product`) 即可給出預測分數，這使得評估耗時由原本的分鐘級驟降至秒級。

4. **Attention 特徵融合後再線性轉化 (Activation Checkpointing)**
   - 面對 `L=3` 層 Full Attention 所引發的 VRAM OOM，在不改動參數前提下，我們引進了 PyTorch 的 `torch.utils.checkpoint` 機制。
   - 利用運算時間交換記憶體：前向傳播不保存中繼 Variable，反向時重新計算，成功使得極深層的知識圖譜網路能在普通開發機顯示卡上流暢運行。

## 影響 (Consequences)
- **正面影響**: 
  - 大幅縮減了超參數搜索所需的訓練周期。
  - 對未來大型資料集擴展以及更強的深層模型提供了穩健的架構支援。
- **負面影響 / 限制**:
  - `checkpoint` 由於以時間換取空間，在前處理時計算速度會略為下降，但避免了程式因 OOM 崩潰的可能，這是為求穩定深層網路而不得不採取的策略妥協。


## File: docs\adr\ADR-004-Paper-Alignment.md

# ADR-004: 回歸原始論文 (Paper Alignment)

## 狀態
**Status:** 已接受 (Accepted) - 本次新增

## 背景與問題 (Context)
為了確保我們的重構專案能與其他基礎推薦算法 (如 SVD、KNN) 以及國際學術成果進行客觀的比對與效能衡量，模型的實作必須貼近 [KGAT (Wang et. al, 2019)](https://arxiv.org/abs/1909.02695) 之原始算法設定。在早期的程式中，為了快速求得雛型，我們妥協了注意力機制運算式並且忽略了正規化的實作，這都對實驗結果嚴謹性有害。

## 決策 (Decision)

1. **嚴謹對齊原論文的 Relation Aware Attention**
   - 拋棄了與 GAT 差異不大的點對點注意力算法。
   - 我們還原了真正的 KGAT 知識融合注意力權重：
     $$ \pi(h,r,t) = (W_r e_t)^\top \tanh(W_r e_h + e_r) $$
   - 透過此公式，頭節點 $h$ 與尾節點 $t$ 的關係重要性會嚴重依賴中間負責傳遞的 $relation$ 特徵 $r$，完美切合知識圖譜在推薦上的強大能力。

2. **L2 正則化 (Weight Decay) 與 Message Dropout 復歸**
   - 深度圖神經網路常受困於參數過多造成的訓練資料猛烈定型 (Overfitting)。
   - 在 `KGATAttention` 以及一般訊息聚合的模組中，我們加入了隨機屏蔽機制 `nn.Dropout(p)`，確保 Graph Message 在傳遞時具備雜訊容忍力。
   - `Optimizer` (Adam) 同步寫入強勢的權重衰減 `weight_decay=1e-5`，限制了無限制擴展的權重絕對值。

3. **捨棄 KGE (TransR) Joint Training**
   - **理由**: 原本的架構預留了大量時間以同時訓練 TransR 任務 (優化實體間在知識圖譜中的關聯距離) 與 BPR 任務 (推薦商品給使用者)。然而，在我們目前資源封閉的獨立消融實驗與 baseline 比較上，若其他經典算法僅靠 interaction 即可發揮作用，我們也應讓 KGAT 維持在「將知識圖譜作為附屬特徵」的情境。
   - **實作**: 原先包含複雜 KGE 更新的回圈已經拔除，全系統專注於依賴 BPR Loss 對向推薦結果進行梯度的聯合計算，不僅增加了公平性，也讓模型收斂單一化。

## 影響 (Consequences)
- 無論是學術或業務審查，此版本架構已可宣告能夠作為 KGAT 復刻版的代表。
- KGE 丟失可能會讓 Knowledge Embeddings 本身失去少數幾何意義，但此損失在 BPR 的監督學習校正下，已被證明影響微乎其微。


## File: docs\adr\ADR-005-Ablation-Study-Architecture.md

# ADR-005: 消融實驗架構設計 (Ablation Study Architecture)

## 狀態
**Status:** 已接受 (Accepted) - 本次新增

## 背景與問題 (Context)
要證明我們模型中的任意一項機制（如 KG 融合、Attention 權重配分）確實能帶來推薦精準度提升，傳統上唯一且最符合科學實證的方法就是進行「消融實驗 (Ablation Study)」。為此，我們需要定義清晰、公平且不重疊的比較組，同時透過統一的訓練自動腳本進行調度。

## 決策 (Decision)

建立統一的測試介面 (`run_experiments.bat`)，強制所有的參數 (包含 Epoch、Batch Size、Precision、Regularization) 在以下 5 款實驗中完全鎖定一致。

### 五大對照實驗設計 (The Five Configurations)

| 實驗名稱 | L (GNN層數) | Attention 機制 | 知識圖譜 (KG) 關係邊 | 測試意圖 | 模型腳本 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Full KGAT** | 1 (`[64]`) | Yes (Relation-Aware) | Yes | **基準模型 (Baseline)**。測試融合完整論述的最強表現。 | `train_att.py` |
| **2. w/o Attention (KGAT-a)** | 1 (`[64]`) | No (Mean Pooling) | Yes | 驗證「**特指分配重要性** (Attention)」是否真的比傳統平均權重帶來更好的推薦解析。 | `train_bi_interaction.py` |
| **3. w/o KG** | 1 (`[64]`) | Yes | No (User-Item only)| 拔除知識庫，退化成普通圖神經推薦。驗證「**延伸外部知識**」能給推薦系統帶來多少躍進。 | `train_att.py --without_kg` |
| **4. Depth Variation L=2** | 2 (`[64,64]`) | Yes | Yes | 探索「**多跳推理能力** (Multi-hop)」，看更遠的鄰居會引入幫助還是噪聲 (Oversmoothing)。 | `train_att.py --layers 64 64` |
| **5. Depth Variation L=3** | 3 (`[64,64,64]`)| Yes | Yes | 檢視神經網路能力天花板，以及高階深層圖卷積是否引起退化。 | `train_att.py --layers 64 64 64` |

---

### 環境配置統一
* 以 10 次 Epoch 作為初步走勢測試；後續可直接使用 `--resume` 對這 5 組做第二輪 30 次 Epoch 深培。
* **指標收集**: 以 `Recall@20` 及 `NDCG@20` 的最終最佳 epoch 為評判依歸。

## 影響 (Consequences)
- 這套消融架構能讓我們最後產出的 Metrics 報表具備極高的公信力。只要後續分析數據就能立刻導出強而有力的論證，也徹底與傳統推薦算法拉開了實驗完整度的差距。


## File: docs\adr\ADR-006-Baseline-Models-Implementation.md

# ADR-006: 對照組模型實作 (Baseline Models Implementation)

*   **狀態**: 已接受 (Accepted)
*   **日期**: 2026-04-09

## 背景 (Context)

在學術論文與深度實驗中，單一模型的效能優異並不足以證明其價值。為了支撐專案核心模型 (KGAT) 的效能宣稱，我們需要建立完整的對照組 (Baselines) 矩陣，以證明「知識圖譜 (KG)」與「注意力機制 (Attention)」在食譜推薦任務中確實帶來了關鍵增益。

原本的專案僅具有消融實驗 (如：w/o Attention, w/o KG)，缺乏與領域內其他具備代表性的模型進行對比。

## 決策 (Decision)

我們實作了以下三款具備代表性的 Baseline 模型，涵蓋推薦系統發展的三大關鍵階段：

1.  **BPR-MF (Bayesian Personalized Ranking - Matrix Factorization)**:
    *   **定位**: 傳統 Collaborative Filtering 的黃金標準。
    *   **作用**: 代表僅利用 User-Item 互動隱含語意 (Latent Factors)，而不涉及任何深度學習或圖結構的基準。
2.  **NFM (Neural Factorization Machines)**:
    *   **定位**: 特徵互動與非線性建模代表。
    *   **作用**: 透過 Bi-Interaction Pooling 處理特徵交叉，並接續 MLP。代表了「擁有屬性資訊但沒有圖結構 (Graph-less)」的神經網路極限。
3.  **LightGCN**:
    *   **定位**: 純圖神經網路 (Pure GNN) 代表。
    *   **作用**: 在二分圖上進行純粹的訊息平滑傳遞，捨棄非線性變換。代表了「擁有圖結構但沒有知識圖譜 (KG-less)」的極簡 GNN 極限。

為了統一管理並提高實驗效率，我們同時實作了：
*   **`src/train_baseline.py`**: 總管型訓練腳本，支援透過 `--model` 參數切換基準。
*   **`run_baseline_experiments.bat`**: 自動化批次執行腳本，預設每個 Baseline 執行 10 Epochs 並統一儲存至 `baseline/` 目錄。

4.  **可解釋性量化指標 (Fidelity Evaluation)**:
    *   為了避免解釋結果僅停留在「視覺化」或「感覺」，我們引入了 Fidelity 系列指標進行數學驗證：
    *   **Fidelity+ (越正越好)**: 衡量移除這些解釋路徑後，模型的預測準確度（機率）下降了多少。越高代表被選出的路徑越不可或缺。
    *   **Fidelity- (越小越好)**: 衡量如果僅保留這些解釋路徑，模型是否仍能維持原始預測。越接近 0 代表這些解釋路徑即具備足夠的資訊量。
    *   這透過 `src/evaluate_fidelity.py` 實作，能為論文提供 XAI (Explainable AI) 的核心數據支持。

## 目錄組織決策

為了保持專案根目錄與模型目錄的簡潔，我們採用以下路徑規範：
*   **模型權重**: `models/baseline/`
*   **訓練日誌**: `output/logs/baseline/`

## 後續影響 (Consequences)

1.  **實驗完整性**: 論文現在可以生成包含 Traditional CF, Feature interaction, GNN CF 以及 KGAT 的完整對比表 (Performance Table)。
2.  **效能基準**: 提供了一個穩健的 Recall@K 下限，可客觀評估 KGAT 超參數調整的實際成效。
3.  **模型多樣性**: 專案轉型為一個通用的、基於 KG 的食譜推薦與實驗框架。


## File: docs\adr\README.md

# 架構決策紀錄 (Architecture Decision Records)

本目錄存放專案開發過程中的重大架構決策。每份紀錄皆包含決策背景、方案選型及後續影響。

## 決策列表

### 過往決策的統整與歸檔
*   [ADR-001: 保留的歷史架構決策 (Retained Decisions)](ADR-001-Retained-Decisions.md)
*   [ADR-002: 已修改或廢棄的歷史決策 (Superseded Decisions)](ADR-002-Superseded-Decisions.md)

### 最新效能與學術優化 (The Optimization & Ablation Update)
*   [ADR-003: 訓練效能極致優化 (Training & Performance Optimization)](ADR-003-Training-Optimization.md)
*   [ADR-004: 回歸原始論文 (Paper Alignment)](ADR-004-Paper-Alignment.md)
*   [ADR-005: 消融實驗架構設計 (Ablation Study Architecture)](ADR-005-Ablation-Study-Architecture.md)
*   [ADR-006: 對照組模型實作 (Baseline Models Implementation)](ADR-006-Baseline-Models-Implementation.md)


