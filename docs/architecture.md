# 專案實驗架構與模組設計 (Architecture & Design)

本專案旨在重新實作與驗證知識圖譜注意力神經網路 (Knowledge Graph Attention Network, KGAT) 在食譜推薦上的效能。歷經多次重構後，目前的系統架構專為「嚴謹對照原論文」與「最大化原生 PyTorch XPU 硬體效能」而打造。

---

## 1. 原論文理論對齊 (Paper Alignment)

為了能在消融實驗中給出具備說服力的對比基準，我們在核心模組 (`src/model/kgat.py`) 中嚴格對齊了 [KGAT (Wang et al., KDD 2019)](https://arxiv.org/abs/1909.02695) 的理論架構：

1. **Relation-Aware Attention 機制**:
   - 捨棄普通的圖注意力網路 (GAT) 點積。
   - 實作關係感知注意力公式：
     $$\pi(h,r,t) = (W_{\text{att}} e_t)^\top \tanh(W_{\text{att}} e_h + e_r)$$
   - 配合 Softmax 正規化得出邊權重 $\alpha_{h,r,t}$，使網路能感知不同關係類型（如：食譜包含食材 vs. 食譜具備標籤）之重要度。
2. **Bi-Interaction 聚合公式**:
   - GNN 訊息傳遞同時包含節點特徵相加與元素級乘積 (Element-wise product)：
     $$h_{\text{out}} = \text{LeakyReLU}\left( W_1(e_u + e_v) + W_2(e_u \odot e_v) \right)$$
3. **BPR Loss 與 $L_2$ 正則化 (Weight Decay)**:
   - 採用成對比較 (Pairwise) 的 Bayesian Personalized Ranking 損失函數。
   - 搭配 $L_2$ 正則化 ($10^{-5}$) 防止過擬合。
4. **Message Dropout 防護**:
   - 在每層 GNN 訊息傳遞後與注意力權重上套用 `nn.Dropout(p=0.1)` 增加抗噪能力。

---

## 2. 原生 PyTorch XPU 與效能優化策略 (Hardware Optimizations)

為了解決原論文架構在消費級硬體（如 Intel Arc A750 8GB VRAM）上的顯存與計算瓶頸，專案導入以下關鍵優化：

1. **原生 PyTorch XPU & BFloat16 混合精度**:
   - 採用 PyTorch 2.4+ 原生 XPU 後端，無需外掛 IPEX 套件。
   - 啟用 BFloat16 混合精度將顯存佔用降低近半，使 $L=3$ 深層模型訓練成為可能。
2. **`index_add_` 聚合與浮點退避策略**:
   - Intel XPU 在對高頻超級節點（如鹽、水等常見食材）進行 BFloat16 原子寫入時存在極大效能退化。
   - 專案在淺層訊息聚合時採用 float32 `out.index_add_(0, edge_index, message)`，並於深層退回 bf16 防止 OOM。
3. **Activation Checkpointing 梯度重算**:
   - 針對多層 ($L=3$) Relation-Aware Attention 產生的巨量臨時邊矩陣，導入 `torch.utils.checkpoint` 技術，在 Forward 階段丟棄中間激活值，Backward 階段重算，解除顯存爆滿限制。
4. **`get_final_embeddings()` 隱含向量快取推論**:
   - 在 Validation/Test 評估前僅執行單次前向傳播提取全圖用戶與實體 Embedding，後續測試僅作向量查表與內積，測試時間由十分鐘降至數秒。

---

## 3. 消融與對比實驗陣列 (Ablation & Baseline Matrix)

專案提供 5 組 KGAT 消融實驗與 3 組經典 Baseline 模型，全數封裝於自動化批次腳本中：

### 3.1 消融實驗組 (`run_experiments.bat` $\rightarrow$ `src/train.py`)
| 實驗名稱 | 指令旗幟 | 核心目的 |
| :--- | :--- | :--- |
| **Full KGAT (L=1)** | `--layers 64` | 完整版 KGAT 基準 |
| **w/o Attention** | `--no_attention --layers 64` | 將注意力退化為均等權重，驗證「注意力機制」效益 |
| **w/o Knowledge Graph**| `--without_kg --layers 64` | 剔除 CKG 知識三元組，驗證「注入外部知識」效益 |
| **Depth L=2** | `--layers 64 64` | 探討 2-hop 遠鄰居資訊傳遞效益 |
| **Depth L=3** | `--layers 64 64 64` | 挑戰 3-hop 極限與過度平滑 (Oversmoothing) 瓶頸 |

### 3.2 經典對照組 (`run_baseline_experiments.bat` $\rightarrow$ `src/train_baseline.py`)
* **BPR-MF**：傳統矩陣分解協同過濾 (`--model BPR-MF`)。
* **LightGCN**：無非線性變換與權重矩陣的純圖卷積協同過濾 (`--model LightGCN`)。
* **NFM (Neural Factorization Machine)**：二階特徵交互搭配深度 MLP 網路 (`--model NFM`)。

---

## 4. 可解釋性評估與 Fidelity 框架 (Explainability Framework)

專案除了評估傳統推薦精準度（HR@K, NDCG@K, Precision@K），亦建立量化 XAI 評估管道：

```
[Target Users] ---> src/evaluate_fidelity.py ---> [KGATAttentionExplainer]
                                                           |
                                           +---------------+---------------+
                                           |                               |
                                    (Occlusion / Fid+)            (Sufficiency / Fid-)
                                           |                               |
                                  遮擋 Top-K 解釋路徑               僅保留 Top-K 解釋路徑
                                           |                               |
                                    預測分數降幅ΔP+                 預測分數殘差ΔP-
```

1. **`KGATAttentionExplainer` (`src/model/explainer_attention.py`)**：結合 Multi-layer Attention 權重與 NetworkX 圖搜尋，提取 User 到 Item 的高貢獻路徑。
2. **`Fidelity+` (必要性)**：$P_{\text{orig}} - P_{F+}$。數值越正，代表該解釋路徑對推薦決策越不可或缺。
3. **`Fidelity-` (充分性)**：$P_{\text{orig}} - P_{F-}$。數值越小，代表僅保留該路徑即可維持原始推薦。
