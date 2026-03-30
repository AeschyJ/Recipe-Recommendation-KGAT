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
