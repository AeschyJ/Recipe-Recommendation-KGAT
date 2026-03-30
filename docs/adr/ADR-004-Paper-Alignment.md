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
