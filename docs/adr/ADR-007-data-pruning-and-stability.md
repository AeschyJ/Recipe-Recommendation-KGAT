# ADR-007: 去除超級節點、提高權重與 Attention 數值穩定性

## Status
Accepted

## Context
在 KGAT 模型的訓練過程中，我們觀察到以下幾個問題：
1.  **超級節點 (Super Nodes)**：部分食材（如 `egg`, `water`, `salt`）或標籤出現頻率過高，導致 Graph 結構過於稠密且缺乏區別度，影響模型學習具體食譜特色的能力。
2.  **Attention 數值不穩定**：在計算 Attention Score 時，原始的 Softmax 實作容易因數值過大導致 `NaN` 或梯度溢出，特別是在 XPU/混合精度訓練環境下。
3.  **食材邏輯權重不足**：模型初期對「食材-食譜」的關聯關注度不足，導致推薦解釋性較弱。

## Decision
我們決定實施以下優化措施：

1.  **數據預處理剪枝 (Data Pruning)**：
    -   在 `src/data/preprocess.py` 中新增邏輯，統計並移除出現頻率 **前 0.5%** 的高頻食材節點（約 74 種，如 `carrot`）並移除部分基本食材（如 `eggs`, `water`, `salt`）。
    -   此舉旨在移除雜訊，迫使模型利用更具特色的食材進行推論。

2.  **Attention 數值穩定化 (Shifted Softmax)**：
    -   在 `src/model/kgat_attention.py` 中，將 Softmax 計算邏輯改為 **Per-node Max Subtraction**。
    -   利用 `torch.scatter_reduce` (reduce='amax') 計算每個目標節點鄰居的最大 Attention Score，並在 Softmax 前減去該值。即 $e_{ij} = e_{ij} - \max(e_{neighborhood})$。
    -   此方法能確保指數運算 `exp()` 不會溢出，大幅提升數值穩定性。

3.  **食材關係偏置 (Ingredient Bias)**：
    -   在 Attention 計算中，針對食材相關的 Edge Type 加入額外的 Bias 或初始高權重。
    -   引導模型在訓練初期更關注食材組成，提升推薦的合理性。

## Consequences
-   **Positive**:
    -   訓練過程更加穩定，減少 Loss 震盪或 NaN 的機率。
    -   模型解釋性提升，生成的 Attention Weights 能更精確反映關鍵食材。
    -   移除超級節點後，計算圖 (Graph) 的規模略微下降，可能有助於訓練速度。
-   **Negative**:
    -   部分基礎食材（如鹽、油）的關聯資訊丟失，若某些食譜僅靠這些食材連結，可能會受影響（但預期此類情況較少建構出有意義的推薦）。
    -   需要確保 PyTorch 版本支援 `scatter_reduce` (需 >= 1.12)。

## Compliance
-   修改 `src/data/preprocess.py`。
-   修改 `src/model/kgat_attention.py`。
