# ADR-002: 在純 PyTorch 中實作真實注意力機制 (Attention Mechanism)

## 狀態
已接受 (Accepted)

## 背景脈絡 (Context)
在將專案從 DGL 遷移至純 PyTorch 實作後 (參考 ADR-001)，初期的 `KGAT` 實作使用了靜態的鄰接矩陣 (Adjacency Matrix) 與固定的歸一化權重 (類似 GCN/GraphSage 的 $D^{-1}A$)。

雖然這使得模型能在不依賴 DGL 的情況下進行訓練與執行，但這偏離了 **知識圖譜注意力網路 (KGAT)** 的核心設計。KGAT 的關鍵特點在於其能夠根據實體與關係嵌入 (Embeddings) 之間的交互作用，動態地為鄰居節點分配權重。原本簡化的 GCN 聚合器無法捕捉這些動態的語義關係。

## 決策 (Decision)
我們決定使用純 PyTorch 運算為 `KGATAttention` 模型實作 **真實的注意力機制 (Real Attention Mechanism)**。

1.  **架構設計**: 在 `src/model/kgat_attention.py` 中建立新的模型類別 `KGATAttention`，以避免干擾目前正在進行的靜態模型訓練。
2.  **實作機制**:
    *   使用 `leaky_relu(a^T [Wh_i || Wh_j])` 計算原始注意力分數 (logits)。
    *   使用 **`torch.scatter_add`** 實作自定義的 **Edge Softmax** 函數，從而在稀疏格式下對目標節點的邊權重進行歸一化。
    *   利用計算出的注意力係數 ($\alpha$) 動態建構稀疏鄰接矩陣，以進行訊息傳遞。
3.  **解釋器 (Explainer)**: 在 `src/model/explainer_attention.py` 中開發專用的 `KGATAttentionExplainer`。該解釋器直接利用模型回傳的注意力權重進行結果解釋，而非依賴梯度顯著性 (Gradient Saliency)。
4.  **硬體支援**: 提供 `src/train_xpu.py` 以支援透過 `intel_extension_for_pytorch` (IPEX) 加速的 Intel Arc 系列顯卡。
5.  **雲端訓練**: 建立 `notebooks/train_attention_colab.ipynb`，提供一個自包含 (Self-contained) 的訓練環境，方便使用者在 Google Colab 上利用 GPU 資源快速訓練新版模型，而不受本地環境限制。

## 後續影響 (Consequences)
### 優點
*   **正確性**: 模型現在忠實地實作了具有動態注意力的 KGAT 架構。
*   **可解釋性**: 注意力權重提供了邊重要性的直接且內在的衡量標準，簡化了解釋生成過程。
*   **效能潛力**: 透過區分重要的鄰居，預計能提升推薦品質。
*   **靈活性**: 純 PyTorch 實作確保了在各種硬體 (CUDA, CPU, XPU) 上的部署均不受外部圖運算函式庫限制。

### 缺點
*   **複雜度**: 手動實作 Edge Softmax 和動態稀疏矩陣建構比使用 DGL 或 PyG 等現成函式庫更為複雜。
*   **記憶體佔用**: 計算與儲存所有邊的注意力係數梯度可能會比靜態 GCN 增加記憶體開銷。

## 相關參考
*   [Xiang Wang et al., "KGAT: Knowledge Graph Attention Network for Recommendation", KDD 2019](https://arxiv.org/abs/1905.07854)
*   ADR-001: 移除 DGL 依賴並遷移至純 PyTorch 實作
