# ADR-001: 移除 DGL 依賴並遷移至純 PyTorch 實作

*   **狀態**: 已接受 (Accepted)
*   **日期**: 2025-12-18
*   **決策者**: User, Assistant

## 背景脈絡 (Context)

專案原先使用 Deep Graph Library (DGL) 來實作 Knowledge Graph Attention Network (KGAT)。然而在 Windows 環境下的開發過程中，遭遇了以下問題：
1.  **安裝困難**: DGL 在 Windows 上的 CUDA 版本對應與依賴安裝極為繁瑣，且常出現版本衝突。
2.  **執行不穩定**: `dgl` 在特定環境下曾發生 Process 凍結或後端通訊錯誤。
3.  **依賴過重**: 對於此專案的規模而言，DGL 引入了過多不必要的複雜度，增加了維護成本。

## 決策方案 (Decision)

決定 **完全移除 DGL 依賴**，並使用 **純 PyTorch** 重寫模型核心與資料載入邏輯。

主要技術變動：
1.  **資料結構**: 不再使用 `dgl.DGLGraph`。改用 PyTorch Tensor (如 `edge_index` 格式，類似 PyTorch Geometric) 來儲存知識圖譜的連接資訊。
2.  **模型實作**: 自行實作 `GNNLayer` 中的訊息傳遞 (Message Passing) 與聚合 (Aggregation) 邏輯。利用 `torch.sparse.mm` 或 `torch.scatter_add` 進行鄰居特徵聚合。
3.  **解釋器**: 原有的 `dgl.nn.GNNExplainer` 也將被移除，未來將根據新的 PyTorch 架構重新實作 Explainer。

## 後續影響 (Consequences)

### 優點
*   **環境友善**: 任何安裝了 PyTorch 的環境皆可直接執行，大幅降低安裝門檻 (Windows/Linux/Mac 通用)。
*   **輕量化**: 減少了第三方依賴包的大小與版本風險。
*   **Debug 容易**: 原生 Tensor 運算更容易進行斷點除錯與數值追蹤，無 DGL 封裝層的黑盒效應。

### 缺點
*   **開發工時**: 需手動實作 GNN 的 Scatter/Gather 邏輯，程式碼量可能會略微增加。
*   **效能調優**: 需自行關注稀疏矩陣運算的記憶體與速度優化，初期效能可能不如高度優化的 DGL。

### 受影響模組
*   `src/model/kgat.py`: 需重寫 `GNNLayer` 與 `KGAT`。
*   `src/model/explainer.py`: 暫時失效，需重構。
*   `src/train.py`: 資料載入與 Batch 採樣邏輯需移除 DGL 相關 code。
