# ADR-006: 針對大規模圖譜的 VRAM 顯存優化 (XPU/GPU)

## 狀態
已接受 (Accepted)

## 背景 (Context)
本專案使用的食譜推薦圖譜包含超過 1,500 萬條邊。在標準的 GNN 訊息傳遞（Message Passing）實現中，`index_add_` 或 `torch.sparse.mm` 的中間過程會產生一個大小為 $E \times \text{embed\_dim}$ 的訊息張量（約 1.9 GB / Layer）。

在顯存有限（如 8GB）的 Intel Arc GPU 或 NVIDIA GPU 上，多層 GNN 會迅速填滿顯存，導致：
1. **CUDA/XPU Out of Memory (OOM)**。
2. **系統記憶體交換 (Swapping)**：驅動程式會將資料搬移至系統 RAM 運算，導致單次迭代時間從數秒飆升至一分鐘，效率極低。

## 決策 (Decision)
實作自定義 Autograd 算子 `SparseAggregateFunction`（位於 `src/model/kgat.py`），採用 **「空間換取時間」** 的重計算（Recomputation）策略：

1. **Forward Pass**：執行聚合運算後，立即釋放暫存的訊息張量（Message Tensor），不將其保留在運算圖中。
2. **Backward Pass**：當需要計算梯度時，利用現有的 Embedding 與邊權重**重新計算**訊息，並直接進行梯度累加。
3. **優化預算**：將稀疏矩陣的索引提取與 Layout 檢查移出模型層級，改由 `forward` 只執行一次。

## 後續影響 (Consequences)
1. **顯存需求大幅降低**：單層 GNN 的顯存佔用從「正比於邊數」降低為「正比於節點數」，使 8GB 級別的顯示卡能穩定處理千萬級規模的圖譜。
2. **訓練速度提升**：雖然 Backward 增加了運算量，但因完全避開了 Swapping，總訓練時間縮短了數倍。
3. **支援大 Batch Size**：現在可以在有限顯存下使用更大型的 Batch Size (如 20480)，進一步加速 Epoch 週期。
4. **程式碼維護**：訊息傳遞邏輯現在封裝在 `SparseAggregateFunction` 中，結構更清晰。
