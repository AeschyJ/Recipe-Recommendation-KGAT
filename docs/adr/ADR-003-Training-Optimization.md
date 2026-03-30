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
