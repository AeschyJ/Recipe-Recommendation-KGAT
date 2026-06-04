# ADR-003: 訓練效能極致優化 (Training & Performance Optimization)

## 狀態
**Status:** 已接受 (Accepted) - 持續更新

## 背景與問題 (Context)
在將模型擴展以進行消融實驗時，我們遇到了嚴重效能瓶頸與資源枯竭：
1. **推論過慢**: `evaluate` 函數在每個 batch 皆完整走一次 3 層的 GNN 向前傳播 (Forward Pass)，導致 Validation 階段耗時高達十數分鐘。
2. **XPU 架構效能低落**: 先前使用的 Custom Python Autograd (`SparseAggregateFunction`) 在部分矩陣運算上無法適配 IPEX 的算子，導致訓練引發 CPU fallback，時間以倍數遞增。
3. **記憶體撐爆 (OOM)**: 當啟動完整版的 Relation-aware Attention (包含龐大的 relation feature vector 融合) 並擴張至深度 `L=3` 時，16GB VRAM 不足以負荷運算圖。
4. **訓練異常緩慢**: `KGAT_BiInteraction` (wo_attn) 退化至 33-60+s/it，而 `KGATAttention` (full_kgat) 亦卡在 6.0s/it 左右，無法發揮硬體效能。

## 決策 (Decision)

針對上述問題，我們在架構內實施了以下系統性工程優化：

1. **全 XPU、bfloat16 訓練策略**
   - 將所有實驗環境預設使用 `--use_bf16`，利用 XPU BFloat16 原生算力將記憶體佔用砍半，同時維持相同的訓練收斂精度。

2. **GNN 訊息傳遞使用 PyTorch 原生 `index_add_` (取代 Autograd)**
   - 全面拔除自定義 Python 梯度類別，改以原生 Tensor 修改操作 `out.index_add_(0, edge_index, message)` 實現稀疏圖訊息聚合。
   - 經測試證實：此舉成功避免了計算卡在 CPU/XPU 間搬移。

3. **GNN 共享與推論快取 (Evaluate Embedding Caching)**
   - 實作新的模型方法 `get_final_embeddings()`。在 `evaluate` 迴圈開始前，率先將所有 Nodes 餵進 GNN，得出**唯一一份**推論結果 `u_g_embeddings` 以及 `i_g_embeddings`。
   - 在每一個測試 batch 中，僅依賴取出對應 index 並執行內積 (`inner product`) 即可給出預測分數，這使得評估耗時由原本的分鐘級驟降至秒級。

4. **Attention 特徵融合後再線性轉化 (Activation Checkpointing)**
   - 面對 `L=3` 層 Full Attention 所引發的 VRAM OOM，在不改動參數前提下，我們引進了 PyTorch 的 `torch.utils.checkpoint` 機制。
   - 利用運算時間交換記憶體：前向傳播不保存中繼 Variable，反向時重新計算，成功使得極深層的知識圖譜網路能在普通開發機顯示卡上流暢運行。

5. **XPU BFloat16 `index_add_` 效能 Bug 修正** *(關鍵發現)*
   - **根因**: 透過逐操作 profiling 發現，Intel XPU 上 `index_add_` 在 bfloat16 精度下
     存在嚴重的效能退化 (**140x slower** vs float32)。真實圖的高度數節點 (max_degree=22,096)
     會觸發大量原子寫入衝突，XPU 的 bf16 kernel 無法有效合併這些衝突寫入。
   - **證據**: 相同操作在 float32 下僅需 0.28s，bfloat16 需要 39.1s (相同 indices 和 tensor size)。
     使用隨機 indices (max_degree=36) 時，bf16 和 f32 均為 ~0.1s，證明問題僅發生在高衝突場景。
   - **解法**: 在 `GNNLayer.forward` 與 `KGATAttention.forward` 中，使用 `torch.autocast(enabled=False)` 將聚合區段
     (`gather → multiply → index_add_`) 強制在 float32 下執行，完成後再轉回原始 dtype。
   - **效果**: 
     - `wo_attn` (BiInteraction) 提升至 **0.24 s/it** (~120倍加速)。
     - `full_kgat` (Attention) 提升至 **2.1 s/it** (~3倍加速)。
   - **備註**: 先前嘗試的 Activation Checkpointing 因誤判硬體瓶頸為記憶體壓力而引入，
     實際反而使 backward 重新計算，將 per-iteration 時間加倍至 60+s/it。已移除。

## 影響 (Consequences)
- **正面影響**:
  - 大幅縮減了超參數搜索所需的訓練周期。
  - `wo_attn` 模型提升至 0.24 s/it，`full_kgat` 模型提升至 2.1 s/it。
  - 對未來大型資料集擴展以及更強的深層模型提供了穩健的架構支援。
- **負面影響 / 限制**:
  - `checkpoint` 由於以時間換取空間，在前處理時計算速度會略為下降，但避免了程式因 OOM 崩潰的可能，這是為求穩定深層網路而不得不採取的策略妥協。
  - float32 聚合在 bf16 訓練中引入了微小的精度差異，但此差異不影響模型收斂。

