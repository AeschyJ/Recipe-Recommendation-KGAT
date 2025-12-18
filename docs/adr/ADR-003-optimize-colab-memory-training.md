# ADR-003: Google Colab 訓練環境優效與記憶體管理策略

## 狀態
已接受 (Accepted)

## 背景 (Context)
在 Google Colab 環境下訓練基於 Food.com 數據集的 KGAT 模型時，面臨嚴峻的記憶體挑戰：
1. **數據規模**：圖譜包含百萬級別的邊，導致 Node Embeddings 與 Edge Indices 佔用大量系統 RAM。
2. **PyTorch 限制**：`torch.sparse.mm` 在反向傳播時會試圖計算稠密梯度矩陣，在大型圖譜上會觸發 800GB+ 的荒謬記憶體申請導致 OOM。
3. **環境限制**：Colab 標準執行階段僅提供約 12GB RAM 與 15GB T4 VRAM，極易崩潰。

## 決策 (Decision)
為了確保模型能在雲端穩定訓練，我們採取了以下優化措施：
1. **Message Passing 模式 (Gather-Scatter)**：棄用 `torch.sparse.mm`，改用 `index_add_` (Scatter Add) 手寫訊息傳遞。這將記憶體消耗控制在與「邊的數量」成正比，而非節點數的平方。
2. **極致記憶體管理**：
    - 在 Forward Pass 中對大型暫存張量（如 `edge_h`, `weighted_msg`）執行顯式 `del` 操作。
    - 每個 Epoch 結束時強制執行 `gc.collect()` 與 `torch.cuda.empty_cache()`。
    - 在優化器中使用 `set_to_none=True` 以釋放梯度記憶體。
3. **精度與相容性控制**：
    - 採用 AMP (自動混合精度) 節省 VRAM。
    - 針對稀疏相關運算，強制轉型為 `float32` 並暫時跳出 AMP 區塊，解決 `Half` 精度不支持的限制。
4. **架構優化**：
    - 使用原始 `indices` 陣列取代封裝過的 Sparse Tensor 物件，減少 Python 物件開銷。
    - 降低嵌入維度至 32 作為初始穩定維度。

## 後續影響 (Consequences)
- **正面影響**：成功將 800GB+ 的異常記憶體申請降至 14GB 左右的穩定負載，使模型能在 Colab T4 GPU 上順利運行。
- **正面影響**：支援 Google Drive 持久化儲存，避免雲端斷線導致訓練進度丟失。
- **負面影響**：頻繁的垃圾回收與顯存清理會輕微降低訓練速度（約 5-10%）。
