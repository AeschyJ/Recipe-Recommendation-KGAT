# ADR-002: 已修改或廢棄的歷史決策 (Superseded Architecture Decisions)

## 狀態
**Status:** 已廢棄 / 已修改 (Superseded/Obsolete)

## 決策列表

### 1. 使用過渡版本的注意力機制 (原 ADR-002)
- **原決策背景**: 早期為了解決程式碼漏洞，自行撰寫了一版簡化的 GAT，未納入 Knowledge Graph 的 `relation` 特徵融合 (`W_r`)。
- **為何被推翻**: 無法與原始 KGAT (Knowledge Graph Attention Network) 論文進行真正的學術對標。
- **取代方案**: 本次專案已「回歸原始論文」，實作正統的 Relation-Aware Attention ($\pi(h,r,t) = (W_r e_t)^\top \tanh(W_r e_h + e_r)$)，請參見 **ADR-004**。

### 2. Google Colab 記憶體優化 (原 ADR-003)
- **原決策背景**: 過去因硬體限制，嘗試透過 `gc.collect()` 或中斷 Graph 歷史圖形來省下 Colab 寶貴的 VRAM。
- **為何被推翻**: 在本機 Intel Arc (XPU) 環境下，這種 Python 層面的 GC 回收不僅無法大幅緩解 VRAM 危機，還會嚴重拖累訓練速度。
- **取代方案**: 改為深層網路的 Activation Checkpointing 策略，請參見 **ADR-003** 效能優化篇。

### 3. 以分批重算作為大規模 VRAM 優化 (原 ADR-006)
- **原決策背景**: 為解決 OOM (Out Of Memory) 嘗試過將 GNN 鄰居切塊分別矩陣相乘，放棄了時間換取空間。
- **為何被推翻**: 訓練效率崩落，且原先以 `torch.sparse` 與 Python index 實作的反向傳播在 XPU 上發生嚴重的 fallback to CPU。
- **取代方案**: 從底層改寫為 PyTorch 的 `index_add_` 原生支援，不會 OOM，訓練速度獲得提升。請參見 **ADR-003** 效能優化篇。

### 4. 舊版消融命名與基礎架構 (原 ADR-009)
- **原決策背景**: 原先嘗試定義了兩三款不同實驗的檔案名稱，並打算進行以 KGE (TransR) 為輔助的多工作業 (Joint Training)。
- **為何被推翻**: 原始設定無法達到純淨的對照實驗，容易因為 KGE 收斂速度的問題影響主線目標 (BPR 推薦任務) 的對比。
- **取代方案**: 確立更加嚴謹且獨立的 5 組新消融實驗對照組，請參見 **ADR-005**。
