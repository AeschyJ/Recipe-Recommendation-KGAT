# ADR-006: 對照組模型實作 (Baseline Models Implementation)

*   **狀態**: 已接受 (Accepted)
*   **日期**: 2026-04-09

## 背景 (Context)

在學術論文與深度實驗中，單一模型的效能優異並不足以證明其價值。為了支撐專案核心模型 (KGAT) 的效能宣稱，我們需要建立完整的對照組 (Baselines) 矩陣，以證明「知識圖譜 (KG)」與「注意力機制 (Attention)」在食譜推薦任務中確實帶來了關鍵增益。

原本的專案僅具有消融實驗 (如：w/o Attention, w/o KG)，缺乏與領域內其他具備代表性的模型進行對比。

## 決策 (Decision)

我們實作了以下三款具備代表性的 Baseline 模型，涵蓋推薦系統發展的三大關鍵階段：

1.  **BPR-MF (Bayesian Personalized Ranking - Matrix Factorization)**:
    *   **定位**: 傳統 Collaborative Filtering 的黃金標準。
    *   **作用**: 代表僅利用 User-Item 互動隱含語意 (Latent Factors)，而不涉及任何深度學習或圖結構的基準。
2.  **NFM (Neural Factorization Machines)**:
    *   **定位**: 特徵互動與非線性建模代表。
    *   **作用**: 透過 Bi-Interaction Pooling 處理特徵交叉，並接續 MLP。代表了「擁有屬性資訊但沒有圖結構 (Graph-less)」的神經網路極限。
3.  **LightGCN**:
    *   **定位**: 純圖神經網路 (Pure GNN) 代表。
    *   **作用**: 在二分圖上進行純粹的訊息平滑傳遞，捨棄非線性變換。代表了「擁有圖結構但沒有知識圖譜 (KG-less)」的極簡 GNN 極限。

為了統一管理並提高實驗效率，我們同時實作了：
*   **`src/train_baseline.py`**: 總管型訓練腳本，支援透過 `--model` 參數切換基準。
*   **`run_baseline_experiments.bat`**: 自動化批次執行腳本，預設每個 Baseline 執行 10 Epochs 並統一儲存至 `baseline/` 目錄。

4.  **可解釋性量化指標 (Fidelity Evaluation)**:
    *   為了避免解釋結果僅停留在「視覺化」或「感覺」，我們引入了 Fidelity 系列指標進行數學驗證：
    *   **Fidelity+ (越正越好)**: 衡量移除這些解釋路徑後，模型的預測準確度（機率）下降了多少。越高代表被選出的路徑越不可或缺。
    *   **Fidelity- (越小越好)**: 衡量如果僅保留這些解釋路徑，模型是否仍能維持原始預測。越接近 0 代表這些解釋路徑即具備足夠的資訊量。
    *   這透過 `src/evaluate_fidelity.py` 實作，能為論文提供 XAI (Explainable AI) 的核心數據支持。

## 目錄組織決策

為了保持專案根目錄與模型目錄的簡潔，我們採用以下路徑規範：
*   **模型權重**: `models/baseline/`
*   **訓練日誌**: `output/logs/baseline/`

## 後續影響 (Consequences)

1.  **實驗完整性**: 論文現在可以生成包含 Traditional CF, Feature interaction, GNN CF 以及 KGAT 的完整對比表 (Performance Table)。
2.  **效能基準**: 提供了一個穩健的 Recall@K 下限，可客觀評估 KGAT 超參數調整的實際成效。
3.  **模型多樣性**: 專案轉型為一個通用的、基於 KG 的食譜推薦與實驗框架。
