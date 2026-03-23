# ADR-009: 支援多維度消融實驗與基礎模型重新命名

## 狀態 (Status)
已接受 (Accepted)

## 背景 (Context)
為驗證 Knowledge Graph Attention Network (KGAT) 之中各模組的有效性，需要進行系統性的消融實驗 (Ablation Study)。
在原先的架構中，不具備注意力機制的模型與訓練腳本被模糊地命名為 `kgat_base` 與 `train.py`。事實上，該「基礎」模型內部採用了 Bi-Interaction 聚合方式（結合對位相乘與相加），只差在沒有引入 Attention 權重計算。
為了讓消融實驗更具說服力，並且反映出架構的實際運作，我們必須：
1. 為現有模型賦予精確的命名。
2. 開發能開關不同模組（如圖譜資訊、傳播層數）的實驗機制，確保比較基礎的一致性。

## 決策 (Decision)
1. **重構與重命名**:
   - 將 `kgat_base` 重命名為 `kgat_bi_interaction`。
   - 將 `train.py` 重命名為 `train_bi_interaction.py`。
   - 替換所有依賴該類別的模組引用 (包含推論與解釋生成腳本)。
2. **消融特徵支援**:
   - 在訓練腳本中擴增 `--layers` 命令列參數，允許自由設定圖神經網路深度 (如 L=1, L=2, L=3)。
   - 在 Attention 訓練腳本中擴增 `--without_kg` 參數，透過在圖譜索引建構階段直接忽略關係與實體節點，來退化成純粹的使用者-物品歷史記錄互動模型。
3. **自動化實驗管線**:
   - 開發 `run_experiments.bat`，一次排程執行 5 組主要的對照實驗：Full KGAT、w/o Attention、w/o Knowledge Graph、Depth L=2、Depth L=3。

## 後續影響 (Consequences)
1. **架構清晰化**: 未來開發者看到 `kgat_bi_interaction.py` 即可明確知道該模型採用的聚合機制為何。
2. **實驗效率提升**: 所有模型變體均透過統一且自動化的批次檔執行，能夠快速驗證假說，日誌與權重檔案也能依結構分隔保存。
3. **推論腳本相容度**: `generate_explanations.py` 已對接新的類別名稱，推論功能未受影響。
