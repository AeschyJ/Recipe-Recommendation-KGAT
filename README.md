# 食譜推薦系統 - Knowledge Graph Attention Network 消融實驗套件

這是一個基於知識圖譜注意力網絡 (KGAT) 的推薦系統專案。專案經過大規模重構，專為 Intel Arc (XPU) 加速、BFloat16 訓練而生，並將注意力放在嚴謹的論文覆現與消融實驗。

## 功能特色
* **資料處理**: 自動讀取 Kaggle Food.com 食譜資料，過濾雜訊後融合使用者互動建立 `Collaborative Knowledge Graph`。
* **KGAT 模型**: 實作最純粹的 Relation-Aware Attention ($\pi(h,r,t) = (W_r e_t)^\top \tanh(W_r e_h + e_r)$)、Bi-Interaction，以及對應的正則化與 Dropout 保護。
* **效能優化**: 具有 `get_final_embeddings()` 查詢快取機制與 PyTorch Checkpointing，能在普通 GPU/XPU 設備上達成深度的圖神經網路訓練。
* **自動化實驗**: 內置一鍵腳本，輕鬆排程多項變數的控制對照。

## 專案依賴
本專案使用 `uv` 進行套件管理。執行：
```bash
uv sync
```
這將會根據 `uv.lock` 建立虛擬環境並安裝所有必要套件 (`torch`, `pandas` etc.)，Intel XPU 需確保系統已安裝 IPEX 與對應驅動。

---

## 快速開始 (Quick Start)

### 1. 準備資料與前處理
請確保已經從 Kaggle 下載 `RAW_recipes.csv` 與 `RAW_interactions.csv` 放入 `data/raw/`，接著執行建立圖譜的預處理：
```bash
.venv\Scripts\python.exe src/data/preprocess.py
```
*(結果將序列化為圖譜與特徵表，存放於 `data/processed/`)*

### 2. 執行消融實驗套件 (Ablation Tests)
我們提供了 `run_experiments.bat` 來自動化所有對照組（基準 KGAT、消除 Attention、消除 KG、各種網路深度等）。
在根目錄直接點擊或透過終端機執行：
```powershell
./run_experiments.bat
```
腳本內建防呆機制，會自動透過專案內的虛擬環境派發下列五種基準訓練腳本：
1. **Full KGAT** (`src/train_att.py`)
2. **w/o Attention** (`src/train_bi_interaction.py`)
3. **w/o KG 推薦** (`src/train_att.py --without_kg`)
4. **Depth L=2 模型**
5. **Depth L=3 模型**

目前預設為 `10` 次 Epoch 以供快跑測試趨勢，所有 Log 與模型權重將獨立匯出。

### 3. 可選：自訂獨立訓練
若只想獨立訓練某一款特定配置的模型：
```bash
.venv\Scripts\python.exe src/train_att.py --epochs 30 --layers 64 --model_dir models/my_kgat --use_bf16
```
若遭逢意外中斷，可利用 `--resume` 旗幟載入舊進度：
```bash
.venv\Scripts\python.exe src/train_att.py --resume models/my_kgat/kgat_checkpoint_e10.pth
```

## 文檔索引
欲深入了解這套系統的心路歷程與各模組實作細節，請參閱：
* [專案實驗架構與模組設計](docs/architecture.md): 本次實驗架構原理與消融變數詳解。
* [API 參考文件](docs/api_reference.md): 兩款核心模型程式切入點與優化演算法說明。
* [架構決策紀錄 (ADR Index)](docs/adr/README.md): 紀錄所有效能卡關與學術選擇的解決歷程。
