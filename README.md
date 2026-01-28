# 食譜推薦系統 (Recipe Recommendation with KGAT)

這是一個基於知識圖譜注意力網絡 (Knowledge Graph Attention Network, KGAT) 的食譜推薦系統專案。本專案利用 Food.com 的資料集，結合使用者互動與食譜的成分、標籤等知識圖譜資訊，提供精準的推薦，並包含一個用於解釋推薦結果的模組 (Graph Explainer)。

## 功能特色

*   **資料處理管道**: 自動下載並處理 Kaggle 上的 Food.com 資料集，將原始 CSV 轉換為知識圖譜結構 (三元組)。
*   **KGAT 模型**: 實作 KGAT 模型，利用圖神經網路 (GNN) 聚合知識圖譜中的鄰居資訊。
*   **可解釋性 AI (XAI)**: 包含一個基於 `dgl.nn.GNNExplainer` 的解釋器模組，用於分析推薦背後的關鍵路徑與節點。

## 專案依賴

本專案使用 `uv` 進行套件管理。主要依賴包括：

*   `torch` (PyTorch)
*   `pandas`, `numpy`, `scikit-learn` (資料處理)

詳細列表請參閱 `pyproject.toml` 或 `requirements.txt`。

## 安裝指南

1.  **安裝 uv** (如果尚未安裝):
    ```bash
    pip install uv
    ```
    或者參考 [uv 官方文檔](https://github.com/astral-sh/uv)。

2.  **建立虛擬環境並同步依賴**:
    在專案根目錄下執行：
    ```bash
    uv sync
    ```
    這將會根據 `uv.lock` 建立虛擬環境並安裝所有必要的套件。

## 快速開始 (Quick Start)

1.  **準備資料**:
    執行資料下載腳本查看指引，需從 Kaggle 下載 `RAW_recipes.csv` 與 `RAW_interactions.csv` 並放入 `data/raw/` 目錄。
    ```bash
    python src/data/download_data.py
    ```

2.  **資料前處理**:
    執行預處理腳本，建立知識圖譜實體與關係。結果將儲存於 `data/processed/`。
    ```bash
    python src/data/preprocess.py
    ```
    *此步驟會過濾互動資料、重新映射使用者與物品 ID，並提取成分 (Ingredients) 與標籤 (Tags) 作為知識圖譜的實體。*

3.  **模型開發**:
    目前的模型核心位於 `src/model/kgat.py`。您可以在自己的訓練腳本中引用它：
    ```python
    from src.model.kgat import KGAT
    # 初始化模型
    model = KGAT(n_users=..., n_entities=..., n_relations=...)
    ```
    *(完整的訓練腳本 `main.py` 仍在開發中)*

3.  **模型訓練**:
    本專案支援多種訓練模式，包含傳統的 KGAT 以及帶有注意力和機制的 KGAT-Attention。

    *   **Intel Arc GPU (XPU) 加速模式** (建議，需 8GB+ VRAM):
        ```bash
        # 大 Batch Size (20480) 搭配 BFloat16 優化與重計算技術
        python src/train.py --use_bf16 --batch_size 20480 --lr 0.001
        ```
    
    *   **CPU 穩定模式** (若無 GPU 或顯存不足):
        ```bash
        python src/train_att.py --cpu --batch_size 1024
        ```

    *   **斷點續訓 (Resume Training)**:
        ```bash
        python src/train.py --resume models/kgat_checkpoint_e10.pth --batch_size 20480
        ```

    *訓練說明：*
    - **VRAM 優化**：針對大規模圖譜（15M 邊），模型已實作重計算（Recomputation）策略，有效解決 8GB 顯示卡溢位問題。
    - **格式**：Checkpoint 會自動儲存包含 Epoch、優化器狀態與超參數的完整資料。

4.  **生成推薦與解釋 (Inference & Explanation)**:
    使用 `src/generate_explanations.py` 腳本對隨機使用者進行推理並生成解釋 JSON 檔：
    ```bash
    # 對 5 個隨機使用者生成解釋，結果存至 output/explanations.json
    python src/generate_explanations.py --num_users 5 --output explanations.json
    ```
    *結果將包含實際的使用者 ID、食譜名稱與解釋路徑。*

5.  **Notebook 實驗**:
    您也可以使用 Jupyter Notebook 進行互動式開發與觀察：
    *   `notebooks/train_colab.ipynb`: 包含完整的模型訓練流程。
    *   `notebooks/inference_xai.ipynb`: 展示如何使用 Explainer 解釋推薦結果。

## 文檔索引

*   [專案架構說明](docs/architecture.md): 了解專案的目錄結構、模組設計與 Notebooks 用途。
*   [API 參考文件](docs/api_reference.md): 詳細的程式碼說明、函數定義與 Notebook 流程解析。
