# 資料字典 (Data Dictionary)

本文件詳細說明 Food.com 資料集的欄位定義，以及經由預處理後產生的知識圖譜結構。

## 原始資料 (Raw Data)

原始資料來自 Kaggle Food.com Recipes and Interactions。請確保檔案存放於 `data/raw/`。

### `RAW_recipes.csv`

包含食譜的詳細資訊。

| 欄位名稱 | 類型 | 說明 |
| :--- | :--- | :--- |
| `id` | int | 食譜唯一識別碼 (原始 ID) |
| `name` | string | 食譜名稱 |
| `minutes` | int | 烹飪時間 (分鐘) |
| `submitted` | date | 上傳日期 |
| `tags` | list (str) | 標籤列表 (如 ['60-minutes-or-less', 'time-to-make', ...]) |
| `nutrition` | list (float) | 營養成分 (cal, fat, sugar, sodium, protein, sat. fat, carbs) |
| `n_steps` | int | 步驟數量 |
| `steps` | list (str) | 烹飪步驟描述 |
| `description` | string | 使用者提供的食譜描述 |
| `ingredients` | list (str) | 成分列表 (如 ['winter squash', 'mexican seasoning', ...]) |
| `n_ingredients`| int | 成分數量 |

### `RAW_interactions.csv`

包含使用者對食譜的評分與評論。

| 欄位名稱 | 類型 | 說明 |
| :--- | :--- | :--- |
| `user_id` | int | 使用者唯一識別碼 (原始 ID) |
| `recipe_id` | int | 食譜 ID (對應 `RAW_recipes.csv` 的 `id`) |
| `date` | date | 互動日期 |
| `rating` | int | 評分 (1-5) |
| `review` | string | 文字評論 |

---

## 預處理資料 (Processed Data)

預處理腳本 `src/data/preprocess.py` 會產生以下 Pickle 檔案，存放於 `data/processed/`。

### 1. `interactions.pkl` (DataFrame)

用於模型訓練的使用者-物品互動矩陣。

| 欄位 | 說明 |
| :--- | :--- |
| `user_id_remap` | [0, n_users) 的連續整數 ID |
| `recipe_id_remap` | [0, n_items) 的連續整數 ID |
| `rating` | 原始評分 |

### 2. `kg_triples.pkl` (Numpy Array)

知識圖譜的三元組 `(Head, Relation, Tail)`。

*   **Head**: 食譜 ID (`recipe_id_remap`)
*   **Relation**: 關係類型 ID
    *   `0`: **Has_Ingredient** (食譜包含某成分)
    *   `1`: **Has_Tag** (食譜擁有某標籤)
*   **Tail**: 實體 ID (Entity ID)
    *   實體 ID 範圍從 `0` 開始編號。
    *   成分 (Ingredients) 與標籤 (Tags) 共享同一個 ID 空間，但彼此 ID 不重疊。

### 3. `stats.pkl` (Dict)

儲存統計資訊與映射表，用於推論時還原原始資訊。

*   `n_users`: 使用者總數
*   `n_items`: 物品 (食譜) 總數
*   `n_entities`: 知識圖譜實體 (成分+標籤) 總數
*   `user_map`: `sklearn.preprocessing.LabelEncoder` 物件 (User ID 轉換)
*   `item_map`: `sklearn.preprocessing.LabelEncoder` 物件 (Recipe ID 轉換)
*   `ingredient_map`: `Dict[str, int]` (成分名稱 -> Entity ID)
*   `tag_map`: `Dict[str, int]` (標籤名稱 -> Entity ID)
