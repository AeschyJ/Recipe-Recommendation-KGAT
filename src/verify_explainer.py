import os
import sys

# 將腳本的上層目錄（即專案根目錄）加入 sys.path
# 假設 verify_explainer.py 位於 src/ 下
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch

from src.model.explainer import KGATExplainer
from src.model.kgat import KGAT
from src.train import construct_adj, load_data

# 1. 載入資料與統計
data_dir = os.path.join(project_root, "data", "processed")
interactions, kg_triples, stats = load_data(data_dir)

n_users = stats["n_users"]
n_items = stats["n_items"]
n_entities = stats["n_entities"]
n_relations = stats["n_relations"]

# 2. 建立 Adjacency Matrix
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

adj = construct_adj(kg_triples, n_users, n_items, n_entities).to(device)

# 3. 載入模型
# 確保 n_all_entities 與訓練時一致
n_all_entities = n_items + n_entities
model = KGAT(n_users, n_all_entities, n_relations).to(device)

# 嘗試載入最新的 checkpoint
model_path = os.path.join(project_root, "models", "kgat_epoch_20.pth")
if not os.path.exists(model_path):
    print(f"Warning: {model_path} not found. Using initialized model for demo.")
else:
    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))

# 4. 執行解釋
explainer = KGATExplainer(model)

# 選擇一個測試案例
# 例如: User 0 與他互動過的第一個 Item
target_user = 0
user_interactions = interactions[interactions[:, 0] == target_user]

if len(user_interactions) > 0:
    target_item = user_interactions[0][1]  # recipe_id

    print(f"Explaining recommendation for User {target_user} -> Item {target_item}")

    try:
        explanation = explainer.explain(adj, target_user, target_item, top_k=5)
        print("Scored:", explanation["target_score"])
        print("Top Paths:", explanation["top_paths"])

        # Visualize (如果是在非圖形介面環境，這可能會跳出視窗或報錯)
        # 為了驗證腳本順暢，我們只在能夠繪圖時繪圖，或是儲存圖片
        # explainer.visualize(explanation)
        print("Explanation extraction successful!")

    except Exception as e:
        print(f"Error explaining: {e}")
        import traceback

        traceback.print_exc()

else:
    print("User 0 has no interactions.")
