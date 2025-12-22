import os
import pickle
import sys

import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Main script logic
data_dir = os.path.join(project_root, "data", "processed")
print(f"Data dir: {data_dir}")

# CACHE WARMING / DEBUG LOAD
interactions_path = os.path.join(data_dir, "interactions.pkl")
print("Pre-loading interactions to avoid OOM...")
try:
    with open(interactions_path, "rb") as f:
        df = pickle.load(f)
    print(f"Pre-load successful. Shape: {df.shape}")
except Exception as e:
    print(f"Error pre-loading: {e}")

# Now import modules
from src.train import construct_adj, load_data

print("Calling load_data...")
interactions, kg_triples, stats = load_data(data_dir)
print("load_data done.")

n_users = stats["n_users"]
n_items = stats["n_items"]
n_entities = stats["n_entities"]
n_relations = stats["n_relations"]

print("Calling construct_adj...")
adj = construct_adj(kg_triples, interactions, n_users, n_items, n_entities)
print("construct_adj done.")

if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
    print("Using Native Intel Arc GPU (XPU)!")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    print(f"Device: {device}")
adj = adj.to(device)
print("adj.to(device) done.")

print("Initializing model...")
from src.model.kgat import KGAT

# 確保 n_all_entities 與訓練時一致
n_all_entities = n_items + n_entities
model = KGAT(n_users, n_all_entities, n_relations).to(device)
print("Model initialized.")

# Load checkpoint
model_path = os.path.join(project_root, "models", "kgat_epoch_20.pth")
if not os.path.exists(model_path):
    print(f"Warning: {model_path} not found. Using initialized model.")
else:
    print(f"Loading model from {model_path}...")
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )

# Explainer
print("Initializing Explainer...")
from src.model.explainer import KGATExplainer

explainer = KGATExplainer(model)

target_user = 5

# 檢測 User 節點在圖中的連通性
user_idx_in_adj = target_user  # User ID is 0~n_users-1
row_indices = adj._indices()[0]
user_degree = (row_indices == user_idx_in_adj).sum()
print(f"Debug: User {target_user} degree in adj: {user_degree}")
if user_degree <= 1:  # 1 for self-loop
    print(
        "Warning: User node seems isolated (only self-loop?). Standard KGAT expects User-Item edges in adj."
    )

target_user = 5
user_interactions = interactions[interactions[:, 0] == target_user]

if len(user_interactions) > 0:
    target_item = user_interactions[0][1]  # recipe_id
    print(
        f"Explaining recommendation for User {target_user} -> Item {target_item} (Type: {type(target_item)})"
    )

    try:
        explanation = explainer.explain(adj, target_user, target_item, top_k=5)
        print("Scored:", explanation["target_score"])
        print("Top Paths:", explanation["top_paths"])
        print("Explanation extraction successful!")

    except Exception as e:
        print(f"Error explaining: {e}")
        import traceback

        traceback.print_exc()
else:
    print("User 0 has no interactions.")
