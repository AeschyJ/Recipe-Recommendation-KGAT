import argparse
import os
import sys

import torch

# 允許載入 argparse.Namespace (PyTorch 2.6+ 安全性要求)
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([argparse.Namespace])

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.explainer_attention import KGATAttentionExplainer
from src.model.kgat_attention import KGATAttention
from src.train_att import get_adj_indices, load_data


def run():
    data_dir = os.path.join(project_root, "data", "processed")
    model_path = os.path.join(project_root, "models", "kgat_att_local_ckpt_e20.pth")

    if not os.path.exists(model_path):
        print(f"Error: Checkpoint {model_path} not found.")
        return

    # 1. Load Data
    interactions, kg_triples, stats = load_data(data_dir)
    n_users = stats["n_users"]
    n_items = stats["n_items"]
    n_entities = stats["n_entities"]
    n_relations = stats["n_relations"]

    # 2. Device Setup
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 3. Construct Graph
    indices, num_nodes = get_adj_indices(
        kg_triples, interactions, n_users, n_items, n_entities
    )
    indices = indices.to(device)

    # 4. Initialize Model
    # We need to get embed_dim and layers from the checkpoint if possible
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    embed_dim = 32
    layers = [32]
    if isinstance(checkpoint, dict) and "args" in checkpoint:
        args = checkpoint["args"]
        embed_dim = getattr(args, "embed_dim", 32)
        layers = getattr(args, "layers", [32])
        print(
            f"Hyperparameters loaded from checkpoint: embed_dim={embed_dim}, layers={layers}"
        )

    model = KGATAttention(
        n_users,
        n_items + n_entities,
        n_relations,
        embed_dim=embed_dim,
        layers=layers,
    ).to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print("Model loaded successfully.")
    model.eval()

    # 5. Explainer
    explainer = KGATAttentionExplainer(model)

    # Pick a user and a recommended item
    # For demonstration, we'll pick an item the user has interacted with
    target_user = 5
    user_interactions = interactions[interactions[:, 0] == target_user]

    if len(user_interactions) > 0:
        target_item = int(user_interactions[0][1])
        print(f"\n--- Explaining for User {target_user} and Item {target_item} ---")

        explanation = explainer.explain(
            indices, num_nodes, target_user, target_item, top_k=5
        )

        if explanation:
            print(f"Prediction Score: {explanation['target_score']:.4f}")
            print("\nTop Explanation Paths:")
            for path, score in explanation["top_paths"]:
                # Path contains global IDs. Let's try to label them.
                path_str = []
                for node_id in path:
                    if node_id < n_users:
                        path_str.append(f"User({node_id})")
                    elif node_id < n_users + n_items:
                        path_str.append(f"Recipe({node_id - n_users})")
                    else:
                        path_str.append(f"Entity({node_id - n_users - n_items})")
                print(" -> ".join(path_str) + f" (Score: {score:.6f})")

            # Since we can't show a plot easily in terminal, we just printed the paths.
            # In a real notebook, you'd call explainer.visualize(explanation)
            print("\nExplanation extraction successful!")
    else:
        print(f"User {target_user} has no interactions to explain.")


if __name__ == "__main__":
    run()
