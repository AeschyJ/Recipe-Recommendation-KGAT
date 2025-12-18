import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

# Import the new Attention-based KGAT
from src.model.kgat_attention import KGATAttention

# Try importing Intel Extension for PyTorch
try:
    import intel_extension_for_pytorch as ipex

    HAS_IPEX = True
except ImportError:
    HAS_IPEX = False
    print(
        "Warning: intel_extension_for_pytorch not found. Arc GPU acceleration will not be available."
    )


def load_data(data_dir="data/processed"):
    print(f"Loading data from {data_dir}...")
    interactions_path = os.path.join(data_dir, "interactions.pkl")
    kg_triples_path = os.path.join(data_dir, "kg_triples.pkl")
    stats_path = os.path.join(data_dir, "stats.pkl")

    df_interactions = pd.read_pickle(interactions_path)
    if isinstance(df_interactions, pd.DataFrame):
        interactions = df_interactions.values
    else:
        interactions = df_interactions

    with open(kg_triples_path, "rb") as f:
        kg_triples = pickle.load(f)

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    return interactions, kg_triples, stats


def construct_adj(kg_triples, n_users, n_items, n_entities):
    print("Constructing Sparse Edge List...")
    num_nodes = n_users + n_items + n_entities

    src = kg_triples[:, 0] + n_users
    dst = kg_triples[:, 2] + n_users + n_items

    all_src = np.concatenate([src, dst])
    all_dst = np.concatenate([dst, src])

    # Self-loops
    all_src = np.concatenate([all_src, np.arange(num_nodes)])
    all_dst = np.concatenate([all_dst, np.arange(num_nodes)])

    indices = torch.LongTensor(np.vstack([all_src, all_dst]))
    values = torch.ones(len(all_src))

    # For KGATAttention, the values in adj are just placeholders,
    # the attention weights are computed dynamically.
    # However, we still pass a SparseTensor for consistent API.
    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
    return adj


def sample_bpr_batch(interactions, n_items, batch_size):
    indices = np.random.randint(0, len(interactions), batch_size)
    batch_data = interactions[indices]

    u = torch.LongTensor(batch_data[:, 0])
    i = torch.LongTensor(batch_data[:, 1])
    j = torch.LongTensor(np.random.randint(0, n_items, batch_size))

    return u, i, j


def bpr_loss(pos_scores, neg_scores):
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))


def train():
    # Detect Intel Arc GPU (XPU)
    if HAS_IPEX and torch.xpu.is_available():
        device = torch.device("xpu")
        print("Using Intel Arc GPU (XPU)!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 1. Load Data
    try:
        interactions, kg_triples, stats = load_data()
    except FileNotFoundError:
        print("Error: Processed data not found.")
        return

    n_users = stats["n_users"]
    n_items = stats["n_items"]
    n_entities = stats["n_entities"]
    n_relations = stats["n_relations"]

    # 2. Construct Graph
    adj = construct_adj(kg_triples, n_users, n_items, n_entities).to(device)

    # 3. Model
    # Note: passing n_items + n_entities for total entity embeddings
    model = KGATAttention(n_users, n_items + n_entities, n_relations).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Optimize with IPEX if available
    if HAS_IPEX:
        model, optimizer = ipex.optimize(
            model, optimizer=optimizer, dtype=torch.float32
        )
        print("Model optimized with IPEX.")

    # 4. Train Loop
    epochs = 20
    batch_size = 1024
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = len(interactions) // batch_size

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch + 1}/{epochs}")
        for _ in pbar:
            u, i, j = sample_bpr_batch(interactions, n_items, batch_size)
            u, i, j = u.to(device), i.to(device), j.to(device)

            pos_scores = model(adj, u, i)
            neg_scores = model(adj, u, j)
            loss = bpr_loss(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch + 1} Avg Loss: {total_loss / n_batches:.4f}")

        if (epoch + 1) % 5 == 0:
            os.makedirs("models", exist_ok=True)
            # Save only state dict (compatible with CPU load)
            torch.save(model.state_dict(), f"models/kgat_att_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    train()
