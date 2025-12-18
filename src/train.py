import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from src.model.kgat import KGAT


def load_data(data_dir="data/processed"):
    print(f"Loading data from {data_dir}...")

    # 使用正確的讀取方式
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
    print("Constructing Sparse Adjacency Matrix...")
    # 總節點數 = User + Item + Entity
    num_nodes = n_users + n_items + n_entities

    # 映射邏輯:
    # Users: 0 ~ n_users-1
    # Items: n_users ~ n_users + n_items - 1
    # Entities: n_users + n_items ~ num_nodes - 1

    # kg_triples: [Item, Relation, Entity]
    # src (Item) 需要偏移 n_users
    # item_id 是 0-indexed 的，所以加上 n_users 變成全域 ID
    src = kg_triples[:, 0] + n_users

    # dst (Entity) 需要偏移 n_users + n_items
    # entity_id 是 0-indexed 的，所以加上 n_users + n_items 變成全域 ID
    dst = kg_triples[:, 2] + n_users + n_items

    # 建立雙向邊: Item <-> Entity
    all_src = np.concatenate([src, dst])
    all_dst = np.concatenate([dst, src])

    # 加入自環 (Self-loop) 給每一個節點
    all_src = np.concatenate([all_src, np.arange(num_nodes)])
    all_dst = np.concatenate([all_dst, np.arange(num_nodes)])

    # 轉換為 Tensor
    # 這裡必須確保 indices 是 LongTensor 且 row 0 是 src, row 1 是 dst (COO 格式通常是 [row, col])
    indices = torch.LongTensor(np.vstack([all_src, all_dst]))
    values = torch.ones(len(all_src))

    # 建立稀疏矩陣
    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

    # Row-Normalization
    # 計算度數 (Degree)
    # 使用 bincount 計算每個節點的度數
    deg = torch.bincount(indices[0], minlength=num_nodes).float()
    deg[deg == 0] = 1  # 避免除以 0

    # Symmetric Normalization: D^-0.5 * A * D^-0.5 或 Mean: D^-1 * A
    # 這裡採用 Mean Aggregation 的前處理: 將 edge value 除以 degree (D^-1 * A)
    # 注意: indices[0] 就是 row index
    norm_values = values / deg[all_src]

    adj_norm = torch.sparse_coo_tensor(indices, norm_values, (num_nodes, num_nodes))
    return adj_norm


def sample_bpr_batch(interactions, n_items, batch_size):
    """隨機採樣 (User, Pos_Item, Neg_Item) 用於 BPR Loss"""
    indices = np.random.randint(0, len(interactions), batch_size)
    batch_data = interactions[indices]

    u = torch.LongTensor(batch_data[:, 0])  # user_id_remap
    i = torch.LongTensor(batch_data[:, 1])  # recipe_id_remap

    # 負採樣：隨機選一個該使用者沒互動過的物品 (簡化實作)
    j = torch.LongTensor(np.random.randint(0, n_items, batch_size))

    return u, i, j


def bpr_loss(pos_scores, neg_scores):
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))


def train():
    # 設定環境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 載入資料
    try:
        interactions, kg_triples, stats = load_data()
    except FileNotFoundError:
        print(
            "Error: Processed data not found. Please run 'python src/data/preprocess.py' first."
        )
        return

    n_users = stats["n_users"]
    n_items = stats["n_items"]
    n_entities = stats["n_entities"]  # 這是 ingredients + tags 的數量
    n_relations = stats["n_relations"]

    # 2. 建立鄰接矩陣 (User + Item + Entity)
    # 我們需要正確傳入 n_items 讓 construct_adj 計算偏移與總節點數
    adj = construct_adj(kg_triples, n_users, n_items, n_entities).to(device)

    # 3. 初始化模型
    # KGAT 的 entity_embed 對應的是 "Items + Entities"
    # 所以 n_entities 參數應該傳入 n_items + n_entities
    n_all_entities = n_items + n_entities

    model = KGAT(n_users, n_all_entities, n_relations).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # 4. 訓練迴圈
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

            try:
                # 正樣本與負樣本分數
                pos_scores = model(adj, u, i)
                neg_scores = model(adj, u, j)

                loss = bpr_loss(pos_scores, neg_scores)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            except RuntimeError as e:
                print(f"\nRuntimeError: {e}")
                print(
                    f"Debug: u max={u.max()}, i max={i.max()}, n_users={n_users}, n_items={n_items}"
                )
                return

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch + 1} Complete. Average Loss: {avg_loss:.4f}")

        # 5. 儲存模組 (每 5 個 epoch 儲存一次)
        if (epoch + 1) % 5 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/kgat_epoch_{epoch + 1}.pth")
            print(f"Model saved to models/kgat_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    train()
