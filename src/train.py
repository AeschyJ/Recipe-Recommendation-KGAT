import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from src.model.kgat import KGAT

# 檢查 Native XPU 支援
HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()


def load_data(data_dir="data/processed"):
    print(f"Loading data from {data_dir}...")

    # 使用正確的讀取方式
    interactions_path = os.path.join(data_dir, "interactions.pkl")
    kg_triples_path = os.path.join(data_dir, "kg_triples.pkl")
    stats_path = os.path.join(data_dir, "stats.pkl")

    with open(interactions_path, "rb") as f:
        df_interactions = pickle.load(f)
    print("Loaded interactions.")
    if isinstance(df_interactions, pd.DataFrame):
        interactions = df_interactions.values
    else:
        interactions = df_interactions

    with open(kg_triples_path, "rb") as f:
        kg_triples = pickle.load(f)
    print("Loaded kg_triples.")

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    print("Loaded stats.")

    return interactions, kg_triples, stats


def construct_adj(kg_triples, interactions, n_users, n_items, n_entities):
    print("Constructing Sparse Adjacency Matrix with Interactions...")
    # 總節點數 = User + Item + Entity
    num_nodes = n_users + n_items + n_entities

    # 1. KG Triples (Item <-> Entity)
    # src: Item (offset by n_users)
    kg_src = kg_triples[:, 0] + n_users
    # dst: Entity (offset by n_users + n_items)
    kg_dst = kg_triples[:, 2] + n_users + n_items

    # 2. Interactions (User <-> Item)
    # src: User (0 ~ n_users-1)
    # dst: Item (n_users ~ n_users + n_items - 1)
    # interactions column 0 is user_id, column 1 is item_id
    int_src = interactions[:, 0]
    int_dst = interactions[:, 1] + n_users

    # Combine all edges (Bidirectional)
    # KG edges
    all_src = np.concatenate([kg_src, kg_dst])
    all_dst = np.concatenate([kg_dst, kg_src])

    # Interaction edges
    all_src = np.concatenate([all_src, int_src, int_dst])
    all_dst = np.concatenate([all_dst, int_dst, int_src])

    # 加入自環 (Self-loop) 給每一個節點
    all_src = np.concatenate([all_src, np.arange(num_nodes)])
    all_dst = np.concatenate([all_dst, np.arange(num_nodes)])

    # 轉換為 Tensor
    # 這裡必須確保 indices 是 LongTensor 且 row 0 是 src, row 1 是 dst (COO 格式通常是 [row, col])
    indices = torch.LongTensor(np.vstack([all_src, all_dst]))
    values = torch.ones(len(all_src))

    # Row-Normalization
    # 計算度數 (Degree)
    # 使用 bincount 計算每個節點的度數
    deg = torch.bincount(indices[0], minlength=num_nodes).float()
    deg[deg == 0] = 1  # 避免除以 0

    # 這裡採用 Mean Aggregation: 將 edge value 除以 degree (D^-1 * A)
    norm_values = values / deg[all_src]

    # 建立正式的規一化稀疏矩陣
    adj_norm = torch.sparse_coo_tensor(indices, norm_values, (num_nodes, num_nodes))

    # 對於 XPU，CSR (Compressed Sparse Row) 格式通常有更好的運算支援與效能
    # 我們確保返回的是 Coalesced 的 COO，或者直接轉為 CSR
    return adj_norm.coalesce()


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


def parse_args():
    parser = argparse.ArgumentParser(description="Train KGAT Model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--use_bf16", action="store_true", help="Use BFloat16 on XPU")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--model_dir", type=str, default="models", help="Directory to save models"
    )
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU")
    return parser.parse_args()


def train():
    args = parse_args()

    # 設定環境
    if args.cpu:
        device = torch.device("cpu")
        print("Forced to use CPU")
    elif HAS_XPU:
        device = torch.device("xpu")
        print("Using Native Intel Arc GPU (XPU)!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

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
    n_entities = stats["n_entities"]
    n_relations = stats["n_relations"]

    # 2. 建立鄰接矩陣 (User + Item + Entity)
    print("Constructing adjacency matrix (CPU)...")
    adj = construct_adj(kg_triples, interactions, n_users, n_items, n_entities)
    print(f"Adjacency matrix created. Edges: {adj._nnz()}")

    print(f"Moving adjacency matrix to {device}...")
    adj = adj.to(device)
    if not adj.is_coalesced():
        print("Coalescing adjacency matrix on device...")
        adj = adj.coalesce()

    if args.use_bf16 and device.type == "xpu":
        adj = adj.bfloat16()
        # 型別轉換可能導致 coalesced 狀態遺失，再次檢查
        if not adj.is_coalesced():
            adj = adj.coalesce()
        print("Adjacency matrix cast to BFloat16.")
    print(f"Done. Coalesced: {adj.is_coalesced()}")

    # 3. 初始化模型
    print("Initializing KGAT model...")
    n_all_entities = n_items + n_entities
    model = KGAT(n_users, n_all_entities, n_relations, embed_dim=args.embed_dim).to(
        device
    )
    if args.use_bf16 and device.type == "xpu":
        model = model.bfloat16()
        print("Model cast to BFloat16.")

    print("Model moved to device.")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # 4. 斷點續訓
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # 智慧恢復參數：若使用者沒在命令列指定，則從 Checkpoint 恢復
            saved_args = checkpoint.get("args")
            if saved_args:
                print("Restoring hyperparameters from checkpoint...")
                # 由於是在 train 內，這裡簡單透過 args 是否等於預設值來判斷
                # 如果目前的 args.batch_size 是預設值 1024，而 saved_args 有不同值，則恢復
                if args.batch_size == 1024 and hasattr(saved_args, "batch_size"):
                    args.batch_size = saved_args.batch_size
                if args.lr == 1e-3 and hasattr(saved_args, "lr"):
                    args.lr = saved_args.lr
                if args.embed_dim == 64 and hasattr(saved_args, "embed_dim"):
                    args.embed_dim = saved_args.embed_dim
                # 保持使用目前的 device/cpu 參數，不從 checkpoint 覆蓋硬體環境

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            print(
                f"Resumed from epoch {start_epoch} (Batch Size: {args.batch_size}, LR: {args.lr})"
            )
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model weights (state_dict only)")

    if HAS_XPU:
        torch.xpu.empty_cache()

    # 5. 訓練迴圈
    epochs = args.epochs
    batch_size = args.batch_size
    print(f"Starting training from epoch {start_epoch} to {epochs}...")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        n_batches = len(interactions) // batch_size

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch + 1}/{epochs}")
        for _ in pbar:
            u, i, j = sample_bpr_batch(interactions, n_items, batch_size)
            u, i, j = u.to(device), i.to(device), j.to(device)

            # 如果使用 BF16，模型輸出會是 BF16
            try:
                # 僅調用一次前向傳播，同時獲得正負樣本分數
                pos_scores, neg_scores = model(adj, u, i, j)

                # 計算 Loss，轉回 Float32 以保證數值穩定性
                loss = bpr_loss(pos_scores.float(), neg_scores.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            except RuntimeError as e:
                print(f"\nRuntimeError during training: {e}")
                import traceback

                traceback.print_exc()
                return

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch + 1} Complete. Average Loss: {avg_loss:.4f}")

        # 6. 儲存模組 (每 2 個 epoch 或是最後一個儲存一次)
        if (epoch + 1) % 2 == 0 or (epoch + 1) == epochs:
            os.makedirs(args.model_dir, exist_ok=True)
            ckpt_path = os.path.join(
                args.model_dir, f"kgat_checkpoint_e{epoch + 1}.pth"
            )
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "args": args,
            }
            torch.save(checkpoint, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    train()
