import argparse
import gc
import logging
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

# Add project root to path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.kgat_attention import KGATAttention

# 檢查 Native XPU 支援
HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()


def parse_args():
    parser = argparse.ArgumentParser(description="Train KGAT Attention Model")
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs to train"
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument(
        "--without_kg",
        action="store_true",
        help="Ablation: Train without Knowledge Graph triples",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/processed", help="Path to processed data"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory to save models",
    )
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    # For layers, we can use a simple string parsing or fixed default.
    # Defaulting to [64] as per the final Colab configuration.
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[64], help="Layer sizes, e.g. 64 64"
    )
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU")
    parser.add_argument(
        "--debug", action="store_true", help="Run with small data for debugging"
    )
    parser.add_argument(
        "--use_bf16", action="store_true", help="Use BFloat16 precision"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--no_compile", action="store_true", help="Disable torch.compile"
    )
    parser.add_argument(
        "--log_dir", type=str, default="output/logs", help="Directory to save logs"
    )
    return parser.parse_args()


def setup_logging(log_dir, model_name="kgat"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{model_name}_{timestamp}.txt")

    # 設定 Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_file


def load_data(data_dir):
    print(f"Loading data from {data_dir}...")
    with open(os.path.join(data_dir, "interactions.pkl"), "rb") as f:
        interactions = pickle.load(f)
    if isinstance(interactions, pd.DataFrame):
        interactions = interactions.values

    with open(os.path.join(data_dir, "kg_triples.pkl"), "rb") as f:
        kg_triples = pickle.load(f)

    with open(os.path.join(data_dir, "stats.pkl"), "rb") as f:
        stats = pickle.load(f)

    return interactions, kg_triples, stats


def get_adj_indices(
    kg_triples, interactions, n_users, n_items, n_entities, without_kg=False
):
    """
    Construct edge indices for the graph.
    Returns: torch.LongTensor relevant for the device
    """
    print("Constructing graph indices with interactions...")
    num_nodes = n_users + n_items + n_entities

    if without_kg:
        print(
            "Ablation: Running without Knowledge Graph (only User-Item bipartite graph)"
        )
        kg_src = np.array([], dtype=np.int64)
        kg_dst = np.array([], dtype=np.int64)
    else:
        # 1. KG Triples (Item <-> Entity)
        kg_src = kg_triples[:, 0] + n_users
        kg_dst = kg_triples[:, 2] + n_users + n_items

    # 2. Interactions (User <-> Item)
    int_src = interactions[:, 0]
    int_dst = interactions[:, 1] + n_users

    # Bi-directional graph
    all_src = np.concatenate([kg_src, kg_dst, int_src, int_dst, np.arange(num_nodes)])
    all_dst = np.concatenate([kg_dst, kg_src, int_dst, int_src, np.arange(num_nodes)])

    # Construct Edge Types
    # KG Triples: Relations are 0 (Ingredient) or 1 (Tag)
    # Inverse KG Triples: Keep same relation or map to new ones? Usually same for similarity.
    if without_kg:
        kg_rels = np.array([], dtype=np.int64)
    else:
        kg_rels = kg_triples[:, 1]

    # Interactions: Let's assign relation ID 2 for User-Item
    # Self-loops: Assign relation ID 3

    n_kg = len(kg_src)
    n_int = len(int_src)
    n_self = num_nodes

    # 這裡我們簡單定義：
    # KG Relations: 0, 1 (原樣)
    # Inverse KG: 0, 1 (Symetric semantic)
    # Interaction: 2
    # Inverse Interaction: 2
    # Self-loop: 3

    rels_kg = kg_rels
    rels_kg_inv = kg_rels  # Reuse same relation ID for inverse
    rels_int = np.full(n_int, 2)
    rels_int_inv = np.full(n_int, 2)
    rels_self = np.full(n_self, 3)

    all_rels = np.concatenate([rels_kg, rels_kg_inv, rels_int, rels_int_inv, rels_self])

    indices = np.vstack([all_src, all_dst])
    edge_types = torch.LongTensor(all_rels)

    return torch.LongTensor(indices), edge_types, num_nodes


def evaluate(
    model, interactions, indices, edge_types, num_nodes, n_items, device="cpu"
):
    """
    Validation Metric: Recall@K
    計算模型在測試集上的 Recall@10, @20, @50。
    為了效率，這裡使用隨機負採樣進行評估 (100個負樣本 + 1個正樣本)。
    """
    model.eval()
    hits_10, hits_20, hits_50 = 0, 0, 0
    total = 0

    batch_size = 512
    n_test = len(interactions)

    # 1. 預先計算並快取全圖最終特徵向量 (可將評估速度由數分鐘縮短至不到一秒)
    with torch.no_grad():
        final_embed = model.get_final_embeddings(indices, edge_types, num_nodes)

    # 2. 生成測試 Batch
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            end = min(i + batch_size, n_test)
            batch = interactions[i:end]

            user_ids = torch.LongTensor(batch[:, 0]).to(device)
            item_ids = torch.LongTensor(batch[:, 1]).to(device)

            # 正樣本分數 (直接 Lookup Cache，不跑 Model Forward)
            u_embed = final_embed[user_ids]
            pos_i_embed = final_embed[model.n_users + item_ids]
            pos_scores = torch.sum(u_embed * pos_i_embed, dim=1)

            # 負樣本評估
            # Repeat Users: [u1, u1... (100 times), u2, u2...]
            users_expanded = user_ids.unsqueeze(1).repeat(1, 100).view(-1)

            # Random Items (0 to n_items-1)
            neg_items = torch.randint(0, n_items, (len(users_expanded),)).to(device)
            neg_items_flatten = neg_items.view(-1)

            u_embed_expanded = final_embed[users_expanded]
            neg_i_embed = final_embed[model.n_users + neg_items_flatten]
            neg_scores = torch.sum(u_embed_expanded * neg_i_embed, dim=1).view(len(batch), 100)

            # Concat positive and negative scores
            all_scores = torch.cat(
                [pos_scores.unsqueeze(1), neg_scores], dim=1
            )  # (B, 101)

            # Calculate Rank using topk
            top50_indices = torch.topk(all_scores, k=50, dim=1).indices

            hits_10 += torch.sum((top50_indices[:, :10] == 0).any(dim=1)).item()
            hits_20 += torch.sum((top50_indices[:, :20] == 0).any(dim=1)).item()
            hits_50 += torch.sum((top50_indices[:, :50] == 0).any(dim=1)).item()
            total += len(batch)

    return hits_10 / total, hits_20 / total, hits_50 / total


def train(args):
    # 0. 初始化 Logging
    log_file = setup_logging(args.log_dir, model_name="kgat")
    logging.info(f"Training started. Args: {args}")
    logging.info(f"Log file: {log_file}")

    # 1. Device Setup
    if args.cpu:
        device = torch.device("cpu")
        logging.info("Using CPU")
    elif HAS_XPU:
        device = torch.device("xpu")
        logging.info("Using Intel Arc GPU (XPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")

    # Memory Cleanup
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()

    # 2. Data Loading
    interactions, kg_triples, stats = load_data(args.data_dir)
    n_users = stats["n_users"]
    n_items = stats["n_items"]
    n_entities = stats["n_entities"]
    n_relations = stats["n_relations"]

    if args.debug:
        logging.info("DEBUG MODE: Using small subset of data")
        interactions = interactions[:2000]
        args.batch_size = 512
        args.epochs = 1

    # 3. Construct Graph
    # 3. Construct Graph
    indices, edge_types, num_nodes = get_adj_indices(
        kg_triples,
        interactions,
        n_users,
        n_items,
        n_entities,
        without_kg=args.without_kg,
    )

    indices = indices.to(device)
    edge_types = edge_types.to(device)

    # Train/Test Split
    np.random.shuffle(interactions)
    split_idx = int(len(interactions) * 0.8)
    train_data = interactions[:split_idx]
    test_data = interactions[split_idx:]

    logging.info(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    # Batch Size Check
    expected_iterations = len(train_data) // args.batch_size
    logging.info(f"Expected iterations per epoch: {expected_iterations}")
    if expected_iterations < 10:
        logging.warning(
            "Warning: Iterations per epoch is very low. Consider reducing batch_size."
        )

    # Release raw triples memory

    # Release raw triples memory
    del kg_triples
    gc.collect()

    # 5. 斷點續訓與超參數恢復
    start_epoch = 0
    checkpoint = None
    if args.resume and os.path.exists(args.resume):
        logging.info(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            saved_args = checkpoint.get("args")
            if saved_args:
                logging.info("Restoring hyperparameters from checkpoint...")
                if args.batch_size == 1024 and hasattr(saved_args, "batch_size"):
                    args.batch_size = saved_args.batch_size
                if args.embed_dim == 64 and hasattr(saved_args, "embed_dim"):
                    args.embed_dim = saved_args.embed_dim
                if args.layers == [64] and hasattr(saved_args, "layers"):
                    args.layers = saved_args.layers
            start_epoch = checkpoint.get("epoch", 0)
        else:
            # state_dict only mode
            pass

    # 4. Model Initialization
    logging.info(
        f"Initializing KGATAttention with embed_dim={args.embed_dim}, layers={args.layers}"
    )
    model = KGATAttention(
        n_users,
        n_items + n_entities,
        n_relations + 2,  # Added 2 relations (Int, Self)
        embed_dim=args.embed_dim,
        layers=args.layers,
    ).to(device)

    # 4.1. BF16 Optimization
    if args.use_bf16:
        logging.info("Enabled BFloat16 precision")
        model = model.bfloat16()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # AMP Scaler (Conditional)
    scaler = None
    if device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        logging.info("Enabled CUDA AMP GradScaler")

    # 6. 載入權重與狀態
    if checkpoint is not None:
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            missing, unexpected = model.load_state_dict(
                checkpoint["model_state_dict"], strict=False
            )
            if missing or unexpected:
                logging.warning(
                    f"Architecture mismatch! Missing: {missing}, Unexpected: {unexpected}"
                )
                logging.warning(
                    "Optimizer/Scheduler/Scaler will NOT be loaded due to mismatch."
                )
            else:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if "scheduler_state_dict" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                if "scaler_state_dict" in checkpoint and scaler:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
            logging.info(
                f"Resumed from epoch {start_epoch} (Batch Size: {args.batch_size}, Embed Dim: {args.embed_dim}, Layers: {args.layers})"
            )
        else:
            model.load_state_dict(checkpoint, strict=False)
            logging.info("Loaded model weights (state_dict only, strict=False)")

    # 6. IPEX Optimization (must be after model load)
    # 6. Model Optimization (Native XPU supports torch.compile)
    if HAS_XPU and not args.cpu and not args.no_compile:
        try:
            model = torch.compile(model)
            logging.info("Model compiled with torch.compile for XPU.")
        except Exception as e:
            logging.warning(f"Warning: torch.compile failed: {e}")
    elif args.no_compile:
        logging.info("Model compilation disabled by user.")

    # 7. Training Loop
    os.makedirs(args.model_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        n_batches = len(train_data) // args.batch_size

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch + 1}/{args.epochs}")

        for _ in pbar:
            idx = np.random.randint(0, len(train_data), args.batch_size)
            batch = train_data[idx]

            u = torch.LongTensor(batch[:, 0]).to(device)
            i = torch.LongTensor(batch[:, 1]).to(device)
            j = torch.LongTensor(np.random.randint(0, n_items, args.batch_size)).to(
                device
            )

            # Memory optimization: set_to_none=True
            optimizer.zero_grad(set_to_none=True)

            # Forward & Loss
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    pos_scores, neg_scores = model(indices, edge_types, num_nodes, u, i, j)
                    loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            elif device.type == "xpu":
                # XPU specific AMP context (if supported) or LP optimization by IPEX
                with torch.autocast(
                    device_type="xpu", enabled=args.use_bf16, dtype=torch.bfloat16
                ):
                    pos_scores, neg_scores = model(indices, edge_types, num_nodes, u, i, j)
                    loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))

                loss.backward()
                optimizer.step()

            else:
                # CPU
                with torch.autocast(
                    device_type="cpu", enabled=args.use_bf16, dtype=torch.bfloat16
                ):
                    pos_scores, neg_scores = model(indices, edge_types, num_nodes, u, i, j)
                    loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / n_batches
        logging.info(f"Epoch {epoch + 1} done. Avg Loss: {avg_loss:.4f}")

        # Validation Per Epoch
        recall_10, recall_20, recall_50 = evaluate(
            model, test_data, indices, edge_types, num_nodes, n_items, device=device
        )
        logging.info(
            f"Epoch {epoch + 1} Evaluation - Recall@10: {recall_10:.4f}, Recall@20: {recall_20:.4f}, Recall@50: {recall_50:.4f}"
        )

        # Step Scheduler
        scheduler.step(recall_20)
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(f"Epoch {epoch + 1} Current LR: {current_lr:.6e}")

        # Explicit GC
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "xpu":
            torch.xpu.empty_cache()

        # Save Checkpoint
        if (epoch + 1) % 2 == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(
                args.model_dir, f"kgat_checkpoint_e{epoch + 1}.pth"
            )
            save_dict = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
                "args": args,
            }
            if scaler:
                save_dict["scaler_state_dict"] = scaler.state_dict()

            torch.save(save_dict, ckpt_path)
            logging.info(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
