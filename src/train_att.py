import argparse
import gc
import os
import pickle
import sys

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
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/processed", help="Path to processed data"
    )
    parser.add_argument(
        "--model_dir", type=str, default="models", help="Directory to save models"
    )
    parser.add_argument("--embed_dim", type=int, default=32, help="Embedding dimension")
    # For layers, we can use a simple string parsing or fixed default.
    # Defaulting to [32] as per the final Colab configuration.
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[32], help="Layer sizes, e.g. 32 32"
    )
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU")
    return parser.parse_args()


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


def get_adj_indices(kg_triples, interactions, n_users, n_items, n_entities):
    """
    Construct edge indices for the graph.
    Returns: torch.LongTensor relevant for the device
    """
    print("Constructing graph indices with interactions...")
    num_nodes = n_users + n_items + n_entities

    # 1. KG Triples (Item <-> Entity)
    kg_src = kg_triples[:, 0] + n_users
    kg_dst = kg_triples[:, 2] + n_users + n_items

    # 2. Interactions (User <-> Item)
    int_src = interactions[:, 0]
    int_dst = interactions[:, 1] + n_users

    # Bi-directional graph
    all_src = np.concatenate([kg_src, kg_dst, int_src, int_dst, np.arange(num_nodes)])
    all_dst = np.concatenate([kg_dst, kg_src, int_dst, int_src, np.arange(num_nodes)])

    indices = np.vstack([all_src, all_dst])
    return torch.LongTensor(indices), num_nodes


def train(args):
    # 1. Device Setup
    if args.cpu:
        device = torch.device("cpu")
        print("Forced to use CPU")
    elif HAS_XPU:
        device = torch.device("xpu")
        print("Using Native Intel Arc GPU (XPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

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

    # 3. Construct Graph
    indices, num_nodes = get_adj_indices(
        kg_triples, interactions, n_users, n_items, n_entities
    )

    indices = indices.to(device)

    # Release raw triples memory
    del kg_triples
    gc.collect()

    # 4. Model Initialization
    print(
        f"Initializing KGATAttention with embed_dim={args.embed_dim}, layers={args.layers}"
    )
    model = KGATAttention(
        n_users,
        n_items + n_entities,
        n_relations,
        embed_dim=args.embed_dim,
        layers=args.layers,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # AMP Scaler (Conditional)
    scaler = None
    if device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        print("Enabled CUDA AMP GradScaler")

    # 5. Resume Checkpoint
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

        # Determine if it's a full checkpoint or state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # 智慧恢復參數：若使用者沒在命令列指定新值，則從 Checkpoint 恢復
            saved_args = checkpoint.get("args")
            if saved_args:
                print("Restoring hyperparameters from checkpoint...")
                if args.batch_size == 1024 and hasattr(saved_args, "batch_size"):
                    args.batch_size = saved_args.batch_size
                if args.embed_dim == 32 and hasattr(saved_args, "embed_dim"):
                    args.embed_dim = saved_args.embed_dim
                if args.layers == [32] and hasattr(saved_args, "layers"):
                    args.layers = saved_args.layers
                # 注意：lr 在此腳本中目前是 hardcoded 1e-3, 若未來加入參數也可在此恢復

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scaler_state_dict" in checkpoint and scaler:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            print(
                f"Resumed from epoch {start_epoch} (Batch Size: {args.batch_size}, Embed Dim: {args.embed_dim})"
            )
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model weights (state_dict only)")

    # 6. IPEX Optimization (must be after model load)
    # 6. Model Optimization (Native XPU supports torch.compile)
    if HAS_XPU:
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile for XPU.")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")

    # 7. Training Loop
    os.makedirs(args.model_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        n_batches = len(interactions) // args.batch_size

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch + 1}/{args.epochs}")

        for _ in pbar:
            idx = np.random.randint(0, len(interactions), args.batch_size)
            batch = interactions[idx]

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
                    pos = model(indices, num_nodes, u, i)
                    neg = model(indices, num_nodes, u, j)
                    loss = -torch.mean(torch.log(torch.sigmoid(pos - neg) + 1e-10))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            elif device.type == "xpu":
                # XPU specific AMP context (if supported) or LP optimization by IPEX
                # For safety given previous OOMs, we run standard FP32 or BF16 if IPEX auto-mixes.
                # Explicit autocast for XPU:
                # with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
                # But let's stick to safe FP32 first unless user asks.
                pos = model(indices, num_nodes, u, i)
                neg = model(indices, num_nodes, u, j)
                loss = -torch.mean(torch.log(torch.sigmoid(pos - neg) + 1e-10))

                loss.backward()
                optimizer.step()

            else:
                # CPU
                pos = model(indices, num_nodes, u, i)
                neg = model(indices, num_nodes, u, j)
                loss = -torch.mean(torch.log(torch.sigmoid(pos - neg) + 1e-10))
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch + 1} done. Avg Loss: {avg_loss:.4f}")

        # Explicit GC
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "xpu":
            torch.xpu.empty_cache()

        # Save Checkpoint
        if (epoch + 1) % 2 == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(
                args.model_dir, f"kgat_att_local_ckpt_e{epoch + 1}.pth"
            )
            save_dict = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "args": args,
            }
            if scaler:
                save_dict["scaler_state_dict"] = scaler.state_dict()

            torch.save(save_dict, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
