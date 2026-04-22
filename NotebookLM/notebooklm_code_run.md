# Source Code: Training & Run Scripts

## File: main.py

```python
def main():
    print("Hello from experiment!")


if __name__ == "__main__":
    main()

```

## File: run_experiments.bat

```text
@echo off
pushd "%~dp0"
echo Starting KGAT Ablation Studies...

if not exist .venv\Scripts\python.exe (
    echo [Error] .venv\Scripts\python.exe not found!
    echo Please ensure the project was initialized in this folder.
    popd
    exit /b 1
)

@REM echo ==============================================
@REM echo [Exp 1/5] Full KGAT (L=1, Attention + Bi-Interaction)
@REM echo ==============================================
@REM .venv\Scripts\python.exe src/train_att.py --epochs 10 --layers 64 --model_dir models/full_kgat --log_dir output/logs/full_kgat --use_bf16 --no_compile

@REM echo ==============================================
@REM echo [Exp 2/5] w/o Attention (KGAT-a, Bi-Interaction Only)
@REM echo ==============================================
@REM .venv\Scripts\python.exe src/train_bi_interaction.py --epochs 10 --layers 64 --model_dir models/wo_attn --log_dir output/logs/wo_attn --use_bf16

@REM echo ==============================================
@REM echo [Exp 3/5] w/o Knowledge Graph (Interaction Only)
@REM echo ==============================================
@REM .venv\Scripts\python.exe src/train_att.py --epochs 10 --layers 64 --without_kg --model_dir models/wo_kg --log_dir output/logs/wo_kg --use_bf16 --no_compile

@REM echo ==============================================
@REM echo [Exp 4/5] Depth Variation L=2
@REM echo ==============================================
@REM .venv\Scripts\python.exe src/train_att.py --epochs 10 --layers 64 64 --model_dir models/depth_2 --log_dir output/logs/depth_2 --use_bf16 --no_compile

echo ==============================================
echo [Exp 5/5] Depth Variation L=3
echo ==============================================
.venv\Scripts\python.exe src/train_att.py --epochs 3 --layers 64 64 64 --model_dir models/depth_3 --log_dir output/logs/depth_3 --use_bf16 --no_compile

echo ==============================================
echo All experiments completed!
echo ==============================================
popd

```

## File: src\train_att.py

```python
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
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation on the loaded checkpoint and exit")
    parser.add_argument("--log_file", type=str, default=None, help="Explicitly specify the log file to use (append mode)")
    return parser.parse_args()


def setup_logging(log_dir, model_name="kgat", log_file=None):
    if log_file is None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{model_name}_{timestamp}.txt")
        file_mode = "w"
    else:
        file_mode = "a"

    # 移除舊的 handlers 以避免重複輸出
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 設定 Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8", mode=file_mode),
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
    Validation Metric: HR@K, Precision@K, NDCG@K
    計算模型在測試集上的表現。(100個負樣本 + 1個正樣本)
    """
    model.eval()
    
    metrics = {
        'hr_10': 0, 'hr_20': 0, 'hr_50': 0,
        'ndcg_10': 0, 'ndcg_20': 0, 'ndcg_50': 0,
        'prec_10': 0, 'prec_20': 0, 'prec_50': 0,
    }
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
            neg_scores = torch.sum(u_embed_expanded * neg_i_embed, dim=1).view(
                len(batch), 100
            )

            # Concat positive and negative scores
            all_scores = torch.cat(
                [pos_scores.unsqueeze(1), neg_scores], dim=1
            )  # (B, 101)

            # Calculate metrics
            _, sorted_indices = torch.sort(all_scores, dim=1, descending=True)
            pos_ranks = (sorted_indices == 0).nonzero(as_tuple=True)[1]

            for k in [10, 20, 50]:
                hits = (pos_ranks < k).float()
                metrics[f'hr_{k}'] += hits.sum().item()
                metrics[f'prec_{k}'] += (hits / k).sum().item()
                metrics[f'ndcg_{k}'] += (hits / torch.log2(pos_ranks.float() + 2)).sum().item()

            total += len(batch)

    for k in metrics:
        metrics[k] /= total

    return metrics


def train(args):
    # 0. 初始化 Logging
    log_file = setup_logging(args.log_dir, model_name="kgat", log_file=args.log_file)
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
    np.random.seed(42)
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

    # 如果僅進行評估
    if args.eval_only:
        logging.info("Running evaluation ONLY mode...")
        metrics = evaluate(model, test_data, indices, edge_types, num_nodes, n_items, device=device)
        logging.info("Evaluation Results:")
        for k in [10, 20, 50]:
            logging.info(f"K={k} -> HR: {metrics[f'hr_{k}']:.4f}, Precision: {metrics[f'prec_{k}']:.4f}, NDCG: {metrics[f'ndcg_{k}']:.4f}")
        return

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
                    pos_scores, neg_scores = model(
                        indices, edge_types, num_nodes, u, i, j
                    )
                    loss = -torch.mean(
                        torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            elif device.type == "xpu":
                # XPU specific AMP context (if supported) or LP optimization by IPEX
                with torch.autocast(
                    device_type="xpu", enabled=args.use_bf16, dtype=torch.bfloat16
                ):
                    pos_scores, neg_scores = model(
                        indices, edge_types, num_nodes, u, i, j
                    )
                    loss = -torch.mean(
                        torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)
                    )

                loss.backward()
                optimizer.step()

            else:
                # CPU
                with torch.autocast(
                    device_type="cpu", enabled=args.use_bf16, dtype=torch.bfloat16
                ):
                    pos_scores, neg_scores = model(
                        indices, edge_types, num_nodes, u, i, j
                    )
                    loss = -torch.mean(
                        torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)
                    )
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / n_batches
        logging.info(f"Epoch {epoch + 1} done. Avg Loss: {avg_loss:.4f}")

        # Validation Per Epoch
        metrics = evaluate(
            model, test_data, indices, edge_types, num_nodes, n_items, device=device
        )
        logging.info(
            f"Epoch {epoch + 1} Evaluation - "
            f"HR@[10,20,50]: [{metrics['hr_10']:.4f}, {metrics['hr_20']:.4f}, {metrics['hr_50']:.4f}] | "
            f"Precision@[10,20,50]: [{metrics['prec_10']:.4f}, {metrics['prec_20']:.4f}, {metrics['prec_50']:.4f}] | "
            f"NDCG@[10,20,50]: [{metrics['ndcg_10']:.4f}, {metrics['ndcg_20']:.4f}, {metrics['ndcg_50']:.4f}]"
        )

        # Step Scheduler
        scheduler.step(metrics['hr_20'])
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(f"Epoch {epoch + 1} Current LR: {current_lr:.6e}")

        # Explicit GC
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "xpu":
            torch.xpu.empty_cache()

        # Save Checkpoint
        if (epoch + 1) % 1 == 0 or (epoch + 1) == args.epochs:
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

```

## File: src\train_bi_interaction.py

```python
import argparse
import logging
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from src.model.kgat_bi_interaction import KGAT_BiInteraction

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


def evaluate(model, test_interactions, adj, n_items, device):
    """計算 HR@K, Precision@K, NDCG@K (使用隨機負採樣)"""
    model.eval()

    metrics = {
        "hr_10": 0,
        "hr_20": 0,
        "hr_50": 0,
        "ndcg_10": 0,
        "ndcg_20": 0,
        "ndcg_50": 0,
        "prec_10": 0,
        "prec_20": 0,
        "prec_50": 0,
    }
    total = 0
    batch_size = 512

    # 1. 預先計算並快取全圖最終特徵向量
    with torch.no_grad():
        final_embed = model.get_final_embeddings(adj)

    with torch.no_grad():
        for start_idx in range(0, len(test_interactions), batch_size):
            end_idx = min(start_idx + batch_size, len(test_interactions))
            batch = test_interactions[start_idx:end_idx]

            u = torch.LongTensor(batch[:, 0]).to(device)
            i = torch.LongTensor(batch[:, 1]).to(device)

            # 2. 正樣本評估
            u_embed = final_embed[u]
            pos_i_embed = final_embed[model.n_users + i]
            pos_scores = torch.sum(u_embed * pos_i_embed, dim=1)  # (B,)

            # 3. 負樣本評估
            neg_items = torch.randint(0, n_items, (len(batch), 100)).to(device)
            u_expanded = u.unsqueeze(1).repeat(1, 100).view(-1)
            neg_items_flatten = neg_items.view(-1)

            u_embed_expanded = final_embed[u_expanded]
            neg_i_embed = final_embed[model.n_users + neg_items_flatten]
            neg_scores = torch.sum(u_embed_expanded * neg_i_embed, dim=1).view(
                len(batch), 100
            )

            # 合併分數並計算排名
            all_scores = torch.cat(
                [pos_scores.unsqueeze(1), neg_scores], dim=1
            )  # (B, 101)

            # --- 新指標計算 ---
            _, sorted_indices = torch.sort(all_scores, dim=1, descending=True)
            pos_ranks = (sorted_indices == 0).nonzero(as_tuple=True)[1]

            for k in [10, 20, 50]:
                hits = (pos_ranks < k).float()
                metrics[f"hr_{k}"] += hits.sum().item()
                metrics[f"prec_{k}"] += (hits / k).sum().item()
                metrics[f"ndcg_{k}"] += (
                    (hits / torch.log2(pos_ranks.float() + 2)).sum().item()
                )

            total += len(batch)

    for k in metrics:
        metrics[k] /= total

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train KGAT Model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[64, 32],
        help="Layer sizes, e.g. 64 32",
    )
    parser.add_argument("--use_bf16", action="store_true", help="Use BFloat16 on XPU")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory to save models",
    )
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU")
    parser.add_argument(
        "--debug", action="store_true", help="Run with small data for debugging"
    )
    parser.add_argument(
        "--log_dir", type=str, default="output/logs", help="Directory to save logs"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Run evaluation on the loaded checkpoint and exit",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Explicitly specify the log file to use (append mode)",
    )
    return parser.parse_args()


def setup_logging(log_dir, model_name="kgat", log_file=None):
    if log_file is None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{model_name}_{timestamp}.txt")
        file_mode = "w"
    else:
        file_mode = "a"

    # 移除舊的 handlers 以避免重複輸出
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 設定 Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8", mode=file_mode),
            logging.StreamHandler(),
        ],
    )
    return log_file


def train():
    args = parse_args()

    # 0. 初始化 Logging
    log_file = setup_logging(args.log_dir, model_name="kgat", log_file=args.log_file)
    logging.info(f"Training started. Args: {args}")
    logging.info(f"Log file: {log_file}")

    # 設定環境
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

    # 1. 載入資料
    try:
        interactions, kg_triples, stats = load_data()
    except FileNotFoundError:
        logging.error(
            "Error: Processed data not found. Please run 'python src/data/preprocess.py' first."
        )
        return

    n_users = stats["n_users"]
    n_items = stats["n_items"]
    n_entities = stats["n_entities"]
    n_relations = stats["n_relations"]

    if args.debug:
        logging.info("DEBUG MODE: Only using 2000 samples.")
        interactions = interactions[:2000]

    # 資料分割 (Train/Test)
    np.random.seed(42)
    np.random.shuffle(interactions)
    split_idx = int(0.8 * len(interactions))
    train_data = interactions[:split_idx]
    test_data = interactions[split_idx:]
    logging.info(f"Data Split - Train: {len(train_data)}, Test: {len(test_data)}")

    # 2. 建立鄰接矩陣 (User + Item + Entity)
    logging.info("Constructing adjacency matrix (CPU)...")
    adj = construct_adj(kg_triples, interactions, n_users, n_items, n_entities)
    logging.info(f"Adjacency matrix created. Edges: {adj._nnz()}")

    logging.info(f"Moving adjacency matrix to {device}...")
    adj = adj.to(device)
    if not adj.is_coalesced():
        logging.info("Coalescing adjacency matrix on device...")
        adj = adj.coalesce()

    if args.use_bf16 and device.type == "xpu":
        adj = adj.bfloat16()
        # 型別轉換可能導致 coalesced 狀態遺失，再次檢查
        if not adj.is_coalesced():
            adj = adj.coalesce()
        logging.info("Adjacency matrix cast to BFloat16.")
    logging.info(f"Done. Coalesced: {adj.is_coalesced()}")

    # 3. 斷點續訓與超參數恢復
    start_epoch = 0
    checkpoint = None
    if args.resume and os.path.exists(args.resume):
        logging.info(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            saved_args = checkpoint.get("args")
            if saved_args:
                logging.info("Restoring hyperparameters from checkpoint...")
                # 如果目前的參數是預設值，則從 checkpoint 恢復，否則優先使用命令列參數
                if args.batch_size == 1024 and hasattr(saved_args, "batch_size"):
                    args.batch_size = saved_args.batch_size
                if args.lr == 1e-3 and hasattr(saved_args, "lr"):
                    args.lr = saved_args.lr
                if args.embed_dim == 64 and hasattr(saved_args, "embed_dim"):
                    args.embed_dim = saved_args.embed_dim
                if args.layers == [64, 32] and hasattr(saved_args, "layers"):
                    args.layers = saved_args.layers
            start_epoch = checkpoint.get("epoch", 0)
        else:
            # state_dict only mode
            pass

    # 4. 初始化模型
    logging.info("Initializing KGAT model...")
    n_all_entities = n_items + n_entities
    model = KGAT_BiInteraction(
        n_users,
        n_all_entities,
        n_relations,
        embed_dim=args.embed_dim,
        layers=args.layers,
    ).to(device)
    if args.use_bf16 and device.type == "xpu":
        model = model.bfloat16()
        logging.info("Model cast to BFloat16.")

    logging.info("Model moved to device.")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # 5. 載入權重與狀態
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
                    "Optimizer and scheduler will NOT be loaded due to architectural mismatch."
                )
            else:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if "scheduler_state_dict" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logging.info(
                f"Resumed from epoch {start_epoch} (Batch Size: {args.batch_size}, LR: {args.lr}, Layers: {args.layers})"
            )
        else:
            model.load_state_dict(checkpoint, strict=False)
            logging.info("Loaded model weights (state_dict only, strict=False)")

    if HAS_XPU:
        torch.xpu.empty_cache()

    if args.eval_only:
        logging.info("Running evaluation ONLY mode...")
        metrics = evaluate(model, test_data, adj, n_items, device)
        logging.info("Evaluation Results:")
        for k in [10, 20, 50]:
            logging.info(
                f"K={k} -> HR: {metrics[f'hr_{k}']:.4f}, Precision: {metrics[f'prec_{k}']:.4f}, NDCG: {metrics[f'ndcg_{k}']:.4f}"
            )
        return

    # 5. 訓練迴圈
    epochs = args.epochs
    batch_size = args.batch_size
    logging.info(f"Starting training from epoch {start_epoch} to {epochs}...")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        n_batches = len(train_data) // batch_size

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch + 1}/{epochs}")
        for _ in pbar:
            u, i, j = sample_bpr_batch(train_data, n_items, batch_size)
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
        logging.info(f"Epoch {epoch + 1} Complete. Average Loss: {avg_loss:.4f}")

        # 評估 Recall Metrics
        metrics = evaluate(model, test_data, adj, n_items, device)
        logging.info(
            f"Epoch {epoch + 1} Evaluation - "
            f"HR@[10,20,50]: [{metrics['hr_10']:.4f}, {metrics['hr_20']:.4f}, {metrics['hr_50']:.4f}] | "
            f"Precision@[10,20,50]: [{metrics['prec_10']:.4f}, {metrics['prec_20']:.4f}, {metrics['prec_50']:.4f}] | "
            f"NDCG@[10,20,50]: [{metrics['ndcg_10']:.4f}, {metrics['ndcg_20']:.4f}, {metrics['ndcg_50']:.4f}]"
        )

        # Step Scheduler
        scheduler.step(metrics["hr_20"])
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(f"Epoch {epoch + 1} Current LR: {current_lr:.6e}")

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
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
                "args": args,
            }
            torch.save(checkpoint, ckpt_path)
            logging.info(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    train()

```

## File: src\run_explainer_attention.py

```python
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

```

## File: src\run_explainer_kgat.py

```python
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

from src.model.explainer import KGATExplainer
from src.model.kgat_bi_interaction import KGAT_BiInteraction
from src.train import construct_adj, load_data


def run():
    data_dir = os.path.join(project_root, "data", "processed")
    model_path = os.path.join(project_root, "models", "kgat_checkpoint_e20.pth")

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
    adj = construct_adj(kg_triples, interactions, n_users, n_items, n_entities)
    adj = adj.to(device)

    # 4. Initialize Model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    embed_dim = 64
    if isinstance(checkpoint, dict) and "args" in checkpoint:
        args = checkpoint["args"]
        embed_dim = getattr(args, "embed_dim", 64)
        print(f"Hyperparameters loaded from checkpoint: embed_dim={embed_dim}")

    n_all_entities = n_items + n_entities
    model = KGAT_BiInteraction(n_users, n_all_entities, n_relations, embed_dim=embed_dim).to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print("Model loaded successfully.")
    model.eval()

    # 5. Explainer
    explainer = KGATExplainer(model)

    # Pick a user and a recommended item
    target_user = 5
    user_interactions = interactions[interactions[:, 0] == target_user]

    if len(user_interactions) > 0:
        target_item = int(user_interactions[0][1])
        print(f"\n--- Explaining for User {target_user} and Item {target_item} ---")

        explanation = explainer.explain(adj, target_user, target_item, top_k=5)

        if explanation:
            print(f"Prediction Score: {explanation['target_score']:.4f}")
            print("\nTop Explanation Paths:")
            for path, score in explanation["top_paths"]:
                path_str = []
                for node_id in path:
                    if node_id < n_users:
                        path_str.append(f"User({node_id})")
                    elif node_id < n_users + n_items:
                        path_str.append(f"Recipe({node_id - n_users})")
                    else:
                        path_str.append(f"Entity({node_id - n_users - n_items})")
                print(" -> ".join(path_str) + f" (Score: {score:.6f})")

            print("\nExplanation extraction successful!")
    else:
        print(f"User {target_user} has no interactions to explain.")


if __name__ == "__main__":
    run()

```

## File: src\generate_explanations.py

```python
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.explainer import KGATExplainer
from src.model.explainer_attention import KGATAttentionExplainer
from src.model.kgat_attention import KGATAttention
from src.model.kgat_bi_interaction import KGAT_BiInteraction
from src.train_bi_interaction import construct_adj, load_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成 KGAT 模型推薦解釋 (Generate KGAT Explanations)"
    )
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(project_root, "models", "kgat_att_checkpoint_e20.pth"),
        help="模型 Checkpoint 路徑",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(project_root, "data", "processed"),
        help="處理後資料的目錄",
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default=os.path.join(project_root, "data", "raw"),
        help="原始資料目錄 (用於讀取名稱)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="explanations.json",
        help="輸出結果檔案路徑 (JSON)",
    )
    parser.add_argument("--num_users", type=int, default=5, help="隨機挑選的使用者數量")
    parser.add_argument(
        "--user_ids",
        type=str,
        default=None,
        help="指定使用者 ID (逗號分隔)，若指定則忽略 num_users",
    )
    parser.add_argument(
        "--user_ids_file",
        type=str,
        default=None,
        help="指定包含使用者 ID 列表的 JSON 檔案路徑",
    )
    parser.add_argument(
        "--top_k_explain", type=int, default=3, help="每個推薦保留的解釋路徑數量"
    )
    return parser.parse_args()


def load_names_and_maps(data_dir, raw_data_dir):
    """
    載入原始資料與對應表，以反查 ID 對應的實際名稱
    """
    print("正在載入名稱對應資訊...")

    # 1. Load Maps from stats.pkl
    stats_path = os.path.join(data_dir, "stats.pkl")
    if not os.path.exists(stats_path):
        print(f"警告: 找不到 {stats_path}，將無法顯示正確名稱。")
        return None, None, None, None

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    # User / Item LabelEncoders
    user_le = stats["user_map"]
    item_le = stats["item_map"]

    # Entity Maps (Name -> ID) -> 需要反轉為 (ID -> Name)
    ingredient_map = stats["ingredient_map"]
    tag_map = stats["tag_map"]

    # 反轉 Entity Map
    id_to_ing = {v: k for k, v in ingredient_map.items()}
    id_to_tag = {v: k for k, v in tag_map.items()}

    # 2. Load Recipe Names from RAW_recipes.csv
    recipes_path = os.path.join(raw_data_dir, "RAW_recipes.csv")
    recipe_name_map = {}

    if os.path.exists(recipes_path):
        print(f"正在讀取食譜名稱: {recipes_path}")
        df_recipes = pd.read_csv(recipes_path)
        for _, row in df_recipes.iterrows():
            recipe_name_map[row["id"]] = row["name"]
    else:
        print(f"警告: 找不到 {recipes_path}，將只顯示 ID。")

    return user_le, item_le, (id_to_ing, id_to_tag), recipe_name_map


def get_node_name(
    node_id, n_users, n_items, user_le, item_le, entity_maps, recipe_name_map
):
    """
    根據 Node ID 返回 (Type, Real ID, Name/Description)
    """
    # 1. User
    if node_id < n_users:
        remapped_uid = node_id
        original_uid = user_le.classes_[remapped_uid] if user_le else str(remapped_uid)
        return "USER", str(original_uid), f"User {original_uid}"

    # 2. Item (Recipe)
    elif node_id < n_users + n_items:
        remapped_iid = node_id - n_users
        try:
            original_iid = (
                item_le.classes_[remapped_iid] if item_le else str(remapped_iid)
            )
            # 嘗試取得名稱
            name = recipe_name_map.get(original_iid, f"Recipe {original_iid}")
        except IndexError:
            # 防呆
            original_iid = str(remapped_iid)
            name = f"Recipe {original_iid}"

        return "RECIPE", str(original_iid), str(name)

    # 3. Entity (Ingredient or Tag)
    else:
        entity_id = node_id - n_users - n_items
        id_to_ing, id_to_tag = entity_maps if entity_maps else ({}, {})

        if entity_id in id_to_ing:
            return "INGREDIENT", str(entity_id), str(id_to_ing[entity_id])
        elif entity_id in id_to_tag:
            return "TAG", str(entity_id), str(id_to_tag[entity_id])
        else:
            return "ENTITY", str(entity_id), f"Entity {entity_id}"


def run():
    args = parse_args()

    # 1. 載入資料
    print(f"正在載入資料：{args.data_dir} ...")
    interactions, kg_triples, stats = load_data(args.data_dir)
    n_users = stats["n_users"]
    n_items = stats["n_items"]
    n_entities = stats["n_entities"]
    n_relations = stats["n_relations"]

    # 載入名稱對應
    user_le, item_le, entity_maps, recipe_name_map = load_names_and_maps(
        args.data_dir, args.raw_data_dir
    )

    # 2. 設定裝置 (Device Setup)
    if args.cpu:
        device = torch.device("cpu")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"使用裝置: {device}")

    # 3. 準備模型與權重
    print(f"正在載入模型：{args.model_path} ...")
    if not os.path.exists(args.model_path):
        print(f"錯誤：找不到模型檔案 {args.model_path}")
        return

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    # 判斷是否為 Attention 模型
    is_attention = False
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        # 檢查是否有 W_att 這個只有 Attention 模型才有的 key
        if any("W_att" in k for k in state_dict.keys()):
            is_attention = True
            print("檢測到 Attention 模型權重。")

    # 提取超參數
    embed_dim = 64
    layers = [64, 32]
    if isinstance(checkpoint, dict) and "args" in checkpoint:
        saved_args = checkpoint["args"]
        embed_dim = getattr(saved_args, "embed_dim", embed_dim)
        layers = getattr(saved_args, "layers", layers)
        print(f"從 Checkpoint 載入超參數: embed_dim={embed_dim}, layers={layers}")

    n_all_entities = n_items + n_entities

    # 初始化模型與解釋器
    if is_attention:
        # Note: train_att.py adds 2 to n_relations (for Interatction and Self-loop)
        model = KGATAttention(
            n_users, n_all_entities, n_relations + 2, embed_dim=embed_dim, layers=layers
        ).to(device)
        explainer = KGATAttentionExplainer(model)

        # 準備 Graph Indices (Attention 模型使用)
        print("正在準備圖索引 (for Attention Mode)...")
        num_nodes = n_users + n_items + n_entities
        kg_src = kg_triples[:, 0] + n_users
        kg_dst = kg_triples[:, 2] + n_users + n_items
        int_src = interactions[:, 0]
        int_dst = interactions[:, 1] + n_users
        all_src = np.concatenate(
            [kg_src, kg_dst, int_src, int_dst, np.arange(num_nodes)]
        )
        all_dst = np.concatenate(
            [kg_dst, kg_src, int_dst, int_src, np.arange(num_nodes)]
        )
        indices = torch.LongTensor(np.vstack([all_src, all_dst])).to(device)

        # Construct Edge Types for Attention
        # KG Relations: 0, 1
        # Inverse KG: 0, 1 (Symetric)
        # Interaction: 2
        # Inverse Interaction: 2
        # Self-loop: 3
        # Must match logic in train_att.py get_adj_indices
        kg_rels = kg_triples[:, 1]
        n_int = len(int_src)
        n_self = num_nodes

        rels_kg = kg_rels
        rels_kg_inv = kg_rels
        rels_int = np.full(n_int, 2)
        rels_int_inv = np.full(n_int, 2)
        rels_self = np.full(n_self, 3)

        all_rels = np.concatenate(
            [rels_kg, rels_kg_inv, rels_int, rels_int_inv, rels_self]
        )
        edge_types = torch.LongTensor(all_rels).to(device)

        graph_input = (indices, edge_types, num_nodes)
    else:
        model = KGAT_BiInteraction(
            n_users, n_all_entities, n_relations, embed_dim=embed_dim, layers=layers
        ).to(device)
        explainer = KGATExplainer(model)

        # 準備鄰接矩陣 (標準 KGAT 使用)
        print("正在建構鄰接矩陣 (for Standard Mode)...")
        adj = construct_adj(kg_triples, interactions, n_users, n_items, n_entities)
        adj = adj.to(device)
        graph_input = adj

    # 載入權重
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("模型載入成功。")

    # 4. 挑選使用者
    if args.user_ids_file and os.path.exists(args.user_ids_file):
        try:
            with open(args.user_ids_file, "r", encoding="utf-8") as f:
                data = json.load(f)

                def extract_ids_recursively(obj):
                    """遞迴提取 JSON 中所有 id 欄位或數值"""
                    ids = []
                    if isinstance(obj, list):
                        for item in obj:
                            ids.extend(extract_ids_recursively(item))
                    elif isinstance(obj, dict):
                        # 如果當前層有 id 或 user_id，直接使用
                        if "id" in obj:
                            ids.append(int(obj["id"]))
                        elif "user_id" in obj:
                            ids.append(int(obj["user_id"]))
                        else:
                            # 否則往下層找 (例如 user: [...], favorite: [...] 等類別)
                            for val in obj.values():
                                ids.extend(extract_ids_recursively(val))
                    elif isinstance(obj, (int, str)):
                        try:
                            ids.append(int(obj))
                        except (ValueError, TypeError):
                            pass
                    return ids

                target_users = extract_ids_recursively(data)
            
            # 去除重複並保持順序
            target_users = list(dict.fromkeys(target_users))
            print(f"從 {args.user_ids_file} 成功載入 {len(target_users)} 個唯一使用者 ID。")
        except Exception as e:
            print(f"錯誤：解析 user_ids_file 失敗 ({e})。請確認 JSON 格式。")
            return
    elif args.user_ids:
        try:
            target_users = [int(uid.strip()) for uid in args.user_ids.split(",")]
        except ValueError:
            print("錯誤：user_ids 格式不正確，請使用逗號分隔的整數。")
            return
    else:
        unique_users = np.unique(interactions[:, 0])
        unique_users = unique_users[unique_users < n_users]
        if len(unique_users) < args.num_users:
            target_users = unique_users.tolist()
        else:
            target_users = np.random.choice(
                unique_users, args.num_users, replace=False
            ).tolist()

    print(f"將對以下使用者進行推理與解釋: {target_users}")

    # 5. 推理與解釋迴圈
    results = []
    all_items = torch.arange(n_items, device=device)

    for user_id in tqdm(target_users, desc="處理使用者"):
        # --- (A) 推理 ---
        u_batch = torch.full((n_items,), user_id, dtype=torch.long, device=device)
        i_batch = all_items

        with torch.no_grad():
            if is_attention:
                # KGATAttention.forward(indices, edge_types, num_nodes, u, i)
                pos_scores = model(
                    graph_input[0], graph_input[1], graph_input[2], u_batch, i_batch
                )
            else:
                # KGAT.forward(adj, u, i, j)
                pos_scores = model(graph_input, u_batch, i_batch)

        best_item_idx = torch.argmax(pos_scores).item()
        best_score = pos_scores[best_item_idx].item()
        recommended_item_id = int(best_item_idx)

        # 獲取名稱資訊
        _, rec_real_id, rec_name = get_node_name(
            n_users + recommended_item_id,
            n_users,
            n_items,
            user_le,
            item_le,
            entity_maps,
            recipe_name_map,
        )
        _, user_real_id, user_name = get_node_name(
            user_id, n_users, n_items, user_le, item_le, entity_maps, recipe_name_map
        )

        # --- (B) 解釋 ---
        if is_attention:
            # KGATAttentionExplainer 使用不同 signature
            explanation_data = explainer.explain(
                graph_input[0],
                graph_input[1],
                graph_input[2],
                user_id,
                recommended_item_id,
                top_k=args.top_k_explain,
            )
        else:
            explanation_data = explainer.explain(
                graph_input, user_id, recommended_item_id, top_k=args.top_k_explain
            )

        user_result = {
            "user_id_remapped": user_id,
            "user_id_original": user_real_id,
            "user_name": user_name,
            "recommended_item_id_remapped": recommended_item_id,
            "recommended_item_id_original": rec_real_id,
            "recommended_item_name": rec_name,
            "score": float(best_score),
            "explanations": [],
        }

        if explanation_data:
            for path, score in explanation_data["top_paths"]:
                path_details = []
                path_desc_list = []

                for node_id in path:
                    node_id = int(node_id)
                    n_type, n_real_id, n_name = get_node_name(
                        node_id,
                        n_users,
                        n_items,
                        user_le,
                        item_le,
                        entity_maps,
                        recipe_name_map,
                    )

                    path_details.append(
                        {
                            "id_remapped": node_id,
                            "id_original": n_real_id,
                            "type": n_type,
                            "name": n_name,
                        }
                    )
                    path_desc_list.append(n_name)

                user_result["explanations"].append(
                    {
                        "path_structure": " -> ".join(
                            [p["type"] for p in path_details]
                        ),
                        "path_description": " -> ".join(path_desc_list),
                        "path_details": path_details,
                        "contribution_score": float(score),
                    }
                )
        else:
            user_result["explanations_note"] = "無法找到解釋路徑。"

        results.append(user_result)

    # 6. 儲存結果
    output_dir = "output/explanations"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.output)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"解釋生成完畢，結果已儲存至：{output_path}")


if __name__ == "__main__":
    run()

```

