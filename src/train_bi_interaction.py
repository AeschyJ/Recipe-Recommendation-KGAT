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
    parser.add_argument(
        "--no_compile", action="store_true", help="Disable torch.compile"
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

    # 6. torch.compile 最佳化 (與 train_att.py 對齊)
    if HAS_XPU and not args.cpu and not args.no_compile:
        try:
            model = torch.compile(model)
            logging.info("Model compiled with torch.compile for XPU.")
        except Exception as e:
            logging.warning(f"Warning: torch.compile failed: {e}")
    elif args.no_compile:
        logging.info("Model compilation disabled by user.")

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

    # 提前解構 adj 索引，避免每個 batch 重複呼叫 .indices()/.values()
    if adj.layout == torch.sparse_coo:
        _adj_target = adj.indices()[0]
        _adj_neighbor = adj.indices()[1]
        _adj_values = adj.values()
    elif adj.layout == torch.sparse_csr:
        _adj_coo = adj.to_sparse_coo().coalesce()
        _adj_target = _adj_coo.indices()[0]
        _adj_neighbor = _adj_coo.indices()[1]
        _adj_values = _adj_coo.values()
    else:
        raise ValueError(f"Unsupported adjacency layout: {adj.layout}")
    logging.info("Adjacency indices pre-extracted for training loop.")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        n_batches = len(train_data) // batch_size

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch + 1}/{epochs}")
        for _ in pbar:
            u, i, j = sample_bpr_batch(train_data, n_items, batch_size)
            u, i, j = u.to(device), i.to(device), j.to(device)

            try:
                optimizer.zero_grad(set_to_none=True)
                if device.type == "xpu":
                    with torch.autocast(
                        device_type="xpu", enabled=args.use_bf16, dtype=torch.bfloat16
                    ):
                        pos_scores, neg_scores = model(
                            adj, u, i, j,
                            _target=_adj_target, _neighbor=_adj_neighbor, _values=_adj_values
                        )
                        loss = bpr_loss(pos_scores.float(), neg_scores.float())
                else:
                    pos_scores, neg_scores = model(
                        adj, u, i, j,
                        _target=_adj_target, _neighbor=_adj_neighbor, _values=_adj_values
                    )
                    loss = bpr_loss(pos_scores.float(), neg_scores.float())

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
        if (epoch + 1) % 1 == 0 or (epoch + 1) == epochs:
            os.makedirs(args.model_dir, exist_ok=True)
            ckpt_path = os.path.join(
                args.model_dir, f"2_kgat_checkpoint_e{epoch + 1}.pth"
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
