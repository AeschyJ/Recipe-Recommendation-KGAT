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

from src.model.kgat import KGATAttention, KGAT_BiInteraction

# 檢查 Native XPU 支援
HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Unified KGAT Model")
    parser.add_argument(
        "--no_attention",
        action="store_true",
        help="Ablation: Use Bi-Interaction aggregator instead of Attention mechanism",
    )
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
    parser.add_argument("--experiment_id", type=str, default="1", help="Experiment ID for checkpoint naming")
    return parser.parse_args()


def setup_logging(log_dir, model_name="kgat", log_file=None, resume_ckpt=None):
    import glob
    is_resume = False
    
    if resume_ckpt and log_file is None:
        ckpt_norm = resume_ckpt.replace("\\", "/")
        # 尋找包含此 checkpoint 的原始 log file
        for fpath in glob.glob(os.path.join(log_dir, "**/*.txt"), recursive=True):
            if "_reformatted" in fpath:
                continue
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        if "Saved checkpoint:" in line:
                            parts = line.split("Saved checkpoint:")
                        elif "Checkpoint saved to" in line:
                            parts = line.split("Checkpoint saved to")
                        else:
                            continue

                        if len(parts) > 1:
                            ckpt = parts[1].strip()
                            if ". Patience:" in ckpt:
                                ckpt = ckpt.split(". Patience:")[0].strip()
                            if ckpt.replace("\\", "/") == ckpt_norm:
                                log_file = fpath
                                break
            except:
                pass
            if log_file:
                break

    if log_file is None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{model_name}_{timestamp}.txt")
        file_mode = "w"
    else:
        file_mode = "a"
        is_resume = True

    # 移除舊的 handlers 以避免重複輸出
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    class FilterResume(logging.Filter):
        def filter(self, record):
            if is_resume:
                skip_phrases = [
                    "Training started. Args:",
                    "Log file:",
                    "Using Intel Arc GPU",
                    "Using NVIDIA GPU",
                    "Using Device",
                    "Using CPU",
                    "Train samples:",
                    "Data Split - Train:",
                    "Expected iterations per epoch:",
                    "Initializing KGAT",
                    "Enabled BFloat16 precision",
                    "Model compilation",
                    "DEBUG MODE:",
                    "Loading checkpoint:",
                    "Restoring hyperparameters from checkpoint",
                    "Resumed from epoch",
                    "Running evaluation ONLY mode:",
                    "Enabled CUDA AMP GradScaler",
                    "Loaded model weights",
                    "Constructing adjacency matrix",
                    "Adjacency matrix created",
                    "Moving adjacency matrix",
                    "Coalescing adjacency matrix",
                    "Adjacency matrix cast",
                    "Graph indices pre-extracted",
                    "Done. Coalesced:",
                    "Evaluating resumed",
                    "Resumed checkpoint HR@20:"
                ]
                for phrase in skip_phrases:
                    if phrase in record.getMessage():
                        return False
            return True

    resume_filter = FilterResume()
    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode=file_mode)
    file_handler.addFilter(resume_filter)
    
    stream_handler = logging.StreamHandler()
    # stream_handler.addFilter(resume_filter) # Depending on preference

    # 設定 Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            file_handler,
            stream_handler,
        ],
    )
    return log_file


def load_data(data_dir="data/processed"):
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


def get_adj_indices(kg_triples, interactions, n_users, n_items, n_entities, without_kg=False):
    """
    Construct edge indices for the graph (used by Attention model).
    """
    print("Constructing graph indices with interactions...")
    num_nodes = n_users + n_items + n_entities

    if without_kg:
        print("Ablation: Running without Knowledge Graph (only User-Item bipartite graph)")
        kg_src = np.array([], dtype=np.int64)
        kg_dst = np.array([], dtype=np.int64)
        kg_rels = np.array([], dtype=np.int64)
    else:
        kg_src = kg_triples[:, 0] + n_users
        kg_dst = kg_triples[:, 2] + n_users + n_items
        kg_rels = kg_triples[:, 1]

    int_src = interactions[:, 0]
    int_dst = interactions[:, 1] + n_users

    all_src = np.concatenate([kg_src, kg_dst, int_src, int_dst, np.arange(num_nodes)])
    all_dst = np.concatenate([kg_dst, kg_src, int_dst, int_src, np.arange(num_nodes)])

    n_int = len(int_src)
    n_self = num_nodes

    rels_kg = kg_rels
    rels_kg_inv = kg_rels  
    rels_int = np.full(n_int, 2)
    rels_int_inv = np.full(n_int, 2)
    rels_self = np.full(n_self, 3)

    all_rels = np.concatenate([rels_kg, rels_kg_inv, rels_int, rels_int_inv, rels_self])

    indices = np.vstack([all_src, all_dst])
    edge_types = torch.LongTensor(all_rels)

    return torch.LongTensor(indices), edge_types, num_nodes


def construct_adj(kg_triples, interactions, n_users, n_items, n_entities):
    """
    Construct Sparse Adjacency Matrix (used by Bi-Interaction model).
    """
    print("Constructing Sparse Adjacency Matrix with Interactions...")
    num_nodes = n_users + n_items + n_entities

    kg_src = kg_triples[:, 0] + n_users
    kg_dst = kg_triples[:, 2] + n_users + n_items

    int_src = interactions[:, 0]
    int_dst = interactions[:, 1] + n_users

    all_src = np.concatenate([kg_src, kg_dst])
    all_dst = np.concatenate([kg_dst, kg_src])

    all_src = np.concatenate([all_src, int_src, int_dst])
    all_dst = np.concatenate([all_dst, int_dst, int_src])

    all_src = np.concatenate([all_src, np.arange(num_nodes)])
    all_dst = np.concatenate([all_dst, np.arange(num_nodes)])

    indices = torch.LongTensor(np.vstack([all_src, all_dst]))
    values = torch.ones(len(all_src))

    deg = torch.bincount(indices[0], minlength=num_nodes).float()
    deg[deg == 0] = 1

    norm_values = values / deg[all_src]

    adj_norm = torch.sparse_coo_tensor(indices, norm_values, (num_nodes, num_nodes))
    return adj_norm.coalesce()


def sample_bpr_batch(interactions, n_items, batch_size):
    """隨機採樣 (User, Pos_Item, Neg_Item) 用於 BPR Loss"""
    indices = np.random.randint(0, len(interactions), batch_size)
    batch_data = interactions[indices]

    u = torch.LongTensor(batch_data[:, 0])
    i = torch.LongTensor(batch_data[:, 1])
    j = torch.LongTensor(np.random.randint(0, n_items, batch_size))

    return u, i, j


def bpr_loss(pos_scores, neg_scores):
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))


def evaluate(
    model, test_interactions, n_items, device, args, graph_data
):
    model.eval()
    
    is_bf16 = next(model.parameters()).dtype == torch.bfloat16
    if is_bf16:
        model = model.float()
    
    metrics = {
        'hr_10': 0, 'hr_20': 0, 'hr_50': 0,
        'ndcg_10': 0, 'ndcg_20': 0, 'ndcg_50': 0,
        'prec_10': 0, 'prec_20': 0, 'prec_50': 0,
    }
    total = 0
    batch_size = 512
    n_test = len(test_interactions)

    with torch.no_grad():
        if not args.no_attention:
            indices, edge_types, num_nodes = graph_data
            final_embed = model.get_final_embeddings(indices, edge_types, num_nodes)
        else:
            adj = graph_data
            final_embed = model.get_final_embeddings(adj)

    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            end = min(i + batch_size, n_test)
            batch = test_interactions[i:end]

            user_ids = torch.LongTensor(batch[:, 0]).to(device)
            item_ids = torch.LongTensor(batch[:, 1]).to(device)

            u_embed = final_embed[user_ids]
            pos_i_embed = final_embed[model.n_users + item_ids]
            pos_scores = torch.sum(u_embed * pos_i_embed, dim=1)

            users_expanded = user_ids.unsqueeze(1).repeat(1, 100).view(-1)
            neg_items = torch.randint(0, n_items, (len(users_expanded),)).to(device)
            neg_items_flatten = neg_items.view(-1)

            u_embed_expanded = final_embed[users_expanded]
            neg_i_embed = final_embed[model.n_users + neg_items_flatten]
            neg_scores = torch.sum(u_embed_expanded * neg_i_embed, dim=1).view(
                len(batch), 100
            )

            all_scores = torch.cat(
                [pos_scores.unsqueeze(1), neg_scores], dim=1
            )

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

    if is_bf16:
        model = model.bfloat16()

    return metrics


def train(args):
    log_file = setup_logging(args.log_dir, model_name="kgat", log_file=args.log_file, resume_ckpt=args.resume)
    logging.info(f"Training started. Args: {args}")
    logging.info(f"Log file: {log_file}")

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

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()

    interactions, kg_triples, stats = load_data(args.data_dir)
    n_users = stats["n_users"]
    n_items = stats["n_items"]
    n_entities = stats["n_entities"]
    n_relations = stats["n_relations"]

    if args.debug:
        logging.info("DEBUG MODE: Using small subset of data")
        interactions = interactions[:2000]
        args.batch_size = 512
        args.epochs = 1 if not args.no_attention else args.epochs

    np.random.seed(42)
    np.random.shuffle(interactions)
    split_idx = int(len(interactions) * 0.8)
    train_data = interactions[:split_idx]
    test_data = interactions[split_idx:]

    logging.info(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    expected_iterations = len(train_data) // args.batch_size
    logging.info(f"Expected iterations per epoch: {expected_iterations}")
    if expected_iterations < 10:
        logging.warning("Warning: Iterations per epoch is very low. Consider reducing batch_size.")

    graph_data = None
    if not args.no_attention:
        indices, edge_types, num_nodes = get_adj_indices(
            kg_triples, interactions, n_users, n_items, n_entities, without_kg=args.without_kg
        )
        indices = indices.to(device)
        edge_types = edge_types.to(device)
        graph_data = (indices, edge_types, num_nodes)
        
        del kg_triples
        gc.collect()
    else:
        adj = construct_adj(kg_triples, interactions, n_users, n_items, n_entities)
        adj = adj.to(device)
        if not adj.is_coalesced():
            adj = adj.coalesce()

        if args.use_bf16 and device.type == "xpu":
            adj = adj.bfloat16()
            if not adj.is_coalesced():
                adj = adj.coalesce()
        
        graph_target = adj.indices()[0].contiguous()
        graph_neighbor = adj.indices()[1].contiguous()
        graph_values = adj.values().contiguous()
        num_nodes = adj.shape[0]
        graph_data = adj
        graph_dense_data = (graph_target, graph_neighbor, graph_values, num_nodes)

        del kg_triples
        gc.collect()

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
                if hasattr(saved_args, "layers"):
                    args.layers = saved_args.layers
            start_epoch = checkpoint.get("epoch", 0)

            # 自動偵測 checkpoint 的模型架構，防止 resume 時架構不匹配
            has_bi_interaction = "W_R" in checkpoint["model_state_dict"]
            if has_bi_interaction and not args.no_attention:
                logging.warning("Checkpoint 包含 BiInteraction 權重 (W_R)，自動啟用 --no_attention")
                args.no_attention = True
            elif not has_bi_interaction and args.no_attention:
                logging.warning("Checkpoint 不包含 BiInteraction 權重，自動關閉 --no_attention")
                args.no_attention = False

    n_all_entities = n_items + n_entities
    if not args.no_attention:
        logging.info(f"Initializing KGATAttention with embed_dim={args.embed_dim}, layers={args.layers}")
        model = KGATAttention(
            n_users, n_all_entities, n_relations + 2, embed_dim=args.embed_dim, layers=args.layers
        ).to(device)
    else:
        logging.info(f"Initializing KGAT_BiInteraction with embed_dim={args.embed_dim}, layers={args.layers}")
        model = KGAT_BiInteraction(
            n_users, n_all_entities, n_relations, embed_dim=args.embed_dim, layers=args.layers
        ).to(device)

    if args.use_bf16:
        logging.info("Enabled BFloat16 precision")
        model = model.bfloat16()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    scaler = None
    if device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        logging.info("Enabled CUDA AMP GradScaler")

    if checkpoint is not None:
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if missing or unexpected:
                logging.warning(f"Architecture mismatch! Missing: {missing}, Unexpected: {unexpected}")
                logging.warning("Optimizer/Scheduler/Scaler will NOT be loaded due to mismatch.")
            else:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if "scheduler_state_dict" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                if "scaler_state_dict" in checkpoint and scaler:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
            logging.info(f"Resumed from epoch {start_epoch} (Batch Size: {args.batch_size}, Embed Dim: {args.embed_dim}, Layers: {args.layers})")
        else:
            model.load_state_dict(checkpoint, strict=False)
            logging.info("Loaded model weights (state_dict only, strict=False)")

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
        metrics = evaluate(model, test_data, n_items, device, args, graph_data)
        logging.info("Evaluation Results:")
        for k in [10, 20, 50]:
            logging.info(f"K={k} -> HR: {metrics[f'hr_{k}']:.4f}, Precision: {metrics[f'prec_{k}']:.4f}, NDCG: {metrics[f'ndcg_{k}']:.4f}")
        return

    os.makedirs(args.model_dir, exist_ok=True)

    best_hr20 = 0.0
    best_epoch = start_epoch
    patience_counter = 0
    patience = 10

    if checkpoint is not None:
        if "best_epoch" in checkpoint:
            best_epoch = checkpoint["best_epoch"]
            patience_counter = checkpoint.get("patience_counter", 0)
        else:
            best_epoch = start_epoch
            patience_counter = 0

        best_ckpt_path = os.path.join(args.model_dir, f"{args.experiment_id}_kgat_checkpoint_e{best_epoch}.pth")
        
        if best_epoch != start_epoch and os.path.exists(best_ckpt_path):
            logging.info(f"Evaluating BEST epoch ({best_epoch}) checkpoint to establish true baseline best_hr20...")
            best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt["model_state_dict"], strict=False)
            
            metrics = evaluate(model, test_data, n_items, device, args, graph_data)
            best_hr20 = metrics['hr_20']
            logging.info(f"Best_hr20 (from epoch {best_epoch}): {best_hr20:.4f}")
            
            logging.info(f"Restoring model weights back to resumed epoch {start_epoch} for continued training...")
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            logging.info("Evaluating resumed checkpoint to establish baseline best_hr20 for this session...")
            metrics = evaluate(model, test_data, n_items, device, args, graph_data)
            best_hr20 = metrics['hr_20']
            logging.info(f"Resumed checkpoint HR@20: {best_hr20:.4f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        n_batches = len(train_data) // args.batch_size

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch + 1}/{args.epochs}")

        for _ in pbar:
            u, i, j = sample_bpr_batch(train_data, n_items, args.batch_size)
            u, i, j = u.to(device), i.to(device), j.to(device)

            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    if not args.no_attention:
                        indices, edge_types, num_nodes = graph_data
                        pos_scores, neg_scores = model(indices, edge_types, num_nodes, u, i, j)
                    else:
                        g_t, g_n, g_v, n_n = graph_dense_data
                        pos_scores, neg_scores = model(g_t, g_n, g_v, n_n, u, i, j)
                    loss = bpr_loss(pos_scores.float(), neg_scores.float())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            elif device.type == "xpu" or device.type == "cpu":
                with torch.autocast(device_type=device.type, enabled=args.use_bf16, dtype=torch.bfloat16):
                    if not args.no_attention:
                        indices, edge_types, num_nodes = graph_data
                        pos_scores, neg_scores = model(indices, edge_types, num_nodes, u, i, j)
                    else:
                        g_t, g_n, g_v, n_n = graph_dense_data
                        pos_scores, neg_scores = model(g_t, g_n, g_v, n_n, u, i, j)
                    loss = bpr_loss(pos_scores.float(), neg_scores.float())

                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / n_batches
        logging.info(f"Epoch {epoch + 1} done. Avg Loss: {avg_loss:.4f}")

        metrics = evaluate(model, test_data, n_items, device, args, graph_data)
        logging.info(
            f"Epoch {epoch + 1} Evaluation - "
            f"HR@[10,20,50]: [{metrics['hr_10']:.4f}, {metrics['hr_20']:.4f}, {metrics['hr_50']:.4f}] | "
            f"Precision@[10,20,50]: [{metrics['prec_10']:.4f}, {metrics['prec_20']:.4f}, {metrics['prec_50']:.4f}] | "
            f"NDCG@[10,20,50]: [{metrics['ndcg_10']:.4f}, {metrics['ndcg_20']:.4f}, {metrics['ndcg_50']:.4f}]"
        )

        scheduler.step(metrics['hr_20'])
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(f"Epoch {epoch + 1} Current LR: {current_lr:.6e}")

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "xpu":
            torch.xpu.empty_cache()

        current_hr20 = metrics['hr_20']
        is_best = False
        if current_hr20 > best_hr20:
            best_hr20 = current_hr20
            best_epoch = epoch + 1
            patience_counter = 0
            is_best = True
        else:
            patience_counter += 1

        ckpt_path = os.path.join(args.model_dir, f"{args.experiment_id}_kgat_checkpoint_e{epoch + 1}.pth")
        save_dict = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
            "args": args,
            "best_hr20": best_hr20,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter
        }
        if scaler:
            save_dict["scaler_state_dict"] = scaler.state_dict()

        torch.save(save_dict, ckpt_path)

        if is_best:
            logging.info(f"Saved NEW BEST checkpoint: {ckpt_path}")
            import glob
            import re
            pattern = os.path.join(args.model_dir, f"{args.experiment_id}_kgat_checkpoint_e*.pth")
            for fpath in glob.glob(pattern):
                match = re.search(r'_e(\d+)\.pth$', fpath)
                if match:
                    ep = int(match.group(1))
                    if ep < best_epoch:
                        try:
                            os.remove(fpath)
                            logging.info(f"Deleted old checkpoint: {fpath}")
                        except Exception:
                            pass
        else:
            logging.info(f"Saved checkpoint: {ckpt_path}. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logging.info(f"Early stop. No improvement for {patience} epochs. Best epoch was {best_epoch} with HR@20: {best_hr20:.4f}")
                break


if __name__ == "__main__":
    args = parse_args()
    train(args)
