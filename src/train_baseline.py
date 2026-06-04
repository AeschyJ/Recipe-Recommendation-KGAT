import argparse
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.bpr_mf import BPRMF
from src.model.lightgcn import LightGCN
from src.model.nfm import NFM

HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Baseline Models")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["BPR-MF", "NFM", "LightGCN"],
        help="Choose baseline model",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--model_dir", type=str, default="models/baseline")
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_dir", type=str, default="output/logs/baseline")
    parser.add_argument(
        "--eval_only", action="store_true", help="Run evaluation on a specific model"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path for checkpoint to evaluate"
    )
    parser.add_argument(
        "--experiment_id", type=int, default=1, help="Experiment ID"
    )
    parser.add_argument("--log_file", type=str, default=None, help="Explicitly specify the log file to use (append mode)")
    return parser.parse_args()


def setup_logging(log_dir, model_name, log_file=None, resume_ckpt=None):
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
                    "Expected iterations per epoch:",
                    "Initializing BPR-MF",
                    "Initializing NFM",
                    "Initializing LightGCN",
                    "Enabled BFloat16 precision",
                    "Model compilation",
                    "DEBUG MODE:",
                    "Loading checkpoint:",
                    "Restoring hyperparameters from checkpoint",
                    "Resumed from epoch",
                    "Running evaluation ONLY mode:",
                    "Enabled CUDA AMP GradScaler",
                    "Loaded model weights",
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
    # stream_handler.addFilter(resume_filter)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            file_handler,
            stream_handler,
        ],
    )
    return log_file


def load_data(data_dir):
    with open(os.path.join(data_dir, "interactions.pkl"), "rb") as f:
        interactions = pickle.load(f)
    if isinstance(interactions, pd.DataFrame):
        interactions = interactions.values

    with open(os.path.join(data_dir, "stats.pkl"), "rb") as f:
        stats = pickle.load(f)

    return interactions, stats


def get_user_item_indices(interactions, n_users, n_items):
    """提取二分圖指標供 LightGCN 使用"""
    int_src = interactions[:, 0]
    int_dst = interactions[:, 1] + n_users

    all_src = np.concatenate([int_src, int_dst])
    all_dst = np.concatenate([int_dst, int_src])

    indices = np.vstack([all_src, all_dst])
    return torch.LongTensor(indices), n_users + n_items


def evaluate(
    model, model_type, interactions, indices, num_nodes, n_items, device="cpu"
):
    """
    通用評估腳本
    計算 HR@10, @20, @50, Precision@10, @20, @50, NDCG@10, @20, @50
    """
    # 記錄原本的 dtype 並暫時轉為 float32 以提高計算精度
    original_dtype = next(model.parameters()).dtype
    if original_dtype == torch.bfloat16:
        model = model.float()

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
    n_test = len(interactions)

    # Fast evaluation for BPR-MF and LightGCN via dot product
    if model_type in ["BPR-MF", "LightGCN"]:
        with torch.no_grad():
            final_embed = model.get_final_embeddings(indices, num_nodes)

    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            end = min(i + batch_size, n_test)
            batch = interactions[i:end]

            user_ids = torch.LongTensor(batch[:, 0]).to(device)
            item_ids = torch.LongTensor(batch[:, 1]).to(device)

            users_expanded = user_ids.unsqueeze(1).repeat(1, 100).view(-1)
            neg_items = torch.randint(0, n_items, (len(users_expanded),)).to(device)
            neg_items_flatten = neg_items.view(-1)

            if model_type in ["BPR-MF", "LightGCN"]:
                u_embed = final_embed[user_ids]
                pos_i_embed = final_embed[model.n_users + item_ids]
                pos_scores = torch.sum(u_embed * pos_i_embed, dim=1)

                u_embed_expanded = final_embed[users_expanded]
                neg_i_embed = final_embed[model.n_users + neg_items_flatten]
                neg_scores = torch.sum(u_embed_expanded * neg_i_embed, dim=1).view(
                    len(batch), 100
                )
            elif model_type == "NFM":
                pos_scores = model.evaluate_forward(user_ids, item_ids)
                neg_scores = model.evaluate_forward(
                    users_expanded, neg_items_flatten
                ).view(len(batch), 100)

            all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

            # 新指標計算
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

    # 恢復原本的 dtype
    if original_dtype == torch.bfloat16:
        model = model.bfloat16()

    return metrics


def train(args):
    model_safe_name = args.model.lower().replace("-", "")
    log_file = setup_logging(args.log_dir, model_name=model_safe_name, log_file=args.log_file, resume_ckpt=args.resume)
    logging.info(f"Training started. Args: {args}")

    if args.cpu:
        device = torch.device("cpu")
    elif HAS_XPU:
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Using Device: {device}")

    interactions, stats = load_data(args.data_dir)
    n_users = stats["n_users"]
    n_items = stats["n_items"]

    if args.debug:
        interactions = interactions[:2000]
        args.batch_size = 512
        args.epochs = 1

    indices, num_nodes = get_user_item_indices(interactions, n_users, n_items)
    indices = indices.to(device)

    # Train/Test Split
    np.random.seed(42)
    np.random.shuffle(interactions)
    split_idx = int(len(interactions) * 0.8)
    train_data = interactions[:split_idx]
    test_data = interactions[split_idx:]

    logging.info(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    # Model Initialization
    if args.model == "BPR-MF":
        model = BPRMF(n_users, n_items, embed_dim=args.embed_dim)
    elif args.model == "NFM":
        model = NFM(n_users, n_items, embed_dim=args.embed_dim)
    elif args.model == "LightGCN":
        model = LightGCN(n_users, n_items, embed_dim=args.embed_dim, layers=3)

    model = model.to(device)

    if args.use_bf16:
        model = model.bfloat16()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    start_epoch = 0
    checkpoint = None
    if args.resume and os.path.exists(args.resume):
        logging.info(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            start_epoch = checkpoint.get("epoch", 0)
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "scaler_state_dict" in checkpoint and scaler:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            logging.info(f"Resumed from epoch {start_epoch}")
        else:
            model.load_state_dict(checkpoint, strict=False)
            logging.info("Loaded model weights (state_dict only, strict=False)")

    if args.eval_only:
        logging.info("Running evaluation ONLY mode...")
        metrics = evaluate(
            model, args.model, test_data, indices, num_nodes, n_items, device=device
        )
        logging.info("Evaluation Results:")
        for k in [10, 20, 50]:
            logging.info(
                f"K={k} -> HR: {metrics[f'hr_{k}']:.4f}, Precision: {metrics[f'prec_{k}']:.4f}, NDCG: {metrics[f'ndcg_{k}']:.4f}"
            )
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
        
        model_safe_name = args.model.lower().replace("-", "")
        best_ckpt_path = os.path.join(args.model_dir, f"{args.experiment_id}_{model_safe_name}_checkpoint_e{best_epoch}.pth")
        
        if best_epoch != start_epoch and os.path.exists(best_ckpt_path):
            logging.info(f"Evaluating BEST epoch ({best_epoch}) checkpoint to establish true baseline best_hr20...")
            best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt["model_state_dict"], strict=False)
            
            metrics = evaluate(
                model, args.model, test_data, indices, num_nodes, n_items, device=device
            )
            best_hr20 = metrics['hr_20']
            logging.info(f"Best_hr20 (from epoch {best_epoch}): {best_hr20:.4f}")
            
            logging.info(f"Restoring model weights back to resumed epoch {start_epoch} for continued training...")
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            logging.info("Evaluating resumed checkpoint to establish baseline best_hr20 for this session...")
            metrics = evaluate(
                model, args.model, test_data, indices, num_nodes, n_items, device=device
            )
            best_hr20 = metrics['hr_20']
            logging.info(f"Resumed checkpoint HR@20: {best_hr20:.4f}")

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

            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    pos_scores, neg_scores = model(indices, num_nodes, u, i, j)
                    loss = -torch.mean(
                        torch.log(torch.sigmoid(pos_scores.float() - neg_scores.float()) + 1e-10)
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            elif device.type == "xpu" or device.type == "cpu":
                with torch.autocast(
                    device_type=device.type, enabled=args.use_bf16, dtype=torch.bfloat16
                ):
                    pos_scores, neg_scores = model(indices, num_nodes, u, i, j)
                    loss = -torch.mean(
                        torch.log(torch.sigmoid(pos_scores.float() - neg_scores.float()) + 1e-10)
                    )
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / n_batches
        logging.info(f"Epoch {epoch + 1} Avg Loss: {avg_loss:.4f}")

        metrics = evaluate(
            model, args.model, test_data, indices, num_nodes, n_items, device=device
        )
        logging.info(
            f"Epoch {epoch + 1} Evaluation - "
            f"HR@[10,20,50]: [{metrics['hr_10']:.4f}, {metrics['hr_20']:.4f}, {metrics['hr_50']:.4f}] | "
            f"Precision@[10,20,50]: [{metrics['prec_10']:.4f}, {metrics['prec_20']:.4f}, {metrics['prec_50']:.4f}] | "
            f"NDCG@[10,20,50]: [{metrics['ndcg_10']:.4f}, {metrics['ndcg_20']:.4f}, {metrics['ndcg_50']:.4f}]"
        )

        scheduler.step(metrics["hr_20"])
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(f"Epoch {epoch + 1} Current LR: {current_lr:.6e}")

        current_hr20 = metrics['hr_20']
        is_best = False
        if current_hr20 > best_hr20:
            best_hr20 = current_hr20
            best_epoch = epoch + 1
            patience_counter = 0
            is_best = True
        else:
            patience_counter += 1

        ckpt_path = os.path.join(
            args.model_dir, f"{args.experiment_id}_{model_safe_name}_checkpoint_e{epoch + 1}.pth"
        )
        save_dict = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_type": args.model,
            "loss": avg_loss,
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
            pattern = os.path.join(args.model_dir, f"{args.experiment_id}_{model_safe_name}_checkpoint_e*.pth")
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
