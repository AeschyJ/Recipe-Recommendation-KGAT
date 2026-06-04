import argparse
import os
import pickle
import sys
import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.kgat import KGATAttention
from src.model.explainer_attention import KGATAttentionExplainer
from src.train import get_adj_indices
from src.generate_explanations import load_names_and_maps, get_node_name

HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fidelity and Extract Explanations for KGAT")
    parser.add_argument("--model_path", type=str, required=True, help="Path to KGAT checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Data directory")
    parser.add_argument("--raw_data_dir", type=str, default="data/raw", help="Raw data directory for names")
    parser.add_argument("--user_ids_file", type=str, required=True, help="JSON file containing user IDs to evaluate")
    parser.add_argument("--output_explain", type=str, default="output/fidelity/explanations.json", help="Path to output explanations")
    parser.add_argument("--output_metrics", type=str, default="output/fidelity/metrics.json", help="Path to output metrics")
    parser.add_argument("--top_k_paths", type=int, default=3, help="Number of paths to extract and use for Fidelity")
    parser.add_argument("--n_hops", type=int, default=None, help="Max hops for path search (default: auto from model layers)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    return parser.parse_args()

def load_data(data_dir):
    with open(os.path.join(data_dir, "interactions.pkl"), "rb") as f:
        interactions = pickle.load(f)
    if isinstance(interactions, pd.DataFrame):
        interactions = interactions.values
    with open(os.path.join(data_dir, "kg_triples.pkl"), "rb") as f:
        kg_triples = pickle.load(f)
    with open(os.path.join(data_dir, "stats.pkl"), "rb") as f:
        stats = pickle.load(f)
    return interactions, kg_triples, stats

def compute_user_interaction_counts(interactions, n_users):
    """計算每位使用者的歷史互動數量。"""
    user_counts = np.zeros(n_users, dtype=np.int32)
    for row in interactions:
        uid = int(row[0])
        if uid < n_users:
            user_counts[uid] += 1
    return user_counts

def compute_item_kg_degree(kg_triples, n_items):
    """計算每個物品在 KG 中的鄰居數（度數）。"""
    item_degrees = np.zeros(n_items, dtype=np.int32)
    for row in kg_triples:
        head = int(row[0])
        tail = int(row[2]) if len(row) > 2 else int(row[1])
        if head < n_items:
            item_degrees[head] += 1
        if tail < n_items:
            item_degrees[tail] += 1
    return item_degrees

def evaluate_fidelity(args):
    # Setup Device
    device = torch.device("cpu")
    if not args.cpu:
        if HAS_XPU: device = torch.device("xpu")
        elif torch.cuda.is_available(): device = torch.device("cuda")

    print(f"Using device: {device}")

    # Load Data
    interactions, kg_triples, stats = load_data(args.data_dir)
    n_users = stats["n_users"]
    n_items = stats["n_items"]
    n_entities = stats["n_entities"]
    n_relations = stats["n_relations"]

    user_le, item_le, entity_maps, recipe_name_map = load_names_and_maps(args.data_dir, args.raw_data_dir)

    indices, edge_types, num_nodes = get_adj_indices(
        kg_triples, interactions, n_users, n_items, n_entities
    )
    indices = indices.to(device)
    edge_types = edge_types.to(device)

    # 預計算用戶互動數與物品 KG 度數
    print("預計算用戶互動數與物品 KG 度數...")
    user_interaction_counts = compute_user_interaction_counts(interactions, n_users)
    item_kg_degrees = compute_item_kg_degree(kg_triples, n_items)

    # Load User IDs
    if not os.path.exists(args.user_ids_file):
        print(f"Error: 找不到使用者清單 {args.user_ids_file}")
        return
    with open(args.user_ids_file, "r", encoding="utf-8") as f:
        target_users = json.load(f)
    print(f"從 {args.user_ids_file} 載入 {len(target_users)} 位使用者")

    # Load Model
    print(f"Loading checkpoint from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    saved_args = checkpoint.get("args")
    embed_dim = getattr(saved_args, "embed_dim", 64) if saved_args else 64
    layers = getattr(saved_args, "layers", [64, 32]) if saved_args else [64, 32]
    
    # 自動推斷 n_hops（若未指定）
    n_hops = args.n_hops
    if n_hops is None:
        n_hops = min(len(layers), 2)  # 預設最多 2 hops
        print(f"自動推斷 n_hops={n_hops}（模型層數={len(layers)}）")
    else:
        print(f"使用指定 n_hops={n_hops}")

    model = KGATAttention(
        n_users, n_items + n_entities, n_relations + 2,
        embed_dim=embed_dim, layers=layers
    ).to(device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # Load Explainer
    explainer = KGATAttentionExplainer(model)

    # Results tracking
    all_results = []
    f_plus_list = []
    f_minus_list = []

    # Pre-calculate base edge hash
    src, dst = indices[0], indices[1]
    edge_hash = src * num_nodes + dst
    all_items = torch.arange(n_items, device=device)

    print(f"開始評估 Fidelity 與擷取推薦解釋（n_hops={n_hops}）...")

    for user_id in tqdm(target_users, desc="Processing Users"):
        u_batch = torch.full((n_items,), user_id, dtype=torch.long, device=device)
        i_batch = all_items

        with torch.no_grad():
            # forward(indices, edge_types, num_nodes, u, i)
            pos_scores = model(indices, edge_types, num_nodes, u_batch, i_batch)
            
        best_item_idx = torch.argmax(pos_scores).item()
        best_score = pos_scores[best_item_idx].item()
        recommended_item_id = int(best_item_idx)
        p_orig = torch.sigmoid(torch.tensor(best_score)).item()

        # 獲取名稱資訊
        _, rec_real_id, rec_name = get_node_name(
            n_users + recommended_item_id, n_users, n_items, user_le, item_le, entity_maps, recipe_name_map
        )
        _, user_real_id, user_name = get_node_name(
            user_id, n_users, n_items, user_le, item_le, entity_maps, recipe_name_map
        )

        # 用戶與物品的額外統計
        user_n_interactions = int(user_interaction_counts[user_id]) if user_id < n_users else 0
        item_kg_deg = int(item_kg_degrees[recommended_item_id]) if recommended_item_id < n_items else 0

        user_result = {
            "user_id_remapped": user_id,
            "user_id_original": user_real_id,
            "user_name": user_name,
            "recommended_item_id_remapped": recommended_item_id,
            "recommended_item_id_original": rec_real_id,
            "recommended_item_name": rec_name,
            "score": float(best_score),
            "original_prob": float(p_orig),
            "user_n_interactions": user_n_interactions,
            "item_kg_degree": item_kg_deg,
            "explanations": [],
            "fidelity": {}
        }

        # 獲取注意力解釋路徑
        explanations = explainer.explain(
            indices, edge_types, num_nodes, user_id, recommended_item_id,
            n_hops=n_hops, top_k=args.top_k_paths
        )
        
        important_edges = set()
        if explanations and "top_paths" in explanations:
            for path, score in explanations["top_paths"]:
                path_details = []
                path_desc_list = []

                for node_id in path:
                    node_id = int(node_id)
                    n_type, n_real_id, n_name = get_node_name(
                        node_id, n_users, n_items, user_le, item_le, entity_maps, recipe_name_map
                    )
                    path_details.append({"id_remapped": node_id, "id_original": n_real_id, "type": n_type, "name": n_name})
                    path_desc_list.append(n_name)

                user_result["explanations"].append({
                    "path_structure": " -> ".join([p["type"] for p in path_details]),
                    "path_description": " -> ".join(path_desc_list),
                    "path_details": path_details,
                    "contribution_score": float(score),
                })
                
                # Retrieve edges for fidelity
                for step in range(len(path) - 1):
                    n1, n2 = int(path[step]), int(path[step+1])
                    important_edges.add((n1, n2))
                    important_edges.add((n2, n1)) # 無向圖需正反兩面遮蔽

        if not important_edges:
            user_result["explanations_note"] = "無法找到解釋路徑。"
            all_results.append(user_result)
            continue

        hashed_important = [n1 * num_nodes + n2 for n1, n2 in important_edges]
        hashed_important = torch.tensor(hashed_important, device=device)

        # 準備 Mask
        mask_important = torch.isin(edge_hash, hashed_important)
        mask_self = (src == dst)
        
        # ======= Fidelity+ (移除重要路徑) =======
        mask_f_plus = ~mask_important
        indices_f_plus = indices[:, mask_f_plus]
        edge_types_f_plus = edge_types[mask_f_plus]
        
        with torch.no_grad():
            f_plus_scores = model(indices_f_plus, edge_types_f_plus, num_nodes, 
                                  torch.tensor([user_id], device=device), 
                                  torch.tensor([recommended_item_id], device=device))
            score_f_plus = f_plus_scores[0].item()
            p_f_plus = torch.sigmoid(torch.tensor(score_f_plus)).item()
            
            f_plus_val = p_orig - p_f_plus
            f_plus_list.append(f_plus_val)
            user_result["fidelity"]["fidelity_plus"] = float(f_plus_val)
            user_result["fidelity"]["prob_f_plus"] = float(p_f_plus)
            
        # ======= Fidelity- (僅保留重要路徑與 Self-loop) =======
        mask_f_minus = mask_important | mask_self
        indices_f_minus = indices[:, mask_f_minus]
        edge_types_f_minus = edge_types[mask_f_minus]
        
        with torch.no_grad():
            f_minus_scores = model(indices_f_minus, edge_types_f_minus, num_nodes, 
                                  torch.tensor([user_id], device=device), 
                                  torch.tensor([recommended_item_id], device=device))
            score_f_minus = f_minus_scores[0].item()
            p_f_minus = torch.sigmoid(torch.tensor(score_f_minus)).item()
            
            f_minus_val = p_orig - p_f_minus
            f_minus_list.append(f_minus_val)
            user_result["fidelity"]["fidelity_minus"] = float(f_minus_val)
            user_result["fidelity"]["prob_f_minus"] = float(p_f_minus)

        all_results.append(user_result)

    # 4. 平均與總結
    os.makedirs(os.path.dirname(args.output_explain), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_metrics), exist_ok=True)
    
    with open(args.output_explain, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 統計有路徑 vs 無路徑
    path_users = [r for r in all_results if r.get("explanations") and len(r["explanations"]) > 0]
    no_path_users = [r for r in all_results if not r.get("explanations") or len(r["explanations"]) == 0]

    metrics_result = {
        "num_evaluated": len(f_plus_list),
        "total_requested": len(target_users),
        "n_hops": n_hops,
        "model_layers": len(layers),
        "path_coverage": {
            "has_path": len(path_users),
            "no_path": len(no_path_users),
            "coverage_rate": len(path_users) / len(target_users) if target_users else 0,
        },
        "no_path_user_stats": {},
    }

    if len(f_plus_list) > 0:
        avg_f_plus = float(np.mean(f_plus_list))
        avg_f_minus = float(np.mean(f_minus_list))
        std_f_plus = float(np.std(f_plus_list))
        std_f_minus = float(np.std(f_minus_list))
        metrics_result["avg_fidelity_plus"] = avg_f_plus
        metrics_result["avg_fidelity_minus"] = avg_f_minus
        metrics_result["std_fidelity_plus"] = std_f_plus
        metrics_result["std_fidelity_minus"] = std_f_minus
        
        print("="*40)
        print("可解釋性量化指標 (Fidelity) 評估結果")
        print(f"樣本數: {len(f_plus_list)} / {len(target_users)}")
        print(f"路徑覆蓋率: {len(path_users)}/{len(target_users)} ({len(path_users)/len(target_users)*100:.1f}%)")
        print(f"Fidelity+ (越正越好): {avg_f_plus:.4f} ± {std_f_plus:.4f}")
        print(f"Fidelity- (越小越好): {avg_f_minus:.4f} ± {std_f_minus:.4f}")
        print("="*40)
    else:
        print("無法評估：未找到任何成功萃取的路徑。")

    # 無路徑用戶統計
    if no_path_users:
        no_path_interactions = [r.get("user_n_interactions", 0) for r in no_path_users]
        no_path_kg_degrees = [r.get("item_kg_degree", 0) for r in no_path_users]
        no_path_probs = [r.get("original_prob", 0) for r in no_path_users]

        path_interactions = [r.get("user_n_interactions", 0) for r in path_users]
        path_kg_degrees = [r.get("item_kg_degree", 0) for r in path_users]
        path_probs = [r.get("original_prob", 0) for r in path_users]

        metrics_result["no_path_user_stats"] = {
            "avg_interactions": float(np.mean(no_path_interactions)),
            "median_interactions": float(np.median(no_path_interactions)),
            "avg_item_kg_degree": float(np.mean(no_path_kg_degrees)),
            "median_item_kg_degree": float(np.median(no_path_kg_degrees)),
            "avg_prob": float(np.mean(no_path_probs)),
        }
        metrics_result["path_user_stats"] = {
            "avg_interactions": float(np.mean(path_interactions)),
            "median_interactions": float(np.median(path_interactions)),
            "avg_item_kg_degree": float(np.mean(path_kg_degrees)),
            "median_item_kg_degree": float(np.median(path_kg_degrees)),
            "avg_prob": float(np.mean(path_probs)),
        }

        print(f"\n--- 有路徑 vs 無路徑 用戶比較 ---")
        print(f"有路徑用戶平均互動數: {np.mean(path_interactions):.1f} | 無路徑: {np.mean(no_path_interactions):.1f}")
        print(f"有路徑用戶平均物品KG度數: {np.mean(path_kg_degrees):.1f} | 無路徑: {np.mean(no_path_kg_degrees):.1f}")
        print(f"有路徑用戶平均預測信心: {np.mean(path_probs):.4f} | 無路徑: {np.mean(no_path_probs):.4f}")

    with open(args.output_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics_result, f, indent=2, ensure_ascii=False)
        
    print(f"\n解釋路徑已儲存至: {args.output_explain}")
    print(f"Fidelity 指標已儲存至: {args.output_metrics}")

if __name__ == "__main__":
    args = parse_args()
    evaluate_fidelity(args)
