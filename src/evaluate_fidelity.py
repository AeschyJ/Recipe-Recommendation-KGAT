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

from src.model.kgat_attention import KGATAttention
from src.model.explainer_attention import KGATAttentionExplainer
from src.train_att import get_adj_indices
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

    print(f"開始評估 Fidelity 與擷取推薦解釋...")

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

        user_result = {
            "user_id_remapped": user_id,
            "user_id_original": user_real_id,
            "user_name": user_name,
            "recommended_item_id_remapped": recommended_item_id,
            "recommended_item_id_original": rec_real_id,
            "recommended_item_name": rec_name,
            "score": float(best_score),
            "original_prob": float(p_orig),
            "explanations": [],
            "fidelity": {}
        }

        # 獲取注意力解釋路徑
        explanations = explainer.explain(
            indices, edge_types, num_nodes, user_id, recommended_item_id, top_k=args.top_k_paths
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

    metrics_result = {
        "num_evaluated": len(f_plus_list),
        "total_requested": len(target_users)
    }

    if len(f_plus_list) > 0:
        avg_f_plus = float(np.mean(f_plus_list))
        avg_f_minus = float(np.mean(f_minus_list))
        metrics_result["avg_fidelity_plus"] = avg_f_plus
        metrics_result["avg_fidelity_minus"] = avg_f_minus
        
        print("="*40)
        print("可解釋性量化指標 (Fidelity) 評估結果")
        print(f"樣本數: {len(f_plus_list)}")
        print(f"Fidelity+ (越正越好, 去除關鍵路徑後的效能降幅): {avg_f_plus:.4f}")
        print(f"Fidelity- (越小越好, 僅保留關鍵路徑的效能降幅): {avg_f_minus:.4f}")
        print("="*40)
    else:
        print("無法評估：未找到任何成功萃取的路徑。")

    with open(args.output_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics_result, f, indent=2, ensure_ascii=False)
        
    print(f"解釋路徑已儲存至: {args.output_explain}")
    print(f"Fidelity 指標已儲存至: {args.output_metrics}")

if __name__ == "__main__":
    args = parse_args()
    evaluate_fidelity(args)
