import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch

# 確保抓到專案路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generate_explanations import get_node_name, load_names_and_maps
from src.model.explainer_attention import KGATAttentionExplainer
from src.model.kgat_attention import KGATAttention
from src.train_att import get_adj_indices


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Recommendation Paths for KGAT"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to KGAT checkpoint"
    )
    parser.add_argument(
        "--user_ids", type=str, help="Comma-separated user IDs (remapped)"
    )
    parser.add_argument(
        "--item_ids",
        type=str,
        help="Comma-separated target item IDs (remapped), match with user_ids",
    )
    parser.add_argument(
        "--user_ids_file", type=str, help="JSON file containing list of user IDs"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/processed", help="Data directory"
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default="data/raw",
        help="Raw data directory for names",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="paper/main/figures",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--top_k", type=int, default=3, help="Top K explanation paths to visualize"
    )
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


def setup_chinese_font():
    try:
        fonts = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "sans-serif"]
        plt.rcParams["font.sans-serif"] = fonts
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def visualize_user_explanations(user_id, target_item_id, explainer, model_inputs, args):
    """為單一使用者產出路徑視覺化圖表"""
    (
        indices,
        edge_types,
        num_nodes,
        n_users,
        n_items,
        user_le,
        item_le,
        entity_maps,
        recipe_name_map,
        device,
    ) = model_inputs

    # 如果沒指定，才用 Argmax
    if target_item_id is None:
        all_items = torch.arange(n_items, device=device)
        u_batch = torch.full((n_items,), user_id, dtype=torch.long, device=device)
        with torch.no_grad():
            scores = explainer.model(indices, edge_types, num_nodes, u_batch, all_items)
        target_item_id = torch.argmax(scores).item()

    print(f"Targeting User: {user_id}, Item: {target_item_id}")

    n_hops = 3
    explanation = explainer.explain(
        indices,
        edge_types,
        num_nodes,
        user_id,
        target_item_id,
        n_hops=n_hops,
        top_k=args.top_k,
    )

    if not explanation or not explanation.get("top_paths"):
        print(f"Warning: User {user_id} and Item {target_item_id} have no valid paths.")
        return

    print(f"\n--- User {user_id} Explanation Paths ---")
    for idx, (path, score) in enumerate(explanation["top_paths"]):
        names = []
        for nid in path:
            _, _, n_name = get_node_name(
                nid, n_users, n_items, user_le, item_le, entity_maps, recipe_name_map
            )
            names.append(n_name)
        print(f"Path {idx + 1} (Score: {score:.6f}): {' -> '.join(names)}")
    print("------------------------------------------\n")

    G = nx.DiGraph()
    node_labels = {}
    node_colors = []
    node_types = {}

    def add_node_to_graph(nid):
        if nid not in node_labels:
            n_type, _, n_name = get_node_name(
                nid, n_users, n_items, user_le, item_le, entity_maps, recipe_name_map
            )
            node_labels[nid] = n_name
            node_types[nid] = n_type
            return n_type, n_name
        return node_types[nid], node_labels[nid]

    for path, score in explanation["top_paths"]:
        for i in range(len(path) - 1):
            u, v = int(path[i]), int(path[i + 1])
            add_node_to_graph(u)
            add_node_to_graph(v)
            if G.has_edge(u, v):
                G[u][v]["weight"] = max(G[u][v]["weight"], score)
            else:
                G.add_edge(u, v, weight=score)

    plt.figure(figsize=(13, 11))
    color_map = {
        "USER": "#AED6F1",
        "RECIPE": "#ABEBC6",
        "INGREDIENT": "#FAD7A0",
        "TAG": "#F9E79F",
    }

    nodes_in_graph = list(G.nodes())
    for nid in nodes_in_graph:
        n_type = node_types[nid]
        if nid == n_users + target_item_id:
            node_colors.append("#EC7063")  # Target
        else:
            node_colors.append(color_map.get(n_type, "lightgray"))

    # 使用更加動態的佈局
    pos = nx.spring_layout(G, k=1.2, iterations=150, seed=42)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=3800,
        alpha=0.9,
        edgecolors="black",
        linewidths=1.5,
    )

    wrapped_labels = {
        k: v.replace(" ", "\n") if len(v) > 12 else v for k, v in node_labels.items()
    }
    nx.draw_networkx_labels(
        G, pos, labels=wrapped_labels, font_size=9, font_weight="bold"
    )

    edges = G.edges()
    weights = [G[u][v]["weight"] for u, v in edges]
    if weights:
        max_w = max(weights)
        # 視覺化調整：讓邊線寬度更明顯
        edge_widths = [(w / (max_w + 1e-9)) * 6 + 1.5 for w in weights]
        nx.draw_networkx_edges(
            G,
            pos,
            width=edge_widths,
            edge_color="#616A6B",
            alpha=0.7,
            arrowsize=25,
            connectionstyle="arc3,rad=0.1",
        )

        # 繪製 Edge Labels 顯示分數
        edge_labels = {(u, v): f"{G[u][v]['weight']:.4f}" for u, v in edges}
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=11,
            font_color="#8cdcfe",
            font_family="sans-serif",
            font_weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.2", alpha=0.8, color="white", edgecolor="none"
            ),
        )

    _, _, user_name = get_node_name(
        user_id, n_users, n_items, user_le, item_le, entity_maps, recipe_name_map
    )
    _, _, item_name = get_node_name(
        n_users + target_item_id,
        n_users,
        n_items,
        user_le,
        item_le,
        entity_maps,
        recipe_name_map,
    )

    plt.title(
        f"Recommendation Explanation (KGAT L=3)\nUser: {user_name} | Target: {item_name}",
        fontsize=18,
        pad=25,
        fontweight="bold",
    )

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="User",
            markerfacecolor="#AED6F1",
            markersize=14,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="History/Neighbor Recipe",
            markerfacecolor="#ABEBC6",
            markersize=14,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Entity (Ingredient/Tag)",
            markerfacecolor="#FAD7A0",
            markersize=14,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Recommended Item",
            markerfacecolor="#EC7063",
            markersize=14,
        ),
    ]
    plt.legend(handles=legend_elements, loc="best", frameon=True, fontsize=11)

    plt.axis("off")

    os.makedirs(args.output_dir, exist_ok=True)
    out_name = f"user_{user_id}.pdf"
    output_path = os.path.join(args.output_dir, out_name)
    plt.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"[OK] Visualization saved to: {output_path}")


def main():
    args = parse_args()
    setup_chinese_font()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    interactions, kg_triples, stats = load_data(args.data_dir)
    n_users, n_items = stats["n_users"], stats["n_items"]
    n_entities, n_relations = stats["n_entities"], stats["n_relations"]

    user_le, item_le, entity_maps, recipe_name_map = load_names_and_maps(
        args.data_dir, args.raw_data_dir
    )
    indices, edge_types, num_nodes = get_adj_indices(
        kg_triples, interactions, n_users, n_items, n_entities
    )
    indices, edge_types = indices.to(device), edge_types.to(device)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args")
    embed_dim = getattr(saved_args, "embed_dim", 64) if saved_args else 64
    layers = getattr(saved_args, "layers", [64, 32]) if saved_args else [64, 32]

    model = KGATAttention(
        n_users,
        n_items + n_entities,
        n_relations + 2,
        embed_dim=embed_dim,
        layers=layers,
    ).to(device)
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    explainer = KGATAttentionExplainer(model)

    def resolve_user_id(uid_str):
        try:
            val = int(uid_str)
            if val < n_users:  # 判斷是否已經是 remapped 也可以，但保守起見先確認
                pass
        except ValueError:
            pass
        if user_le is not None:
            for i, c in enumerate(user_le.classes_):
                if str(c) == str(uid_str):
                    return i
        try:
            return int(uid_str)
        except ValueError:
            return None

    def resolve_item_id(iid_str):
        if not iid_str:
            return None
        # Check specific name in recipe_name_map
        for orig_id, name in recipe_name_map.items():
            if str(name).lower() == str(iid_str).lower():
                iid_str = orig_id  # resolved to original ID
                break

        if item_le is not None:
            for i, c in enumerate(item_le.classes_):
                if str(c) == str(iid_str):
                    return i
        try:
            return int(iid_str)
        except ValueError:
            return None

    user_list = []
    item_list = []

    if args.user_ids:
        raw_users = [u.strip() for u in args.user_ids.split(",")]
        raw_items = (
            [i.strip() for i in args.item_ids.split(",")]
            if args.item_ids
            else [None] * len(raw_users)
        )

        for u, i in zip(raw_users, raw_items):
            r_u = resolve_user_id(u)
            r_i = resolve_item_id(i)
            if r_u is not None:
                user_list.append(r_u)
                item_list.append(r_i)

    if not user_list:
        print("Error: 未提供使用者 ID，或解析失敗。")
        return

    print(f"Starting visualization for {len(user_list)} cases...")
    model_inputs = (
        indices,
        edge_types,
        num_nodes,
        n_users,
        n_items,
        user_le,
        item_le,
        entity_maps,
        recipe_name_map,
        device,
    )

    for uid, iid in zip(user_list, item_list):
        try:
            visualize_user_explanations(uid, iid, explainer, model_inputs, args)
        except Exception as e:
            print(f"Failed to visualize User {uid}, Item {iid}: {e}")


if __name__ == "__main__":
    main()
