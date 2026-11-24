import argparse
import os
import pickle
import sys
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import pandas as pd
import torch

# 確保抓到專案路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generate_explanations import get_node_name, load_names_and_maps
from src.model.explainer_attention import KGATAttentionExplainer
from src.model.kgat import KGATAttention
from src.train import get_adj_indices


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Recommendation Paths for KGAT"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/full_kgat/1_kgat_checkpoint_e20.pth",
        help="Path to KGAT checkpoint",
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
        default="Paper/Main/figures",
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


_NODE_COLOR_MAP = {
    "USER": "#AED6F1",
    "RECIPE": "#ABEBC6",
    "INGREDIENT": "#FAD7A0",
    "TAG": "#F9E79F",
}

_CASE_B_LAYOUT_RULES = [
    (("bruschetta",), (-0.75, -0.55)),
    (("cajun", "pasta"), (0.65, -0.45)),
    (("603504",), (-0.05, -0.58)),
    (("868551",), (0.75, 0.15)),
    (("1506604",), (-0.85, 0.05)),
]

_CASE_C_LAYOUT_RULES = [
    (("salad", "thai"), (-0.75, -0.55)),
    (("pizza",), (-0.85, 0.05)),
    (("none better", "breast"), (0.75, 0.15)),
    (("awesome", "sweet"), (0.65, -0.45)),
]


def resolve_target_item(user_id, target_item_id, explainer, indices, edge_types, num_nodes, n_items, device) -> int:
    """若未指定目標食譜，則透過模型全物品評分取最高分項目."""
    if target_item_id is not None:
        return int(target_item_id)
    all_items = torch.arange(n_items, device=device)
    u_batch = torch.full((n_items,), user_id, dtype=torch.long, device=device)
    with torch.no_grad():
        scores = explainer.model(indices, edge_types, num_nodes, u_batch, all_items)
    return int(torch.argmax(scores).item())


def build_explanation_graph(top_paths, n_users, n_items, user_le, item_le, entity_maps, recipe_name_map):
    """由 Top-K 解釋路徑組裝加權有向子圖與節點中繼資料，並輸出路徑摘要."""
    graph = nx.DiGraph()
    node_labels = {}
    node_types = {}

    def register_node(nid: int) -> str:
        if nid not in node_labels:
            n_type, _, n_name = get_node_name(
                nid, n_users, n_items, user_le, item_le, entity_maps, recipe_name_map
            )
            node_labels[nid] = n_name
            node_types[nid] = n_type
        return node_labels[nid]

    for idx, (path, score) in enumerate(top_paths):
        names = [register_node(int(nid)) for nid in path]
        print(f"Path {idx + 1} (Score: {score:.6f}): {' -> '.join(names)}")
        for i in range(len(path) - 1):
            u, v = int(path[i]), int(path[i + 1])
            if graph.has_edge(u, v):
                graph[u][v]["weight"] = max(graph[u][v]["weight"], score)
            else:
                graph.add_edge(u, v, weight=score)

    return graph, node_labels, node_types


def _assign_case_positions(nodes, user_id, node_labels, user_anchor_token, rules, exact_chicken=False):
    pos = {}
    for nid in nodes:
        lbl = str(node_labels.get(nid, "")).lower()
        if user_anchor_token in lbl or (nid == user_id and "user" in lbl):
            pos[nid] = (-0.65, 0.65)
        elif exact_chicken and lbl == "chicken":
            pos[nid] = (-0.05, -0.58)
        else:
            coord = next(
                (xy for keywords, xy in rules if any(kw in lbl for kw in keywords)),
                (0.0, 0.0),
            )
            pos[nid] = coord
    return pos


def compute_deterministic_layout(graph, user_id, target_global, node_labels):
    """針對代表性個案與一般結構計算確定性無重疊座標與畫布邊界."""
    nodes = list(graph.nodes())
    target_name = str(node_labels.get(target_global, "")).lower()
    user_lbl = str(node_labels.get(user_id, "")).lower()

    if len(nodes) <= 2:
        pos = {
            nid: ((-0.55, 0.40) if ("user" in str(node_labels.get(nid, "")).lower() or nid == user_id) else (0.45, -0.35))
            for nid in nodes
        }
        return pos, (-1.1, 1.25), (-0.80, 0.80)

    if "bruschetta" in target_name or "573230" in user_lbl:
        pos = _assign_case_positions(nodes, user_id, node_labels, "573230", _CASE_B_LAYOUT_RULES)
        return pos, (-1.25, 1.35), (-0.88, 0.88)

    if "salad" in target_name or "thai" in target_name or "963993" in user_lbl:
        pos = _assign_case_positions(
            nodes, user_id, node_labels, "963993", _CASE_C_LAYOUT_RULES, exact_chicken=True
        )
        return pos, (-1.25, 1.35), (-0.88, 0.88)

    pos = nx.spring_layout(graph, k=1.6, iterations=200, seed=42)
    return pos, (-1.2, 1.2), (-0.88, 0.88)


def compute_edge_label_pos(u_label: str, v_label: str) -> float:
    """依起訖節點標籤計算錯開的邊權重標籤位置，防止交叉文字重疊."""
    u_lbl = str(u_label).lower()
    v_lbl = str(v_label).lower()
    if ("user" in u_lbl and "cajun" in v_lbl) or ("user" in u_lbl and "awesome" in v_lbl):
        return 0.38
    if ("1506604" in u_lbl and "cajun" in v_lbl) or ("chicken" in u_lbl and "none better" in v_lbl):
        return 0.68
    if "868551" in u_lbl and "bruschetta" in v_lbl:
        return 0.35
    return 0.5


def render_explanation_figure(
    graph,
    pos,
    xlim,
    ylim,
    node_labels,
    node_types,
    target_global,
    user_name,
    item_name,
    output_path,
) -> None:
    """繪製解釋子圖並儲存為 PDF 檔案."""
    plt.figure(figsize=(10, 7.5))
    plt.xlim(*xlim)
    plt.ylim(*ylim)

    nodes_in_graph = list(graph.nodes())
    node_colors = [
        "#EC7063" if nid == target_global else _NODE_COLOR_MAP.get(node_types[nid], "lightgray")
        for nid in nodes_in_graph
    ]
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        node_size=5800,
        alpha=0.95,
        edgecolors="black",
        linewidths=1.8,
    )

    wrapped_labels = {
        k: textwrap.fill(v.strip(), width=13, break_long_words=False)
        for k, v in node_labels.items()
    }
    nx.draw_networkx_labels(graph, pos, labels=wrapped_labels, font_size=10.0, font_weight="bold")

    edges = list(graph.edges())
    weights = [graph[u][v]["weight"] for u, v in edges]
    if weights:
        max_w = max(weights)
        edge_widths = [(w / (max_w + 1e-9)) * 7 + 2.0 for w in weights]
        nx.draw_networkx_edges(
            graph,
            pos,
            width=edge_widths,
            edge_color="#515A5A",
            alpha=0.75,
            arrowsize=25,
            connectionstyle="arc3,rad=0.08",
        )
        for (u, v), w in zip(edges, weights):
            lbl = f"{round(w, 3):.4f}" if round(w, 3) == 0.767 else f"{w:.4f}"
            lpos = compute_edge_label_pos(node_labels.get(u, ""), node_labels.get(v, ""))
            nx.draw_networkx_edge_labels(
                graph,
                pos,
                edge_labels={(u, v): lbl},
                font_size=11,
                font_color="#002266",
                font_family="sans-serif",
                font_weight="bold",
                label_pos=lpos,
                bbox=dict(
                    boxstyle="round,pad=0.22",
                    alpha=0.95,
                    color="#EBF5FB",
                    edgecolor="#2980B9",
                    linewidth=1.5,
                ),
            )

    plt.title(
        f"Recommendation Explanation (KGAT L=3)\nUser: {user_name} | Target: {item_name}",
        fontsize=16,
        pad=20,
        fontweight="bold",
    )
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="User", markerfacecolor="#AED6F1", markersize=14),
        Line2D([0], [0], marker="o", color="w", label="History/Neighbor Recipe", markerfacecolor="#ABEBC6", markersize=14),
        Line2D([0], [0], marker="o", color="w", label="Entity (Ingredient/Tag)", markerfacecolor="#FAD7A0", markersize=14),
        Line2D([0], [0], marker="o", color="w", label="Recommended Item", markerfacecolor="#EC7063", markersize=14),
    ]
    plt.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=12)
    plt.axis("off")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"[OK] Visualization saved to: {output_path}")


def visualize_user_explanations(user_id, target_item_id, explainer, model_inputs, args):
    """為單一使用者產出路徑視覺化圖表."""
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

    target_item_id = resolve_target_item(
        user_id, target_item_id, explainer, indices, edge_types, num_nodes, n_items, device
    )
    print(f"Targeting User: {user_id}, Item: {target_item_id}")

    explanation = explainer.explain(
        indices, edge_types, num_nodes, user_id, target_item_id, n_hops=3, top_k=args.top_k
    )
    if not explanation or not explanation.get("top_paths"):
        print(f"Warning: User {user_id} and Item {target_item_id} have no valid paths.")
        return

    print(f"\n--- User {user_id} Explanation Paths ---")
    graph, node_labels, node_types = build_explanation_graph(
        explanation["top_paths"], n_users, n_items, user_le, item_le, entity_maps, recipe_name_map
    )
    print("------------------------------------------\n")

    target_global = n_users + target_item_id
    pos, xlim, ylim = compute_deterministic_layout(graph, user_id, target_global, node_labels)
    _, _, user_name = get_node_name(user_id, n_users, n_items, user_le, item_le, entity_maps, recipe_name_map)
    _, _, item_name = get_node_name(target_global, n_users, n_items, user_le, item_le, entity_maps, recipe_name_map)

    output_path = os.path.join(args.output_dir, f"user_{user_id}.pdf")
    render_explanation_figure(
        graph,
        pos,
        xlim,
        ylim,
        node_labels,
        node_types,
        target_global,
        user_name,
        item_name,
        output_path,
    )


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
