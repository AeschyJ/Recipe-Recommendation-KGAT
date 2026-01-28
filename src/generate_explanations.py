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
from src.model.kgat import KGAT
from src.train import construct_adj, load_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成 KGAT 模型推薦解釋 (Generate KGAT Explanations)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(project_root, "models", "kgat_checkpoint_e20.pth"),
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

    # stats 中包含 LabelEncoder (user_map, item_map) 和 dict (ingredient_map, tag_map)
    # 我們需要反向對應：Remapped ID -> Original ID -> Name

    # User / Item LabelEncoders
    # classes_[i] 就是 Remapped ID i 對應的 Original ID (String or Int)
    user_le = stats["user_map"]
    item_le = stats["item_map"]

    # Entity Maps (Name -> ID) -> 需要反轉為 (ID -> Name)
    ingredient_map = stats["ingredient_map"]
    tag_map = stats["tag_map"]

    # 反轉 Entity Map
    id_to_ing = {v: k for k, v in ingredient_map.items()}
    id_to_tag = {v: k for k, v in tag_map.items()}

    # 2. Load Recipe Names from RAW_recipes.csv
    # 我們需要 mapping: Original Recipe ID -> Recipe Name
    recipes_path = os.path.join(raw_data_dir, "RAW_recipes.csv")
    recipe_name_map = {}

    if os.path.exists(recipes_path):
        print(f"正在讀取食譜名稱: {recipes_path}")
        df_recipes = pd.read_csv(recipes_path)
        # 建立 Original Recipe ID -> Name 的對應
        # 假設 csv 有 "id" 和 "name" 欄位
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
        # 我們不知道這個 entity_id 是 ingredient 還是 tag，因為 ID 空間是共用的 (0 ~ n_entities-1)
        # 根據 preprocess.py，ingredient 和 tag 的 ID 是統一分配的 (next_entity_id += 1)
        # 所以直接查兩個 map 即可

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
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"使用裝置: {device}")

    # 3. 建構圖 (Construct Graph)
    print("正在建構鄰接矩陣...")
    adj = construct_adj(kg_triples, interactions, n_users, n_items, n_entities)
    adj = adj.to(device)

    # 4. 初始化模型與載入權重
    print(f"正在載入模型：{args.model_path} ...")
    if not os.path.exists(args.model_path):
        print(f"錯誤：找不到模型檔案 {args.model_path}")
        return

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    embed_dim = 64
    if isinstance(checkpoint, dict) and "args" in checkpoint:
        saved_args = checkpoint["args"]
        embed_dim = getattr(saved_args, "embed_dim", 64)
        print(f"從 Checkpoint 載入超參數: embed_dim={embed_dim}")

    n_all_entities = n_items + n_entities
    model = KGAT(n_users, n_all_entities, n_relations, embed_dim=embed_dim).to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("模型載入成功。")

    # 5. 初始化 Explainer
    explainer = KGATExplainer(model)

    # 6. 挑選使用者
    if args.user_ids:
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

    # 7. 推理與解釋迴圈
    results = []

    # 準備所有 item IDs 用於預測
    all_items = torch.arange(n_items, device=device)

    for user_id in tqdm(target_users, desc="處理使用者"):
        # --- (A) 推理 ---
        u_batch = torch.full((n_items,), user_id, dtype=torch.long, device=device)
        i_batch = all_items
        j_batch = torch.zeros_like(i_batch)  # Dummy

        with torch.no_grad():
            pos_scores, _ = model(adj, u_batch, i_batch, j_batch)

        best_item_idx = torch.argmax(pos_scores).item()
        best_score = pos_scores[best_item_idx].item()

        recommended_item_id = int(best_item_idx)

        # 獲取推薦物品的名稱資訊
        _, rec_real_id, rec_name = get_node_name(
            n_users + recommended_item_id,
            n_users,
            n_items,
            user_le,
            item_le,
            entity_maps,
            recipe_name_map,
        )

        # 獲取使用者名稱資訊
        _, user_real_id, user_name = get_node_name(
            user_id, n_users, n_items, user_le, item_le, entity_maps, recipe_name_map
        )

        # --- (B) 解釋 ---
        explanation_data = explainer.explain(
            adj, user_id, recommended_item_id, top_k=args.top_k_explain
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

    # 8. 儲存結果
    import os

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"解釋生成完畢，結果已儲存至：{output_path}")


if __name__ == "__main__":
    run()
