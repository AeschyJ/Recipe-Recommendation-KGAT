"""
compute_dataset_stats.py
========================
從 data/processed 讀取預處理後的 pickle 檔案，
計算資料集統計表所需數值，並將結果輸出為
data/processed/dataset_stats.csv。

統計欄位
--------
- n_users          : 使用者數（remapped）
- n_items          : 食譜數（remapped）
- n_interactions   : 互動紀錄筆數
- n_kg_triples     : KG 三元組數
- n_entities       : KG 實體數（食材 + 標籤，不含食譜節點）
- n_relations      : KG 關係種類數
- n_ingredients    : 食材實體數
- n_tags           : 標籤實體數
- sparsity         : 互動矩陣稀疏度 (%)
- ckg_nodes        : CKG 總節點數（使用者 + 食譜 + KG 實體）
- ckg_edges        : CKG 總邊數（含反向邊與自迴圈）

用法
----
    python scripts/compute_dataset_stats.py [--data_dir data]
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_stats(data_dir: str) -> dict:
    processed_dir = os.path.join(data_dir, "processed")

    # ---------- 讀取 pickle ----------
    interactions_path = os.path.join(processed_dir, "interactions.pkl")
    kg_triples_path = os.path.join(processed_dir, "kg_triples.pkl")
    stats_path = os.path.join(processed_dir, "stats.pkl")

    for p in [interactions_path, kg_triples_path, stats_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"找不到檔案: {p}\n"
                "請先執行 src/data/preprocess.py 完成資料預處理。"
            )

    print("載入 interactions.pkl ...")
    df_interactions = pd.read_pickle(interactions_path)

    print("載入 kg_triples.pkl ...")
    kg_triples = load_pickle(kg_triples_path)  # numpy array shape (N, 3)

    print("載入 stats.pkl ...")
    stats_meta = load_pickle(stats_path)

    # ---------- 計算統計 ----------
    n_users = int(stats_meta["n_users"])
    n_items = int(stats_meta["n_items"])
    n_entities = int(stats_meta["n_entities"])
    n_relations = int(stats_meta["n_relations"])

    n_ingredients = len(stats_meta.get("ingredient_map", {}))
    n_tags = len(stats_meta.get("tag_map", {}))

    n_interactions = len(df_interactions)

    kg_arr = np.array(kg_triples)
    n_kg_triples = len(kg_arr)

    # 稀疏度 = 1 - (n_interactions / (n_users * n_items))
    sparsity = (1 - n_interactions / (n_users * n_items)) * 100

    # ---------- CKG 結構統計 ----------
    # 依照 src/train_att.py get_adj_indices() 的邏輯：
    #   all_src = [kg_src, kg_dst, int_src, int_dst, self_loops]
    #   all_dst = [kg_dst, kg_src, int_dst, int_src, self_loops]
    # 節點空間：使用者 | 食譜 | KG 實體
    ckg_nodes = n_users + n_items + n_entities

    # 邊數拆解：
    #   KG 正向   : n_kg_triples
    #   KG 反向   : n_kg_triples
    #   互動 正向  : n_interactions
    #   互動 反向  : n_interactions
    #   Self-loop : ckg_nodes
    ckg_edges = (
        n_kg_triples       # KG 正向邊
        + n_kg_triples     # KG 反向邊
        + n_interactions   # 互動正向邊
        + n_interactions   # 互動反向邊
        + ckg_nodes        # 自迴圈
    )

    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "n_kg_triples": n_kg_triples,
        "n_entities": n_entities,
        "n_relations": n_relations,
        "n_ingredients": n_ingredients,
        "n_tags": n_tags,
        "sparsity (%)": round(sparsity, 4),
        "ckg_nodes": ckg_nodes,
        "ckg_edges": ckg_edges,
    }


def main():
    parser = argparse.ArgumentParser(
        description="計算 Food.com 資料集統計表並輸出 CSV"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="資料根目錄（預設: data）",
    )
    args = parser.parse_args()

    stats = compute_stats(args.data_dir)

    # ---------- 列印摘要 ----------
    print("\n" + "=" * 45)
    print("  資料集統計摘要 (Food.com)")
    print("=" * 45)
    for k, v in stats.items():
        print(f"  {k:<22}: {v:>12,}" if isinstance(v, int) else f"  {k:<22}: {v:>12}")
    print("=" * 45)

    # ---------- 輸出 CSV ----------
    output_path = os.path.join(args.data_dir, "processed", "dataset_stats.csv")
    df_stats = pd.DataFrame(list(stats.items()), columns=["metric", "value"])
    df_stats.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n統計表已儲存至: {output_path}")


if __name__ == "__main__":
    main()
