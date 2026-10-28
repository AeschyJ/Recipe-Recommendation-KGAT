"""KGAT 注意力路徑搜尋效能基準測試腳本 (CSR vs NetworkX).

本腳本系統性評估並對比以下指標：
1. 建圖開支 (Graph Construction Overhead):
   - CSR 矩陣建構 (scipy.sparse.csr_matrix)
   - NetworkX 靜態拓撲預先建圖 (nx.Graph)
   - NetworkX 動態帶權建圖 (nx.Graph with attention weights > 1e-6)
2. 尋路遍歷延遲 (Traversal Latency @ cutoff=3 / L=3 hops):
   - NetworkX 單向 DFS (all_simple_paths)
   - NetworkX 雙向折半搜尋 (Bidirectional Meet-in-the-middle)
   - CSR 單向遍歷 (CSR forward DFS + binary search)
   - CSR 雙向局部交集 (CSR bidirectional meet-in-the-middle)
3. 數學與拓撲等價性驗證 (100% Path Equivalence Verification)
4. 多維度消融拆解 (Ablation: Data Structure vs Algorithmic Complexity)

測試結果同步輸出至控制台、output/benchmark_path_search.log 與 output/benchmark_path_search.json。
"""

import argparse
import gc
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import psutil
import scipy.sparse as sp
import networkx as nx

# 設定工作目錄錨定
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.train import get_adj_indices


def setup_logger(log_path: Path):
    """配置雙路徑日誌記錄器 (Console + File)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler (UTF-8)
    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    # Stream handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def get_current_rss_mb() -> float:
    """獲取當前行程 RSS 記憶體 (MB)."""
    return psutil.Process().memory_info().rss / (1024 ** 2)


def benchmark_graph_construction(
    src_np: np.ndarray,
    dst_np: np.ndarray,
    num_nodes: int,
    logger: logging.Logger,
    n_repeats: int = 3,
) -> Dict[str, Any]:
    """評估建圖耗時與記憶體佔用."""
    logger.info("=" * 80)
    logger.info(">>> 階段 1：圖資料結構建置開支評估 (Graph Construction Overhead Benchmark)")
    logger.info("=" * 80)

    num_edges = len(src_np)
    logger.info(f"全圖總節點數: {num_nodes:,} | 總有向邊數: {num_edges:,}")

    # 1. CSR 矩陣建構
    gc.collect()
    rss_before_csr = get_current_rss_mb()
    csr_times = []
    csr_adj = None
    for r in range(n_repeats):
        t0 = time.perf_counter()
        # 建立 CSR 矩陣並排序索引
        adj = sp.csr_matrix(
            (np.ones(num_edges, dtype=np.int32), (src_np, dst_np)),
            shape=(num_nodes, num_nodes),
        )
        adj.sort_indices()
        t1 = time.perf_counter()
        csr_times.append(t1 - t0)
        if r == 0:
            csr_adj = adj

    rss_after_csr = get_current_rss_mb()
    csr_raw_mb = (
        csr_adj.data.nbytes + csr_adj.indices.nbytes + csr_adj.indptr.nbytes
    ) / (1024 ** 2)
    csr_rss_delta_mb = max(0.0, rss_after_csr - rss_before_csr)
    csr_time_mean_ms = float(np.mean(csr_times)) * 1000
    csr_time_std_ms = float(np.std(csr_times)) * 1000

    logger.info(
        f"[CSR Matrix]       建構耗時: {csr_time_mean_ms:8.2f} ± {csr_time_std_ms:5.2f} ms | "
        f"核心陣列記憶體: {csr_raw_mb:7.2f} MB | 行程 RSS 增幅: {csr_rss_delta_mb:7.2f} MB"
    )

    # 2. NetworkX 靜態拓撲建構 (Static unweighted)
    gc.collect()
    rss_before_nx_static = get_current_rss_mb()
    t0 = time.perf_counter()
    G_static = nx.Graph()
    # 使用 tuple 邊列表進行批次添加
    G_static.add_edges_from(zip(src_np, dst_np))
    t1 = time.perf_counter()
    nx_static_time_s = t1 - t0
    rss_after_nx_static = get_current_rss_mb()
    nx_static_rss_delta_mb = max(0.0, rss_after_nx_static - rss_before_nx_static)

    logger.info(
        f"[NetworkX 靜態]    建構耗時: {nx_static_time_s:8.2f} s ({nx_static_time_s*1000:8.1f} ms) | "
        f"無向邊數: {G_static.number_of_edges():,} | 行程 RSS 增幅: {nx_static_rss_delta_mb:7.2f} MB"
    )

    # 3. NetworkX 動態帶權建構 (Dynamic per-query simulation with threshold > 1e-6)
    gc.collect()
    rss_before_nx_dyn = get_current_rss_mb()
    # 產生具備長尾特徵之注意力模擬權重
    sim_weights = np.random.uniform(1e-5, 1.0, size=num_edges).astype(np.float32)
    edge_indices = np.vstack([src_np, dst_np]).T

    t0 = time.perf_counter()
    G_dyn = nx.Graph()
    for (s, d), w in zip(edge_indices, sim_weights):
        if w > 1e-6:
            G_dyn.add_edge(int(s), int(d), weight=float(w))
    t1 = time.perf_counter()
    nx_dyn_time_s = t1 - t0
    rss_after_nx_dyn = get_current_rss_mb()
    nx_dyn_rss_delta_mb = max(0.0, rss_after_nx_dyn - rss_before_nx_dyn)

    logger.info(
        f"[NetworkX 動態帶權] 建構耗時: {nx_dyn_time_s:8.2f} s ({nx_dyn_time_s*1000:8.1f} ms) | "
        f"帶權邊數: {G_dyn.number_of_edges():,} | 行程 RSS 增幅: {nx_dyn_rss_delta_mb:7.2f} MB"
    )

    # 釋放動態圖以省記憶體
    del G_dyn, sim_weights, edge_indices
    gc.collect()

    return {
        "csr": {
            "time_ms_mean": csr_time_mean_ms,
            "time_ms_std": csr_time_std_ms,
            "raw_array_mb": csr_raw_mb,
            "rss_delta_mb": csr_rss_delta_mb,
        },
        "nx_static": {
            "time_s": nx_static_time_s,
            "time_ms": nx_static_time_s * 1000,
            "rss_delta_mb": nx_static_rss_delta_mb,
            "edges": G_static.number_of_edges(),
        },
        "nx_dynamic": {
            "time_s": nx_dyn_time_s,
            "time_ms": nx_dyn_time_s * 1000,
            "rss_delta_mb": nx_dyn_rss_delta_mb,
        },
        "csr_adj": csr_adj,
        "G_static": G_static,
    }


def search_nx_unidirectional(G: nx.Graph, source: int, target: int, cutoff: int = 3) -> List[List[int]]:
    """NetworkX 預設單向 DFS 尋路 (all_simple_paths)."""
    if source not in G or target not in G:
        return []
    return list(nx.all_simple_paths(G, source=source, target=target, cutoff=cutoff))


def search_nx_bidirectional(G: nx.Graph, source: int, target: int) -> List[List[int]]:
    """NetworkX 雙向折半中間相遇搜尋 (Meet-in-the-Middle on Python Dicts)."""
    if source not in G or target not in G:
        return []
    if source == target:
        return [[source]]

    paths = []
    nbrs_src = G[source]
    nbrs_tgt = G[target]

    # 直連 1-hop
    if target in nbrs_src:
        paths.append([source, target])

    # 2-hop: source -> n -> target
    common = (set(nbrs_src) & set(nbrs_tgt)) - {source, target}
    for n in common:
        paths.append([source, n, target])

    # 3-hop: source -> r -> m -> target
    for r in nbrs_src:
        if r == source or r == target:
            continue
        for m in G[r]:
            if m != source and m != r and m != target and m in nbrs_tgt:
                paths.append([source, r, m, target])
    return paths


def search_csr_unidirectional(
    indptr: np.ndarray, col_indices: np.ndarray, source: int, target: int
) -> List[List[int]]:
    """CSR 單向遍歷搜尋 (DFS forward + Binary search on sorted columns)."""
    num_nodes = len(indptr) - 1
    if source < 0 or source >= num_nodes or target < 0 or target >= num_nodes:
        return []
    if source == target:
        return [[source]]

    paths = []
    src_nbrs = col_indices[indptr[source] : indptr[source + 1]]

    for r in src_nbrs:
        if r == target:
            paths.append([source, target])
            continue
        if r == source:
            continue

        start_r, end_r = indptr[r], indptr[r + 1]
        cols_r = col_indices[start_r:end_r]

        # 2-hop: 檢查 target 是否在 r 之鄰域中
        pos_t = np.searchsorted(cols_r, target)
        if pos_t < len(cols_r) and cols_r[pos_t] == target:
            paths.append([source, int(r), target])

        # 3-hop: 遍歷 m in cols_r
        for m in cols_r:
            if m == source or m == r or m == target:
                continue

            # 單向檢查: 透過二分搜尋檢查 target 是否存在於 m 的鄰域清單中
            start_m, end_m = indptr[m], indptr[m + 1]
            cols_m = col_indices[start_m:end_m]
            pos = np.searchsorted(cols_m, target)
            if pos < len(cols_m) and cols_m[pos] == target:
                paths.append([source, int(r), int(m), target])
    return paths


def search_csr_bidirectional(
    indptr: np.ndarray, col_indices: np.ndarray, source: int, target: int
) -> List[List[int]]:
    """CSR 雙向局部交集向量化尋路 (Meet-in-the-Middle with Hash Set)."""
    num_nodes = len(indptr) - 1
    if source < 0 or source >= num_nodes or target < 0 or target >= num_nodes:
        return []
    if source == target:
        return [[source]]

    paths = []
    src_nbrs = col_indices[indptr[source] : indptr[source + 1]]
    tgt_nbrs = col_indices[indptr[target] : indptr[target + 1]]
    tgt_set = set(tgt_nbrs) - {source, target}

    # 直連 1-hop
    if target in src_nbrs:
        paths.append([source, target])

    # 2-hop: source -> n -> target
    common = set(src_nbrs) & tgt_set
    for n in common:
        paths.append([source, int(n), target])

    # 3-hop: source -> r -> m -> target
    for r in src_nbrs:
        if r == source or r == target:
            continue
        r_nbrs = col_indices[indptr[r] : indptr[r + 1]]
        for m in r_nbrs:
            if m in tgt_set and m != r:
                paths.append([source, int(r), int(m), target])
    return paths


def normalize_paths(paths: List[List[int]]) -> Set[Tuple[int, ...]]:
    """標準化路徑集合以供精確等價性驗證."""
    return set(tuple(p) for p in paths)


def run_benchmark():
    parser = argparse.ArgumentParser(description="KGAT Path Extraction Benchmark (CSR vs NetworkX)")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--demo_cases", type=str, default="demo/web/demo_cases.json")
    parser.add_argument("--output_log", type=str, default="output/benchmark_path_search.log")
    parser.add_argument("--output_json", type=str, default="output/benchmark_path_search.json")
    parser.add_argument("--repeats", type=int, default=5, help="Number of repetitions per query")
    args = parser.parse_args()

    log_path = Path(args.output_log)
    logger = setup_logger(log_path)

    logger.info("=" * 80)
    logger.info("   KGAT 可解釋性路徑搜尋效能極限基準測試 (CSR vs NetworkX Benchmark)")
    logger.info("   國立臺灣大學 碩士學位口試展示與論文實證數據生成模組")
    logger.info("=" * 80)

    # 載入統計資訊與圖結構
    stats_path = Path(args.data_dir) / "stats.pkl"
    kg_path = Path(args.data_dir) / "kg_triples.pkl"
    int_path = Path(args.data_dir) / "interactions.pkl"

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    with open(kg_path, "rb") as f:
        kg_triples = pickle.load(f)
    with open(int_path, "rb") as f:
        interactions = pickle.load(f)
    if hasattr(interactions, "values"):
        interactions = interactions.values

    n_users = stats["n_users"]
    n_items = stats["n_items"]
    n_entities = stats["n_entities"]

    indices, edge_types, num_nodes = get_adj_indices(
        kg_triples, interactions, n_users, n_items, n_entities
    )
    src_np = indices[0].numpy()
    dst_np = indices[1].numpy()

    # 執行階段 1: 建圖開支測試
    const_res = benchmark_graph_construction(src_np, dst_np, num_nodes, logger, n_repeats=3)
    csr_adj = const_res["csr_adj"]
    G_static = const_res["G_static"]
    indptr = csr_adj.indptr
    col_indices = csr_adj.indices

    # 挑選具備代表性之多層級測試使用者與物品 (15 組)
    logger.info("\n" + "=" * 80)
    logger.info(">>> 階段 2：建構代表性測試樣本群 (涵蓋低/中/高互動度與口試核心案例)")
    logger.info("=" * 80)

    # 計算使用者度數
    users_unique, user_degrees = np.unique(interactions[:, 0], return_counts=True)
    user_deg_map = dict(zip(users_unique, user_degrees))

    sample_cohort: List[Dict[str, Any]] = []

    # 1. 口試 5 大經典學術案例
    demo_json_path = Path(args.demo_cases)
    if demo_json_path.exists():
        with open(demo_json_path, "r", encoding="utf-8") as f:
            demo_data = json.load(f)
        for c in demo_data.get("academic_cases", []):
            uid = c["user_id"]
            rec_id = c["data"]["3"]["recommended_item_id"]
            target_node = n_users + rec_id
            deg = user_deg_map.get(uid, 1)
            sample_cohort.append(
                {
                    "group": "Academic Cases",
                    "case_name": c["name"],
                    "user_id": uid,
                    "target_node": target_node,
                    "target_item_id": rec_id,
                    "user_degree": int(deg),
                }
            )

        # 2. 中低度使用者 (從 user_cache 抽樣 5 位)
        uc = demo_data.get("user_cache", {})
        count_mid = 0
        for u_str, u_info in uc.items():
            uid = int(u_str)
            if "3" in u_info.get("depths", {}):
                rec_id = u_info["depths"]["3"]["recommended_item_id"]
                target_node = n_users + rec_id
                deg = user_deg_map.get(uid, 1)
                if 2 <= deg <= 10:
                    sample_cohort.append(
                        {
                            "group": "Mid-Degree Users",
                            "case_name": f"User {uid} (deg={deg})",
                            "user_id": uid,
                            "target_node": target_node,
                            "target_item_id": rec_id,
                            "user_degree": int(deg),
                        }
                    )
                    count_mid += 1
                    if count_mid >= 5:
                        break
    else:
        # Fallback: 若無 demo_cases.json，自 interactions 抽樣
        low_users = [u for u, d in user_deg_map.items() if d == 1][:5]
        for i, uid in enumerate(low_users):
            target_item = (500 * (i + 1)) % n_items
            sample_cohort.append(
                {
                    "group": "Academic Cases (Fallback)",
                    "case_name": f"Low-Deg User {uid} (deg=1)",
                    "user_id": int(uid),
                    "target_node": n_users + target_item,
                    "target_item_id": target_item,
                    "user_degree": 1,
                }
            )
        mid_users = [u for u, d in user_deg_map.items() if 4 <= d <= 8][:5]
        for i, uid in enumerate(mid_users):
            target_item = (700 * (i + 1)) % n_items
            sample_cohort.append(
                {
                    "group": "Mid-Degree Users (Fallback)",
                    "case_name": f"Mid-Deg User {uid} (deg={user_deg_map[uid]})",
                    "user_id": int(uid),
                    "target_node": n_users + target_item,
                    "target_item_id": target_item,
                    "user_degree": int(user_deg_map[uid]),
                }
            )

    # 3. 高度使用者 (度數 >= 20，挑戰圖爆炸)
    high_deg_users = [u for u, d in user_deg_map.items() if 25 <= d <= 50]
    for i, h_uid in enumerate(high_deg_users[:5]):
        h_deg = user_deg_map[h_uid]
        target_item = (1000 * (i + 1)) % n_items
        sample_cohort.append(
            {
                "group": "High-Degree Stress Users",
                "case_name": f"Stress User {h_uid} (deg={h_deg})",
                "user_id": int(h_uid),
                "target_node": n_users + target_item,
                "target_item_id": target_item,
                "user_degree": int(h_deg),
            }
        )

    # 4. 超巨型熱門食材節點壓力測試 (Top Ingredient Entities deg > 20,000)
    top_ingredient_recipes = kg_triples[kg_triples[:, 2] == 14883][:, 0]
    if len(top_ingredient_recipes) > 0:
        hub_users = [u for u, d in user_deg_map.items() if 10 <= d <= 25][:2]
        for i, u_hub in enumerate(hub_users):
            rec_hub = int(top_ingredient_recipes[i * 100])
            d_hub = user_deg_map[u_hub]
            sample_cohort.append(
                {
                    "group": "Super-Hub Ingredient Stress",
                    "case_name": f"SuperHub User {u_hub} (deg={d_hub})",
                    "user_id": int(u_hub),
                    "target_node": n_users + rec_hub,
                    "target_item_id": rec_hub,
                    "user_degree": int(d_hub),
                }
            )

    # 5. 極限非連通與零路徑測試 (Zero-Path Disconnected Pair)
    sample_cohort.append(
        {
            "group": "Disconnected Zero-Path",
            "case_name": "Disconnected Pair (U10-I99999)",
            "user_id": 10,
            "target_node": n_users + 99999,
            "target_item_id": 99999,
            "user_degree": int(user_deg_map.get(10, 1)),
        }
    )

    logger.info(
        f"成功組裝 {len(sample_cohort)} 組測試樣本 "
        f"(5組經典案例 + 5組中度樣本 + 5組高度壓力 + 2組超熱門食材 + 1組完全非連通)\n"
    )

    # 執行階段 3: 遍歷延遲與等價性測試
    logger.info("=" * 80)
    logger.info(">>> 階段 3：四象限路徑搜尋延遲評估與等價性檢驗 (2x2 Factorial Traversal Benchmark)")
    logger.info("=" * 80)
    header = (
        f"{'Case / User ID':<28} | {'Deg':<4} | {'Paths':<6} | "
        f"{'NX Uni (ms)':<12} | {'NX Bi (ms)':<11} | {'CSR Uni (ms)':<12} | {'CSR Bi (ms)':<11} | {'Speedup':<8} | {'Match'}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    query_results = []

    for item in sample_cohort:
        uid = item["user_id"]
        target = item["target_node"]
        deg = item["user_degree"]
        case_label = f"{item['case_name'][:26]}"

        # 1. NetworkX Unidirectional
        t_nx_uni_runs = []
        paths_nx_uni = []
        # 若度數過高，為免 benchmark 卡死，減少 repeats
        repeats_nx_uni = 2 if deg > 20 else args.repeats
        for _ in range(repeats_nx_uni):
            t0 = time.perf_counter()
            paths_nx_uni = search_nx_unidirectional(G_static, uid, target, cutoff=3)
            t_nx_uni_runs.append((time.perf_counter() - t0) * 1000)
        nx_uni_mean = float(np.mean(t_nx_uni_runs))
        nx_uni_std = float(np.std(t_nx_uni_runs))

        # 2. NetworkX Bidirectional
        t_nx_bi_runs = []
        paths_nx_bi = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            paths_nx_bi = search_nx_bidirectional(G_static, uid, target)
            t_nx_bi_runs.append((time.perf_counter() - t0) * 1000)
        nx_bi_mean = float(np.mean(t_nx_bi_runs))
        nx_bi_std = float(np.std(t_nx_bi_runs))

        # 3. CSR Unidirectional
        t_csr_uni_runs = []
        paths_csr_uni = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            paths_csr_uni = search_csr_unidirectional(indptr, col_indices, uid, target)
            t_csr_uni_runs.append((time.perf_counter() - t0) * 1000)
        csr_uni_mean = float(np.mean(t_csr_uni_runs))
        csr_uni_std = float(np.std(t_csr_uni_runs))

        # 4. CSR Bidirectional
        t_csr_bi_runs = []
        paths_csr_bi = []
        for _ in range(args.repeats * 2):  # 執行更多次取精準平均
            t0 = time.perf_counter()
            paths_csr_bi = search_csr_bidirectional(indptr, col_indices, uid, target)
            t_csr_bi_runs.append((time.perf_counter() - t0) * 1000)
        csr_bi_mean = float(np.mean(t_csr_bi_runs))
        csr_bi_std = float(np.std(t_csr_bi_runs))

        # 驗證數學等價性
        set_nx_uni = normalize_paths(paths_nx_uni)
        set_nx_bi = normalize_paths(paths_nx_bi)
        set_csr_uni = normalize_paths(paths_csr_uni)
        set_csr_bi = normalize_paths(paths_csr_bi)

        is_match = (
            set_nx_uni == set_nx_bi == set_csr_uni == set_csr_bi
        )
        match_str = "PASS (100%)" if is_match else "FAIL"

        speedup = nx_uni_mean / max(1e-5, csr_bi_mean)

        row_str = (
            f"{case_label:<28} | {deg:<4} | {len(paths_csr_bi):<6} | "
            f"{nx_uni_mean:8.2f} ms  | {nx_bi_mean:7.3f} ms  | {csr_uni_mean:8.3f} ms   | {csr_bi_mean:7.3f} ms  | "
            f"{speedup:7.1f}x | {match_str}"
        )
        logger.info(row_str)

        query_results.append(
            {
                "case_name": item["case_name"],
                "group": item["group"],
                "user_id": uid,
                "target_node": target,
                "user_degree": deg,
                "paths_count": len(paths_csr_bi),
                "is_match": is_match,
                "latency_ms": {
                    "nx_unidirectional": {"mean": nx_uni_mean, "std": nx_uni_std},
                    "nx_bidirectional": {"mean": nx_bi_mean, "std": nx_bi_std},
                    "csr_unidirectional": {"mean": csr_uni_mean, "std": csr_uni_std},
                    "csr_bidirectional": {"mean": csr_bi_mean, "std": csr_bi_std},
                },
                "speedup_nx_uni_vs_csr_bi": speedup,
                "speedup_algorithmic_nx": nx_uni_mean / max(1e-5, nx_bi_mean),
                "speedup_algorithmic_csr": csr_uni_mean / max(1e-5, csr_bi_mean),
                "speedup_datastruct_uni": nx_uni_mean / max(1e-5, csr_uni_mean),
                "speedup_datastruct_bi": nx_bi_mean / max(1e-5, csr_bi_mean),
            }
        )

    # 執行階段 4: 消融實驗與統計分析 (Ablation Analysis)
    logger.info("\n" + "=" * 80)
    logger.info(">>> 階段 4：多軸向消融實驗分解 (Ablation Study Decomposition)")
    logger.info("=" * 80)

    avg_nx_uni = np.mean([r["latency_ms"]["nx_unidirectional"]["mean"] for r in query_results])
    avg_nx_bi = np.mean([r["latency_ms"]["nx_bidirectional"]["mean"] for r in query_results])
    avg_csr_uni = np.mean([r["latency_ms"]["csr_unidirectional"]["mean"] for r in query_results])
    avg_csr_bi = np.mean([r["latency_ms"]["csr_bidirectional"]["mean"] for r in query_results])

    overall_traversal_speedup = avg_nx_uni / avg_csr_bi
    overall_algorithmic_gain_nx = avg_nx_uni / avg_nx_bi
    overall_algorithmic_gain_csr = avg_csr_uni / avg_csr_bi
    overall_hardware_gain_uni = avg_nx_uni / avg_csr_uni
    overall_hardware_gain_bi = avg_nx_bi / avg_csr_bi

    # 端到端加速比 (含動態建圖 vs CSR 預建)
    nx_dynamic_total_ms = const_res["nx_dynamic"]["time_ms"] + avg_nx_uni
    end_to_end_speedup = nx_dynamic_total_ms / avg_csr_bi

    logger.info(f"1. 平均尋路延遲 (Average Query Latency):")
    logger.info(f"   - NetworkX 單向 DFS (all_simple_paths):  {avg_nx_uni:9.2f} ms")
    logger.info(f"   - NetworkX 雙向折半 (Meet-in-the-Middle): {avg_nx_bi:9.3f} ms")
    logger.info(f"   - CSR 單向遍歷 (DFS + C-Array Search):    {avg_csr_uni:9.3f} ms")
    logger.info(f"   - CSR 雙向局部交集 (CSR Vectorized):      {avg_csr_bi:9.3f} ms")
    logger.info(f"\n2. 消融維度拆解 (Ablation Decomposition):")
    logger.info(
        f"   - [演算法複雜度維度 O(d^3) -> O(d^2)]:"
        f"\n     * 在 NetworkX 下切換雙向演算法之提速: {overall_algorithmic_gain_nx:8.1f}x"
        f"\n     * 在 CSR 結構下切換雙向演算法之提速:   {overall_algorithmic_gain_csr:8.1f}x"
    )
    logger.info(
        f"   - [硬體/資料結構維度 Python Dict -> C Contiguous Array]:"
        f"\n     * 單向搜尋下更換 CSR 資料結構之提速:   {overall_hardware_gain_uni:8.1f}x"
        f"\n     * 雙向搜尋下更換 CSR 資料結構之提速:   {overall_hardware_gain_bi:8.1f}x"
    )
    logger.info(f"\n3. 綜合搜尋提速比 (Traversal Speedup):         {overall_traversal_speedup:8.1f}x")
    logger.info(
        f"4. 端到端單次查詢綜合提速比 (End-to-End Speedup):   {end_to_end_speedup:8.1f}x"
        f"\n   (含 NetworkX 每次查詢動態建圖 {const_res['nx_dynamic']['time_ms']:.1f} ms + 搜尋延遲 vs CSR 向量化查詢)"
    )

    # 彙整匯出 JSON 資料
    export_payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_nodes": num_nodes,
            "num_edges": len(src_np),
            "num_samples": len(sample_cohort),
        },
        "construction_benchmark": {
            "csr": const_res["csr"],
            "nx_static": const_res["nx_static"],
            "nx_dynamic": const_res["nx_dynamic"],
        },
        "traversal_benchmark_summary": {
            "avg_latency_ms": {
                "nx_unidirectional": float(avg_nx_uni),
                "nx_bidirectional": float(avg_nx_bi),
                "csr_unidirectional": float(avg_csr_uni),
                "csr_bidirectional": float(avg_csr_bi),
            },
            "overall_speedup": {
                "traversal_pure_speedup": float(overall_traversal_speedup),
                "end_to_end_dynamic_speedup": float(end_to_end_speedup),
                "algorithmic_gain_nx": float(overall_algorithmic_gain_nx),
                "algorithmic_gain_csr": float(overall_algorithmic_gain_csr),
                "hardware_gain_uni": float(overall_hardware_gain_uni),
                "hardware_gain_bi": float(overall_hardware_gain_bi),
            },
        },
        "query_results": query_results,
    }

    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, ensure_ascii=False, indent=2)

    logger.info(f"\n基準測試完成！日誌已儲存至: {log_path}")
    logger.info(f"結構化結果已儲存至: {json_path}")


if __name__ == "__main__":
    run_benchmark()
