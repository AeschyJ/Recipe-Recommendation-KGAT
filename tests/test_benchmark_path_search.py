"""Unit tests for KGAT Path Search Benchmark functions (CSR vs NetworkX).

Tests equivalence, edge cases, boundaries, self-loops, and multi-hop paths.
"""

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

from scripts.benchmark_path_search import (
    normalize_paths,
    search_csr_bidirectional,
    search_csr_unidirectional,
    search_nx_bidirectional,
    search_nx_unidirectional,
)


def _build_test_csr(edges, num_nodes):
    src = np.array([e[0] for e in edges], dtype=np.int32)
    dst = np.array([e[1] for e in edges], dtype=np.int32)
    adj = sp.csr_matrix(
        (np.ones(len(edges), dtype=np.int32), (src, dst)),
        shape=(num_nodes, num_nodes),
    )
    adj.sort_indices()
    return adj


def test_self_loops_and_zero_hop():
    """驗證 source == target 與自環節點之邊界行為，確保不產生重複節點 (例如 [1, 1] 或 [1, 2, 3, 3])."""
    edges = [(0, 0), (1, 1), (2, 2), (3, 3), (1, 2), (2, 1), (2, 3), (3, 2)]
    num_nodes = 4
    adj = _build_test_csr(edges, num_nodes)
    G = nx.Graph()
    G.add_edges_from(edges)

    # 1. source == target
    nx_uni = search_nx_unidirectional(G, 1, 1, cutoff=3)
    nx_bi = search_nx_bidirectional(G, 1, 1)
    csr_uni = search_csr_unidirectional(adj.indptr, adj.indices, 1, 1)
    csr_bi = search_csr_bidirectional(adj.indptr, adj.indices, 1, 1)

    assert nx_uni == [[1]]
    assert nx_bi == [[1]]
    assert csr_uni == [[1]]
    assert csr_bi == [[1]]

    # 2. 1 -> 3 (2-hop: 1 -> 2 -> 3) 且 3 有自環 (3, 3)
    # 舊版 nx_bi 會產出錯誤的 [1, 2, 3, 3]，修復後必須完全一致
    nx_uni_13 = normalize_paths(search_nx_unidirectional(G, 1, 3, cutoff=3))
    nx_bi_13 = normalize_paths(search_nx_bidirectional(G, 1, 3))
    csr_uni_13 = normalize_paths(search_csr_unidirectional(adj.indptr, adj.indices, 1, 3))
    csr_bi_13 = normalize_paths(search_csr_bidirectional(adj.indptr, adj.indices, 1, 3))

    expected_13 = {(1, 2, 3)}
    assert nx_uni_13 == expected_13
    assert nx_bi_13 == expected_13
    assert csr_uni_13 == expected_13
    assert csr_bi_13 == expected_13


def test_out_of_bounds_and_missing_nodes():
    """驗證超出邊界或不存在於圖中的節點能安全回傳空列表，不引發例外."""
    edges = [(1, 2), (2, 1)]
    num_nodes = 5
    adj = _build_test_csr(edges, num_nodes)
    G = nx.Graph()
    G.add_edges_from(edges)

    # 節點 99 不在 G 中且超出 CSR shape
    assert search_nx_unidirectional(G, 1, 99, cutoff=3) == []
    assert search_nx_bidirectional(G, 1, 99) == []
    assert search_csr_unidirectional(adj.indptr, adj.indices, 1, 99) == []
    assert search_csr_bidirectional(adj.indptr, adj.indices, 1, 99) == []

    # 負數節點
    assert search_csr_unidirectional(adj.indptr, adj.indices, -1, 2) == []
    assert search_csr_bidirectional(adj.indptr, adj.indices, -1, 2) == []


def test_multi_hop_general_graph_equivalence():
    """驗證包含 1-hop、2-hop、3-hop 的複合圖結構在四種演算法下完全等價."""
    # 圖結構：
    # 1 -> 2 -> 3 -> 4 (3-hop: 1-2-3-4)
    # 1 -> 5 -> 4      (2-hop: 1-5-4)
    # 1 -> 4           (1-hop: 1-4)
    # 自環 (1,1), (4,4)
    edges = [
        (1, 1), (4, 4),
        (1, 4), (4, 1),
        (1, 5), (5, 1), (5, 4), (4, 5),
        (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3),
    ]
    num_nodes = 6
    adj = _build_test_csr(edges, num_nodes)
    G = nx.Graph()
    G.add_edges_from(edges)

    paths_nx_uni = normalize_paths(search_nx_unidirectional(G, 1, 4, cutoff=3))
    paths_nx_bi = normalize_paths(search_nx_bidirectional(G, 1, 4))
    paths_csr_uni = normalize_paths(search_csr_unidirectional(adj.indptr, adj.indices, 1, 4))
    paths_csr_bi = normalize_paths(search_csr_bidirectional(adj.indptr, adj.indices, 1, 4))

    expected = {
        (1, 4),
        (1, 5, 4),
        (1, 2, 3, 4),
    }

    assert paths_nx_uni == expected
    assert paths_nx_bi == expected
    assert paths_csr_uni == expected
    assert paths_csr_bi == expected


def test_disconnected_pairs():
    """驗證不連通分量回傳空路徑."""
    edges = [(1, 2), (2, 1), (3, 4), (4, 3)]
    num_nodes = 6
    adj = _build_test_csr(edges, num_nodes)
    G = nx.Graph()
    G.add_edges_from(edges)

    assert search_nx_unidirectional(G, 1, 4, cutoff=3) == []
    assert search_nx_bidirectional(G, 1, 4) == []
    assert search_csr_unidirectional(adj.indptr, adj.indices, 1, 4) == []
    assert search_csr_bidirectional(adj.indptr, adj.indices, 1, 4) == []


def test_random_graph_strict_bijection():
    """隨機生成多種類型小圖，驗證 30 組隨機節點對在四種演算法下的絕對集合恆等."""
    rng = np.random.default_rng(42)
    n_nodes = 40
    # 隨機產生無向邊 + 自環
    n_edges = 120
    u_list = rng.integers(0, n_nodes, size=n_edges)
    v_list = rng.integers(0, n_nodes, size=n_edges)
    edges = set()
    for u, v in zip(u_list, v_list):
        edges.add((u, v))
        edges.add((v, u))
    for i in range(n_nodes):
        edges.add((i, i))

    edge_list = list(edges)
    adj = _build_test_csr(edge_list, n_nodes)
    G = nx.Graph()
    G.add_edges_from(edge_list)

    for _ in range(30):
        s = int(rng.integers(0, n_nodes))
        t = int(rng.integers(0, n_nodes))

        res_nx_uni = normalize_paths(search_nx_unidirectional(G, s, t, cutoff=3))
        res_nx_bi = normalize_paths(search_nx_bidirectional(G, s, t))
        res_csr_uni = normalize_paths(search_csr_unidirectional(adj.indptr, adj.indices, s, t))
        res_csr_bi = normalize_paths(search_csr_bidirectional(adj.indptr, adj.indices, s, t))

        assert res_nx_uni == res_nx_bi, f"NX Bi mismatch on ({s}, {t}): {res_nx_uni} vs {res_nx_bi}"
        assert res_nx_uni == res_csr_uni, f"CSR Uni mismatch on ({s}, {t}): {res_nx_uni} vs {res_csr_uni}"
        assert res_nx_uni == res_csr_bi, f"CSR Bi mismatch on ({s}, {t}): {res_nx_uni} vs {res_csr_bi}"
