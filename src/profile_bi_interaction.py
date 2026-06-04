"""
驗證 float32 index_add_ 修復後的完整訓練迴圈效能
"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

from src.model.kgat import KGAT_BiInteraction
from src.train import load_data, construct_adj, sample_bpr_batch, bpr_loss

LOG = "profile_results.txt"


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def profile():
    if os.path.exists(LOG):
        os.remove(LOG)

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    log(f"Device: {device}")

    interactions, kg_triples, stats = load_data()
    n_users = stats["n_users"]
    n_items = stats["n_items"]
    n_entities = stats["n_entities"]
    n_relations = stats["n_relations"]

    np.random.seed(42)
    np.random.shuffle(interactions)
    train_data = interactions[: int(0.8 * len(interactions))]

    adj = construct_adj(kg_triples, interactions, n_users, n_items, n_entities)
    adj = adj.to(device).coalesce().bfloat16()

    graph_target = adj.indices()[0].contiguous()
    graph_neighbor = adj.indices()[1].contiguous()
    graph_values = adj.values().contiguous()
    num_nodes = adj.shape[0]
    del adj
    torch.xpu.empty_cache()

    log(f"Nodes: {num_nodes}, Edges: {graph_target.shape[0]}")

    n_all_entities = n_items + n_entities
    model = KGAT_BiInteraction(
        n_users, n_all_entities, n_relations,
        embed_dim=64, layers=[64]
    ).bfloat16().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    batch_size = 1024

    # Warmup
    log("\n--- Warmup ---")
    model.train()
    u, i, j = sample_bpr_batch(train_data, n_items, batch_size)
    u, i, j = u.to(device), i.to(device), j.to(device)
    optimizer.zero_grad(set_to_none=True)
    pos_s, neg_s = model(graph_target, graph_neighbor, graph_values, num_nodes, u, i, j)
    loss = bpr_loss(pos_s.float(), neg_s.float())
    loss.backward()
    optimizer.step()
    del pos_s, neg_s, loss
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    log(f"  Mem: Alloc={torch.xpu.memory_allocated()/1024**2:.1f}MB, "
        f"Res={torch.xpu.memory_reserved()/1024**2:.1f}MB")

    # Profile 10 iterations
    log("\n--- Profiling 10 real training iterations ---")
    for it in range(10):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        u, i, j = sample_bpr_batch(train_data, n_items, batch_size)
        u, i, j = u.to(device), i.to(device), j.to(device)

        optimizer.zero_grad(set_to_none=True)

        t_fwd = time.perf_counter()
        pos_s, neg_s = model(graph_target, graph_neighbor, graph_values, num_nodes, u, i, j)
        loss = bpr_loss(pos_s.float(), neg_s.float())
        torch.xpu.synchronize()
        t_fwd_end = time.perf_counter()

        loss.backward()
        torch.xpu.synchronize()
        t_bwd_end = time.perf_counter()

        optimizer.step()
        torch.xpu.synchronize()
        t_end = time.perf_counter()

        del pos_s, neg_s, loss
        log(f"  Iter {it+1}: Total={t_end-t0:.4f}s | Fwd={t_fwd_end-t_fwd:.4f}s | "
            f"Bwd={t_bwd_end-t_fwd_end:.4f}s | Step={t_end-t_bwd_end:.4f}s")

    log(f"\n--- Final Memory ---")
    log(f"  Allocated: {torch.xpu.memory_allocated()/1024**2:.1f} MB")
    log(f"  Reserved:  {torch.xpu.memory_reserved()/1024**2:.1f} MB")
    log("Done!")


if __name__ == "__main__":
    profile()
