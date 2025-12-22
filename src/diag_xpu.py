import time

import torch


def benchmark_xpu_ops():
    if not torch.xpu.is_available():
        print("XPU not available")
        return

    device = "xpu"
    N = 487000
    E = 15214975
    dim = 64

    print(f"Benchmarking GNN ops on {device}")
    print(f"N={N}, E={E}, dim={dim}")

    target = torch.randint(0, N, (E,)).to(device)
    neighbor = torch.randint(0, N, (E,)).to(device)
    values = torch.randn(E).to(device)
    features = torch.randn(N, dim).to(device)

    # Test cases: Float32 vs BF16
    for dtype in [torch.float32, torch.bfloat16]:
        print(f"\n--- DTYPE: {dtype} ---")
        f = features.to(dtype)
        v = values.to(dtype)

        # 1. Indexing (Gather)
        torch.xpu.synchronize()
        start = time.time()
        msg_gather = f[neighbor]
        torch.xpu.synchronize()
        t_gather = time.time() - start
        print(f"Gather features[neighbor]: {t_gather:.4f}s")

        # 2. Multiplication
        torch.xpu.synchronize()
        start = time.time()
        msg = msg_gather * v.unsqueeze(1)
        torch.xpu.synchronize()
        t_mul = time.time() - start
        print(f"Multiply: {t_mul:.4f}s")

        # 3. index_add_
        res = torch.zeros(N, dim, device=device, dtype=dtype)
        torch.xpu.synchronize()
        start = time.time()
        res.index_add_(0, target, msg)
        torch.xpu.synchronize()
        t_index_add = time.time() - start
        print(f"index_add_: {t_index_add:.4f}s")

        # 4. Total GNN Layer (Forward)
        torch.xpu.synchronize()
        start = time.time()
        msg = f[neighbor] * v.unsqueeze(1)
        res = torch.zeros(N, dim, device=device, dtype=dtype)
        res.index_add_(0, target, msg)
        torch.xpu.synchronize()
        t_total = time.time() - start
        print(f"Total iteration (Manual): {t_total:.4f}s")

    # 5. Test Sparse MM
    print("\n--- Sparse MM (CSR) ---")
    adj_coo = torch.sparse_coo_tensor(
        torch.stack([target, neighbor]), v.float(), (N, N)
    ).to(device)
    adj_csr = adj_coo.to_sparse_csr()
    feat_f32 = f.float()

    try:
        torch.xpu.synchronize()
        start = time.time()
        res_mm = torch.sparse.mm(adj_csr, feat_f32)
        torch.xpu.synchronize()
        print(f"Sparse CSR MM Success! Time: {time.time() - start:.4f}s")
    except Exception as e:
        print(f"Sparse CSR MM Failed: {e}")


if __name__ == "__main__":
    benchmark_xpu_ops()
