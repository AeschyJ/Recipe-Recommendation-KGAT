import torch
import sys

for path in sys.argv[1:]:
    cp = torch.load(path, map_location="cpu", weights_only=False)
    a = cp.get("args")
    if a:
        layers = getattr(a, "layers", "N/A")
        embed_dim = getattr(a, "embed_dim", "N/A")
        print(f"{path}: layers={layers}, embed_dim={embed_dim}")
    else:
        print(f"{path}: No saved args")
