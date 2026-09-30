"""Independent arithmetic diagnostic for the CPU oracle's small GEMM paths."""
import numpy as np
import torch
torch.set_num_threads(1)
torch.manual_seed(1024)
for rows in [2, 3, 4]:
    for width in [2, 4, 8, 13, 16, 32]:
        for outputs in [1, 2, 3, 4, 7, 8, 9, 12, 15, 16]:
            x = torch.randn(rows, width)
            w = torch.randn(outputs, width)
            target = (x @ w.T).numpy()
            candidates = []
            for lanes, fused in [(1, True), (4, False), (8, False), (16, False)]:
                a = np.zeros((rows, outputs, lanes), dtype=np.float32)
                for k in range(width):
                    p = x[:,k,None].numpy().astype(np.float64) * w[None,:,k].numpy().astype(np.float64)
                    if not fused: p = p.astype(np.float32).astype(np.float64)
                    a[:,:,k%lanes] = p + a[:,:,k%lanes].astype(np.float64)
                while a.shape[-1] > 1:
                    half = a.shape[-1]//2
                    a = a[:,:,:half] + a[:,:,half:]
                candidates.append((int(np.count_nonzero(a[:,:,0] != target)), lanes, fused))
            print(rows, width, outputs, sorted(candidates), flush=True)
