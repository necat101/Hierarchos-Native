"""Locate FP32 projection reduction differences; oracle-only diagnostics."""
import json
import numpy as np
import torch
import verify_falcon_h1 as v

torch.set_num_threads(1)
model = v.tiny()
folder = v.ROOT / '.falcon-h1-fixtures/dual/gated_after'
trace = json.loads((folder / 'native-backward-trace.json').read_text())
x = torch.tensor(trace['forward.layers.0.mamba.input']).reshape(7, 32)
w = model.model.layers[0].mamba.in_proj.weight.detach()
reference = torch.nn.functional.linear(x, w).numpy()
native = np.array(trace['forward.layers.0.mamba.in_proj'], dtype=np.float32).reshape(reference.shape)
print('native', np.max(np.abs(native-reference)), np.count_nonzero(native != reference))
products = x.numpy()[:, None, :].astype(np.float64) * w.numpy()[None, :, :].astype(np.float64)
for fused in [False, True]:
    for lanes in [1, 2, 4, 8, 16]:
        acc = np.zeros((*reference.shape, lanes), dtype=np.float32)
        for k in range(x.shape[-1]):
            p = products[:, :, k] if fused else products[:, :, k].astype(np.float32)
            acc[:, :, k % lanes] = acc[:, :, k % lanes].astype(np.float64) + p
        for reduction in ['left', 'pair', 'halves']:
            if reduction == 'left':
                out = np.zeros_like(reference)
                for k in range(lanes): out = out + acc[:, :, k]
            else:
                a = acc.copy()
                while a.shape[-1] > 1:
                    half = a.shape[-1] // 2
                    a = a[:, :, ::2] + a[:, :, 1::2] if reduction == 'pair' else a[:, :, :half] + a[:, :, half:]
                out = a[:, :, 0]
            print(fused, lanes, reduction, 'hf', np.max(np.abs(out-reference)), np.count_nonzero(out != reference),
                  'native', np.max(np.abs(out-native)), np.count_nonzero(out != native))
