"""Independent CPU AdamW trajectories for the additional 64-wide precision profile.

Reads initial weights and token records; never consumes GPU output or modifies
an existing reference. Equal supervision counts make a four-example batch an
independent reference for two averaged two-example microbatches.
"""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--transformers', type=Path, required=True)
    p.add_argument('--steps', type=int, default=20)
    a = p.parse_args()
    os.environ.update(ATEN_CPU_CAPABILITY='avx2', MKL_ENABLE_INSTRUCTIONS='AVX2', ONEDNN_MAX_CPU_ISA='AVX2')
    sys.path.insert(0, str(a.transformers / 'src'))
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaForCausalLM
    torch.set_num_threads(1)
    rows = [json.loads(line) for line in (a.fixtures / 'dataset.jsonl').read_text().splitlines()]
    extra = []
    for row in rows:
        row = {k: v for k, v in row.items() if k != 'image'}
        extra.append({**row, 'input_ids': [(x + 3) % 64 for x in row['input_ids']],
                      'targets': [(x + 3) % 64 for x in row['targets']]})
    accumulation = [{k: v for k, v in row.items() if k != 'image'} for row in rows] + extra
    with (a.fixtures / 'accumulation.jsonl').open('x') as f:
        f.writelines(json.dumps(row) + '\n' for row in accumulation)
    results = {}
    for name, records in [('ordinary', rows), ('accumulation', accumulation)]:
        model = LlamaForCausalLM.from_pretrained(a.fixtures / 'model/text', attn_implementation='eager').float().train()
        ids = torch.tensor([r['input_ids'] for r in records])
        targets = torch.tensor([r['targets'] for r in records])
        weights = torch.tensor([r['loss_weights'] for r in records], dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9, .999), eps=.001,
                                     weight_decay=0., foreach=False, fused=False)
        losses = []
        for _ in range(a.steps):
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids, attention_mask=torch.ones_like(ids)).logits
            per_token = torch.nn.functional.cross_entropy(logits.reshape(-1, 64), targets.flatten(), reduction='none')
            loss = (per_token * weights.flatten()).sum() / weights.sum()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        destination = a.fixtures / ('cpu-text-' + name + '.safetensors')
        if destination.exists():
            raise FileExistsError(destination)
        save_file({k: v.detach().contiguous().clone() for k, v in model.state_dict().items()}, str(destination))
        results[name] = dict(losses=losses, weights=destination.name,
                             weights_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    report = dict(torch=torch.__version__, numpy=np.__version__, steps=a.steps,
                  transformers_revision=subprocess.check_output(['git', '-C', str(a.transformers), 'rev-parse', 'HEAD'], text=True).strip(),
                  cpu_capability=torch.backends.cpu.get_cpu_capability(), results=results,
                  initial_weights_sha256=hashlib.sha256((a.fixtures / 'model/text/model.safetensors').read_bytes()).hexdigest())
    with (a.fixtures / 'cpu_text.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
