"""Freeze a complete sparse-attention/cache-lifetime autograd graph on CPU.

Cache updates use functional copies (the mathematical overwrite graph), not
detached inference buffers. No step VJPs or native gradients build the oracle.
The forward expression is the existing pinned V4.1 sparse-attention mirror.
Inputs are post-RoPE/post-quantizer FP32 payloads; no quantizer VJP is implied.
"""
import hashlib
import json
from pathlib import Path

import torch
from validate_deepseek_v41_block_grad import bits, deterministic
from validate_deepseek_v41_model import sparse_attn
from validate_deepseek_v41_quantization import quantize_case

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures/deepseek_v41_attention_tape.json"


def case(name, salt, batch, heads, dim, window, prefill, decode, sources, ratio):
    # Each layer has a private ring. Equal nonnegative source IDs share one
    # compressed cache, while different IDs have independent lifetimes.
    layers = len(sources)
    capacity = (prefill + decode) // max(ratio, 1) + 2
    caches, cache_specs, leaves = [], [], []
    writes, reads, events = [], [], []
    serial = salt

    def tensor(shape, scale=0.5):
        nonlocal serial
        serial += 1
        return deterministic(shape, serial, scale)

    def cache(kind, rows):
        x = tensor((batch, rows, dim)).requires_grad_()
        leaves.append(x)
        caches.append(x)
        cache_specs.append(dict(kind=kind, capacity=rows, initial_bits=bits(x)))
        return len(caches) - 1

    windows = [cache("window", window) for _ in sources]
    compressed = {source: cache("compressed", capacity) for source in sorted(set(sources)) if source >= 0}
    sinks = [tensor((heads,)) for _ in sources]

    def write(cid, kind, rows, position):
        raw = tensor((batch, rows, dim))
        mode, scale_mode, block = ("fp4", "e4m3", 16) if kind == "append" else ("fp8", "ue8m0", 32)
        values, scales = quantize_case(raw.reshape(-1, dim).numpy(), block, mode, scale_mode)
        source = torch.from_numpy(values).reshape(raw.shape).requires_grad_()
        if kind == "seed":
            slots = [row % window if row >= rows - window else -1 for row in range(rows)]
        elif kind == "decode":
            slots = [position % window]
        else:
            slots = list(range(position, position + rows))
        updated = caches[cid].clone()
        for row, slot in enumerate(slots):
            if slot >= 0:
                updated[:, slot] = source[:, row]
        caches[cid] = updated
        writes.append(dict(cache=cid, kind=kind, rows=rows, position=position,
                           raw_bits=bits(raw), source_bits=bits(source), scale_bits=bits(torch.from_numpy(scales)),
                           cache_after_bits=bits(updated), _source=source))
        events.append(dict(kind="write", index=len(writes) - 1))
        return len(writes) - 1

    def attend(layer, seq, position, prefill_write=None, warm=False, seed=True):
        cid = windows[layer]
        w = caches[cid] if prefill_write is None else writes[prefill_write]["_source"]
        width = w.shape[1]
        if warm:
            wi = list(range(window))
            topk = window
        elif position == 0:
            topk = min(seq, window)
            wi = [p if p <= row else -1
                  for row in range(seq)
                  for p in range(max(row - window + 1, 0), max(row - window + 1, 0) + topk)]
        else:
            topk = window
            wi = [((position % window + 1 + slot) % window)
                  if (position >= window or position % window + 1 + slot >= window) else -1
                  for slot in range(window)]
        wi = wi * batch
        source_id = sources[layer]
        cc = compressed.get(source_id)
        c = caches[cc] if cc is not None else w[:, :0]
        ready = capacity if warm else ((position + seq) // ratio if ratio else 0)
        ctopk = min(ready, 3) if cc is not None else 0
        ci = []
        for _ in range(batch):
            for row in range(seq):
                visible = ready if warm else (position + row + 1) // max(ratio, 1)
                # Include duplicate compressed gathers and masked positions.
                ci.extend([width + ((layer + slot) % visible)
                           if visible > 0 and (slot + row + layer) % 5 != 0 else -1
                           for slot in range(ctopk)])
        q = tensor((batch, seq, heads, dim)).requires_grad_()
        upstream = tensor(q.shape)
        # One explicit zero gradient remains a real graph use.
        if not warm and position == prefill and layer == layers - 1:
            upstream.zero_()
        wi_tensor = torch.tensor(wi, dtype=torch.long).reshape(batch, seq, topk)
        ci_tensor = torch.tensor(ci, dtype=torch.long).reshape(batch, seq, ctopk)
        indices = torch.cat((wi_tensor, ci_tensor), dim=-1)
        result = sparse_attn(q, torch.cat((w, c), dim=1), sinks[layer], indices, dim ** -0.5)
        reads.append(dict(layer=layer, seq=seq, window_cache=cid,
                          prefill_write=prefill_write, compressed_cache=cc, window_rows=width,
                          topk=topk, compressed_topk=ctopk, wi=wi, ci=ci,
                          query_bits=bits(q), sink_bits=bits(sinks[layer]),
                          upstream_bits=bits(upstream), output_bits=bits(result), seed=seed,
                          _q=q, _out=result, _g=upstream))
        events.append(dict(kind="read", index=len(reads) - 1))

    # A differentiable incoming cache proves that untouched initial rows pass
    # their gradients through the overwrite chain. No detach between steps.
    for layer in range(layers):
        attend(layer, 1, 0, warm=True)
    seed_writes = [write(cid, "seed", prefill, 0) for cid in windows]
    if ratio:
        for cid in compressed.values():
            if prefill // ratio:
                write(cid, "append", prefill // ratio, 0)
    for layer in range(layers):
        attend(layer, prefill, 0, prefill_write=seed_writes[layer])
    for position in range(prefill, prefill + decode):
        # A source publishes completed groups once; all its readers use the
        # same physical cache before the next token's publication.
        if ratio and (position + 1) % ratio == 0:
            for cid in compressed.values():
                write(cid, "append", 1, (position + 1) // ratio - 1)
        for layer, cid in enumerate(windows):
            write(cid, "decode", 1, position)
            attend(layer, 1, position, seed=not (layer == 0 and position == prefill + 1))

    active = [r for r in reads if r["seed"]]
    torch.autograd.backward([r["_out"] for r in active], [r["_g"] for r in active])
    for w in writes:
        g = w.pop("_source").grad
        w["grad_source_bits"] = None if g is None else bits(g)
    for r in reads:
        g = r.pop("_q").grad
        r["grad_query_bits"] = None if g is None else bits(g)
        del r["_out"], r["_g"]
    for spec, leaf in zip(cache_specs, leaves):
        spec["grad_initial_bits"] = None if leaf.grad is None else bits(leaf.grad)
    return dict(name=name, batch=batch, heads=heads, dim=dim, caches=cache_specs,
                writes=writes, reads=reads, events=events)


def main():
    torch.set_num_threads(1)
    mirror = HERE / "validate_deepseek_v41_model.py"
    payload = dict(tolerance=2e-7, rtol=0, comparison="strict_less_than",
        torch_version=torch.__version__, torch_git_version=torch.version.git_version,
        sparse_forward_mirror_sha256=hashlib.sha256(mirror.read_bytes()).hexdigest(),
        contract="FP8/UE8M0 window and FP4/E4M3 compressed payloads; FP32 post-quantization VJPs; fixed indices; full functional overwrite autograd graph",
        cases=[case(*spec) for spec in [
            ("sliding-prefill-longer-than-ring", 3100, 1, 2, 32, 4, 6, 7, [-1], 0),
            ("batched-shared-ratio2", 3200, 2, 3, 32, 4, 3, 8, [0, 0, 0], 2),
            ("two-independent-ratio1-sources", 3300, 1, 4, 64, 3, 2, 6, [0, 0, 2, 2], 1),
        ]])
    if FIXTURE.exists() and json.loads(FIXTURE.read_text()) != payload:
        raise SystemExit("attention-tape fixture changed; refusing to overwrite frozen CPU evidence")
    FIXTURE.write_text(json.dumps(payload, indent=2) + "\n")
    print("PASS CPU attention tape fixture:", len(payload["cases"]), "graphs,",
          sum(len(c["reads"]) for c in payload["cases"]), "reads,",
          sum(len(c["writes"]) for c in payload["cases"]), "writes;",
          hashlib.sha256(FIXTURE.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
