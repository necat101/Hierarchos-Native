"""Independent released-width 384-expert/top-6 Gate.forward CPU regression.

Only the router is materialized. The full-logit schedule case separately runs
six selected experts at production width; this checks selection over all 384
candidates without allocating their payloads. Existing fixtures are immutable.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import validate_deepseek_v41_mixed as mixed

FIXTURE = mixed.FIXTURE.parent / "deepseek_v41_router_width.json"
WEIGHT = FIXTURE.with_suffix(".bf16.bin")
SEED = 4141212


def generate(path):
    namespace = mixed.source_namespace(path)  # checks the pinned source SHA256
    args = SimpleNamespace(dim=5120, get_moe_config=lambda _: (384, 6),
                           score_func="sqrtsoftplus", gate_temp=1.0,
                           norm_topk_prob=True, route_scale=1.5, vision_enabled=False)
    gate = namespace["Gate"](0, args)
    rng = torch.Generator().manual_seed(SEED)
    weight = (torch.randn(384, 5120, generator=rng)*.015).bfloat16()
    with torch.no_grad():
        gate.weight.data = weight
        gate.bias.copy_(torch.linspace(-.15, .15, 384))
    inputs = torch.randn(8, 5120, generator=rng).bfloat16()
    bits = lambda x: x.float().contiguous().view(torch.uint32).flatten().tolist()
    cases = []
    for name, x in [("vector-one", inputs[:1]), ("vector-three", inputs[1:4]),
                    ("matrix-four", inputs[4:])]:
        with torch.inference_mode():
            normalized, indices = gate(x)
            gate.norm_topk_prob, gate.route_scale = False, 1.0
            raw, raw_indices = gate(x)
            gate.norm_topk_prob, gate.route_scale = True, 1.5
            assert torch.equal(raw_indices, indices)
            dots = F.linear(x.float(), weight.float())
        cases.append(dict(name=name, rows=len(x), input_bits=bits(x),
                          dot_bits=bits(dots), indices=indices.flatten().tolist(),
                          raw_bits=bits(raw), normalized_bits=bits(normalized)))
    blob = weight.view(torch.uint16).numpy().astype("<u2").tobytes()
    result = dict(seed=SEED, source_sha256=mixed.packed.MODEL_SHA256,
                  torch_revision=torch.version.git_version, dim=5120, experts=384, topk=6,
                  weight_file=WEIGHT.name, weight_sha256=hashlib.sha256(blob).hexdigest(),
                  bias_bits=bits(gate.bias), cases=cases)
    return result, blob


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--write-fixture", action="store_true")
    args = parser.parse_args()
    if args.write_fixture and (FIXTURE.exists() or WEIGHT.exists()):
        raise FileExistsError("Router-width expectations are frozen; repair native execution")
    torch.set_num_threads(1)
    result, blob = generate(args.reference_path)
    if args.write_fixture:
        WEIGHT.write_bytes(blob)
        FIXTURE.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False))
    else:
        assert WEIGHT.read_bytes() == blob
        assert json.loads(FIXTURE.read_text()) == result
    print("PASS released-width CPU Gate.forward: 384 experts, top-6, rows 1/3/4")


if __name__ == "__main__":
    main()
