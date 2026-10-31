"""Freeze/replay the real layer-27 MoE boundary tie without native expectations.

The original 2,048-token CPU prompt is observed at gate row 1,019, then stopped
before expert execution in that layer. Four copies of its input retain the
gate's matrix reduction path in a small default GPU test. The unchanged pinned
Gate.forward must reproduce the original expert/weight pairs before freezing.
Normal replay reads the same hash-verified checkpoint and frozen input row;
--capture-model-row additionally re-observes the independent model prefix.
No full-model fixture, arithmetic or parity tolerance is rewritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from safetensors import safe_open
import torch
import torch.nn.functional as F

import validate_deepseek_v41_long_context_model as long_context
import validate_deepseek_v41_model as model

FIXTURE = model.FIXTURE.parent / "deepseek_v41_gate_tie.json"
PARENT_SHA256 = "82dcb51beea92833fa8dc382daf30846cd195f5974a17e9eaf0f9c7dd314bbbc"
LAYER, ROW, REPLAY_ROWS = 27, 1019, 4
TIED_EXPERTS = [284, 351]
TIE_SCORE = 1.0706253051757812


def bits(value):
    return value.float().contiguous().view(torch.uint32).flatten().tolist()


def floats(values):
    return torch.tensor(values, dtype=torch.uint32).view(torch.float32)


def parent_and_weights():
    payload = long_context.FIXTURE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == PARENT_SHA256, "parent CPU fixture changed"
    parent = json.loads(payload)["cases"][0]
    assert parent["seed"] == 4141112
    step = parent["steps"][0]
    assert step["name"] == "long-prefill-2048" and step["seq"] == 2048
    root = long_context.FIXTURE.parent / parent["checkpoint"]["directory"]
    for name, digest in parent["checkpoint"]["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]

    def read(name):
        with safe_open(str(root / index[name]), framework="pt") as reader:
            return reader.get_tensor(name).float()

    return parent, step, read(f"layers.{LAYER}.ffn.gate.weight"), read(f"layers.{LAYER}.ffn.gate.bias")


class Captured(Exception):
    """Stop only after the unmodified CPU gate has returned its real output."""


class RowObserver:
    def __init__(self, step, weight, bias):
        self.expected = step["routing"][str(LAYER)]
        self.weight, self.bias = weight, bias
        self.row = None

    def begin(self, spec):
        assert spec["name"] == "long-prefill-2048"

    def attach(self, net):
        def capture(gate, args, output):
            x = args[0]
            weights, indices = output
            assert x.shape == (2048, 32)
            assert bits(gate.weight) == bits(self.weight)
            assert bits(gate.bias) == bits(self.bias)
            assert indices[ROW].tolist() == self.expected["indices"][ROW * 6:(ROW + 1) * 6]
            assert bits(weights[ROW]) == self.expected["weight_bits"][ROW * 6:(ROW + 1) * 6]
            self.row = x[ROW].float().clone()
            print(f"Captured independent CPU layer {LAYER}, row {ROW}", flush=True)
            raise Captured

        net.layers[LAYER].ffn.gate.register_forward_hook(capture)


def generate(args):
    torch.set_num_threads(1)
    parent, step, weight, bias = parent_and_weights()
    namespace = model.load_reference(args.reference_path, args.kernel_reference_path)
    if args.capture_model_row:
        observer = RowObserver(step, weight, bias)
        print("Replaying pinned CPU model up to the known gate tie", flush=True)
        try:
            long_context.make_case(args.reference_path, args.kernel_reference_path,
                                   long_context.SPEC, observer)
        except Captured:
            pass
        assert observer.row is not None, "reference never reached the boundary tie"
        row = observer.row
    else:
        frozen = json.loads(FIXTURE.read_text())
        row = floats(frozen["case"]["input_bits"]).reshape(REPLAY_ROWS, 32)[0]
    x = row.repeat(REPLAY_ROWS, 1)
    gate_args = SimpleNamespace(dim=32, score_func="sqrtsoftplus", gate_temp=1.0,
                               norm_topk_prob=True, route_scale=1.5, vision_enabled=False,
                               get_moe_config=lambda _: (384, 6))
    gate = namespace["Gate"](LAYER, gate_args)
    with torch.no_grad():
        gate.weight.copy_(weight)
        gate.bias.copy_(bias)
        normalized, indices = gate(x)
        gate.norm_topk_prob, gate.route_scale = False, 1.0
        raw, raw_indices = gate(x)
    assert torch.equal(indices, raw_indices)
    expected_indices = step["routing"][str(LAYER)]["indices"][ROW * 6:(ROW + 1) * 6]
    expected_bits = step["routing"][str(LAYER)]["weight_bits"][ROW * 6:(ROW + 1) * 6]
    assert indices.tolist() == [expected_indices] * REPLAY_ROWS
    assert bits(normalized) == expected_bits * REPLAY_ROWS, "small replay changed the original gate"
    dots = F.linear(x, weight)
    scores = F.softplus(dots).sqrt()
    biased = scores + bias
    assert all(torch.isfinite(v).all() for v in (x, weight, bias, dots, scores, biased, raw, normalized))
    tie_bits = bits(torch.tensor([TIE_SCORE]))[0]
    assert bits(biased[0, TIED_EXPERTS]) == [tie_bits, tie_bits]
    assert bits(scores[0, TIED_EXPERTS])[0] != bits(scores[0, TIED_EXPERTS])[1]
    assert TIED_EXPERTS[1] in expected_indices and TIED_EXPERTS[0] not in expected_indices
    # The former lower-id tie policy must choose the wrong boundary expert.
    legacy = torch.argsort(biased, dim=-1, descending=True, stable=True)[:, :6]
    assert TIED_EXPERTS[0] in legacy[0] and TIED_EXPERTS[1] not in legacy[0]
    case = dict(name="4141112-long-prefill-2048-layer-27-row-1019", rows=REPLAY_ROWS,
                dim=32, experts=384, topk=6, input_bits=bits(x), weight_bits=bits(weight),
                bias_bits=bits(bias), dot_bits=bits(dots), indices=indices.flatten().tolist(),
                raw_bits=bits(raw), normalized_bits=bits(normalized))
    return dict(parent_fixture_sha256=PARENT_SHA256, revision=parent["revision"],
                source_sha256=parent["source_sha256"], torch_revision=torch.version.git_version,
                checkpoint_files=parent["checkpoint"]["files"], layer=LAYER, row=ROW,
                source_rows=step["seq"], tied_experts=TIED_EXPERTS, tied_score_bits=tie_bits,
                score_bits=bits(scores), biased_score_bits=bits(biased),
                legacy_indices=legacy.flatten().tolist(),
                counts=torch.bincount(indices.flatten(), minlength=384).tolist(), case=case)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-path", type=Path, required=True)
    p.add_argument("--kernel-reference-path", type=Path, required=True)
    p.add_argument("--capture-model-row", action="store_true")
    p.add_argument("--write-fixture", action="store_true")
    args = p.parse_args()
    if args.write_fixture and FIXTURE.exists():
        raise FileExistsError("gate-tie CPU fixture is frozen; repair native execution")
    if args.write_fixture and not args.capture_model_row:
        raise ValueError("initial freeze requires the independent model row")
    result = generate(args)
    if args.write_fixture:
        FIXTURE.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False))
    elif result != json.loads(FIXTURE.read_text()):
        raise AssertionError("gate-tie fixture differs from the independent CPU gate")
    print("PASS real boundary tie: expert 351 selected, 284 excluded; all routing bits exact", flush=True)


if __name__ == "__main__":
    main()
