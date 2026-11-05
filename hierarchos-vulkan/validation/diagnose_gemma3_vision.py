"""Locate the first SigLIP boundary where native arithmetic leaves the oracle.

Runs the unmodified upstream Gemma3 package with forward hooks, runs the native
probe's stage dump on the same pixels, and prints the maximum absolute
difference per boundary. Preparation only; not an acceptance gate.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from qualify_multimodal_training import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkout = Path(os.environ.get("HIERARCHOS_TRANSFORMERS_CHECKOUT", str(Path(__file__).resolve().parents[3] / "transformers")))
    sys.path.insert(0, str(checkout / "src"))
    import torch

    folder = args.fixtures.resolve() / args.case
    fixture = json.loads((folder / "input.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    stages_path = args.output / f"{args.case}-native.json"
    subprocess.run(
        [str(args.binary.resolve()), "--model", str(folder / "model"),
         "--fixture", str(folder / "input.json"), "--stages", str(stages_path),
         "--output", str(args.output / "unused")],
        check=True,
    )
    native = json.loads(stages_path.read_text())
    frozen = json.loads((folder / "cpu.json").read_text())
    from transformers import Gemma3ForConditionalGeneration
    torch.set_num_threads(1)
    model = Gemma3ForConditionalGeneration.from_pretrained(folder / "model").float().train()
    # The fixture oracle is built with pinned eager attention; from_pretrained
    # would otherwise restore the package default (SDPA) and hand this
    # diagnostic a different arithmetic path than the frozen expectations.
    model.set_attn_implementation("eager")
    batch, seq = fixture["batch_size"], fixture["seq_len"]
    ids = torch.tensor(fixture["input_ids"], dtype=torch.long).view(batch, seq)
    mask = torch.tensor(fixture["attention_mask"], dtype=torch.long).view(batch, seq)
    side = model.config.vision_config.image_size
    pixels = torch.tensor(frozen["pixels"], dtype=torch.float32).view(fixture["image_count"], 3, side, side)
    if native["pixels"] != frozen["pixels"]:
        raise ValueError("native normalization changed the diagnostic input")
    seen = {}

    def keep(name):
        def hook(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            seen[name] = tensor.detach().flatten().tolist()
        return hook

    def keep_input(name):
        def hook(module, inputs):
            seen[name] = inputs[0].detach().flatten().tolist()
        return hook

    def keep_pool(module, inputs, output):
        # Native pooled rows use [image, token, hidden], after HF's NCHW pool.
        seen["pooled"] = output.flatten(2).transpose(1, 2).detach().flatten().tolist()

    tower = model.model.vision_tower
    handles = [tower.embeddings.register_forward_hook(keep("tokens"))]
    for index, layer in enumerate(tower.encoder.layers):
        handles.append(layer.register_forward_hook(keep(f"layer_{index}")))
    first = tower.encoder.layers[0]
    handles.append(first.layer_norm1.register_forward_hook(keep("layer_0/ln1")))
    handles.append(first.self_attn.q_proj.register_forward_hook(keep("layer_0/q")))
    handles.append(first.self_attn.k_proj.register_forward_hook(keep("layer_0/k")))
    handles.append(first.self_attn.v_proj.register_forward_hook(keep("layer_0/v")))
    handles.append(first.self_attn.out_proj.register_forward_pre_hook(keep_input("layer_0/attention")))
    handles.append(first.self_attn.out_proj.register_forward_hook(keep("layer_0/out_proj")))
    handles.append(first.layer_norm2.register_forward_hook(keep("layer_0/ln2")))
    handles.append(first.mlp.fc1.register_forward_hook(keep("layer_0/fc1")))
    handles.append(first.mlp.activation_fn.register_forward_hook(keep("layer_0/gte")))
    handles.append(tower.post_layernorm.register_forward_hook(keep("post_norm")))
    projector = model.model.multi_modal_projector
    handles.append(projector.avg_pool.register_forward_hook(keep_pool))
    handles.append(projector.mm_soft_emb_norm.register_forward_hook(keep("pooled_norm")))
    handles.append(projector.register_forward_hook(keep("projected")))
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask, pixel_values=pixels, use_cache=False).logits.flatten().tolist()
    for handle in handles:
        handle.remove()
    reference = frozen["steps"][0]["forward"]["logits"]
    if logits != reference:
        raise ValueError(f"unmodified CPU diagnostic differs from frozen oracle: {max(abs(a-b) for a,b in zip(logits,reference,strict=True))}")
    json.dump({"stages":seen,"logits":logits},
        (args.output / f"{args.case}-cpu.json").open("w"), allow_nan=False)

    rows = []
    native_stages = native["stages"]
    if {"layer_0/ln1_mean", "layer_0/ln2_mean"} <= set(native_stages):
        vision = model.config.vision_config
        eps = vision.layer_norm_eps
        dim = vision.hidden_size
        tokens = torch.tensor(native_stages["tokens"], dtype=torch.float32).view(1, -1, dim)
        first_layer = tower.encoder.layers[0]
        residual = tokens + torch.tensor(
            native_stages["layer_0/out_proj"], dtype=torch.float32).view(1, -1, dim)
        moments = []
        for prefix, source, module in [("ln1", tokens, first_layer.layer_norm1),
                                       ("ln2", residual, first_layer.layer_norm2)]:
            _, mean, rstd = torch.native_layer_norm(source, [dim], module.weight, module.bias, eps)
            moments.append((f"layer_0/{prefix}_mean", mean.flatten().tolist()))
            moments.append((f"layer_0/{prefix}_rstd", rstd.flatten().tolist()))
        for name, expected in moments:
            actual = native_stages.get(name)
            worst = max(abs(a - b) for a, b in zip(actual, expected))
            rows.append({"stage": name, "values": len(expected), "max_abs": worst,
                         "pass": worst < 2e-7})
    for name, expected in seen.items():
        actual = native["stages"].get(name)
        if actual is None or len(actual) != len(expected):
            rows.append({"stage": name, "status": "missing or reshaped"})
            continue
        worst = max(abs(a - b) for a, b in zip(actual, expected))
        rows.append({"stage": name, "values": len(expected), "max_abs": worst,
                     "pass": worst < 2e-7})
    json.dump({"case": args.case, "stages": rows, "cpu_replay_max_abs":0.0,
        "cpu_sha256":digest(folder/"cpu.json"), "binary_sha256":digest(args.binary),
        "diagnostic_sha256":digest(__file__)},
        (args.output / f"{args.case}-diff.json").open("w"), indent=1)
    for row in rows:
        if "max_abs" in row:
            print(f"{'ok  ' if row['pass'] else 'DRIFT'} {row['stage']:12s} {row['max_abs']:.3e} ({row['values']})")
        else:
            print(f"skip {row['stage']}: {row['status']}")


if __name__ == "__main__":
    main()
