"""Compare the native GPT-2-wide stage tape against ATen one operation at a time.

Every reference call consumes the *native* input of that operation, so a
difference localizes the kernel that produced the stage instead of accumulating
upstream drift. This is a diagnostic: it reads the frozen model and native stage
dump and never writes expectations or changes a gate.

Run with the oracle venv:

    .venv-torch/Scripts/python.exe validation/diagnose_gpt2_wide_stages.py \
        --model target/rdna3-port/cpu-shared-v2/gpt2_wide/unmasked/model \
        --fixture target/rdna3-port/cpu-shared-v2/gpt2_wide/unmasked/input.json \
        --stages target/rdna3-port/gpt2wide-stages.json \
        --output target/rdna3-port/gpt2-wide-stages-report.json
"""
from __future__ import annotations

import argparse
import json
import math
import hashlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--stages", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--standalone", action="store_true",
                        help="Read the standalone GPT2 logits harness tape and model")
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa",
                        help="Actual CPU oracle backend; the retained GPT-2-wide oracle uses SDPA")
    args = parser.parse_args()

    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    text = args.model if args.standalone else args.model / "text"
    config = json.loads((text / "config.json").read_text())
    weights = load_file(str(text / "model.safetensors"))
    fixture = json.loads(args.fixture.read_text())
    stages_document = json.loads(args.stages.read_text())
    tape = stages_document if args.standalone else stages_document["stages"]

    batch = fixture["batch_size"]
    seq = fixture["seq_len"]
    rows = batch * seq
    dim = config["n_embd"]
    heads = config["n_head"]
    head_dim = dim // heads
    eps = config["layer_norm_epsilon"]
    scale = 1.0 / math.sqrt(head_dim)

    ids = torch.tensor(fixture["input_ids"], dtype=torch.long).view(batch, seq)
    query_mask = torch.tensor(fixture["attention_mask"]).view(batch,seq).bool()

    def stage(name: str) -> torch.Tensor:
        return torch.tensor(tape[name], dtype=torch.float32)

    def weight(name: str) -> torch.Tensor:
        return weights[name].float()

    ops: list[dict] = []

    def record(
        label: str,
        native: torch.Tensor,
        reference: torch.Tensor,
        rows_mask: torch.Tensor | None = None,
    ) -> dict:
        if rows_mask is not None:
            native = native.reshape(rows, -1)[rows_mask]
            reference = reference.reshape(rows, -1)[rows_mask]
        native = native.flatten().float()
        reference = reference.flatten().float()
        if native.numel() != reference.numel():
            raise ValueError(
                f"{label}: {native.numel()} native values vs {reference.numel()} reference"
            )
        difference = (native - reference).abs()
        words = (native.view(torch.int32) != reference.view(torch.int32)).sum().item()
        ulp = (
            (
                native.view(torch.int32).to(torch.int64)
                - reference.view(torch.int32).to(torch.int64)
            )
            .abs()
            .max()
            .item()
        )
        entry = {
            "stage": label,
            "values": native.numel(),
            "max_abs": float(difference.max()) if native.numel() else 0.0,
            "max_ulp": int(ulp),
            "word_differences": int(words),
            "exact": bool(words == 0),
        }
        ops.append(entry)
        return entry

    # The composed `lookup` replaces image-token rows with projected vision
    # features, so validate the text rows only. A failure here means the tape,
    # weight layout or row order is wrong and later comparisons are meaningless.
    image_token = None if args.standalone else json.loads(
        (args.model / "image" / "config.json").read_text())["image_token_index"]
    text_rows = torch.ones(rows, dtype=torch.bool) if args.standalone else (ids != image_token).flatten()
    lookup_reference = weight("transformer.wte.weight")[ids].reshape(rows, dim)
    lookup_native = lookup_reference if args.standalone else stage("lookup")
    lookup = record("lookup/text_rows", lookup_native, lookup_reference, text_rows)
    if not lookup["exact"]:
        raise SystemExit(
            "embedding lookup does not match the tape; refusing to interpret later stages"
        )
    positions = torch.arange(seq).repeat(batch)
    position_reference = weight("transformer.wpe.weight")[positions].reshape(rows, dim)
    record(
        "input/lookup+position",
        stage("input"),
        lookup_native.reshape(rows, dim) + position_reference,
        text_rows,
    )

    def layer_norm(x: torch.Tensor, index: int, suffix: str):
        w = weight(f"transformer.h.{index}.{suffix}.weight")
        b = weight(f"transformer.h.{index}.{suffix}.bias")
        normalized, mean, rstd = torch.native_layer_norm(x, (dim,), w, b, eps)
        return normalized, mean, rstd

    def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        query = q.view(batch, seq, heads, head_dim).transpose(1, 2)
        key = k.view(batch, seq, heads, head_dim).transpose(1, 2)
        value = v.view(batch, seq, heads, head_dim).transpose(1, 2)
        allowed = torch.ones(seq,seq,dtype=torch.bool).tril()[None,None] & query_mask[:,None,None,:]
        if args.attention == 'sdpa':
            output = torch.nn.functional.scaled_dot_product_attention(query,key,value,attn_mask=allowed)
        else:
            scores = torch.matmul(query,key.transpose(-1,-2)) * scale
            probabilities = torch.softmax(scores.masked_fill(~allowed,torch.finfo(torch.float32).min),dim=-1)
            output = torch.matmul(probabilities,value)
        # Native masks query outputs; CPU padded queries have no objective.
        return output.transpose(1,2).reshape(rows,dim) * query_mask.reshape(rows,1)

    input_native = stage("input").reshape(rows, dim)
    for index in range(config["n_layer"]):
        prefix = f"layer_{index}"
        if args.standalone:
            tape[f"{prefix}/n1"] = tape[f"{prefix}/ln1"]
            tape[f"{prefix}/n3"] = tape[f"{prefix}/ln2"]
            tape[f"{prefix}/a"] = tape[f"{prefix}/attention"]
            tape[f"{prefix}/r"] = tape[f"{prefix}/residual"]
            tape[f"{prefix}/qkv"] = torch.cat([
                stage(f"{prefix}/{name}").reshape(rows,dim) for name in ('q','k','v')
            ],dim=1).flatten().tolist()
        if index:
            input_native = stage(f"layer_{index - 1}/out").reshape(rows, dim)
        n1_native = stage(f"{prefix}/n1").reshape(rows, dim)
        qkv_native = stage(f"{prefix}/qkv").reshape(rows, 3 * dim)

        n1, mean, rstd = layer_norm(input_native, index, "ln_1")
        if not args.standalone:
            record(f"{prefix}/n1/mean", stage(f"{prefix}/n1/mean"), mean)
            record(f"{prefix}/n1/rstd", stage(f"{prefix}/n1/rstd"), rstd)
        record(f"{prefix}/n1", n1_native, n1)

        qkv = torch.addmm(
            weight(f"transformer.h.{index}.attn.c_attn.bias"),
            n1_native,
            weight(f"transformer.h.{index}.attn.c_attn.weight"),
        )
        record(f"{prefix}/qkv", qkv_native, qkv)
        q_native, k_native, v_native = qkv_native.split([dim, dim, dim], dim=1)
        q, k, v = qkv.split([dim, dim, dim], dim=1)
        record(f"{prefix}/q", q_native, q)
        record(f"{prefix}/k", k_native, k)
        record(f"{prefix}/v", v_native, v)

        attended = attention(q_native, k_native, v_native)
        a_native = stage(f"{prefix}/a").reshape(rows, dim)
        record(f"{prefix}/a", a_native, attended)

        projected = torch.addmm(
            weight(f"transformer.h.{index}.attn.c_proj.bias"),
            a_native,
            weight(f"transformer.h.{index}.attn.c_proj.weight"),
        )
        o_native = stage(f"{prefix}/o").reshape(rows, dim)
        record(f"{prefix}/o", o_native, projected)

        residual_native = stage(f"{prefix}/r").reshape(rows, dim)
        record(f"{prefix}/r", residual_native, input_native + o_native)

        n3_native = stage(f"{prefix}/n3").reshape(rows, dim)
        n3, mean, rstd = layer_norm(residual_native, index, "ln_2")
        if not args.standalone:
            record(f"{prefix}/n3/mean", stage(f"{prefix}/n3/mean"), mean)
            record(f"{prefix}/n3/rstd", stage(f"{prefix}/n3/rstd"), rstd)
        record(f"{prefix}/n3", n3_native, n3)

        up = torch.addmm(
            weight(f"transformer.h.{index}.mlp.c_fc.bias"),
            n3_native,
            weight(f"transformer.h.{index}.mlp.c_fc.weight"),
        )
        u_native = stage(f"{prefix}/u").reshape(rows, dim * 4)
        record(f"{prefix}/u", u_native, up)

        activation = 0.5 * u_native * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (u_native + 0.044715 * torch.pow(u_native, 3.0))
            )
        )
        activation_native = stage(f"{prefix}/activation").reshape(rows, dim * 4)
        record(f"{prefix}/activation", activation_native, activation)

        down = torch.addmm(
            weight(f"transformer.h.{index}.mlp.c_proj.bias"),
            activation_native,
            weight(f"transformer.h.{index}.mlp.c_proj.weight"),
        )
        d_native = stage(f"{prefix}/d").reshape(rows, dim)
        record(f"{prefix}/d", d_native, down)

        out_native = stage(f"{prefix}/out").reshape(rows, dim)
        record(f"{prefix}/out", out_native, residual_native + d_native)

    final_native = stage("final_norm").reshape(rows, dim)
    final_norm, _, _ = torch.native_layer_norm(
        out_native,
        (dim,),
        weight("transformer.ln_f.weight"),
        weight("transformer.ln_f.bias"),
        eps,
    )
    record("final_norm", final_native, final_norm)
    logits = torch.nn.functional.linear(final_native, weight(
        "lm_head.weight" if "lm_head.weight" in weights else "transformer.wte.weight"))
    record(
        "logits",
        torch.tensor(stages_document["logits"], dtype=torch.float32),
        logits,
    )

    diverged = [entry for entry in ops if not entry["exact"]]
    first = diverged[0] if diverged else None
    report = {
        "schema": 1,
        "scope": "GPT-2 operation-local ATen oracle, each operation uses native inputs",
        "attention_implementation": args.attention,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "stages_sha256": hashlib.sha256(args.stages.read_bytes()).hexdigest(),
        "model_sha256": {p.relative_to(args.model).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.model.rglob('*') if p.is_file()},
        "harness": {
            "torch": torch.__version__,
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "threads": torch.get_num_threads(),
            "batch": batch,
            "seq_len": seq,
            "hidden": dim,
            "heads": heads,
            "layers": config["n_layer"],
            "image_token": image_token,
        },
        "ops": ops,
        "first_divergence": first,
        "divergent_operations": len(diverged),
        "all_exact": not diverged,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"torch {torch.__version__} ({torch.backends.cpu.get_cpu_capability()}), threads 1")
    print(f"{'stage':<26} {'values':>6} {'max_abs':>12} {'ulp':>4} {'words':>6}")
    for entry in ops:
        print(
            f"{entry['stage']:<26} {entry['values']:>6} {entry['max_abs']:>12.6g} "
            f"{entry['max_ulp']:>4} {entry['word_differences']:>6}"
        )
    if first:
        print(
            f"\nFIRST DIVERGENCE: {first['stage']} max_abs={first['max_abs']:.6g} "
            f"ulp={first['max_ulp']} words={first['word_differences']}"
        )
    else:
        print("\nall stages bit-exact")


if __name__ == "__main__":
    main()
