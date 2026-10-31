"""Production-width PEFT-LoRA CPU PyTorch oracle for the native V4.1 graph.

This is the adapter twin of ``validate_deepseek_v41_production_model.py``. One
released-width block (5120 hidden, 64x512 heads, Q/O ranks 1280/1024, eight
output groups, 2304-wide FP4 experts) carries an additive HF PEFT-LoRA adapter
on packed FP8 attention, the packed shared expert and one routed FP4 expert.
The vocabulary is reduced to 512 so the package fits a few hundred MB; the
vocabulary/head contract is already gated by the 24-layer production case, and
an adapter never touches it.

Reference arithmetic is PEFT's wrapper:

    y = base(x) + (alpha / r) * B(A(x))

with ``base`` returning the graph's BF16 projection result, ``A``/``B`` and the
scaled sum in FP32, and one rounding back to BF16. The native kernel implements
exactly that boundary, so acceptance stays the unchanged ``<2e-7``, ``rtol=0``
gate. Base (unadapted) logits are stored beside the adapted ones so the native
test proves the delta was applied, not just that the graph ran.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import safetensors
import safetensors.torch
import torch
import torch.nn.functional as F
from torch import nn

import validate_deepseek_v41_attention_model as attention_model
import validate_deepseek_v41_mixed as mixed
import validate_deepseek_v41_model as model
import validate_deepseek_v41_production_model as production

FIXTURE = model.FIXTURE.parent / "deepseek_v41_lora_model.json"
SEED = 4141095
ADAPTER_SEED = 4141096
RANK = 16
ALPHA = 16.0
# The adapter gate uses two single-token steps: adapter arithmetic is
# row-independent and rides the same qualified projection kernels. The
# multi-row grouped body is gated separately by `PREFILL_STEPS` below on the
# unadapted graph, so a shape-dependent projection cannot hide here.
STEPS = [
    dict(name="lora-prefill", pos=0, seq=1, ids=[3]),
    dict(name="lora-decode", pos=1, seq=1, ids=[17]),
]
# A four-token prompt at released width is the first multi-row gate. The CPU
# bmm picks a different K-block body for one row than for several, so this case
# exists to pin the multi-row grouping instead of relying on the single-token
# steps above.
PREFILL_STEPS = [
    dict(name="prefill-four", pos=0, seq=4, ids=[3, 17, 41, 59]),
    dict(name="prefill-four-decode", pos=4, seq=1, ids=[23]),
]
ARGS = dict(
    production.ARGS,
    n_layers=1,
    vocab_size=512,
    max_seq_len=8,
    compress_ratios=(0,),
    kv_source_layers=(),
    index_source_layers=(),
    engram_layer_ids=(),
    engram_num_embeddings=(),
)
# (checkpoint module path, HF adapter spelling, owner path, attribute)
TARGETS = [
    ("layers.0.attn.wq_a", "layers.0.self_attn.wq_a", "attn", "wq_a", 5120, 1280),
    ("layers.0.attn.wq_b", "layers.0.self_attn.wq_b", "attn", "wq_b", 1280, 32768),
    ("layers.0.attn.wkv", "layers.0.self_attn.wkv", "attn", "wkv", 5120, 512),
    ("layers.0.attn.wo_b", "layers.0.self_attn.wo_b", "attn", "wo_b", 8192, 5120),
    ("layers.0.ffn.shared_experts.w1", "layers.0.ffn.shared_experts.w1", "shared", "w1", 5120, 2304),
    ("layers.0.ffn.experts.0.w2", "layers.0.ffn.experts.0.w2", "expert0", "w2", 2304, 5120),
]


class Lora(nn.Module):
    """PEFT's inference wrapper over one packed projection."""

    def __init__(self, base, a, b, scale):
        super().__init__()
        self.base = base
        self.register_buffer("lora_a", a.float())
        self.register_buffer("lora_b", b.float())
        self.scale = float(scale)

    def forward(self, x):
        y = self.base(x)
        delta = F.linear(F.linear(x.float(), self.lora_a), self.lora_b) * self.scale
        return (y.float() + delta).to(y.dtype)


def native_config():
    value = model.config(2)
    value["text_config"].update({
        "vocab_size": 512, "hidden_size": 5120, "moe_intermediate_size": 2304,
        "num_hidden_layers": 1, "num_attention_heads": 64, "num_key_value_heads": 1,
        "head_dim": 512, "qk_rope_head_dim": 64, "q_lora_rank": 1280,
        "o_lora_rank": 1024, "o_groups": 8, "sliding_window": 128,
        "max_position_embeddings": 8, "rms_norm_eps": 1e-20,
        "compress_ratios": [0], "kv_source_layer_ids": [], "index_source_layer_ids": [],
        "index_n_heads": 32, "index_head_dim": 128, "index_topk": 512,
        "candidate_source_layer_id": -1, "candidate_topk_blocks": 0,
        "candidate_block_size": 0, "n_routed_experts": 2, "n_shared_experts": 1,
        "num_experts_per_tok": 2, "routed_scaling_factor": 1.5, "swiglu_limit": 10.0,
        "scoring_func": "sqrtsoftplus", "norm_topk_prob": True, "hc_mult": 4,
        "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "engram_layer_ids": [],
        "engram_num_embeddings": [], "engram_vocab_size": 0, "engram_n_heads": 8,
        "engram_head_dim": 256, "dspark_block_size": 0, "dspark_target_layer_ids": [],
        "rope_scaling": dict(rope_type="yarn", factor=16, beta_fast=32, beta_slow=1,
                             original_max_position_embeddings=65536),
    })
    return value


def check_shapes(net):
    layer = net.layers[0]
    assert layer.attn.wq_a.spec["rows"] == 1280
    assert layer.attn.wq_b.spec["rows"] == 64 * 512
    assert layer.ffn.experts[0].w2.spec["rows"] == 5120


def attach_adapters(net, rng):
    layer = net.layers[0]
    owners = {
        "attn": layer.attn,
        "shared": layer.ffn.shared_experts,
        "expert0": layer.ffn.experts[0],
    }
    adapter = {}
    for target, spelling, owner, attr, in_dim, out_dim in TARGETS:
        base = getattr(owners[owner], attr)
        a = (torch.randn((RANK, in_dim), generator=rng) * 0.02).float()
        b = (torch.randn((out_dim, RANK), generator=rng) * 0.02).float()
        setattr(owners[owner], attr, Lora(base, a, b, ALPHA / RANK))
        adapter[spelling] = (a, b)
    return adapter


def write_adapter(adapter, root):
    root.mkdir(parents=True, exist_ok=True)
    tensors = {}
    targets = []
    for spelling, (a, b) in adapter.items():
        tensors[f"base_model.model.{spelling}.lora_A.weight"] = a.contiguous()
        tensors[f"base_model.model.{spelling}.lora_B.weight"] = b.contiguous()
        targets.append(spelling)
    safetensors.torch.save_file(tensors, root / "adapter_model.safetensors")
    (root / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": RANK,
        "lora_alpha": ALPHA, "lora_dropout": 0.0, "bias": "none",
        "fan_in_fan_out": False, "inference_mode": True,
        "target_modules": ["wq_a", "wq_b", "wkv", "wo_b", "w1", "w2"],
    }, separators=(",", ":")))


def entries(net, packed):
    """Checkpoint tensors in write order; adapter factors are not base weights."""
    out = []
    for name, tensor in net.state_dict().items():
        if name.endswith(("lora_a", "lora_b")):
            continue
        if tensor.dtype == torch.bfloat16:
            out.append((name, "bfloat16", list(tensor.shape),
                        tensor.contiguous().view(torch.uint16).clone()))
        else:
            out.append((name, "float32", list(tensor.shape), tensor.detach().contiguous().clone()))
    for name, spec in packed.items():
        fp4 = spec["dtype"] == "fp4_row32"
        rows, cols = spec["rows"], spec["cols"]
        out.append((name, "float4_e2m1fn_x2" if fp4 else "float8_e4m3fn",
                    [rows, cols // 2 if fp4 else cols], spec["data"].clone()))
        scale_rows = rows if fp4 else (rows + 31) // 32
        out.append((name.removesuffix("weight") + "scale", "float8_e8m0fnu",
                    [scale_rows, cols // 32], spec["scales"].view(scale_rows, cols // 32).clone()))
    return out


def write_package(net, packed, root):
    root.mkdir(parents=True, exist_ok=True)
    # The head is stored BF16 exactly as released; its FP32 promotion during
    # the forward sees the same values.
    net.head.weight.data = net.head.weight.detach().bfloat16()
    # `rows` must outlive `serialize`: TensorSpec holds raw data pointers into
    # these clones, and letting the list fall out of scope first is a segfault.
    rows = entries(net, packed)
    specs, weight_map = {}, {}
    for name, dtype, shape, tensor in rows:
        specs[name] = safetensors.TensorSpec(
            dtype=dtype, shape=shape, data_ptr=tensor.data_ptr(),
            data_len=tensor.numel() * tensor.element_size())
        weight_map[name] = "model.safetensors"
    data = safetensors.serialize(specs)
    (root / "model.safetensors").write_bytes(data)
    (root / "config.json").write_text(json.dumps(native_config(), separators=(",", ":")))
    (root / "model.safetensors.index.json").write_text(
        json.dumps(dict(weight_map=weight_map), separators=(",", ":")))
    return {"model.safetensors": hashlib.sha256(data).hexdigest()}


def forward(net, steps):
    head_inputs = []
    net.head.register_forward_pre_hook(lambda m, i: head_inputs.append(i[0].clone()))
    torch.set_num_threads(1)
    outputs = []
    with torch.inference_mode():
        for spec in steps:
            ids = torch.tensor(spec["ids"]).view(1, spec["seq"])
            net(ids, spec["pos"])
            logits = net.head(head_inputs[-1], full_logits=True)
            outputs.append(spec | {"logits": logits.flatten().tolist(),
                                   "shape": list(logits.shape)})
    return outputs


def make_case(reference_path, kernel_path):
    namespace = model.load_reference(reference_path, kernel_path, hash_config={}, tokens=None)
    expert_ns = mixed.source_namespace(reference_path)
    torch.set_default_dtype(torch.bfloat16)
    args = namespace["ModelArgs"](**ARGS)
    net = namespace["Transformer"](args)
    torch.set_default_dtype(torch.float32)
    namespace["sparse_attn"] = attention_model.sparse_attn
    rng = torch.Generator().manual_seed(SEED)
    packed = {}
    production.fill_parameters(list(net.named_parameters()), rng)
    for layer in net.layers:
        production.transform_layer(layer, packed, expert_ns)
        production.round_stream_dtype(layer)
    # The final norm is outside the blocks; the production case rounds every
    # non-HC parameter to the BF16 stream dtype after construction.
    net.norm.weight.data = net.norm.weight.detach().bfloat16()
    check_shapes(net)
    base_steps = forward(net, STEPS)
    # Start the multi-token case from an empty ring. Only slots below `seq` are
    # read during the prefill, but the decode after it walks the seeded ring.
    for layer in net.layers:
        layer.attn.window_kv_cache.zero_()
    prefill_steps = forward(net, PREFILL_STEPS)

    adapter = attach_adapters(net, torch.Generator().manual_seed(ADAPTER_SEED))
    adapted_steps = forward(net, STEPS)
    folder = f"deepseek_v41_lora_{SEED}"
    root = FIXTURE.parent / folder
    files = write_package(net, packed, root)
    write_adapter(adapter, root / "adapter")
    adapter_files = {
        name: hashlib.sha256((root / "adapter" / name).read_bytes()).hexdigest()
        for name in ("adapter_config.json", "adapter_model.safetensors")
    }
    worst_delta = max(
        abs(a - b)
        for step, base in zip(adapted_steps, base_steps)
        for a, b in zip(step["logits"], base["logits"])
    )
    assert worst_delta > 1e-4, "the adapter must change the logits measurably"
    return dict(
        seed=SEED, revision=model.backbone.REVISION,
        source_sha256=model.backbone.SOURCE_SHA256,
        precision="production_width_peft_lora_bf16_residual_hc_cpu_pytorch",
        stream_precision="bf16", config=native_config(), batch_size=1,
        max_prefill=1, max_positions=8,
        checkpoint=dict(directory=folder, files=files,
                        tensor_count=len(net.state_dict()), packed_count=len(packed)),
        adapter=dict(directory="adapter", files=adapter_files, rank=RANK, alpha=ALPHA,
                     targets=[target for target, *_ in TARGETS],
                     worst_logit_delta=worst_delta),
        steps=adapted_steps, base_steps=base_steps, prefill_steps=prefill_steps,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--kernel-reference-path", type=Path, required=True)
    parser.add_argument("--write-fixture", action="store_true")
    parser.add_argument("--native-output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    fixture = make_case(args.reference_path, args.kernel_reference_path)
    if args.write_fixture:
        FIXTURE.write_text(json.dumps(fixture, separators=(",", ":"), allow_nan=False))
    elif json.loads(FIXTURE.read_text()) != fixture:
        raise AssertionError("PEFT-LoRA fixture differs from the pinned CPU PyTorch reference")
    print("PASS production-width PEFT-LoRA CPU oracle:",
          sum(len(step["logits"]) for step in fixture["steps"]), "logits;",
          sum(len(step["logits"]) for step in fixture["prefill_steps"]),
          "four-token logits;",
          f"adapter delta {fixture['adapter']['worst_logit_delta']:.6g}")
    if args.native_output:
        native = json.loads(args.native_output.read_text())
        binding = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        if native.get("fixture_sha256") != binding:
            raise AssertionError("native output does not use the identical pinned fixture")
        # Either native regression writes its own group; a file that names a
        # group must match it step for step.
        groups = [(key, fixture[key], native[key])
                  for key in ("steps", "prefill_steps") if key in native]
        if not groups:
            raise AssertionError("native output carries no logit group")
        for _, expected_steps, actual_steps in groups:
            if len(expected_steps) != len(actual_steps):
                raise AssertionError("native step count mismatch")
            for expected, actual in zip(expected_steps, actual_steps):
                if expected["name"] != actual["name"] or expected["shape"] != actual["shape"]:
                    raise AssertionError("native step identity/shape mismatch")
                values = torch.tensor(actual["logits"], dtype=torch.float32).reshape(actual["shape"])
                reference = torch.tensor(expected["logits"], dtype=torch.float32).reshape(expected["shape"])
                drift = (values.double() - reference.double()).abs().max().item()
                print(f"PASS {expected['name']}: {drift:.17g}")
                if drift >= 2e-7:
                    raise AssertionError(f"{expected['name']} exceeds the strict <2e-7 gate")


if __name__ == "__main__":
    main()
