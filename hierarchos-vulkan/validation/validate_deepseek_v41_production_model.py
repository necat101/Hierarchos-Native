"""Production-width, real-vocabulary V4.1 CPU PyTorch oracle.

The released text config is 5120 hidden, 64x512 attention heads, Q/O ranks
1280/1024, 8 output groups, 2304-wide experts, 32x128 index heads, HC 4 and a
129280-token vocabulary. A literal 40-layer/384-expert/1M-context run needs the
~600B-parameter checkpoint, so this case fixes the released *widths* and
vocabulary in twenty-four layers with two routed experts, two single-token
steps, and no compression/index schedule. Twenty-four layers keep this a
production-width case that exercises repeated inter-layer accumulation rather
than a single residual hand-off, and reach past every place where the released
schedule changes shape (ratio 2 at layer 2, ratio 1 at layer 20, KV/index
sources at 2/8/14/20) without enabling compression itself. Everything else
follows the already-qualified BF16 stream contract.

Weights are written straight to sharded SafeTensors. They are never expanded
into JSON: the vocabulary pair alone is 1.3B values. Twenty-four
production-width layers, the embedding and the FP32 head cannot be resident on
a consumer host, so the embedding and head are allocated on demand in
memory-mapped scratch files, and the package is written in two passes that
release the blocks first. The values, RNG order and forward arithmetic are
untouched by that staging.
"""
import argparse
import atexit
import gc
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import safetensors
import torch

import validate_deepseek_v41_attention_model as attention_model
import validate_deepseek_v41_bf16_model as bf16
import validate_deepseek_v41_mixed as mixed
import validate_deepseek_v41_model as model

FIXTURE = model.FIXTURE.parent / "deepseek_v41_production_model.json"
SEED = 4141090
TOKEN_IDS = [129279, 1]
STEPS = [
    dict(name="production-prefill", pos=0, seq=1, ids=TOKEN_IDS[:1]),
    dict(name="production-decode", pos=1, seq=1, ids=TOKEN_IDS[1:]),
]
ARGS = dict(
    max_batch_size=1, max_seq_len=8, temperature=0, dtype="bf16", expert_dtype=None,
    vocab_size=129280, dim=5120, moe_inter_dim=2304, n_layers=24, n_mtp_layers=0,
    n_heads=64, n_routed_experts=2, n_activated_experts=2, route_scale=1.5,
    swiglu_limit=10.0, q_lora_rank=1280, head_dim=512, rope_head_dim=64, o_groups=8,
    o_lora_rank=1024, window_size=128, compress_ratios=(0,) * 24, kv_source_layers=(),
    index_source_layers=(), compress_rope_theta=160000, original_seq_len=65536,
    index_n_heads=32, index_head_dim=128, index_topk=512, candidate_source_layer=-1,
    candidate_topk_blocks=0, candidate_block_size=0, hc_mult=4,
    engram_layer_ids=(), engram_num_embeddings=(), engram_max_ngram_size=1,
    engram_vocab_size=1, engram_n_heads=1, engram_head_dim=1,
    engram_compressed_vocab_size=1,
)


def native_config():
    value = model.config(2)
    value["text_config"].update({
        "vocab_size": 129280, "hidden_size": 5120, "moe_intermediate_size": 2304,
        "num_hidden_layers": 24, "num_attention_heads": 64, "num_key_value_heads": 1,
        "head_dim": 512, "qk_rope_head_dim": 64, "q_lora_rank": 1280,
        "o_lora_rank": 1024, "o_groups": 8, "sliding_window": 128,
        "max_position_embeddings": 8, "rms_norm_eps": 1e-20,
        "compress_ratios": [0] * 24, "kv_source_layer_ids": [], "index_source_layer_ids": [],
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


_SCRATCH = []
_PACKED_FILES = []
_PACKED_COUNT = [0]
_SWEPT = False


def page_backed(shape, dtype, folder=None):
    """Allocate one large tensor in a memory-mapped scratch file.

    The BF16 embedding, the FP32 head and the packed block weights are the only
    multi-gigabyte residents of this oracle; at twenty-four layers they add up
    to roughly 9 GiB, which does not fit this host anonymously. The same bytes
    live in a file mapping instead: the OS can evict pages the forward is not
    touching, and the values, dtype, shape, RNG order and shard hashes are
    unchanged.
    """
    _ensure_swept()
    if folder is None:
        folder = Path(tempfile.mkdtemp(prefix="deepseek-v41-vocabulary-"))
        _SCRATCH.append(folder)
        path = folder / "tensor.bin"
    else:
        _PACKED_COUNT[0] += 1
        path = folder / f"tensor-{_PACKED_COUNT[0]:05d}.bin"
    count = 1
    for size in shape:
        count *= size
    itemsize = torch.empty((), dtype=dtype).element_size()
    with path.open("wb") as handle:
        handle.truncate(count * itemsize)
    mapped = np.memmap(path, dtype=np.uint8, mode="r+", shape=(count * itemsize,))
    return torch.frombuffer(mapped, dtype=dtype).view(*shape)


def packed_folder():
    """One scratch directory for every page-backed packed matrix."""
    _ensure_swept()
    if not _PACKED_FILES:
        folder = Path(tempfile.mkdtemp(prefix="deepseek-v41-packed-"))
        _SCRATCH.append(folder)
        _PACKED_FILES.append(folder)
    return _PACKED_FILES[0]


def _purge(folder):
    """Best-effort removal of one scratch folder; True once it is gone.

    Windows keeps a memory-mapped file's name locked until the last view is
    dropped, so a single attempt can fail while tensors are still alive; collect
    and retry rather than treating that as a permanent failure.
    """
    for _ in range(3):
        try:
            shutil.rmtree(folder)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            gc.collect()
    return False


def sweep_stale_scratch():
    """Reclaim scratch left behind by a run that never reached release_scratch().

    A killed process cannot run cleanup, and neither can one whose mapping is
    still alive when it exits. Reclaiming whatever is left before the next run
    allocates anything keeps the temp root bounded however the previous run
    ended. A folder still mapped by a live sibling process stays: its deletion
    fails and it is skipped.
    """
    swept = 0
    for prefix in ("deepseek-v41-packed-", "deepseek-v41-vocabulary-"):
        for folder in Path(tempfile.gettempdir()).glob(f"{prefix}*"):
            if folder.is_dir() and _purge(folder):
                swept += 1
    return swept


def _ensure_swept():
    global _SWEPT
    if not _SWEPT:
        _SWEPT = True
        sweep_stale_scratch()


def release_scratch():
    """Delete the scratch files once every view of them is dropped."""
    folders, _SCRATCH[:] = list(_SCRATCH), []
    _PACKED_FILES.clear()
    _PACKED_COUNT[0] = 0
    gc.collect()
    leftover = [folder for folder in folders if not _purge(folder)]
    if leftover:
        print(
            f"warning: {len(leftover)} page-backed scratch folder(s) are still "
            f"mapped, e.g. {leftover[0]}; the next run sweeps them",
            file=sys.stderr,
        )


# An interrupted or raising run still releases its own scratch.
atexit.register(release_scratch)


def defer_large_allocations(namespace):
    """Let the blocks exist before the embedding and FP32 head are allocated.

    `Transformer.__init__` allocates 1.23 GiB of embedding, the blocks, then a
    2.47 GiB FP32 head, so twenty-four production-width layers need 20.9 GiB
    before a single weight is filled. These shims defer the two vocabulary tensors
    until the caller asks for them and back them with a file mapping; shapes,
    names, dtypes and draw order are unchanged.
    """
    original_embed = namespace["ParallelEmbedding"]
    original_head = namespace["ParallelHead"]
    world_size = namespace.get("world_size", 1)
    rank = namespace.get("rank", 0)

    class DeferredEmbedding(original_embed):
        def __init__(self, vocab_size, dim):
            torch.nn.Module.__init__(self)
            self.vocab_size = vocab_size
            self.dim = dim
            self.part_vocab_size = vocab_size // world_size
            self.vocab_start_idx = rank * self.part_vocab_size
            self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
            self.weight = torch.nn.Parameter(torch.empty(1, dim))

        def materialize(self):
            # The reference allocates during `Transformer.__init__`, while the
            # BF16 default dtype is active; this case stores the same BF16
            # embedding, so name the dtype instead of inheriting the caller's.
            self.weight = torch.nn.Parameter(
                page_backed((self.part_vocab_size, self.dim), torch.bfloat16))

    class DeferredHead(original_head):
        def __init__(self, vocab_size, dim, norm_eps=1e-6, hc_eps=1e-6):
            torch.nn.Module.__init__(self)
            self.vocab_size = vocab_size
            self.dim = dim
            self.norm_eps = norm_eps
            self.hc_eps = hc_eps
            self.part_vocab_size = vocab_size // world_size
            # The release stores BF16; the reference promotes it for the logits.
            self.weight = torch.nn.Parameter(
                torch.empty(1, dim, dtype=torch.float32))

        def materialize(self):
            self.weight = torch.nn.Parameter(
                page_backed((self.part_vocab_size, self.dim), torch.float32))

    namespace["ParallelEmbedding"] = DeferredEmbedding
    namespace["ParallelHead"] = DeferredHead


def fill_parameters(named, rng):
    """Deterministic init with the released rounding boundaries, chunked so the
    129280x5120 tensors never need a second full-size temporary."""
    with torch.no_grad():
        for name, parameter in named:
            if "hc_" in name and parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
            if name == "embed.weight":
                for start in range(0, parameter.shape[0], 8192):
                    end = min(start + 8192, parameter.shape[0])
                    chunk = torch.randn((end - start, parameter.shape[1]),
                                        generator=rng) * 0.3
                    parameter.data[start:end] = chunk.to(torch.bfloat16)
                continue
            if name == "head.weight":
                for start in range(0, parameter.shape[0], 8192):
                    end = min(start + 8192, parameter.shape[0])
                    chunk = torch.randn((end - start, parameter.shape[1]),
                                        generator=rng) * 0.08
                    parameter.data[start:end] = chunk.bfloat16().float()
                continue
            value = torch.randn(parameter.shape, generator=rng) * 0.05
            if "norm.weight" in name or name.endswith(("q_weight", "k_weight")):
                value = 1 + torch.randn(parameter.shape, generator=rng) * 0.02
            elif name.endswith(("hc_attn_scale", "hc_ffn_scale")):
                value = torch.tensor([0.8, 1.1, 0.9])
            elif "hc_" in name:
                value *= 0.4
            elif name.endswith("ffn.gate.bias"):
                value = torch.tensor([-0.15, 0.0])
            parameter.copy_(value.to(parameter.dtype))


def round_stream_dtype(layer):
    """Round one block's remaining parameters to the BF16 stream dtype.

    This is the same conversion the validator applies to every non-HC parameter
    after the forward; doing it per block only moves the moment, so a
    twenty-four-layer package never keeps all blocks in FP32 at once.
    """
    for name, parameter in layer.named_parameters():
        if "hc_" in name:
            continue
        if parameter.dtype != torch.bfloat16:
            parameter.data = parameter.data.detach().bfloat16()


def fill_and_quantize(net, packed, expert_ns, rng, build_layer, layer_count):
    """Fill each block and immediately replace its BF16 weights with the packed
    specs, in `Transformer.named_parameters()` order (embed, blocks, norm, head).

    The draws are identical to one `fill_parameters(net.named_parameters())`
    pass; only the lifetime changes. Production-width FP32 blocks are 0.87 GiB
    each, so they are built, filled, packed and rounded one at a time. The FP32
    head is materialized last, after the blocks it would otherwise overlap have
    already shrunk.
    """
    net.embed.materialize()
    fill_parameters([(f"embed.{name}", parameter)
                     for name, parameter in net.embed.named_parameters()], rng)
    for index in range(layer_count):
        layer = build_layer(index)
        net.layers.append(layer)
        fill_parameters([(f"layers.{index}.{name}", parameter)
                         for name, parameter in layer.named_parameters()], rng)
        transform_layer(layer, packed, expert_ns)
        round_stream_dtype(layer)
    fill_parameters([(f"norm.{name}", parameter)
                     for name, parameter in net.norm.named_parameters()], rng)
    net.head.materialize()
    fill_parameters([(f"head.{name}", parameter)
                     for name, parameter in net.head.named_parameters()], rng)


def compact_spec(spec):
    """Replace one packed matrix's Python byte lists with page-backed tensors.

    `quantize_weight` returns `.tolist()` output, which costs a pointer per
    byte. The forward and the writer accept tensors, so each matrix is
    converted as soon as it is produced, and the tensor lives in a scratch file
    rather than in anonymous memory: twenty-four layers of packed weights are
    the largest resident in this case.
    """
    for key in ("data", "scales"):
        if isinstance(spec[key], torch.Tensor):
            continue
        values = spec[key]
        packed = page_backed((len(values),), torch.uint8, packed_folder())
        packed.copy_(torch.tensor(values, dtype=torch.uint8))
        spec[key] = packed


def transform_layer(layer, packed, expert_ns):
    prefix = f"layers.{layer.layer_id}"
    attention = layer.attn
    for name in ("wq_a", "wq_b", "wkv", "wo_b"):
        spec = attention_model.packed_model.quantize_weight(
            getattr(attention, name).weight, "fp8_block32")
        compact_spec(spec)
        packed[f"{prefix}.attn.{name}.weight"] = spec
        setattr(attention, name, mixed.Linear(spec))
    attention.window_kv_cache = attention.window_kv_cache.bfloat16()
    moe = layer.ffn
    for index, expert in enumerate(moe.experts):
        spec = {name: attention_model.packed_model.quantize_weight(
            getattr(expert, name).weight, "fp4_row32")
            for name in ("w1", "w2", "w3")}
        for weight in spec.values():
            compact_spec(weight)
        moe.experts[index] = mixed.expert(expert_ns, spec, expert.swiglu_limit)
        for name, weight in spec.items():
            packed[f"{prefix}.ffn.experts.{index}.{name}.weight"] = weight
    expert = moe.shared_experts
    spec = {name: attention_model.packed_model.quantize_weight(
        getattr(expert, name).weight, "fp8_block32")
        for name in ("w1", "w2", "w3")}
    for weight in spec.values():
        compact_spec(weight)
    moe.shared_experts = mixed.expert(expert_ns, spec, expert.swiglu_limit)
    for name, weight in spec.items():
        packed[f"{prefix}.ffn.shared_experts.{name}.weight"] = weight


def make_case(path, kernel_path, observer=None, sparse_sink=None):
    namespace = model.load_reference(path, kernel_path, hash_config={}, tokens=None)
    expert_ns = mixed.source_namespace(path)
    torch.set_default_dtype(torch.bfloat16)
    defer_large_allocations(namespace)
    args = namespace["ModelArgs"](**ARGS)
    # `Transformer.__init__` builds every block eagerly. Twenty-four
    # production-width FP32 blocks are 20.9 GiB, which cannot coexist with the
    # vocabulary tensors, so the shell is created empty and the blocks are
    # appended after each one has been packed. Every block still sees the
    # released args.
    args.n_layers = 0
    net = namespace["Transformer"](args)
    args.n_layers = ARGS["n_layers"]
    torch.set_default_dtype(torch.float32)
    namespace["sparse_attn"] = attention_model.sparse_attn
    rng = torch.Generator().manual_seed(SEED)
    packed = {}
    fill_and_quantize(net, packed, expert_ns, rng,
                      lambda layer_id: namespace["Block"](layer_id, args),
                      ARGS["n_layers"])
    if sparse_sink is not None:
        # Wrap the CPU mirror installed above so the capture sees the exact
        # inputs and FP32 result the graph uses.
        def capture(q, kv, sink, indices, scale):
            fp32 = model.sparse_attn(q.float(), kv.float(), sink, indices, scale)
            if not sparse_sink:
                sparse_sink.append(dict(
                    q=q.float().flatten().tolist(), kv=kv.float().flatten().tolist(),
                    sink=sink.float().flatten().tolist(),
                    indices=indices.flatten().tolist(), scale=scale,
                    out=fp32.flatten().tolist(),
                    rounded=fp32.to(q.dtype).float().flatten().tolist()))
            return fp32.to(q.dtype)

        namespace["sparse_attn"] = capture
    # The qualified BF16 contract rounds every non-HC parameter (wo_a included);
    # the head stays FP32 because the release promotes it for the logits, but
    # every value is already BF16-exact.
    for name, parameter in net.named_parameters():
        if "hc_" in name or name == "head.weight":
            continue
        if parameter.dtype != torch.bfloat16:
            parameter.data = parameter.data.detach().bfloat16()
    head_inputs, collapsed_inputs = [], []
    net.head.register_forward_pre_hook(lambda m, i: head_inputs.append(i[0].clone()))
    net.norm.register_forward_pre_hook(lambda m, i: collapsed_inputs.append(i[0].clone()))
    if observer is not None:
        observer.attach(net)
    steps = []
    torch.set_num_threads(1)
    with torch.inference_mode():
        for spec in STEPS:
            ids = torch.tensor(spec["ids"]).view(1, spec["seq"])
            if observer is not None:
                observer.begin(spec)
            net(ids, spec["pos"])
            if observer is not None:
                observer.end(spec)
            logits = net.head(head_inputs[-1], full_logits=True)
            steps.append(spec | {"logits": logits.flatten().tolist(),
                                 "shape": list(logits.shape),
                                 "collapsed": collapsed_inputs[-1].flatten().tolist(),
                                 "normalized": head_inputs[-1].flatten().tolist()})
    # The release stores the head as BF16; the reference promotes it to FP32 so
    # the logits come out in FP32. `write_checkpoint` keeps only the 16-bit copy
    # once the forward is done, after the block tensors have been written.
    return net, packed, steps


def compact_packed(packed):
    """Final safety net: no packed matrix may reach the writer as a byte list.

    `transform_layer` already compacts every matrix as it is produced; this
    repeats the conversion so a spec added elsewhere cannot reintroduce the
    pointer-per-byte cost at write time.
    """
    for spec in packed.values():
        compact_spec(spec)


def checkpoint_entries(net, packed, head):
    """Tensors for one write pass, in state-dict order."""
    entries = []
    for name, tensor in net.state_dict().items():
        if (name == "head.weight") != head:
            continue
        if tensor.dtype == torch.bfloat16:
            entries.append((name, "bfloat16", list(tensor.shape),
                            tensor.contiguous().view(torch.uint16)))
        else:
            entries.append((name, "float32", list(tensor.shape), tensor.contiguous()))
    if head:
        return entries
    for name, spec in packed.items():
        fp4 = spec["dtype"] == "fp4_row32"
        rows, cols = spec["rows"], spec["cols"]
        entries.append((name, "float4_e2m1fn_x2" if fp4 else "float8_e4m3fn",
                        [rows, cols // 2 if fp4 else cols],
                        spec["data"]))
        scale_rows = rows if fp4 or spec["dtype"] == "engram_fp8_row32" else (rows + 31) // 32
        entries.append((name.removesuffix("weight") + "scale", "float8_e8m0fnu",
                        [scale_rows, cols // 32],
                        spec["scales"].view(scale_rows, cols // 32)))
    return entries


def write_shards(entries, root, first, shard_bytes=320 * 1024 * 1024):
    def nbytes(tensor):
        return tensor.numel() * tensor.element_size()

    shards, current, size = [], [], 0
    for entry in entries:
        length = nbytes(entry[3])
        if current and size + length > shard_bytes:
            shards.append(current)
            current, size = [], 0
        current.append(entry)
        size += length
    if current:
        shards.append(current)

    hashes = {}
    weight_map = {}
    for offset, shard in enumerate(shards):
        filename = f"model-{first + offset:02d}.safetensors"
        specs = {name: safetensors.TensorSpec(dtype=dtype, shape=shape,
                 data_ptr=tensor.data_ptr(), data_len=nbytes(tensor))
                 for name, dtype, shape, tensor in shard}
        data = safetensors.serialize(specs)
        (root / filename).write_bytes(data)
        hashes[filename] = hashlib.sha256(data).hexdigest()
        weight_map.update({name: filename for name, _, _, _ in shard})
    return hashes, weight_map, first + len(shards)


def write_checkpoint(net, packed, root, shard_bytes=320 * 1024 * 1024):
    """Write the package in two passes, releasing weights between them.

    Holding the 1.23 GiB embedding, 10.4 GiB of BF16 blocks and the 2.47 GiB
    FP32 head together costs 14.1 GiB before any serialization buffer, which
    does not fit a consumer host. The blocks, embedding and packed matrices go
    to disk first and are dropped; only then is the head rounded to the BF16
    copy the release stores. Values and shard contents are unchanged.
    """
    root.mkdir(parents=True, exist_ok=True)
    compact_packed(packed)
    hashes, weight_map, next_index = write_shards(
        checkpoint_entries(net, packed, head=False), root, 0, shard_bytes)
    net.embed = None
    net.layers = torch.nn.ModuleList()
    packed.clear()
    net.head.weight.data = net.head.weight.detach().bfloat16()
    head_hashes, head_map, _ = write_shards(
        checkpoint_entries(net, packed, head=True), root, next_index, shard_bytes)
    hashes.update(head_hashes)
    weight_map.update(head_map)
    # Drop the last view of the page-backed head so its scratch file can go.
    net.head = None
    (root / "config.json").write_text(json.dumps(native_config(), separators=(",", ":")))
    (root / "model.safetensors.index.json").write_text(
        json.dumps(dict(weight_map=weight_map), separators=(",", ":")))
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--kernel-reference-path", type=Path, required=True)
    parser.add_argument("--write-fixture", action="store_true")
    parser.add_argument("--trace-output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    observer = bf16.Bf16Observer() if args.trace_output else None
    sparse_sink = [] if args.trace_output else None
    net, packed, steps = make_case(
        args.reference_path, args.kernel_reference_path, observer, sparse_sink)
    folder = f"deepseek_v41_production_{SEED}"
    # `write_checkpoint` releases the blocks and the packed matrices as it goes,
    # so count them while they are still resident.
    tensor_count, packed_count = len(net.state_dict()), len(packed)
    try:
        files = write_checkpoint(net, packed, FIXTURE.parent / folder)
    finally:
        release_scratch()
    fixture = dict(
        seed=SEED, revision=model.backbone.REVISION,
        source_sha256=model.backbone.SOURCE_SHA256,
        precision="production_width_real_vocabulary_bf16_residual_hc_cpu_pytorch",
        stream_precision="bf16", config=native_config(), batch_size=1,
        max_prefill=1, max_positions=8,
        checkpoint=dict(directory=folder, files=files,
                        tensor_count=tensor_count, packed_count=packed_count),
        steps=steps,
    )
    if args.trace_output:
        args.trace_output.write_text(json.dumps(dict(steps=observer.steps), allow_nan=False))
        args.trace_output.with_name("production-reference-sparse.json").write_text(
            json.dumps(sparse_sink[0] if sparse_sink else None, allow_nan=False))
    if args.write_fixture:
        FIXTURE.write_text(json.dumps(fixture, separators=(",", ":"), allow_nan=False))
    elif json.loads(FIXTURE.read_text()) != fixture:
        raise AssertionError("production fixture differs from the pinned CPU PyTorch reference")
    print("PASS production-width CPU oracle:", sum(len(s["logits"]) for s in steps), "logits")


if __name__ == "__main__":
    main()
