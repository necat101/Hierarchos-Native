#!/usr/bin/env python3
"""Compare native Vulkan logits against tiny Hugging Face reference models.

Creates deterministic tiny dense, MoE, masked-LM and encoder-decoder checkpoints
using the adjacent Transformers checkout. Runs the Rust/Vulkan logit probe on
the same SafeTensors packages and checks every visible token's logits.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "src"))

import transformers  # noqa: E402
from transformers import (  # noqa: E402
    BartConfig,
    BartForConditionalGeneration,
    BertConfig,
    BertForMaskedLM,
    DbrxConfig,
    DbrxForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    MiniMaxM2Config,
    MiniMaxM2ForCausalLM,
    MiniMaxM3VLForCausalLM,
    MiniMaxM3VLTextConfig,
    MistralConfig,
    MistralForCausalLM,
    MixtralConfig,
    MixtralForCausalLM,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
    Qwen4ExpForCausalLM,
    Qwen4ExpTextConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SwitchTransformersConfig,
    SwitchTransformersForConditionalGeneration,
    T5Config,
    T5ForConditionalGeneration,
)


MANIFEST = ROOT / "Cargo.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--atol", type=float, default=2.0e-4)
    parser.add_argument("--rtol", type=float, default=2.0e-4)
    parser.add_argument("--keep-fixtures", type=Path)
    parser.add_argument("--families", nargs="+", help="run only these family names")
    return parser.parse_args()


def build_binary(name: str = "hierarchos-vulkan-transformer-logits") -> Path:
    result = subprocess.run(
        [
            "cargo",
            "build",
            "--manifest-path",
            str(MANIFEST),
            "--bin",
            name,
            "--message-format=json-render-diagnostics",
        ],
        cwd=ROOT.parent,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    for line in result.stdout.splitlines():
        message = json.loads(line)
        if message.get("reason") == "compiler-artifact" and message.get("executable"):
            if message["target"]["name"] == name:
                return Path(message["executable"])
    raise RuntimeError(f"Cargo did not report an executable for {name}")


def tiny_models() -> list[tuple[str, torch.nn.Module, bool]]:
    common_decoder = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 16,
        "tie_word_embeddings": False,
    }
    return [
        (
            "gpt2",
            GPT2LMHeadModel(
                GPT2Config(
                    vocab_size=32,
                    n_positions=16,
                    n_ctx=16,
                    n_embd=16,
                    n_layer=2,
                    n_head=2,
                    n_inner=32,
                    resid_pdrop=0.0,
                    embd_pdrop=0.0,
                    attn_pdrop=0.0,
                    tie_word_embeddings=False,
                    bos_token_id=1,
                    eos_token_id=2,
                )
            ),
            False,
        ),
        (
            "llama",
            LlamaForCausalLM(
                LlamaConfig(
                    **common_decoder,
                    attention_bias=False,
                    mlp_bias=False,
                    rms_norm_eps=1.0e-5,
                )
            ),
            False,
        ),
        (
            "qwen2",
            Qwen2ForCausalLM(
                Qwen2Config(
                    **common_decoder,
                    attention_bias=True,
                    rms_norm_eps=1.0e-6,
                    use_sliding_window=False,
                )
            ),
            False,
        ),
        (
            "mistral",
            MistralForCausalLM(
                MistralConfig(
                    **common_decoder,
                    rms_norm_eps=1.0e-5,
                    sliding_window=8,
                )
            ),
            False,
        ),
        (
            "mixtral",
            MixtralForCausalLM(
                MixtralConfig(
                    **common_decoder,
                    num_local_experts=3,
                    num_experts_per_tok=2,
                    router_jitter_noise=0.1,
                    sliding_window=4,
                )
            ),
            False,
        ),
        (
            "qwen3_5_full",
            Qwen3_5ForCausalLM(
                Qwen3_5TextConfig(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    max_position_embeddings=16,
                    linear_conv_kernel_dim=2,
                    linear_key_head_dim=4,
                    linear_value_head_dim=4,
                    linear_num_key_heads=2,
                    linear_num_value_heads=4,
                    layer_types=["full_attention"],
                    use_cache=False,
                    tie_word_embeddings=False,
                    bos_token_id=1,
                    eos_token_id=2,
                )
            ),
            False,
        ),
        (
            "qwen4_exp_linear",
            Qwen4ExpForCausalLM(
                Qwen4ExpTextConfig(
                    vocab_size=32,
                    hidden_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    max_position_embeddings=16,
                    linear_conv_kernel_dim=2,
                    linear_key_head_dim=4,
                    linear_value_head_dim=4,
                    linear_num_key_heads=2,
                    linear_num_value_heads=4,
                    moe_intermediate_size=8,
                    shared_expert_intermediate_size=8,
                    num_experts=3,
                    num_experts_per_tok=2,
                    layer_types=["linear_attention"],
                    hc_count=2,
                    hc_lowrank=4,
                    ple_layer_ids=[],
                    use_cache=False,
                    tie_word_embeddings=False,
                    bos_token_id=1,
                    eos_token_id=2,
                )
            ),
            False,
        ),
        (
            "qwen4_exp_qsa",
            Qwen4ExpForCausalLM(
                Qwen4ExpTextConfig(
                    vocab_size=32,
                    hidden_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    max_position_embeddings=16,
                    linear_conv_kernel_dim=2,
                    linear_key_head_dim=4,
                    linear_value_head_dim=4,
                    linear_num_key_heads=2,
                    linear_num_value_heads=4,
                    moe_intermediate_size=8,
                    shared_expert_intermediate_size=8,
                    num_experts=3,
                    num_experts_per_tok=2,
                    layer_types=["qwen_sparse_attention"],
                    hc_count=2,
                    hc_lowrank=4,
                    ple_layer_ids=[],
                    indexer_n_heads=2,
                    indexer_kv_heads=1,
                    indexer_head_dim=8,
                    indexer_budget=16,
                    indexer_compress_ratio=2,
                    use_cache=False,
                    tie_word_embeddings=False,
                    bos_token_id=1,
                    eos_token_id=2,
                )
            ),
            False,
        ),
        (
            "minimax_m2",
            MiniMaxM2ForCausalLM(
                MiniMaxM2Config(
                    **common_decoder,
                    head_dim=16,
                    num_local_experts=3,
                    num_experts_per_tok=2,
                    router_jitter_noise=0.1,
                    bos_token_id=1,
                    eos_token_id=2,
                )
            ),
            False,
        ),
        (
            "minimax_m3_dense",
            MiniMaxM3VLForCausalLM(
                MiniMaxM3VLTextConfig(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=8,
                    dense_intermediate_size=32,
                    shared_intermediate_size=8,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    max_position_embeddings=16,
                    tie_word_embeddings=False,
                    bos_token_id=1,
                    eos_token_id=2,
                    num_local_experts=3,
                    num_experts_per_tok=2,
                    router_jitter_noise=0.0,
                    rotary_dim=8,
                    index_n_heads=1,
                    index_head_dim=8,
                    index_block_size=2,
                    index_topk_blocks=2,
                    index_local_blocks=1,
                    layer_types=["full_attention", "full_attention"],
                    mlp_layer_types=["dense", "dense"],
                )
            ),
            False,
        ),
        (
            "minimax_m3",
            MiniMaxM3VLForCausalLM(
                MiniMaxM3VLTextConfig(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=8,
                    dense_intermediate_size=32,
                    shared_intermediate_size=8,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    max_position_embeddings=16,
                    tie_word_embeddings=False,
                    bos_token_id=1,
                    eos_token_id=2,
                    num_local_experts=3,
                    num_experts_per_tok=2,
                    router_jitter_noise=0.0,
                    rotary_dim=8,
                    index_n_heads=1,
                    index_head_dim=8,
                    index_block_size=2,
                    index_topk_blocks=2,
                    index_local_blocks=1,
                    layer_types=["full_attention", "minimax_m3_sparse"],
                    mlp_layer_types=["dense", "sparse"],
                )
            ),
            False,
        ),
        (
            "dbrx",
            DbrxForCausalLM(
                DbrxConfig(
                    vocab_size=32,
                    d_model=16,
                    n_heads=2,
                    n_layers=2,
                    max_seq_len=16,
                    attn_config={"kv_n_heads": 1, "rope_theta": 10000.0, "clip_qkv": 2.0},
                    ffn_config={"ffn_hidden_size": 32, "moe_num_experts": 3, "moe_top_k": 2, "moe_jitter_eps": 0.1},
                )
            ),
            False,
        ),
        (
            "t5",
            T5ForConditionalGeneration(
                T5Config(
                    vocab_size=32,
                    d_model=16,
                    d_kv=8,
                    d_ff=32,
                    num_layers=2,
                    num_decoder_layers=2,
                    num_heads=2,
                    dropout_rate=0.0,
                    decoder_start_token_id=0,
                )
            ),
            False,
        ),
        (
            "bart",
            BartForConditionalGeneration(
                BartConfig(
                    vocab_size=32,
                    d_model=16,
                    encoder_layers=2,
                    decoder_layers=2,
                    encoder_attention_heads=2,
                    decoder_attention_heads=2,
                    encoder_ffn_dim=32,
                    decoder_ffn_dim=32,
                    max_position_embeddings=16,
                    dropout=0.0,
                    attention_dropout=0.0,
                    activation_dropout=0.0,
                )
            ),
            False,
        ),
        (
            "switch_transformers",
            SwitchTransformersForConditionalGeneration(
                SwitchTransformersConfig(
                    vocab_size=32,
                    d_model=16,
                    d_kv=8,
                    d_ff=32,
                    num_layers=3,
                    num_decoder_layers=2,
                    num_heads=2,
                    num_experts=3,
                    num_sparse_encoder_layers=3,
                    num_sparse_decoder_layers=1,
                    expert_capacity=8,
                    dropout_rate=0.0,
                    decoder_start_token_id=0,
                )
            ),
            False,
        ),
        (
            "bert",
            BertForMaskedLM(
                BertConfig(
                    vocab_size=32,
                    hidden_size=16,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    intermediate_size=32,
                    max_position_embeddings=16,
                    type_vocab_size=2,
                    hidden_dropout_prob=0.0,
                    attention_probs_dropout_prob=0.0,
                    tie_word_embeddings=True,
                )
            ),
            True,
        ),
    ]


def compare_family(
    binary: Path,
    root: Path,
    name: str,
    model: torch.nn.Module,
    token_types: bool,
    device_index: int | None,
    atol: float,
    rtol: float,
    padded: bool = False,
) -> dict[str, object]:
    case = "padded" if padded else "unmasked"
    model_dir = root / name / case
    model.eval()
    model.save_pretrained(model_dir, safe_serialization=True)

    input_ids = torch.tensor([[1, 5, 9, 3, 7, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    if padded:
        attention_mask[0, -2:] = 0
    if padded:
        input_ids = torch.cat([input_ids, input_ids.flip(1)])
        attention_mask = torch.cat([attention_mask, torch.tensor([[0, 1, 1, 1, 1, 1]])])
    token_type_ids = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long) if token_types else None
    if token_type_ids is not None:
        token_type_ids = token_type_ids.expand_as(input_ids)
    encoder_ids = torch.tensor([[4, 8, 12, 6, 2]], dtype=torch.long)
    encoder_mask = torch.tensor([[1, 1, 1, 0, 0] if padded else [1, 1, 1, 1, 1]])
    if padded:
        encoder_ids = torch.cat([encoder_ids, encoder_ids.flip(1)])
        encoder_mask = torch.cat([encoder_mask, torch.tensor([[0, 1, 1, 1, 1]])])
    with torch.no_grad():
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        if model.config.is_encoder_decoder:
            kwargs = {
                "input_ids": encoder_ids,
                "attention_mask": encoder_mask,
                "decoder_input_ids": input_ids,
                "decoder_attention_mask": attention_mask,
            }
        reference = model(**kwargs).logits.detach().cpu().to(torch.float32).reshape(-1)

    fixture = {
        "batch_size": input_ids.shape[0],
        "seq_len": input_ids.shape[1],
        "input_ids": input_ids.reshape(-1).tolist(),
        "attention_mask": attention_mask.to(torch.float32).reshape(-1).tolist(),
    }
    if token_type_ids is not None:
        fixture["token_type_ids"] = token_type_ids.reshape(-1).tolist()
    if model.config.is_encoder_decoder:
        fixture["encoder_input_ids"] = encoder_ids.reshape(-1).tolist()
        fixture["encoder_attention_mask"] = encoder_mask.to(torch.float32).reshape(-1).tolist()
    fixture_path = model_dir / "vulkan_logits_fixture.json"
    report_path = model_dir / "vulkan_logits_report.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")

    command = [
        str(binary),
        "--model",
        str(model_dir),
        "--fixture",
        str(fixture_path),
        "--output",
        str(report_path),
    ]
    if device_index is not None:
        command.extend(["--device-index", str(device_index)])
    subprocess.run(command, cwd=ROOT.parent, check=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    actual = torch.tensor(report["logits"], dtype=torch.float32)
    if actual.shape != reference.shape:
        raise AssertionError(f"{name}: Vulkan logits {actual.shape} != HF logits {reference.shape}")

    # Fully masked query rows are backend-dependent in HF; compare the
    # observable logits at visible positions, including mixed padding in batch.
    visible = attention_mask.bool().reshape(-1, 1).expand(-1, model.config.vocab_size).reshape(-1)
    actual, reference = actual[visible], reference[visible]

    delta = (actual - reference).abs()
    max_abs = float(delta.max().item())
    denominator = reference.abs().clamp_min(1.0e-8)
    max_rel = float((delta / denominator).max().item())
    close = torch.isclose(actual, reference, atol=atol, rtol=rtol)
    failing = int((~close).sum().item())
    result = {
        "family": name,
        "case": case,
        "architecture": report["architecture"],
        "device": report["device"],
        "values": actual.numel(),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "failing_values": failing,
        "atol": atol,
        "rtol": rtol,
    }
    if failing:
        worst = int(delta.argmax().item())
        result["worst_index"] = worst
        result["hf"] = float(reference[worst].item())
        result["vulkan"] = float(actual[worst].item())
        raise AssertionError(json.dumps(result, indent=2))
    if not (math.isfinite(max_abs) and math.isfinite(max_rel)):
        raise AssertionError(f"{name}: non-finite parity metric")
    return result


def main() -> None:
    args = parse_args()
    torch.manual_seed(0x48494552)
    source = Path(transformers.__file__).resolve()
    if not source.is_relative_to(REPO / "src"):
        raise RuntimeError(f"expected adjacent Transformers checkout, imported {source}")
    models = tiny_models()
    if args.families:
        unknown = set(args.families) - {name for name, _, _ in models}
        if unknown:
            raise ValueError(f"unknown families: {sorted(unknown)}")
        models = [entry for entry in models if entry[0] in args.families]
    binary = build_binary()
    if args.keep_fixtures is not None:
        args.keep_fixtures.mkdir(parents=True, exist_ok=True)
        fixture_root = args.keep_fixtures
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-hf-parity-")
        fixture_root = Path(cleanup.name)

    results = []
    try:
        for name, model, token_types in models:
            for padded in (False, True):
                results.append(
                    compare_family(
                        binary,
                        fixture_root,
                        name,
                        model,
                        token_types,
                        args.device_index,
                        args.atol,
                        args.rtol,
                        padded,
                    )
                )
    finally:
        if cleanup is not None:
            cleanup.cleanup()
    print(
        json.dumps(
            {
                "result": "pass",
                "transformers_version": transformers.__version__,
                "transformers_source": str(source),
                "families": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
