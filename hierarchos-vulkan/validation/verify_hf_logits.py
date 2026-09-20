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
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent

# Source-backed validation may need dependencies newer than the user's global
# Python environment.  Prefer an explicitly configured validation dependency
# directory, then the existing local K3 oracle cache when present.  This only
# affects the Python validation process; production execution remains native.
_validation_pydeps = os.environ.get("HIERARCHOS_VALIDATION_PYDEPS")
if _validation_pydeps:
    sys.path.insert(0, str(Path(_validation_pydeps).expanduser().resolve()))
else:
    _local_oracle_pydeps = ROOT / ".k3-oracle-pydeps"
    if _local_oracle_pydeps.is_dir():
        sys.path.insert(0, str(_local_oracle_pydeps.resolve()))
    _validation_pydeps_dir = ROOT / ".validation-pydeps"
    if _validation_pydeps_dir.is_dir():
        sys.path.insert(0, str(_validation_pydeps_dir.resolve()))

from safetensors.torch import save_file  # noqa: E402


def _transformers_source_root() -> Path:
    """Resolve the local Transformers checkout used as the reference implementation."""
    candidates: list[Path] = []
    configured = os.environ.get("HIERARCHOS_TRANSFORMERS_CHECKOUT")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            REPO / "transformers",
            REPO.parent / "transformers",
            Path.home() / "transformers",
        ]
    )
    for checkout in candidates:
        source_root = checkout / "src"
        if (source_root / "transformers" / "__init__.py").is_file():
            return source_root.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise RuntimeError(
        "local Hugging Face Transformers checkout not found; set "
        f"HIERARCHOS_TRANSFORMERS_CHECKOUT to the checkout root (searched: {searched})"
    )


TRANSFORMERS_SOURCE = _transformers_source_root()
sys.path.insert(0, str(TRANSFORMERS_SOURCE))

import transformers  # noqa: E402
from transformers import (  # noqa: E402
    BartConfig,
    BartForConditionalGeneration,
    BertConfig,
    BertForMaskedLM,
    DeepseekV3Config,
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    DbrxConfig,
    DbrxForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    GptOssConfig,
    GptOssForCausalLM,
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    Gemma4ForCausalLM,
    Gemma4TextConfig,
    Kimi_K25Config,
    Kimi_K25ForConditionalGeneration,
    Kimi_K25VisionConfig,
    LlamaConfig,
    LlamaForCausalLM,
    MiniMaxM2Config,
    MiniMaxM2ForCausalLM,
    MiniMaxM3VLForCausalLM,
    MiniMaxM3VLTextConfig,
    MistralConfig,
    MistralForCausalLM,
    Mistral4Config,
    Mistral4ForCausalLM,
    MixtralConfig,
    MixtralForCausalLM,
    Phi3Config,
    Phi3ForCausalLM,
    Phi4MultimodalAudioConfig,
    Phi4MultimodalConfig,
    Phi4MultimodalForCausalLM,
    Phi4MultimodalVisionConfig,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig,
    Qwen4ExpForCausalLM,
    Qwen4ExpTextConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SwitchTransformersConfig,
    SwitchTransformersForConditionalGeneration,
    T5Config,
    T5ForConditionalGeneration,
)

from kimi_k3_reference import tiny_kimi_k3_model  # noqa: E402


MANIFEST = ROOT / "Cargo.toml"

# High-signal parity set used by COMPATIBILITY.md. Keep this deliberately
# narrower than the architecture registry: every entry must have both live
# forward parity and the strict two-step AdamW parameter check.
#
# Kimi K3 uses Moonshot's released remote modeling code with the local
# Transformers checkout as its validation oracle. The mixed text graph has now
# passed the same strict two-step AdamW parameter-parity contract as the other
# headline families; the KDA-only and MLA-only fixtures remain focused probes.
HEADLINE_ADAMW_FAMILIES = (
    "deepseek_v4",
    "phi4_multimodal_text",
    "phi3",
    "kimi_k25_text",
    "kimi_k3_text",
    "gpt_oss",
    "mistral4",
    "minimax_m3",
    "gemma4",
    "minimax_m2",
)

KIMI_K3_ORACLE_FAMILIES = (
    "kimi_k3_kda",
    "kimi_k3_mla",
    "kimi_k3_text",
)

QWEN35_STRICT_FAMILIES = (
    "qwen3_5_full",
    "qwen3_5_linear",
    "qwen3_5_mixed",
    "qwen3_5_moe",
)

QWEN4_EXP_STRICT_FAMILIES = (
    "qwen4_exp_linear",
    "qwen4_exp_qsa",
    "qwen4_exp_ple",
    "qwen4_exp_mixed_ple",
)

STRICT_LOGIT_FAMILIES = (
    *KIMI_K3_ORACLE_FAMILIES,
    "gpt_oss",
    *QWEN35_STRICT_FAMILIES,
    *QWEN4_EXP_STRICT_FAMILIES,
)


def save_reference_checkpoint(name: str, model: torch.nn.Module, model_dir: Path) -> None:
    """Serialize the normalized reference graph without vendor reverse conversions."""
    if name in KIMI_K3_ORACLE_FAMILIES:
        # Moonshot's remote class uses the pre-main list form of
        # `_tied_weights_keys`, while current Transformers main expects a
        # mapping during save_pretrained(). The tiny K3 fixture deliberately
        # has no tied word embeddings, so writing its exact state_dict is the
        # semantically equivalent normalized checkpoint and avoids changing
        # either upstream source tree.
        model_dir.mkdir(parents=True, exist_ok=True)
        model.config.save_pretrained(model_dir)
        save_file(model.state_dict(), model_dir / "model.safetensors")
        return
    model.save_pretrained(
        model_dir,
        safe_serialization=True,
        save_original_format=False,
    )


def model_vocab_size(model: torch.nn.Module) -> int:
    """Return the LM vocabulary size for direct and multimodal wrapper configs."""
    vocab_size = getattr(model.config, "vocab_size", None)
    if vocab_size is None and hasattr(model.config, "text_config"):
        vocab_size = getattr(model.config.text_config, "vocab_size", None)
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError(f"{type(model.config).__name__} does not expose a positive LM vocabulary size")
    return vocab_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--atol", type=float, default=2.0e-4)
    parser.add_argument("--rtol", type=float, default=2.0e-4)
    parser.add_argument("--keep-fixtures", type=Path)
    parser.add_argument("--families", nargs="+", help="run only these family names")
    parser.add_argument(
        "--headline-eight",
        "--headline-strict",
        dest="headline_eight",
        action="store_true",
        help="run the source-backed families used by the strict two-step AdamW compatibility claim",
    )
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


def tiny_models(*, training_reference: bool = False) -> list[tuple[str, torch.nn.Module, bool]]:
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
    models = [
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
            "gpt_oss",
            GptOssForCausalLM(
                GptOssConfig(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=8,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    max_position_embeddings=16,
                    sliding_window=4,
                    num_local_experts=3,
                    num_experts_per_tok=2,
                    attention_dropout=0.0,
                    attention_bias=True,
                    tie_word_embeddings=False,
                    bos_token_id=1,
                    eos_token_id=2,
                    pad_token_id=0,
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
            "phi3",
            Phi3ForCausalLM(
                Phi3Config(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    max_position_embeddings=16,
                    original_max_position_embeddings=16,
                    resid_pdrop=0.0,
                    embd_pdrop=0.0,
                    attention_dropout=0.0,
                    rms_norm_eps=1.0e-5,
                    tie_word_embeddings=False,
                    bos_token_id=1,
                    eos_token_id=2,
                    pad_token_id=0,
                )
            ),
            False,
        ),
        (
            "phi4_multimodal_text",
            Phi4MultimodalForCausalLM(
                Phi4MultimodalConfig(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    max_position_embeddings=16,
                    original_max_position_embeddings=16,
                    resid_pdrop=0.0,
                    embd_pdrop=0.0,
                    attention_dropout=0.0,
                    rms_norm_eps=1.0e-5,
                    tie_word_embeddings=False,
                    bos_token_id=1,
                    eos_token_id=2,
                    pad_token_id=0,
                    sliding_window=None,
                    vision_config=Phi4MultimodalVisionConfig(
                        hidden_size=8,
                        intermediate_size=16,
                        num_hidden_layers=1,
                        num_attention_heads=1,
                        image_size=14,
                        patch_size=14,
                        crop_size=14,
                    ),
                    audio_config=Phi4MultimodalAudioConfig(
                        hidden_size=8,
                        intermediate_size=8,
                        num_blocks=1,
                        num_attention_heads=1,
                        ext_pw_out_channel=8,
                        depthwise_separable_out_channel=8,
                        nemo_conv_channels=8,
                        input_size=8,
                    ),
                )
            ),
            False,
        ),
        (
            "gemma3",
            Gemma3ForCausalLM(
                Gemma3TextConfig(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    max_position_embeddings=16,
                    rms_norm_eps=1.0e-6,
                    attention_dropout=0.0,
                    query_pre_attn_scalar=8,
                    sliding_window=4,
                    layer_types=["sliding_attention", "full_attention"],
                    tie_word_embeddings=False,
                    bos_token_id=1,
                    eos_token_id=2,
                    pad_token_id=0,
                )
            ),
            False,
        ),
        (
            "gemma4",
            Gemma4ForCausalLM(
                Gemma4TextConfig(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    global_head_dim=8,
                    max_position_embeddings=16,
                    rms_norm_eps=1.0e-6,
                    attention_dropout=0.0,
                    sliding_window=4,
                    layer_types=["sliding_attention", "full_attention"],
                    rope_parameters={
                        "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
                        "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0},
                    },
                    hidden_size_per_layer_input=0,
                    num_kv_shared_layers=0,
                    attention_k_eq_v=False,
                    enable_moe_block=False,
                    tie_word_embeddings=False,
                    bos_token_id=1,
                    eos_token_id=2,
                    pad_token_id=0,
                )
            ),
            False,
        ),
        (
            "kimi_k25_text",
            Kimi_K25ForConditionalGeneration(
                Kimi_K25Config(
                    text_config=DeepseekV3Config(
                        vocab_size=32,
                        hidden_size=16,
                        intermediate_size=32,
                        moe_intermediate_size=8,
                        num_hidden_layers=2,
                        num_attention_heads=2,
                        num_key_value_heads=2,
                        n_shared_experts=1,
                        n_routed_experts=3,
                        routed_scaling_factor=1.0,
                        kv_lora_rank=4,
                        q_lora_rank=4,
                        qk_rope_head_dim=4,
                        v_head_dim=8,
                        qk_nope_head_dim=4,
                        n_group=1,
                        topk_group=1,
                        num_experts_per_tok=2,
                        first_k_dense_replace=0,
                        norm_topk_prob=True,
                        max_position_embeddings=16,
                        attention_dropout=0.0,
                        num_mtp_layers=0,
                        tie_word_embeddings=False,
                        bos_token_id=1,
                        eos_token_id=2,
                        pad_token_id=0,
                    ),
                    vision_config=Kimi_K25VisionConfig(
                        patch_size=2,
                        pos_emb_height=2,
                        pos_emb_width=2,
                        pos_emb_time=1,
                        num_attention_heads=1,
                        num_hidden_layers=1,
                        hidden_size=8,
                        intermediate_size=16,
                        merge_kernel_size=(1, 1),
                        max_position_embeddings=8,
                    ),
                    projection_hidden_size=8,
                    image_token_id=29,
                    video_token_id=30,
                    vision_start_token_id=27,
                    vision_end_token_id=28,
                    tie_word_embeddings=False,
                )
            ),
            False,
        ),
        (
            "mistral4",
            Mistral4ForCausalLM(
                Mistral4Config(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    moe_intermediate_size=8,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    n_shared_experts=1,
                    n_routed_experts=3,
                    routed_scaling_factor=1.0,
                    kv_lora_rank=4,
                    q_lora_rank=4,
                    qk_rope_head_dim=4,
                    v_head_dim=8,
                    qk_nope_head_dim=4,
                    n_group=1,
                    topk_group=1,
                    num_experts_per_tok=2,
                    first_k_dense_replace=0,
                    norm_topk_prob=True,
                    max_position_embeddings=16,
                    attention_dropout=0.0,
                    tie_word_embeddings=False,
                    bos_token_id=1,
                    eos_token_id=2,
                    pad_token_id=0,
                )
            ),
            False,
        ),
        (
            "deepseek_v4",
            DeepseekV4ForCausalLM(
                DeepseekV4Config(
                    vocab_size=32,
                    hidden_size=16,
                    moe_intermediate_size=8,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    q_lora_rank=4,
                    num_experts_per_tok=2,
                    n_routed_experts=3,
                    n_shared_experts=1,
                    scoring_func="sqrtsoftplus",
                    norm_topk_prob=True,
                    routed_scaling_factor=1.5,
                    max_position_embeddings=16,
                    layer_types=[
                        "heavily_compressed_attention",
                        "compressed_sparse_attention",
                    ],
                    compress_rates={
                        "compressed_sparse_attention": 2,
                        "heavily_compressed_attention": 2,
                    },
                    compress_rope_theta=10_000.0,
                    hc_mult=2,
                    hc_sinkhorn_iters=2,
                    hc_eps=1.0e-6,
                    mlp_layer_types=["hash_moe", "moe"],
                    sliding_window=4,
                    o_groups=2,
                    o_lora_rank=2,
                    index_n_heads=2,
                    index_head_dim=8,
                    index_topk=2,
                    num_nextn_predict_layers=0,
                    partial_rotary_factor=0.5,
                    router_jitter_noise=0.0,
                    attention_dropout=0.0,
                    tie_word_embeddings=False,
                    bos_token_id=1,
                    eos_token_id=2,
                    pad_token_id=0,
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
            "qwen3_5_linear",
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
                    layer_types=["linear_attention"],
                    use_cache=False,
                    tie_word_embeddings=False,
                    bos_token_id=1,
                    eos_token_id=2,
                )
            ),
            False,
        ),
        (
            "qwen3_5_mixed",
            Qwen3_5ForCausalLM(
                Qwen3_5TextConfig(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=4,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    max_position_embeddings=16,
                    linear_conv_kernel_dim=2,
                    linear_key_head_dim=4,
                    linear_value_head_dim=4,
                    linear_num_key_heads=2,
                    linear_num_value_heads=4,
                    use_cache=False,
                    tie_word_embeddings=False,
                    bos_token_id=1,
                    eos_token_id=2,
                )
            ),
            False,
        ),
        (
            "qwen3_5_moe",
            Qwen3_5MoeForCausalLM(
                Qwen3_5MoeTextConfig(
                    vocab_size=32,
                    hidden_size=16,
                    num_hidden_layers=2,
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
                    layer_types=["linear_attention", "full_attention"],
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
                    indexer_budget=2,
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
            "qwen4_exp_ple",
            Qwen4ExpForCausalLM(
                Qwen4ExpTextConfig(
                    vocab_size=32,
                    hidden_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    max_position_embeddings=32,
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
                    ple_layer_ids=[1],
                    ple_embed_dim=8,
                    ple_conv_kernel_size=3,
                    ngram_size=3,
                    heads_per_ngram=1,
                    ngram_vocab_size_base=17,
                    make_ngram_vocab_size_divisible_by=8,
                    split_ngram_parts=1,
                    seed=17,
                    use_cache=False,
                    tie_word_embeddings=False,
                    bos_token_id=1,
                    eos_token_id=2,
                )
            ),
            False,
        ),
        (
            "qwen4_exp_mixed_ple",
            Qwen4ExpForCausalLM(
                Qwen4ExpTextConfig(
                    vocab_size=32,
                    hidden_size=16,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    max_position_embeddings=32,
                    linear_conv_kernel_dim=2,
                    linear_key_head_dim=4,
                    linear_value_head_dim=4,
                    linear_num_key_heads=2,
                    linear_num_value_heads=4,
                    moe_intermediate_size=8,
                    shared_expert_intermediate_size=8,
                    num_experts=3,
                    num_experts_per_tok=2,
                    layer_types=["linear_attention", "qwen_sparse_attention"],
                    hc_count=2,
                    hc_lowrank=4,
                    ple_layer_ids=[1],
                    ple_embed_dim=8,
                    ple_conv_kernel_size=3,
                    ngram_size=3,
                    heads_per_ngram=1,
                    ngram_vocab_size_base=17,
                    make_ngram_vocab_size_divisible_by=8,
                    split_ngram_parts=1,
                    seed=29,
                    indexer_n_heads=2,
                    indexer_kv_heads=1,
                    indexer_head_dim=8,
                    indexer_budget=2,
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
    for name, schedule in (
        ("kimi_k3_kda", "kda"),
        ("kimi_k3_mla", "mla"),
        ("kimi_k3_text", "mixed"),
    ):
        model = tiny_kimi_k3_model(training_reference=training_reference, schedule=schedule)
        if model is not None:
            models.append((name, model, False))
    for name, model, _ in models:
        if name == "deepseek_v4":
            # DeepseekV4HashRouter intentionally initializes this persistent
            # lookup buffer to zeros.  A real checkpoint supplies a token ->
            # top-k expert table; populate a deterministic valid table for the
            # tiny fixture so both implementations exercise hash routing rather
            # than an invalid duplicate-expert placeholder.
            tid2eid = model.model.layers[0].mlp.gate.tid2eid
            with torch.no_grad():
                for token_id in range(tid2eid.shape[0]):
                    for slot in range(tid2eid.shape[1]):
                        tid2eid[token_id, slot] = (token_id + slot) % model.config.num_local_experts
    return models


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
    if name in STRICT_LOGIT_FAMILIES:
        # K3 acceptance is intentionally stronger than the broad compatibility
        # suite. GPT-OSS uses the same strict absolute-only contract against the
        # current local Transformers implementation.
        atol = min(atol, 2.0e-7)
        rtol = 0.0
    # Compare against Transformers' normalized in-memory parameter graph. Some
    # current families (notably DeepSeek-V4) register a reverse conversion for
    # Hub serialization that rewrites `self_attn.*` tensors back to vendor
    # checkpoint names.  The native loader/training parity path intentionally
    # targets the normalized graph, so keep the same format here as in the
    # two-step AdamW verifier.
    save_reference_checkpoint(name, model, model_dir)

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
    visible = attention_mask.bool().reshape(-1, 1).expand(-1, model_vocab_size(model)).reshape(-1)
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
    if not source.is_relative_to(TRANSFORMERS_SOURCE):
        raise RuntimeError(
            f"expected local Transformers checkout under {TRANSFORMERS_SOURCE}, imported {source}"
        )
    models = tiny_models()
    if args.headline_eight and args.families:
        raise ValueError("--headline-strict cannot be combined with --families")
    if args.headline_eight:
        model_by_name = {entry[0]: entry for entry in models}
        missing = [name for name in HEADLINE_ADAMW_FAMILIES if name not in model_by_name]
        if missing:
            raise ValueError(f"headline families are missing fixtures: {missing}")
        models = [model_by_name[name] for name in HEADLINE_ADAMW_FAMILIES]
    elif args.families:
        unknown = set(args.families) - {name for name, _, _ in models}
        if unknown:
            raise ValueError(f"unknown families: {sorted(unknown)}")
        models = [entry for entry in models if entry[0] in args.families]
    binary = build_binary()
    if args.keep_fixtures is not None:
        fixture_root = args.keep_fixtures.resolve()
        fixture_root.mkdir(parents=True, exist_ok=True)
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-hf-parity-")
        fixture_root = Path(cleanup.name)

    results = []
    try:
        for name, model, token_types in models:
            # The CPU KDA oracle intentionally implements the dense sequence
            # path used for numerical parity; padding/varlen FLA behavior is a
            # separate kernel contract and is not emulated here.
            padded_cases = (False,) if name in KIMI_K3_ORACLE_FAMILIES else (False, True)
            for padded in padded_cases:
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
