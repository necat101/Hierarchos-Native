#!/usr/bin/env python3
"""Moonshot Kimi-K3 reference helpers for Vulkan parity validation.

The released Kimi-K3 remote code depends on ``fla-core`` kernels.  The Vulkan
parity environment used by this repository is CPU-only and intentionally does
not make FLA a production dependency, so this module supplies small,
autograd-friendly PyTorch equivalents for the exact FLA primitives used by
``modeling_kimi_linear.py``.  Everything else -- Kimi's module graph, tensor
names, MLA, SiTU, LatentMoE, AttnRes, routing, and checkpoint serialization --
comes directly from Moonshot's remote model code.

These replacements are validation-only.  They are not imported by the native
runtime.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.util
import math
import os
import sys
import types
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


KIMI_K3_FAMILY = "kimi_k3_text"


def find_kimi_k3_remote_source() -> Path | None:
    """Find the locally cached Moonshot Kimi-K3 remote-code snapshot."""
    configured = os.environ.get("HIERARCHOS_KIMI_K3_REMOTE_DIR")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    snapshots = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--moonshotai--Kimi-K3"
        / "snapshots"
    )
    if snapshots.is_dir():
        candidates.extend(sorted((path for path in snapshots.iterdir() if path.is_dir()), reverse=True))
    for candidate in candidates:
        if (candidate / "configuration_kimi_k3.py").is_file() and (
            candidate / "modeling_kimi_linear.py"
        ).is_file():
            return candidate.resolve()
    return None


class _ReferenceShortConvolution(nn.Module):
    """Pure-PyTorch equivalent of FLA ShortConvolution for Kimi's causal case."""

    def __init__(self, hidden_size: int, kernel_size: int, activation: str | None = None, **_: Any):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.weight = nn.Parameter(torch.empty(hidden_size, kernel_size))

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if cu_seqlens is not None:
            raise NotImplementedError("the Kimi-K3 parity fixture uses dense unpadded sequences")
        if x.ndim != 3:
            raise ValueError(f"ShortConvolution expected [batch, sequence, channels], got {tuple(x.shape)}")
        batch, seq_len, channels = x.shape
        if channels != self.hidden_size:
            raise ValueError(f"ShortConvolution got {channels} channels, expected {self.hidden_size}")

        # FLA/PyTorch conv1d uses correlation.  Kimi applies left causal padding
        # of K-1, exactly matching the native depthwise Vulkan primitive.
        history_len = self.kernel_size - 1
        if cache is None:
            history = x.new_zeros(batch, history_len, channels)
        else:
            history = cache
            if history.ndim == 3 and history.shape[1] == channels and history.shape[2] == history_len:
                history = history.transpose(1, 2)
            if tuple(history.shape) != (batch, history_len, channels):
                raise ValueError(
                    "ShortConvolution cache has shape "
                    f"{tuple(history.shape)}, expected {(batch, history_len, channels)}"
                )
        extended = torch.cat((history, x), dim=1)
        y = F.conv1d(
            extended.transpose(1, 2),
            self.weight.unsqueeze(1),
            groups=channels,
        ).transpose(1, 2)
        if self.activation == "silu":
            y = F.silu(y)
        elif self.activation not in (None, "identity"):
            raise NotImplementedError(f"unsupported ShortConvolution activation {self.activation!r}")

        final_state = None
        if output_final_state:
            final_state = extended[:, -history_len:, :] if history_len else extended[:, :0, :]
        return y[:, -seq_len:, :], final_state


class _ReferenceFusedRMSNormGated(nn.Module):
    """Kimi's FLA gated RMSNorm expressed with ordinary PyTorch ops."""

    def __init__(self, hidden_size: int, eps: float = 1.0e-6, activation: str = "sigmoid", **_: Any):
        super().__init__()
        if activation != "sigmoid":
            raise NotImplementedError(f"Kimi reference gated RMSNorm only supports sigmoid, got {activation!r}")
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        inv_rms = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
        return (xf * inv_rms * self.weight.float() * gate.float().sigmoid()).to(dtype)


def _reference_kda(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    transpose_state_layout: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Source-equivalent KDA recurrence used by Moonshot's KimiDeltaAttention."""
    if cu_seqlens is not None:
        raise NotImplementedError("the Kimi-K3 parity fixture uses dense unpadded sequences")
    if not (use_qk_l2norm_in_kernel and use_gate_in_kernel and use_beta_sigmoid_in_kernel):
        raise NotImplementedError("Kimi-K3 parity requires the production KDA kernel options")
    if not transpose_state_layout:
        raise NotImplementedError("Kimi-K3 uses transpose_state_layout=True")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"Kimi KDA expects equal Q/K/V geometry, got {q.shape}, {k.shape}, {v.shape}")

    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if value_dim != key_dim:
        raise ValueError("the current Kimi-K3 projection uses equal key/value head dimensions")
    if g.shape != q.shape:
        raise ValueError(f"Kimi KDA forget gate shape {g.shape} does not match Q shape {q.shape}")
    if tuple(beta.shape) != (batch, seq_len, heads):
        raise ValueError(f"Kimi KDA beta shape {beta.shape} is not {(batch, seq_len, heads)}")

    state_shape = (batch, heads, key_dim, value_dim)
    if initial_state is None:
        state = q.new_zeros(state_shape, dtype=torch.float32)
    else:
        state = initial_state.float()
        if tuple(state.shape) != state_shape:
            raise ValueError(f"Kimi KDA state shape {state.shape} is not {state_shape}")

    dt = dt_bias.float().reshape(heads, key_dim)
    rate = A_log.float().exp().reshape(1, heads, 1)
    query_scale = 1.0 / math.sqrt(float(key_dim))
    outputs: list[torch.Tensor] = []
    for token in range(seq_len):
        q_t = q[:, token].float()
        k_t = k[:, token].float()
        v_t = v[:, token].float()
        forget = g[:, token].float()
        u = forget + dt.unsqueeze(0)
        if safe_gate or lower_bound is not None:
            if lower_bound is None:
                raise ValueError("safe KDA gate requires lower_bound")
            log_decay = float(lower_bound) * torch.sigmoid(rate * u)
        else:
            log_decay = -rate * F.softplus(u)
        beta_t = beta[:, token].float().sigmoid()

        q_norm = q_t * torch.rsqrt(q_t.square().sum(dim=-1, keepdim=True) + 1.0e-6)
        k_norm = k_t * torch.rsqrt(k_t.square().sum(dim=-1, keepdim=True) + 1.0e-6)
        state = state * log_decay.exp().unsqueeze(-1)
        memory = torch.einsum("bhkv,bhk->bhv", state, k_norm)
        delta = (v_t - memory) * beta_t.unsqueeze(-1)
        state = state + torch.einsum("bhk,bhv->bhkv", k_norm, delta)
        out = torch.einsum("bhkv,bhk->bhv", state, q_norm) * query_scale
        outputs.append(out.to(v.dtype))
    output = torch.stack(outputs, dim=1)
    return output, state if output_final_state else None


def _tensor_cache(function):
    # Dense deterministic fixtures do not need FLA's memoization wrapper.
    return functools.wraps(function)(function)


def _prepare_lens_from_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask.to(torch.int32).sum(dim=-1)


def _prepare_cu_seqlens_from_mask(mask: torch.Tensor) -> torch.Tensor:
    lens = _prepare_lens_from_mask(mask)
    zero = lens.new_zeros(1)
    return torch.cat((zero, lens.cumsum(dim=0)), dim=0)


def _install_reference_fla() -> None:
    """Expose only the FLA symbols imported by Moonshot's Kimi remote code."""
    if "fla" in sys.modules:
        return
    if importlib.util.find_spec("fla") is not None:
        return
    fla = types.ModuleType("fla")
    modules = types.ModuleType("fla.modules")
    ops = types.ModuleType("fla.ops")
    kda = types.ModuleType("fla.ops.kda")
    ops_utils = types.ModuleType("fla.ops.utils")
    index = types.ModuleType("fla.ops.utils.index")
    utils = types.ModuleType("fla.utils")
    modules.FusedRMSNormGated = _ReferenceFusedRMSNormGated
    modules.ShortConvolution = _ReferenceShortConvolution
    kda.chunk_kda = _reference_kda
    kda.fused_recurrent_kda = _reference_kda
    index.prepare_lens_from_mask = _prepare_lens_from_mask
    index.prepare_cu_seqlens_from_mask = _prepare_cu_seqlens_from_mask
    utils.tensor_cache = _tensor_cache
    fla.modules = modules
    fla.ops = ops
    fla.utils = utils
    ops.kda = kda
    ops.utils = ops_utils
    ops_utils.index = index
    sys.modules.update(
        {
            "fla": fla,
            "fla.modules": modules,
            "fla.ops": ops,
            "fla.ops.kda": kda,
            "fla.ops.utils": ops_utils,
            "fla.ops.utils.index": index,
            "fla.utils": utils,
        }
    )


def _install_transformers_remote_code_compat() -> None:
    """Bridge API moves between Moonshot's snapshot and local Transformers main.

    Kimi-K3's cached remote code imports ``OutputRecorder`` from
    ``transformers.utils.generic``.  Current Transformers main moved the same
    dataclass to ``transformers.utils.output_capturing`` while retaining the
    rest of the API used by this snapshot.  Re-export it only in this oracle
    process so the reference continues to execute against the user's local
    Transformers checkout without modifying that checkout.
    """
    import transformers.utils.generic as generic

    if not hasattr(generic, "OutputRecorder"):
        from transformers.utils.output_capturing import OutputRecorder

        generic.OutputRecorder = OutputRecorder


def _load_remote_modules(remote_dir: Path):
    _install_reference_fla()
    _install_transformers_remote_code_compat()
    package_name = "_hierarchos_moonshot_kimi_k3"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(remote_dir)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    configuration = importlib.import_module(f"{package_name}.configuration_kimi_k3")
    modeling = importlib.import_module(f"{package_name}.modeling_kimi_linear")
    # Moonshot's snapshot predates the Transformers-main rename of the causal
    # mask argument from `input_embeds` to `inputs_embeds`. Keep the vendor
    # model unchanged and adapt the imported helper in this validation process.
    create_causal_mask = modeling.create_causal_mask

    def create_causal_mask_compat(*args, **kwargs):
        if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        # Current Transformers derives the same cache offset from
        # `past_key_values`/`position_ids`; the older helper accepted the
        # explicit vector as a separate keyword.
        kwargs.pop("cache_position", None)
        return create_causal_mask(*args, **kwargs)

    modeling.create_causal_mask = create_causal_mask_compat
    return configuration, modeling


def _deterministic_parameter_fill(model: nn.Module) -> None:
    """Fill all tiny-oracle parameters deterministically and avoid router ties."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            count = parameter.numel()
            if name.endswith("A_log"):
                values = torch.linspace(1.15, 1.75, count, dtype=torch.float32).log()
            elif name.endswith("dt_bias"):
                values = torch.linspace(-1.35, -0.85, count, dtype=torch.float32)
            elif name.endswith("e_score_correction_bias"):
                values = torch.linspace(-0.017, 0.019, count, dtype=torch.float32)
            elif name.endswith("norm.weight") or "layernorm.weight" in name or "res_norm.weight" in name:
                values = 1.0 + torch.linspace(-0.035, 0.035, count, dtype=torch.float32)
            else:
                digest = hashlib.sha256(name.encode("utf-8")).digest()
                scale = 0.018 + (digest[0] / 255.0) * 0.022
                offset = digest[1] % 31
                index = torch.arange(count, dtype=torch.int64)
                centered = ((index * 17 + offset) % 31).to(torch.float32) - 15.0
                values = centered * (scale / 31.0)
            parameter.copy_(values.reshape(parameter.shape).to(parameter.dtype))
        embedding = getattr(getattr(model, "model", None), "embed_tokens", None)
        if embedding is not None and embedding.padding_idx is not None:
            embedding.weight[embedding.padding_idx].zero_()


def _install_differentiable_moe_reference(modeling) -> None:
    """Enable source-equivalent Kimi routed-MoE math under autograd for parity.

    Moonshot's released inference module intentionally asserts in training mode
    and wraps its expert dispatch in ``no_grad``.  This replacement preserves
    the same router/top-k/expert/shared-expert equations while making the tiny
    validation fixture differentiable, so native backward/AdamW can be checked.
    """

    def gate_forward(self, hidden_states):
        bsz, seq_len, hidden = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden)
        logits = F.linear(flat.float(), self.weight.float(), None)
        if self.moe_router_activation_func == "sigmoid":
            scores = logits.sigmoid()
        elif self.moe_router_activation_func == "softmax":
            scores = logits.softmax(dim=1)
        else:
            raise NotImplementedError(self.moe_router_activation_func)
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        if self.num_expert_group > 1 and self.num_expert_group > self.topk_group:
            group_scores = scores_for_choice.view(
                bsz * seq_len, self.num_expert_group, -1
            ).topk(2, dim=-1)[0].sum(dim=-1)
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores).scatter(1, group_idx, 1)
            score_mask = group_mask.unsqueeze(-1).expand(
                bsz * seq_len,
                self.num_expert_group,
                self.num_experts // self.num_expert_group,
            ).reshape(bsz * seq_len, -1)
            scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        topk_idx = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weight = scores.gather(1, topk_idx)
        if self.top_k > 1 and self.moe_renormalize:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1.0e-20)
        return topk_idx, topk_weight * self.routed_scaling_factor

    def sparse_forward(self, hidden_states):
        identity = hidden_states
        original_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        routed = hidden_states.reshape(-1, hidden_states.shape[-1])
        if self.use_latent_moe:
            routed = self.routed_expert_down_proj(routed)
        expert_outputs = torch.stack([expert(routed) for expert in self.experts], dim=1)
        gather_index = topk_idx.unsqueeze(-1).expand(-1, -1, expert_outputs.shape[-1])
        selected = expert_outputs.gather(1, gather_index)
        output = (selected * topk_weight.to(selected.dtype).unsqueeze(-1)).sum(dim=1)
        if self.use_latent_moe:
            if self.latent_moe_use_norm:
                output = self.routed_expert_norm(output)
            output = self.routed_expert_up_proj(output)
        output = output.view(*original_shape)
        if self.config.num_shared_experts is not None:
            output = output + self.shared_experts(identity)
        return output

    modeling.KimiMoEGate.forward = gate_forward
    modeling.KimiSparseMoeBlock.forward = sparse_forward


def tiny_kimi_k3_model(
    *,
    training_reference: bool = False,
    schedule: str = "mixed",
) -> torch.nn.Module | None:
    """Build a deterministic tiny model from Moonshot's actual Kimi remote code.

    ``schedule`` selects the hybrid-attention boundary exercised by the fixture:
    ``kda`` uses four KDA layers, ``mla`` uses four full-attention layers, and
    ``mixed`` uses the production-style hybrid shape of three KDA layers followed
    by one MLA layer.
    """
    remote_dir = find_kimi_k3_remote_source()
    if remote_dir is None:
        return None
    configuration, modeling = _load_remote_modules(remote_dir)
    if training_reference:
        _install_differentiable_moe_reference(modeling)
    schedules = {
        "kda": ([1, 2, 3, 4], []),
        "mla": ([], [1, 2, 3, 4]),
        "mixed": ([1, 2, 3], [4]),
    }
    try:
        kda_layers, full_attn_layers = schedules[schedule]
    except KeyError as exc:
        raise ValueError(f"unsupported tiny Kimi-K3 schedule {schedule!r}") from exc
    config = configuration.KimiLinearConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        rms_norm_eps=1.0e-6,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        mla_use_nope=True,
        mla_use_output_gate=True,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 2,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
            "kda_layers": kda_layers,
            "full_attn_layers": full_attn_layers,
        },
        first_k_dense_replace=1,
        moe_intermediate_size=16,
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=1,
        routed_expert_hidden_size=8,
        latent_moe_use_norm=True,
        moe_router_activation_func="sigmoid",
        moe_renormalize=True,
        routed_scaling_factor=1.0,
        use_grouped_topk=False,
        num_expert_group=1,
        topk_group=1,
        attention_dropout=0.0,
        attn_res_block_size=2,
        tie_word_embeddings=False,
        use_cache=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = modeling.KimiLinearForCausalLM(config)
    # The released constructor prefers FlashAttention2.  CPU parity deliberately
    # uses the eager implementation from the same Moonshot module.
    model.config._attn_implementation = "eager"
    _deterministic_parameter_fill(model)
    return model
