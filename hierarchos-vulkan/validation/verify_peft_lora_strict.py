from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
TRANSFORMERS_ROOT = Path(r"C:\Users\User\transformers")
FAMILY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HIERARCHOS_PEFT_ORACLE_FAMILY", "gpt2")
SHARED_REFERENCE_FAMILIES = {
    "deepseek_v4", "phi4_multimodal_text", "phi3", "kimi_k25_text",
    "kimi_k3_kda", "kimi_k3_mla", "kimi_k3_text", "gpt_oss", "smollm3",
    "qwen2_5_gqa", "qwen2_5_sliding_tied", "qwen3_5_full", "qwen3_5_mixed",
    "qwen3_5_moe", "qwen4_exp_linear", "qwen4_exp_qsa", "qwen4_exp_ple",
    "qwen4_exp_mixed_ple", "mistral4", "minimax_m3_dense", "minimax_m3",
    "gemma4", "gemma3", "minimax_m2",
}
if FAMILY not in {"gpt2", "llama", "mixtral", "qwen3_next", "qwen3_5_linear", "falcon_h1_gated_after", "falcon_h1_gated_before", "falcon_h1_silu"} | SHARED_REFERENCE_FAMILIES:
    raise ValueError(f"unsupported PEFT oracle family: {FAMILY}")
FIXTURE_ROOT = ROOT / ".peft-oracle-fixtures" / f"{FAMILY}-strict"
BASE_DIR = FIXTURE_ROOT / "base"
ADAPTER_DIR = FIXTURE_ROOT / "adapter"
FIXTURE_JSON = FIXTURE_ROOT / "fixture.json"
NATIVE_BASE_JSON = FIXTURE_ROOT / "native-base.json"
NATIVE_PEFT_JSON = FIXTURE_ROOT / "native-peft.json"
HF_JSON = FIXTURE_ROOT / "hf.json"
TRAINING_JSON = FIXTURE_ROOT / "training.json"
NATIVE_STEP1_DIR = FIXTURE_ROOT / "native-step1"
NATIVE_TRAINED_DIR = FIXTURE_ROOT / "native-trained"
NATIVE_BASE_AFTER_DIR = FIXTURE_ROOT / "native-base-after"
NATIVE_GRADIENT_JSON = FIXTURE_ROOT / "native-gradients.json"
HF_STEP1_DIR = FIXTURE_ROOT / "hf-step1"
HF_TRAINED_DIR = FIXTURE_ROOT / "hf-trained"
TOLERANCE = 2.0e-7


def local_transformers_provenance():
    """Reject a cached/site-packages import instead of silently changing the oracle."""
    import transformers

    source = Path(transformers.__file__).resolve()
    expected = (TRANSFORMERS_ROOT / "src" / "transformers").resolve()
    if not source.is_relative_to(expected):
        raise RuntimeError(f"Transformers oracle must load from {expected}, got {source}")
    return {"transformers_root": str(TRANSFORMERS_ROOT),
            "transformers_source": str(source), "transformers_version": transformers.__version__}


def _run(args: list[str], *, cwd: Path = ROOT) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def _native_binary() -> Path:
    name = "hierarchos-vulkan-transformer-logits.exe" if os.name == "nt" else "hierarchos-vulkan-transformer-logits"
    binary = ROOT / "target" / "debug" / name
    if os.environ.get("HIERARCHOS_PEFT_USE_PREBUILT") != "1":
        _run(["cargo", "build", "--bin", "hierarchos-vulkan-transformer-logits", "-j", "1"])
    if not binary.is_file():
        raise RuntimeError(f"missing native validation binary: {binary}")
    return binary


def _training_binary() -> Path:
    name = "transformer_parity.exe" if os.name == "nt" else "transformer_parity"
    binary = ROOT / "target" / "debug" / name
    if os.environ.get("HIERARCHOS_PEFT_USE_PREBUILT") != "1":
        _run(["cargo", "build", "--bin", "transformer_parity", "-j", "1"])
    if not binary.is_file():
        raise RuntimeError(f"missing native validation binary: {binary}")
    return binary


def _max_abs(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise RuntimeError(f"length mismatch: native={len(left)} reference={len(right)}")
    return max((abs(a - b) for a, b in zip(left, right)), default=0.0)


def _tensor_map_max_abs(left, right):
    if set(left) != set(right):
        raise RuntimeError("tensor key mismatch")
    worst = (0.0, "")
    for key in sorted(left):
        delta = float((left[key].float() - right[key].float()).abs().max())
        worst = max(worst, (delta, key))
    return worst


def _reference_model():
    if FAMILY.startswith("falcon_h1_"):
        from verify_falcon_h1 import tiny
        # HF PEFT explicitly rejects Mamba out_proj/conv1d for falcon_h1.
        # Validate every other actual nn.Linear, including Mamba in_proj.
        return tiny(norm=FAMILY != "falcon_h1_silu", before=FAMILY == "falcon_h1_gated_before"), (
            r"model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
            r"feed_forward\.(?:gate_proj|up_proj|down_proj)|mamba\.in_proj)"
        )
    from transformers import (GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM,
                              MixtralConfig, MixtralForCausalLM, Qwen3NextConfig, Qwen3NextForCausalLM,
                              Qwen3_5ForCausalLM, Qwen3_5TextConfig)
    if FAMILY == "gpt2":
        return GPT2LMHeadModel(GPT2Config(
            vocab_size=32, n_positions=8, n_ctx=8, n_embd=16, n_layer=2, n_head=2,
            n_inner=32, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
            use_cache=False, tie_word_embeddings=True)), ["c_proj"]
    common = dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                  num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=16,
                  attention_dropout=0.0, use_cache=False, tie_word_embeddings=True,
                  attn_implementation="eager")
    if FAMILY == "llama":
        return LlamaForCausalLM(LlamaConfig(**common)), "all-linear"
    if FAMILY == "mixtral":
        return MixtralForCausalLM(MixtralConfig(**common, num_local_experts=2, num_experts_per_tok=2)), "all-linear"
    if FAMILY == "qwen3_5_linear":
        return Qwen3_5ForCausalLM(Qwen3_5TextConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            max_position_embeddings=16, linear_conv_kernel_dim=2,
            linear_key_head_dim=4, linear_value_head_dim=4,
            linear_num_key_heads=2, linear_num_value_heads=4,
            layer_types=["linear_attention"], use_cache=False,
            tie_word_embeddings=False, bos_token_id=1, eos_token_id=2,
        )), "all-linear"
    if FAMILY in SHARED_REFERENCE_FAMILIES:
        from verify_hf_logits import tiny_models
        model_by_name = {name: model for name, model, _ in tiny_models(training_reference=True)}
        base = model_by_name[FAMILY]
        # PEFT's generic ``all-linear`` selector can see modules which are not
        # ordinary native VulkanLinear nodes (multimodal towers, packed routed
        # experts, DeepSeek-V4's rank-changing o_a projection, indexer-only
        # projections, etc.).  Keep the strict oracle on the exact canonical
        # linear graph exposed by ``VulkanTransformer::peft_layer_linears``.
        # This is architecture discovery, not a numerical workaround: every
        # selected module must exist in both the HF module graph and the native
        # PEFT traversal, while non-representable operations stay explicit
        # exclusions in the registry/audit.
        if FAMILY == "deepseek_v4":
            # Compressor/indexer branches are conditional in the tiny V4
            # topology, so keep the two-step gradient oracle on projections
            # that are executed in every decoder layer.  The native registry
            # still enumerates the conditional compressor targets separately.
            targets = r"model\.layers\.\d+\.self_attn\.(?:q_a_proj|q_b_proj|kv_proj|o_b_proj)"
        elif FAMILY in {"phi4_multimodal_text", "phi3"}:
            targets = r"model\.layers\.\d+\.(?:self_attn\.(?:qkv_proj|o_proj)|mlp\.down_proj)"
        elif FAMILY == "kimi_k25_text":
            targets = (
                r"model\.language_model\.layers\.\d+\."
                r"(?:self_attn\.(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)|"
                r"mlp\.shared_experts\.(?:gate_proj|up_proj|down_proj))"
            )
        elif FAMILY in {"kimi_k3_kda", "kimi_k3_mla", "kimi_k3_text"}:
            # KimiLinear has a dedicated native KDA/MLA topology.  Start the
            # strict family oracle from the common output projection while the
            # registry separately records the additional targetable KDA/MLA
            # linears exposed by the native traversal.
            targets = ["o_proj"]
        elif FAMILY in {"minimax_m3_dense", "minimax_m3"}:
            # Indexer q_proj participates only in discrete top-k selection;
            # HF itself supplies no adjoint for this side path.
            targets = r"model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
        elif FAMILY in {"gpt_oss", "minimax_m2"}:
            targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif FAMILY.startswith("qwen4_exp_"):
            targets = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "out_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
                "gate_proj", "up_proj", "down_proj", "shared_expert_gate",
                "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight",
                "key_proj", "value_proj",
            ]
        else:
            targets = "all-linear"
        return base, targets
    return Qwen3NextForCausalLM(Qwen3NextConfig(
        **common, head_dim=8, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_conv_kernel_dim=3,
        num_experts=2, num_experts_per_tok=2, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, mlp_only_layers=[0, 1],
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={"rope_type":"default", "rope_theta":10000.0, "partial_rotary_factor":1.0},
    )), "all-linear"


def _native_lifecycle_targets(targets):
    """Use explicit native target names when the HF fixture uses one regex.

    PEFT interprets a string target_modules value as a full-match regex.
    transformer_parity --lora-target deliberately accepts repeated
    canonical/suffix names instead, so passing the regex text through that CLI
    would turn it into a literal suffix selector. The adapter-load portion of
    this oracle still exercises the exact HF regex. For the native-created
    lifecycle leg, spell out the same canonical linear subset so the exported
    adapter can be loaded back by both native Vulkan and HF PEFT.
    """
    if not isinstance(targets, str) or targets == "all-linear":
        return [targets] if isinstance(targets, str) else list(targets)
    if FAMILY == "deepseek_v4":
        return ["q_a_proj", "q_b_proj", "kv_proj", "o_b_proj"]
    if FAMILY in {"phi4_multimodal_text", "phi3"}:
        return ["qkv_proj", "o_proj", "down_proj"]
    if FAMILY == "kimi_k25_text":
        return [
            "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj",
            "o_proj", "gate_proj", "up_proj", "down_proj",
        ]
    base, _ = _reference_model()
    names = [name for name, module in base.named_modules()
             if isinstance(module, torch.nn.Linear) and re.fullmatch(targets, name)]
    if not names:
        raise RuntimeError(f"PEFT regex selected no native lifecycle targets: {targets}")
    return names


def _verify_lifecycle(binary, training_binary, input_ids, attention_mask, targets, reference_model_cls):
    from peft import PeftModel

    if os.environ.get("HIERARCHOS_PEFT_USE_PREBUILT") != "1":
        _run(["cargo", "build", "--bin", "peft_checkpoint", "-j", "1"])
    checkpoint = ROOT / "target" / "debug" / ("peft_checkpoint.exe" if os.name == "nt" else "peft_checkpoint")
    native_created = FIXTURE_ROOT / "native-created"
    target_args = [arg for target in _native_lifecycle_targets(targets)
                   for arg in ["--lora-target", target]]
    _run([str(training_binary), "--model", str(BASE_DIR), "--fixture", str(TRAINING_JSON),
          "--output", str(native_created), "--lora-rank", "2", "--lora-alpha", "4",
          *target_args, "--steps", "1"])

    def logits(model, adapter=None, label="logits"):
        output = FIXTURE_ROOT / f"{label}.json"
        args = [str(binary), "--model", str(model), "--fixture", str(FIXTURE_JSON), "--output", str(output)]
        if adapter is not None:
            args += ["--lora-adapter", str(adapter)]
        _run(args)
        return json.loads(output.read_text(encoding="utf-8"))["logits"]

    metrics = {}
    for label, adapter in [("hf-created", ADAPTER_DIR), ("native-created", native_created),
                           ("native-trained", NATIVE_TRAINED_DIR)]:
        reference_model = reference_model_cls.from_pretrained(
            BASE_DIR,
            attn_implementation="eager",
        )
        if FAMILY in {"kimi_k3_kda", "kimi_k3_mla", "kimi_k3_text"}:
            # Moonshot's released KimiLinear constructor intentionally forces
            # FlashAttention2 even when Transformers supplies an eager override.
            # The strict CPU oracle uses the same remote eager implementation by
            # resetting this dispatch field immediately after construction, just
            # like validation/kimi_k3_reference.py does for the original fixture.
            reference_model.config._attn_implementation = "eager"
        hf = PeftModel.from_pretrained(reference_model, adapter).eval()
        with torch.no_grad():
            reference = hf(input_ids=input_ids, attention_mask=attention_mask).logits.float().flatten().tolist()
        unmerged = logits(BASE_DIR, adapter, label + "-unmerged")
        metrics[label + "_reload_hf_max_abs"] = _max_abs(unmerged, reference)
        training_report = adapter / "training_report.json"
        if training_report.is_file():
            before_save = json.loads(training_report.read_text(encoding="utf-8"))["logits_after"]
            metrics[label + "_save_reload_max_abs"] = _max_abs(unmerged, before_save)
        merged_dir = FIXTURE_ROOT / (label + "-merged")
        restored_dir = FIXTURE_ROOT / (label + "-unmerged-base")
        for directory in (merged_dir, restored_dir):
            directory.mkdir()
            shutil.copy2(BASE_DIR / "config.json", directory / "config.json")
        _run([str(checkpoint), "merge", str(BASE_DIR / "model.safetensors"), str(adapter),
              str(merged_dir / "model.safetensors")])
        merged_logits = logits(merged_dir, label=label + "-merged")
        metrics[label + "_merged_unmerged_max_abs"] = _max_abs(merged_logits, unmerged)
        hf_merged = hf.merge_and_unload().eval()
        with torch.no_grad():
            merged_reference = hf_merged(input_ids=input_ids, attention_mask=attention_mask).logits.float().flatten().tolist()
        metrics[label + "_merged_hf_max_abs"] = _max_abs(merged_logits, merged_reference)
        _run([str(checkpoint), "unmerge", str(merged_dir / "model.safetensors"), str(adapter),
              str(restored_dir / "model.safetensors")])
        restored = logits(restored_dir, label=label + "-restored")
        original = json.loads(NATIVE_BASE_JSON.read_text(encoding="utf-8"))["logits"]
        metrics[label + "_unmerge_base_max_abs"] = _max_abs(restored, original)
    return metrics


def _saved_lora_key(parameter_name: str) -> str:
    return (
        parameter_name.replace(".lora_A.default.weight", ".lora_A.weight")
        .replace(".lora_B.default.weight", ".lora_B.weight")
    )


def _infer_first_step_adamw_gradient(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    *,
    learning_rate: float,
    eps: float,
) -> dict[str, torch.Tensor]:
    if set(before) != set(after):
        raise RuntimeError("native step-1 adapter key mismatch")
    inferred = {}
    for key in sorted(before):
        # With zero weight decay and zero-initialized Adam moments, the first
        # bias-corrected AdamW step is:
        #
        #   delta = lr * g / (abs(g) + eps)
        #
        # The native shader follows this standard form. Inverting the update
        # gives us an external gradient check without adding a validation-only
        # Python execution path to the Rust/Vulkan runtime.
        ratio = (before[key].float() - after[key].float()) / learning_rate
        if torch.any(ratio.abs() >= 1.0):
            raise RuntimeError(f"cannot invert first AdamW step for {key}: |delta/lr| >= 1")
        inferred[key] = eps * ratio / (1.0 - ratio.abs())
    return inferred


def _fill_adapter(model) -> None:
    cursor = 1
    with torch.no_grad():
        for module_name, module in model.named_modules():
            if not hasattr(module, "lora_A") or "default" not in module.lora_A:
                continue
            a = module.lora_A["default"].weight
            b = module.lora_B["default"].weight
            a_values = torch.arange(cursor, cursor + a.numel(), dtype=torch.float32).reshape_as(a)
            cursor += a.numel()
            b_values = torch.arange(cursor, cursor + b.numel(), dtype=torch.float32).reshape_as(b)
            cursor += b.numel()
            a.copy_((a_values.remainder(17) - 8.0) * 0.0025)
            b.copy_((b_values.remainder(19) - 9.0) * 0.0020)
    if cursor == 1:
        raise RuntimeError("PEFT did not attach any LoRA modules")


def main() -> None:
    if not TRANSFORMERS_ROOT.exists():
        raise RuntimeError(f"local Transformers checkout is missing: {TRANSFORMERS_ROOT}")
    validation_pydeps = os.environ.get("HIERARCHOS_VALIDATION_PYDEPS")
    if validation_pydeps:
        sys.path.insert(0, str(Path(validation_pydeps).expanduser().resolve()))
    else:
        local_oracle_pydeps = ROOT / ".k3-oracle-pydeps"
        if local_oracle_pydeps.is_dir():
            sys.path.insert(0, str(local_oracle_pydeps.resolve()))
        validation_pydeps_dir = ROOT / ".validation-pydeps"
        if validation_pydeps_dir.is_dir():
            sys.path.insert(0, str(validation_pydeps_dir.resolve()))
    sys.path.insert(0, str(TRANSFORMERS_ROOT / "src"))

    oracle_provenance = local_transformers_provenance()
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file

    if not FIXTURE_ROOT.resolve().is_relative_to((ROOT / ".peft-oracle-fixtures").resolve()):
        raise RuntimeError("fixture cleanup escaped the PEFT fixture directory")
    if FIXTURE_ROOT.exists():
        # A new oracle run must not inherit stale reports, but previous traces,
        # saved-module evidence and checkpoints are useful continuation state.
        # Rename within the validated fixture parent instead of deleting them.
        previous_run = FIXTURE_ROOT.with_name(FIXTURE_ROOT.name + ".prior-" + str(time.time_ns()))
        FIXTURE_ROOT.rename(previous_run)
        print(f"[peft-oracle] preserved previous fixture at {previous_run}", flush=True)
    BASE_DIR.mkdir(parents=True)
    ADAPTER_DIR.mkdir(parents=True)

    torch.manual_seed(1234)
    torch.set_num_threads(1)
    base, target_modules = _reference_model()
    # Match the independent base training fixture: stochastic router jitter
    # must be disabled on BOTH sides unless replaying identical Philox draws.
    for module in base.modules():
        for field in ("jitter_noise", "moe_jitter_eps"):
            if hasattr(module, field):
                setattr(module, field, 0.0)
    if hasattr(base.config, "router_jitter_noise"):
        base.config.router_jitter_noise = 0.0
    base.eval()
    # Keep the fixture in Transformers' current normalized parameter graph.
    # Some families (notably Mixtral) have a save-time reverse conversion to
    # legacy Hub tensor names/shapes; comparing that serialized legacy layout
    # directly with the native normalized export produces a false frozen-base
    # key mismatch even when every parameter value is unchanged.
    if FAMILY in {"kimi_k3_kda", "kimi_k3_mla", "kimi_k3_text"}:
        # Reuse the strict K3 oracle's normalized serializer. Moonshot's
        # released remote class still exposes the pre-main list form of
        # `_tied_weights_keys`, while the local Transformers main checkout
        # expects a mapping during save_pretrained(). The tiny K3 fixtures have
        # untied embeddings, so saving the exact state_dict plus config is the
        # equivalent normalized checkpoint and avoids mutating either upstream
        # source tree.
        from verify_hf_logits import save_reference_checkpoint

        save_reference_checkpoint(FAMILY, base, BASE_DIR)
    else:
        base.save_pretrained(
            BASE_DIR,
            safe_serialization=True,
            save_original_format=False,
        )

    peft_config = LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base, peft_config).eval()
    _fill_adapter(peft_model)
    peft_model.save_pretrained(ADAPTER_DIR, safe_serialization=True)

    input_ids = torch.tensor([[1, 7, 3, 11]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    peft_model.disable_adapter_layers()
    with torch.no_grad():
        hf_base = peft_model(input_ids=input_ids, attention_mask=attention_mask).logits.float().cpu().reshape(-1).tolist()
    peft_model.enable_adapter_layers()
    with torch.no_grad():
        hf_peft = peft_model(input_ids=input_ids, attention_mask=attention_mask).logits.float().cpu().reshape(-1).tolist()

    FIXTURE_JSON.write_text(
        json.dumps(
            {
                "batch_size": 1,
                "seq_len": 4,
                "input_ids": input_ids.reshape(-1).tolist(),
                "attention_mask": attention_mask.float().reshape(-1).tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    HF_JSON.write_text(json.dumps({"base_logits": hf_base, "peft_logits": hf_peft}), encoding="utf-8")

    binary = _native_binary()
    _run([str(binary), "--model", str(BASE_DIR), "--fixture", str(FIXTURE_JSON), "--output", str(NATIVE_BASE_JSON)])
    _run(
        [
            str(binary),
            "--model",
            str(BASE_DIR),
            "--fixture",
            str(FIXTURE_JSON),
            "--lora-adapter",
            str(ADAPTER_DIR),
            "--output",
            str(NATIVE_PEFT_JSON),
        ]
    )

    native_base = json.loads(NATIVE_BASE_JSON.read_text(encoding="utf-8"))["logits"]
    native_peft = json.loads(NATIVE_PEFT_JSON.read_text(encoding="utf-8"))["logits"]
    base_drift = _max_abs(native_base, hf_base)
    peft_drift = _max_abs(native_peft, hf_peft)
    delta_drift = _max_abs(
        [p - b for p, b in zip(native_peft, native_base)],
        [p - b for p, b in zip(hf_peft, hf_base)],
    )

    targets = torch.tensor([[7, 3, 11, -100]], dtype=torch.long)
    hyper = {
        "learning_rate": 1.0e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1.0e-3,
        "weight_decay": 0.0,
    }
    TRAINING_JSON.write_text(
        json.dumps(
            {
                "batch_size": 1,
                "seq_len": 4,
                "input_ids": input_ids.reshape(-1).tolist(),
                "targets": [int(t) if t >= 0 else 0xFFFFFFFF for t in targets.reshape(-1)],
                "attention_mask": attention_mask.float().reshape(-1).tolist(),
                "loss_weights": targets.ne(-100).float().reshape(-1).tolist(),
                **hyper,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    peft_model.train()
    trainable_parameters = [
        (name, parameter)
        for name, parameter in peft_model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("PEFT oracle exposes no trainable adapter parameters")
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable_parameters],
        lr=hyper["learning_rate"],
        betas=(hyper["beta1"], hyper["beta2"]),
        eps=hyper["eps"],
        weight_decay=hyper["weight_decay"],
    )
    hf_gradients = {}
    hf_losses = []
    for step_index in range(2):
        optimizer.zero_grad(set_to_none=True)
        outputs = peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        loss = outputs.loss
        hf_losses.append(float(loss.detach().cpu()))
        loss.backward()
        if step_index == 0:
            for name, parameter in trainable_parameters:
                if parameter.grad is None:
                    raise RuntimeError(f"PEFT trainable parameter has no gradient: {name}")
                saved_key = _saved_lora_key(name)
                if saved_key == name:
                    raise RuntimeError(
                        f"strict LoRA fixture unexpectedly exposed a non-LoRA trainable parameter: {name}"
                    )
                hf_gradients[saved_key] = parameter.grad.detach().float().cpu().clone()
        optimizer.step()
        if step_index == 0:
            peft_model.save_pretrained(HF_STEP1_DIR, safe_serialization=True)
    peft_model.save_pretrained(HF_TRAINED_DIR, safe_serialization=True)

    training_binary = _training_binary()
    _run(
        [
            str(training_binary),
            "--model",
            str(BASE_DIR),
            "--fixture",
            str(TRAINING_JSON),
            "--output",
            str(NATIVE_STEP1_DIR),
            "--training-checkpoint",
            str(FIXTURE_ROOT / "native-resume-state"),
            "--base-output",
            str(NATIVE_BASE_AFTER_DIR),
            "--gradient-output",
            str(NATIVE_GRADIENT_JSON),
            "--lora-adapter",
            str(ADAPTER_DIR),
            "--steps",
            "1",
        ]
    )
    _run(
        [
            str(training_binary),
            "--model",
            str(BASE_DIR),
            "--fixture",
            str(TRAINING_JSON),
            "--output",
            str(NATIVE_TRAINED_DIR),
            "--lora-adapter",
            str(ADAPTER_DIR),
            "--steps",
            "2",
        ]
    )

    # Check the frozen-base contract before optimizer-resume validation so a
    # base mutation is reported directly instead of surfacing later as an
    # opaque checkpoint-identity mismatch.
    base_before = load_file(str(BASE_DIR / "model.safetensors"))
    base_after = load_file(str(NATIVE_BASE_AFTER_DIR / "model.safetensors"))
    frozen_base_drift, frozen_base_worst = _tensor_map_max_abs(base_before, base_after)
    print(
        json.dumps(
            {
                "pre_resume_frozen_base_max_abs": frozen_base_drift,
                "pre_resume_frozen_base_worst": frozen_base_worst,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    initial_adapter = load_file(str(ADAPTER_DIR / "adapter_model.safetensors"))
    resumed_dir = FIXTURE_ROOT / "native-resumed"
    _run([str(training_binary), "--model", str(BASE_DIR), "--fixture", str(TRAINING_JSON),
          "--output", str(resumed_dir), "--resume-lora", str(FIXTURE_ROOT / "native-resume-state"),
          "--steps", "1"])
    native_step1 = load_file(str(NATIVE_STEP1_DIR / "adapter_model.safetensors"))
    native_step2 = load_file(str(NATIVE_TRAINED_DIR / "adapter_model.safetensors"))
    hf_step1 = load_file(str(HF_STEP1_DIR / "adapter_model.safetensors"))
    hf_step2 = load_file(str(HF_TRAINED_DIR / "adapter_model.safetensors"))
    resumed = load_file(str(resumed_dir / "adapter_model.safetensors"))
    resume_drift, _ = _tensor_map_max_abs(resumed, native_step2)
    resume_hf_drift, _ = _tensor_map_max_abs(resumed, hf_step2)

    step1_drift, step1_worst = _tensor_map_max_abs(native_step1, hf_step1)
    step2_drift, step2_worst = _tensor_map_max_abs(native_step2, hf_step2)
    native_gradient_json = json.loads(NATIVE_GRADIENT_JSON.read_text(encoding="utf-8"))
    if set(native_gradient_json) != set(hf_gradients):
        missing_native = sorted(set(hf_gradients) - set(native_gradient_json))
        extra_native = sorted(set(native_gradient_json) - set(hf_gradients))
        raise RuntimeError(
            "native/HF gradient key mismatch: "
            f"missing_native={missing_native} extra_native={extra_native}"
        )
    native_gradients = {
        key: torch.tensor(values, dtype=torch.float32).reshape_as(hf_gradients[key])
        for key, values in native_gradient_json.items()
    }
    gradient_drift, gradient_worst = _tensor_map_max_abs(
        native_gradients,
        hf_gradients,
    )

    trainable_key_set_match = set(hf_gradients) == set(initial_adapter)
    lifecycle = _verify_lifecycle(
        binary,
        training_binary,
        input_ids,
        attention_mask,
        target_modules,
        type(base),
    )

    report = {
        **oracle_provenance,
        "family": FAMILY,
        "peft_tolerance": TOLERANCE,
        "base_max_abs": base_drift,
        "peft_max_abs": peft_drift,
        "lora_delta_max_abs": delta_drift,
        "hf_losses": hf_losses,
        "step1_adapter_max_abs": step1_drift,
        "step1_adapter_worst": step1_worst,
        "two_step_adamw_max_abs": step2_drift,
        "two_step_adamw_worst": step2_worst,
        "gradient_max_abs": gradient_drift,
        "gradient_worst": gradient_worst,
        "frozen_base_max_abs": frozen_base_drift,
        "frozen_base_worst": frozen_base_worst,
        "trainable_key_set_match": trainable_key_set_match,
        "resume_uninterrupted_max_abs": resume_drift,
        "resume_hf_max_abs": resume_hf_drift,
        "lifecycle": lifecycle,
        "pass": (
            base_drift <= TOLERANCE
            and peft_drift <= TOLERANCE
            and delta_drift <= TOLERANCE
            and step1_drift <= TOLERANCE
            and step2_drift <= TOLERANCE
            and gradient_drift <= TOLERANCE
            and frozen_base_drift == 0.0
            and trainable_key_set_match
            and resume_drift <= TOLERANCE
            and resume_hf_drift <= TOLERANCE
            and all(value <= TOLERANCE for value in lifecycle.values())
        ),
    }
    (FIXTURE_ROOT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
