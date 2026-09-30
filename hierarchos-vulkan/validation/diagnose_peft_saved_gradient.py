"""One-step backward diagnostic; never writes full PEFT qualification reports."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import tempfile

import numpy as np
import torch
import verify_peft_multi_adapter as multi
import verify_peft_lora_strict as strict
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family")
    parser.add_argument("--module", default="model.embed_tokens")
    parser.add_argument("--adapter", choices=["a", "b"], default="a")
    parser.add_argument(
        "--gemma4-internals",
        action="store_true",
        help="capture non-mutating Gemma4 layer-0 projection/norm boundaries for diagnosis",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = tempfile.mkdtemp(prefix="gradient-probe-", dir=strict.FIXTURE_ROOT)
    from pathlib import Path
    output = Path(output)
    # Full saved-module qualification now uses unique short scratch paths.
    # Recreate its deterministic payload instead of depending on obsolete
    # multi-saved-* directories left by a different run or model checkpoint.
    adapter = output / "adapter"
    multi.SAVED = args.module
    multi._write_adapter(adapter, 101 if args.adapter == "a" else 202)
    fixture = json.loads(multi.FIXTURE_JSON.read_text())
    ids = torch.tensor(fixture["input_ids"]).reshape(fixture["batch_size"], fixture["seq_len"])
    mask = torch.tensor(fixture["attention_mask"]).reshape_as(ids)
    model = PeftModel.from_pretrained(multi.load_base(), adapter, is_trainable=True).train()
    gemma4_internal = {}
    gemma4_modules = {}
    gemma4_handles = []
    gemma4_rope_calls = []
    gemma4_modeling = None
    gemma4_apply_rotary_original = None

    def capture_gemma4_boundary(label):
        def hook(_module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            gemma4_internal[label] = value.detach().float().cpu()
            if inputs:
                gemma4_internal[label + ".input_value"] = inputs[0].detach().float().cpu()
            if value.requires_grad:
                value.register_hook(
                    lambda grad, key=label: gemma4_internal.__setitem__(
                        key + ".grad_output", grad.detach().float().cpu()
                    )
                )
            if inputs and inputs[0].requires_grad:
                inputs[0].register_hook(
                    lambda grad, key=label: gemma4_internal.__setitem__(
                        key + ".grad_input", grad.detach().float().cpu()
                    )
                )
        return hook

    if args.gemma4_internals:
        if args.family != "gemma4":
            raise SystemExit("--gemma4-internals is only valid for the gemma4 diagnostic")
        import transformers.models.gemma4.modeling_gemma4 as gemma4_modeling_module

        gemma4_modeling = gemma4_modeling_module
        gemma4_apply_rotary_original = gemma4_modeling.apply_rotary_pos_emb

        def capture_gemma4_rotary(x, cos, sin, unsqueeze_dim=1):
            value = gemma4_apply_rotary_original(x, cos, sin, unsqueeze_dim=unsqueeze_dim)
            entry = {
                "input": x.detach().float().cpu(),
                "cos": cos.detach().float().cpu(),
                "sin": sin.detach().float().cpu(),
                "output": value.detach().float().cpu(),
                "unsqueeze_dim": unsqueeze_dim,
            }
            if value.requires_grad:
                value.register_hook(
                    lambda grad, captured=entry: captured.__setitem__(
                        "grad_output", grad.detach().float().cpu()
                    )
                )
            if x.requires_grad:
                x.register_hook(
                    lambda grad, captured=entry: captured.__setitem__(
                        "grad_input", grad.detach().float().cpu()
                    )
                )
            gemma4_rope_calls.append(entry)
            return value

        gemma4_modeling.apply_rotary_pos_emb = capture_gemma4_rotary
        module_map = dict(model.named_modules())
        layer_indices = sorted(
            {
                int(name.split("model.layers.", 1)[1].split(".", 1)[0])
                for name in module_map
                if "model.layers." in name and name.endswith(".self_attn.q_norm")
            }
        )
        if not layer_indices:
            raise RuntimeError("could not discover Gemma4 decoder layers for internal tracing")
        suffixes = {}
        for layer_index in layer_indices:
            prefix = f"layer{layer_index}"
            # The input norm is a ModulesToSaveWrapper only when this diagnostic
            # is probing that exact norm. Embedding/head probes leave it as the
            # ordinary Gemma4 module. Prefer the active replacement when present.
            suffixes.update(
                {
                    f"{prefix}.decoder": (f"model.layers.{layer_index}",),
                    f"{prefix}.input_layernorm": (
                        f"model.layers.{layer_index}.input_layernorm.modules_to_save.default",
                        f"model.layers.{layer_index}.input_layernorm",
                    ),
                    f"{prefix}.post_attention_layernorm": (
                        f"model.layers.{layer_index}.post_attention_layernorm.modules_to_save.default",
                        f"model.layers.{layer_index}.post_attention_layernorm",
                    ),
                    f"{prefix}.pre_feedforward_layernorm": (
                        f"model.layers.{layer_index}.pre_feedforward_layernorm.modules_to_save.default",
                        f"model.layers.{layer_index}.pre_feedforward_layernorm",
                    ),
                    f"{prefix}.post_feedforward_layernorm": (
                        f"model.layers.{layer_index}.post_feedforward_layernorm.modules_to_save.default",
                        f"model.layers.{layer_index}.post_feedforward_layernorm",
                    ),
                    f"{prefix}.q_proj": (f"model.layers.{layer_index}.self_attn.q_proj",),
                    f"{prefix}.k_proj": (f"model.layers.{layer_index}.self_attn.k_proj",),
                    f"{prefix}.v_proj": (f"model.layers.{layer_index}.self_attn.v_proj",),
                    f"{prefix}.q_norm": (f"model.layers.{layer_index}.self_attn.q_norm",),
                    f"{prefix}.k_norm": (f"model.layers.{layer_index}.self_attn.k_norm",),
                    f"{prefix}.v_norm": (f"model.layers.{layer_index}.self_attn.v_norm",),
                    f"{prefix}.o_proj": (f"model.layers.{layer_index}.self_attn.o_proj",),
                    f"{prefix}.mlp": (f"model.layers.{layer_index}.mlp",),
                    f"{prefix}.mlp.gate_proj": (f"model.layers.{layer_index}.mlp.gate_proj",),
                    f"{prefix}.mlp.up_proj": (f"model.layers.{layer_index}.mlp.up_proj",),
                    f"{prefix}.mlp.down_proj": (f"model.layers.{layer_index}.mlp.down_proj",),
                }
            )
        suffixes["final_norm"] = ("model.norm",)
        for label, candidate_suffixes in suffixes.items():
            matches = []
            suffix = candidate_suffixes[-1]
            for candidate_suffix in candidate_suffixes:
                matches = [
                    (name, module)
                    for name, module in module_map.items()
                    if name.endswith(candidate_suffix)
                ]
                if matches:
                    suffix = candidate_suffix
                    break
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one Gemma4 module ending in {suffix!r}, found "
                    f"{[name for name, _ in matches]}"
                )
            gemma4_handles.append(
                matches[0][1].register_forward_hook(capture_gemma4_boundary(label))
            )
            gemma4_modules[label] = matches[0][1]
    head_capture = {}
    head_hook = None
    embedding_capture = {}
    embedding_hook = None
    if args.module == "lm_head":
        for module_name, module in model.named_modules():
            if module_name.endswith("lm_head.modules_to_save.default"):
                def capture_head(_module, inputs, output):
                    head_capture["input"] = inputs[0].detach().float().cpu()
                    output.register_hook(
                        lambda grad: head_capture.__setitem__(
                            "grad_output", grad.detach().float().cpu()
                        )
                    )
                head_hook = module.register_forward_hook(capture_head)
                head_capture["module_name"] = module_name
                head_capture["weight_shape"] = tuple(module.weight.shape)
                break
    if args.module.endswith("embed_tokens") or args.module.endswith("wte"):
        for module_name, module in model.named_modules():
            if module_name.endswith(f"{args.module}.modules_to_save.default"):
                def capture_embedding(_module, _inputs, output):
                    embedding_capture["output"] = output.detach().float().cpu()
                    output.register_hook(
                        lambda grad: embedding_capture.__setitem__(
                            "grad_output", grad.detach().float().cpu()
                        )
                    )
                embedding_hook = module.register_forward_hook(capture_embedding)
                embedding_capture["module_name"] = module_name
                embedding_capture["scale"] = float(
                    getattr(module, "embed_scale", torch.tensor(1.0)).detach().float().cpu()
                )
                break
    model(input_ids=ids, attention_mask=mask, labels=ids).loss.backward()
    if gemma4_modeling is not None and gemma4_apply_rotary_original is not None:
        gemma4_modeling.apply_rotary_pos_emb = gemma4_apply_rotary_original
    for handle in gemma4_handles:
        handle.remove()
    if head_hook is not None:
        head_hook.remove()
    if embedding_hook is not None:
        embedding_hook.remove()
    expected = {
        strict._saved_lora_key(name).replace(".modules_to_save.default.", "."): param.grad.detach().clone()
        for name, param in model.named_parameters() if param.requires_grad
    }
    native_binary = strict._training_binary()
    # Hash source and embedded-artifact inputs independently. Equal timestamps
    # or a Cargo rebuild do not prove that GLSL was compiled into its SPIR-V.
    artifact_paths = [native_binary]
    for stem in ["transformer_rope_forward", "transformer_rope_backward",
                 "transformer_attention_forward", "transformer_attention_backward",
                 "transformer_rms_norm_forward", "transformer_rms_norm_input_grad",
                 "linear_forward", "linear_forward_lane4", "linear_weight_grad"]:
        artifact_paths.extend(strict.ROOT / "shaders" / (stem + suffix)
                              for suffix in [".comp", ".spv"])
    artifact_paths.extend(strict.ROOT / "shaders" / name for name in
                          ["fp32_sleef_trig.glsl", "fp32_sleef_exp.glsl", "fp32_sqrt_recip.glsl"])
    artifact_hashes = {str(path.relative_to(strict.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in artifact_paths if path.is_file()}
    multi._run([
        str(native_binary), "--model", str(multi.BASE_DIR),
        "--fixture", str(strict.FIXTURE_ROOT / "training.json"),
        "--lora-adapter", str(adapter), "--steps", "1", "--output", str(output / "native"),
        "--gradient-output", str(output / "gradients.json"),
        "--debug-first-layer-output", str(output / "trace.json"),
    ])
    actual = {
        key: torch.tensor(value).reshape_as(expected[key])
        for key, value in json.loads((output / "gradients.json").read_text()).items()
    }
    drift, worst = strict._tensor_map_max_abs(actual, expected)
    diff = (actual[worst] - expected[worst]).abs().flatten()
    top_values, top_indices = torch.topk(diff, min(16, diff.numel()))
    flat_actual = actual[worst].flatten()
    flat_expected = expected[worst].flatten()
    top_differences = [
        {
            "flat_index": int(index),
            "actual": float(flat_actual[index]),
            "expected": float(flat_expected[index]),
            "abs_diff": float(value),
        }
        for value, index in zip(top_values, top_indices)
    ]
    trace = json.loads((output / "trace.json").read_text())
    if args.gemma4_internals:
        # Preserve the actual oracle tensors for offline operand-matched replay.
        # Replaying derivatives later must not mutate the comparison snapshot.
        torch.save({"boundaries": gemma4_internal, "rope": gemma4_rope_calls,
                    "weights": {name: value.detach().clone() for name, value in model.named_parameters()},
                    "gradients": expected}, output / "oracle-tensors.pt")
    gemma4_comparisons = None
    gemma4_attention_replay = None
    gemma4_rope_replay = None
    gemma4_o_proj_replay = None
    gemma4_forward_residual_replay = None
    if args.gemma4_internals:
        native_keys = {}
        for layer_index in layer_indices:
            label_prefix = f"layer{layer_index}"
            trace_prefix = f"layers.{layer_index}"
            layer_grad_output_key = (
                f"layers.{layer_index + 1}.grad_input"
                if layer_index + 1 < len(layer_indices)
                else "grad_final_norm_input"
            )
            native_keys.update(
                {
                    f"{label_prefix}.decoder": (
                        f"{trace_prefix}.output",
                        layer_grad_output_key,
                        f"{trace_prefix}.grad_input",
                    ),
                    f"{label_prefix}.input_layernorm": (
                        f"{trace_prefix}.norm_output",
                        f"{trace_prefix}.grad_norm_output",
                        None,
                    ),
                    f"{label_prefix}.q_proj": (
                        f"{trace_prefix}.q_linear_output",
                        f"{trace_prefix}.grad_q_linear_output",
                        None,
                    ),
                    f"{label_prefix}.k_proj": (
                        f"{trace_prefix}.k_linear_output",
                        f"{trace_prefix}.grad_k_linear_output",
                        None,
                    ),
                    f"{label_prefix}.v_proj": (
                        f"{trace_prefix}.v_linear_output",
                        f"{trace_prefix}.grad_v_linear_output",
                        None,
                    ),
                    f"{label_prefix}.q_norm": (
                        f"{trace_prefix}.q_norm_output",
                        f"{trace_prefix}.grad_q_norm_output",
                        f"{trace_prefix}.grad_q_linear_output",
                    ),
                    f"{label_prefix}.k_norm": (
                        f"{trace_prefix}.k_norm_output",
                        f"{trace_prefix}.grad_k_norm_output",
                        f"{trace_prefix}.grad_k_linear_output",
                    ),
                    f"{label_prefix}.v_norm": (
                        f"{trace_prefix}.v_norm_output",
                        f"{trace_prefix}.grad_v_norm_output",
                        f"{trace_prefix}.grad_v_linear_output",
                    ),
                    f"{label_prefix}.o_proj": (
                        f"{trace_prefix}.attention_projection",
                        None,
                        f"{trace_prefix}.grad_attention_output",
                    ),
                    f"{label_prefix}.post_attention_layernorm": (
                        f"{trace_prefix}.post_attention_norm_output",
                        f"{trace_prefix}.grad_residual1",
                        f"{trace_prefix}.grad_post_attention_norm",
                    ),
                    f"{label_prefix}.pre_feedforward_layernorm": (
                        f"{trace_prefix}.mlp_norm_output",
                        f"{trace_prefix}.grad_mlp_norm_output",
                        f"{trace_prefix}.grad_residual1",
                    ),
                    f"{label_prefix}.post_feedforward_layernorm": (
                        f"{trace_prefix}.post_mlp_norm_output",
                        f"layers.{layer_index + 1}.grad_input"
                        if layer_index + 1 < len(layer_indices)
                        else "grad_final_norm_input",
                        f"{trace_prefix}.grad_post_mlp_norm",
                    ),
                    f"{label_prefix}.mlp": (
                        f"{trace_prefix}.mlp_output",
                        None,
                        f"{trace_prefix}.grad_mlp_norm_output",
                    ),
                    f"{label_prefix}.mlp.gate_proj": (
                        f"{trace_prefix}.mlp_gate_output",
                        f"{trace_prefix}.grad_mlp_gate_output",
                        f"{trace_prefix}.grad_mlp_gate_input",
                    ),
                    f"{label_prefix}.mlp.up_proj": (
                        f"{trace_prefix}.mlp_up_output",
                        f"{trace_prefix}.grad_mlp_up_output",
                        f"{trace_prefix}.grad_mlp_up_input",
                    ),
                    f"{label_prefix}.mlp.down_proj": (
                        f"{trace_prefix}.mlp_output",
                        f"{trace_prefix}.grad_mlp_output",
                        f"{trace_prefix}.grad_mlp_product",
                    ),
                }
            )
        native_keys["final_norm"] = (
            "final_norm_output",
            "grad_final_norm_output",
            "grad_final_norm_input",
        )
        gemma4_comparisons = {}
        for label, (value_key, grad_output_key, grad_input_key) in native_keys.items():
            row = {}
            hf_value = gemma4_internal.get(label)
            if hf_value is not None and value_key in trace:
                native_value = torch.tensor(trace[value_key], dtype=torch.float32).reshape_as(hf_value)
                row["value_max_abs"] = float((native_value - hf_value).abs().max())
            hf_grad_output = gemma4_internal.get(label + ".grad_output")
            if (
                hf_grad_output is not None
                and grad_output_key is not None
                and grad_output_key in trace
            ):
                native_grad_output = torch.tensor(
                    trace[grad_output_key], dtype=torch.float32
                ).reshape_as(hf_grad_output)
                row["grad_output_max_abs"] = float(
                    (native_grad_output - hf_grad_output).abs().max()
                )
            hf_grad_input = gemma4_internal.get(label + ".grad_input")
            if (
                hf_grad_input is not None
                and grad_input_key is not None
                and grad_input_key in trace
            ):
                native_grad_input = torch.tensor(
                    trace[grad_input_key], dtype=torch.float32
                ).reshape_as(hf_grad_input)
                row["grad_input_max_abs"] = float(
                    (native_grad_input - hf_grad_input).abs().max()
                )
            if label.endswith(".o_proj"):
                hf_input = gemma4_internal.get(label + ".input_value")
                layer_index = int(label.split(".", 1)[0].removeprefix("layer"))
                attention_key = f"layers.{layer_index}.attention_output"
                if hf_input is not None and attention_key in trace:
                    native_input = torch.tensor(
                        trace[attention_key], dtype=torch.float32
                    ).reshape_as(hf_input)
                    row["input_max_abs"] = float(
                        (native_input - hf_input).abs().max()
                    )
            if label.endswith(".mlp.down_proj"):
                hf_input = gemma4_internal.get(label + ".input_value")
                layer_index = int(label.split(".", 1)[0].removeprefix("layer"))
                product_key = f"layers.{layer_index}.mlp_product"
                if hf_input is not None and product_key in trace:
                    native_input = torch.tensor(
                        trace[product_key], dtype=torch.float32
                    ).reshape_as(hf_input)
                    row["input_max_abs"] = float(
                        (native_input - hf_input).abs().max()
                    )
            gemma4_comparisons[label] = row

        # Replay every decoder layer's Q/K/V RMSNorm backward from the exact
        # native operands. The older detailed replay below only covers layer 0,
        # which can hide a tiny layer-1 seed later amplified by per-head norms.
        for layer_index in layer_indices:
            trace_prefix = f"layers.{layer_index}"
            for short_label in ("q_norm", "k_norm", "v_norm"):
                label = f"layer{layer_index}.{short_label}"
                prefix = short_label[0]
                input_key = f"{trace_prefix}.{prefix}_linear_output"
                grad_output_key = f"{trace_prefix}.grad_{prefix}_norm_output"
                grad_input_key = f"{trace_prefix}.grad_{prefix}_linear_output"
                if not all(key in trace for key in (input_key, grad_output_key, grad_input_key)):
                    continue
                module = gemma4_modules[label]
                hf_input = gemma4_internal.get(label + ".input_value")
                hf_grad_output = gemma4_internal.get(label + ".grad_output")
                hf_grad_input = gemma4_internal.get(label + ".grad_input")
                if hf_input is None or hf_grad_output is None or hf_grad_input is None:
                    continue
                native_input = torch.tensor(trace[input_key], dtype=torch.float32).reshape_as(hf_input)
                native_grad_output = torch.tensor(
                    trace[grad_output_key], dtype=torch.float32
                ).reshape_as(hf_grad_output)
                native_grad_input = torch.tensor(
                    trace[grad_input_key], dtype=torch.float32
                ).reshape_as(hf_grad_input)
                replay_input = native_input.detach().clone().requires_grad_(True)
                replay_weight = (
                    module.weight.detach().float().cpu()
                    if hasattr(module, "weight")
                    else torch.ones(hf_input.shape[-1], dtype=torch.float32)
                )
                replay_mean = replay_input.pow(2).mean(-1, keepdim=True) + module.eps
                replay_rstd = torch.pow(replay_mean, -0.5)
                replay_output = replay_input * replay_rstd * replay_weight
                (replay_output * native_grad_output).sum().backward()
                gemma4_comparisons[label]["native_operand_rms_replay"] = {
                    "input_vs_hf_max_abs": float((native_input - hf_input).abs().max()),
                    "grad_output_vs_hf_max_abs": float(
                        (native_grad_output - hf_grad_output).abs().max()
                    ),
                    "native_grad_input_vs_torch_max_abs": float(
                        (native_grad_input - replay_input.grad).abs().max()
                    ),
                    "torch_native_operands_vs_hf_grad_input_max_abs": float(
                        (replay_input.grad - hf_grad_input).abs().max()
                    ),
                }

        # Replay Gemma4's two residual joins from the exact operands captured
        # on each side.  The HF graph spells both joins as `residual + branch`;
        # this separates a branch-value mismatch from a native add mismatch.
        gemma4_forward_residual_replay = {}
        for layer_index in layer_indices:
            prefix = f"layer{layer_index}"
            trace_prefix = f"layers.{layer_index}"
            decoder_input = gemma4_internal[prefix + ".decoder.input_value"]
            hf_post_attention = gemma4_internal[prefix + ".post_attention_layernorm"]
            hf_pre_ff_input = gemma4_internal[prefix + ".pre_feedforward_layernorm.input_value"]
            hf_post_ff = gemma4_internal[prefix + ".post_feedforward_layernorm"]
            hf_decoder_output = gemma4_internal[prefix + ".decoder"]

            native_input = torch.tensor(trace[trace_prefix + ".input"], dtype=torch.float32).reshape_as(decoder_input)
            native_post_attention = torch.tensor(
                trace[trace_prefix + ".post_attention_norm_output"], dtype=torch.float32
            ).reshape_as(hf_post_attention)
            native_residual1 = torch.tensor(
                trace[trace_prefix + ".residual1"], dtype=torch.float32
            ).reshape_as(hf_pre_ff_input)
            native_post_ff = torch.tensor(
                trace[trace_prefix + ".post_mlp_norm_output"], dtype=torch.float32
            ).reshape_as(hf_post_ff)
            native_output = torch.tensor(
                trace[trace_prefix + ".output"], dtype=torch.float32
            ).reshape_as(hf_decoder_output)

            hf_attention_join = decoder_input + hf_post_attention
            native_attention_join = native_input + native_post_attention
            hf_mlp_join = hf_pre_ff_input + hf_post_ff
            native_mlp_join = native_residual1 + native_post_ff
            gemma4_forward_residual_replay[prefix] = {
                "hf_attention_replay_vs_hf_residual1_max_abs": float(
                    (hf_attention_join - hf_pre_ff_input).abs().max()
                ),
                "native_attention_replay_vs_native_residual1_max_abs": float(
                    (native_attention_join - native_residual1).abs().max()
                ),
                "native_attention_replay_vs_hf_residual1_max_abs": float(
                    (native_attention_join - hf_pre_ff_input).abs().max()
                ),
                "attention_branch_vs_hf_max_abs": float(
                    (native_post_attention - hf_post_attention).abs().max()
                ),
                "hf_mlp_replay_vs_hf_output_max_abs": float(
                    (hf_mlp_join - hf_decoder_output).abs().max()
                ),
                "native_mlp_replay_vs_native_output_max_abs": float(
                    (native_mlp_join - native_output).abs().max()
                ),
                "native_mlp_replay_vs_hf_output_max_abs": float(
                    (native_mlp_join - hf_decoder_output).abs().max()
                ),
                "mlp_branch_vs_hf_max_abs": float(
                    (native_post_ff - hf_post_ff).abs().max()
                ),
            }

        # Saved-embedding runs can remain exact through the decoder and first
        # diverge at the final Gemma4 RMSNorm. Replay that exact boundary using
        # the local Transformers spelling so mean/pow drift is distinguishable
        # from output multiplication association without changing the runtime.
        final_label = "final_norm"
        final_module = gemma4_modules[final_label]
        final_hf_input = gemma4_internal[final_label + ".input_value"]
        final_hf_output = gemma4_internal[final_label]
        final_native_input = torch.tensor(
            trace["final_prefix"], dtype=torch.float32
        ).reshape_as(final_hf_input)
        final_native_mean = torch.tensor(
            trace["final_norm_mean"], dtype=torch.float32
        ).reshape(final_native_input.shape[:-1] + (1,))
        final_native_rstd = torch.tensor(
            trace["final_norm_rstd"], dtype=torch.float32
        ).reshape_as(final_native_mean)
        final_native_output = torch.tensor(
            trace["final_norm_output"], dtype=torch.float32
        ).reshape_as(final_hf_output)
        final_torch_mean = (
            final_native_input.pow(2).mean(-1, keepdim=True) + final_module.eps
        )
        final_torch_rstd = torch.pow(final_torch_mean, -0.5)
        final_weight = final_module.weight.detach().float()
        final_reference_output = (final_native_input * final_torch_rstd) * final_weight
        final_native_rstd_left = (final_native_input * final_native_rstd) * final_weight
        final_native_rstd_right = final_native_input * (final_native_rstd * final_weight)
        final_reduction_variants = {}
        final_flat_input = final_native_input.reshape(-1, final_native_input.shape[-1])
        final_flat_torch_mean = final_torch_mean.reshape(-1)
        for lane_count in (1, 2, 4, 8, 16):
            if final_flat_input.shape[-1] % lane_count:
                continue
            for fold_mode in ("sequential", "tree"):
                means = []
                for input_row in final_flat_input:
                    lanes = torch.zeros(lane_count, dtype=torch.float32)
                    for column, value in enumerate(input_row):
                        square = (value * value).float()
                        lane = column % lane_count
                        lanes[lane] = (lanes[lane] + square).float()
                    if fold_mode == "sequential":
                        total = torch.tensor(0.0, dtype=torch.float32)
                        for lane_value in lanes:
                            total = (total + lane_value).float()
                    else:
                        reduced = lanes
                        while reduced.numel() > 1:
                            reduced = (reduced[0::2] + reduced[1::2]).float()
                        total = reduced[0]
                    means.append(
                        (total / float(input_row.numel()) + final_module.eps).float()
                    )
                candidate_mean = torch.stack(means).reshape_as(final_torch_mean)
                candidate_rstd = torch.pow(candidate_mean, -0.5)
                candidate_output = (final_native_input * candidate_rstd) * final_weight
                final_reduction_variants[f"lane{lane_count}_{fold_mode}"] = {
                    "mean_vs_torch_max_abs": float(
                        (candidate_mean.reshape(-1) - final_flat_torch_mean).abs().max()
                    ),
                    "output_vs_hf_max_abs": float(
                        (candidate_output - final_hf_output).abs().max()
                    ),
                }
        gemma4_comparisons[final_label].update(
            {
                "input_max_abs": float((final_native_input - final_hf_input).abs().max()),
                "mean_vs_torch_max_abs": float(
                    (final_native_mean - final_torch_mean).abs().max()
                ),
                "rstd_vs_torch_pow_max_abs": float(
                    (final_native_rstd - final_torch_rstd).abs().max()
                ),
                "native_output_vs_torch_reference_max_abs": float(
                    (final_native_output - final_reference_output).abs().max()
                ),
                "native_output_vs_native_rstd_left_assoc_max_abs": float(
                    (final_native_output - final_native_rstd_left).abs().max()
                ),
                "native_output_vs_native_rstd_right_assoc_max_abs": float(
                    (final_native_output - final_native_rstd_right).abs().max()
                ),
                "torch_reference_vs_hf_output_max_abs": float(
                    (final_reference_output - final_hf_output).abs().max()
                ),
                "mean_reduction_variants": final_reduction_variants,
            }
        )

        # Compare the internal gated-MLP arithmetic that is otherwise hidden
        # behind the down-projection module hook.  Gemma4 uses PyTorch's C
        # tanh-GELU (`nn.functional.gelu(..., approximate="tanh")`) followed by
        # an elementwise product with the up projection.  Tiny differences here
        # can be magnified by the following pow-based RMSNorm during backward.
        for layer_index in layer_indices:
            gate_label = f"layer{layer_index}.mlp.gate_proj"
            up_label = f"layer{layer_index}.mlp.up_proj"
            gate_hf = gemma4_internal.get(gate_label)
            up_hf = gemma4_internal.get(up_label)
            activation_key = f"layers.{layer_index}.mlp_activation"
            product_key = f"layers.{layer_index}.mlp_product"
            if (
                gate_hf is not None
                and up_hf is not None
                and activation_key in trace
                and product_key in trace
            ):
                hf_activation = torch.nn.functional.gelu(gate_hf, approximate="tanh")
                hf_product = hf_activation * up_hf
                native_activation = torch.tensor(
                    trace[activation_key], dtype=torch.float32
                ).reshape_as(hf_activation)
                native_product = torch.tensor(
                    trace[product_key], dtype=torch.float32
                ).reshape_as(hf_product)
                gemma4_comparisons[f"layer{layer_index}.mlp.internal"] = {
                    "activation_max_abs": float(
                        (native_activation - hf_activation).abs().max()
                    ),
                    "product_max_abs": float((native_product - hf_product).abs().max()),
                    "native_product_vs_native_activation_times_hf_up_max_abs": float(
                        (native_product - native_activation * up_hf).abs().max()
                    ),
                }

        # Diagnose the first current Gemma4 forward divergence without changing
        # production kernels.  PyTorch's tiny CPU GEMMs can switch reduction
        # topology by shape; replay the captured layer-0 MLP down projection
        # with explicit FP32 rounding so we can distinguish base GEMM order
        # from LoRA branch arithmetic before retaining another kernel variant.
        down_label = "layer0.mlp.down_proj"
        down_module = gemma4_modules.get(down_label)
        down_input = gemma4_internal.get(down_label + ".input_value")
        down_output = gemma4_internal.get(down_label)
        if (
            down_module is not None
            and down_input is not None
            and down_output is not None
            and hasattr(down_module, "base_layer")
        ):
            def f32(value):
                return struct.unpack("<f", struct.pack("<f", float(value)))[0]

            def replay_linear(input_tensor, weight, lanes, fused, reduction):
                flat_input = input_tensor.detach().float().cpu().reshape(-1, input_tensor.shape[-1])
                flat_weight = weight.detach().float().cpu()
                result = torch.empty(
                    (flat_input.shape[0], flat_weight.shape[0]), dtype=torch.float32
                )
                for row_index in range(flat_input.shape[0]):
                    for out_index in range(flat_weight.shape[0]):
                        acc = [0.0] * lanes
                        for inner in range(flat_input.shape[1]):
                            lane = inner % lanes
                            x_value = float(flat_input[row_index, inner])
                            w_value = float(flat_weight[out_index, inner])
                            if fused:
                                # A binary32 product plus binary32 accumulator
                                # is exactly representable in binary64 before
                                # the one final binary32 rounding, i.e. FP32 FMA.
                                acc[lane] = f32(x_value * w_value + acc[lane])
                            else:
                                product = f32(x_value * w_value)
                                acc[lane] = f32(acc[lane] + product)
                        if reduction == "sequential":
                            total = 0.0
                            for value in acc:
                                total = f32(total + value)
                        elif reduction == "cross":
                            values = list(acc)
                            while len(values) > 1:
                                half = len(values) // 2
                                values = [
                                    f32(values[index] + values[index + half])
                                    for index in range(half)
                                ]
                            total = values[0]
                        else:
                            values = list(acc)
                            while len(values) > 1:
                                values = [
                                    f32(values[index] + values[index + 1])
                                    for index in range(0, len(values), 2)
                                ]
                            total = values[0]
                        result[row_index, out_index] = total
                return result.reshape(*input_tensor.shape[:-1], flat_weight.shape[0])

            base_weight = down_module.base_layer.weight
            base_hf = torch.nn.functional.linear(down_input.float(), base_weight.float())
            reduction_candidates = {}
            for lanes in (1, 2, 4, 8, 16):
                for fused in (False, True):
                    reductions = ("sequential", "tree")
                    if lanes > 1:
                        reductions = reductions + ("cross",)
                    for reduction in reductions:
                        replay = replay_linear(
                            down_input, base_weight, lanes, fused, reduction
                        )
                        reduction_candidates[
                            f"lanes{lanes}_{'fma' if fused else 'muladd'}_{reduction}"
                        ] = float((replay - base_hf).abs().max())

            gemma4_comparisons[down_label]["base_reduction_candidates"] = dict(
                sorted(reduction_candidates.items(), key=lambda item: item[1])
            )
            gemma4_comparisons[down_label]["hf_base_plus_lora_vs_hook_max_abs"] = None
            if hasattr(down_module, "lora_A") and down_module.lora_A:
                adapter_name = next(iter(down_module.lora_A))
                a_weight = down_module.lora_A[adapter_name].weight
                b_weight = down_module.lora_B[adapter_name].weight
                scale = float(down_module.scaling[adapter_name])
                a_hf = torch.nn.functional.linear(down_input.float(), a_weight.float())
                b_hf = torch.nn.functional.linear(a_hf, b_weight.float()) * scale
                combined_hf = base_hf + b_hf
                gemma4_comparisons[down_label]["hf_base_plus_lora_vs_hook_max_abs"] = float(
                    (combined_hf - down_output.float()).abs().max()
                )
                for component_name, component_input, component_weight, component_hf in (
                    ("lora_a", down_input, a_weight, a_hf),
                    ("lora_b", a_hf, b_weight, torch.nn.functional.linear(a_hf, b_weight.float())),
                ):
                    component_candidates = {}
                    for lanes in (1, 2, 4, 8, 16):
                        if component_input.shape[-1] < lanes:
                            continue
                        for fused in (False, True):
                            reductions = ("sequential", "tree")
                            if lanes > 1:
                                reductions = reductions + ("cross",)
                            for reduction in reductions:
                                replay = replay_linear(
                                    component_input, component_weight, lanes, fused, reduction
                                )
                                component_candidates[
                                    f"lanes{lanes}_{'fma' if fused else 'muladd'}_{reduction}"
                                ] = float((replay - component_hf).abs().max())
                    gemma4_comparisons[down_label][f"{component_name}_reduction_candidates"] = dict(
                        sorted(component_candidates.items(), key=lambda item: item[1])
                    )

        # Inspect every down-projection adjoint using identical operands. Tiny
        # upstream errors can be amplified by the earlier per-head RMSNorms.
        mlp_input_branches = {}
        for layer_index, projection in (
            (index, projection) for index in layer_indices
            for projection in ("down", "gate", "up")
        ):
            label = f"layer{layer_index}.mlp.{projection}_proj"
            module = gemma4_modules[label]
            incoming = gemma4_internal[label + ".grad_output"]
            expected_projection_grad = gemma4_internal[label + ".grad_input"]
            incoming_key = "grad_post_mlp_norm" if projection == "down" else f"grad_mlp_{projection}_output"
            result_key = "grad_mlp_product" if projection == "down" else f"grad_mlp_{projection}_input"
            native_incoming = torch.tensor(
                trace[f"layers.{layer_index}.{incoming_key}"], dtype=torch.float32
            ).reshape_as(incoming)
            native_result = torch.tensor(
                trace[f"layers.{layer_index}.{result_key}"], dtype=torch.float32
            ).reshape_as(expected_projection_grad)
            adapter_name = next(iter(module.lora_A))
            weights = [module.base_layer.weight.detach(),
                       module.lora_B[adapter_name].weight.detach(),
                       module.lora_A[adapter_name].weight.detach()]
            scale = float(module.scaling[adapter_name])
            parts = []
            for source in (incoming, native_incoming):
                flat = source.reshape(-1, source.shape[-1])
                base = flat @ weights[0]
                a_grad = (flat * scale) @ weights[1]
                lora = a_grad @ weights[2]
                parts.append((base + lora).reshape_as(expected_projection_grad))
                if projection != "down" and len(parts) == 1:
                    mlp_input_branches[(layer_index, projection)] = {
                        "base": base.reshape_as(expected_projection_grad),
                        "lora": lora.reshape_as(expected_projection_grad),
                    }
            candidates = {}
            flat = native_incoming.reshape(-1, native_incoming.shape[-1])
            for part, values, weight in (
                ("base", flat, weights[0]),
                ("lora_b", flat * scale, weights[1]),
                ("lora_a", (flat * scale) @ weights[1], weights[2]),
            ):
                reference = values @ weight
                candidates[part] = {}
                for lanes in (1, 2, 4, 8, 16):
                    for fused in (False, True):
                        for reduction in ("sequential", "cross"):
                            replay = replay_linear(values, weight.T, lanes, fused, reduction)
                            candidates[part][f"{lanes}_{fused}_{reduction}"] = float(
                                (replay - reference).abs().max()
                            )
            gemma4_comparisons[label]["backward_replay"] = {
                # Gate/up share their input tensor: its hook sees the sum of
                # both branches, not this projection's independent adjoint.
                "hf_vs_hook": float((parts[0] - expected_projection_grad).abs().max()) if projection == "down" else None,
                "native_operands_vs_native": float((parts[1] - native_result).abs().max()),
                "incoming_drift": float((incoming - native_incoming).abs().max()),
                "reduction_candidates": candidates,
            }

        import itertools
        for layer_index in layer_indices:
            branches = {f"{proj}_{part}": tensor
                        for proj in ("up", "gate")
                        for part, tensor in mlp_input_branches[(layer_index, proj)].items()}
            expected_mlp_grad = gemma4_internal[f"layer{layer_index}.mlp.grad_input"]
            orders = {}
            for order in itertools.permutations(branches):
                result = branches[order[0]]
                for key in order[1:]:
                    result = result + branches[key]
                orders["+".join(order)] = float((result - expected_mlp_grad).abs().max())
            gemma4_comparisons[f"layer{layer_index}.mlp"]["input_branch_orders"] = dict(
                sorted(orders.items(), key=lambda item: item[1])
            )

        for layer_index in layer_indices:
            prefix = f"layers.{layer_index}."
            output_grad_key = (f"layers.{layer_index + 1}.grad_input"
                               if layer_index + 1 < len(layer_indices)
                               else "grad_final_norm_input")
            for norm, xkey, dykey, branchkey, totalkey, residualkey in (
                ("input_layernorm", "input", "grad_norm_output", "grad_input_attention",
                 "grad_input", prefix + "grad_residual1"),
                ("pre_feedforward_layernorm", "residual1", "grad_mlp_norm_output", "grad_residual1_mlp",
                 "grad_residual1", output_grad_key),
            ):
                label = f"layer{layer_index}.{norm}"
                module = gemma4_modules[label]
                x = torch.tensor(trace[prefix + xkey]).reshape_as(gemma4_internal[label + ".input_value"])
                dy = torch.tensor(trace[prefix + dykey]).reshape_as(x)
                native_branch = torch.tensor(trace[prefix + branchkey]).reshape_as(x)
                native_total = torch.tensor(trace[prefix + totalkey]).reshape_as(x)
                residual = torch.tensor(trace[residualkey]).reshape_as(x)
                with torch.enable_grad():
                    probe = x.clone().requires_grad_()
                    module(probe).backward(dy)
                mean = x.square().mean(-1, keepdim=True) + module.eps
                weighted = dy * module.weight.detach()
                direct = weighted * mean.pow(-0.5)
                through = (((weighted * x).sum(-1, keepdim=True) * -0.5)
                           * mean.pow(-1.5) / x.shape[-1]) * (2.0 * x)
                gemma4_comparisons[label]["residual_backward_replay"] = {
                    "branch_vs_native": float((probe.grad - native_branch).abs().max()),
                    "separate_vs_native_total": float((residual + probe.grad - native_total).abs().max()),
                    "joined_vs_native_total": float(((residual + direct) + through - native_total).abs().max()),
                    "staged_vs_autograd_branch": float((direct + through - probe.grad).abs().max()),
                }
                hf_x = gemma4_internal[label + ".input_value"]
                hf_dy = gemma4_internal[label + ".grad_output"]
                hf_residual = gemma4_internal[
                    f"layer{layer_index}.post_attention_layernorm.grad_output"
                    if norm == "input_layernorm" else f"layer{layer_index}.decoder.grad_output"
                ]
                hf_total = gemma4_internal.get(label + ".grad_input")
                if hf_total is None:
                    continue  # Frozen layer-0 embeddings need no input adjoint.
                hf_mean = hf_x.square().mean(-1, keepdim=True) + module.eps
                hf_weighted = hf_dy * module.weight.detach()
                hf_direct = hf_weighted * hf_mean.pow(-0.5)
                hf_through = (((hf_weighted * hf_x).sum(-1, keepdim=True) * -0.5)
                              * hf_mean.pow(-1.5) / hf_x.shape[-1]) * (2.0 * hf_x)
                gemma4_comparisons[label]["residual_backward_replay"].update({
                    "hf_separate_vs_hook": float((hf_residual + (hf_direct + hf_through) - hf_total).abs().max()),
                    "hf_joined_vs_hook": float(((hf_residual + hf_direct) + hf_through - hf_total).abs().max()),
                })

        # Replay the PEFT-wrapped layer-0 output projection with both the HF
        # and native incoming adjoints.  Gemma4's per-head Q/K/V RMSNorms can
        # amplify single-ulp errors here by orders of magnitude, so distinguish
        # an upstream adjoint mismatch from arithmetic inside native linear
        # backward before changing either shared kernel.
        o_proj = gemma4_modules["layer0.o_proj"]
        hf_o_grad_output = gemma4_internal.get("layer0.o_proj.grad_output")
        hf_o_grad_input = gemma4_internal.get("layer0.o_proj.grad_input")
        if (
            hf_o_grad_output is not None
            and hf_o_grad_input is not None
            and "grad_post_attention_norm" in trace
            and "layers.0.grad_attention_output" in trace
            and hasattr(o_proj, "base_layer")
            and hasattr(o_proj, "lora_A")
        ):
            native_o_grad_output = torch.tensor(
                trace["grad_post_attention_norm"], dtype=torch.float32
            ).reshape_as(hf_o_grad_output)
            native_o_grad_input = torch.tensor(
                trace["layers.0.grad_attention_output"], dtype=torch.float32
            ).reshape_as(hf_o_grad_input)
            adapter_name = next(iter(o_proj.lora_A))
            base_weight = o_proj.base_layer.weight.detach().float().cpu()
            a_weight = o_proj.lora_A[adapter_name].weight.detach().float().cpu()
            b_weight = o_proj.lora_B[adapter_name].weight.detach().float().cpu()
            scale = float(o_proj.scaling[adapter_name])

            def replay_o_proj_backward(grad_output):
                flat_grad = grad_output.reshape(-1, base_weight.shape[0])
                base_grad = torch.matmul(flat_grad, base_weight)
                scaled_grad = (flat_grad * scale).float()
                grad_a_output = torch.matmul(scaled_grad, b_weight)
                lora_grad = torch.matmul(grad_a_output, a_weight)
                return (base_grad + lora_grad).reshape_as(hf_o_grad_input)

            replay_hf = replay_o_proj_backward(hf_o_grad_output)
            replay_native = replay_o_proj_backward(native_o_grad_output)
            gemma4_o_proj_replay = {
                "native_grad_output_vs_hf_max_abs": float(
                    (native_o_grad_output - hf_o_grad_output).abs().max()
                ),
                "hf_replay_vs_hf_grad_input_max_abs": float(
                    (replay_hf - hf_o_grad_input).abs().max()
                ),
                "native_replay_vs_native_grad_input_max_abs": float(
                    (replay_native - native_o_grad_input).abs().max()
                ),
                "native_replay_vs_hf_grad_input_max_abs": float(
                    (replay_native - hf_o_grad_input).abs().max()
                ),
                "native_grad_input_vs_hf_max_abs": float(
                    (native_o_grad_input - hf_o_grad_input).abs().max()
                ),
            }

            per_layer_o_proj = {}
            for layer_index in layer_indices:
                label = f"layer{layer_index}.o_proj"
                layer_module = gemma4_modules[label]
                layer_hf_grad_output = gemma4_internal.get(label + ".grad_output")
                layer_hf_grad_input = gemma4_internal.get(label + ".grad_input")
                trace_prefix = f"layers.{layer_index}."
                if layer_hf_grad_output is None or layer_hf_grad_input is None:
                    continue
                layer_native_grad_output = torch.tensor(
                    trace[trace_prefix + "grad_post_attention_norm"], dtype=torch.float32
                ).reshape_as(layer_hf_grad_output)
                layer_native_grad_input = torch.tensor(
                    trace[trace_prefix + "grad_attention_output"], dtype=torch.float32
                ).reshape_as(layer_hf_grad_input)
                layer_adapter_name = next(iter(layer_module.lora_A))
                layer_base_weight = layer_module.base_layer.weight.detach().float().cpu()
                layer_a_weight = layer_module.lora_A[layer_adapter_name].weight.detach().float().cpu()
                layer_b_weight = layer_module.lora_B[layer_adapter_name].weight.detach().float().cpu()
                layer_scale = float(layer_module.scaling[layer_adapter_name])

                def replay_layer_o_proj(grad_output):
                    flat_grad = grad_output.reshape(-1, layer_base_weight.shape[0])
                    base_grad = torch.matmul(flat_grad, layer_base_weight)
                    scaled_grad = (flat_grad * layer_scale).float()
                    grad_a_output = torch.matmul(scaled_grad, layer_b_weight)
                    lora_grad = torch.matmul(grad_a_output, layer_a_weight)
                    return (base_grad + lora_grad).reshape_as(layer_hf_grad_input)

                layer_hf_replay = replay_layer_o_proj(layer_hf_grad_output)
                layer_native_replay = replay_layer_o_proj(layer_native_grad_output)
                per_layer_o_proj[f"layer{layer_index}"] = {
                    "native_grad_output_vs_hf": float(
                        (layer_native_grad_output - layer_hf_grad_output).abs().max()
                    ),
                    "hf_replay_vs_hf_grad_input": float(
                        (layer_hf_replay - layer_hf_grad_input).abs().max()
                    ),
                    "native_replay_vs_native_grad_input": float(
                        (layer_native_replay - layer_native_grad_input).abs().max()
                    ),
                    "native_replay_vs_hf_grad_input": float(
                        (layer_native_replay - layer_hf_grad_input).abs().max()
                    ),
                    "native_grad_input_vs_hf": float(
                        (layer_native_grad_input - layer_hf_grad_input).abs().max()
                    ),
                }
            gemma4_o_proj_replay["per_layer"] = per_layer_o_proj

        # Decompose the three Gemma4 attention projection contributions that
        # feed layer-0 input_layernorm.  The Q/K/V norm derivatives are highly
        # sensitive to tiny incoming-adjoint differences, so distinguish those
        # upstream differences from native linear-backward arithmetic and from
        # the final Q + (K + V) accumulation order.
        qkv_projection_replay = {}
        replay_hf_contributions = {}
        replay_native_contributions = {}
        for short_label in ("q", "k", "v"):
            module = gemma4_modules[f"layer0.{short_label}_proj"]
            hf_grad_output = gemma4_internal.get(
                f"layer0.{short_label}_proj.grad_output"
            )
            if hf_grad_output is None or not hasattr(module, "base_layer"):
                continue
            native_grad_output = torch.tensor(
                trace[f"grad_{short_label}_linear_output"], dtype=torch.float32
            ).reshape_as(hf_grad_output)
            native_grad_input = torch.tensor(
                trace[f"grad_norm_{short_label}"], dtype=torch.float32
            ).reshape(ids.shape[0], ids.shape[1], -1)
            base_weight = module.base_layer.weight.detach().float().cpu()
            adapter_name = next(iter(module.lora_A))
            a_weight = module.lora_A[adapter_name].weight.detach().float().cpu()
            b_weight = module.lora_B[adapter_name].weight.detach().float().cpu()
            scale = float(module.scaling[adapter_name])

            def replay_projection_input(grad_output):
                flat_grad = grad_output.reshape(-1, base_weight.shape[0])
                base_grad = torch.matmul(flat_grad, base_weight)
                scaled_grad = (flat_grad * scale).float()
                grad_a_output = torch.matmul(scaled_grad, b_weight)
                lora_grad = torch.matmul(grad_a_output, a_weight)
                return (base_grad + lora_grad).reshape_as(native_grad_input)

            replay_hf = replay_projection_input(hf_grad_output)
            replay_native = replay_projection_input(native_grad_output)
            replay_hf_contributions[short_label] = replay_hf
            replay_native_contributions[short_label] = replay_native
            qkv_projection_replay[short_label] = {
                "native_grad_output_vs_hf_max_abs": float(
                    (native_grad_output - hf_grad_output).abs().max()
                ),
                "native_replay_vs_native_grad_input_max_abs": float(
                    (replay_native - native_grad_input).abs().max()
                ),
                "native_replay_vs_hf_replay_max_abs": float(
                    (replay_native - replay_hf).abs().max()
                ),
            }

        if set(replay_hf_contributions) == {"q", "k", "v"}:
            hf_qkv_sum = replay_hf_contributions["q"] + (
                replay_hf_contributions["k"] + replay_hf_contributions["v"]
            )
            native_qkv_sum = replay_native_contributions["q"] + (
                replay_native_contributions["k"] + replay_native_contributions["v"]
            )
            native_kernel_contributions = {
                short_label: torch.tensor(
                    trace[f"grad_norm_{short_label}"], dtype=torch.float32
                ).reshape_as(hf_qkv_sum)
                for short_label in ("q", "k", "v")
            }
            native_total = torch.tensor(
                trace["grad_norm_output"], dtype=torch.float32
            ).reshape_as(hf_qkv_sum)
            hf_total = gemma4_internal.get("layer0.input_layernorm.grad_output")
            association_variants = {
                "q_plus_kv": lambda values: values["q"] + (values["k"] + values["v"]),
                "qk_plus_v": lambda values: (values["q"] + values["k"]) + values["v"],
                "k_plus_qv": lambda values: values["k"] + (values["q"] + values["v"]),
                "kq_plus_v": lambda values: (values["k"] + values["q"]) + values["v"],
                "v_plus_qk": lambda values: values["v"] + (values["q"] + values["k"]),
                "vq_plus_k": lambda values: (values["v"] + values["q"]) + values["k"],
            }
            qkv_projection_replay["sum"] = {
                "hf_replay_vs_hf_total_max_abs": float(
                    (hf_qkv_sum - hf_total).abs().max()
                ),
                "native_replay_vs_native_total_max_abs": float(
                    (native_qkv_sum - native_total).abs().max()
                ),
                "native_replay_vs_hf_total_max_abs": float(
                    (native_qkv_sum - hf_total).abs().max()
                ),
                "native_total_vs_hf_total_max_abs": float(
                    (native_total - hf_total).abs().max()
                ),
                "hf_associations_vs_hf_total": {
                    name: float((combine(replay_hf_contributions) - hf_total).abs().max())
                    for name, combine in association_variants.items()
                },
                "native_replay_associations_vs_hf_total": {
                    name: float((combine(replay_native_contributions) - hf_total).abs().max())
                    for name, combine in association_variants.items()
                },
                "native_kernel_associations_vs_hf_total": {
                    name: float((combine(native_kernel_contributions) - hf_total).abs().max())
                    for name, combine in association_variants.items()
                },
            }
        # Tensor hooks do not insert module-level backward barriers. Preserve
        # that graph when examining accumulation across all six Q/K/V paths.
        for layer_index in layer_indices:
            branches = {}
            for projection in ("q", "k", "v"):
                label = f"layer{layer_index}.{projection}_proj"
                module = gemma4_modules[label]
                incoming = gemma4_internal[label + ".grad_output"]
                adapter_name = next(iter(module.lora_A))
                flat = incoming.reshape(-1, incoming.shape[-1])
                branches[projection + "_base"] = flat @ module.base_layer.weight.detach()
                branches[projection + "_lora"] = (
                    (flat * module.scaling[adapter_name]) @ module.lora_B[adapter_name].weight.detach()
                ) @ module.lora_A[adapter_name].weight.detach()
            expected_qkv_grad = gemma4_internal[
                f"layer{layer_index}.input_layernorm.grad_output"
            ].reshape_as(branches["q_base"])
            orders = {}
            for order in itertools.permutations(branches):
                result = branches[order[0]]
                for key in order[1:]:
                    result = result + branches[key]
                orders["+".join(order)] = float(
                    (result - expected_qkv_grad).abs().max()
                )
            gemma4_comparisons[f"layer{layer_index}.input_layernorm"]["qkv_branch_orders"] = dict(
                sorted(orders.items(), key=lambda item: item[1])[:8]
            )

        # Detailed mean/rstd scratch is currently exported for layer 0 only.
        input_norm_label = "layer0.input_layernorm"
        input_norm_module = gemma4_modules[input_norm_label]
        input_norm_hf_input = gemma4_internal.get(input_norm_label + ".input_value")
        if input_norm_hf_input is not None:
            input_norm_hf_mean = (
                input_norm_hf_input.float().pow(2).mean(-1, keepdim=True)
                + input_norm_module.eps
            )
            input_norm_hf_rstd = torch.pow(input_norm_hf_mean, -0.5)
            input_norm_native_input = torch.tensor(
                trace["layers.0.input"], dtype=torch.float32
            ).reshape_as(input_norm_hf_input)
            input_norm_native_mean = torch.tensor(
                trace["norm_mean"], dtype=torch.float32
            ).reshape_as(input_norm_hf_mean)
            input_norm_native_rstd = torch.tensor(
                trace["norm_rstd"], dtype=torch.float32
            ).reshape_as(input_norm_hf_rstd)
            input_norm_weight = (
                input_norm_module.weight.detach().float().cpu()
                if hasattr(input_norm_module, "weight")
                else torch.ones(input_norm_hf_input.shape[-1], dtype=torch.float32)
            )
            input_norm_replay = (
                input_norm_native_input
                * torch.pow(input_norm_native_mean, -0.5)
                * input_norm_weight
            )
            input_norm_hf_grad_output = gemma4_internal.get(
                input_norm_label + ".grad_output"
            )
            if input_norm_hf_grad_output is not None:
                input_norm_native_grad_output = torch.tensor(
                    trace["layers.0.grad_norm_output"], dtype=torch.float32
                ).reshape_as(input_norm_hf_grad_output)
                native_unscaled = input_norm_native_input * input_norm_native_rstd
                hf_unscaled = input_norm_hf_input * input_norm_hf_rstd
                native_product = input_norm_native_grad_output * native_unscaled
                hf_product = input_norm_hf_grad_output * hf_unscaled
                native_torch_sum = native_product.reshape(-1, native_product.shape[-1]).sum(0)
                hf_torch_sum = hf_product.reshape(-1, hf_product.shape[-1]).sum(0)

                def sequential_row_sum(product):
                    flat = product.reshape(-1, product.shape[-1])
                    result = torch.zeros(flat.shape[-1], dtype=torch.float32)
                    for row in flat:
                        result = (result + row).float()
                    return result

                native_sequential_sum = sequential_row_sum(native_product)
                hf_sequential_sum = sequential_row_sum(hf_product)
                if "norm_weight_grad" in trace and input_norm_module.weight.grad is not None:
                    input_norm_native_weight_grad = torch.tensor(
                        trace["norm_weight_grad"], dtype=torch.float32
                    ).reshape_as(input_norm_weight)
                    input_norm_hf_weight_grad = input_norm_module.weight.grad.detach().float().cpu()
                    gemma4_comparisons[input_norm_label]["weight_grad_reduction"] = {
                        "native_kernel_vs_hf_max_abs": float(
                            (input_norm_native_weight_grad - input_norm_hf_weight_grad).abs().max()
                        ),
                        "hf_torch_replay_vs_hf_max_abs": float(
                            (hf_torch_sum - input_norm_hf_weight_grad).abs().max()
                        ),
                        "hf_sequential_replay_vs_hf_max_abs": float(
                            (hf_sequential_sum - input_norm_hf_weight_grad).abs().max()
                        ),
                        "native_torch_replay_vs_native_kernel_max_abs": float(
                            (native_torch_sum - input_norm_native_weight_grad).abs().max()
                        ),
                        "native_sequential_replay_vs_native_kernel_max_abs": float(
                            (native_sequential_sum - input_norm_native_weight_grad).abs().max()
                        ),
                        "native_torch_replay_vs_hf_max_abs": float(
                            (native_torch_sum - input_norm_hf_weight_grad).abs().max()
                        ),
                        "native_sequential_replay_vs_hf_max_abs": float(
                            (native_sequential_sum - input_norm_hf_weight_grad).abs().max()
                        ),
                    }
            reduction_variants = {}
            flat_input_norm = input_norm_native_input.reshape(
                -1, input_norm_native_input.shape[-1]
            )
            flat_hf_mean = input_norm_hf_mean.reshape(-1)
            flat_native_mean = input_norm_native_mean.reshape(-1)
            for lane_count in (1, 2, 4, 8, 16):
                if flat_input_norm.shape[-1] % lane_count:
                    continue
                for fold_mode in ("sequential", "tree"):
                    means = []
                    for input_row in flat_input_norm:
                        lanes = torch.zeros(lane_count, dtype=torch.float32)
                        for column, value in enumerate(input_row):
                            square = (value * value).float()
                            lane = column % lane_count
                            lanes[lane] = (lanes[lane] + square).float()
                        if fold_mode == "sequential":
                            total = torch.tensor(0.0, dtype=torch.float32)
                            for lane_value in lanes:
                                total = (total + lane_value).float()
                        else:
                            reduced = lanes
                            while reduced.numel() > 1:
                                reduced = (reduced[0::2] + reduced[1::2]).float()
                            total = reduced[0]
                        means.append(
                            (total / float(input_row.numel()) + input_norm_module.eps).float()
                        )
                    candidate = torch.stack(means)
                    reduction_variants[f"lane{lane_count}_{fold_mode}"] = {
                        "vs_hf_max_abs": float((candidate - flat_hf_mean).abs().max()),
                        "vs_native_max_abs": float((candidate - flat_native_mean).abs().max()),
                    }
            gemma4_comparisons[input_norm_label].update(
                {
                    "input_max_abs": float(
                        (input_norm_native_input - input_norm_hf_input).abs().max()
                    ),
                    "mean_max_abs": float(
                        (input_norm_native_mean - input_norm_hf_mean).abs().max()
                    ),
                    "rstd_max_abs": float(
                        (input_norm_native_rstd - input_norm_hf_rstd).abs().max()
                    ),
                    "native_rstd_vs_torch_pow_native_mean_max_abs": float(
                        (
                            input_norm_native_rstd
                            - torch.pow(input_norm_native_mean, -0.5)
                        )
                        .abs()
                        .max()
                    ),
                    "native_output_vs_torch_pow_native_operands_max_abs": float(
                        (
                            torch.tensor(trace["norm_output"], dtype=torch.float32)
                            .reshape_as(input_norm_replay)
                            - input_norm_replay
                        )
                        .abs()
                        .max()
                    ),
                    "mean_reduction_variants": reduction_variants,
                }
            )

        # Gemma4's final per-layer RMSNorm sits immediately before the second
        # residual join.  Replaying it from the exact native operands tells us
        # whether the large MLP-side adjoint drift comes from the shared RMS
        # derivative itself or from an upstream residual/MLP mismatch.
        post_norm_label = "layer0.post_feedforward_layernorm"
        post_norm_module = gemma4_modules[post_norm_label]
        post_norm_hf_input = gemma4_internal.get(post_norm_label + ".input_value")
        post_norm_hf_grad_output = gemma4_internal.get(post_norm_label + ".grad_output")
        if post_norm_hf_input is not None and post_norm_hf_grad_output is not None:
            post_norm_hf_mean = (
                post_norm_hf_input.float().pow(2).mean(-1, keepdim=True)
                + post_norm_module.eps
            )
            post_norm_hf_rstd = torch.pow(post_norm_hf_mean, -0.5)
            post_norm_native_input = torch.tensor(
                trace["post_mlp_norm_input"], dtype=torch.float32
            ).reshape_as(post_norm_hf_input)
            post_norm_native_mean = torch.tensor(
                trace["post_mlp_norm_mean"], dtype=torch.float32
            ).reshape_as(post_norm_hf_mean)
            post_norm_native_rstd = torch.tensor(
                trace["post_mlp_norm_rstd"], dtype=torch.float32
            ).reshape_as(post_norm_hf_rstd)
            post_norm_native_grad_output = torch.tensor(
                trace["layers.1.grad_input"], dtype=torch.float32
            ).reshape_as(post_norm_hf_grad_output)
            post_norm_native_grad_input = torch.tensor(
                trace["grad_post_mlp_norm"], dtype=torch.float32
            ).reshape_as(post_norm_hf_input)
            post_norm_weight = post_norm_module.weight.detach().float().cpu()

            replay_input = post_norm_native_input.detach().clone().requires_grad_(True)
            replay_mean = replay_input.pow(2).mean(-1, keepdim=True) + post_norm_module.eps
            replay_rstd = torch.pow(replay_mean, -0.5)
            replay_output = replay_input * replay_rstd * post_norm_weight
            (replay_output * post_norm_native_grad_output).sum().backward()

            weighted_grad = post_norm_native_grad_output * post_norm_weight
            native_dot = (weighted_grad * post_norm_native_input).sum(-1, keepdim=True)
            derivative_variants = {
                "torch_pow_neg_three_halves": torch.pow(post_norm_native_mean, -1.5),
                "rsqrt_cubed": (
                    torch.rsqrt(post_norm_native_mean)
                    * torch.rsqrt(post_norm_native_mean)
                    * torch.rsqrt(post_norm_native_mean)
                ),
                "rsqrt_over_mean": torch.rsqrt(post_norm_native_mean) / post_norm_native_mean,
                "reciprocal_mean_times_rsqrt": (
                    torch.ones_like(post_norm_native_mean) / post_norm_native_mean
                ) * torch.rsqrt(post_norm_native_mean),
            }

            def post_explicit_backward(derivative):
                grad_mean = (native_dot * -0.5) * derivative
                grad_square = grad_mean / float(post_norm_native_input.shape[-1])
                direct = weighted_grad * post_norm_native_rstd
                through_square = grad_square * (2.0 * post_norm_native_input)
                return direct + through_square

            gemma4_comparisons[post_norm_label].update(
                {
                    "input_max_abs": float(
                        (post_norm_native_input - post_norm_hf_input).abs().max()
                    ),
                    "mean_max_abs": float(
                        (post_norm_native_mean - post_norm_hf_mean).abs().max()
                    ),
                    "rstd_max_abs": float(
                        (post_norm_native_rstd - post_norm_hf_rstd).abs().max()
                    ),
                    "native_mean_vs_torch_native_input_max_abs": float(
                        (post_norm_native_mean - replay_mean.detach()).abs().max()
                    ),
                    "native_rstd_vs_torch_native_input_max_abs": float(
                        (post_norm_native_rstd - replay_rstd.detach()).abs().max()
                    ),
                    "native_output_vs_torch_pow_native_operands_max_abs": float(
                        (
                            torch.tensor(trace["post_mlp_norm_output"], dtype=torch.float32)
                            .reshape_as(replay_output)
                            - replay_output.detach()
                        )
                        .abs()
                        .max()
                    ),
                    "native_grad_output_vs_hf_max_abs": float(
                        (post_norm_native_grad_output - post_norm_hf_grad_output).abs().max()
                    ),
                    "native_grad_input_vs_torch_pow_native_operands_max_abs": float(
                        (post_norm_native_grad_input - replay_input.grad).abs().max()
                    ),
                    "explicit_backward_vs_native": {
                        name: float(
                            (post_explicit_backward(value) - post_norm_native_grad_input)
                            .abs()
                            .max()
                        )
                        for name, value in derivative_variants.items()
                    },
                    "explicit_backward_vs_torch_replay": {
                        name: float(
                            (post_explicit_backward(value) - replay_input.grad).abs().max()
                        )
                        for name, value in derivative_variants.items()
                    },
                }
            )
        for short_label in ("q_norm", "k_norm", "v_norm"):
            label = f"layer0.{short_label}"
            hf_input = gemma4_internal.get(label + ".input_value")
            module = gemma4_modules[label]
            prefix = short_label[0]
            if hf_input is None:
                continue
            hf_mean = hf_input.float().pow(2).mean(-1, keepdim=True) + module.eps
            hf_rstd = torch.pow(hf_mean, -0.5)
            native_input = torch.tensor(
                trace[prefix + "_norm_input"], dtype=torch.float32
            ).reshape_as(hf_input)
            native_mean = torch.tensor(
                trace[prefix + "_norm_mean"], dtype=torch.float32
            ).reshape_as(hf_mean)
            native_rstd = torch.tensor(
                trace[prefix + "_norm_rstd"], dtype=torch.float32
            ).reshape_as(hf_rstd)
            native_grad_output = torch.tensor(
                trace[f"grad_{prefix}_norm_output"], dtype=torch.float32
            ).reshape_as(gemma4_internal[label + ".grad_output"])
            native_grad_input = torch.tensor(
                trace[f"grad_{prefix}_linear_output"], dtype=torch.float32
            ).reshape_as(hf_input)
            replay_input = native_input.detach().clone().requires_grad_(True)
            replay_weight = (
                module.weight.detach().float().cpu()
                if hasattr(module, "weight")
                else torch.ones(hf_input.shape[-1], dtype=torch.float32)
            )
            replay_mean = replay_input.pow(2).mean(-1, keepdim=True) + module.eps
            replay_rstd = torch.pow(replay_mean, -0.5)
            replay_output = replay_input * replay_rstd * replay_weight
            (replay_output * native_grad_output).sum().backward()

            # Decompose the RMSNorm backward using the exact native operands so
            # a tiny upstream mean/rstd difference is not confused with a
            # derivative-order mismatch.  PyTorch's pow(-0.5) backward is
            # sensitive to the concrete -1.5 evaluation and FP32 operation
            # ordering, even though several algebraic forms are real-number
            # equivalent.
            weighted_grad = native_grad_output * replay_weight
            native_dot = (weighted_grad * native_input).sum(-1, keepdim=True)
            torch_pow_neg_three_halves = torch.pow(native_mean, -1.5)
            torch_rsqrt_native_mean = torch.rsqrt(native_mean)
            derivative_variants = {
                "torch_pow_neg_three_halves": torch_pow_neg_three_halves,
                "rsqrt_cubed": (
                    torch_rsqrt_native_mean
                    * torch_rsqrt_native_mean
                    * torch_rsqrt_native_mean
                ),
                "rsqrt_over_mean": torch_rsqrt_native_mean / native_mean,
                "reciprocal_mean_times_rsqrt": (
                    torch.ones_like(native_mean) / native_mean
                ) * torch_rsqrt_native_mean,
            }

            def explicit_backward(derivative):
                grad_mean = (native_dot * -0.5) * derivative
                grad_square = grad_mean / float(native_input.shape[-1])
                direct = weighted_grad * native_rstd
                through_square = grad_square * (2.0 * native_input)
                return direct + through_square

            derivative_errors = {
                name: float((explicit_backward(value) - native_grad_input).abs().max())
                for name, value in derivative_variants.items()
            }
            derivative_vs_torch_replay = {
                name: float((explicit_backward(value) - replay_input.grad).abs().max())
                for name, value in derivative_variants.items()
            }
            gemma4_comparisons[label].update(
                {
                    "input_max_abs": float((native_input - hf_input).abs().max()),
                    "mean_max_abs": float((native_mean - hf_mean).abs().max()),
                    "native_mean_vs_torch_native_input_max_abs": float(
                        (native_mean - replay_mean.detach()).abs().max()
                    ),
                    "rstd_max_abs": float((native_rstd - hf_rstd).abs().max()),
                    "native_rstd_vs_torch_native_input_max_abs": float(
                        (native_rstd - replay_rstd.detach()).abs().max()
                    ),
                    "native_rstd_vs_torch_pow_native_mean_max_abs": float(
                        (
                            native_rstd
                            - torch.pow(native_mean, -0.5)
                        )
                        .abs()
                        .max()
                    ),
                    "native_output_vs_torch_pow_native_operands_max_abs": float(
                        (
                            torch.tensor(trace[f"{prefix}_norm_output"], dtype=torch.float32)
                            .reshape_as(replay_output)
                            - replay_output.detach()
                        )
                        .abs()
                        .max()
                    ),
                    "native_grad_input_vs_torch_pow_native_operands_max_abs": float(
                        (native_grad_input - replay_input.grad).abs().max()
                    ),
                    "pow_neg_three_halves_vs_rsqrt_cubed_max_abs": float(
                        (
                            derivative_variants["torch_pow_neg_three_halves"]
                            - derivative_variants["rsqrt_cubed"]
                        )
                        .abs()
                        .max()
                    ),
                    "pow_neg_three_halves_vs_rsqrt_over_mean_max_abs": float(
                        (
                            derivative_variants["torch_pow_neg_three_halves"]
                            - derivative_variants["rsqrt_over_mean"]
                        )
                        .abs()
                        .max()
                    ),
                    "explicit_backward_vs_native": derivative_errors,
                    "explicit_backward_vs_torch_replay": derivative_vs_torch_replay,
                }
            )

        # Replay the layer-0 eager attention graph with the exact tensors that
        # entered the native attention kernel and the exact native upstream
        # adjoint.  This isolates attention-backward arithmetic from RoPE,
        # RMSNorm, LoRA, residual, and replacement-bank differences.  Keep the
        # KV repetition expressed as expand+reshape, matching Transformers'
        # repeat_kv implementation so its autograd reduction topology is also
        # part of the reference.
        config = json.loads((strict.FIXTURE_ROOT / "base" / "config.json").read_text())
        fixture = json.loads((strict.FIXTURE_ROOT / "fixture.json").read_text())
        batch = fixture["batch_size"]
        seq_len = fixture["seq_len"]
        num_heads = config["num_attention_heads"]
        num_kv_heads = config["num_key_value_heads"]
        head_dim = config.get("head_dim", config["hidden_size"] // num_heads)
        groups = num_heads // num_kv_heads

        def attention_tensor(key, heads):
            return (
                torch.tensor(trace[key], dtype=torch.float32)
                .reshape(batch, seq_len, heads, head_dim)
                .transpose(1, 2)
                .contiguous()
            )

        # Gemma4 applies RoPE before transposing Q/K into attention layout.
        # The local Transformers implementation therefore observes rotary
        # operands as [batch, seq, heads, head_dim] (unsqueeze_dim=2), which
        # is also the row-major layout used by the native trace. Keep this
        # separate from attention_tensor(), whose post-RoPE attention replay
        # intentionally converts to [batch, heads, seq, head_dim].
        def rope_tensor(key, heads):
            return torch.tensor(trace[key], dtype=torch.float32).reshape(
                batch, seq_len, heads, head_dim
            )

        replay_q = attention_tensor("q_rotary_output", num_heads).requires_grad_(True)
        replay_k = attention_tensor("k_rotary_output", num_kv_heads).requires_grad_(True)
        replay_v = attention_tensor("v_norm_output", num_kv_heads).requires_grad_(True)
        replay_grad_out = attention_tensor("grad_attention_output", num_heads)

        def repeat_kv_exact(hidden_states):
            if groups == 1:
                return hidden_states
            bsz, kv_heads, slen, dim = hidden_states.shape
            return (
                hidden_states[:, :, None, :, :]
                .expand(bsz, kv_heads, groups, slen, dim)
                .reshape(bsz, kv_heads * groups, slen, dim)
            )

        replay_k_repeated = repeat_kv_exact(replay_k)
        replay_v_repeated = repeat_kv_exact(replay_v)
        replay_scores = torch.matmul(replay_q, replay_k_repeated.transpose(2, 3))
        # Gemma4 text attention deliberately overrides the generic attention
        # default and passes scaling=1.0 into eager_attention_forward.  Using
        # head_dim**-0.5 here makes the diagnostic replay a different graph and
        # hides the real native-vs-HF attention error behind a huge false delta.
        replay_scores = replay_scores * 1.0
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), torch.finfo(torch.float32).min),
            diagonal=1,
        )
        replay_scores = replay_scores + causal_mask
        replay_probs = torch.nn.functional.softmax(replay_scores, dim=-1, dtype=torch.float32)
        replay_attention = torch.matmul(replay_probs, replay_v_repeated)
        (replay_attention * replay_grad_out).sum().backward()

        native_attention = attention_tensor("attention_output", num_heads)
        native_grad_q = attention_tensor("grad_q_rotary_output", num_heads)
        native_grad_k = attention_tensor("grad_k_rotary_output", num_kv_heads)
        native_grad_v = attention_tensor("grad_v_norm_output", num_kv_heads)

        def max_abs(left, right):
            return float((left.detach() - right.detach()).abs().max())

        gemma4_attention_replay = {
            "attention_output_max_abs": max_abs(native_attention, replay_attention),
            "grad_q_max_abs": max_abs(native_grad_q, replay_q.grad),
            "grad_k_max_abs": max_abs(native_grad_k, replay_k.grad),
            "grad_v_max_abs": max_abs(native_grad_v, replay_v.grad),
        }

        # The compact native trace also retains each decoder layer's attention
        # boundaries.  Replay them independently so a tiny upstream seed in a
        # later layer cannot be hidden by the legacy single-boundary replay
        # above.  This is especially useful for saved-embedding gradients,
        # where a sub-ulp attention adjoint difference can be amplified by the
        # following small-RMS Gemma4 normalization.
        per_layer_attention = {}
        for layer_index in layer_indices:
            prefix = f"layers.{layer_index}."
            layer_q = attention_tensor(prefix + "q_rotary_output", num_heads).requires_grad_(True)
            layer_k = attention_tensor(prefix + "k_rotary_output", num_kv_heads).requires_grad_(True)
            layer_v = attention_tensor(prefix + "v_norm_output", num_kv_heads).requires_grad_(True)
            layer_grad_out = attention_tensor(prefix + "grad_attention_output", num_heads)
            layer_k_repeated = repeat_kv_exact(layer_k)
            layer_v_repeated = repeat_kv_exact(layer_v)
            layer_scores = torch.matmul(layer_q, layer_k_repeated.transpose(2, 3))
            layer_scores = layer_scores + causal_mask
            layer_probs = torch.nn.functional.softmax(layer_scores, dim=-1, dtype=torch.float32)
            layer_attention = torch.matmul(layer_probs, layer_v_repeated)
            (layer_attention * layer_grad_out).sum().backward()
            per_layer_attention[f"layer{layer_index}"] = {
                "attention_output_vs_native": max_abs(
                    layer_attention,
                    attention_tensor(prefix + "attention_output", num_heads),
                ),
                "grad_q_vs_native": max_abs(
                    layer_q.grad,
                    attention_tensor(prefix + "grad_q_rotary_output", num_heads),
                ),
                "grad_k_vs_native": max_abs(
                    layer_k.grad,
                    attention_tensor(prefix + "grad_k_rotary_output", num_kv_heads),
                ),
                "grad_v_vs_native": max_abs(
                    layer_v.grad,
                    attention_tensor(prefix + "grad_v_norm_output", num_kv_heads),
                ),
            }
        gemma4_attention_replay["per_layer"] = per_layer_attention

        # The first two Gemma4 rotary calls are layer-0 Q and K respectively.
        # Compare both native boundaries directly to the HF graph, then replay
        # the inverse rotation with native operands/upstream adjoints.  This
        # separates attention-backward drift from RoPE-backward drift without
        # changing the production graph.
        if len(gemma4_rope_calls) >= 2:
            gemma4_rope_replay = {}
            gemma4_rope_replay["native_params"] = {
                "attention_factor": float(trace["debug_rope_attention_factor"][0]),
                "llama4_scaling_beta": float(trace["debug_rope_llama4_scaling_beta"][0]),
            }

            def rotate_half(value):
                half = value.shape[-1] // 2
                return torch.cat((-value[..., half:], value[..., :half]), dim=-1)

            for label, call, native_value_key, native_grad_output_key, native_grad_input_key in [
                (
                    "q",
                    gemma4_rope_calls[0],
                    "q_rotary_output",
                    "grad_q_rotary_output",
                    "grad_q_norm_output",
                ),
                (
                    "k",
                    gemma4_rope_calls[1],
                    "k_rotary_output",
                    "grad_k_rotary_output",
                    "grad_k_norm_output",
                ),
            ]:
                hf_input = call["input"]
                hf_output = call["output"]
                native_heads = num_heads if label == "q" else num_kv_heads
                native_input = rope_tensor(f"{label}_norm_output", native_heads)
                native_output = rope_tensor(native_value_key, native_heads)
                native_grad_output = rope_tensor(native_grad_output_key, native_heads)
                native_grad_input = rope_tensor(native_grad_input_key, native_heads)

                replay_input = native_input.detach().clone().requires_grad_(True)
                cos = call["cos"].unsqueeze(call["unsqueeze_dim"])
                sin = call["sin"].unsqueeze(call["unsqueeze_dim"])
                replay_output = (replay_input * cos) + (rotate_half(replay_input) * sin)
                (replay_output * native_grad_output).sum().backward()

                inferred_trig = None
                forward_delta = (native_output - replay_output).abs()
                worst_forward = None
                if float(forward_delta.max()) != 0.0:
                    # The native shader and HF both use the same two rounded
                    # products plus an add. Infer whether whichever Q/K branch
                    # still diverges is instead seeing different shared cos/sin
                    # coefficients by searching a small FP32 ULP neighborhood
                    # around HF's pair and scoring exact native-output
                    # reproduction across every Q/K head that shares that
                    # position/frequency.
                    worst_flat = int(forward_delta.reshape(-1).argmax())
                    b_idx, pos_idx, h_idx, dim_idx = np.unravel_index(
                        worst_flat, tuple(native_output.shape)
                    )
                    head_dim = native_output.shape[-1]
                    half = head_dim // 2
                    first_dim = dim_idx if dim_idx < half else dim_idx - half
                    second_dim = first_dim + half
                    hf_c = np.float32(cos[b_idx, pos_idx, 0, first_dim].item())
                    hf_s = np.float32(sin[b_idx, pos_idx, 0, first_dim].item())
                    worst_forward = {
                        "flat_index": worst_flat,
                        "index": [int(b_idx), int(pos_idx), int(h_idx), int(dim_idx)],
                        "native_input": float(native_input[b_idx, pos_idx, h_idx, dim_idx]),
                        "hf_input": float(hf_input[b_idx, pos_idx, h_idx, dim_idx]),
                        "native_output": float(native_output[b_idx, pos_idx, h_idx, dim_idx]),
                        "hf_output": float(hf_output[b_idx, pos_idx, h_idx, dim_idx]),
                        "torch_replay_output": float(replay_output[b_idx, pos_idx, h_idx, dim_idx]),
                        "hf_cos": float(hf_c),
                        "hf_sin": float(hf_s),
                    }

                    def ulp_neighbors(value, radius=16):
                        values = [np.float32(value)]
                        lo = np.float32(value)
                        hi = np.float32(value)
                        for _ in range(radius):
                            lo = np.nextafter(lo, np.float32(-np.inf), dtype=np.float32)
                            hi = np.nextafter(hi, np.float32(np.inf), dtype=np.float32)
                            values.extend([np.float32(lo), np.float32(hi)])
                        return values

                    q_input = rope_tensor("q_norm_output", num_heads).numpy()
                    q_native = rope_tensor("q_rotary_output", num_heads).numpy()
                    k_input = rope_tensor("k_norm_output", num_kv_heads).numpy()
                    k_native = rope_tensor("k_rotary_output", num_kv_heads).numpy()

                    def score_pair(c_value, s_value):
                        score = 0
                        for source, target in ((q_input, q_native), (k_input, k_native)):
                            for head in range(source.shape[2]):
                                x0 = np.float32(source[b_idx, pos_idx, head, first_dim])
                                x1 = np.float32(source[b_idx, pos_idx, head, second_dim])
                                y0 = np.float32(
                                    np.float32(x0 * c_value) + np.float32(-x1 * s_value)
                                )
                                y1 = np.float32(
                                    np.float32(x1 * c_value) + np.float32(x0 * s_value)
                                )
                                score += int(
                                    y0.view(np.uint32)
                                    == np.float32(target[b_idx, pos_idx, head, first_dim]).view(np.uint32)
                                )
                                score += int(
                                    y1.view(np.uint32)
                                    == np.float32(target[b_idx, pos_idx, head, second_dim]).view(np.uint32)
                                )
                        return score

                    best = None
                    for c_value in ulp_neighbors(hf_c):
                        for s_value in ulp_neighbors(hf_s):
                            candidate = (score_pair(c_value, s_value), c_value, s_value)
                            if best is None or candidate[0] > best[0]:
                                best = candidate
                    inferred_trig = {
                        "position": int(pos_idx),
                        "frequency_index": int(first_dim),
                        "hf_cos": float(hf_c),
                        "hf_sin": float(hf_s),
                        "hf_exact_outputs": score_pair(hf_c, hf_s),
                        "best_cos": float(best[1]),
                        "best_sin": float(best[2]),
                        "best_exact_outputs": int(best[0]),
                        "total_outputs": int(2 * (q_input.shape[2] + k_input.shape[2])),
                    }

                gemma4_rope_replay[label] = {
                    "native_input_vs_hf_max_abs": max_abs(native_input, hf_input),
                    "native_output_vs_hf_max_abs": max_abs(native_output, hf_output),
                    "native_grad_output_vs_hf_max_abs": max_abs(
                        native_grad_output, call["grad_output"]
                    ),
                    "native_grad_input_vs_hf_max_abs": max_abs(
                        native_grad_input, call["grad_input"]
                    ),
                    "torch_forward_from_native_input_vs_native_max_abs": max_abs(
                        replay_output, native_output
                    ),
                    "torch_backward_from_native_operands_vs_native_max_abs": max_abs(
                        replay_input.grad, native_grad_input
                    ),
                    "torch_backward_from_native_operands_vs_hf_max_abs": max_abs(
                        replay_input.grad, call["grad_input"]
                    ),
                    "worst_forward": worst_forward,
                    "inferred_trig": inferred_trig,
                }

            # The compact production trace also retains every layer's Q/K
            # rotary inputs and outputs.  Compare those forward boundaries
            # against the exact HF apply_rotary_pos_emb calls so a later-layer
            # attention mismatch is not incorrectly attributed to the
            # layer-0 backward replay above.
            for layer_index in layer_indices:
                call_base = layer_index * 2
                if call_base + 1 >= len(gemma4_rope_calls):
                    break
                for offset, label in enumerate(("q", "k")):
                    call = gemma4_rope_calls[call_base + offset]
                    hf_input = call["input"]
                    hf_output = call["output"]
                    native_heads = num_heads if label == "q" else num_kv_heads
                    native_input = rope_tensor(
                        f"layers.{layer_index}.{label}_norm_output", native_heads
                    )
                    native_output = rope_tensor(
                        f"layers.{layer_index}.{label}_rotary_output", native_heads
                    )
                    cos = call["cos"].unsqueeze(call["unsqueeze_dim"])
                    sin = call["sin"].unsqueeze(call["unsqueeze_dim"])
                    replay_output = (native_input * cos) + (rotate_half(native_input) * sin)
                    gemma4_rope_replay[f"layer{layer_index}_{label}_forward"] = {
                        "native_input_vs_hf_max_abs": max_abs(native_input, hf_input),
                        "native_output_vs_hf_max_abs": max_abs(native_output, hf_output),
                        "torch_forward_from_native_input_vs_native_max_abs": max_abs(
                            replay_output, native_output
                        ),
                    }
    head_reduction = None
    embedding_reduction = None
    if args.module == "lm_head" and "input" in head_capture and "grad_output" in head_capture:
        weight = expected[worst]
        worst_flat = int(top_indices[0])
        out_dim, in_dim = head_capture["weight_shape"]
        out_row, in_col = divmod(worst_flat, in_dim)
        hf_x = head_capture["input"].reshape(-1, in_dim)
        hf_g = head_capture["grad_output"].reshape(-1, out_dim)

        def sequential_products(g_col, x_col):
            value = torch.tensor(0.0, dtype=torch.float32)
            for row in range(g_col.numel()):
                value = value + g_col[row] * x_col[row]
            return float(value)

        def reverse_products(g_col, x_col):
            return sequential_products(g_col.flip(0), x_col.flip(0))

        def pairwise_products(g_col, x_col):
            values = [g_col[row] * x_col[row] for row in range(g_col.numel())]
            while len(values) > 1:
                values = [
                    values[index] + values[index + 1]
                    if index + 1 < len(values) else values[index]
                    for index in range(0, len(values), 2)
                ]
            return float(values[0]) if values else 0.0

        def kahan_products(g_col, x_col):
            value = torch.tensor(0.0, dtype=torch.float32)
            compensation = torch.tensor(0.0, dtype=torch.float32)
            for row in range(g_col.numel()):
                product = g_col[row] * x_col[row]
                corrected = product - compensation
                updated = value + corrected
                compensation = (updated - value) - corrected
                value = updated
            return float(value)

        def reductions(g_col, x_col):
            products = g_col * x_col
            return {
                "torch_sum": float(products.sum()),
                "torch_dot": float(torch.dot(g_col, x_col)),
                "matmul": float((g_col.reshape(1, -1) @ x_col.reshape(-1, 1)).item()),
                "sequential": sequential_products(g_col, x_col),
                "reverse": reverse_products(g_col, x_col),
                "pairwise": pairwise_products(g_col, x_col),
                "kahan": kahan_products(g_col, x_col),
                "double": float((g_col.double() * x_col.double()).sum()),
            }

        hf_g_col = hf_g[:, out_row].contiguous()
        hf_x_col = hf_x[:, in_col].contiguous()
        head_reduction = {
            "module_name": head_capture["module_name"],
            "weight_shape": list(head_capture["weight_shape"]),
            "rows": int(hf_g_col.numel()),
            "worst_flat_index": worst_flat,
            "out_row": out_row,
            "in_col": in_col,
            "native_gradient": float(actual[worst].flatten()[worst_flat]),
            "hf_gradient": float(weight.flatten()[worst_flat]),
            "hf_operands": reductions(hf_g_col, hf_x_col),
        }
        if "final_norm_output" in trace and "grad_logits" in trace:
            native_x = torch.tensor(trace["final_norm_output"], dtype=torch.float32).reshape_as(hf_x)
            native_g = torch.tensor(trace["grad_logits"], dtype=torch.float32).reshape_as(hf_g)
            x_delta = (native_x - hf_x).abs().flatten()
            g_delta = (native_g - hf_g).abs().flatten()
            x_max, x_index = torch.max(x_delta, dim=0)
            g_max, g_index = torch.max(g_delta, dim=0)
            native_x_col = native_x[:, in_col].contiguous()
            native_g_col = native_g[:, out_row].contiguous()
            head_reduction.update({
                "head_input_max_abs": float(x_max),
                "head_input_worst_index": int(x_index),
                "grad_logits_max_abs": float(g_max),
                "grad_logits_worst_index": int(g_index),
                "native_operands": reductions(native_g_col, native_x_col),
                "native_grad_hf_input": reductions(native_g_col, hf_x_col),
                "hf_grad_native_input": reductions(hf_g_col, native_x_col),
            })
    if embedding_capture.get("grad_output") is not None and "grad_token_embedding_output" in trace:
        hf_output_grad = embedding_capture["grad_output"]
        scale = embedding_capture["scale"]
        hf_weight_input_grad = hf_output_grad * scale
        native_weight_input_grad = torch.tensor(
            trace["grad_token_embedding_output"], dtype=torch.float32
        ).reshape_as(hf_weight_input_grad)
        boundary_delta = (native_weight_input_grad - hf_weight_input_grad).abs().flatten()
        boundary_max, boundary_index = torch.max(boundary_delta, dim=0)

        # Reconstruct the saved embedding's weight gradient using the same
        # source-order fold as the native sparse embedding kernel.  This tells
        # us whether any remaining drift is introduced by the scatter itself or
        # already exists in the gradient presented to it.
        embedding_weight_keys = [
            key for key in expected if key.endswith(f"{args.module}.weight")
        ]
        if len(embedding_weight_keys) != 1:
            raise RuntimeError(
                "expected one saved embedding weight gradient, found "
                f"{embedding_weight_keys}"
            )
        embedding_weight_key = embedding_weight_keys[0]
        expected_embedding_grad = expected[embedding_weight_key]
        reconstructed = torch.zeros_like(expected_embedding_grad)
        flat_ids = ids.reshape(-1)
        flat_grad = hf_weight_input_grad.reshape(flat_ids.numel(), -1)
        for position, token in enumerate(flat_ids.tolist()):
            reconstructed[token] = reconstructed[token] + flat_grad[position]
        reconstructed_delta = (reconstructed - expected_embedding_grad).abs().flatten()
        reconstructed_max, reconstructed_index = torch.max(reconstructed_delta, dim=0)
        embedding_reduction = {
            "module_name": embedding_capture["module_name"],
            "weight_key": embedding_weight_key,
            "scale": scale,
            "native_vs_hf_weight_input_grad_max_abs": float(boundary_max),
            "native_vs_hf_weight_input_grad_worst_index": int(boundary_index),
            "hf_source_order_reconstruction_max_abs": float(reconstructed_max),
            "hf_source_order_reconstruction_worst_index": int(reconstructed_index),
        }
    report = {"family": args.family, "module": args.module, "adapter": args.adapter,
              "gradient_max_abs": drift, "worst_tensor": worst,
              "top_differences": top_differences,
              "head_reduction": head_reduction,
              "embedding_reduction": embedding_reduction,
              "gemma4_internals": gemma4_comparisons,
              "gemma4_o_proj_replay": gemma4_o_proj_replay,
              "gemma4_qkv_projection_replay": qkv_projection_replay if args.gemma4_internals else None,
              "gemma4_attention_replay": gemma4_attention_replay,
              "gemma4_rope_replay": gemma4_rope_replay,
              "gemma4_forward_residual_replay": gemma4_forward_residual_replay,
              "artifact_sha256": artifact_hashes,
              "adapter_path": str(adapter),
              "trace_keys": sorted(trace.keys()),
              "output": str(output)}
    (output / "diagnostic.json").write_text(json.dumps(report, indent=2))
    if args.gemma4_internals and gemma4_comparisons is not None:
        compact = {
            "gradient_max_abs": drift,
            "worst_tensor": worst,
            "qkv_branch_orders": {
                key: value.get("qkv_branch_orders")
                for key, value in gemma4_comparisons.items()
                if value.get("qkv_branch_orders") is not None
            },
            "mlp_input_branch_orders": {
                key: value.get("input_branch_orders")
                for key, value in gemma4_comparisons.items()
                if value.get("input_branch_orders") is not None
            },
            "qkv_projection_sum": (
                qkv_projection_replay.get("sum")
                if qkv_projection_replay is not None
                else None
            ),
            "final_norm": gemma4_comparisons.get("final_norm"),
            "embedding_reduction": embedding_reduction,
            "forward_residual_replay": gemma4_forward_residual_replay,
        }
        print("GEMMA4_COMPACT=" + json.dumps(compact), flush=True)
    print(json.dumps(report), flush=True)
    if drift > 2e-7:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
