"""Offline, operand-matched replay of a captured Gemma4 backward graph."""
import argparse
import json
import math
import struct
from pathlib import Path

import torch
from gemma4_sleef_port_diag_tmp import sleef_pow_neg_three_halves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("--trace", default="trace.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    trace = json.loads((args.probe / args.trace).read_text())
    saved = torch.load(args.probe / "oracle-tensors.pt", weights_only=True)
    config = json.loads((args.probe.parent / "base/config.json").read_text())
    fixture = json.loads((args.probe.parent / "fixture.json").read_text())
    b, s = fixture["batch_size"], fixture["seq_len"]
    h, k, d = config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
    hf = saved["boundaries"]
    report = {}

    def tensor(key, shape):
        return torch.tensor(trace[key]).reshape(shape)

    def diff(x, y):
        return float((x.detach() - y.detach()).abs().max())

    def f32(value):
        return struct.unpack("<f", struct.pack("<f", float(value)))[0]

    if "final_norm" in hf:
        final_shape = hf["final_norm"].shape
        hf_final_input = hf["final_norm.input_value"]
        hf_final_grad_output = hf["final_norm.grad_output"]
        hf_final_grad_input = hf["final_norm.grad_input"]
        native_final_input = tensor("final_prefix", hf_final_input.shape)
        native_final_output = tensor("final_norm_output", final_shape)
        native_final_grad_output = tensor("grad_final_norm_output", hf_final_grad_output.shape)
        native_final_grad_input = tensor("grad_final_norm_input", hf_final_grad_input.shape)
        native_final_mean = tensor(
            "final_norm_mean", hf_final_input.shape[:-1] + (1,)
        )
        native_final_rstd = tensor(
            "final_norm_rstd", hf_final_input.shape[:-1] + (1,)
        )
        final_weight = saved["weights"]["base_model.model.model.norm.weight"]

        def final_norm_backward(input_value, grad_value):
            replay_input = input_value.detach().clone().requires_grad_()
            replay_mean = (
                replay_input.pow(2).mean(-1, keepdim=True)
                + config["rms_norm_eps"]
            )
            replay_output = (
                replay_input * replay_mean.pow(-0.5)
            ) * final_weight
            replay_output.backward(grad_value)
            return replay_input.grad, replay_mean, replay_output

        hf_replay_grad, hf_replay_mean, _ = final_norm_backward(
            hf_final_input, hf_final_grad_output
        )
        native_input_hf_grad, _, _ = final_norm_backward(
            native_final_input, hf_final_grad_output
        )
        native_operand_replay_grad, native_operand_mean, native_operand_output = (
            final_norm_backward(native_final_input, native_final_grad_output)
        )

        def dot_variant(left, right, *, fused, lanes):
            accumulators = [f32(0.0)] * lanes
            for index in range(left.numel()):
                lane = index % lanes
                if fused and hasattr(math, "fma"):
                    accumulators[lane] = f32(
                        math.fma(
                            float(left[index]),
                            float(right[index]),
                            accumulators[lane],
                        )
                    )
                else:
                    product = f32(float(left[index]) * float(right[index]))
                    accumulators[lane] = f32(accumulators[lane] + product)
            total = f32(0.0)
            for accumulator in accumulators:
                total = f32(total + accumulator)
            return total

        final_candidates = []
        flat_input = native_final_input.reshape(-1, native_final_input.shape[-1])
        flat_grad = native_final_grad_output.reshape_as(flat_input)
        flat_hf_grad_input = hf_final_grad_input.reshape_as(flat_input)
        flat_native_grad_input = native_final_grad_input.reshape_as(flat_input)
        flat_mean = native_final_mean.reshape(-1)
        flat_rstd = native_final_rstd.reshape(-1)
        weight_flat = final_weight.reshape(-1)
        for fused in (False, True):
            for lanes in (1, 2, 4, 8, 16):
                if lanes > flat_input.shape[-1]:
                    continue
                for power_mode in ("torch", "sleef"):
                    for grad_mean_assoc in ("left", "power_first"):
                        for mean_scale in ("divide", "reciprocal"):
                            for through_assoc in ("two_x", "two_grad"):
                                candidate = torch.empty_like(flat_input)
                                for row in range(flat_input.shape[0]):
                                    weighted = torch.empty_like(flat_input[row])
                                    for column in range(flat_input.shape[-1]):
                                        weighted[column] = f32(
                                            f32(flat_grad[row, column])
                                            * f32(weight_flat[column])
                                        )
                                    dot = dot_variant(
                                        weighted,
                                        flat_input[row],
                                        fused=fused,
                                        lanes=lanes,
                                    )
                                    mean_value = f32(flat_mean[row])
                                    power = (
                                        f32(float(native_operand_mean.pow(-1.5).reshape(-1)[row]))
                                        if power_mode == "torch"
                                        else f32(sleef_pow_neg_three_halves(mean_value))
                                    )
                                    if grad_mean_assoc == "left":
                                        grad_mean = f32(f32(dot * f32(-0.5)) * power)
                                    else:
                                        grad_mean = f32(dot * f32(f32(-0.5) * power))
                                    if mean_scale == "divide":
                                        grad_square = f32(
                                            grad_mean / f32(flat_input.shape[-1])
                                        )
                                    else:
                                        grad_square = f32(
                                            grad_mean
                                            * f32(1.0 / flat_input.shape[-1])
                                        )
                                    for column in range(flat_input.shape[-1]):
                                        direct = f32(
                                            f32(weighted[column]) * f32(flat_rstd[row])
                                        )
                                        if through_assoc == "two_x":
                                            through = f32(
                                                grad_square
                                                * f32(
                                                    f32(2.0)
                                                    * f32(flat_input[row, column])
                                                )
                                            )
                                        else:
                                            through = f32(
                                                f32(grad_square * f32(2.0))
                                                * f32(flat_input[row, column])
                                            )
                                        candidate[row, column] = f32(direct + through)
                                description = (
                                    f"fused={int(fused)},lanes={lanes},"
                                    f"power={power_mode},grad_mean={grad_mean_assoc},"
                                    f"mean_scale={mean_scale},through={through_assoc}"
                                )
                                final_candidates.append(
                                    {
                                        "description": description,
                                        "vs_hf": diff(candidate, flat_hf_grad_input),
                                        "vs_native": diff(
                                            candidate, flat_native_grad_input
                                        ),
                                    }
                                )
        final_candidates_by_hf = sorted(
            final_candidates, key=lambda item: (item["vs_hf"], item["vs_native"])
        )[:12]
        final_candidates_by_native = sorted(
            final_candidates, key=lambda item: (item["vs_native"], item["vs_hf"])
        )[:12]
        report["final_norm"] = {
            "input_native_vs_hf": diff(native_final_input, hf_final_input),
            "output_native_vs_hf": diff(native_final_output, hf["final_norm"]),
            "grad_output_native_vs_hf": diff(
                native_final_grad_output, hf_final_grad_output
            ),
            "grad_input_native_vs_hf": diff(
                native_final_grad_input, hf_final_grad_input
            ),
            "hf_replay_grad_vs_hf": diff(hf_replay_grad, hf_final_grad_input),
            "torch_native_input_hf_grad_vs_hf": diff(
                native_input_hf_grad, hf_final_grad_input
            ),
            "torch_native_operands_vs_native": diff(
                native_operand_replay_grad, native_final_grad_input
            ),
            "torch_native_operands_vs_hf": diff(
                native_operand_replay_grad, hf_final_grad_input
            ),
            "mean_native_vs_torch_native_input": diff(
                native_final_mean, native_operand_mean
            ),
            "rstd_native_vs_torch_native_input": diff(
                native_final_rstd, native_operand_mean.pow(-0.5)
            ),
            "output_native_vs_torch_native_operands": diff(
                native_final_output, native_operand_output
            ),
            "best_arithmetic_vs_hf": final_candidates_by_hf,
            "best_arithmetic_vs_native": final_candidates_by_native,
        }

    for layer in range(config["num_hidden_layers"]):
        prefix = f"layers.{layer}."
        label = f"layer{layer}."
        def attn(key, heads):
            return tensor(prefix + key, (b, s, heads, d)).transpose(1, 2).contiguous()
        q = attn("q_rotary_output", h).requires_grad_()
        key = attn("k_rotary_output", k).requires_grad_()
        v = attn("v_norm_output", k).requires_grad_()
        def repeat(x):
            return x[:, :, None].expand(b, k, h // k, s, d).reshape(b, h, s, d)
        scores = q @ repeat(key).transpose(-1, -2)
        mask = torch.triu(torch.full((s, s), torch.finfo(torch.float32).min), diagonal=1)
        probs = (scores + mask).softmax(-1)
        out = probs @ repeat(v)
        out.backward(attn("grad_attention_output", h))
        attention_grads = {"q": q.grad.transpose(1, 2), "k": key.grad.transpose(1, 2), "v": v.grad.transpose(1, 2)}
        q_grad_key = "grad_q_rotary_output" if layer == 0 else prefix + "grad_q_rotary_output"
        k_grad_key = "grad_k_rotary_output" if layer == 0 else prefix + "grad_k_rotary_output"
        v_grad_key = "grad_v_norm_output" if layer == 0 else prefix + "grad_v_norm_output"
        attention_report = {}
        if v_grad_key in trace:
            attention_report["v_grad"] = diff(
                attention_grads["v"], tensor(v_grad_key, attention_grads["v"].shape)
            )
        if q_grad_key in trace:
            attention_report["q_grad"] = diff(
                attention_grads["q"], tensor(q_grad_key, attention_grads["q"].shape)
            )
        if k_grad_key in trace:
            attention_report["k_grad"] = diff(
                attention_grads["k"], tensor(k_grad_key, attention_grads["k"].shape)
            )
        report[label + "attention"] = attention_report
        for branch, heads in (("q", h), ("k", k), ("v", k)):
            norm_label = label + branch + "_norm"
            shape = hf[norm_label + ".input_value"].shape
            x = tensor(prefix + branch + "_linear_output", shape).requires_grad_()
            grad = attention_grads[branch] if branch == "v" else tensor(prefix + "grad_" + branch + "_norm_output", shape)
            weight = saved["weights"].get(f"base_model.model.model.layers.{layer}.self_attn.{branch}_norm.weight", torch.ones(d))
            mean = x.square().mean(-1, keepdim=True) + config["rms_norm_eps"]
            y = x * mean.pow(-0.5) * weight
            y.backward(grad)
            dx = tensor(prefix + "grad_" + branch + "_linear_output", shape)
            report[norm_label] = {
                "local_dx": diff(x.grad, dx),
                "incoming": diff(grad, hf[norm_label + ".grad_output"]),
                "forward": diff(y, hf[norm_label]),
            }
            weighted = grad * weight
            products = weighted * x.detach()
            seqdot = torch.zeros_like(mean)
            for column in range(d):
                seqdot += products[..., column:column+1]
            derivative = mean.detach().pow(-1.5)
            port_derivative = torch.tensor([sleef_pow_neg_three_halves(value) for value in mean.detach().flatten().tolist()]).reshape_as(mean)
            report[norm_label]["pow_port_vs_torch"] = diff(port_derivative, derivative)
            exact_derivative = torch.tensor([math.pow(value, -1.5) for value in mean.detach().flatten().tolist()]).reshape_as(mean)
            report[norm_label]["pow_double_vs_torch"] = diff(exact_derivative, derivative)
            port_dx = weighted * mean.detach().pow(-0.5) + ((seqdot * -0.5) * port_derivative / d) * (2 * x.detach())
            report[norm_label]["port_dx_vs_native"] = diff(port_dx, dx)
            rstd = mean.detach().pow(-0.5)
            for name, dot in (("sequential", seqdot), ("torch", products.sum(-1, keepdim=True))):
                direct = weighted * rstd
                through = ((dot * -0.5) * derivative / d) * (2 * x.detach())
                candidate = direct + through
                report[norm_label][name] = {"vs_native": diff(candidate, dx), "vs_autograd": diff(candidate, x.grad)}
        for branch, heads in (("q", h), ("k", k)):
            call = saved["rope"][2 * layer + (branch == "k")]
            x = tensor(prefix + branch + "_norm_output", (b, s, heads, d)).requires_grad_()
            cos, sin = call["cos"].unsqueeze(2), call["sin"].unsqueeze(2)
            rotated = torch.cat((-x[..., d//2:], x[..., :d//2]), -1)
            y = x * cos + rotated * sin
            y.backward(attention_grads[branch])
            report[label + branch + "_rope"] = {"local_dx": diff(x.grad, tensor(prefix + "grad_" + branch + "_norm_output", x.shape))}

        # The decoder input participates in two uses: the attention pre-norm
        # and the direct first residual.  A Tensor hook on HF's
        # pre_feedforward_layernorm input therefore captures the already-
        # accumulated gradient of residual1, which is exactly the residual
        # operand consumed by the native residual-aware input RMS backward.
        # Replay the three relevant FP32 associations so a one-ulp decoder
        # mismatch can be attributed to the join rather than to attention.
        input_norm_label = label + "input_layernorm"
        residual1_label = label + "pre_feedforward_layernorm"
        decoder_label = label + "decoder"
        hf_input = hf[input_norm_label + ".input_value"]
        hf_norm_grad = hf[input_norm_label + ".grad_output"]
        hf_residual1_grad = hf[residual1_label + ".grad_input"]
        hf_decoder_grad = hf[decoder_label + ".grad_input"]
        input_weight = saved["weights"][
            f"base_model.model.model.layers.{layer}.input_layernorm.weight"
        ]
        hf_mean = hf_input.square().mean(-1, keepdim=True) + config["rms_norm_eps"]
        hf_rstd = hf_mean.pow(-0.5)
        hf_weighted = hf_norm_grad * input_weight
        hf_dot = (hf_weighted * hf_input).sum(-1, keepdim=True)
        hf_power = hf_mean.pow(-1.5)
        hf_grad_square = ((hf_dot * -0.5) * hf_power) / hf_input.shape[-1]
        hf_direct = hf_weighted * hf_rstd
        hf_through = hf_grad_square * (2 * hf_input)

        native_input = tensor(prefix + "input", hf_input.shape)
        native_norm_grad = tensor(prefix + "grad_norm_output", hf_norm_grad.shape)
        native_residual1_grad = tensor(prefix + "grad_residual1", hf_residual1_grad.shape)
        native_decoder_grad = tensor(prefix + "grad_input", hf_decoder_grad.shape)
        native_mean_key = (
            prefix + "norm_mean"
            if prefix + "norm_mean" in trace
            else "norm_mean" if layer == 0 and "norm_mean" in trace else None
        )
        native_rstd_key = (
            prefix + "norm_rstd"
            if prefix + "norm_rstd" in trace
            else "norm_rstd" if layer == 0 and "norm_rstd" in trace else None
        )
        native_mean = (
            tensor(native_mean_key, hf_mean.shape)
            if native_mean_key is not None
            else native_input.square().mean(-1, keepdim=True)
            + config["rms_norm_eps"]
        )
        native_rstd = (
            tensor(native_rstd_key, hf_rstd.shape)
            if native_rstd_key is not None
            else native_mean.pow(-0.5)
        )
        native_weighted = native_norm_grad * input_weight
        native_dot = (native_weighted * native_input).sum(-1, keepdim=True)
        native_power = torch.tensor(
            [
                sleef_pow_neg_three_halves(value)
                for value in native_mean.flatten().tolist()
            ],
            dtype=native_mean.dtype,
        ).reshape_as(native_mean)
        native_grad_square = (
            (native_dot * -0.5) * native_power
        ) / native_input.shape[-1]
        native_direct = native_weighted * native_rstd
        native_through = native_grad_square * (2 * native_input)

        def join_candidates(residual, direct, through):
            return {
                "residual_plus_direct_then_through": (residual + direct) + through,
                "residual_plus_branch": residual + (direct + through),
                "residual_plus_through_then_direct": (residual + through) + direct,
            }

        hf_candidates = join_candidates(hf_residual1_grad, hf_direct, hf_through)
        native_candidates = join_candidates(
            native_residual1_grad, native_direct, native_through
        )
        report[label + "first_residual_join"] = {
            "residual_operand_native_vs_hf": diff(
                native_residual1_grad, hf_residual1_grad
            ),
            "norm_grad_operand_native_vs_hf": diff(
                native_norm_grad, hf_norm_grad
            ),
            "input_operand_native_vs_hf": diff(native_input, hf_input),
            "mean_native_vs_hf_replay": diff(native_mean, hf_mean),
            "rstd_native_vs_hf_replay": diff(native_rstd, hf_rstd),
            "decoder_grad_native_vs_hf": diff(
                native_decoder_grad, hf_decoder_grad
            ),
            "hf_candidates_vs_hf": {
                name: diff(value, hf_decoder_grad)
                for name, value in hf_candidates.items()
            },
            "native_candidates_vs_native": {
                name: diff(value, native_decoder_grad)
                for name, value in native_candidates.items()
            },
            "native_candidates_vs_hf": {
                name: diff(value, hf_decoder_grad)
                for name, value in native_candidates.items()
            },
        }

        # Gemma4 applies a second RMSNorm to the MLP output immediately before
        # the residual add.  Saved-embedding probes are particularly sensitive
        # to this derivative because the MLP output can be much larger than the
        # residual adjoint. Replay it from the exact captured forward operand so
        # we can distinguish local RMS arithmetic from upstream gradient drift.
        norm_label = label + "post_feedforward_layernorm"
        hf_norm = hf[norm_label]
        x = tensor(prefix + "mlp_output", hf_norm.shape).requires_grad_()
        grad = hf[norm_label + ".grad_output"]
        weight = saved["weights"][
            f"base_model.model.model.layers.{layer}.post_feedforward_layernorm.weight"
        ]
        mean = x.square().mean(-1, keepdim=True) + config["rms_norm_eps"]
        rstd = mean.pow(-0.5)
        y = x * rstd * weight
        y.backward(grad)
        native_dx = tensor(prefix + "grad_post_mlp_norm", x.shape)
        native_grad_key = "grad_final_norm_input" if layer + 1 == config["num_hidden_layers"] else f"layers.{layer + 1}.grad_input"
        native_grad = tensor(native_grad_key, grad.shape)
        x_native_grad = tensor(prefix + "mlp_output", hf_norm.shape).requires_grad_()
        native_mean = x_native_grad.square().mean(-1, keepdim=True) + config["rms_norm_eps"]
        native_y = x_native_grad * native_mean.pow(-0.5) * weight
        native_y.backward(native_grad)
        products = (grad * weight) * x.detach()
        torchdot = products.sum(-1, keepdim=True)
        seqdot = torch.zeros_like(mean)
        for column in range(x.shape[-1]):
            seqdot += products[..., column : column + 1]
        lane8 = torch.zeros((*mean.shape[:-1], 8), dtype=x.dtype)
        for column in range(x.shape[-1]):
            lane8[..., column & 7] += products[..., column]
        lane8dot = torch.zeros_like(mean)
        for lane in range(8):
            lane8dot += lane8[..., lane : lane + 1]

        reduction_candidates = {}
        for lane_count in (1, 2, 4, 8, 16, 32):
            if lane_count > x.shape[-1]:
                continue
            lanes = torch.zeros((*mean.shape[:-1], lane_count), dtype=x.dtype)
            for column in range(x.shape[-1]):
                lanes[..., column % lane_count] += products[..., column]

            lane_seq = torch.zeros_like(mean)
            for lane in range(lane_count):
                lane_seq += lanes[..., lane : lane + 1]

            lane_tree_values = [lanes[..., lane : lane + 1].clone() for lane in range(lane_count)]
            while len(lane_tree_values) > 1:
                next_values = []
                for index in range(0, len(lane_tree_values), 2):
                    if index + 1 < len(lane_tree_values):
                        next_values.append(lane_tree_values[index] + lane_tree_values[index + 1])
                    else:
                        next_values.append(lane_tree_values[index])
                lane_tree_values = next_values
            lane_tree = lane_tree_values[0]
            reduction_candidates[f"lane{lane_count}_seq"] = diff(lane_seq, torchdot)
            reduction_candidates[f"lane{lane_count}_tree"] = diff(lane_tree, torchdot)

        # x86 vector reductions often combine low/high halves before the final
        # horizontal tree. Keep that ordering explicit for 8- and 16-lane
        # candidates instead of assuming adjacent-pair reduction.
        for lane_count in (8, 16):
            if lane_count > x.shape[-1]:
                continue
            lanes = torch.zeros((*mean.shape[:-1], lane_count), dtype=x.dtype)
            for column in range(x.shape[-1]):
                lanes[..., column % lane_count] += products[..., column]
            half = lane_count // 2
            folded = [lanes[..., i : i + 1] + lanes[..., i + half : i + half + 1] for i in range(half)]
            while len(folded) > 1:
                folded = [
                    folded[i] + folded[i + 1] if i + 1 < len(folded) else folded[i]
                    for i in range(0, len(folded), 2)
                ]
            reduction_candidates[f"lane{lane_count}_halves_tree"] = diff(folded[0], torchdot)

        derivative = mean.detach().pow(-1.5)
        port_derivative = torch.tensor(
            [sleef_pow_neg_three_halves(value) for value in mean.detach().flatten().tolist()]
        ).reshape_as(mean)

        def rms_dx(dot, power):
            weighted = grad * weight
            grad_mean = (dot * -0.5) * power
            grad_square = grad_mean / x.shape[-1]
            return weighted * rstd + grad_square * (2 * x.detach())

        # PyTorch autograd preserves the primitive-operation graph rather than
        # an algebraically simplified RMS derivative.  Enumerate the handful
        # of FP32 association choices at the pow/mean/pow chain so a captured
        # native mismatch can be attributed to one concrete primitive order.
        def rms_dx_association(dot, power, *, pow_first, reciprocal_mean, two_first):
            weighted = grad * weight
            if pow_first:
                grad_mean = dot * (-0.5 * power)
            else:
                grad_mean = (dot * -0.5) * power
            if reciprocal_mean:
                grad_square = grad_mean * (1.0 / x.shape[-1])
            else:
                grad_square = grad_mean / x.shape[-1]
            if two_first:
                through = (grad_square * 2.0) * x.detach()
            else:
                through = grad_square * (2.0 * x.detach())
            return weighted * rstd + through

        association_candidates = {}
        for pow_first in (False, True):
            for reciprocal_mean in (False, True):
                for two_first in (False, True):
                    candidate = rms_dx_association(
                        seqdot,
                        derivative,
                        pow_first=pow_first,
                        reciprocal_mean=reciprocal_mean,
                        two_first=two_first,
                    )
                    key = (
                        f"pow_first={int(pow_first)},"
                        f"reciprocal_mean={int(reciprocal_mean)},"
                        f"two_first={int(two_first)}"
                    )
                    association_candidates[key] = {
                        "vs_native": diff(candidate, native_dx),
                        "vs_autograd": diff(candidate, x.grad),
                    }

        # Mirror the GLSL pow-RMS backward one scalar FP32 operation at a time.
        # Tensor expressions can use vectorized CPU kernels with a different
        # contraction/association policy, so this distinguishes a source/SPIR-V
        # mismatch from a genuine disagreement with the PyTorch autograd graph.
        # These per-layer scalar buffers are not part of the compact trace.
        # The captured post-MLP forward output is exact, and the same operand
        # replay above reconstructs the reference norm exactly, so these are
        # the native-equivalent forward scalars for the backward experiment.
        native_mean_trace = mean.detach()
        native_rstd_trace = rstd.detach()
        scalar_shader = torch.empty_like(native_dx)
        flat_x = x.detach().reshape(-1, x.shape[-1])
        flat_grad = grad.detach().reshape(-1, x.shape[-1])
        flat_weight = weight.detach().reshape(-1)
        flat_mean = native_mean_trace.detach().reshape(-1)
        flat_rstd = native_rstd_trace.detach().reshape(-1)
        flat_out = scalar_shader.reshape(-1, x.shape[-1])
        for row in range(flat_x.shape[0]):
            dot = f32(0.0)
            weighted_values = []
            for column in range(x.shape[-1]):
                weighted = f32(f32(flat_grad[row, column]) * f32(flat_weight[column]))
                weighted_values.append(weighted)
                product = f32(weighted * f32(flat_x[row, column]))
                dot = f32(dot + product)
            power = f32(sleef_pow_neg_three_halves(f32(flat_mean[row])))
            grad_mean = f32(f32(dot * f32(-0.5)) * power)
            grad_square = f32(grad_mean / f32(x.shape[-1]))
            for column in range(x.shape[-1]):
                direct = f32(weighted_values[column] * f32(flat_rstd[row]))
                two_x = f32(f32(2.0) * f32(flat_x[row, column]))
                through = f32(grad_square * two_x)
                flat_out[row, column] = f32(direct + through)

        report[norm_label] = {
            "forward": diff(y, hf_norm),
            "torch_autograd_vs_native": diff(x.grad, native_dx),
            "torch_autograd_vs_hf": diff(x.grad, hf[norm_label + ".grad_input"]),
            "native_vs_hf": diff(native_dx, hf[norm_label + ".grad_input"]),
            "incoming_native_vs_hf": diff(native_grad, grad),
            "torch_with_native_incoming_vs_native": diff(x_native_grad.grad, native_dx),
            "pow_port_vs_torch": diff(port_derivative, derivative),
            "seq_torch_power_vs_native": diff(rms_dx(seqdot, derivative), native_dx),
            "seq_port_power_vs_native": diff(rms_dx(seqdot, port_derivative), native_dx),
            "torchdot_vs_seqdot": diff(torchdot, seqdot),
            "torchdot_torch_power_vs_native": diff(rms_dx(torchdot, derivative), native_dx),
            "torchdot_torch_power_vs_autograd": diff(rms_dx(torchdot, derivative), x.grad),
            "dim": x.shape[-1],
            "reduction_candidates_vs_torchdot": reduction_candidates,
            "lane8_torch_power_vs_native": diff(rms_dx(lane8dot, derivative), native_dx),
            "lane8_port_power_vs_native": diff(rms_dx(lane8dot, port_derivative), native_dx),
            "association_candidates": association_candidates,
            "scalar_shader_vs_native": diff(scalar_shader, native_dx),
            "scalar_shader_vs_autograd": diff(scalar_shader, x.grad),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
