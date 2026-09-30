"""Replay a tiny unscaled causal attention trace; never writes qualification reports.

This diagnostic compares identical recorded native Q/K/V operands. Intermediate
GPU scores are not recorded: the shader arithmetic below is a CPU replay, whose
agreement with the recorded output is reported separately.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


F = np.float32
runtime = ctypes.CDLL("ucrtbase" if os.name == "nt" else "libm.so.6")
runtime.fmaf.argtypes = [ctypes.c_float] * 3
runtime.fmaf.restype = ctypes.c_float


def fma(a, b, c):
    return F(runtime.fmaf(float(a), float(b), float(c)))


def shader_exp(d):
    q = int(np.rint(F(d * F(1.44269504088896340736))))
    s = fma(F(q), F(-0.693145751953125), d)
    s = fma(F(q), F(-1.428606765330187045e-06), s)
    u = F(0.000198527617612853646278381)
    for coefficient in [0.00139304355252534151077271, 0.00833336077630519866943359,
                        0.0416664853692054748535156, 0.166666671633720397949219, 0.5]:
        u = fma(u, s, F(coefficient))
    return F(F(1.0 + fma(F(s * s), u, s)) * F(2.0 ** q))


def compare(label, actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    delta = np.abs(actual - expected)
    index = np.unravel_index(int(delta.argmax()), delta.shape)
    print(json.dumps({"comparison": label, "max_abs": float(delta[index]),
                      "index": [int(i) for i in index], "actual": float(actual[index]),
                      "expected": float(expected[index]),
                      "different_elements": int(np.count_nonzero(delta))}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--fixture-root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    shader_root = Path(__file__).resolve().parents[1] / "shaders"
    print(json.dumps({
        "trace": str(args.trace.resolve()), "torch_version": torch.__version__,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "warning": "CPU replay of shader SOURCE, not an observation of GPU intermediates; verify embedded SPIR-V matches source",
        "attention_artifacts": {
            name: hashlib.sha256((shader_root / name).read_bytes()).hexdigest()
            for name in ["transformer_attention_forward.comp", "transformer_attention_forward.spv"]
        },
    }), flush=True)
    trace = json.loads(args.trace.read_text())
    config = json.loads((args.fixture_root / "base/config.json").read_text())
    fixture = json.loads((args.fixture_root / "fixture.json").read_text())
    batch, seq = fixture["batch_size"], fixture["seq_len"]
    heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
    dim = config.get("head_dim", config["hidden_size"] // heads)
    # The present trace records the first MiniMax M2 layer before projection.
    if config["model_type"] != "minimax_m2" or not all(fixture["attention_mask"]):
        raise ValueError("diagnostic currently requires an unpadded MiniMax M2 fixture")
    def tensor(key, n_heads):
        return torch.tensor(trace[key]).reshape(batch, seq, n_heads, dim).transpose(1, 2)
    q = tensor("q_rotary_output", heads)
    k = tensor("k_rotary_output", kv_heads).repeat_interleave(heads // kv_heads, dim=1)
    v = tensor("v_norm_input", kv_heads).repeat_interleave(heads // kv_heads, dim=1)
    native = tensor("attention_output", heads).numpy()
    scale = F(dim ** -0.5)
    scores = (q @ k.transpose(-2, -1)) * float(scale)
    shader_scores = np.zeros(tuple(scores.shape), dtype=np.float32)
    for b, h, i, j in np.ndindex(shader_scores.shape):
        value = F(0)
        for d in range(dim):
            value = fma(q[b, h, i, d], k[b, h, j, d], value)
        shader_scores[b, h, i, j] = F(value * scale)
    compare("score_sequential_fma_vs_torch_bmm", shader_scores, scores.numpy())
    visible = np.broadcast_to(np.tril(np.ones((seq, seq), dtype=bool)), shader_scores.shape).copy()
    visible[:, :, 0, :] = False  # A one-key softmax cannot expose its score drift.
    if visible.any():
        compare("visible_multi_key_scores_fma_vs_torch", shader_scores[visible], scores.numpy()[visible])
    unfused_scores = np.zeros_like(shader_scores)
    for b, h, i, j in np.ndindex(unfused_scores.shape):
        value = F(0)
        for d in range(dim):
            product = F(F(q[b, h, i, d]) * F(k[b, h, j, d]))
            value = F(value + product)
        unfused_scores[b, h, i, j] = F(value * scale)
    compare("score_materialized_products_vs_torch_bmm", unfused_scores, scores.numpy())
    mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    probs = scores.masked_fill(mask, torch.finfo(torch.float32).min).softmax(-1)
    shader_probs = np.zeros_like(shader_scores)
    for b, h, i in np.ndindex(batch, heads, seq):
        row = shader_scores[b, h, i, :i + 1]
        exp = [shader_exp(F(x - row.max())) for x in row]
        total = F(0)
        for x in exp:
            total = F(total + x)
        inverse = F(F(1) / total)
        shader_probs[b, h, i, :i + 1] = [F(x * inverse) for x in exp]
    compare("shader_softmax_replay_vs_torch", shader_probs, probs.numpy())
    torch_from_shader_scores = torch.tensor(shader_scores).masked_fill(mask, -float("inf")).softmax(-1)
    compare("softmax_identical_scores", shader_probs, torch_from_shader_scores.numpy())
    compare("torch_attention_vs_native", (probs @ v).numpy(), native)
    compare("torch_bmm_shader_probs_vs_native", (torch.tensor(shader_probs) @ v).numpy(), native)
    replay = np.zeros_like(native)
    for b, h, i, d in np.ndindex(replay.shape):
        value = F(0)
        for j in range(i + 1):
            value = fma(shader_probs[b, h, i, j], v[b, h, j, d], value)
        replay[b, h, i, d] = value
    compare("complete_shader_replay_vs_native", replay, native)
    compare("value_bmm_identical_probabilities", replay, (torch.tensor(shader_probs) @ v).numpy())
    unfused_value = np.zeros_like(native)
    for b, h, i, d in np.ndindex(unfused_value.shape):
        value = F(0)
        for j in range(i + 1):
            value = F(value + F(shader_probs[b, h, i, j] * F(v[b, h, j, d])))
        unfused_value[b, h, i, d] = value
    compare("value_materialized_products_vs_torch_bmm", unfused_value,
            (torch.tensor(shader_probs) @ v).numpy())

    if "grad_attention_output" in trace and "grad_v_linear_output" in trace:
        grad_out = tensor("grad_attention_output", heads).numpy()
        native_grad_v = tensor("grad_v_linear_output", kv_heads).numpy()
        for label, probabilities in [("torch", probs.numpy()), ("sleef_replay", shader_probs)]:
            repeated_grad = torch.tensor(probabilities).transpose(-2, -1) @ torch.tensor(grad_out)
            torch_grad = repeated_grad.reshape(batch, kv_heads, heads // kv_heads, seq, dim).sum(2)
            compare(label + "_grad_v_torch_bmm_vs_native", torch_grad.numpy(), native_grad_v)
            for fused in (False, True):
                result = np.zeros_like(native_grad_v)
                for b, kh, key, d in np.ndindex(result.shape):
                    total = F(0)
                    for h in range(kh * (heads // kv_heads), (kh + 1) * (heads // kv_heads)):
                        for query in range(key, seq):
                            p, g = probabilities[b, h, query, key], grad_out[b, h, query, d]
                            total = fma(p, g, total) if fused else F(total + F(p * g))
                    result[b, kh, key, d] = total
                compare(label + ("_fma" if fused else "_muladd") + "_grad_v_vs_native", result, native_grad_v)


if __name__ == "__main__":
    main()
