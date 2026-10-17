#!/usr/bin/env python3
"""Compare the Vulkan RWKV cell against the CPU runtime on one tiny package.

Hierarchos is its own architecture: there is no Transformers oracle for the
coherent-v9 RWKV cell, so the reference here is the independent pure-Rust
implementation in `hierarchos-inference`. This harness

1. generates a deterministic tiny coherent-v9 package (`model.safetensors`) that
   both runtimes load through their own loaders,
2. runs a short recurrent chain on the CPU probe (`rwkv-cell-probe`) and on the
   Vulkan packed-cell runner with the same inputs, each step consuming that
   backend's own packed state from the previous step,
3. compares the output vector and the packed next state after every step, and
4. runs a perturbed negative control so a broken comparison cannot pass silently.

The two paths share only the file format and the documented packed-state layout,
so a mismatch is a real disagreement rather than a shared-bug blind spot.

Three state-clamp modes run back to back:

* `0.0` - clamp disabled. The Vulkan shaders read `state_clamp <= 0` as "no
  clamp"; the CPU runtime is required to agree (`finite_clamp`), so this mode
  would fail loudly if either side treated zero as "clamp to zero".
* `50.0` - the production `recurrent_state_clamp` default; nothing clamps in
  practice, so this is the realistic operating point.
* `0.25` - the clamp is active on both sides, which exercises the clamp path
  (and the harness asserts the returned state really is bounded).

The package uses `width=128` and `batch=4`, which clears the Intel Gen9 matmul
gate (`rows >= 4`, `input_dim >= 32`, `output_dim >= 128`). The packed-cell
runner resolves the fused packed-cell shaders rather than the vendor linear
modules, so `variant-dispatch:` is expected to be empty; the harness records the
plan and the dispatch counts so nobody reads this agreement as vendor-module
coverage it never had.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
INFERENCE_MANIFEST = REPO / "hierarchos-inference" / "Cargo.toml"
VULKAN_MANIFEST = ROOT / "Cargo.toml"

CELL_PREFIX = "h_rnn"
ADAPTER_PREFIX = "h_embed"
WIDTH = 128
HEAD_SIZE = 16
BATCH = 4
INPUT_DIM = 128
ADAPTER_RANK = 8
STEPS = 3
# Mirrors the runner's own clamps (`rwkv-packed-cell-step` passes 12.0/4.0) and
# the CPU model defaults, so both sides clamp identically.
KEY_CLAMP = 12.0
DEEPEMBED_CLAMP = 4.0
# (clamp, name, binding): `disabled` must not clamp, `production` must stay far
# from the limit, and `active` must land exactly on it so the clamp path is
# exercised rather than merely configured.
STATE_CLAMP_MODES = ((0.0, "disabled", "disabled"), (50.0, "production", "unbound"), (0.25, "active", "bound"))


def rwkv_lora_rank(width: int, scale: float) -> int:
    """The rank rule both loaders agree on (`hierarchos-inference::rwkv`)."""
    if width < 128:
        return 8
    raw = scale * (width**0.5)
    return max(int(round(raw / 32.0)), 1) * 32


def build_binary(manifest: Path, name: str) -> Path:
    result = subprocess.run(
        [
            "cargo",
            "build",
            "--manifest-path",
            str(manifest),
            "--bin",
            name,
            "--message-format=json-render-diagnostics",
        ],
        cwd=REPO,
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


def tensor(shape: tuple[int, ...], *, generator) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.1


def write_package(directory: Path) -> None:
    """Write the tensors both loaders read, with the shared naming/layout."""
    generator = torch.Generator().manual_seed(0x48494552)
    width = WIDTH
    w_rank = rwkv_lora_rank(width, 2.5)
    a_rank = rwkv_lora_rank(width, 2.5)
    g_rank = rwkv_lora_rank(width, 5.0)
    tensors: dict[str, torch.Tensor] = {}
    for name in (
        "x_r",
        "x_w",
        "x_k",
        "x_v",
        "x_a",
        "x_g",
        "w0",
        "a0",
        "k_k",
        "k_a",
        "r_k",
        "x_k_cm",
        "ln1.weight",
        "ln1.bias",
        "ln2.weight",
        "ln2.bias",
        "ln_x.weight",
        "ln_x.bias",
    ):
        value = tensor((width,), generator=generator)
        if name.endswith("ln1.weight") or name.endswith("ln2.weight") or name.endswith("ln_x.weight"):
            value = value.abs() + 0.5
        tensors[f"{CELL_PREFIX}.{name}"] = value
    for name, shape in (
        ("w1", (width, w_rank)),
        ("w2", (w_rank, width)),
        ("a1", (width, a_rank)),
        ("a2", (a_rank, width)),
        ("g1", (width, g_rank)),
        ("g2", (g_rank, width)),
        ("receptance.weight", (width, width)),
        ("key.weight", (width, width)),
        ("value.weight", (width, width)),
        ("output.weight", (width, width)),
        ("key_cm.weight", (4 * width, width)),
        ("value_cm.weight", (width, 4 * width)),
    ):
        tensors[f"{CELL_PREFIX}.{name}"] = tensor(shape, generator=generator)
    # The adapter is neutralised deliberately: `up = 0` makes its output exactly
    # `bias`, so the CPU side can be handed the same DeepEmbed vector without
    # either runtime re-deriving the adapter.
    tensors[f"{ADAPTER_PREFIX}.down.weight"] = tensor((ADAPTER_RANK, INPUT_DIM), generator=generator)
    tensors[f"{ADAPTER_PREFIX}.up.weight"] = torch.zeros((4 * width, ADAPTER_RANK), dtype=torch.float32)
    tensors[f"{ADAPTER_PREFIX}.bias"] = torch.ones((4 * width,), dtype=torch.float32)
    directory.mkdir(parents=True, exist_ok=True)
    save_file({name: value.contiguous() for name, value in tensors.items()}, directory / "model.safetensors")


def make_inputs() -> dict[str, object]:
    generator = torch.Generator().manual_seed(0x48494553)
    state_size = 4 + HEAD_SIZE
    initial_state = tensor((BATCH, WIDTH, state_size), generator=generator).reshape(-1)
    xs = [tensor((BATCH, WIDTH), generator=generator).reshape(-1).tolist() for _ in range(STEPS)]
    return {"initial_state": initial_state.tolist(), "xs": xs}


def run_vulkan(binary: Path, model_dir: Path, payload: dict, case_path: Path) -> dict:
    case_path.write_text(json.dumps(payload), encoding="utf-8")
    env = dict(os.environ, HIERARCHOS_VULKAN_VARIANT_TRACE="1")
    completed = subprocess.run(
        [
            str(binary),
            "--case",
            str(case_path),
            "--model-dir",
            str(model_dir),
            "--cell-prefix",
            CELL_PREFIX,
            "--adapter-prefix",
            ADAPTER_PREFIX,
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    result = json.loads(completed.stdout)
    trace = {}
    for line in completed.stderr.splitlines():
        for key in ("variant-plan", "variant-dispatch"):
            if line.startswith(f"{key}:"):
                trace[key] = line.split(":", 1)[1].strip()
    result["_trace"] = trace
    return result


def run_cpu(binary: Path, model_dir: Path, payload: dict, case_path: Path) -> dict:
    case_path.write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(
        [str(binary), "--case", str(case_path), "--model-dir", str(model_dir), "--cell-prefix", CELL_PREFIX],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


class Mismatch(AssertionError):
    pass


def compare(label: str, vulkan: list[float], cpu: list[float], atol: float, rtol: float) -> dict:
    if len(vulkan) != len(cpu):
        raise Mismatch(f"{label}: Vulkan {len(vulkan)} values != CPU {len(cpu)}")
    max_abs = 0.0
    for index, (left, right) in enumerate(zip(vulkan, cpu, strict=True)):
        if not (left == left and right == right):
            raise Mismatch(f"{label}: non-finite comparison at index {index} (vulkan={left!r} cpu={right!r})")
        delta = abs(left - right)
        max_abs = max(max_abs, delta)
        bound = atol + rtol * max(abs(left), abs(right))
        if delta > bound:
            raise Mismatch(
                f"{label}: |vulkan-cpu| {delta:.9e} exceeds "
                f"atol {atol:.3e} + rtol {rtol:.3e}*scale (bound {bound:.9e}) at index {index}; "
                f"vulkan={left!r} cpu={right!r}"
            )
    print(f"  {label}: {len(vulkan)} values, max_abs {max_abs:.9e}")
    return {"label": label, "values": len(vulkan), "max_abs": max_abs, "atol": atol, "rtol": rtol}


def vulkan_payload(x: list[float], packed_state: list[float], state_clamp: float) -> dict:
    return {
        "batch": BATCH,
        "width": WIDTH,
        "head_size": HEAD_SIZE,
        "input_dim": INPUT_DIM,
        "state_mode": "explicit-output",
        "state_clamp": state_clamp,
        "x": x,
        "packed_state": packed_state,
        "token_features": [0.0] * (BATCH * INPUT_DIM),
        "grad_output": [0.0] * (BATCH * WIDTH),
        "grad_packed_new_state": [0.0] * len(packed_state),
    }


def cpu_payload(x: list[float], packed_state: list[float], state_clamp: float) -> dict:
    return {
        "batch": BATCH,
        "width": WIDTH,
        "head_size": HEAD_SIZE,
        "state_clamp": state_clamp,
        "key_clamp": KEY_CLAMP,
        "deepembed_clamp": DEEPEMBED_CLAMP,
        "x": x,
        "packed_state": packed_state,
        # `up = 0, bias = 1`: the adapter output is exactly one for every row.
        "deepembed": [1.0] * (BATCH * WIDTH * 4),
    }


def run_chain(
    vulkan_binary: Path,
    cpu_binary: Path,
    model_dir: Path,
    work: Path,
    xs: list[list[float]],
    initial_state: list[float],
    state_clamp: float,
    binding: str,
    atol: float,
    rtol: float,
) -> dict:
    vulkan_state = list(initial_state)
    cpu_state = list(initial_state)
    comparisons = []
    trace: dict[str, str] = {}
    for step, x in enumerate(xs):
        vulkan = run_vulkan(
            vulkan_binary, model_dir, vulkan_payload(x, vulkan_state, state_clamp), work / "vulkan-case.json"
        )
        cpu = run_cpu(cpu_binary, model_dir, cpu_payload(x, cpu_state, state_clamp), work / "cpu-case.json")
        if step == 0:
            trace = vulkan["_trace"]
        comparisons.append(compare(f"step {step} cell output", vulkan["output"], cpu["output"], atol, rtol))
        comparisons.append(
            compare(f"step {step} packed next state", vulkan["packed_new_state"], cpu["packed_new_state"], atol, rtol)
        )
        for side, values in (("vulkan", vulkan["packed_new_state"]), ("cpu", cpu["packed_new_state"])):
            peak = max(abs(value) for value in values)
            bound = state_clamp * (1.0 + 1e-6)
            if binding == "bound" and peak < state_clamp * (1.0 - 1e-6):
                raise Mismatch(
                    f"{side} state peak {peak:.9e} never reaches the {state_clamp} clamp; "
                    "the active mode would not exercise the clamp path"
                )
            if binding == "unbound" and peak >= bound:
                raise Mismatch(
                    f"{side} state peak {peak:.9e} touches the production clamp {state_clamp}; "
                    "the fixture no longer matches the production regime"
                )
            if binding == "disabled" and state_clamp <= 0.0 and peak == 0.0:
                raise Mismatch(f"{side} state is all zeros under a disabled clamp")
        # Hand each backend its own state back, exactly as a live session would.
        vulkan_state = vulkan["packed_new_state"]
        cpu_state = cpu["packed_new_state"]
    return {"state_clamp": state_clamp, "comparisons": comparisons, "trace": trace}


def negative_control(
    vulkan_binary: Path,
    cpu_binary: Path,
    model_dir: Path,
    work: Path,
    xs: list[list[float]],
    initial_state: list[float],
    atol: float,
    rtol: float,
) -> dict:
    """A deliberately perturbed CPU input must fail the same comparison.

    Without this the harness could pass because `compare` never rejects
    anything; the perturbation is far above fp32 reordering noise but far below
    a structural bug, so it pins the tolerance's discriminating power.
    """
    x = xs[0]
    vulkan = run_vulkan(vulkan_binary, model_dir, vulkan_payload(x, initial_state, 0.0), work / "vulkan-case.json")
    perturbed = [value * 1.01 for value in x]
    cpu = run_cpu(cpu_binary, model_dir, cpu_payload(perturbed, initial_state, 0.0), work / "cpu-case.json")
    try:
        compare("negative control", vulkan["output"], cpu["output"], atol, rtol)
    except Mismatch as exc:
        print(f"  negative control rejected the perturbation as expected: {exc}")
        return {"result": "rejected", "detail": str(exc)}
    raise Mismatch("negative control passed: the comparison does not discriminate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atol", type=float, default=1.0e-4, help="absolute part of the agreement bound")
    parser.add_argument("--rtol", type=float, default=1.0e-5, help="relative part of the agreement bound")
    parser.add_argument("--keep-fixtures", type=Path)
    args = parser.parse_args()

    vulkan_binary = build_binary(VULKAN_MANIFEST, "hierarchos-vulkan-rwkv-packed-cell-step")
    cpu_binary = build_binary(INFERENCE_MANIFEST, "rwkv-cell-probe")
    cleanup = None
    if args.keep_fixtures is not None:
        fixture_root = args.keep_fixtures.resolve()
        fixture_root.mkdir(parents=True, exist_ok=True)
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="hierarchos-rwkv-bridge-")
        fixture_root = Path(cleanup.name)
    try:
        model_dir = fixture_root / "model"
        write_package(model_dir)
        inputs = make_inputs()
        xs = inputs["xs"]
        initial_state = inputs["initial_state"]
        modes = []
        for state_clamp, mode_name, binding in STATE_CLAMP_MODES:
            print(f"state_clamp={state_clamp} ({mode_name})")
            mode = run_chain(
                vulkan_binary,
                cpu_binary,
                model_dir,
                fixture_root,
                xs,
                initial_state,
                state_clamp,
                binding,
                args.atol,
                args.rtol,
            )
            mode["mode"] = mode_name
            modes.append(mode)
        print("negative control")
        control = negative_control(
            vulkan_binary, cpu_binary, model_dir, fixture_root, xs, initial_state, args.atol, args.rtol
        )
    except subprocess.CalledProcessError as exc:
        print(exc.stdout or "", file=sys.stdout)
        print(exc.stderr or "", file=sys.stderr)
        raise
    finally:
        if cleanup is not None:
            cleanup.cleanup()

    plan = modes[0]["trace"].get("variant-plan", "")
    dispatch = modes[0]["trace"].get("variant-dispatch", "")
    if not plan:
        raise Mismatch("the Vulkan leg printed no variant-plan trace; the runner lost its plan reporting")
    if dispatch:
        # The packed cell resolves its own fused shaders; a vendor linear
        # dispatch here would mean the graph changed and the tolerance needs a
        # fresh look rather than a silent pass.
        raise Mismatch(f"packed-cell path unexpectedly dispatched vendor modules: {dispatch}")
    print(
        json.dumps(
            {
                "result": "pass",
                "width": WIDTH,
                "head_size": HEAD_SIZE,
                "batch": BATCH,
                "steps": STEPS,
                "state_clamp_modes": modes,
                "variant_plan": plan,
                "vendor_dispatch": dispatch,
                "negative_control": control["result"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
