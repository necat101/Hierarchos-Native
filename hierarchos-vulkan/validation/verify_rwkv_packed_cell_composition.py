#!/usr/bin/env python3
"""Cross-check the fused packed RWKV cell against the vendor-linear composition.

`RwkvPackedCellOp` normally runs the recurrent cell with a single fused portable
shader (`packed_cell_channel_mix_state_forward_fused.spv`). Nothing else
dispatches on that path, so `verify_rwkv_cpu_bridge.py` cannot say anything about
the tuned vendor matmul modules.

Setting `HIERARCHOS_VULKAN_DISABLE_CHANNEL_MIX_FORWARD_FUSION` forces the
multi-kernel composition instead: the channel-mix producer stays fused at
layer-norm/mix granularity, while `key_cm` and `value_cm` run through the
vendor `linear-forward` and `linear-residual-forward` modules that the rest of
the training graph uses. The packed-cell op follows the same switch (it gates
its fused shader on the channel mix's fused capability), so one environment
variable selects a complete alternative cell graph.

This harness runs both graphs over identical package/inputs and compares the
cell output, the packed next state and every returned gradient after each step,
each leg consuming its own recurrent state. It asserts that the composed leg
really dispatched both vendor linear families, because an agreement produced by
two portable paths would not close the coverage gap, and it runs a perturbed
negative control through the same comparison.

Usage:
    python hierarchos-vulkan/validation/verify_rwkv_packed_cell_composition.py
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
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_rwkv_cpu_bridge as bridge  # noqa: E402  (shared fixture builder)

FUSION_DISABLE_ENV = "HIERARCHOS_VULKAN_DISABLE_CHANNEL_MIX_FORWARD_FUSION"
VENDOR_FAMILIES = ("linear-forward", "linear-residual-forward")
STEPS = bridge.STEPS
# Eight rows clear both Gen9 matmul gates: at four to seven rows the value_cm
# projection (`output_dim` 128) stays on the portable module, so a four-row
# fixture would not exercise the vendor residual-linear arm at all.
BATCH = 8


def write_package(directory: Path) -> None:
    """Bridge fixture, with a non-trivial adapter so its gradient is compared.

    `verify_rwkv_cpu_bridge.py` zeroes `up.weight` so the CPU side can be handed
    the adapter output directly. Both legs here are Vulkan, so the adapter can be
    live: without that, `token_feature_grad` would be structurally zero and the
    comparison would silently skip a real gradient.
    """
    bridge.write_package(directory)
    generator = torch.Generator().manual_seed(0x48494554)
    tensors = load_file(directory / "model.safetensors")
    width = bridge.WIDTH
    tensors[f"{bridge.ADAPTER_PREFIX}.up.weight"] = bridge.tensor(
        (4 * width, bridge.ADAPTER_RANK), generator=generator
    )
    tensors[f"{bridge.ADAPTER_PREFIX}.bias"] = bridge.tensor((4 * width,), generator=generator)
    save_file({name: value.contiguous() for name, value in tensors.items()}, directory / "model.safetensors")


def make_inputs() -> dict:
    generator = torch.Generator().manual_seed(0x48494553)
    state_size = 4 + bridge.HEAD_SIZE
    initial_state = bridge.tensor((BATCH, bridge.WIDTH, state_size), generator=generator).reshape(-1)
    xs = [bridge.tensor((BATCH, bridge.WIDTH), generator=generator).reshape(-1).tolist() for _ in range(STEPS)]
    return {"initial_state": initial_state.tolist(), "xs": xs}


def make_gradients() -> dict[str, list[float]]:
    generator = torch.Generator().manual_seed(0x48494555)
    return {
        "grad_output": bridge.tensor((BATCH, bridge.WIDTH), generator=generator).reshape(-1).tolist(),
        "grad_packed_new_state": bridge.tensor(
            (BATCH, bridge.WIDTH, 4 + bridge.HEAD_SIZE), generator=generator
        )
        .reshape(-1)
        .tolist(),
    }


def payload(x: list[float], packed_state: list[float], grads: dict[str, list[float]]) -> dict:
    return {
        "batch": BATCH,
        "width": bridge.WIDTH,
        "head_size": bridge.HEAD_SIZE,
        "input_dim": bridge.INPUT_DIM,
        "state_mode": "explicit-output",
        "state_clamp": 0.0,
        "x": x,
        "packed_state": packed_state,
        "token_features": [0.0] * (BATCH * bridge.INPUT_DIM),
        **grads,
    }


def run(binary: Path, model_dir: Path, case: dict, case_path: Path, *, composed: bool) -> dict:
    case_path.write_text(json.dumps(case), encoding="utf-8")
    env = dict(os.environ, HIERARCHOS_VULKAN_VARIANT_TRACE="1")
    env.pop(FUSION_DISABLE_ENV, None)
    if composed:
        env[FUSION_DISABLE_ENV] = "1"
    completed = subprocess.run(
        [
            str(binary),
            "--case",
            str(case_path),
            "--model-dir",
            str(model_dir),
            "--cell-prefix",
            bridge.CELL_PREFIX,
            "--adapter-prefix",
            bridge.ADAPTER_PREFIX,
        ],
        cwd=bridge.REPO,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    result = json.loads(completed.stdout)
    trace = {}
    for line in completed.stderr.splitlines():
        for key in ("variant-plan", "variant-dispatch", "packed-cell-forward-kernel"):
            if line.startswith(f"{key}:"):
                trace[key] = line.split(":", 1)[1].strip()
    result["_trace"] = trace
    return result


def dispatch_counts(trace: dict) -> dict[str, int]:
    counts = {}
    for entry in trace.get("variant-dispatch", "").split():
        family, _, count = entry.partition("=")
        counts[family] = int(count)
    return counts


def negative_control(
    binary: Path,
    model_dir: Path,
    work: Path,
    x: list[float],
    initial_state: list[float],
    grads: dict[str, list[float]],
    atol: float,
    rtol: float,
) -> dict:
    """A perturbed composed leg must fail the same comparison.

    The perturbation is far above fp32 reassociation noise but far below a
    structural bug, so this pins the tolerance's discriminating power instead of
    letting a comparison that never rejects anything report agreement.
    """
    fused = run(binary, model_dir, payload(x, initial_state, grads), work / "fused-case.json", composed=False)
    perturbed = [value * 1.01 for value in x]
    composed = run(
        binary,
        model_dir,
        payload(perturbed, initial_state, grads),
        work / "composed-case.json",
        composed=True,
    )
    try:
        bridge.compare("negative control cell output", fused["output"], composed["output"], atol, rtol)
    except bridge.Mismatch as exc:
        print(f"  negative control rejected the perturbation as expected: {exc}")
        return {"result": "rejected", "detail": str(exc)}
    raise bridge.Mismatch("negative control passed: the comparison does not discriminate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--atol", type=float, default=1.0e-4, help="absolute part of the agreement bound")
    parser.add_argument("--rtol", type=float, default=1.0e-5, help="relative part of the agreement bound")
    parser.add_argument("--keep-fixtures", type=Path)
    parser.add_argument("--record", type=Path, help="write the JSON receipt here")
    args = parser.parse_args()

    binary = bridge.build_binary(bridge.VULKAN_MANIFEST, "hierarchos-vulkan-rwkv-packed-cell-step")
    cleanup = None
    if args.keep_fixtures is not None:
        fixture_root = args.keep_fixtures.resolve()
        fixture_root.mkdir(parents=True, exist_ok=True)
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="hierarchos-rwkv-composition-")
        fixture_root = Path(cleanup.name)
    try:
        model_dir = fixture_root / "model"
        write_package(model_dir)
        inputs = make_inputs()
        grads = make_gradients()
        fused_state = list(inputs["initial_state"])
        composed_state = list(inputs["initial_state"])
        comparisons = []
        traces = {}
        for step, step_input in enumerate(inputs["xs"]):
            fused = run(
                binary,
                model_dir,
                payload(step_input, fused_state, grads),
                fixture_root / "fused-case.json",
                composed=False,
            )
            composed = run(
                binary,
                model_dir,
                payload(step_input, composed_state, grads),
                fixture_root / "composed-case.json",
                composed=True,
            )
            if step == 0:
                traces = {"fused": fused["_trace"], "composed": composed["_trace"]}
            for field in (
                "output",
                "packed_new_state",
                "grad_x",
                "grad_packed_state",
                "token_feature_grad",
            ):
                comparisons.append(
                    bridge.compare(
                        f"step {step} {field}",
                        fused[field],
                        composed[field],
                        args.atol,
                        args.rtol,
                    )
                )
            fused_state = fused["packed_new_state"]
            composed_state = composed["packed_new_state"]

        for leg, expected in (("fused", "fused"), ("composed", "composed")):
            observed = traces[leg].get("packed-cell-forward-kernel")
            if observed != expected:
                raise bridge.Mismatch(
                    f"{leg} leg reported packed-cell forward kernel {observed!r}, expected {expected!r}; "
                    f"check that {FUSION_DISABLE_ENV} selects the composed graph"
                )
        fused_counts = dispatch_counts(traces["fused"])
        composed_counts = dispatch_counts(traces["composed"])
        if fused_counts:
            raise bridge.Mismatch(
                f"fused leg dispatched vendor modules {fused_counts}; it is supposed to run the fused "
                "shader only, so this run cannot be read as fused-vs-composed"
            )
        missing = [family for family in VENDOR_FAMILIES if composed_counts.get(family, 0) == 0]
        if missing:
            raise bridge.Mismatch(
                f"composed leg did not dispatch {missing} (counts {composed_counts}); the composition "
                "ran portable modules, so it does not exercise the vendor-linear graph this check exists "
                "to cover"
            )

        print("negative control")
        control = negative_control(
            binary,
            model_dir,
            fixture_root,
            inputs["xs"][0],
            inputs["initial_state"],
            grads,
            args.atol,
            args.rtol,
        )
    finally:
        if cleanup is not None:
            cleanup.cleanup()

    worst = max(row["max_abs"] for row in comparisons)
    summary = {
        "result": "pass",
        "width": bridge.WIDTH,
        "head_size": bridge.HEAD_SIZE,
        "batch": BATCH,
        "steps": STEPS,
        "comparisons": len(comparisons),
        "worst_max_abs": worst,
        "atol": args.atol,
        "rtol": args.rtol,
        "forward_kernels": {
            "fused": traces["fused"].get("packed-cell-forward-kernel", ""),
            "composed": traces["composed"].get("packed-cell-forward-kernel", ""),
        },
        "vendor_plan": traces["composed"].get("variant-plan", ""),
        "fused_dispatch": traces["fused"].get("variant-dispatch", ""),
        "composed_dispatch": traces["composed"].get("variant-dispatch", ""),
        "negative_control": control["result"],
        "rows": comparisons,
    }
    if args.record is not None:
        args.record.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except bridge.Mismatch as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
