#!/usr/bin/env python3
"""Qualify the vendor modules the packed RWKV cell composes.

`RwkvPackedCellOp` builds its channel mix from the vendor
`linear-forward`/`linear-residual-forward` modules and its time-mix projection
from the vendor `linear3-forward` module. Two switches make that composition
observable:

* `HIERARCHOS_VULKAN_DISABLE_TIME_MIX_FORWARD_FUSION` forces the time-mix
  projection through its portable `linear3-forward` composition. Without it the
  fused projection/key/state kernel bypasses the vendor module entirely, so
  `linear3-forward` would have no RWKV coverage.
* `HIERARCHOS_VULKAN_DISABLE_CHANNEL_MIX_FORWARD_FUSION` does the same for the
  channel mix, whose fused forward would otherwise absorb `key_cm`/`value_cm`.
* `HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS` selects the portable modules for
  every family, which is the A/B arm.

This harness runs the packed cell twice over identical package/inputs, once with
the tuned modules and once with the portable ones, each leg carrying its own
packed recurrent state. It compares the cell output, the packed next state and
every returned gradient after each step, asserts that the vendor leg really
dispatched all three linear families (an agreement produced without them would
not close the coverage gap), and runs a perturbed negative control through the
same comparison.

The tuned modules are per-vendor, so qualification names the vendor it intends
to cover: `--expected-vendor amd` requires the resolved plan to carry
`amd-rdna3-*` modules for all three families and must be run on an RDNA3
adapter, while `--expected-vendor intel` requires `intel-gen9-*`. `auto` accepts
whichever vendor the attached adapter resolves, but still refuses a plan that
falls back to the portable modules. A `HIERARCHOS_VULKAN_FORCE_VENDOR` override
is rejected outright: it changes which module the plan names without changing
the hardware that would execute it.

Usage:
    python hierarchos-vulkan/validation/verify_rwkv_packed_cell_composition.py
    python hierarchos-vulkan/validation/verify_rwkv_packed_cell_composition.py --expected-vendor amd --record receipt.json
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

TIME_MIX_FUSION_DISABLE_ENV = "HIERARCHOS_VULKAN_DISABLE_TIME_MIX_FORWARD_FUSION"
CHANNEL_MIX_FUSION_DISABLE_ENV = "HIERARCHOS_VULKAN_DISABLE_CHANNEL_MIX_FORWARD_FUSION"
VENDOR_DISABLE_ENV = "HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS"
VENDOR_FAMILIES = ("linear-forward", "linear-residual-forward", "linear3-forward")
VENDOR_PREFIXES = {"intel": "intel-", "amd": "amd-"}
STEPS = bridge.STEPS
# Eight rows clear both vendor matmul gates: the Gen9 gate needs `output_dim`
# 192 below eight rows (value_cm has 128) and the RDNA3 gate needs eight rows
# outright, so a four-row fixture would not exercise the vendor residual-linear
# arm on either part.
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


def make_inputs(batch: int = BATCH) -> dict:
    generator = torch.Generator().manual_seed(0x48494553)
    state_size = 4 + bridge.HEAD_SIZE
    initial_state = bridge.tensor((batch, bridge.WIDTH, state_size), generator=generator).reshape(-1)
    xs = [bridge.tensor((batch, bridge.WIDTH), generator=generator).reshape(-1).tolist() for _ in range(STEPS)]
    return {"initial_state": initial_state.tolist(), "xs": xs}


def make_gradients(batch: int = BATCH) -> dict[str, list[float]]:
    generator = torch.Generator().manual_seed(0x48494555)
    return {
        "grad_output": bridge.tensor((batch, bridge.WIDTH), generator=generator).reshape(-1).tolist(),
        "grad_packed_new_state": bridge.tensor(
            (batch, bridge.WIDTH, 4 + bridge.HEAD_SIZE), generator=generator
        )
        .reshape(-1)
        .tolist(),
    }


def payload(x: list[float], packed_state: list[float], grads: dict[str, list[float]], batch: int = BATCH) -> dict:
    return {
        "batch": batch,
        "width": bridge.WIDTH,
        "head_size": bridge.HEAD_SIZE,
        "input_dim": bridge.INPUT_DIM,
        "state_mode": "explicit-output",
        "state_clamp": 0.0,
        "x": x,
        "packed_state": packed_state,
        "token_features": [0.0] * (batch * bridge.INPUT_DIM),
        **grads,
    }


def run(binary: Path, model_dir: Path, case: dict, case_path: Path, *, vendor: bool) -> dict:
    case_path.write_text(json.dumps(case), encoding="utf-8")
    env = dict(os.environ, HIERARCHOS_VULKAN_VARIANT_TRACE="1")
    env[TIME_MIX_FUSION_DISABLE_ENV] = "1"
    env[CHANNEL_MIX_FUSION_DISABLE_ENV] = "1"
    env.pop(VENDOR_DISABLE_ENV, None)
    if not vendor:
        env[VENDOR_DISABLE_ENV] = "1"
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
        for key in ("variant-plan", "variant-dispatch"):
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


def plan_variants(trace: dict) -> dict[str, str]:
    variants = {}
    for entry in trace.get("variant-plan", "").split():
        family, _, variant = entry.partition("=")
        variants[family] = variant
    return variants


def require_vendor_plan(trace: dict, expected_vendor: str) -> dict[str, str]:
    """Fail unless all three composed projections resolved a tuned module.

    Returns the family/variant pairs so the receipt names what was actually
    qualified. A portable resolution is a coverage failure here, not a pass with
    a footnote: it means the vendor leg ran the same arithmetic as the portable
    leg through a different scheduler, which is not the comparison this harness
    exists to make.
    """
    variants = plan_variants(trace)
    for family in VENDOR_FAMILIES:
        if family not in variants:
            raise bridge.Mismatch(
                f"variant plan has no entry for {family}; the runner lost its plan trace"
            )
        if variants[family] == "portable":
            raise bridge.Mismatch(
                f"{family} resolved the portable module on this adapter; {expected_vendor} "
                "qualification needs the tuned module, so run this on an adapter that has one"
            )
    if expected_vendor != "auto":
        prefix = VENDOR_PREFIXES[expected_vendor]
        for family in VENDOR_FAMILIES:
            if not variants[family].startswith(prefix):
                raise bridge.Mismatch(
                    f"{family} resolved {variants[family]!r}, not an {expected_vendor} module; "
                    f"this host cannot qualify the {expected_vendor} path"
                )
    return {family: variants[family] for family in VENDOR_FAMILIES}


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
    """A perturbed vendor leg must fail the same comparison.

    The perturbation is far above fp32 reassociation noise but far below a
    structural bug, so this pins the tolerance's discriminating power instead of
    letting a comparison that never rejects anything report agreement.
    """
    baseline = run(binary, model_dir, payload(x, initial_state, grads), work / "vendor-case.json", vendor=True)
    perturbed = [value * 1.01 for value in x]
    control = run(
        binary,
        model_dir,
        payload(perturbed, initial_state, grads),
        work / "perturbed-case.json",
        vendor=True,
    )
    try:
        bridge.compare("negative control cell output", baseline["output"], control["output"], atol, rtol)
    except bridge.Mismatch as exc:
        print(f"  negative control rejected the perturbation as expected: {exc}")
        return {"result": "rejected", "detail": str(exc)}
    raise bridge.Mismatch("negative control passed: the comparison does not discriminate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--atol", type=float, default=1.0e-4, help="absolute part of the agreement bound")
    parser.add_argument("--rtol", type=float, default=1.0e-5, help="relative part of the agreement bound")
    parser.add_argument(
        "--expected-vendor",
        choices=("auto", "intel", "amd"),
        default="auto",
        help="require the resolved modules to be this vendor's tuned variants",
    )
    parser.add_argument("--keep-fixtures", type=Path)
    parser.add_argument("--record", type=Path, help="write the JSON receipt here")
    args = parser.parse_args()

    if "HIERARCHOS_VULKAN_FORCE_VENDOR" in os.environ:
        raise bridge.Mismatch(
            "HIERARCHOS_VULKAN_FORCE_VENDOR is set; qualification must run on the adapter it "
            "names, because a forced plan dispatches modules the hardware was not classified for"
        )

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
        vendor_state = list(inputs["initial_state"])
        portable_state = list(inputs["initial_state"])
        comparisons = []
        traces = {}
        for step, step_input in enumerate(inputs["xs"]):
            vendor = run(
                binary,
                model_dir,
                payload(step_input, vendor_state, grads),
                fixture_root / "vendor-case.json",
                vendor=True,
            )
            portable = run(
                binary,
                model_dir,
                payload(step_input, portable_state, grads),
                fixture_root / "portable-case.json",
                vendor=False,
            )
            if step == 0:
                traces = {"vendor": vendor["_trace"], "portable": portable["_trace"]}
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
                        vendor[field],
                        portable[field],
                        args.atol,
                        args.rtol,
                    )
                )
            vendor_state = vendor["packed_new_state"]
            portable_state = portable["packed_new_state"]

        portable_counts = dispatch_counts(traces["portable"])
        if portable_counts:
            raise bridge.Mismatch(
                f"portable leg dispatched vendor modules {portable_counts}; it must run the "
                "portable modules only, so this run cannot be read as a vendor A/B"
            )
        vendor_counts = dispatch_counts(traces["vendor"])
        missing = [family for family in VENDOR_FAMILIES if vendor_counts.get(family, 0) == 0]
        if missing:
            raise bridge.Mismatch(
                f"vendor leg did not dispatch {missing} (counts {vendor_counts}); the packed cell "
                "did not exercise the vendor-linear graph this check exists to cover"
            )
        vendor_variants = require_vendor_plan(traces["vendor"], args.expected_vendor)
        qualified_vendor = "amd" if any(
            variant.startswith("amd-") for variant in vendor_variants.values()
        ) else "intel"
        if args.expected_vendor != "auto" and qualified_vendor != args.expected_vendor:
            raise bridge.Mismatch(
                f"qualified {qualified_vendor} modules while asking for {args.expected_vendor}"
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
        "time_mix_forward_fusion": "disabled",
        "channel_mix_forward_fusion": "disabled",
        "expected_vendor": args.expected_vendor,
        "qualified_vendor": qualified_vendor,
        "vendor_variants": vendor_variants,
        "vendor_plan": traces["vendor"].get("variant-plan", ""),
        "vendor_dispatch": traces["vendor"].get("variant-dispatch", ""),
        "portable_dispatch": traces["portable"].get("variant-dispatch", ""),
        "negative_control": control["result"],
        "negative_control_detail": control.get("detail", ""),
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
