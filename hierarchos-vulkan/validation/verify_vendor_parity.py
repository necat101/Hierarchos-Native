#!/usr/bin/env python3
"""Prove vendor-aware kernel selection cannot change numerical results.

`src/vendor.rs` swaps in vendor-specific SPIR-V modules for measured kernel
families (currently Intel Gen9 tiled matmuls) while every other vendor keeps the
portable modules the backend shipped before tuning. That is only safe if the two
paths are numerically identical, and the claim has to be checkable on whatever
adapter is installed - including a machine that is *not* the vendor the variant
was tuned for.

This harness runs the headline forward-logit qualification twice on the
executing adapter:

1. with the vendor plan resolved normally for the physical device, and
2. with `HIERARCHOS_VULKAN_FORCE_VENDOR=amd`, which pins kernel selection to the
   portable modules (the AMD target's modules) regardless of the hardware.

Both runs must pass at the unchanged absolute gate, and every comparison must
report the same `max_abs` and `failing_values`. On an Intel adapter that is the
cross-vendor statement: the AMD path's portable modules reproduce exactly what
the tuned Intel modules compute. On an AMD or NVIDIA adapter the forced run is
the real run, so the check degrades into the ordinary qualification plus a
portable-path regression guard.

Exit status is 0 only when both runs pass and agree.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
HEADLINE = ROOT / "validation" / "verify_hf_logits.py"
FORCED_VENDOR = "amd"


def _json_report(stdout: str) -> dict:
    """Extract the harness's trailing JSON report from its noisy stdout."""
    normalized = stdout.replace("\r\n", "\n")
    start = normalized.rindex('{\n  "result"')
    end = normalized.index("\n}\n", start) + 2
    return json.loads(normalized[start:end])


def _run_headline(timeout: int, extra_env: dict[str, str]) -> tuple[int, str]:
    env = dict(os.environ)
    env.update(extra_env)
    completed = subprocess.run(
        [sys.executable, str(HEADLINE), "--headline-strict"],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
        timeout=timeout,
    )
    return completed.returncode, completed.stdout


def _comparisons(report: dict) -> dict[tuple[str, str], tuple[float, int]]:
    """One `(max_abs, failing_values)` row per `(family, case)` comparison."""
    rows: dict[tuple[str, str], tuple[float, int]] = {}
    for entry in report.get("families", []):
        rows[(entry["family"], entry["case"])] = (
            entry["max_abs"],
            entry["failing_values"],
        )
    return rows


def _device_vendor_plan() -> None:
    """Print the resolved kernel plan so the log shows what the default run used."""
    for profile in ("debug", "release"):
        for name in ("hierarchos-vulkan-devices", "hierarchos-vulkan-devices.exe"):
            binary = ROOT / "target" / profile / name
            if binary.is_file():
                completed = subprocess.run(
                    [str(binary), "--kernel-plan"],
                    capture_output=True,
                    text=True,
                    cwd=REPO,
                )
                if completed.returncode != 0:
                    print("warning: kernel-plan probe failed; continuing", flush=True)
                    return
                try:
                    devices = json.loads(completed.stdout)
                except json.JSONDecodeError:
                    print("warning: kernel-plan probe was not JSON; continuing", flush=True)
                    return
                for device in devices:
                    plan = device.get("vendor_kernel_plan")
                    if not plan:
                        continue
                    print(
                        f"adapter {device.get('name')!r} vendor={plan['vendor_id']} "
                        f"device={plan['device_id']}",
                        flush=True,
                    )
                    for kernel in plan["kernels"]:
                        print(
                            f"  {kernel['family']:<26} {kernel['variant']}"
                            f"{' (geometry gated)' if kernel['geometry_gated'] else ''}",
                            flush=True,
                        )
                return
    print("note: no --kernel-plan binary found; run the qualification tests instead", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=1800, help="per-run timeout in seconds")
    args = parser.parse_args()

    print(f"python: {sys.executable}")
    _device_vendor_plan()

    print(
        f"\n[1/2] headline logits with the resolved plan (FORCE_VENDOR unset)",
        flush=True,
    )
    default_status, default_stdout = _run_headline(args.timeout, {})
    print(default_stdout, flush=True)
    if default_status != 0:
        print(f"FAIL: resolved-plan run exited {default_status}")
        return 1

    print(
        f"\n[2/2] headline logits with portable modules "
        f"(HIERARCHOS_VULKAN_FORCE_VENDOR={FORCED_VENDOR})",
        flush=True,
    )
    forced_status, forced_stdout = _run_headline(
        args.timeout, {"HIERARCHOS_VULKAN_FORCE_VENDOR": FORCED_VENDOR}
    )
    print(forced_stdout, flush=True)
    if forced_status != 0:
        print(f"FAIL: forced-portable run exited {forced_status}")
        return 1

    default_rows = _comparisons(_json_report(default_stdout))
    forced_rows = _comparisons(_json_report(forced_stdout))
    if not default_rows:
        print("FAIL: the resolved-plan run reported no comparisons")
        return 1

    mismatches = [
        (key, default_rows.get(key), forced_rows.get(key))
        for key in sorted(set(default_rows) | set(forced_rows))
        if default_rows.get(key) != forced_rows.get(key)
    ]
    worst = max(value[0] for value in default_rows.values())
    print("\n=== vendor parity ===")
    print(f"comparisons: {len(default_rows)} resolved, {len(forced_rows)} forced-portable")
    print(f"worst max_abs on both paths: {worst:.3e}")
    for key, left, right in mismatches:
        print(f"  MISMATCH {key}: resolved={left} forced={right}")
    if mismatches:
        print(f"FAIL: {len(mismatches)} comparison(s) differ between the kernel paths")
        return 1

    print(
        "PASS: every headline comparison is identical with the tuned modules and "
        "with the portable (AMD-target) modules"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
