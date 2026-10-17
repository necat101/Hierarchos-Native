#!/usr/bin/env python3
"""Time the fused packed RWKV cell against the vendor-linear composition.

`rwkv_packed_cell_composition_bench` builds both cell graphs on one device and
interleaves their `forward_backward` calls, alternating which graph runs first,
so a steady clock cannot favor either. This driver supplies the same generated
package the parity harness uses and reports the medians plus their ratio.

The result is the evidence for whether the fused packed kernel earns its keep;
it is not a cross-vendor claim. Run it on each adapter that matters.

Usage:
    python hierarchos-vulkan/validation/bench_rwkv_packed_cell_composition.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_rwkv_cpu_bridge as bridge  # noqa: E402
import verify_rwkv_packed_cell_composition as composition  # noqa: E402


def build_release_binary(manifest: Path, name: str) -> Path:
    """Timing must come from optimized host code, not a debug build."""
    result = subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            str(manifest),
            "--bin",
            name,
            "--message-format=json-render-diagnostics",
        ],
        cwd=bridge.REPO,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8, help="rows per cell step (default clears the vendor gates)")
    parser.add_argument("--keep-fixtures", type=Path)
    parser.add_argument("--record", type=Path, help="write the JSON timing receipt here")
    args = parser.parse_args()

    binary = build_release_binary(
        bridge.VULKAN_MANIFEST, "hierarchos-vulkan-rwkv-packed-cell-composition-bench"
    )
    cleanup = None
    if args.keep_fixtures is not None:
        fixture_root = args.keep_fixtures.resolve()
        fixture_root.mkdir(parents=True, exist_ok=True)
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="hierarchos-rwkv-bench-")
        fixture_root = Path(cleanup.name)
    try:
        model_dir = fixture_root / "model"
        composition.write_package(model_dir)
        inputs = composition.make_inputs(batch=args.batch)
        grads = composition.make_gradients(batch=args.batch)
        case = composition.payload(inputs["xs"][0], inputs["initial_state"], grads, batch=args.batch)
        case_path = fixture_root / "bench-case.json"
        case_path.write_text(json.dumps(case), encoding="utf-8")
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
                "--iterations",
                str(args.iterations),
                "--warmup",
                str(args.warmup),
            ],
            cwd=bridge.REPO,
            check=True,
            capture_output=True,
            text=True,
        )
        summary = json.loads(completed.stdout)
    finally:
        if cleanup is not None:
            cleanup.cleanup()

    if summary["max_abs_output"] != 0.0 or summary["max_abs_packed_state"] != 0.0:
        raise SystemExit(
            "the timed graphs disagree numerically, so this timing compares different work: "
            f"output {summary['max_abs_output']:.3e}, state {summary['max_abs_packed_state']:.3e}"
        )
    if args.record is not None:
        args.record.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
