#!/usr/bin/env python3
"""Phase LXXXVIII boundary sweep: which staging module does production pick?

Every shape is run through the **v0.2.6 production binary** with no override of
any kind. Two facts are recorded per shape:

* the cooperative shader the GPU profile actually dispatched
  (`..._p3_c1` inside the contract, `..._p3` when the contract cannot be
  proven), read from the timestamp profile stream rather than from a label;
* the cooperative arm's distance from the harness's own FP32 reference
  (`drift_vs_fp32_max_abs`), so a fast shape is never recorded without its
  numerical cost.

A shape is run on its own (`--shapes <one>`), so a run has exactly one
cooperative shader and the profile cannot be mislabelled.

Usage:
  run_boundary_sweep.py --exe EXE --out DIR [--iterations 8] [--skip-reference]
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VALIDATION = HERE.parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "phase_lxxxi_evidence"))

from parse_gpu_profile import parse_stream  # noqa: E402

COOPERATIVE_PREFIX = "wmma_gemm_f16_f16acc_twowave_p3"

# The ten aligned neighbour geometries Phase LXXXVII measured, plus the
# unaligned neighbours the promotion brief names. "expected" is the module the
# contract requires, not a preference.
ALIGNED = (
    "128x4096x768",
    "128x16384x768",
    "256x8192x768",
    "512x1024x768",
    "1024x768x768",
    "512x768x1536",
    "2048x768x1024",
    "480x768x768",
    "512x736x768",
    "512x768x752",
)
UNALIGNED = (
    "511x768x768",
    "255x768x768",
    "512x767x768",
    "512x2047x768",
    "512x768x767",
    "2048x768x2047",
    "16x128x32",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str]) -> dict:
    options = {"iterations": 8, "skip_reference": False}
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--skip-reference":
            options["skip_reference"] = True
            index += 1
        elif arg in ("--exe", "--out", "--iterations"):
            value = argv[index + 1]
            key = arg[2:].replace("-", "_")
            options[key] = int(value) if key == "iterations" else value
            index += 2
        else:
            raise SystemExit(f"unsupported argument {arg!r}")
    for key in ("exe", "out"):
        if key not in options:
            raise SystemExit(f"missing --{key}")
    return options


def dispatched_shader(stream: list[str]) -> tuple[str, int]:
    batches, _ = parse_stream(stream)
    names = {
        batch["shaders"][0][0]
        for batch in batches
        if len(batch["shaders"]) == 1 and batch["shaders"][0][0].startswith(COOPERATIVE_PREFIX)
    }
    if len(names) != 1:
        raise SystemExit(f"expected one cooperative shader, saw {sorted(names)}")
    solo = sum(
        1
        for batch in batches
        if len(batch["shaders"]) == 1 and batch["shaders"][0][0] in names
    )
    return names.pop(), solo


def main() -> int:
    options = parse_args(sys.argv[1:])
    exe = Path(options["exe"]).resolve()
    out = Path(options["out"]).resolve()
    if not exe.is_file():
        raise SystemExit(f"missing executable: {exe}")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for expectation, shapes in (("c1", ALIGNED), ("p3", UNALIGNED)):
        for shape in shapes:
            stem = out / shape
            env = dict(os.environ)
            env["HIERARCHOS_VULKAN_PROFILE_KERNELS"] = "1"
            env["HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE"] = "1"
            env.pop("HIERARCHOS_WMMA_FULL_FP16_VARIANT", None)
            command = [
                str(exe),
                "--wmma-mode",
                "full-fp16",
                "--rounds",
                "1",
                "--iterations",
                str(options["iterations"]),
                "--json",
                "--shapes",
                shape,
            ]
            if options["skip_reference"]:
                command.append("--skip-reference")
            with open(stem.with_suffix(".json"), "wb") as stdout, open(
                out / (shape + "_err.txt"), "wb"
            ) as stderr:
                completed = subprocess.run(command, stdout=stdout, stderr=stderr, env=env)
            if completed.returncode != 0:
                raise SystemExit(f"{shape}: benchmark failed; see {shape}_err.txt")
            report = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
            stream = (out / (shape + "_err.txt")).read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            shader, solo = dispatched_shader(stream)
            arm = next(
                entry
                for entry in report["shapes"][0]["arms"]
                if entry["arm"] == "wmma-full-f16acc"
            )
            selected = "c1" if shader.endswith("_c1") else "p3"
            rows.append(
                {
                    "shape": shape,
                    "expected": expectation,
                    "selected": selected,
                    "shader": shader,
                    "solo_dispatches": solo,
                    "median_ms": arm["median_ms"],
                    "drift_vs_fp32_max_abs": arm["drift_vs_fp32_max_abs"],
                    "staging_counters": report.get("staging_counters"),
                    "verdict": "OK" if selected == expectation else "MISMATCH",
                }
            )
            print(
                f"{shape:<16} expected {expectation:<2} got {selected:<2} "
                f"({shader}) dispatches={solo} drift={arm['drift_vs_fp32_max_abs']:.3e}",
                flush=True,
            )

    mismatches = [row for row in rows if row["verdict"] != "OK"]
    document = {
        "exe": str(exe),
        "exe_sha256": sha256(exe),
        "iterations": options["iterations"],
        "skip_reference": options["skip_reference"],
        "rows": rows,
        "mismatches": len(mismatches),
    }
    (out / "SELECTOR_SWEEP.json").write_text(json.dumps(document, indent=1), encoding="utf-8")
    finite = all(
        row["drift_vs_fp32_max_abs"] == row["drift_vs_fp32_max_abs"] for row in rows
    )
    print()
    print(f"selector mismatches: {len(mismatches)}")
    if not finite:
        print("drift column contains NaN (reference skipped for some shapes)")
    if mismatches:
        for row in mismatches:
            print(f"  {row['shape']}: expected {row['expected']}, selected {row['selected']}")
        return 1
    print("SELECTOR SWEEP: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
