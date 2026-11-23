#!/usr/bin/env python3
"""Collect the Phase LXXXVII shader / module / harness identities.

Everything recorded here is re-derived on the host, not copied from a previous
phase's notes. The output (`identities.json`) is what the phase README quotes.

Usage: collect_identities.py <out.json>
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VULKAN = HERE.parents[1]
REPO = VULKAN.parent
SHADERS = VULKAN / "shaders"
CANDIDATES = HERE / "candidates"
EVIDENCE_LXXXVI = VULKAN / "validation" / "phase_lxxxvi_wmma_optimization"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fnv1a64(path: Path) -> str:
    value = 0xCBF29CE484222325
    for byte in path.read_bytes():
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"0x{value:016x}"


def run(*command: str) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return (completed.stdout + completed.stderr).strip()


def describe_module(path: Path) -> dict:
    return {
        "path": str(path.relative_to(REPO)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "spirv_fnv1a64": fnv1a64(path),
    }


def main() -> int:
    out = Path(sys.argv[1])

    production = {
        "p3": {
            "source": describe_module(SHADERS / "wmma_gemm_f16_f16acc_twowave_p3.comp"),
            "spirv": describe_module(SHADERS / "wmma_gemm_f16_f16acc_twowave_p3.spv"),
        },
    }
    candidates = {}
    for name in (
        "wmma_gemm_f16_f16acc_twowave_p3_lean",
        "wmma_gemm_f16_f16acc_twowave_p3_batch",
        "wmma_gemm_f16_f16acc_twowave_p3_leanpad",
        "wmma_gemm_f16_f16acc_twowave_p3_pad",
        "wmma_gemm_f16_f16acc_twowave_p3_ablnoepi",
    ):
        candidates[name] = {
            "source": describe_module(CANDIDATES / f"{name}.comp"),
            "spirv": describe_module(CANDIDATES / f"{name}.spv"),
        }

    harnesses = {}
    for label, path in (
        ("bench_lxxxvii", VULKAN / "target-lxxxvii" / "release" / "wmma_bench.exe"),
        ("qualify_lxxxvii", VULKAN / "target-lxxxvii" / "release" / "wmma_qualify.exe"),
        (
            "bench_v025_archived",
            EVIDENCE_LXXXVI / "bin" / "wmma_bench_v025.exe",
        ),
        (
            "qualify_v025_archived",
            EVIDENCE_LXXXVI / "bin" / "wmma_qualify_v025.exe",
        ),
    ):
        harnesses[label] = {"path": str(path.relative_to(REPO)).replace("\\", "/")} | (
            {"bytes": path.stat().st_size, "sha256": sha256(path)}
            if path.is_file()
            else {"missing": True}
        )

    toolchain = {}
    for label, command in (
        ("glslc", ("glslc", "--version")),
        ("rustc", ("rustc", "--version")),
        ("cargo", ("cargo", "--version")),
    ):
        try:
            toolchain[label] = run(*command).splitlines()[0]
        except (IndexError, FileNotFoundError):
            toolchain[label] = "unavailable"

    document = {
        "source_derivation": {
            "p3_source": "hierarchos-vulkan/shaders/wmma_gemm_f16_f16acc_twowave_p3.comp",
            "lean_source_recovered_from": (
                "hierarchos-vulkan/validation/phase_lxxxvi_wmma_optimization/candidates/"
                "wmma_gemm_f16_f16acc_twowave_p3_lean.comp"
            ),
            "generator": (
                "hierarchos-vulkan/validation/phase_lxxxvii_wmma_optimization/"
                "make_candidates.py"
            ),
            "build_recipe": (
                "glslc -fshader-stage=compute --target-env=vulkan1.2 <src> -o <out>"
            ),
        },
        "production": production,
        "candidates": candidates,
        "harnesses": harnesses,
        "toolchain": toolchain,
    }
    out.write_text(json.dumps(document, indent=1), encoding="utf-8")
    print(json.dumps(document, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
