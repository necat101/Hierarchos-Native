#!/usr/bin/env python3
"""Rebuild the promoted attention-forward artifact and prove it is the pinned bytes.

`validation/check_shader_provenance.py` is the repository-wide gate: it rebuilds
every manifest entry with the release that entry records and compares bytes. On
this machine that gate reports `wrong_version` for pre-existing, untouched
artifacts too -- `PATH`'s `glslc` advertises two releases (`shaderc v2023.8
v2025.3`) and the checker reads the second, and the only installed
`glslangValidator` is 15.3.0 while most entries pin 16.5.0/16.6.0 -- so its
verdict says nothing about this phase's artifact.

This script runs the same rebuild for the one artifact this phase promoted,
reports the compiler banner and both releases, and compares:

  * the committed SPIR-V against the recorded Phase IX candidate bytes, and
  * the committed SPIR-V against a fresh rebuild under the manifest recipe.

A version-pin disagreement is reported separately from a byte difference,
because only the second one means the artifact drifted.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "validation" / "training_shader_manifest.json"
ARTIFACT = "transformer_attention_forward_rowpacked64_cached_rdna3.spv"
PHASE_IX = (
    ROOT
    / "validation"
    / "phase_ix_row_packed_forward_20261124"
    / "shaders"
    / ARTIFACT
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entry = next(
        (item for item in document["shaders"] if item["artifact"] == ARTIFACT), None
    )
    if entry is None:
        print(json.dumps({"status": "FAIL", "reason": "not in manifest"}, indent=2))
        return 1

    source = ROOT / "shaders" / entry["source"]
    committed = ROOT / "shaders" / ARTIFACT
    compiler = shutil.which(entry["compiler"])
    report: dict = {
        "artifact": ARTIFACT,
        "source": entry["source"],
        "compiler": entry["compiler"],
        "recorded_version": entry.get("compiler_version"),
        "flags": entry["flags"],
        "target_env": entry.get("target_env"),
        "committed_sha256": sha256(committed),
        "source_sha256": sha256(source),
        "phase_ix_sha256": sha256(PHASE_IX) if PHASE_IX.is_file() else None,
    }
    report["identical_to_phase_ix"] = report["committed_sha256"] == report["phase_ix_sha256"]

    if compiler is None:
        report["status"] = "FAIL"
        report["reason"] = "compiler_missing"
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    banner = subprocess.run(
        [compiler, "--version"], text=True, capture_output=True, check=False
    ).stdout.strip()
    report["compiler_banner"] = banner
    report["compiler_path"] = compiler

    with tempfile.TemporaryDirectory(prefix="phase-x-promoted-") as scratch:
        rebuilt = Path(scratch) / ARTIFACT
        result = subprocess.run(
            [compiler, *entry["flags"], f"shaders/{entry['source']}", "-o", str(rebuilt)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        report["compile_exit"] = result.returncode
        if result.returncode != 0:
            report["status"] = "FAIL"
            report["reason"] = "compile_failed"
            report["error"] = (result.stdout + result.stderr)[-2000:]
            print(json.dumps(report, indent=2, sort_keys=True))
            return 1
        report["rebuilt_sha256"] = sha256(rebuilt)

    validation = subprocess.run(
        ["spirv-val", "--target-env", entry.get("target_env", "vulkan1.2"), str(committed)],
        text=True,
        capture_output=True,
        check=False,
    )
    report["spirv_val_exit"] = validation.returncode
    report["spirv_val_output"] = (validation.stdout + validation.stderr).strip()

    reproduces = report["rebuilt_sha256"] == report["committed_sha256"]
    report["reproduces_byte_for_byte"] = reproduces
    report["status"] = (
        "PASS" if reproduces and report["identical_to_phase_ix"] and validation.returncode == 0
        else "FAIL"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
