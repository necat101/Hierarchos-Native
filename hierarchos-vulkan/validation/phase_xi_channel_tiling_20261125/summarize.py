#!/usr/bin/env python3
"""Assemble the Phase XI receipt summary from the raw artifacts.

Usage: summarize.py RAW_DIR > summary.json

Nothing here re-derives a number: every field is read from a file the pipeline
wrote, and a missing file makes the field absent rather than zero. The gates are
reported as the pipeline recorded them (`verdict=` lines and the parity summary
line), so a summary cannot say PASS where the gate said otherwise.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def read_json(path: Path):
    text = read(path)
    return json.loads(text) if text.strip() else None


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: summarize.py RAW_DIR", file=sys.stderr)
        return 2
    raw = Path(argv[1])

    identity = {}
    for line in read(raw / "00_identity.txt").splitlines():
        if "=" in line and not line.startswith(" "):
            key, value = line.split("=", 1)
            identity[key] = value

    shader_hashes = {}
    for line in read(raw / "01_shader-hashes.txt").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = dict(
            token.split("=", 1) for token in line.split() if "=" in token and not token.startswith("*")
        )
        name = line.split()[0].lstrip("*")
        if "chunk" in fields:
            shader_hashes[name] = {
                "chunk": int(fields["chunk"]),
                "blocks": int(fields["blocks"]),
                "unroll": int(fields["unroll"]),
                "spirv_sha256": fields["spv"],
                "preprocessed_glsl_sha256": fields["preprocessed_glsl"],
            }
        else:
            shader_hashes[name] = {"sha256": fields.get("sha256") or line.split()[0]}

    production_hashes = {}
    for line in read(raw / "03_production-hashes.txt").splitlines():
        parts = line.split()
        if len(parts) == 2:
            production_hashes[Path(parts[1]).name] = parts[0]

    parity_text = read(raw / "11_parity.log")
    parity = {
        "pass_lines": parity_text.count("verdict=PASS"),
        "fail_lines": parity_text.count("verdict=FAIL"),
    }
    summary_line = re.search(r"^\[phase-xi-parity-summary\].*$", parity_text, re.MULTILINE)
    if summary_line:
        parity["summary"] = dict(
            token.split("=", 1) for token in summary_line.group(0).split() if "=" in token
        )

    ownership_text = read(raw / "10_ownership.log")
    ownership = {
        "pass_lines": ownership_text.count("verdict=PASS"),
        "fail_lines": ownership_text.count("verdict=FAIL"),
    }
    ownership_first = re.search(r"^\[phase-xi-ownership\].*$", ownership_text, re.MULTILINE)
    if ownership_first:
        ownership["example"] = dict(
            token.split("=", 1) for token in ownership_first.group(0).split() if "=" in token
        )

    equivalence = {
        "diff_bytes": (raw / "32_equivalence.diff").stat().st_size
        if (raw / "32_equivalence.diff").exists()
        else None,
        "production_run_end": None,
        "candidate_run_end": None,
    }
    for key, name in (
        ("production_run_end", "30_equivalence-production.log"),
        ("candidate_run_end", "31_equivalence-candidate.log"),
    ):
        match = re.search(r"^\[phase-xi-step-run-end\].*$", read(raw / name), re.MULTILINE)
        if match:
            equivalence[key] = match.group(0)
    candidate_logs = sorted(raw.glob("31_equivalence-*.log"))
    if candidate_logs:
        match = re.search(
            r"^\[phase-xi-step-run-end\].*$", read(candidate_logs[0]), re.MULTILINE
        )
        if match:
            equivalence["candidate_run_end"] = match.group(0)
        checkpoints = [
            line
            for line in read(candidate_logs[0]).splitlines()
            if line.startswith("[phase-xi-checkpoint]")
        ]
        equivalence["candidate_checkpoints"] = checkpoints
    production_checkpoints = [
        line
        for line in read(raw / "30_equivalence-production.log").splitlines()
        if line.startswith("[phase-xi-checkpoint]")
    ]
    equivalence["production_checkpoints"] = production_checkpoints

    print(
        json.dumps(
            {
                "phase": "XI",
                "objective": "RDNA3 GQA key-pass channel tiling",
                "identity": identity,
                "experimental_shader_hashes": shader_hashes,
                "production_shader_hashes": production_hashes,
                "state_inventory": read_json(raw / "04_state-inventory.json"),
                "ownership_gate": ownership,
                "parity_gate": parity,
                "kernel_timing": read_json(raw / "22_timing.json"),
                "kernel_timing_all_variants": read_json(
                    raw / "25_timing-nine-variant.json"
                ),
                "training_equivalence": equivalence,
                "abba": read_json(raw / "33_abba.json"),
                "attribution": read_json(raw / "42_attribution.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
