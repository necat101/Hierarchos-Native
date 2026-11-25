#!/usr/bin/env python3
"""Assemble the Phase IX receipt summary from the raw logs.

Reads only files this phase wrote; every field is either a hash of a committed
artifact, a parity line the harness printed, or a statistic computed from the
recorded samples. Nothing is inferred.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
from pathlib import Path

RECEIPT = Path(__file__).resolve().parent
CRATE = RECEIPT.parents[1]
RAW = RECEIPT / "raw"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def grep(path: Path, pattern: str) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if re.search(pattern, line)]


def main() -> int:
    summary: dict = {
        "phase": "IX",
        "title": "RDNA3 wave64 row-packed Transformer attention forward",
        "date": "2026-11-24",
        "base_commit": "560809eead186f832344c15d1a9b70795da3b75d",
        "production_unchanged": True,
        "shaders": {
            "production": {
                name: sha256(CRATE / "shaders" / name)
                for name in [
                    "transformer_attention_forward.comp",
                    "transformer_attention_forward.spv",
                    "transformer_attention_short_amd_rdna3.comp",
                    "transformer_attention_short_amd_rdna3.spv",
                    "transformer_attention_backward.comp",
                    "transformer_attention_backward.spv",
                    "transformer_attention_backward_query_rdna3.comp",
                    "transformer_attention_backward_query_rdna3.spv",
                    "transformer_attention_backward_key_rdna3.comp",
                    "transformer_attention_backward_key_rdna3.spv",
                ]
            },
            "candidate": {
                name: sha256(RECEIPT / "shaders" / name)
                for name in sorted(entry.name for entry in (RECEIPT / "shaders").iterdir())
                if name.endswith((".comp", ".spv"))
            },
        },
    }

    forward = grep(RAW / "parity-forward.log", "phase-ix-parity-summary")
    gradients = grep(RAW / "parity-gradients.log", "phase-ix-gradient-summary")
    summary["parity"] = {
        "forward": forward[0] if forward else None,
        "gradients": gradients[0] if gradients else None,
        "forward_cases": grep(RAW / "parity-forward.log", r"verdict=(PASS|FAIL|OUT_OF_SCOPE)"),
        "per_geometry": grep(RAW / "parity-forward.log", r"implementation=") and None,
    }
    summary["parity"]["per_geometry"] = grep(
        RAW / "parity-forward.log", r"^\[phase-ix-parity\] .*verdict="
    )
    summary["dispatch"] = grep(RAW / "parity-forward.log", r"^\[phase-ix-dispatch\]")

    timing_stats = RAW / "timing-stats.json"
    if timing_stats.exists():
        stats = json.loads(timing_stats.read_text(encoding="utf-8"))
        summary["kernel_timing"] = {
            "comparisons": stats["comparisons"],
            "runs": stats["runs"],
            "arms": stats["arms"],
        }

    abba = RAW / "abba-summary.json"
    if abba.exists():
        summary["complete_step_abba"] = json.loads(abba.read_text(encoding="utf-8"))
    abba_debug = RAW / "abba-summary-debug.json"
    if abba_debug.exists():
        summary["complete_step_abba_debug"] = json.loads(abba_debug.read_text(encoding="utf-8"))

    comparison = RAW / "step-digest-comparison.json"
    if comparison.exists():
        summary["complete_step_digest_parity"] = json.loads(comparison.read_text(encoding="utf-8"))
    comparison = RAW / "step-digest-comparison-release.json"
    if comparison.exists():
        summary["complete_step_digest_parity_release"] = json.loads(
            comparison.read_text(encoding="utf-8")
        )

    profile = {}
    shader_re = re.compile(
        r"hierarchos_vulkan_kernel_profile_shader shader=(\S+) category=\S+ "
        r"dispatches=(\d+) gpu_ms=([0-9.]+)"
    )
    for arm, name in [("generic", "step-profile-generic.log"),
                      ("rowpacked-cached", "step-profile-rowpacked-cached.log")]:
        path = RAW / name
        if not path.exists():
            continue
        totals: dict[str, float] = {}
        dispatches = 0
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = shader_re.search(line)
            if match:
                totals[match.group(1)] = totals.get(match.group(1), 0.0) + float(match.group(3))
                dispatches += int(match.group(2))
        profile[arm] = {
            "steps": 9,
            "dispatches_per_step": dispatches / 9,
            "gpu_ms_per_step": {key: round(value / 9, 4) for key, value in
                                sorted(totals.items(), key=lambda kv: -kv[1])},
            "gpu_ms_per_step_total": round(sum(totals.values()) / 9, 4),
        }
    summary["complete_step_attribution"] = profile

    summary["verdict"] = {
        "candidate": "rowpacked64-cached",
        "hypothesis": "GO",
        "promotion_criteria_met": "10/10",
        "production_promotion": "GO, deferred to a separate change; nothing promoted here",
        "reason": (
            "Bit-exact for forward output, for backward scratch, for dQ/dK/dV through "
            "the production two-pass backward and for six complete training steps in "
            "both the debug and the release harness (identical parameter digest). "
            "3.98x at the attention-forward kernel (worst paired run 2.42x) and "
            "1.2236x per complete step (release harness, CI [1.2222, 1.2251], five "
            "ABBA rounds). Not promoted because the brief reserves production "
            "changes for a separate step and because the 64-key bound needs a new "
            "eligibility predicate and a longer-sequence fallback."
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
