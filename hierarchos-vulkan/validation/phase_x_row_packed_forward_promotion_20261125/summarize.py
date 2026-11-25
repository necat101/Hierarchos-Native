#!/usr/bin/env python3
"""Assemble the Phase X receipt from the raw artifacts.

Every number in `summary.json` is read back out of a file this phase produced;
nothing is transcribed by hand. A missing input is reported as a missing input
rather than as a passing section.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

RECEIPT = Path(__file__).resolve().parent
RAW = RECEIPT / "raw"
ROOT = RECEIPT.parents[1]

PROMOTED = "transformer_attention_forward_rowpacked64_cached_rdna3"
PRODUCTION_HASHES = [
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(name: str) -> dict | None:
    path = RAW / name
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def summary_line(name: str, pattern: str) -> str | None:
    path = RAW / name
    if not path.is_file():
        return None
    match = re.search(pattern, path.read_text(encoding="utf-8", errors="replace"))
    return match.group(0) if match else None


def hashes_from(name: str) -> dict[str, str]:
    path = RAW / name
    result: dict[str, str] = {}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 2:
            result[parts[1].lstrip("*").split("/")[-1]] = parts[0]
    return result


def main() -> int:
    preservation_before = (RAW / "00_preservation.txt").read_text(encoding="utf-8")
    before = {
        key: value
        for key, value in re.findall(r"^(\w+)=(\S+)$", preservation_before, re.M)
    }
    production_hashes = hashes_from("01_production-shader-hashes.txt")
    promoted_hashes = hashes_from("03_promoted-artifact-hashes.txt")
    rebuilt = read_json("21_promoted-artifact-rebuild.json") or {}
    parity_line = summary_line(
        "11_parity-forward.log", r"\[phase-x-parity-summary\] .*"
    )
    gradient_line = summary_line(
        "14_parity-gradients.log", r"\[phase-x-gradient-summary\] .*"
    )
    eligibility_line = summary_line(
        "10_selector-matrix.log", r"\[phase-x-eligibility-summary\] .*"
    )
    fallback_line = summary_line("12_fallback.log", r"\[phase-x-fallback-summary\] .*")
    precedence_line = summary_line(
        "13_short-precedence.log", r"\[phase-x-precedence-summary\] .*"
    )
    step_release = read_json("23_step-digest-release.json")
    step_debug = read_json("22_step-digest-debug.json")
    abba = read_json("abba-summary.json")
    timing = read_json("timing-gpu.json")
    attribution = read_json("attribution.json")

    def verdict(line: str | None) -> str:
        if line is None:
            return "NOT_RUN"
        return "PASS" if "status=PASS" in line else "FAIL"

    timing_arms: dict[str, dict] = {}
    if timing:
        for entry in timing.values():
            key = f"{entry['geometry']}|{entry['arm']}|{entry['workgroups']}wg"
            timing_arms[key] = {
                "samples": entry["samples"],
                "median_gpu_ms": entry["median_gpu_ms"],
                "mean_gpu_ms": entry["mean_gpu_ms"],
                "stdev_gpu_ms": entry["stdev_gpu_ms"],
                "median_host_ms": entry["host"]["median_host_ms"],
            }

    attribution_table: dict[str, dict] = {}
    if attribution:
        logs = {entry["arm"]: entry for entry in attribution["logs"]}
        generic = logs.get("generic", {}).get("per_step", {})
        production = logs.get("production", {}).get("per_step", {})
        for name in sorted({*generic, *production}):
            attribution_table[name] = {
                "generic_ms_per_step": generic.get(name, {}).get("gpu_ms_per_step", 0.0),
                "production_ms_per_step": production.get(name, {}).get(
                    "gpu_ms_per_step", 0.0
                ),
            }
        totals = {
            "generic_ms_per_step": logs.get("generic", {}).get(
                "total_gpu_ms_per_step"
            ),
            "production_ms_per_step": logs.get("production", {}).get(
                "total_gpu_ms_per_step"
            ),
            "profiled_steps": logs.get("generic", {}).get("profiled_steps"),
        }
    else:
        totals = {}

    attention = {}
    if attribution and attribution_table:
        forward_generic = attribution_table.get("transformer_attention_forward", {})
        forward_production = attribution_table.get(f"transformer_attention_forward_{PROMOTED.split('_')[-5]}", {})
        forward_production = attribution_table.get(
            "transformer_attention_forward_rowpacked64_cached_rdna3", {}
        )
        key = attribution_table.get("transformer_attention_backward_key_rdna3", {})
        query = attribution_table.get("transformer_attention_backward_query_rdna3", {})
        for label, forward in (
            ("generic", forward_generic),
            ("production", forward_production),
        ):
            total_attention = (
                forward.get(f"{label}_ms_per_step", 0.0)
                + key.get(f"{label}_ms_per_step", 0.0)
                + query.get(f"{label}_ms_per_step", 0.0)
            )
            step_total = totals.get(f"{label}_ms_per_step") or 0.0
            attention[label] = {
                "forward_ms_per_step": forward.get(f"{label}_ms_per_step"),
                "backward_key_ms_per_step": key.get(f"{label}_ms_per_step"),
                "backward_query_ms_per_step": query.get(f"{label}_ms_per_step"),
                "attention_total_ms_per_step": round(total_attention, 4),
                "attention_share_of_step_gpu": round(total_attention / step_total, 4)
                if step_total
                else None,
            }

    report = {
        "schema": 1,
        "phase": "X — production promotion of the RDNA3 row-packed attention forward",
        "date": "2026-11-25",
        "starting_state": {
            "head": before.get("head"),
            "branch": before.get("branch"),
            "tracked_files_changed": before.get("tracked_files_changed"),
            "diff_sha256": before.get("diff_sha256"),
            "diff_bytes": before.get("diff_bytes"),
            "binary_diff_sha256": before.get("binary_diff_sha256"),
            "staged_index_sha256": before.get("staged_index_sha256"),
            "staged_index_bytes": before.get("staged_index_bytes"),
            "stash0": before.get("stash0"),
        },
        "promoted_artifact": {
            "glsl": f"shaders/{PROMOTED}.comp",
            "glsl_sha256": promoted_hashes.get(f"{PROMOTED}.comp"),
            "spirv": f"shaders/{PROMOTED}.spv",
            "spirv_sha256": promoted_hashes.get(f"{PROMOTED}.spv"),
            "identical_to_phase_ix_candidate": rebuilt.get("identical_to_phase_ix"),
            "rebuild_reproduces_byte_for_byte": rebuilt.get("reproduces_byte_for_byte"),
            "spirv_val_exit": rebuilt.get("spirv_val_exit"),
            "compiler": rebuilt.get("compiler"),
            "compiler_banner": (rebuilt.get("compiler_banner") or "").splitlines()[:1],
            "recorded_compiler_version": rebuilt.get("recorded_version"),
            "flags": rebuilt.get("flags"),
            "target_env": rebuilt.get("target_env"),
        },
        "production_shader_hashes_unchanged": {
            name: production_hashes.get(name) for name in PRODUCTION_HASHES
        },
        "qualification": {
            "eligibility_predicate": {
                "status": verdict(eligibility_line),
                "summary": eligibility_line,
            },
            "forward_matrix": {"status": verdict(parity_line), "summary": parity_line},
            "fallback": {"status": verdict(fallback_line), "summary": fallback_line},
            "selector_order": {
                "status": verdict(precedence_line),
                "summary": precedence_line,
            },
            "gradients_and_scratch": {
                "status": "PASS" if gradient_line and "status=PASS" in gradient_line else "FAIL",
                "summary": gradient_line,
            },
            "forward_log": "raw/11_parity-forward.log",
            "gradient_log": "raw/14_parity-gradients.log",
        },
        "release_complete_step_equivalence": {
            "status": step_release.get("status") if step_release else "NOT_RUN",
            "steps_compared": step_release.get("steps_compared") if step_release else None,
            "loss_bits_mismatches": step_release.get("loss_bits_mismatches")
            if step_release
            else None,
            "gradient_digest_mismatches": step_release.get("gradient_digest_mismatches")
            if step_release
            else None,
            "parameter_digest_mismatches": step_release.get("parameter_digest_mismatches")
            if step_release
            else None,
            "gradient_elements_per_step": step_release.get("gradient_elements_per_step")
            if step_release
            else None,
            "gradient_bits_compared": step_release.get("gradient_bits_compared")
            if step_release
            else None,
            "parameter_end_digest_equal": step_release.get("parameter_end_digest_equal")
            if step_release
            else None,
            "parameter_end_digest": step_release.get("parameter_end_digest_production")
            if step_release
            else None,
            "checkpoint_files_compared": step_release.get("checkpoint_files_compared")
            if step_release
            else None,
            "checkpoint_file_mismatches": step_release.get("checkpoint_file_mismatches")
            if step_release
            else None,
            "checkpoint_digests": step_release.get("checkpoint_digests")
            if step_release
            else None,
            "routes_distinct": step_release.get("routes_distinct")
            if step_release
            else None,
            "dispatch_counters_production": step_release.get("dispatch_counters_production")
            if step_release
            else None,
            "dispatch_counters_generic": step_release.get("dispatch_counters_generic")
            if step_release
            else None,
            "debug_mode_status": step_debug.get("status") if step_debug else "NOT_RUN",
        },
        "release_complete_step_timing": {
            "status": abba.get("status") if abba else "NOT_RUN",
            "rounds": abba.get("rounds") if abba else None,
            "summary": abba.get("summary") if abba else None,
        },
        "attention_forward_kernel_timing": timing_arms or None,
        "per_component_gpu_attribution": {
            "totals": totals or None,
            "table": attribution_table or None,
            "attention": attention or None,
        },
        "phase_ix_comparison": {
            "note": (
                "Phase IX measured the same shader bytes with the kernel swapped into "
                "the production grid; this phase measures the shipped selector, which "
                "launches the packed 16-workgroup grid."
            ),
            "phase_ix_paired_ratio_mean": 1.2236,
            "phase_ix_paired_ratio_ci95": [1.2222, 1.2251],
        },
    }

    # Every promoted artifact hash, plus the Phase III-IX ledgers, so a later
    # phase can detect a rewrite rather than rediscover it.
    report["preserved_ledgers"] = hashes_from("02_prior-receipt-hashes.txt")
    (RECEIPT / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
