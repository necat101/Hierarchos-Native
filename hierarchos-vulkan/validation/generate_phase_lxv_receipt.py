#!/usr/bin/env python3
"""Build `phase_lxv_receipt.json` from the committed Phase LXV evidence.

Every field is derived, not typed: the margin block comes from the per-seed
accuracy receipts, the performance block from the raw ABBA samples, the dispatch
block from the receipts' own dispatch accounting, and the hardware summary from
the RGP evidence manifest.

Hosted CI never runs this. It encodes measurements taken on this RDNA3 host; the
gate recomputes the same numbers from the same evidence so a hand-edited receipt
cannot survive.

Usage: python generate_phase_lxv_receipt.py [--performance-only]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import check_phase_lxv as c
import phase_lxv_stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# Which ABBA sweep is which. `arm` is the benchmark arm whose ratio is reported;
# production sweeps carry the non-inferiority verdict.
SWEEPS = {
    "abba_a_vs_final": {"arm": "wmma-mixed-f32acc", "production": True,
                        "reference": "candidate-a", "candidate": "final"},
    "abba_a_vs_a_null": {"arm": "wmma-mixed-f32acc", "production": False,
                         "reference": "candidate-a", "candidate": "candidate-a"},
    "abba_full_main_vs_final": {"arm": "wmma-full-f16acc", "production": False,
                                "reference": "main", "candidate": "final"},
    "abba_main_vs_final": {"arm": "wmma-mixed-f32acc", "production": False,
                           "reference": "main", "candidate": "final"},
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accuracy_margin(receipts):
    metrics = c.gate_metrics(receipts)
    return {
        "target_headroom": c.MARGIN_TARGET,
        "target_rule": ("a two-ULP gate must measure at most 1.5 ULP "
                        "(75 % of its original limit)"),
        "gates": metrics,
        "zero_margin_gates": [m["gate"] for m in metrics if m["headroom_ratio"] >= 1.0],
        "target_met": all(m["meets_margin_target"] for m in metrics),
        "seeds_with_a_failure": sorted(
            seed for seed in c.SEEDS
            if receipts[(seed, "full-fp16", "tile16")]["failed"] > 0
        ),
        "seeds": list(c.SEEDS),
        "configs": ["%s/%s" % pair for pair in c.CONFIGS],
    }


def performance():
    out = {}
    for name, spec in SWEEPS.items():
        directory = c.EVIDENCE / name
        if not directory.is_dir():
            sys.exit(f"missing ABBA evidence directory {directory}")
        stats = phase_lxv_stats.summarize(directory, spec["arm"])
        verdict, reasons = c.non_inferiority(stats) if spec["production"] else ("NOT-A-GATE", [])
        out[name] = {
            "arm": spec["arm"],
            "reference": spec["reference"],
            "candidate": spec["candidate"],
            "production": spec["production"],
            "verdict": verdict,
            "gate_satisfied": verdict == "NON-INFERIOR",
            "reasons": reasons,
            "shapes": stats,
        }
    return out


def hardware_summary():
    manifest = json.loads((c.EVIDENCE / "hardware.json").read_text(encoding="utf-8"))
    captured = {}
    for arm, entry in manifest.items():
        kernels = []
        for code_object in entry["code_objects"]:
            if not code_object["wmma_total"]:
                continue
            kernels.append({
                "code_object_sha256": code_object["code_object_sha256"],
                "instructions": code_object["instructions"],
                "wmma": code_object["wmma"],
                "static_wmma": code_object["wmma_total"],
                "vgpr_count": (code_object["resources"] or {}).get("vgpr_count"),
                "sgpr_count": (code_object["resources"] or {}).get("sgpr_count"),
                "lds_bytes": (code_object["resources"] or {}).get("lds_bytes"),
                "scratch_enabled": (code_object["resources"] or {}).get("scratch_en"),
                "scratch_bytes_per_wave": (code_object["resources"] or {}).get("scratch_bytes_per_wave"),
                "wavefront_size": (code_object["resources"] or {}).get("wavefront_size"),
                "scratch_instructions": code_object["scratch_instructions"],
                "isa_listing": code_object["listing"],
            })
        captured[arm] = {
            "profile": entry["profile"],
            "profile_bytes": entry["profile_bytes"],
            "profile_sha256": entry["profile_sha256"],
            "kernels": kernels,
        }
    return {
        "tool": "Radeon GPU Profiler 2.7.1.13 (RDTS 2026-10-02-1872)",
        "adapter": "AMD Radeon Graphics (Radeon 780M, gfx1103), PCI 1002:15bf",
        "driver": "32.0.31007.6002 (packaging 26.10.07.06-260512a-200783C-Asus)",
        "os": "Windows 11 Enterprise 26100",
        "capture_method": ("RadeonDeveloperPanelCLI.exe -m profiling "
                           "--rgp-auto-capture=dispatch:1:N -p wmma_bench"),
        "isa_tool": ("amd_comgr_3.dll AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE, "
                     "isa amdgcn-amd-amdhsa--gfx1103"),
        "dynamic_hit_counts": ("not claimed: the SQTT trace capture succeeds "
                               "(--rgp-instruction-tracing --rgp-sqtt-buffer-size=maximum) "
                               "but per-instruction hit counts are a GUI reduction of that stream"),
        "captures": captured,
    }


def build(performance_only=False):
    receipts = c.accuracy_receipts()
    default = receipts[(c.SEEDS[0], "full-fp16", "tile16")]
    receipt = {
        "schema": 1,
        "phase": "LXV",
        "base_revision": c.BASE_REVISION,
        "trainer": {
            "source": "hierarchos-vulkan/src/bin/train.rs",
            "source_sha256": digest(ROOT / "hierarchos-vulkan/src/bin/train.rs"),
            "note": "unchanged in this phase; no rebuild was required",
        },
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
        },
        "hardware": hardware_summary(),
        "accuracy": {
            "default_seed": c.SEEDS[0],
            "gates": default["gates"],
            "failed_under_tile16": default["failed"],
            "failed_under_split8": receipts[(c.SEEDS[0], "full-fp16", "split8")]["failed"],
            "worst_vs_fp32": max(row["max_abs"] for row in default["results"]
                                 if row["max_abs_limit"] == 0.05),
            "receipts": sorted(p.name for p in (c.EVIDENCE / "accuracy").glob("*.json")),
        },
        "accuracy_margin": accuracy_margin(receipts),
        "dispatch": {
            "mac_floor": c.MAC_FLOOR,
            "cooperative_dispatches": default["dispatch_counts"]["cooperative"],
            "plain_dispatches": default["dispatch_counts"]["plain_f16"],
            "shape_fallbacks": default["dispatch_counts"]["shape_fallbacks"],
        },
        "sixteen_bank": {
            "implemented": False,
            "reason": "vgpr-wall",
            "detail": ("eight banks already use 8 banks x 4 tiles x 8 VGPRs = 256 VGPRs "
                       "per wave, the architectural maximum, plus 768 B/wave of spill; "
                       "sixteen banks would need 512"),
        },
        "performance": performance(),
    }
    if performance_only:
        existing = json.loads(c.RECEIPT.read_text(encoding="utf-8"))
        existing["performance"] = receipt["performance"]
        receipt = existing
    c.RECEIPT.write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print("wrote", c.RECEIPT)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--performance-only", action="store_true")
    build(parser.parse_args().performance_only)
