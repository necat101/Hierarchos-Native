#!/usr/bin/env python3
"""Phase LXVI gate: shared statistics, historical correction, K=96 margin.

What this gate can and cannot decide
-----------------------------------
Hosted CI is offline. It recomputes statistics from committed raw samples and
checks that the numerical simulation reproduces. It does **not** run a GPU, open
Radeon GPU Profiler, or measure a millisecond. Anything that required hardware is
recorded as a hardware artifact with its own hash and is verified structurally
only; the gate refuses to let such an artifact be described as a hosted
measurement.

Checks:

1. the canonical statistics module has no normal-approximation fallback on a
   live path, computes the reference quantiles, and rejects unsupported sample
   counts;
2. the historical correction record reproduces from the preserved raw samples,
   and the superseded intervals it names really are narrower than the corrected
   ones (so the record cannot be rewritten to hide the defect);
3. the Phase LXV receipt still reproduces, allowing only the documented
   three-decimal table quantisation that predates the correction;
4. the K=96 simulation evidence reproduces, every one of the eight seeds and four
   layouts is present, the limits are unchanged, and no candidate is claimed to
   meet the 25 % headroom target when it does not;
5. required hardware evidence that is absent is reported as absent, not as a
   pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import abba_stats
import phase_lxvi_historical_ci

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RECEIPT = HERE / "phase_lxv_receipt.json"
CORRECTION = HERE / "phase_lxvi_statistics_correction.json"
SIMULATION = HERE / "phase_lxvi_evidence" / "k96_margin.json"
RGP_RECORD = HERE / "phase_lxvi_evidence" / "rgp_instruction_timing.json"
TRAINING_RECORD = HERE / "phase_lxvi_evidence" / "training_step.json"

EVIDENCE_KIND = {
    "k96_margin": "offline recomputation (numerical simulation)",
    "statistics_correction": "offline recomputation",
    "phase_lxv_receipt": "structural verification of recorded hardware evidence",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_statistics_module():
    for df, want in (
        (7, 2.364624), (11, 2.200985), (12, 2.178813), (13, 2.160369),
        (15, 2.131450), (23, 2.068658), (31, 2.039513), (32, 2.036933),
        (33, 2.034515), (47, 2.011741), (63, 1.998341),
    ):
        got = abba_stats.t_critical(df)
        require(abs(got - want) < 5.0e-6, f"t quantile at df={df}: {got} != {want}")
    require(abba_stats.t_critical(10_000) > 1.96,
            "the quantile collapsed to the normal value at a finite df")
    for bad in (0, -1, abba_stats.MAX_DF + 1):
        try:
            abba_stats.t_critical(bad)
        except abba_stats.StatisticsError:
            continue
        raise ValueError(f"t_critical({bad}) was accepted instead of failing closed")
    # No module-level t table anywhere on a live path, and the only place the
    # literal 1.96 may appear is the explicitly archival `mean_ci_normal`.
    import ast

    source = (HERE / "abba_stats.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                require(not (isinstance(target, ast.Name) and "T95" in target.id),
                        "abba_stats.py still defines a t table")
    import io
    import tokenize

    archival = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "mean_ci_normal"
    )
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        if token.string == "1.96":
            require(
                archival.lineno <= token.start[0] <= archival.end_lineno,
                "abba_stats.py uses the normal quantile outside mean_ci_normal",
            )
    for path in ("wmma_abba_analyze.py", "phase_lxv_stats.py"):
        text = (HERE / path).read_text(encoding="utf-8")
        require("import abba_stats" in text, f"{path} does not use the shared module")
        body = text.split('"""')
        code = body[-1] if len(body) > 2 else text
        require("1.96" not in code, f"{path} still carries a normal-approximation quantile")
        require("T95" not in code, f"{path} still carries a t table")


def check_correction():
    committed = json.loads(CORRECTION.read_text(encoding="utf-8"))
    rebuilt = phase_lxvi_historical_ci.build_record()
    require(committed["entries"] == rebuilt["entries"],
            "the correction record does not reproduce from the raw samples")
    require(committed["unrecomputable_sweeps"] == rebuilt["unrecomputable_sweeps"],
            "the unrecomputable-sweep block changed")
    for entry in committed["entries"]:
        lo, hi = entry["corrected_ci95"]
        z_lo, z_hi = entry["z_1p96_ci95"]
        # The defect was substituting z=1.96 above the table. A corrected t
        # interval must be at least as wide as that substitution, always.
        require(hi - lo >= (z_hi - z_lo) - 1.0e-12,
                f"{entry['sweep']}/{entry['shape']}: the corrected interval is "
                "narrower than the superseded normal-approximation interval")
        require(entry["corrected_quantile"] >= 1.96,
                f"{entry['sweep']}/{entry['shape']}: corrected quantile below 1.96")
    return committed


def check_lxv_receipt():
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    correction = json.loads(CORRECTION.read_text(encoding="utf-8"))
    corrected = {
        (entry["sweep"], entry["shape"]): entry["corrected_ci95"]
        for entry in correction["entries"]
    }
    for name, entry in receipt["performance"].items():
        stats = abba_stats.summarize(HERE / "phase_lxv_evidence" / name, entry["arm"])
        for shape, want in entry["shapes"].items():
            got = stats[shape]
            for key in ("median_reference_ms", "median_candidate_ms", "mean_ratio"):
                require(abs(got[key] - want[key]) <= 1.0e-6,
                        f"{name}/{shape}: {key} does not reproduce")
            # The archived receipt was printed from a three-decimal t table. Its
            # interval may differ from the corrected one by that quantisation,
            # but it must equal the documented corrected value to be accepted.
            documented = corrected.get((name, shape))
            require(documented is not None,
                    f"{name}/{shape}: no correction-record entry")
            for index, key in enumerate(("ci95_low", "ci95_high")):
                exact = abs(got[key] - want[key]) <= 1.0e-6
                matches_correction = abs(got[key] - documented[index]) <= 1.0e-9
                require(exact or matches_correction,
                        f"{name}/{shape}: {key} reproduces neither the receipt nor "
                        "the corrected interval")


def check_simulation():
    committed = json.loads(SIMULATION.read_text(encoding="utf-8"))
    require(committed.get("kind", "").startswith("numerical simulation"),
            "the K=96 evidence must be labelled a simulation")
    results = committed["results"]
    seeds = set(committed["seeds"])
    require(len(seeds) == 8, "the simulation must carry eight seeds")
    fixtures = {row["fixture"] for row in results}
    require("128x128x96" in fixtures and "37x8192x48" in fixtures,
            "the K=96 and K=48 fixtures must both be present")
    layouts = {(row["trans_a"], row["trans_b"]) for row in results}
    require(len(layouts) == 4, "all four A/B layouts must be present")
    for row in results:
        require(row["fp32_limit"] == 0.05,
                f"{row['fixture']}: the FP32 gate limit moved")
        require(row["oracle_limit"] == 0.03125,
                f"{row['fixture']}: the matched-oracle gate limit moved")
        require(row["fp32_gate_passes"] == (row["max_abs_vs_fp32"] <= 0.05),
                f"{row['fixture']}: the FP32 gate verdict contradicts its measurement")
        require(row["tile16_gate_passes"] == (row["max_abs_vs_tile16"] <= 0.03125),
                f"{row['fixture']}: the oracle gate verdict contradicts its measurement")
        require(row["trans_a"] in (True, False) and row["trans_b"] in (True, False),
                "layout flags must be booleans")
        require(row["seed"] in seeds, "a result row carries an undeclared seed")
    current = [r for r in results
               if r["model"] == "split8-roundrobin-pairwise" and r["fixture"] == "128x128x96"]
    require(len(current) == 32, "the current structure must cover 8 seeds x 4 layouts")
    require(all(r["fp32_gate_passes"] for r in current),
            "the current eight-bank structure lost its FP32 accuracy gate")
    require(any(r["target_met_vs_tile16"] for r in current) is False,
            "the evidence claims the 25 % margin target is met when it is not")
    # A candidate may only be described as meeting the target if its own rows say so.
    for row in results:
        if row.get("target_met_vs_tile16"):
            require(row["max_abs_vs_tile16"] <= committed["target_for_25pct_headroom"],
                    f"{row['model']}: target claim contradicts its own measurement")


def check_rgp_timing_record(path):
    """If a GUI-derived hit-count export exists, it must carry its scope."""
    if not path.exists():
        return False
    record = json.loads(path.read_text(encoding="utf-8"))
    for key in ("rgp_version", "adapter", "capture_sha256", "capture_bytes",
                "event_id", "pipeline_hash", "static_wmma", "sampling_scope",
                "total_wavefronts", "analyzed_wavefronts", "per_instruction"):
        require(key in record, f"RGP evidence is missing {key}")
    require(len(record["capture_sha256"]) == 64,
            "RGP evidence must carry a full SHA-256 of its capture")
    require(record["sampling_scope"] in ("single_shader_engine",) or
            "single shader engine" in record["sampling_scope"].lower(),
            "RGP evidence must state the single-shader-engine sampling scope")
    require(record["analyzed_wavefronts"] <= record["total_wavefronts"],
            "analyzed wavefronts cannot exceed total wavefronts")
    require(record.get("whole_gpu_normalisation", None) is None,
            "sampled counts must not be presented as whole-GPU totals")
    require(isinstance(record["per_instruction"], list) and record["per_instruction"],
            "RGP evidence must carry per-instruction hit counts")
    for row in record["per_instruction"]:
        require("instruction" in row and "hit_count" in row,
                "every per-instruction row needs instruction and hit_count")
    return True


def check_absent_hardware_evidence():
    """Report what this tree cannot substantiate, rather than implying it does."""
    missing = []
    if not check_rgp_timing_record(RGP_RECORD):
        missing.append(
            "RGP dynamic per-instruction hit counts: no GUI-derived export is "
            "present. The captured .rgp files hold the SQTT stream, but the "
            "hit-count reduction is a Radeon GPU Profiler GUI operation."
        )
    if not TRAINING_RECORD.exists():
        missing.append(
            "Steady-state training-step measurement: no verified checkpoint, "
            "dataset and paired training-step raw samples are present."
        )
    if not (HERE / "phase_lxvi_evidence" / "full_fp16_bank_reuse.json").exists():
        missing.append(
            "Full FP16 bank-reuse candidate: not built or measured in this phase."
        )
    return missing


def validate(with_missing=True):
    check_statistics_module()
    correction = check_correction()
    check_lxv_receipt()
    check_simulation()
    missing = check_absent_hardware_evidence() if with_missing else []
    return {
        "kind": EVIDENCE_KIND,
        "correction_entries": len(correction["entries"]),
        "conclusion_changes": correction["conclusion_changes"],
        "missing_evidence": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true",
                        help="fail when required hardware evidence is absent")
    args = parser.parse_args()
    try:
        result = validate()
    except ValueError as error:
        print(f"Phase LXVI gate FAILED: {error}", file=sys.stderr)
        return 1
    print("Phase LXVI gate")
    print(f"  correction entries verified: {result['correction_entries']}")
    print(f"  conclusion changes under the corrected interval: "
          f"{len(result['conclusion_changes'])}")
    print("  hosted classification: structural + offline recomputation only")
    if result["missing_evidence"]:
        print("  evidence that is still absent (not claimed as verified):")
        for line in result["missing_evidence"]:
            print(f"    - {line}")
    if args.strict and result["missing_evidence"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
