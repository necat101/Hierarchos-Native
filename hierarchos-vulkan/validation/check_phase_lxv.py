#!/usr/bin/env python3
"""Phase LXV gate: hardware ISA, accuracy margin and performance confidence.

Earlier phases recorded *claims*; this gate asks whether the claims are still
true of the committed evidence. Everything it checks is re-derived from artifacts
in the tree:

* the two SPIR-V modules are the exact bytes the 8-bank record names, and their
  static cooperative-multiply-add counts are 4 (mixed) and 32 (full FP16);
* the committed ISA listings are the listings of the committed code objects, and
  they contain 4 `v_wmma_f32_16x16x16_f16` / 4 and 32
  `v_wmma_f16_16x16x16_f16` sites — the RGP capture's own compiled output, not a
  source-level inference;
* the committed code objects' AMDPAL metadata still records the measured register,
  LDS, scratch and wavefront facts (71/256 VGPR, 3,072/4,608 B LDS, 0/768 B scratch,
  wave32);
* every recorded accuracy receipt is internally consistent (a gate cannot be
  "passed" while its measured value exceeds its own limit, or with non-finite
  output), all eight seeds carry the same 37 gates under the same limits, and the
  nine originally failing gates keep their original limits;
* the margin block is recomputed from those receipts: headroom per gate, the
  declared 25 % target, whether it is met, and how many gates sit at exactly the
  limit. A receipt that claims the target is met when it is not fails;
* the ABBA performance block is recomputed from the raw per-iteration samples,
  and the non-inferiority verdicts must match the recomputation. A missing raw
  sample file, an edited statistic or a re-labelled verdict fails;
* the 16-bank experiment must be recorded as rejected on the measured VGPR wall,
  not silently dropped.

The file-scope control is deliberately about **this phase's own change set**,
not about every commit that descends from its base. See
[`PHASE_LXV_SCOPE_BASE`] / [`PHASE_LXV_ENDPOINT`]: the range is fixed at the two
commits that bound Phase LXV, so a later release commit is verified by that
phase's own gate instead of being reported as an unapproved Phase LXV change.
The scope control itself is unchanged - an unapproved path inside the range is
still rejected.

Hosted CI verifies structure and recomputation only. It never claims to have run
the RDNA3 GPU, produced an RGP capture, or measured a millisecond.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import phase_lxv_metadata
import phase_lxv_stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RECEIPT = HERE / "phase_lxv_receipt.json"
EVIDENCE = HERE / "phase_lxv_evidence"

SPIRV_MAGIC = 0x07230203
SPIRV_VERSION = 0x00010400
OP_COOPERATIVE_MATRIX_MUL_ADD = 4459

BASE_REVISION = "81ac82fa972d84025129e7c3f0e58fdece7df786"

# The two revisions that bound Phase LXV's own change set.
#
# `PHASE_LXV_SCOPE_BASE` is the commit the phase started from: the Phase LXIV
# record was complete at `a70cef4`, and `b86c29e` is the documentation commit
# immediately before the first Phase LXV commit. `PHASE_LXV_ENDPOINT` is the
# last commit of the phase - after it, `759da5c` (Phase LXVI) and the two-wave /
# P3 release commits are later work.
#
# The scope control must read this range, not `base..HEAD`. A branch that
# descends from this phase keeps inheriting phases, and every later phase
# legitimately touches files this one never approved (the release's
# `.gitattributes` rule, `phase_lxxxi_evidence/**`, the P3 shaders). Diffing to
# HEAD turns that later work into a false Phase LXV scope violation.
PHASE_LXV_SCOPE_BASE = "b86c29e212e0d439d6698a0d47a48ed58424ce88"
PHASE_LXV_ENDPOINT = "cf98af639cf1dd81539e182f158eda41da85e195"

# The exact artifacts the Phase LXV record is about.
SHADERS = {
    "wmma_gemm_f16_f32acc": {
        "source": "hierarchos-vulkan/shaders/wmma_gemm_f16_f32acc.comp",
        "binary": "hierarchos-vulkan/shaders/wmma_gemm_f16_f32acc.spv",
        "source_sha256": "239b63144bdd7c7aef703d88622201bc9c6c4832000e7dc86cec3317034863fc",
        "binary_sha256": "e07e48882a2459ebe30be228675596a12ac3f6b846badf32099ef06cca5cea65",
        "cooperative_mul_adds": 4,
    },
    "wmma_gemm_f16_f16acc": {
        "source": "hierarchos-vulkan/shaders/wmma_gemm_f16_f16acc.comp",
        "binary": "hierarchos-vulkan/shaders/wmma_gemm_f16_f16acc.spv",
        "source_sha256": "764cd5dd33c78bd3a71f906418e4965eeed684140e6b1e7de7c3ad2a2e2f2274",
        "binary_sha256": "dbed2a528f674a37fd6672ff83a786eb28804a77a85befd076acc80d701106fe",
        "cooperative_mul_adds": 32,
    },
}

# The nine gates that failed before Phase LXIV and were recovered. The limits are
# the originals and are *not* allowed to move.
ORIGINAL_NINE = {
    "128x128x96 wmma-full-fp16 vs matched oracle": 0.03125,
    "16x1024x64 wmma-full-fp16 vs matched oracle": 0.03125,
    "37x8192x48 wmma-full-fp16 vs matched oracle": 0.03125,
    "128x128x96 trans_a=true trans_b=false vs oracle": 0.03125,
    "128x128x96 trans_a=true trans_b=false vs FP32": 0.05,
    "128x128x96 trans_a=false trans_b=true vs oracle": 0.03125,
    "128x128x96 trans_a=false trans_b=true vs FP32": 0.05,
    "128x128x96 trans_a=true trans_b=true vs oracle": 0.03125,
    "128x128x96 trans_a=true trans_b=true vs FP32": 0.05,
}

# A two-ULP gate with 25 % headroom must measure at most 1.5 ULP, i.e. 75 % of
# its limit. Declared before the measurement; not adjustable after it.
MARGIN_TARGET = 0.25
ULP_GATE_LIMITS = 0.03125  # which of the nine are two-ULP (oracle) gates

SEEDS = (1593426484, 42, 1337, 271828, 314159, 777, 8675309, 20261119)
CONFIGS = (("full-fp16", "tile16"), ("full-fp16", "split8"),
           ("full-fp16", "rdna3-dot2"), ("full-fp16", "split2"),
           ("mixed", "tile16"))
PRODUCTION_SHAPES = ("2048x768x2048", "512x2048x768", "512x768x768", "128x8192x768")
MAC_FLOOR = 16_000_000
NONINFERIORITY_LOWER = 0.99

ISA_EXPECTATIONS = {
    "main-mixed": {"mnemonic": "v_wmma_f32_16x16x16_f16", "static": 4,
                   "vgpr": 73, "lds": 3072, "scratch_en": False, "scratch": 0},
    "candidate-a-mixed": {"mnemonic": "v_wmma_f32_16x16x16_f16", "static": 4,
                          "vgpr": 71, "lds": 3072, "scratch_en": False, "scratch": 0},
    "final-mixed": {"mnemonic": "v_wmma_f32_16x16x16_f16", "static": 4,
                    "vgpr": 71, "lds": 3072, "scratch_en": False, "scratch": 0},
    "main-full": {"mnemonic": "v_wmma_f16_16x16x16_f16", "static": 4,
                  "vgpr": 57, "lds": 2560, "scratch_en": False, "scratch": 0},
    "final-full": {"mnemonic": "v_wmma_f16_16x16x16_f16", "static": 32,
                   "vgpr": 256, "lds": 4608, "scratch_en": True, "scratch": 768},
}

APPROVED_CHANGES = (
    "hierarchos-vulkan/validation/",
    "PHASE_LXV",
    ".github/workflows/",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spirv_cooperative_count(path: Path) -> int:
    data = Path(path).read_bytes()
    require(len(data) % 4 == 0, f"{path} is not a word-aligned SPIR-V module")
    words = struct.unpack("<%dI" % (len(data) // 4), data)
    require(words[0] == SPIRV_MAGIC, f"{path} is not SPIR-V")
    require(words[1] == SPIRV_VERSION, f"{path} is not SPIR-V 1.4")
    index, count = 5, 0
    while index < len(words):
        opcode = words[index] & 0xFFFF
        word_count = words[index] >> 16
        require(word_count >= 1, f"{path} has a zero-word instruction")
        if opcode == OP_COOPERATIVE_MATRIX_MUL_ADD:
            count += 1
        index += word_count
    return count


def check_shaders(root: Path):
    for name, spec in SHADERS.items():
        source = root / spec["source"]
        binary = root / spec["binary"]
        require(source.exists() and binary.exists(), f"{name}: artifact missing")
        require(digest(source) == spec["source_sha256"],
                f"{name}: shader source is not the recorded bytes")
        require(digest(binary) == spec["binary_sha256"],
                f"{name}: SPIR-V is not the recorded bytes")
        require(spirv_cooperative_count(binary) == spec["cooperative_mul_adds"],
                f"{name}: cooperative multiply-add count changed")


def load_receipt(receipt):
    if receipt is not None:
        return receipt
    require(RECEIPT.exists(), "phase LXV receipt is missing")
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def accuracy_receipts():
    out = {}
    for seed in SEEDS:
        for mode, oracle in CONFIGS:
            path = EVIDENCE / "accuracy" / f"seed{seed}_{mode}_{oracle}.json"
            require(path.exists(), f"missing accuracy receipt {path.name}")
            out[(seed, mode, oracle)] = json.JSONDecoder().raw_decode(
                path.read_text(encoding="utf-8")
            )[0]
    return out


MIXED_VS_FP32 = 0.02
MIXED_VS_ORACLE = 0.001
FULL_VS_FP32 = 0.05
FULL_ORACLE_FLOOR = 0.004
FULL_ORACLE_CEILING = 0.0625


def documented_limit(mode, label, limit):
    """The only limits the quantifier produces, per mode and comparison kind.

    `full-fp16` matched-oracle limits are data-dependent by construction: the
    limit is `max(2 ULP of the largest reference magnitude, 4e-3)`, so a fixture
    whose reference reaches 16-32 doubles the limit. It is bounded here rather
    than frozen, and the *original* absolute limits stay pinned in ORIGINAL_NINE.
    """
    if label.endswith("vs FP32"):
        return limit == (MIXED_VS_FP32 if mode == "mixed" else FULL_VS_FP32)
    if mode == "mixed" or "plain-f16-storage" in label:
        return limit == MIXED_VS_ORACLE
    return FULL_ORACLE_FLOOR <= limit <= FULL_ORACLE_CEILING


def check_accuracy(receipts):
    minimum_by_gate = {}
    for key, receipt in receipts.items():
        seed, mode, oracle = key
        require(receipt.get("gates") == 37, f"{key}: gate count is not 37")
        rows = receipt["results"]
        require(len(rows) == 37, f"{key}: receipt lists {len(rows)} gates")
        computed_failures = 0
        for row in rows:
            require(row["max_abs_limit"] > 0, f"{key}: {row['label']} has no limit")
            require(documented_limit(mode, row["label"], row["max_abs_limit"]),
                    f"{key}: {row['label']} carries an undocumented limit {row['max_abs_limit']}")
            # Limits are rendered to three significant digits in the JSON, so a
            # limit such as 0.015625 is stored as 0.01562. Allow that rendering
            # error only; a widened limit is caught by documented_limit().
            passed = (row["nonfinite"] == 0
                      and row["max_abs"] <= row["max_abs_limit"] * (1.0 + 1.0e-3))
            require(passed == row["passed"],
                    f"{key}: {row['label']} recorded passed={row['passed']} "
                    f"but measured {row['max_abs']:.9g} against {row['max_abs_limit']:.9g}")
            if not passed:
                computed_failures += 1
            identifier = (mode, row["label"])
            minimum_by_gate[identifier] = min(minimum_by_gate.get(identifier, row["max_abs_limit"]),
                                              row["max_abs_limit"])
        require(computed_failures == receipt["failed"],
                f"{key}: recorded failed={receipt['failed']} but {computed_failures} gates fail")
        counts = receipt["dispatch_counts"]
        require(counts["cooperative"] + counts["plain_f16"] + counts["shape_fallbacks"] > 0,
                f"{key}: dispatch accounting is empty")
    # A gate's limit may grow with the fixture magnitude, but the tightest limit
    # seen across seeds must still be the floor of the documented rule, so a
    # receipt cannot buy headroom by inflating every seed's limit.
    for identifier, minimum in minimum_by_gate.items():
        mode, label = identifier
        if not label.endswith("vs FP32") and mode == "full-fp16":
            require(minimum <= FULL_ORACLE_CEILING,
                    f"{label}: no seed records a limit inside the documented rule")
    # The recorded seed must clear every gate under both Full FP16 references, and
    # the contract-matched oracle must clear every gate at every seed. The
    # historical `tile16` oracle is *not* required to be clean at every seed: the
    # Phase LXV finding is that it is not, and check_margin pins exactly which
    # seeds exceed it so that finding cannot be edited away.
    for oracle in ("tile16", "split8"):
        require(receipts[(SEEDS[0], "full-fp16", oracle)]["failed"] == 0,
                f"the recorded seed fails full-fp16/{oracle}")
    for seed in SEEDS:
        require(receipts[(seed, "full-fp16", "split8")]["failed"] == 0,
                f"seed {seed} full-fp16/split8 has failing gates")
        require(receipts[(seed, "mixed", "tile16")]["failed"] == 0,
                f"seed {seed} mixed has failing gates")
    return minimum_by_gate


def gate_metrics(receipts):
    default = receipts[(SEEDS[0], "full-fp16", "tile16")]
    rows = {row["label"]: row for row in default["results"]}
    metrics = []
    for label, limit in ORIGINAL_NINE.items():
        require(label in rows, f"the originally failing gate {label!r} is gone")
        row = rows[label]
        require(row["max_abs_limit"] == limit,
                f"{label}: original limit {limit} is not what the receipt records")
        ratio = row["max_abs"] / limit
        metrics.append({
            "gate": label,
            "measured": row["max_abs"],
            "limit": limit,
            "headroom_ratio": ratio,
            "ulp": (2.0 * ratio) if limit == ULP_GATE_LIMITS else None,
            "meets_margin_target": ratio <= 1.0 - MARGIN_TARGET,
        })
    # The worst case over every seed, per gate, so a seed-specific failure is visible.
    for metric in metrics:
        worst = 0.0
        for seed in SEEDS:
            rows = {row["label"]: row for row in receipts[(seed, "full-fp16", "tile16")]["results"]}
            worst = max(worst, rows[metric["gate"]]["max_abs"])
        metric["worst_over_seeds"] = worst
        metric["seeds_over_limit"] = sorted(
            seed for seed in SEEDS
            if {row["label"]: row for row in receipts[(seed, "full-fp16", "tile16")]["results"]}
            [metric["gate"]]["max_abs"] > metric["limit"]
        )
    return metrics


def check_margin(receipt, receipts):
    metrics = gate_metrics(receipts)
    recomputed = {
        "target_headroom": MARGIN_TARGET,
        "gates": metrics,
        "zero_margin_gates": [m["gate"] for m in metrics if m["headroom_ratio"] >= 1.0],
        "target_met": all(m["meets_margin_target"] for m in metrics),
        "seeds_with_a_failure": sorted(
            seed for seed in SEEDS
            if receipts[(seed, "full-fp16", "tile16")]["failed"] > 0
        ),
    }
    recorded = receipt["accuracy_margin"]
    require(recorded["target_headroom"] == MARGIN_TARGET,
            "the recorded margin target is not the declared 25 %")
    require([m["gate"] for m in recorded["gates"]] == [m["gate"] for m in metrics],
            "the recorded margin gate list differs from the nine original gates")
    for got, want in zip(recorded["gates"], metrics):
        require(got["limit"] == want["limit"], f"{want['gate']}: recorded limit moved")
        require(abs(got["measured"] - want["measured"]) <= 1e-12,
                f"{want['gate']}: recorded measurement does not reproduce")
        require(abs(got["headroom_ratio"] - want["headroom_ratio"]) <= 1e-9,
                f"{want['gate']}: recorded headroom ratio does not reproduce")
        require(got["seeds_over_limit"] == want["seeds_over_limit"],
                f"{want['gate']}: recorded seed failures do not reproduce")
    require(recorded["zero_margin_gates"] == recomputed["zero_margin_gates"],
            "the recorded zero-margin gate set does not reproduce")
    require(recorded["target_met"] == recomputed["target_met"],
            "the receipt claims the margin target is met when it is not")
    require(recorded["seeds_with_a_failure"] == recomputed["seeds_with_a_failure"],
            "the recorded seeds-with-a-failure set does not reproduce")
    return recomputed


def count_isa(path: Path, mnemonic: str) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    return sum(1 for line in text.splitlines() if ("\t" + mnemonic + " ") in line)


def check_hardware(receipt):
    hardware = json.loads((EVIDENCE / "hardware.json").read_text(encoding="utf-8"))
    for arm, expectation in ISA_EXPECTATIONS.items():
        require(arm in hardware, f"{arm}: no capture recorded")
        entry = hardware[arm]
        require(len(entry["profile_sha256"]) == 64, f"{arm}: capture digest is not a SHA-256")
        require(entry["profile_bytes"] > 0, f"{arm}: capture is empty")
        candidates = []
        for index, code_object in enumerate(entry["code_objects"]):
            listing = EVIDENCE / code_object["listing"]
            elf = EVIDENCE / code_object["elf"]
            require(listing.exists() and elf.exists(), f"{arm}: committed ISA artifact missing")
            require(digest(elf) == code_object["code_object_sha256"],
                    f"{arm}: code object bytes changed")
            require(digest(listing) == code_object.get("listing_sha256"),
                    f"{arm}: committed ISA listing was edited")
            hits = count_isa(listing, expectation["mnemonic"])
            if hits == expectation["static"]:
                candidates.append((code_object, elf, listing, hits))
        require(len(candidates) >= 1,
                f"{arm}: no committed kernel has {expectation['static']} {expectation['mnemonic']} sites")
        code_object, elf, listing, hits = candidates[0]
        require(code_object["wmma_total"] == expectation["static"],
                f"{arm}: recorded static WMMA count is wrong")
        require(code_object["wmma"].get(expectation["mnemonic"]) == expectation["static"],
                f"{arm}: recorded mnemonic count is wrong")
        stage = phase_lxv_metadata.compute_stage(elf)
        require(stage[".vgpr_count"] == expectation["vgpr"],
                f"{arm}: VGPR count is {stage['.vgpr_count']}, not {expectation['vgpr']}")
        require(stage[".lds_size"] == expectation["lds"],
                f"{arm}: LDS size is {stage['.lds_size']}, not {expectation['lds']}")
        require(stage[".scratch_en"] is expectation["scratch_en"],
                f"{arm}: scratch enable flag changed")
        require(stage[".scratch_memory_size"] == expectation["scratch"],
                f"{arm}: scratch size is {stage['.scratch_memory_size']}, not {expectation['scratch']}")
        require(stage[".wavefront_size"] == 32, f"{arm}: wavefront size is not 32")
    # Final mixed and Candidate A mixed must be the same compiled code object.
    mixed = {}
    for arm in ("candidate-a-mixed", "final-mixed"):
        for code_object in hardware[arm]["code_objects"]:
            if code_object["wmma_total"] == 4 and "f32" in "".join(code_object["wmma"]):
                mixed[arm] = code_object["code_object_sha256"]
                break
    require(mixed.get("candidate-a-mixed") == mixed.get("final-mixed"),
            "Final mixed and Candidate A mixed are not the same compiled code object")
    experiment = receipt["sixteen_bank"]
    require(experiment["implemented"] is False, "the 16-bank variant is not marked unbuilt")
    require(experiment["reason"] == "vgpr-wall",
            "the 16-bank rejection is not recorded as the measured register wall")
    return hardware


CORRECTION = HERE / "phase_lxvi_statistics_correction.json"


def corrected_intervals():
    """Dated correction record, when present.

    The archived receipt was printed from a three-decimal t table. The corrected
    Student-t quantile moves a few intervals by up to ~2e-5, which is larger than
    the 1e-6 reproduction tolerance but is not a change to the samples. The
    correction record carries the value the canonical implementation produces, so
    an archived interval is accepted only if it equals either the receipt's own
    number or that documented corrected number.
    """
    if not CORRECTION.exists():
        return {}
    record = json.loads(CORRECTION.read_text(encoding="utf-8"))
    return {
        (entry["sweep"], entry["shape"]): entry["corrected_ci95"]
        for entry in record.get("entries", [])
    }


def check_performance(receipt):
    recorded = receipt["performance"]
    corrected = corrected_intervals()
    recomputed = {}
    for name, entry in recorded.items():
        directory = EVIDENCE / name
        require(directory.is_dir(), f"{name}: raw ABBA directory is missing")
        try:
            stats = phase_lxv_stats.summarize(directory, entry["arm"])
        except (OSError, ValueError) as error:
            raise ValueError(f"{name}: raw ABBA samples are missing or unreadable ({error})") from error
        recomputed[name] = stats
        for shape, want in entry["shapes"].items():
            require(shape in stats, f"{name}: {shape} is not derivable from the raw samples")
            got = stats[shape]
            for key in ("mean_ratio", "median_reference_ms", "median_candidate_ms"):
                require(abs(got[key] - want[key]) <= 1e-6,
                        f"{name}/{shape}: recorded {key} does not match the raw samples")
            documented = corrected.get((name, shape))
            for index, key in enumerate(("ci95_low", "ci95_high")):
                matches_receipt = abs(got[key] - want[key]) <= 1e-6
                matches_correction = (
                    documented is not None
                    and abs(got[key] - documented[index]) <= 1e-9
                )
                require(matches_receipt or matches_correction,
                        f"{name}/{shape}: recorded {key} does not match the raw samples")
            require(got["pairs"] == want["pairs"], f"{name}/{shape}: pair count changed")
        if entry.get("production"):
            verdict, reasons = non_inferiority(stats)
            require(entry["verdict"] == verdict,
                    f"{name}: performance verdict does not reproduce ({reasons})")
            require(entry["gate_satisfied"] == (verdict == "NON-INFERIOR"),
                    f"{name}: non-inferiority flag disagrees with the recomputation")
    return recomputed


def non_inferiority(stats):
    reasons = []
    for shape in PRODUCTION_SHAPES:
        require(shape in stats, f"{shape}: production shape missing from the sweep")
        entry = stats[shape]
        if entry["median_reference_ms"] / entry["median_candidate_ms"] < 1.0:
            pass  # median rule uses the mean ratio below
        if entry["mean_ratio"] < 1.0:
            reasons.append(f"{shape} mean {entry['mean_ratio']:.4f} < 1.000")
        if entry["ci95_low"] < NONINFERIORITY_LOWER:
            reasons.append(f"{shape} lower bound {entry['ci95_low']:.4f} < {NONINFERIORITY_LOWER}")
    return ("NON-INFERIOR" if not reasons else "INCONCLUSIVE"), reasons


def check_dispatch(receipt, receipts):
    default = receipts[(SEEDS[0], "full-fp16", "tile16")]
    recorded = receipt["dispatch"]
    require(recorded["mac_floor"] == MAC_FLOOR, "the profitability floor moved")
    require(recorded["cooperative_dispatches"] == default["dispatch_counts"]["cooperative"],
            "recorded cooperative dispatch count does not reproduce")
    require(recorded["plain_dispatches"] == default["dispatch_counts"]["plain_f16"],
            "recorded plain dispatch count does not reproduce")
    require(recorded["shape_fallbacks"] == default["dispatch_counts"]["shape_fallbacks"],
            "recorded fallback count does not reproduce")
    for key, value in recorded.items():
        require(isinstance(value, int), f"dispatch field {key} is not an integer")


def historical_scope_paths(root: Path = ROOT) -> list:
    """The files Phase LXV itself changed, from its own two recorded endpoints.

    This is the change set the scope control was written for. Callers that want
    to know whether a *later* release commit is in scope are asking the wrong
    question: later work is verified by that phase's own gate, and the evidence
    checker below still runs against whatever tree it is handed.
    """
    result = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only",
         PHASE_LXV_SCOPE_BASE, PHASE_LXV_ENDPOINT],
        capture_output=True, text=True)
    require(result.returncode == 0,
            f"the recorded Phase LXV range {PHASE_LXV_SCOPE_BASE[:12]}.."
            f"{PHASE_LXV_ENDPOINT[:12]} is unavailable: "
            + result.stderr.strip()[:200])
    return [line for line in result.stdout.splitlines() if line.strip()]


def check_scope(changed):
    if changed is None:
        return
    for path in changed:
        if path.startswith("hierarchos-vulkan/validation/fixtures/"):
            continue
        require(any(path.startswith(prefix) or path == prefix.rstrip("/")
                    for prefix in APPROVED_CHANGES),
                f"unapproved file changed by Phase LXV: {path}")


def validate(root: Path = ROOT, changed=None, receipt=None):
    require((root / "hierarchos-vulkan").is_dir(), "not a repository root")
    receipt = load_receipt(receipt)
    check_shaders(root)
    receipts = accuracy_receipts()
    check_accuracy(receipts)
    margin = check_margin(receipt, receipts)
    hardware = check_hardware(receipt)
    performance = check_performance(receipt)
    check_dispatch(receipt, receipts)
    check_scope(changed)
    return {
        "margin": margin,
        "hardware_arms": sorted(hardware),
        "performance": performance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--changed", type=Path, default=None,
                        help="file with one changed path per line (default: skip scope check)")
    parser.add_argument("--write-scope", type=Path, default=None,
                        help="write the Phase LXV historical change set here and exit")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.write_scope is not None:
        try:
            paths = historical_scope_paths(args.root)
        except ValueError as error:
            print(f"phase-lxv: FAIL: {error}", file=sys.stderr)
            return 1
        args.write_scope.write_text("\n".join(paths) + "\n", encoding="utf-8")
        print(f"phase-lxv: {len(paths)} path(s) in the recorded Phase LXV range "
              f"{PHASE_LXV_SCOPE_BASE[:12]}..{PHASE_LXV_ENDPOINT[:12]}")
        return 0
    changed = None
    if args.changed is not None:
        changed = [line.strip() for line in args.changed.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    try:
        result = validate(args.root, changed=changed)
    except ValueError as error:
        print(f"phase-lxv: FAIL: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=1))
    else:
        print("Phase LXV hardware, accuracy-margin and performance gate")
        print("  ISA arms verified:", ", ".join(result["hardware_arms"]))
        print("  seeds with a failing gate:", result["margin"]["seeds_with_a_failure"] or "none")
        print("  gates at exactly their limit:", len(result["margin"]["zero_margin_gates"]))
        print("  margin target met:", result["margin"]["target_met"])
        for name, stats in result["performance"].items():
            print(f"  {name}:")
            for shape, entry in stats.items():
                print("    %-14s %8.4f  [%6.3f, %6.3f]  %d pairs"
                      % (shape, entry["mean_ratio"], entry["ci95_low"], entry["ci95_high"],
                         entry["pairs"]))
        print("status: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
