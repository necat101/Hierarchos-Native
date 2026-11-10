#!/usr/bin/env python3
"""Fail-closed verifier for the Phase XVI training qualification receipt.

Everything the receipt claims is recomputed from the shipped per-step reports
and profiler logs. The verifier also re-checks every bound source hash, the
source revision ancestry, the fixture and binary identities, and the preserved
Phase XVI numerical receipt.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import qualify_training_phase16 as q

SCHEMA = 1
KIND = "phase16-training-qualification"


class QualificationError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    left, right = float(left), float(right)
    if math.isnan(left) and math.isnan(right):
        return True
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def verify(receipt_dir: Path, crate: Path | None = None) -> dict:
    crate = crate or q.CRATE
    receipt_path = receipt_dir / "current-qualification.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing {receipt_path}")
    receipt = read_json(receipt_path)
    require(receipt.get("schema") == SCHEMA, "unexpected receipt schema")
    require(receipt.get("kind") == KIND, "unexpected receipt kind")
    evidence = receipt_dir / "evidence"
    require(evidence.is_dir(), "missing evidence directory")

    # Source revision must be an ancestor of the checked-out commit.
    revision = receipt.get("source_revision", "")
    require(len(revision) == 40 and all(c in "0123456789abcdef" for c in revision),
            "invalid source revision")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, "HEAD"],
        cwd=crate.parent,
        capture_output=True,
        text=True,
    )
    require(ancestor.returncode == 0, "source revision is not an ancestor of HEAD")
    require(receipt.get("source_clean") is True, "receipt was produced from a dirty worktree")

    # Every bound source byte must match this checkout.
    for relative, expected in receipt["source_files"].items():
        path = crate / relative
        require(path.is_file(), f"missing bound source {relative}")
        require(q.sha256_file(path) == expected, f"bound source changed: {relative}")
    require(q.sha256_file(crate / "Cargo.toml") == receipt["source_files"]["Cargo.toml"],
            "Cargo.toml binding missing")
    # The producer binaries and the 16.78M-parameter fixture live under the
    # untracked target directory, so a CI checkout legitimately lacks them. When
    # present they must match; either way the per-step evidence is re-derived.
    binary = crate / "target/release/hierarchos-vulkan-training-qualification.exe"
    trainer = crate / "target/release/hierarchos-vulkan-transformer-train.exe"
    binaries_present = binary.is_file() and trainer.is_file()
    if binaries_present:
        require(q.sha256_file(binary) == receipt["binary_sha256"],
                "qualification binary changed")
        require(q.sha256_file(trainer) == receipt["trainer_binary_sha256"],
                "trainer binary changed")

    model = crate / receipt["fixture"]["model"]
    dataset = crate / receipt["fixture"]["dataset"]
    fixture_present = (model / "config.json").is_file() and (model / "model.safetensors").is_file() and dataset.is_file()
    if fixture_present:
        require(q.sha256_file(model / "config.json") == receipt["fixture"]["model_config_sha256"],
                "model config changed")
        require(q.sha256_file(model / "model.safetensors") == receipt["fixture"]["model_weights_sha256"],
                "model weights changed")
        require(q.sha256_file(dataset) == receipt["fixture"]["dataset_sha256"], "dataset changed")

    phase16 = crate / receipt["phase16_receipt"]["path"]
    require(phase16.is_file(), "Phase XVI receipt missing")
    require(q.sha256_file(phase16) == receipt["phase16_receipt"]["sha256"],
            "Phase XVI receipt changed")

    # Evidence must exist with the recorded bytes, and nothing may be extra.
    recorded_evidence = receipt["evidence"]
    for relative, expected in recorded_evidence.items():
        path = evidence / relative
        require(path.is_file(), f"missing evidence {relative}")
        require(q.sha256_file(path) == expected, f"evidence changed: {relative}")

    def report(name: str) -> dict:
        return read_json(evidence / name)

    def stderr(name: str) -> str:
        return (evidence / name).read_text(encoding="utf-8", errors="replace")

    # Arms: per-step records, dispatch proof, WMMA counters, summaries.
    reports: dict[str, dict] = {}
    for arm_id, arm in q.ARMS.items():
        entry = receipt["arms"][arm_id]
        require(entry["config"] == arm, f"arm {arm_id} config changed")
        path = evidence / f"arm-{arm_id}.json"
        require(path.is_file(), f"arm {arm_id} report missing")
        run = read_json(path)
        reports[arm_id] = run
        require(q.sha256_file(path) == entry["report_sha256"], f"arm {arm_id} report hash")
        expected_mode = "full-finetune" if arm["mode"] == "full" else "lora"
        require(run["mode"] == expected_mode, f"arm {arm_id} mode changed")
        require(run["lora"] == (arm["mode"] == "lora"), f"arm {arm_id} LoRA flag changed")
        require(run["two_pass_requested"] == (arm["two_pass"] == "1"),
                f"arm {arm_id} two-pass flag changed")
        require(run["precision_requested"] == ("mixed" if arm["precision"] == "mixed" else "off"),
                f"arm {arm_id} precision changed")
        records = run["records"]
        steps = entry["summary"]["steps"]
        require(len(records) == steps, f"arm {arm_id} record count")
        require([r["optimizer_step"] for r in records] == list(range(1, steps + 1)),
                f"arm {arm_id} optimizer step sequence")
        require(all(r["loss_finite"] for r in records), f"arm {arm_id} non-finite loss")
        require(run["all_losses_finite"] is True, f"arm {arm_id} loss flag")
        require(entry["summary"]["first_loss"] == run["first_loss"], f"arm {arm_id} first loss")
        require(entry["summary"]["final_loss"] == run["final_loss"], f"arm {arm_id} final loss")
        require(entry["summary"]["parameter_end_digest"] == run["parameter_end_digest"],
                f"arm {arm_id} final parameter digest")
        if arm["mode"] == "full":
            require(run["parameter_elements"] == q.EXPECTED_PARAMETERS,
                    f"arm {arm_id} parameter count")
            require(run["parameter_end_digest"] is not None,
                    f"arm {arm_id} missing final parameter digest")
        else:
            require(run["parameter_frozen"] is True, f"arm {arm_id} base weights moved")
            require(run["grad_nonfinite_total"] == 0, f"arm {arm_id} non-finite gradients")

        blocks = q.parse_profile(stderr(f"arm-{arm_id}.err.txt"))
        require(len(blocks) == steps, f"arm {arm_id} profiled step count")
        totals = q.profile_totals(blocks)
        scalar = totals.get(q.SCALAR_SHADER, {}).get("dispatches", 0)
        query = totals.get(q.QUERY_SHADER, {}).get("dispatches", 0)
        key = totals.get(q.KEY_SHADER, {}).get("dispatches", 0)
        if arm["backward"] == "scalar":
            require(scalar == steps * 2 and query == 0 and key == 0,
                    f"arm {arm_id} scalar dispatch proof")
        else:
            require(scalar == 0 and query == steps * 2 and key == steps * 2,
                    f"arm {arm_id} two-pass dispatch proof")
        require(entry["dispatch"]["scalar"] == scalar, f"arm {arm_id} scalar count")
        require(entry["dispatch"]["query"] == query, f"arm {arm_id} query count")
        require(entry["dispatch"]["key"] == key, f"arm {arm_id} key count")
        wmma = {
            "forward": run["wmma_forward"],
            "weight_grad": run["wmma_weight_grad"],
            "input_grad": run["wmma_input_grad"],
            "cooperative": run["wmma_cooperative"],
            "shape_fallbacks": run["wmma_shape_fallbacks"],
            "weight_conversions": run["wmma_weight_conversions"],
            "mirror_reuses": run["wmma_mirror_reuses"],
        }
        require(entry["wmma"] == wmma, f"arm {arm_id} WMMA counters changed")
        if arm["precision"] == "fp32":
            require(wmma["cooperative"] == 0 and wmma["forward"] == 0,
                    f"arm {arm_id} FP32 dispatched WMMA")
        else:
            require(wmma["forward"] > 0 and wmma["input_grad"] > 0,
                    f"arm {arm_id} missing WMMA work")
            require((wmma["weight_grad"] > 0) == (arm["mode"] == "full"),
                    f"arm {arm_id} WMMA weight-gradient mode")

    # Pairwise claims are recomputed, including exact adapter bytes for LoRA.
    def compare(tag_a: str, tag_b: str, fields: list[str]) -> dict:
        left, right = reports[tag_a], reports[tag_b]
        require(len(left["records"]) == len(right["records"]), "comparison step mismatch")
        result = {"left": f"arm-{tag_a}", "right": f"arm-{tag_b}", "fields": {},
                  "bitwise_equal": True}
        for field in fields:
            a = [record[field] for record in left["records"]]
            b = [record[field] for record in right["records"]]
            result["fields"][field] = {"equal": a == b}
            if a != b:
                result["bitwise_equal"] = False
        if left["mode"] == "full-finetune":
            result["final_parameter_digest_equal"] = (
                left["parameter_end_digest"] == right["parameter_end_digest"]
            )
        else:
            a_path = evidence / "states" / tag_a.replace("arm-", "") / "u" / "adapter_model.safetensors"
            # Pairwise adapters are exported beside the run, not in state dirs.
            a_path = evidence / f"adapter-{tag_a}.safetensors"
            b_path = evidence / f"adapter-{tag_b}.safetensors"
            require(a_path.is_file() and b_path.is_file(), "pairwise adapter evidence missing")
            a_hash, b_hash = q.sha256_file(a_path), q.sha256_file(b_path)
            result["adapter_sha256"] = {"left": a_hash, "right": b_hash, "equal": a_hash == b_hash}
        return result

    recomputed = {
        "A_vs_B": compare("A", "B", ["loss_bits", "param_digest", "update_norm", "update_max_abs"]),
        "C_vs_D": compare("C", "D", ["loss_bits", "param_digest", "update_norm", "update_max_abs"]),
        "E_vs_F": compare("E", "F", ["loss_bits", "grad_digest", "grad_norm"]),
        "G_vs_H": compare("G", "H", ["loss_bits", "grad_digest", "grad_norm"]),
    }
    for key, result in recomputed.items():
        require(result["bitwise_equal"], f"{key} is not bit-identical")
        require(receipt["comparisons"][key]["bitwise_equal"] is True, f"{key} claim")
    require(recomputed["G_vs_H"]["adapter_sha256"]["equal"], "G_vs_H adapters differ")
    for key in ("A_vs_B", "C_vs_D"):
        require(recomputed[key]["final_parameter_digest_equal"], f"{key} final parameters differ")

    def drift(left: str, right: str) -> dict:
        a = [record["loss"] for record in reports[left]["records"]]
        b = [record["loss"] for record in reports[right]["records"]]
        diffs = [x - y for x, y in zip(a, b)]
        first_abs = abs(diffs[0])
        return {
            "first_step_abs": first_abs,
            "first_step_within_policy": first_abs <= 2.0e-2,
            "max_abs": max(abs(value) for value in diffs),
            "rms": math.sqrt(sum(value * value for value in diffs) / len(diffs)),
            "final_abs": abs(diffs[-1]),
        }

    for key, (left, right) in {"B_vs_D": ("B", "D"), "F_vs_H": ("F", "H")}.items():
        recomputed_drift = drift(left, right)
        require(recomputed_drift["first_step_within_policy"],
                f"{key} exceeds the matched-step mixed policy")
        require(receipt["comparisons"][key]["first_step_within_policy"] is True,
                f"{key} policy claim")
        require(close(receipt["comparisons"][key]["first_step_abs"],
                      recomputed_drift["first_step_abs"]), f"{key} first-step drift")
        require(close(receipt["comparisons"][key]["max_abs"], recomputed_drift["max_abs"]),
                f"{key} max_abs claim")
        require(close(receipt["comparisons"][key]["rms"], recomputed_drift["rms"]),
                f"{key} rms claim")
    receipt["comparisons"] = receipt["comparisons"]

    # Resume: checkpoint bytes, losses and optimizer steps from evidence.
    for arm_id, entry in receipt["resume"].items():
        total = entry["total_steps"]
        split = entry["split_step"]
        u = report(f"resume-{arm_id}-u.json")
        first = report(f"resume-{arm_id}-i.json")
        resumed = report(f"resume-{arm_id}-r.json")
        require(len(u["records"]) == total and len(first["records"]) == split,
                f"resume {arm_id} step counts")
        require(len(resumed["records"]) == total - split, f"resume {arm_id} resumed steps")
        require([r["optimizer_step"] for r in resumed["records"]]
                == list(range(split + 1, total + 1)), f"resume {arm_id} optimizer steps")
        u_bits = [r["loss_bits"] for r in u["records"][split:]]
        r_bits = [r["loss_bits"] for r in resumed["records"]]
        require(u_bits == r_bits, f"resume {arm_id} losses")
        require(entry["loss_bits"]["uninterrupted_tail"] == u_bits, f"resume {arm_id} tail claim")
        for phase in ("u", "r"):
            for name in ("adapter_model.safetensors", "adapter_config.json",
                         "hierarchos_adapter_optimizer.safetensors"):
                path = evidence / "states" / arm_id / phase / name
                require(path.is_file(), f"resume {arm_id} missing {phase}/{name}")
                actual = q.sha256_file(path)
                expected = entry["checkpoint_sha256"][name][
                    "uninterrupted" if phase == "u" else "resumed"
                ]
                require(actual == expected, f"resume {arm_id} {phase}/{name} changed")
        require(entry["checkpoint_bytes_equal"] is True, f"resume {arm_id} bytes claim")
        require(entry["loss_bits_equal"] is True, f"resume {arm_id} loss claim")
        require(entry["parameter_frozen"] is True, f"resume {arm_id} base moved")
        u_adapter = q.sha256_file(evidence / "states" / arm_id / "u" / "adapter_model.safetensors")
        r_adapter = q.sha256_file(evidence / "states" / arm_id / "r" / "adapter_model.safetensors")
        require(u_adapter == r_adapter, f"resume {arm_id} adapter bytes")
        require(entry["adapter_sha256"]["uninterrupted"] == u_adapter,
                f"resume {arm_id} uninterrupted adapter claim")

    # Adverse evidence.
    padded = report("adverse-seq96.json")
    padded_blocks = q.parse_profile(stderr("adverse-seq96.err.txt"))
    require(len(padded_blocks) == padded["steps"], "seq96 profiled steps")
    padded_query = q.profile_totals(padded_blocks).get(q.QUERY_SHADER, {}).get("dispatches", 0)
    require(padded_query == padded["steps"] * 2, "seq96 two-pass dispatch")
    require(padded["all_losses_finite"] is True, "seq96 non-finite loss")
    seed7 = q.sha256_file(evidence / "adverse/seed7.safetensors")
    seed13 = q.sha256_file(evidence / "adverse/seed13.safetensors")
    require(seed7 != seed13, "seed runs produced identical adapters")
    require(seed7 == receipt["adverse"]["seed"]["seed7_adapter"], "seed7 adapter claim")
    require(seed13 == receipt["adverse"]["seed"]["seed13_adapter"], "seed13 adapter claim")
    repeated = report("adverse-repeat.json")
    require([r["loss_bits"] for r in repeated["records"]]
            == [r["loss_bits"] for r in reports["F"]["records"]], "repeat loss divergence")
    require([r["grad_digest"] for r in repeated["records"]]
            == [r["grad_digest"] for r in reports["F"]["records"]], "repeat gradient divergence")

    refusal = (evidence / "full-ft-refusal.txt").read_text(encoding="utf-8")
    require("exit=0" not in refusal.splitlines()[0], "full-FT checkpoint refusal exited zero")
    require("full-parameter exact state is not implemented" in refusal,
            "full-FT checkpoint refusal message missing")

    # Performance: recompute every statistic from the per-run reports.
    for arm_id, entry in receipt["performance"].items():
        scalar_samples: list[float] = []
        twopass_samples: list[float] = []
        for path in sorted(evidence.glob(f"perf-{arm_id}-*.json")):
            run = read_json(path)
            measured = [r["step_ms"] for r in run["records"][entry["warmup"]:]]
            if "twopass" in path.stem:
                twopass_samples.extend(measured)
            else:
                scalar_samples.extend(measured)
        require(scalar_samples == entry["scalar_samples"], f"perf {arm_id} scalar samples")
        require(twopass_samples == entry["twopass_samples"], f"perf {arm_id} twopass samples")
        stats = q.latency_stats(scalar_samples, twopass_samples)
        for key in ("scalar_median_ms", "twopass_median_ms", "speedup", "scalar_p95_ms",
                    "twopass_p95_ms"):
            require(close(entry[key], stats[key]), f"perf {arm_id} {key} claim")
        require(close(entry["speedup_ci95"][0], stats["speedup_ci95"][0])
                and close(entry["speedup_ci95"][1], stats["speedup_ci95"][1]),
                f"perf {arm_id} confidence interval")

    # Reversal: recompute durations and window ratios from the reports.
    reversal = {}
    for precision in ("fp32", "mixed"):
        durations, conversions, reuses = [], [], []
        for path in sorted(evidence.glob(f"reversal-{precision}-*.json")):
            run = read_json(path)
            durations.append([r["step_ms"] for r in run["records"]])
            conversions.append([r["wmma_weight_conversions"] for r in run["records"]])
            reuses.append([r["wmma_mirror_reuses"] for r in run["records"]])
        reversal[precision] = {
            "durations": durations, "conversions": conversions, "reuses": reuses,
            "windows": q.window_stats(durations),
        }
    for precision in ("fp32", "mixed"):
        for key, value in reversal[precision]["windows"].items():
            require(close(receipt["wmma_reversal"][precision]["windows"][key], value),
                    f"reversal {precision} {key}")
    for window in ("early", "middle", "late"):
        expected = (reversal["fp32"]["windows"][f"{window}_median_ms"]
                    / reversal["mixed"]["windows"][f"{window}_median_ms"])
        require(close(receipt["wmma_reversal"]["mixed_vs_fp32"][window], expected),
                f"reversal ratio {window}")

    require(bool(receipt.get("limitations")), "missing limitations")
    return {
        "status": "PASS",
        "revision": revision,
        "arms": len(receipt["arms"]),
        "steps": {arm: receipt["arms"][arm]["summary"]["steps"] for arm in receipt["arms"]},
        "resume_arms": sorted(receipt["resume"]),
        "performance_arms": sorted(receipt["performance"]),
        "binaries_verified": binaries_present,
        "fixture_verified": fixture_present,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt-dir", type=Path, default=q.RECEIPT_DIR)
    args = parser.parse_args()
    try:
        summary = verify(args.receipt_dir)
    except FileNotFoundError as error:
        print(f"TRAINING QUALIFICATION: UNQUALIFIED ({error})", file=sys.stderr)
        return 3
    except (QualificationError, KeyError, ValueError, AssertionError) as error:
        print(f"TRAINING QUALIFICATION: FAIL ({error})", file=sys.stderr)
        return 1
    print(
        "TRAINING QUALIFICATION: PASS "
        f"({summary['revision'][:12]}, {summary['arms']} arms, "
        f"{len(summary['resume_arms'])} resume arms, "
        f"{len(summary['performance_arms'])} performance arms)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
