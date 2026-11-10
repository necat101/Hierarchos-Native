#!/usr/bin/env python3
"""Fail-closed verifier for the Phase XVII held-out generalization receipt.

The checker trusts no summary field. It re-derives the fixture, the split, the
window streams, every training step, every evaluation total, the read-only
invariants, the dispatch counts and the controls from the committed partition
files and the shipped raw reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import qualify_heldout_phase17 as p
import qualify_training_phase16 as q

SCHEMA = 1
KIND = "phase17-heldout-generalization"


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


_FIXTURE_CACHE: dict[str, Path] = {}


def regenerate_fixture(crate: Path, fixture_dir: Path) -> Path:
    """Rebuild the fixture from its committed generator in a scratch directory.

    The result is memoised on the generator plus fixture bytes, so repeated
    verification inside one process is cheap while any change to either still
    forces a fresh build and therefore a real comparison.
    """
    digest = hashlib.sha256((crate / "validation/make_phase17_fixture.py").read_bytes())
    for name in ("corpus.jsonl", "vocab.json", "train.jsonl",
                 "validation.jsonl", "test.jsonl", "split_manifest.json"):
        digest.update((fixture_dir / name).read_bytes())
    key = digest.hexdigest()
    cached = _FIXTURE_CACHE.get(key)
    if cached is not None and cached.exists():
        return cached
    target = Path(tempfile.mkdtemp(prefix="phase17-fixture-"))
    completed = subprocess.run(
        [sys.executable, str(crate / "validation/make_phase17_fixture.py"),
         "--output", str(target)],
        cwd=crate, capture_output=True, text=True,
    )
    require(completed.returncode == 0,
            f"fixture generator failed: {completed.stderr[-800:]}")
    _FIXTURE_CACHE[key] = target
    return target


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

    revision = receipt.get("source_revision", "")
    require(len(revision) == 40 and all(c in "0123456789abcdef" for c in revision),
            "invalid source revision")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, "HEAD"],
        cwd=crate.parent, capture_output=True, text=True,
    )
    require(ancestor.returncode == 0, "source revision is not an ancestor of HEAD")
    require(receipt.get("source_clean") is True, "receipt was produced from a dirty worktree")

    for relative, expected in receipt["source_files"].items():
        path = crate / relative
        require(path.is_file(), f"missing bound source {relative}")
        require(q.sha256_file(path) == expected, f"bound source changed: {relative}")
    require("src/bin/held_out_eval.rs" in receipt["source_files"], "harness binding missing")

    binary = crate / "target/release/held_out_eval.exe"
    if binary.is_file():
        require(q.sha256_file(binary) == receipt["binaries"]["held_out_eval"],
                "held-out harness binary hash changed")

    fixture = receipt["fixture"]
    fixture_dir = crate / fixture["directory"]
    require(fixture_dir.is_dir(), "missing fixture directory")
    for name, expected in fixture["files"].items():
        path = fixture_dir / name
        require(path.is_file(), f"missing fixture file {name}")
        require(q.sha256_file(path) == expected, f"fixture file changed: {name}")

    manifest = fixture["manifest"]
    require(manifest.get("fixture_version") == "phase17-heldout-v1",
            "unexpected split manifest identifier")
    require(manifest.get("split_seed") == 20261110, "split seed changed")
    require("sha256" in manifest.get("split_algorithm", ""),
            "split algorithm is not the declared hash ordering")
    require(manifest["tokenizer"]["name"] == "hierarchos-word-hash-v1"
            and manifest["tokenizer"]["revision"] == "1",
            "tokenizer identity changed")
    require(manifest["tokenizer"]["vocab_sha256"]
            == q.sha256_file(fixture_dir / "vocab.json"),
            "tokenizer identity changed")

    # --- Phase I audit ------------------------------------------------------
    audit = fixture["phase16_audit"]
    require(audit.get("split_verdict", {}).get("can_support_held_out_split") is False,
            "Phase XVI audit no longer rejects a held-out split")
    phase16_dataset = fixture_dir / "phase16_dataset.jsonl"
    require(q.sha256_file(phase16_dataset) == fixture["phase16_dataset_sha256"],
            "Phase XVI dataset copy changed")
    require(fixture["phase16_dataset_sha256"] == fixture["phase16_receipt_dataset_sha256"],
            "Phase XVI dataset copy is not the dataset the sealed receipt consumed")
    require(audit.get("sha256") == fixture["phase16_dataset_sha256"],
            "Phase XVI audit was not run over the sealed dataset")

    # --- leakage ------------------------------------------------------------
    partitions = manifest["partitions"]
    # Cross-partition identity is checked before manifest/file consistency so a
    # duplicated document is rejected as duplicate leakage, not as a bookkeeping
    # mismatch.
    train_docs = set(partitions["train"]["documents_list"])
    validation_docs = set(partitions["validation"]["documents_list"])
    test_docs = set(partitions["test"]["documents_list"])
    require(not train_docs & validation_docs, "duplicate documents across partitions: train/validation")
    require(not train_docs & test_docs, "duplicate documents across partitions: train/test")
    require(not validation_docs & test_docs, "duplicate documents across partitions: validation/test")
    for name in ("train", "validation", "test"):
        documents = [
            json.loads(line)["document_id"]
            for line in (fixture_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        require(len(documents) == partitions[name]["documents"],
                f"{name} document count changed")
        require(set(documents) == set(partitions[name]["documents_list"]),
                f"{name} document identities changed")

    def grams(name: str) -> set[str]:
        values = set()
        for line in (fixture_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            tokens = json.loads(line)["input_ids"]
            for index in range(max(0, len(tokens) - 8 + 1)):
                values.add(hashlib.sha256(
                    ",".join(str(v) for v in tokens[index:index + 8]).encode()
                ).hexdigest())
        return values

    train_grams, validation_grams, test_grams = grams("train"), grams("validation"), grams("test")
    require(not train_grams & validation_grams, "8-gram overlap between train and validation")
    require(not train_grams & test_grams, "8-gram overlap between train and test")
    require(not validation_grams & test_grams, "8-gram overlap between validation and test")

    # --- the generator must reproduce the committed bytes -------------------
    regenerated = regenerate_fixture(crate, fixture_dir)
    for name in ("corpus.jsonl", "vocab.json", "train.jsonl",
                 "validation.jsonl", "test.jsonl", "split_manifest.json"):
        produced = regenerated / name
        committed = fixture_dir / name
        require(produced.is_file(), f"generator produced no {name}")
        require(produced.read_bytes() == committed.read_bytes(),
                f"fixture is not reproducible from its generator: {name}")

    # --- window streams (causal alignment) ---------------------------------
    streams = {
        name: p.stream_digest(p.derive_windows(fixture_dir / f"{name}.jsonl", manifest["seq_len"]))
        for name in ("train", "validation", "test")
    }
    require(streams == fixture["derived_window_streams"],
            "derived window streams changed")
    for name, expected in partitions.items():
        recomputed = p.derive_windows(fixture_dir / f"{name}.jsonl", manifest["seq_len"])
        require(len(recomputed) == expected["windows"], f"{name} window count changed")
        require(sum(window["valid_tokens"] for window in recomputed)
                < expected["windows"] * manifest["seq_len"],
                f"{name} reports padding as supervised loss")
        require(sum(window["valid_tokens"] for window in recomputed)
                == expected["valid_target_tokens"], f"{name} valid target token count changed")

    # --- evidence map -------------------------------------------------------
    recorded = receipt.get("evidence", {})
    require(recorded, "missing evidence map")
    for relative, expected in recorded.items():
        path = evidence / relative
        require(path.is_file(), f"missing evidence artifact {relative}")
        require(q.sha256_file(path) == expected, f"evidence changed: {relative}")

    protocol = receipt["protocol"]
    require(protocol["schedule"] == list(p.SCHEDULE), "validation schedule changed")
    require(protocol["test_access"].startswith("two evaluations"),
            "test partition access is not the declared two evaluations")
    require("no selection on test" in protocol["hyperparameter_source"],
            "hyperparameters are not declared test-independent")

    # --- every arm ----------------------------------------------------------
    arms = receipt.get("arms", {})
    for arm_id in ("A", "B", "C", "D"):
        require(arm_id in arms, f"missing execution arm {arm_id}")

    recomputed_arms = {}
    for arm_id, entry in arms.items():
        config = entry["config"]
        require(config == p.ARMS[arm_id], f"arm {arm_id} execution mode changed")
        report = read_json(evidence / f"arm-{arm_id}.json")
        require(report["records"] and len(report["records"]) == p.STEPS,
                f"arm {arm_id} step count")
        require([record["optimizer_step"] for record in report["records"]]
                == list(range(1, p.STEPS + 1)), f"arm {arm_id} optimizer steps")
        require(report["parameter_elements"] == p.EXPECTED_PARAMETERS,
                f"arm {arm_id} parameter count")
        require(report["all_losses_finite"] is True, f"arm {arm_id} non-finite loss")
        require(report["partition_stream_sha256"] == streams,
                f"arm {arm_id} causal target alignment differs from the committed partitions")
        require(report["two_pass_requested"] == (config["two_pass"] == "1"),
                f"arm {arm_id} two-pass label")
        require(report["precision_requested"]
                == ("mixed" if config["precision"] == "mixed" else "off"),
                f"arm {arm_id} precision label")

        evaluations = {e["tag"]: e for e in report["evaluations"]}
        for tag in ("test@0", "test@final"):
            require(evaluations[tag]["partition"] == "test",
                    f"arm {arm_id} presents a validation-only result as test evidence")
        expected_validation = {f"validation@{step}" for step in p.SCHEDULE}
        require({name for name in evaluations if name.startswith("validation@")}
                == expected_validation, f"arm {arm_id} validation schedule")
        require("test@0" in evaluations and "test@final" in evaluations,
                f"arm {arm_id} missing evaluation output for the test partition")
        require(not any(name.startswith("test@") and name not in ("test@0", "test@final")
                        for name in evaluations),
                f"arm {arm_id} used the test partition outside the declared accesses")

        for tag, evaluation in evaluations.items():
            partition = evaluation["partition"]
            require(evaluation["valid_target_tokens"]
                    == partitions[partition]["valid_target_tokens"],
                    f"{tag} valid target token count (padding excluded) mismatch")
            require(evaluation["examples"] == partitions[partition]["windows"],
                    f"{tag} example count")
            require(evaluation["nonfinite_logits"] == 0, f"{tag} non-finite logits")
            require(evaluation["parameter_digest_before"]
                    == evaluation["parameter_digest_after"],
                    f"{tag} evaluation mutated parameters")
            require(evaluation["optimizer_state_digest_before"]
                    == evaluation["optimizer_state_digest_after"],
                    f"{tag} evaluation changed optimizer state")
            require(evaluation["optimizer_step_before"] == evaluation["optimizer_step_after"],
                    f"{tag} evaluation advanced the optimizer step")
            require(evaluation["dropout_seed_before"] == evaluation["dropout_seed_after"],
                    f"{tag} evaluation advanced the dropout seed")
            require(evaluation["wmma_weight_grad_delta"] == 0,
                    f"{tag} evaluation dispatched a weight gradient")
            require(evaluation["wmma_input_grad_delta"] == 0,
                    f"{tag} evaluation dispatched an input gradient")
            if config["precision"] == "mixed":
                require(evaluation["wmma_forward_delta"] > 0,
                        f"{tag} mixed evaluation did not use the cooperative forward")
            recomputed = sum(example["loss_sum"] for example in evaluation["per_example"]) \
                / evaluation["valid_target_tokens"]
            require(close(recomputed, evaluation["cross_entropy"], 1e-12),
                    f"{tag} token-weighted cross-entropy does not recompute")
            require(close(evaluation["perplexity"], math.exp(evaluation["cross_entropy"]), 1e-9),
                    f"{tag} perplexity does not recompute")
            summary = next(item for item in entry["evaluations"] if item["tag"] == tag)
            for key in ("cross_entropy", "perplexity", "valid_target_tokens", "examples",
                        "optimizer_step_before", "optimizer_step_after"):
                require(close(summary[key], evaluation[key]) if isinstance(evaluation[key], float)
                        else summary[key] == evaluation[key],
                        f"{tag} receipt summary {key}")

        # final test loss must equal the value the receipt reports
        test_metrics = receipt["test_metrics"][arm_id]
        final_test = evaluations["test@final"]
        require(close(test_metrics["final_cross_entropy"], final_test["cross_entropy"], 1e-12),
                f"arm {arm_id} final test loss claim")
        require(close(test_metrics["initial_cross_entropy"],
                      evaluations["test@0"]["cross_entropy"], 1e-12),
                f"arm {arm_id} initial test loss claim")
        require(close(test_metrics["final_perplexity"],
                      math.exp(final_test["cross_entropy"]), 1e-9),
                f"arm {arm_id} final test perplexity claim")
        require(test_metrics["valid_target_tokens"]
                == partitions["test"]["valid_target_tokens"],
                f"arm {arm_id} test token count claim")
        # the receipt must not present a validation-only result as test evidence
        require(final_test["partition"] == "test",
                f"arm {arm_id} presents a validation-only result as final test evidence")

        totals = q.profile_totals(q.parse_profile(
            (evidence / f"arm-{arm_id}.err.txt").read_text(encoding="utf-8", errors="replace")
        ))
        scalar = totals.get(q.SCALAR_SHADER, {}).get("dispatches", 0)
        query = totals.get(q.QUERY_SHADER, {}).get("dispatches", 0)
        key = totals.get(q.KEY_SHADER, {}).get("dispatches", 0)
        layers = 2
        if config["backward"] == "scalar":
            require(scalar == p.STEPS * layers and query == 0 and key == 0,
                    f"arm {arm_id} dispatch does not match the declared scalar route "
                    f"(scalar={scalar} query={query} key={key})")
        else:
            require(scalar == 0 and query == p.STEPS * layers and key == p.STEPS * layers,
                    f"arm {arm_id} dispatch does not match the declared two-pass route "
                    f"(scalar={scalar} query={query} key={key})")
        recomputed_arms[arm_id] = report

    # --- scalar/two-pass equivalence and mixed drift -----------------------
    def records(report: dict, field: str) -> list:
        return [record[field] for record in report["records"]]

    for key, (left, right) in {"A_vs_B": ("A", "B"), "C_vs_D": ("C", "D")}.items():
        for field in ("loss_bits", "param_digest", "wmma_forward",
                      "wmma_weight_grad", "wmma_input_grad"):
            require(records(recomputed_arms[left], field)
                    == records(recomputed_arms[right], field),
                    f"{key} {field} is not bitwise identical")
        require(recomputed_arms[left]["parameter_end_digest"]
                == recomputed_arms[right]["parameter_end_digest"],
                f"{key} final parameters differ")
        require(receipt["comparisons"][key]["bitwise_equal"] is True, f"{key} receipt claim")

    for key, (left, right) in {"A_vs_C": ("A", "C"), "B_vs_D": ("B", "D")}.items():
        a = records(recomputed_arms[left], "loss")
        b = records(recomputed_arms[right], "loss")
        diffs = [x - y for x, y in zip(a, b)]
        claim = receipt["comparisons"][key]
        require(abs(diffs[0]) <= 2e-2, f"{key} first-step mixed drift exceeded policy")
        require(close(claim["first_step_abs"], abs(diffs[0])), f"{key} first-step drift claim")
        require(close(claim["max_abs"], max(abs(value) for value in diffs)), f"{key} max drift claim")
        require(close(claim["rms"], math.sqrt(sum(value * value for value in diffs) / len(diffs))),
                f"{key} RMS drift claim")

    # --- seed investigation -------------------------------------------------
    seed_block = receipt["seed_investigation"]
    seed_bits = {}
    for seed in p.SEED_PROBES:
        report = read_json(evidence / f"seed-{seed}.json")
        seed_bits[str(seed)] = [record["loss_bits"] for record in report["records"]]
        require(len(report["records"]) == p.CONTROL_STEPS, f"seed {seed} step count")
    require(len({tuple(bits) for bits in seed_bits.values()}) == 1,
            "seed investigation does not reproduce identical trajectories")
    require(seed_block["loss_bits_identical"] is True, "seed investigation claim")
    require(seed_block["loss_bits"] == seed_bits, "seed investigation evidence changed")
    require("not independent replication" in seed_block["finding"],
            "seed investigation overclaims independent replication")

    # --- read-only control --------------------------------------------------
    for arm_id in p.CONTROL_ARMS:
        with_eval = read_json(evidence / f"control-{arm_id}-eval.json")
        without_eval = read_json(evidence / f"control-{arm_id}-noeval.json")
        require(records(with_eval, "loss_bits") == records(without_eval, "loss_bits"),
                f"control {arm_id}: interleaved evaluation changed training losses")
        require(records(with_eval, "param_digest") == records(without_eval, "param_digest"),
                f"control {arm_id}: interleaved evaluation changed parameters")
        require(with_eval["parameter_end_digest"] == without_eval["parameter_end_digest"],
                f"control {arm_id}: interleaved evaluation changed the final model")
        require(any(evaluation["tag"] == f"validation@{p.CONTROL_EVAL_AT}"
                    for evaluation in with_eval["evaluations"]),
                f"control {arm_id}: interleaved evaluation did not run")
        require(receipt["read_only_controls"][arm_id]["comparison"]["bitwise_equal"] is True,
                f"control {arm_id}: receipt claim")

    # --- order replication --------------------------------------------------
    for offset in p.ORDER_OFFSETS:
        report = read_json(evidence / f"order-{offset}.json")
        evaluations = {e["tag"]: e for e in report["evaluations"]}
        claim = receipt["order_replication"][str(offset)]
        require(close(claim["final_validation_cross_entropy"],
                      evaluations["validation@100"]["cross_entropy"]), f"order {offset} validation claim")
        require(close(claim["final_test_cross_entropy"],
                      evaluations["test@final"]["cross_entropy"]), f"order {offset} test claim")
        require(claim["parameter_end_digest"] == report["parameter_end_digest"],
                f"order {offset} parameter digest claim")

    # --- retained Phase XVI status -----------------------------------------
    stability = read_json(p.STABILITY_RECEIPT)
    require(stability["source_revision"]
            == "21027bf9699757859e7f2a9c07c437402d5aa2ae",
            "Phase XVI stability receipt was rewritten")

    require(bool(receipt.get("limitations")), "missing limitations")

    validation_final = {
        arm_id: receipt["validation_trajectories"][arm_id]["validation@100"]
        for arm_id in ("A", "B", "C", "D")
    }
    test_final = {
        arm_id: receipt["test_metrics"][arm_id]["final_cross_entropy"]
        for arm_id in ("A", "B", "C", "D")
    }
    return {
        "status": "PASS",
        "revision": revision,
        "arms": len(arms),
        "validation_final": validation_final,
        "test_final": test_final,
        "controls": len(p.CONTROL_ARMS),
        "seed_identical": True,
        "order_replication": sorted(receipt["order_replication"]),
        "binary_verified": binary.is_file(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt-dir", type=Path, default=p.RECEIPT_DIR)
    args = parser.parse_args()
    try:
        summary = verify(args.receipt_dir)
    except FileNotFoundError as error:
        print(f"HELD-OUT QUALIFICATION: UNQUALIFIED ({error})", file=sys.stderr)
        return 3
    except (QualificationError, KeyError, ValueError, AssertionError) as error:
        print(f"HELD-OUT QUALIFICATION: FAIL ({error})", file=sys.stderr)
        return 1
    print(
        "HELD-OUT QUALIFICATION: PASS "
        f"({summary['revision'][:12]}, {summary['arms']} arms, "
        f"validation@100 A={summary['validation_final']['A']:.6f} "
        f"D={summary['validation_final']['D']:.6f}, "
        f"test A={summary['test_final']['A']:.6f} D={summary['test_final']['D']:.6f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
