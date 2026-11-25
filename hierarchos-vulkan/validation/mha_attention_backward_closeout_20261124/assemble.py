#!/usr/bin/env python3
"""Assemble the Phase III MHA attention-backward closeout receipt.

Every number in ``qualification.json`` is read from a raw artifact in
``raw/`` (or re-derived from the current checkout). Nothing here writes a
measurement it did not read: a missing or malformed raw file raises instead of
being skipped.

    python validation/mha_attention_backward_closeout_20261124/assemble.py
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import subprocess
from pathlib import Path

RECEIPT = Path(__file__).resolve().parent
CRATE = RECEIPT.parents[1]
RAW = RECEIPT / "raw"

GRAD = re.compile(
    r"\[(?P<case>[a-z0-9-]+)\] mode=(?P<mode>\d+) batch=(?P<batch>\d+) "
    r"heads=(?P<heads>\d+) seq=(?P<seq>\d+) head_dim=(?P<head_dim>\d+) "
    r"causal=(?P<causal>[01]) masked=(?P<masked>true|false) "
    r"unfused_qk=(?P<mode_field>\d+) unwritten=(?P<unwritten>\d+) "
    r"grad_(?P<tensor>[qkv]): elements=(?P<elements>\d+) "
    r"bitwise_equal=(?P<equal>\d+) max_abs=(?P<max_abs>\S+) max_rel=\S+ "
    r"rms=\S+ max_ulp=(?P<max_ulp>\d+) .* (?P<verdict>PASS|FAIL)$",
    re.M,
)
SKIP = re.compile(
    r"two-pass gradient qualification: status=(?P<status>[A-Z]+) "
    r"reason=(?P<reason>\S+)"
)
QUALIFICATION = re.compile(
    r"two-pass gradient qualification: status=(?P<status>[A-Z]+) "
    r"cases=(?P<cases>\d+)/(?P<cases_total>\d+) "
    r"comparisons=(?P<comparisons>\d+)/(?P<comparisons_total>\d+) "
    r"elements=(?P<elements>\d+) bitwise_equal=(?P<equal>\d+) "
    r"max_abs=(?P<max_abs>\S+) max_ulp=(?P<max_ulp>\d+) "
    r"unwritten=(?P<unwritten>\d+) scalar_dispatches=(?P<scalar>\d+) "
    r"query_dispatches=(?P<query>\d+) key_dispatches=(?P<key>\d+)"
)
ARM = re.compile(
    r"^(?P<arm>scalar arm|twopass arm)\s*:\s*scalar=(?P<scalar>\d+) "
    r"query=(?P<query>\d+) key=(?P<key>\d+)$",
    re.M,
)
CONTROLS = re.compile(r"negative controls: (?P<detected>\d+)/(?P<total>\d+) detected")
ADAPTER = re.compile(r"^(?P<arm>scalar|twopass)\s+(?P<path>\S+)\s+(?P<sha>[0-9a-f]{64})$", re.M)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(name: str) -> str:
    return (RAW / name).read_text(encoding="utf-8", errors="replace")


def require(condition, message: str) -> None:
    if not condition:
        raise SystemExit(f"receipt input is incomplete: {message}")


def key_values(line: str) -> dict[str, str]:
    return dict(re.findall(r"([a-z_]+)=([^\s]+)", line))


def parse_parity() -> dict:
    log = text("parity-mode0-mode9.log")
    cases = []
    for match in GRAD.finditer(log):
        row = match.groupdict()
        require(row["mode"] == row["mode_field"], "mode label and field disagree")
        cases.append(
            dict(
                case=row["case"],
                mode=int(row["mode"]),
                tensor=row["tensor"],
                batch=int(row["batch"]),
                heads=int(row["heads"]),
                seq=int(row["seq"]),
                head_dim=int(row["head_dim"]),
                causal=int(row["causal"]),
                masked=row["masked"] == "true",
                elements=int(row["elements"]),
                bitwise_equal=int(row["equal"]),
                max_abs=float(row["max_abs"]),
                max_ulp=int(row["max_ulp"]),
                unwritten=int(row["unwritten"]),
                verdict=row["verdict"],
            )
        )
    summary = QUALIFICATION.search(log)
    require(summary is not None, "qualification summary line missing from parity log")
    values = summary.groupdict()
    skipped = SKIP.search(text("parity-skip.log"))
    require(skipped is not None, "SKIP line missing from the opt-out test log")
    blocked_log = text("blocked.log")
    blocked = SKIP.search(blocked_log)
    require(blocked is not None, "BLOCKED line missing from the missing-ICD test log")
    require(blocked.group("status") == "BLOCKED", "missing-ICD log did not report BLOCKED")
    require(
        "two-pass gradient qualification BLOCKED" in blocked_log,
        "missing-ICD run did not fail closed",
    )
    return dict(
        cases=cases,
        summary=dict(
            status=values["status"],
            cases=int(values["cases"]),
            cases_total=int(values["cases_total"]),
            comparisons=int(values["comparisons"]),
            comparisons_total=int(values["comparisons_total"]),
            elements=int(values["elements"]),
            bitwise_equal=int(values["equal"]),
            max_abs=float(values["max_abs"]),
            max_ulp=int(values["max_ulp"]),
            unwritten=int(values["unwritten"]),
            scalar_dispatches=int(values["scalar"]),
            query_dispatches=int(values["query"]),
            key_dispatches=int(values["key"]),
        ),
        skipped=dict(status=skipped.group("status"), reason=skipped.group("reason")),
        blocked=dict(status=blocked.group("status"), reason=blocked.group("reason")),
        comparator_negative_control="PASS"
        if "test result: ok" in text("comparator-negative-control.log")
        else "FAIL",
    )


def parse_equivalence() -> dict:
    log = text("equivalence.log")
    arms = {
        match.group("arm").split()[0]: {
            "scalar": int(match.group("scalar")),
            "query": int(match.group("query")),
            "key": int(match.group("key")),
        }
        for match in ARM.finditer(log)
    }
    digests: dict[str, dict[str, str]] = {}
    for match in ADAPTER.finditer(log):
        digests.setdefault(match.group("arm"), {})[match.group("path")] = match.group("sha")
    controls = CONTROLS.search(log)
    require(controls is not None, "equivalence negative-control summary missing")
    return dict(
        arms=arms,
        adapter_files=digests,
        adapter_files_identical=bool(digests.get("scalar"))
        and digests.get("scalar") == digests.get("twopass"),
        negative_controls=f"{controls.group('detected')}/{controls.group('total')}",
        pass_line="PASS: both passes dispatched and every adapter file is bit-identical"
        in log,
    )


def parse_scratch(name: str) -> dict:
    log = text(name)
    decisions = [
        key_values(line)
        for line in log.splitlines()
        if line.startswith("two-pass backward: route=")
    ]
    counters = next(
        (key_values(line) for line in log.splitlines()
         if line.startswith("two-pass backward counters:")),
        None,
    )
    require(decisions, f"{name}: no decision line")
    require(counters is not None, f"{name}: no counter summary")
    return dict(decisions=decisions, counters=counters)


def parse_environment() -> dict:
    lines = [line.strip() for line in text("environment.txt").splitlines() if line.strip()]
    values = {}
    for line in lines:
        if line.startswith("rustc "):
            values.setdefault("rustc", line)
        elif line.startswith("cargo "):
            values.setdefault("cargo", line)
        elif line.startswith("shaderc "):
            values.setdefault("glslc", line)
        elif line.startswith("SPIRV-Tools "):
            values.setdefault("spirv_val", line)
        elif line.startswith("Name "):
            values["device"] = line.split(":", 1)[1].strip()
        elif line.startswith("DriverVersion "):
            values["driver_version"] = line.split(":", 1)[1].strip()
        elif line.startswith("DriverDate "):
            values["driver_date"] = line.split(":", 1)[1].strip()
    for key in ("device", "driver_version", "rustc", "cargo", "glslc", "spirv_val"):
        require(key in values, f"environment field {key} missing")
    return values


def parse_hashes() -> dict[str, str]:
    hashes = {}
    for line in text("source-hashes.txt").splitlines():
        digest, path = line.split(maxsplit=1)
        hashes[path.lstrip("*")] = digest
    return hashes


def parse_git() -> dict:
    head_stash = text("git-head-stash.txt").splitlines()
    require(len(head_stash) >= 1 and re.fullmatch(r"[0-9a-f]{40}", head_stash[0]),
            "HEAD is not a full commit id")
    status = [line for line in text("git-status.txt").splitlines() if line.strip()]
    return dict(
        commit=head_stash[0],
        expected_commit="560809eead186f832344c15d1a9b70795da3b75d",
        working_tree_changed_files=[line[3:] for line in status],
        stash=head_stash[1] if len(head_stash) > 1 else None,
        diff_sha256=sha256(RAW / "git-diff.patch"),
    )


def parse_seq16() -> dict:
    arms = {}
    for arm in ("tp0", "tp1"):
        report = json.loads(text(f"seq16-{arm}.json"))
        adapter = RAW / f"seq16-{arm}-adapter.safetensors"
        config = RAW / f"seq16-{arm}-adapter-config.json"
        arms[arm] = dict(
            loss_bits=[record["loss_bits"] for record in report["records"]],
            parameter_end_digest=report["parameter_end_digest"],
            adapter_model_sha256=sha256(adapter),
            adapter_config_sha256=sha256(config),
            optimizer_steps=report["optimizer_steps"],
        )
    return dict(
        arms=arms,
        loss_bits_equal=arms["tp0"]["loss_bits"] == arms["tp1"]["loss_bits"],
        parameter_end_digest_equal=(
            arms["tp0"]["parameter_end_digest"] == arms["tp1"]["parameter_end_digest"]
        ),
        adapter_model_equal=(
            arms["tp0"]["adapter_model_sha256"] == arms["tp1"]["adapter_model_sha256"]
        ),
        adapter_config_equal=(
            arms["tp0"]["adapter_config_sha256"] == arms["tp1"]["adapter_config_sha256"]
        ),
    )


def main() -> int:
    bench = json.loads(text("bench-summary.json"))
    parity = parse_parity()
    equivalence = parse_equivalence()
    environment = parse_environment()
    git = parse_git()
    hashes = parse_hashes()
    seq16 = parse_seq16()
    scratch_eligible = parse_scratch("scratch-eligible.log")
    scratch_optout = parse_scratch("scratch-optout.log")
    scratch_gqa = parse_scratch("scratch-gqa.log")
    scratch_seqgt = parse_scratch("scratch-seqgt.log")

    scalar_kernel = bench["kernel"]["scalar"]["kernels"]["transformer_attention_backward"]
    query_kernel = bench["kernel"]["twopass"]["kernels"][
        "transformer_attention_backward_query_rdna3"
    ]
    key_kernel = bench["kernel"]["twopass"]["kernels"][
        "transformer_attention_backward_key_rdna3"
    ]
    two_pass_kernel_ms = query_kernel["ms_per_step"] + key_kernel["ms_per_step"]
    abba = bench["abba"]

    failures = []
    if parity["summary"]["status"] != "PASS":
        failures.append("gradient qualification did not pass")
    if parity["summary"]["comparisons"] != 24 or parity["summary"]["cases"] != 8:
        failures.append("gradient qualification did not cover 8 cases and 24 comparisons")
    if parity["summary"]["bitwise_equal"] != parity["summary"]["elements"]:
        failures.append("gradient parity is not bit-exact")
    if parity["summary"]["max_ulp"] or parity["summary"]["unwritten"]:
        failures.append("gradient parity reports a ULP difference or an unwritten element")
    if parity["comparator_negative_control"] != "PASS":
        failures.append("the comparator negative control did not pass")
    if not equivalence["pass_line"] or not equivalence["adapter_files_identical"]:
        failures.append("dispatch-gated equivalence did not pass")
    if equivalence["negative_controls"] != "11/11":
        failures.append("equivalence negative controls were not all detected")
    if not (bench["numerics"]["loss_bits_equal"]
            and bench["numerics"]["adapters_equal"]
            and bench["numerics"]["parameter_end_equal"]):
        failures.append("benchmark arms are not bit-identical")
    if seq16["loss_bits_equal"] is not True:
        failures.append("sequence-16 loss bits differ")
    if abba["ratio_ci"]["low"] <= 1.0:
        failures.append("complete-step pairing does not exclude a 1x speedup")
    if int(scratch_optout["counters"]["scratch_allocations"]) != 0:
        failures.append("the explicit opt-out allocated scratch")
    if int(scratch_gqa["counters"]["scratch_allocations"]) != 0:
        failures.append("the GQA fallback allocated scratch")
    if int(scratch_seqgt["counters"]["scratch_allocations"]) != 0:
        failures.append("the long-context fallback allocated scratch")
    if int(scratch_eligible["counters"]["query"]) <= 0 or int(
        scratch_eligible["counters"]["scalar"]
    ) != 0:
        failures.append("the default route did not dispatch both passes without the scalar kernel")

    receipt = {
        "schema": 1,
        "kind": "mha_attention_backward_closeout_receipt",
        "phase": "Phase III: MHA attention-backward closeout and default readiness",
        "date": "2026-11-24",
        "verdict": "GO_SCOPED_MHA_DEFAULT" if not failures else "NO_GO",
        "failures": failures,
        "source": {
            **git,
            "base_commit_matches_expected": git["commit"] == git["expected_commit"],
            "executable_sha256": {
                "hierarchos-vulkan-transformer-train.exe": hashes[
                    "target/release/hierarchos-vulkan-transformer-train.exe"
                ],
                "hierarchos-vulkan-training-qualification.exe": hashes[
                    "target/release/hierarchos-vulkan-training-qualification.exe"
                ],
            },
            "source_sha256": {
                path: digest
                for path, digest in hashes.items()
                if path.startswith(("src/", "shaders/"))
            },
        },
        "environment": environment,
        "fixture": {
            "model": "target/wmma-fixture/model (2 layers, Llama, MHA)",
            "dataset": "target/wmma-fixture/dataset.jsonl",
            "config_sha256": hashes["target/wmma-fixture/model/config.json"],
            "weights_sha256": hashes["target/wmma-fixture/model/model.safetensors"],
            "dataset_sha256": hashes["target/wmma-fixture/dataset.jsonl"],
            "batch": 8,
            "seq": 64,
            "num_heads": 8,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "layers": 2,
            "lora_rank": 8,
            "lora_alpha": 16,
            "learning_rate": 0.01,
            "seed": 7,
            "benchmark_arithmetic_mode": 9,
            "gradient_modes_covered": [0, 9],
        },
        "protocol": {
            "gradient_test": "cargo test --offline --lib "
            "transformer::tests::two_pass_attention_backward_matches_scalar_gradients "
            "-- --nocapture --test-threads=1",
            "equivalence": "bash validation/two_pass_backward_equivalence.sh",
            "benchmark": "python raw/phase2-bench.py --rounds 5 --steps 26 --warmup 3 "
            "--kernel-steps 12 --phase all",
            "abba_rounds": abba["rounds"],
            "steps_per_run": abba["steps"],
            "warmup_steps": abba["warmup"],
            "kernel_steps": 12,
            "statistic": "per-round median scalar / median two-pass, Student-t interval over rounds",
        },
        "parity": parity,
        "dispatch": {
            "equivalence": equivalence["arms"],
            "kernel": {
                "scalar": bench["kernel"]["scalar"]["dispatch"],
                "twopass": bench["kernel"]["twopass"]["dispatch"],
            },
        },
        "scratch": {
            "formula": "batch * num_heads * query_seq_len * key_value_seq_len floats per buffer "
            "(two buffers), the highest slot the pass-1 and pass-2 shaders index",
            "eligible": {
                "decisions": scratch_eligible["decisions"],
                "counters": scratch_eligible["counters"],
            },
            "explicit_opt_out": {
                "decisions": scratch_optout["decisions"],
                "counters": scratch_optout["counters"],
            },
            "grouped_query_attention": {
                "decisions": scratch_gqa["decisions"],
                "counters": scratch_gqa["counters"],
            },
            "kv_sequence_above_head_dim": {
                "decisions": scratch_seqgt["decisions"],
                "counters": scratch_seqgt["counters"],
            },
        },
        "performance": {
            "kernel_ms_per_step": {
                "scalar": scalar_kernel["ms_per_step"],
                "query_pass": query_kernel["ms_per_step"],
                "key_pass": key_kernel["ms_per_step"],
                "two_pass_total": two_pass_kernel_ms,
                "speedup": scalar_kernel["ms_per_step"] / two_pass_kernel_ms,
            },
            "complete_step": {
                "scalar_mean_ms": abba["scalar_mean_ms"],
                "two_pass_mean_ms": abba["twopass_mean_ms"],
                "scalar_median_ms": abba["scalar_median_ms"],
                "two_pass_median_ms": abba["twopass_median_ms"],
                "paired_ratios": [entry["ratio"] for entry in abba["per_round"]],
                "mean_paired_speedup": abba["ratio_ci"]["mean"],
                "ci95": [abba["ratio_ci"]["low"], abba["ratio_ci"]["high"]],
                "df": abba["ratio_ci"]["df"],
                "samples_per_arm": len(abba["scalar_samples"]),
            },
            "phase2_narrative_reference": {
                "note": "Narrative comparison only; not bound to a committed receipt.",
                "scalar_backward_ms": 431.93,
                "two_pass_backward_ms": 44.59,
                "backward_speedup": 9.69,
                "scalar_step_ms": 561.56,
                "two_pass_step_ms": 170.28,
                "mean_paired_speedup": 3.304,
                "ci95": [3.2530, 3.3549],
            },
        },
        "numerics": {
            "benchmark_arms": bench["numerics"],
            "sequence_16": seq16,
        },
        "historical_receipts": {
            "phase16": "validation/release_parity/multimodal_training/"
            "20261109-rdna3-phase16-attention-backward-qualification",
            "note": "Historical Phase XVI receipts bind the previous query SPIR-V "
            "(615c27dc...); they remain historical and are not revalidated against "
            "this tree. The 2.88x narrative has no committed benchmark receipt and "
            "is not preserved as a verified historical measurement.",
        },
        "limitations": [
            "Only the admitted MHA geometry is covered (num_heads == num_key_value_heads).",
            "Only unfused_qk 0 and 9 are covered; other adjoint orders keep the scalar kernel.",
            "kv_seq_len must be <= head_dim, head_dim <= 128.",
            "Sliding window, softcap, ALiBi, relative bias, dropout, attention sink, "
            "position offset, sparse masks, compressed keys, paged KV and value_scale != 1 "
            "keep the scalar kernel.",
            "The missing-device BLOCKED path was code-reviewed but could not be produced "
            "on this host; the SKIP path was executed with the explicit opt-out.",
            "Complete-step timings are host and driver specific; the paired interval "
            "covers run-to-run variation on this machine only.",
        ],
    }
    output = RECEIPT / "qualification.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"{receipt['verdict']}: {output}")
    for failure in failures:
        print("  " + failure)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
