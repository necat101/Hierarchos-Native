"""Attribute the RDNA3 ``gpt2_wide/unmasked`` residual without touching the gate.

The probe reports under ``target/rdna3-port/diag-gpt2wide-*`` are retained
diagnostic copies of the same frozen fixture that the qualification suite
gates. This script does not run a GPU and never relaxes acceptance: it compares
each report against the frozen CPU oracle, counts the values that cross the
unchanged ``max_abs < 2e-7`` / ``rtol = 0`` gate, and separates text rows from
image-injected rows so a routing defect can be told apart from a reduction-order
tail.

Inputs are untracked build artifacts; the receipt binds them by SHA-256.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
from typing import Any

from qualify_multimodal_training import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = Path(__file__).resolve().parent
LIMIT = 2e-7
CASE = "gpt2_wide/unmasked"
MODES = ("default", "no_text", "no_packed")
ATTEMPT_HYPOTHESES = {
    "k128": "restrict the 128-block GEMM to exactly K=128",
    "nowelford": "disable the one-chunk small Welford norm",
    "avx2gemm": "install the AVX2 ATen-association row GEMM at width 16",
}
PHASES = ("alternate", "step-1", "step-2", "final")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def display(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def phase_logits(document: dict[str, Any], phase: str) -> list[float]:
    """Both the oracle and the probe wrap steps as ``forward``/``training``."""
    if phase == "alternate":
        entry = document["alternate"]
    elif phase == "final":
        entry = document["final"]
    elif phase.startswith("step-"):
        entry = document["steps"][int(phase.split("-")[1]) - 1]
    else:
        raise ValueError(f"unknown phase {phase}")
    if "logits" in entry:
        return entry["logits"]
    return entry["forward"]["logits"]


def f32_ulp(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", abs(value)))[0]
    exponent = (bits >> 23) & 0xFF
    if exponent == 0:
        return 2.0**-149
    return 2.0 ** (exponent - 127 - 23)


def group_stats(
    differences: dict[int, float], indices: list[int], mask: list[bool]
) -> dict[str, Any]:
    selected = [i for i in indices if mask[i]]
    if not selected:
        return {"values": 0, "max_abs": None, "failing_values": 0, "nonzero": 0}
    maxima = sorted(((differences[i], i) for i in selected), reverse=True)
    return {
        "values": len(selected),
        "max_abs": maxima[0][0],
        "worst_index": maxima[0][1],
        "failing_values": sum(1 for value, _ in maxima if value >= LIMIT),
        "nonzero": sum(1 for value, _ in maxima if value > 0),
    }


def distribution(
    got: list[float], expected: list[float], visible: list[bool], text_visible: list[bool]
) -> dict[str, Any]:
    if len(got) != len(expected) or len(got) != len(visible):
        raise ValueError("inconsistent comparison coverage")
    if not all(math.isfinite(value) for value in [*got, *expected]):
        raise ValueError("non-finite logit")
    indices = [i for i, flag in enumerate(visible) if flag]
    differences = {i: abs(got[i] - expected[i]) for i in indices}
    ranked = sorted(((differences[i], i) for i in indices), reverse=True)
    failing = [
        {
            "index": index,
            "max_abs": value,
            "got": got[index],
            "expected": expected[index],
        }
        for value, index in ranked
        if value >= LIMIT
    ]
    worst = ranked[0] if ranked else (None, None)
    return {
        "values": len(indices),
        "max_abs": worst[0],
        "worst_index": worst[1],
        "failing_values": len(failing),
        "failing": failing[:8],
        "nonzero": sum(1 for value, _ in ranked if value > 0),
        "text_rows": group_stats(differences, indices, text_visible),
        "image_rows": group_stats(differences, indices, [not flag for flag in text_visible]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("target/rdna3-port"))
    parser.add_argument("--case", default=CASE)
    parser.add_argument("--profiles", nargs="+", default=list(MODES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    fixture = root / "cpu-shared-v2" / args.case
    oracle = load(fixture / "cpu.json")
    visible = [bool(flag) for flag in oracle["visible"]]
    text_visible = [bool(flag) for flag in oracle["text_visible"]]
    inputs = load(fixture / "input.json")
    oracle_rows = int(inputs["batch_size"]) * int(inputs["seq_len"])
    vocab = len(oracle["final"]["logits"]) // oracle_rows

    def rows(index: int) -> int:
        return index // vocab

    runs: dict[str, Any] = {}
    failing_phases: dict[str, Any] = {}
    for mode in args.profiles:
        report_path = root / f"diag-gpt2wide-{mode}" / "report.json"
        report = load(report_path)
        phases: dict[str, Any] = {}
        worst = 0.0
        total_failing = 0
        failing_detail: dict[str, Any] = {}
        for phase in PHASES:
            stats = distribution(
                phase_logits(report, phase),
                phase_logits(oracle, phase),
                visible,
                text_visible,
            )
            stats["pass"] = stats["max_abs"] is not None and stats["max_abs"] < LIMIT
            for row in stats["failing"]:
                row["row"] = rows(row["index"])
            phases[phase] = stats
            if stats["max_abs"] is not None:
                worst = max(worst, stats["max_abs"])
            if stats["failing_values"]:
                failing_detail[phase] = stats["failing"]
            total_failing += stats["failing_values"]
        runs[mode] = {
            "report": display(report_path),
            "report_sha256": digest(report_path),
            "max_abs": worst,
            "failing_values": total_failing,
            "pass": worst < LIMIT,
            "phases": phases,
        }
        failing_phases[mode] = failing_detail

    base = runs["default"]
    default_final = base["phases"]["final"]
    ulp = f32_ulp(default_final["failing"][0]["expected"]) if default_final["failing"] else None
    ulp_span = (
        round(default_final["max_abs"] / ulp, 3)
        if ulp and default_final["max_abs"] is not None
        else None
    )

    suite_path = root / "cpu-shared-v2" / "suite-final" / "qualification.json"
    padded_control: dict[str, Any] = {
        "case": "gpt2_wide/padded",
        "visible_values": None,
        "worst_logit": None,
        "receipt": None,
        "receipt_sha256": None,
    }
    padded_fixture = root / "cpu-shared-v2" / "gpt2_wide" / "padded" / "cpu.json"
    if padded_fixture.is_file():
        padded_control["visible_values"] = sum(
            1 for flag in load(padded_fixture)["visible"] if flag
        )
    if suite_path.is_file():
        suite = load(suite_path)
        padded_control["receipt"] = display(suite_path)
        padded_control["receipt_sha256"] = digest(suite_path)
        for result in suite.get("results", []):
            if result.get("case") != "gpt2_wide/padded":
                continue
            values = [
                float(row["max_abs"])
                for row in result.get("checks") or result.get("metrics") or []
                if str(row.get("metric", "")).endswith("logits")
                and row.get("max_abs") is not None
            ]
            padded_control["worst_logit"] = max(values) if values else None
            padded_control["pass"] = bool(result.get("pass"))

    switches = {}
    for mode in ("no_text", "no_packed"):
        if mode not in runs:
            continue
        switches[mode] = {
            "max_abs": runs[mode]["max_abs"],
            "failing_values": runs[mode]["failing_values"],
            "delta_max_abs": runs[mode]["max_abs"] - base["max_abs"],
            "delta_failing_values": runs[mode]["failing_values"] - base["failing_values"],
        }

    attempts = []
    for mode, hypothesis in ATTEMPT_HYPOTHESES.items():
        attempt_path = root / f"diag-gpt2wide-{mode}" / "report.json"
        if not attempt_path.is_file():
            continue
        attempt = load(attempt_path)
        final_stats = distribution(
            phase_logits(attempt, "final"),
            phase_logits(oracle, "final"),
            visible,
            text_visible,
        )
        attempts.append(
            {
                "hypothesis": hypothesis,
                "report": display(attempt_path),
                "report_sha256": digest(attempt_path),
                "max_abs": final_stats["max_abs"],
                "failing_values": final_stats["failing_values"],
                "delta_max_abs": final_stats["max_abs"] - default_final["max_abs"],
                "closed": final_stats["max_abs"] < LIMIT,
            }
        )

    def rerun_worst(result: dict[str, Any]) -> float | None:
        values = [
            float(row["max_abs"])
            for row in result.get("checks") or result.get("metrics") or []
            if str(row.get("metric", "")).endswith("logits")
            and row.get("max_abs") is not None
        ]
        return max(values) if values else None

    rerun_path = root / "gpt2-wide-gate-rerun" / "qualification.json"
    gate_rerun = None
    if rerun_path.is_file():
        rerun = load(rerun_path)
        gate_rerun = {
            "receipt": display(rerun_path),
            "receipt_sha256": digest(rerun_path),
            "pass": bool(rerun.get("pass")),
            "cases": {
                result.get("case"): {
                    "pass": bool(result.get("pass")),
                    "worst_logit": rerun_worst(result),
                }
                for result in rerun.get("results", [])
            },
        }

    findings = [
        (
            f"The residual is global, not a path or mask defect: "
            f"{default_final['nonzero']}/{default_final['values']} final logits differ, "
            f"with {default_final['text_rows']['nonzero']}/"
            f"{default_final['text_rows']['values']} text-row and "
            f"{default_final['image_rows']['nonzero']}/"
            f"{default_final['image_rows']['values']} image-row values nonzero."
        ),
        (
            f"Only {default_final['failing_values']} of "
            f"{default_final['values']} final logits cross 2e-7; the worst is "
            f"{default_final['max_abs']!r} over {ulp_span} FP32 ulps at the "
            f"oracle value."
        ),
        (
            "Disabling the RDNA3 short-attention module makes the case worse "
            f"({switches.get('no_text', {}).get('max_abs')!r}, "
            f"{switches.get('no_text', {}).get('failing_values')} failing values), "
            "so that module is not the source of the residual."
        ),
        (
            "Disabling the packed-norm path changes nothing "
            f"({switches.get('no_packed', {}).get('max_abs')!r}), so the packed "
            "norm is not the source either."
        ),
        (
            "The padded control gates "
            f"{padded_control['visible_values']} values instead of "
            f"{default_final['values']}; the same few-ulp tail is sampled less, "
            "which is why it stays green without being bit-exact."
        ),
    ]
    if attempts:
        findings.append(
            "Candidate association changes were measured and rejected: "
            + "; ".join(
                f"{attempt['hypothesis']} -> {attempt['max_abs']!r} "
                f"({attempt['failing_values']} failing values)"
                for attempt in attempts
            )
            + "; none closes the strict gate."
        )

    receipt = {
        "schema": 1,
        "profile": "gpt2_wide",
        "case": args.case,
        "target": "rdna3",
        "device_id": "0x15bf",
        "acceptance": {
            "absolute_limit": LIMIT,
            "rtol": 0,
            "comparison": "strict_less_than",
            "unchanged": True,
        },
        "fixture": {
            "cpu_json": display(fixture / "cpu.json"),
            "cpu_json_sha256": digest(fixture / "cpu.json"),
            "visible_values": sum(1 for flag in visible if flag),
            "text_visible_values": sum(
                1 for i, flag in enumerate(visible) if flag and text_visible[i]
            ),
            "rows": oracle_rows,
            "vocab": vocab,
        },
        "padded_control": padded_control,
        "runs": runs,
        "switches": switches,
        "closure_attempts": attempts,
        "gate_rerun": gate_rerun,
        "findings": findings,
        "conclusion": {
            "classification": "reduction_order_tail",
            "path_specific": False,
            "mask_or_routing_defect": False,
            "short_attention_module_effect": "improves",
            "packed_norm_effect": "none",
            "profile_gate": "FAIL",
            "waived": False,
            "closure_status": (
                "closed_by_candidate"
                if any(attempt["closed"] for attempt in attempts)
                else "open"
            ),
            "fix_requires": (
                "bit-matching the composed GPT-2-wide GEMM association to the "
                "frozen ATen AVX2 oracle; the gate is not relaxed"
            ),
        },
    }
    output = (
        args.output
        or VALIDATION
        / "release_parity/multimodal_training/20261106-rdna3-profile-completion/gpt2-wide-residual.json"
    )
    write_json(output, receipt)
    print(
        f"GPT-2-wide residual: max {receipt['runs']['default']['max_abs']!r}, "
        f"{receipt['runs']['default']['failing_values']} failing values, "
        f"classification={receipt['conclusion']['classification']}"
    )


if __name__ == "__main__":
    main()
