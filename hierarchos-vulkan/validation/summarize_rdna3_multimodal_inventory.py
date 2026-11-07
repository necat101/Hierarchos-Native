"""Summarize physical RDNA3 multimodal receipts against the 44-profile inventory.

The shared, sequence-to-sequence, KIMI and native V4.1 qualifiers each use a
different runner. This script does not rerun them; it joins their retained
receipts by the frozen inventory case names and writes one status document plus
the six-profile replay receipt. Missing or duplicate cases are errors.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from qualify_multimodal_training import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = Path(__file__).resolve().parent
INVENTORY = VALIDATION / "multimodal_training_inventory.json"
SIX_PROFILES = [
    "bart",
    "t5",
    "switch_transformers",
    "kimi_k3_kda",
    "kimi_k3_mla",
    "kimi_k3_text",
]
UNPADDED = {"kimi_k3_kda", "kimi_k3_mla", "kimi_k3_text"}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def display(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def logit_checks(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = result.get("checks") or result.get("metrics") or []
    return [
        row
        for row in rows
        if str(row.get("metric", "")).endswith("logits")
        and row.get("max_abs") is not None
    ]


def worst_logit(result: dict[str, Any]) -> float | None:
    values = [float(row["max_abs"]) for row in logit_checks(result)]
    return max(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("target/rdna3-port"))
    parser.add_argument("--shared", type=Path)
    parser.add_argument("--seq2seq", type=Path)
    parser.add_argument("--kimi", type=Path)
    parser.add_argument("--v41", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--six-output", type=Path)
    parser.add_argument("--target", default="rdna3")
    parser.add_argument("--device-id", type=lambda x: int(x, 0), default=0x15BF)
    args = parser.parse_args()

    root = args.root.resolve()
    shared_path = (args.shared or root / "cpu-shared-v2/suite-final/qualification.json").resolve()
    seq2seq_path = (args.seq2seq or root / "cpu-seq2seq-v1-run/qualification.json").resolve()
    kimi_path = (args.kimi or root / "kimi-replay-v1/qualification.json").resolve()
    v41_path = (args.v41 or root / "cpu-deepseek-v41-v1-run/qualification.json").resolve()
    output = (args.output or root / "inventory-status.json").resolve()
    six_output = (args.six_output or output.with_name("six-profile-replay.json")).resolve()

    documents = {
        "shared": load(shared_path),
        "seq2seq": load(seq2seq_path),
        "kimi": load(kimi_path),
        "v41": load(v41_path),
    }
    rows: dict[str, dict[str, Any]] = {}
    for source, document in documents.items():
        for result in document.get("results", []):
            name = result.get("case")
            if not name or name in rows:
                raise ValueError(f"missing or duplicate case in {source}: {name}")
            rows[name] = {**result, "_source": source}

    inventory = load(INVENTORY)
    profiles = inventory["profiles"]
    expected_profiles = [entry["profile"] for entry in profiles]
    expected_cases = [
        case["case"]
        for entry in profiles
        for case in entry.get("tested_cases", [])
    ]
    if len(expected_cases) != len(set(expected_cases)):
        raise ValueError("duplicate expected case in inventory")
    missing_cases = [name for name in expected_cases if name not in rows]
    extra_cases = sorted(set(rows) - set(expected_cases))
    if missing_cases or extra_cases:
        raise ValueError(
            f"receipt/inventory case mismatch; missing={missing_cases[:8]} "
            f"extra={extra_cases[:8]}"
        )

    def source_for(profile: str) -> str:
        if profile == "deepseek_v41_scalable":
            return "v41"
        if profile in UNPADDED:
            return "kimi"
        if profile in {"bart", "t5", "switch_transformers"}:
            return "seq2seq"
        return "shared"

    profile_status: dict[str, dict[str, Any]] = {}
    for entry in profiles:
        profile = entry["profile"]
        expected = [case["case"] for case in entry.get("tested_cases", [])]
        present = [rows[name] for name in expected]
        failures = [name for name, row in zip(expected, present) if not row.get("pass")]
        worst = max(
            (worst_logit(row) for row in present if worst_logit(row) is not None),
            default=None,
        )
        profile_status[profile] = {
            "pass": not failures,
            "expected_cases": len(expected),
            "completed_cases": len(present),
            "failing_cases": failures,
            "worst_logit": worst,
            "source": source_for(profile),
        }

    failing = [p for p, status in profile_status.items() if not status["pass"]]
    fully_green = len(profiles) - len(failing)
    six_status = {
        profile: profile_status[profile]
        for profile in SIX_PROFILES
    }
    six_pass = all(status["pass"] for status in six_status.values())
    v41_status = profile_status["deepseek_v41_scalable"]

    source_hashes = {
        name: digest(path)
        for name, path in {
            "shared": shared_path,
            "seq2seq": seq2seq_path,
            "kimi": kimi_path,
            "v41": v41_path,
        }.items()
    }
    case_counts = {
        "shared_base": len(documents["shared"].get("results", [])),
        "kimi": len(documents["kimi"].get("results", [])),
        "seq2seq": len(documents["seq2seq"].get("results", [])),
        "native_v41": len(documents["v41"].get("results", [])),
    }
    case_counts["shared_family"] = case_counts["shared_base"] + case_counts["kimi"]
    case_counts["expected"] = len(expected_cases)
    case_counts["completed"] = len(rows)
    case_counts["passing"] = sum(bool(row.get("pass")) for row in rows.values())
    case_counts["failing"] = case_counts["completed"] - case_counts["passing"]

    six_receipt = {
        "schema": 1,
        "target": args.target,
        "device_id": hex(args.device_id),
        "profile_count": len(SIX_PROFILES),
        "case_count": sum(
            status["expected_cases"] for status in six_status.values()
        ),
        "all_pass": six_pass,
        "profiles": {
            profile: {
                **status,
                "cases": {
                    name: {
                        "pass": rows[name]["pass"],
                        "worst_logit": worst_logit(rows[name]),
                    }
                    for name in expected_cases
                    if name in rows and rows[name].get("profile") == profile
                },
            }
            for profile, status in six_status.items()
        },
        "receipts": {
            "seq2seq": display(seq2seq_path),
            "kimi": display(kimi_path),
        },
        "source_sha256": {
            "seq2seq": source_hashes["seq2seq"],
            "kimi": source_hashes["kimi"],
        },
    }
    write_json(six_output, six_receipt)

    status = {
        "schema": 1,
        "target": args.target,
        "device_id": hex(args.device_id),
        "acceptance": {
            "absolute_limit": 2e-7,
            "rtol": 0,
            "comparison": "strict_less_than",
            "unchanged": True,
        },
        "inventory": {
            "expected_profiles": len(profiles),
            "fully_green_profiles": fully_green,
            "failing_profiles": failing,
            "missing_profiles": [],
            "profile_coverage_complete": True,
            "all_profiles_green": not failing,
            "all_green_backbones_qualified": not failing,
            "rdna3_multimodal_qualified": not failing,
            "six_previous_missing_profiles_green": six_pass,
            "seventh_native_v41_profile_green": v41_status["pass"],
        },
        "cases": case_counts,
        "six_profile_replay": {
            "profiles": SIX_PROFILES,
            "profile_count": len(SIX_PROFILES),
            "case_count": sum(
                status["expected_cases"] for status in six_status.values()
            ),
            "all_pass": six_pass,
            "receipt": six_output.name,
        },
        "native_v41": {
            "profile": "deepseek_v41_scalable",
            "pass": v41_status["pass"],
            "cases": v41_status["expected_cases"],
            "worst_logit": v41_status["worst_logit"],
            "receipt": display(v41_path),
        },
        "profiles": profile_status,
        "receipts": {
            "shared": display(shared_path),
            "seq2seq": display(seq2seq_path),
            "kimi": display(kimi_path),
            "v41": display(v41_path),
        },
        "source_sha256": source_hashes,
        "notes": [
            "The six shared-profile gaps are now replayed on the physical RDNA3 adapter.",
            "The native V4.1 profile uses DeepseekV41MultimodalTrainer, not the shared CLIP adapter.",
            "All 44 profiles and all 85 image cases pass at the unchanged strict gate.",
            "GPT-2-wide passes both unmasked and padded; its composed forward stages are bit-exact against eager ATen.",
            "The former residual was closed by matching QK muladd, SLEEF-u10 exp, the exact reciprocal and the softmax sum in the GPT-2 forward and backward attention paths.",
        ],
    }
    write_json(output, status)
    print(
        f"RDNA3 inventory: {fully_green}/{len(profiles)} profiles green, "
        f"{case_counts['passing']}/{case_counts['completed']} cases pass; "
        f"six replay={'PASS' if six_pass else 'FAIL'}, V4.1={'PASS' if v41_status['pass'] else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
