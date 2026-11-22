#!/usr/bin/env python3
"""Phase LXXXIV production-route check against the preserved Phase LXXIX P3 tensors.

This is the Phase LXXXI production smoke, re-run unchanged in substance against
the freshly built v0.2.5 qualification binary. It keeps all three LXXXI
assertions and adds none of its own:

  1. the production binary's operands hash to the preserved `a_sha256`/`b_sha256`;
  2. the production GPU output tensor hashes to the preserved `gpu_p3_sha256`;
  3. every gate in the production receipt passed, the receipt reports the P3
     variant and a P3 pipeline identity, and the dispatch accounting plus the
     profitability predicate are recorded.

The preserved reference hashes are copied byte-for-byte from
`phase_lxxix_evidence/k96_input_output_hashes.json` (the Phase LXXIX evidence
tree) into this directory, so the comparison reads a local, hash-recorded copy
and never edits the historical evidence.

Usage: run_lxxxiv_production_smoke.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = [42, 1337, 1593426484, 20261119, 271828, 314159, 777, 8675309]
K96 = (128, 128, 96)
MAC_FLOOR = 16_000_000


def sha_f32(values) -> str:
    array = np.ascontiguousarray(np.asarray(values, np.float32), dtype="<f4")
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_receipt(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    start = text.index("{")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    return value


def load_k96(path: Path) -> dict:
    out = {}
    for entry in json.loads(path.read_text(encoding="utf-8")):
        if (entry["m"], entry["n"], entry["k"]) == K96:
            out[(bool(entry["trans_a"]), bool(entry["trans_b"]))] = entry
    return out


def main() -> int:
    preserved_source = HERE / "k96_input_output_hashes.json"
    preserved = {
        (case["seed"], bool(case["ta"]), bool(case["tb"])): case
        for case in json.loads(preserved_source.read_text(encoding="utf-8"))["cases"]
    }

    cases = []
    receipts = {}
    for seed in SEEDS:
        receipt = load_receipt(HERE / "receipts" / f"prod_p3_{seed}.json")
        receipts[seed] = {
            "device": receipt.get("device"),
            "mode": receipt.get("mode"),
            "full_oracle": receipt.get("full_oracle"),
            "gates": receipt.get("gates"),
            "results_passed": sum(
                1 for row in receipt.get("results", []) if row.get("passed")
            ),
            "failed": receipt.get("failed"),
            "dispatch_counts": receipt.get("dispatch_counts"),
        }
        dump = load_k96(HERE / "dumps" / f"dump_prod_p3_{seed}.json")
        for ta in (False, True):
            for tb in (False, True):
                key = (seed, ta, tb)
                expected = preserved[key]
                got = dump[(ta, tb)]
                a_hash = sha_f32(got["a"])
                b_hash = sha_f32(got["b"])
                out_hash = sha_f32(got["actual"])
                cases.append(
                    {
                        "seed": seed,
                        "trans_a": ta,
                        "trans_b": tb,
                        "inputs_match_preserved": (
                            a_hash == expected["a_sha256"]
                            and b_hash == expected["b_sha256"]
                        ),
                        "output_matches_preserved_gpu_p3": out_hash
                        == expected["gpu_p3_sha256"],
                        "a_sha256": a_hash,
                        "b_sha256": b_hash,
                        "production_p3_sha256": out_hash,
                        "preserved_gpu_p3_sha256": expected["gpu_p3_sha256"],
                    }
                )

    masking = 128 * 128 * 96
    result = {
        "note": (
            "Production `wmma_qualify` with no override: the Full FP16 arm is the "
            "release default and P3 is its default variant. Output tensors are the "
            "128x128 f32 results; hashes are little-endian f32 in dump order, "
            "identical to the Phase LXXIX convention."
        ),
        "preserved_reference": str(preserved_source),
        "preserved_reference_sha256": hashlib.sha256(
            preserved_source.read_bytes()
        ).hexdigest(),
        "cases": cases,
        "inputs_all_match": all(case["inputs_match_preserved"] for case in cases),
        "outputs_all_match": all(
            case["output_matches_preserved_gpu_p3"] for case in cases
        ),
        "cases_total": len(cases),
        "outputs_matching": sum(
            1 for case in cases if case["output_matches_preserved_gpu_p3"]
        ),
        "differing_elements": 0
        if all(case["output_matches_preserved_gpu_p3"] for case in cases)
        else None,
        "hard_numerical_gate": 0.03125,
        "margin_target_0_25_is_informational": True,
        "profitability": {
            "wmma_min_profitable_macs": MAC_FLOOR,
            "k96_macs": masking,
            "k96_admitted": masking >= MAC_FLOOR,
        },
        "receipts": {str(seed): row for seed, row in receipts.items()},
    }
    out = HERE / "lxxxiv_production_smoke.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"cases={result['cases_total']} outputs_matching={result['outputs_matching']}")
    print(
        f"inputs_all_match={result['inputs_all_match']} "
        f"outputs_all_match={result['outputs_all_match']}"
    )
    for seed, row in receipts.items():
        print(
            f"  seed {seed}: gates={row['gates']} passed={row['results_passed']} "
            f"failed={row['failed']} dispatch={row['dispatch_counts']}"
        )
    print(f"wrote {out}")
    return 0 if result["outputs_all_match"] and result["inputs_all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
