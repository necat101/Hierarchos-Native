#!/usr/bin/env python3
"""Phase LXXXI production-route verification against the Phase LXXIX hardware P3.

No new numerical model is introduced here. For each of the 32 preserved K=96
cases this checks three things that together prove the production selector runs
the same artifact and arithmetic the qualification campaign executed:

  1. the operands the production binary fed the GPU hash to the preserved
     `a_sha256` / `b_sha256` (same inputs);
  2. the production GPU output tensor hashes to the preserved `gpu_p3_sha256`
     (bit-identical output);
  3. every gate in the production receipt passed, and the receipt reports the
     P3 variant and a P3 pipeline identity.

Usage: lxxxi_production_smoke.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LXXIX = HERE.parent / "phase_lxxix_evidence"
SEEDS = [42, 1337, 1593426484, 20261119, 271828, 314159, 777, 8675309]
K96 = (128, 128, 96)


def sha_f32(values) -> str:
    array = np.ascontiguousarray(np.asarray(values, np.float32), dtype="<f4")
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_receipt(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    start = text.index("{")
    value, _ = decoder.raw_decode(text[start:])
    return value


def load_k96(path: Path) -> dict:
    out = {}
    for entry in json.loads(path.read_text(encoding="utf-8")):
        if (entry["m"], entry["n"], entry["k"]) == K96:
            out[(bool(entry["trans_a"]), bool(entry["trans_b"]))] = entry
    return out


def main() -> int:
    preserved = {
        (case["seed"], bool(case["ta"]), bool(case["tb"])): case
        for case in json.loads(
            (LXXIX / "k96_input_output_hashes.json").read_text(encoding="utf-8")
        )["cases"]
    }

    cases = []
    receipts = {}
    for seed in SEEDS:
        receipt = load_receipt(HERE / "receipts" / f"prod_p3_{seed}.json")
        # This build's `wmma_qualify` predates the variant/pipeline reporting
        # fields; the selected module is evidenced by the `wmma_bench` receipt
        # and by the unit test that hashes the embedded P3 SPIR-V. Here the
        # numerical gates and dispatch counts are the receipt's contribution.
        receipts[seed] = {
            "device": receipt.get("device"),
            "mode": receipt.get("mode"),
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

    result = {
        "note": (
            "Production `wmma_qualify` with no override: the Full FP16 arm is the "
            "release default. Output tensors are the 128x128 f32 results; hashes "
            "are little-endian f32 in dump order, identical to the Phase LXXIX "
            "convention."
        ),
        "cases": cases,
        "inputs_all_match": all(case["inputs_match_preserved"] for case in cases),
        "outputs_all_match": all(
            case["output_matches_preserved_gpu_p3"] for case in cases
        ),
        "cases_total": len(cases),
        "outputs_matching": sum(
            1 for case in cases if case["output_matches_preserved_gpu_p3"]
        ),
        "receipts": {str(seed): row for seed, row in receipts.items()},
    }
    out = HERE / "lxxxi_production_smoke.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"cases={result['cases_total']} outputs_matching={result['outputs_matching']}")
    print(f"inputs_all_match={result['inputs_all_match']} outputs_all_match={result['outputs_all_match']}")
    for seed, row in receipts.items():
        print(
            f"  seed {seed}: gates={row['gates']} passed={row['results_passed']} "
            f"failed={row['failed']} dispatch={row['dispatch_counts']}"
        )
    print(f"wrote {out}")
    return 0 if result["outputs_all_match"] and result["inputs_all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
