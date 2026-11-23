#!/usr/bin/env python3
"""Compare a candidate's production dumps against the preserved v0.2.5 dumps.

Both sides are produced by the *unmodified* production harness with no oracle
override. The baseline side is the archived v0.2.5 `wmma_qualify` resolving the
committed production selector (P3, no variant override). The candidate side is
the same harness with the candidate module selected.

The comparison is per case and bit-exact on the observed output tensor: every
`actual` list must have the same length and the same values, value for value.
There is no aggregate, no tolerance and no averaging, so a case that got worse
cannot be hidden by cases that got better.

Usage: compare_dumps.py <baseline-dump-dir> <candidate-dump-dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    baseline_dir = Path(sys.argv[1])
    candidate_dir = Path(sys.argv[2])
    baseline_files = sorted(baseline_dir.glob("dump_prod_p3_*.json"))
    if not baseline_files:
        raise SystemExit(f"no baseline dumps in {baseline_dir}")

    total_cases = 0
    identical = 0
    differing: list[str] = []
    missing: list[str] = []

    for baseline_path in baseline_files:
        candidate_path = candidate_dir / baseline_path.name
        if not candidate_path.is_file():
            missing.append(baseline_path.name)
            continue
        baseline = load(baseline_path)
        candidate = load(candidate_path)
        if len(baseline) != len(candidate):
            raise SystemExit(
                f"{baseline_path.name}: {len(candidate)} candidate cases, "
                f"{len(baseline)} baseline cases"
            )
        for index, (base_case, cand_case) in enumerate(zip(baseline, candidate)):
            shape = f"{base_case['m']}x{base_case['n']}x{base_case['k']}"
            orientation = f"trans_a={base_case['trans_a']} trans_b={base_case['trans_b']}"
            label = f"{baseline_path.name}[{index}] {shape} {orientation}"
            total_cases += 1
            base_actual = base_case["actual"]
            cand_actual = cand_case["actual"]
            if len(base_actual) != len(cand_actual):
                differing.append(f"{label}: length {len(cand_actual)} != {len(base_actual)}")
                continue
            # Inputs must be the same observation, or the comparison is void.
            if base_case["a"] != cand_case["a"] or base_case["b"] != cand_case["b"]:
                differing.append(f"{label}: operands differ between the two runs")
                continue
            mismatches = [
                (i, b, c)
                for i, (b, c) in enumerate(zip(base_actual, cand_actual))
                if b != c
            ]
            if mismatches:
                first = mismatches[0]
                differing.append(
                    f"{label}: {len(mismatches)} differing elements, first at "
                    f"{first[0]}: baseline {first[1]!r} candidate {first[2]!r}"
                )
            else:
                identical += 1

    for name in missing:
        print(f"MISSING candidate dump: {name}")
    for line in differing:
        print(f"DIFFERS {line}")
    print(
        f"cases={total_cases} bit_identical={identical} differing={len(differing)} "
        f"missing_files={len(missing)}"
    )
    if differing or missing or identical != total_cases:
        print("BIT-IDENTICAL COMPARISON: FAIL")
        return 1
    print("BIT-IDENTICAL COMPARISON: PASS (every production case, every element)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
