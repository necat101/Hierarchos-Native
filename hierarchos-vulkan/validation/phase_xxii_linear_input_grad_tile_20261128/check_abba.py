#!/usr/bin/env python3
"""Check the Phase XXII ABBA logs against the predeclared rule.

The timing reduction stays in the established Phase XII reducer; this script
adds the evidence checks that reducer does not make: every process exit, the
candidate announcement (so arm B really selected the tiled kernel), and the
end-to-end parameter digest equality between the two arms.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TILE_MARK = "[phase-xxii-tile] dispatch"
RUN_END = "[phase-xii-step-run-end]"


def main() -> int:
    raw = Path(sys.argv[1])
    rounds = int(sys.argv[2])
    abba = json.loads((raw / "abba.json").read_text(encoding="utf-8"))
    ok = True

    for index in range(rounds):
        digests = {}
        for position in range(4):
            tag = "A" if position in (0, 3) else "B"
            path = raw / f"abba-r{index}-p{position}-{tag}.log"
            text = path.read_text(encoding="utf-8", errors="replace")
            end = {}
            for line in text.splitlines():
                if RUN_END in line:
                    end = dict(token.split("=", 1) for token in line.split() if "=" in token)
            tiles = text.count(TILE_MARK)
            if tag == "B" and tiles != 1:
                print(f"FAIL r{index} p{position}: candidate announcement {tiles}, expected 1")
                ok = False
            if tag == "A" and tiles != 0:
                print(f"FAIL r{index} p{position}: control log announced the candidate")
                ok = False
            digests[tag] = end.get("parameter_end_digest")
        if digests.get("A") != digests.get("B") or digests.get("A") is None:
            print(f"FAIL r{index}: digests differ A={digests.get('A')} B={digests.get('B')}")
            ok = False
        else:
            print(f"round {index}: parameter_end_digest={digests['A']} (identical)")

    ci = abba["ratio_ci95"]
    improved = abba["rounds_improved"]
    median = abba["median_round_ratio"]
    print(
        f"ratios median={median} CI95={ci} rounds_improved={improved}/{abba['rounds_total']} "
        f"speedup={abba['speedup']}"
    )
    if not (ci[1] < 1.0 and improved >= 4 and median <= 0.990):
        print("FAIL: the predeclared performance gate is not met")
        ok = False
    print("VERDICT=" + ("GO" if ok else "NO-GO"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
