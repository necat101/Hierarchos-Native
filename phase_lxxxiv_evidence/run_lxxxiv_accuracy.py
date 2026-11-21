#!/usr/bin/env python3
"""Phase LXXXIV fresh accuracy sweep.

Runs the *committed* production harness (`wmma_qualify`, unmodified) over the
full Phase LXV matrix: eight seeds x five mode/oracle configurations, plus one
production operand/result dump per seed for the preserved K=96 comparison.

Nothing here changes a seed, an oracle, a tolerance, a reference tensor or a
gate limit. The invocation is the one the committed receipts were produced with:

    wmma_qualify --wmma-mode full-fp16 --full-oracle <tile16|split8|rdna3-dot2|split2> \
                 --seed <seed> --json
    wmma_qualify --wmma-mode mixed     --seed <seed> --json

The full-fp16 dump uses no override at all: the Full FP16 arm is the release
default, so `--dump-operands` records exactly what the production selector fed
the GPU.

Usage: run_lxxxiv_accuracy.py --exe <wmma_qualify.exe> [--out <dir>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SEEDS = (1593426484, 42, 1337, 271828, 314159, 777, 8675309, 20261119)
CONFIGS = (
    ("full-fp16", "tile16"),
    ("full-fp16", "split8"),
    ("full-fp16", "rdna3-dot2"),
    ("full-fp16", "split2"),
    ("mixed", "tile16"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_receipt(text: str) -> dict:
    return json.JSONDecoder().raw_decode(text[text.index("{"):])[0]


def run(exe: Path, argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([str(exe), *argv], capture_output=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()

    exe = args.exe.resolve()
    if not exe.is_file():
        raise SystemExit(f"missing harness: {exe}")
    accuracy = args.out / "accuracy"
    dumps = args.out / "dumps"
    accuracy.mkdir(parents=True, exist_ok=True)
    dumps.mkdir(parents=True, exist_ok=True)

    exe_hash = sha256(exe)
    rows = []
    for seed in SEEDS:
        for mode, oracle in CONFIGS:
            # `mixed` is qualified against its own contract-matched oracle; the
            # historical invocation passes no `--full-oracle` for it because the
            # Mixed arm does not consult the Full FP16 variant.
            argv = ["--wmma-mode", mode]
            if mode == "full-fp16":
                argv += ["--full-oracle", oracle]
            argv += ["--seed", str(seed), "--json"]
            stem = f"seed{seed}_{mode}_{oracle}"
            completed = run(exe, argv)
            (accuracy / f"{stem}.json").write_bytes(completed.stdout)
            (accuracy / f"{stem}.err").write_bytes(completed.stderr)
            record = {
                "seed": seed,
                "mode": mode,
                "oracle": oracle,
                "argv": argv,
                "rc": completed.returncode,
            }
            if completed.stdout:
                try:
                    receipt = parse_receipt(completed.stdout.decode("utf-8", "replace"))
                    record.update(
                        {
                            "gates": receipt["gates"],
                            "failed": receipt["failed"],
                            "device": receipt.get("device"),
                            "full_oracle": receipt.get("full_oracle"),
                            "dispatch_counts": receipt["dispatch_counts"],
                        }
                    )
                except Exception as error:  # noqa: BLE001 - preserved, not hidden
                    record["parse_error"] = repr(error)
            rows.append(record)
            print(
                f"seed {seed:>10} {mode:<9} {oracle:<11} rc={completed.returncode} "
                f"gates={record.get('gates')} failed={record.get('failed')}",
                flush=True,
            )

        # Production dump for the preserved K=96 comparison: no oracle override,
        # no variant override, no qualification flag.
        stem = dumps / f"dump_prod_p3_{seed}.json"
        if stem.exists():
            stem.unlink()
        completed = run(
            exe,
            [
                "--wmma-mode",
                "full-fp16",
                "--seed",
                str(seed),
                "--dump-operands",
                str(stem),
                "--json",
            ],
        )
        (dumps / f"dump_prod_p3_{seed}.err").write_bytes(completed.stderr)
        print(f"seed {seed:>10} production dump rc={completed.returncode}", flush=True)

    result = {
        "harness": str(exe),
        "harness_sha256": exe_hash,
        "seeds": list(SEEDS),
        "configs": [list(pair) for pair in CONFIGS],
        "receipts": rows,
        "receipts_total": len(rows),
        "receipts_failed_nonzero": sum(
            1 for row in rows if (row.get("failed") or 0) > 0
        ),
        "configs_all_gates_pass": sorted(
            {
                f"{row['mode']}/{row['oracle']}"
                for row in rows
                if (row.get("failed") or 0) == 0
            }
        ),
        "configs_with_failing_gates": sorted(
            {
                f"{row['mode']}/{row['oracle']}"
                for row in rows
                if (row.get("failed") or 0) > 0
            }
        ),
    }
    (args.out / "accuracy_matrix.json").write_text(
        json.dumps(result, indent=1), encoding="utf-8"
    )
    print(
        f"receipts={result['receipts_total']} "
        f"receipts_with_failing_gates={result['receipts_failed_nonzero']}"
    )
    print(f"configs_with_failing_gates={result['configs_with_failing_gates']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
