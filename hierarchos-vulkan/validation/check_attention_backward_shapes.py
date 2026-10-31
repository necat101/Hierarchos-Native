"""Fail when an attention-backward dispatch stops completing on this device.

The strict attention backward resets the device when a dispatch exceeds the
driver's watchdog, and which shapes do that has already changed once: the head
dimension above the 256-wide cache bound used to fall through to a replay that
recomputed every head-dimension-independent statistic once per element, which
made the dispatch about `head_dim` times more expensive. That was invisible to
every bit-exactness check, because those shapes never produced an output to
compare. This check asserts the weaker property those checks cannot: the dispatch
finishes.

Each case is run with the same module on both sides of the harness, so a failure
is the module's and never the baseline's, and `--bench-only` still requires the
harness to compare the two output vectors bit for bit before it reports a timing.
A device loss is fatal to the process, so a non-zero exit status is the signal;
the harness's own error text is reported with it.

    python validation/check_attention_backward_shapes.py
    python validation/check_attention_backward_shapes.py --candidate-dir <dir> \
        --baseline-dir <dir> --binary <path>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "target" / "debug" / "attention_kernel_cache_check.exe"

# Head dimensions around the 256-wide cache bound, and the batch ladder the
# private-cache change was measured on.
HEAD_DIM_CASES = [
    {"mode": 1, "head_dim": 32, "batch": 1, "query": 8, "key": 8},
    {"mode": 1, "head_dim": 64, "batch": 1, "query": 8, "key": 8},
    {"mode": 1, "head_dim": 128, "batch": 1, "query": 8, "key": 8},
    {"mode": 1, "head_dim": 256, "batch": 1, "query": 8, "key": 8},
    {"mode": 1, "head_dim": 257, "batch": 1, "query": 8, "key": 8},
    {"mode": 1, "head_dim": 512, "batch": 1, "query": 8, "key": 8},
    {"mode": 1, "head_dim": 1024, "batch": 1, "query": 8, "key": 8},
]
BATCH_CASES = [
    {"mode": 0, "head_dim": 32, "batch": batch, "query": 64, "key": 64}
    for batch in (2, 8, 32, 128)
]

# The sequence ladder from the device sweep. Every case here is a shape that
# completes on this part, so it is an assertion rather than a survey: if the
# driver watchdog boundary narrows -- a slower kernel, a colder part, more
# head-dimension work in the same dispatch -- the last known-good shape in a mode
# fails first and this check catches it. The measured boundary is recorded beside
# it instead of being asserted, because asserting a device reset would reset the
# device on every run.
SEQUENCE_CASES = [
    {"mode": 0, "head_dim": 32, "batch": 1, "query": 64, "key": 64},
    {"mode": 0, "head_dim": 32, "batch": 1, "query": 128, "key": 128},
    {"mode": 0, "head_dim": 32, "batch": 1, "query": 256, "key": 256},
    {"mode": 1, "head_dim": 32, "batch": 1, "query": 64, "key": 64},
    {"mode": 1, "head_dim": 32, "batch": 1, "query": 128, "key": 128},
    {"mode": 1, "head_dim": 32, "batch": 1, "query": 224, "key": 224},
]
KNOWN_LIMITS = {
    "measured_on": "Intel(R) HD Graphics 520, 2026-10-15",
    "budget_seconds": 3.0,
    "last_good": {"mode0": "seq 256, 2576 ms", "mode1": "seq 240, 2973 ms"},
    "first_loss": {"mode0": "seq 320", "mode1": "seq 248"},
    "note": "the limit is per-invocation work, so batching does not move it; "
            "forward survives 1558 ms at seq 1024",
}
CASE_TIMEOUT_SECONDS = 600


def arguments(case: dict, baseline_dir: str, candidate_dir: str) -> list[str]:
    return [
        "--candidate-dir",
        candidate_dir,
        "--bench",
        "--bench-only",
        "--bench-mode",
        str(case["mode"]),
        "--bench-repeats",
        "1",
        "--bench-batch",
        str(case["batch"]),
        "--bench-query",
        str(case["query"]),
        "--bench-key",
        str(case["key"]),
        "--bench-qk-dim",
        str(case["head_dim"]),
        "--bench-value-dim",
        str(case["head_dim"]),
    ]


def run_case(binary: Path, case: dict, baseline_dir: str, candidate_dir: str) -> dict:
    command = [str(binary), baseline_dir, *arguments(case, baseline_dir, candidate_dir)]
    label = f"mode{case['mode']} head_dim={case['head_dim']} batch={case['batch']}"
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=CASE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {"case": label, "status": "timeout"}
    report = result.stdout + result.stderr
    if result.returncode != 0:
        detail = next(
            (line for line in report.splitlines() if line.startswith("Error")),
            report.strip().splitlines()[-1] if report.strip() else "no output",
        )
        return {"case": label, "status": "failed", "detail": detail, "exit": result.returncode}
    # The harness benchmarks the forward pass first and the backward pass second;
    # the backward dispatch is the one this check exists for, so report that one.
    timing = next(
        (line for line in report.splitlines() if line.startswith("benchmark ") and "backward=true" in line),
        "",
    ) or next((line for line in report.splitlines() if line.startswith("benchmark ")), "")
    return {"case": label, "status": "ok", "timing": timing}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--candidate-dir", default="shaders")
    parser.add_argument(
        "--baseline-dir",
        default=None,
        help="defaults to --candidate-dir, so both sides are the same module",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--cases", choices=["all", "head-dim", "batch", "sequence"], default="all"
    )
    args = parser.parse_args()

    if not args.binary.is_file():
        print(f"error: harness binary not found: {args.binary}")
        print("build it first: cargo build --bin attention_kernel_cache_check")
        return 1
    baseline_dir = args.baseline_dir or args.candidate_dir

    cases = []
    if args.cases in ("all", "head-dim"):
        cases += HEAD_DIM_CASES
    if args.cases in ("all", "sequence"):
        cases += SEQUENCE_CASES
    if args.cases in ("all", "batch"):
        cases += BATCH_CASES

    rows = [run_case(args.binary, case, baseline_dir, args.candidate_dir) for case in cases]
    failures = [row for row in rows if row["status"] != "ok"]
    for row in rows:
        extra = row.get("detail", row.get("timing", ""))
        print(f"{row['status']:7} {row['case']}  {extra}".rstrip())
    report = {
        "binary": str(args.binary),
        "baseline_dir": baseline_dir,
        "candidate_dir": args.candidate_dir,
        "known_limits": KNOWN_LIMITS,
        "cases": rows,
        "pass": not failures,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failures:
        print(
            f"FAIL: {len(failures)} of {len(rows)} attention-backward shapes did not complete"
        )
        return 1
    print(f"PASS: every attention-backward shape completed ({len(rows)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
