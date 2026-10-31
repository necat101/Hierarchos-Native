#!/usr/bin/env python3
"""Measure how a training step's throughput scales with the tokens it carries.

The training step's GPU cost barely moves with the number of tokens (see
`rank_kernel_profile.py` and the third pass in
PROGRESS_HIERARCHOS_VULKAN_TRAINING_OPTIMIZATIONS.md), because a step is thousands
of small dispatches whose per-launch cost is fixed. If that is true, the cheapest
available throughput win is not a kernel change but carrying more tokens per step,
so that the same 6912 dispatches amortize over more useful work.

This sweeps `training_step_probe` over batch sizes (rows per step) and token counts
(sequence length) and reports wall-clock and GPU milliseconds per step, tokens per
second, and the dispatch and barrier counts. Two knobs are separated on purpose:
`--batch` adds independent sequences to the same step, which is the on-policy way to
amortize (the update just averages over more data), while `--tokens` lengthens the
sequence, which changes what the model is being asked to do.

Every configuration starts from a freshly initialized model package, so all of them
begin at the same weights and their losses are comparable.

Usage:
    python validation/sweep_step_tokens.py
    python validation/sweep_step_tokens.py --batches 2,8,32 --tokens 8,32 --json target/token_sweep.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
SCRATCH = CRATE / "target" / "token_sweep"


def probe_args(batch, tokens, steps, warmup, sequences, device_index):
    return [
        "--batch", str(batch),
        "--tokens", str(tokens),
        "--sequences", str(sequences),
        "--steps", str(steps),
        "--warmup", str(warmup),
        "--device-index", str(device_index),
    ]


def run_probe(binary, batch, tokens, steps, warmup, sequences, device_index, profile):
    """Run one configuration on a fresh fixture and return its parsed report."""
    fixture = SCRATCH / f"fixture_b{batch}_t{tokens}_s{sequences}"
    if fixture.exists():
        shutil.rmtree(fixture)
    environment = dict(os.environ)
    profile_log = SCRATCH / f"profile_b{batch}_t{tokens}_s{sequences}.log"
    if profile:
        environment["HIERARCHOS_VULKAN_PROFILE_KERNELS"] = "1"
    with open(profile_log, "w", encoding="utf-8") as log:
        completed = subprocess.run(
            [
                str(binary),
                "--model-dir", str(fixture),
                *probe_args(batch, tokens, steps, warmup, sequences, device_index),
            ],
            cwd=str(CRATE),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"batch {batch} tokens {tokens}: probe exited {completed.returncode}; "
            f"see {profile_log}"
        )
    report = json.loads(completed.stdout[completed.stdout.index("{"):])
    report["gpu_ms_per_step"] = None
    if profile:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import rank_kernel_profile

        submissions = rank_kernel_profile.parse(profile_log)
        steps_run = [
            submission
            for submission in submissions
            if submission["dispatches"] >= report["dispatch_count"] // 2
        ]
        if steps_run:
            report["gpu_ms_per_step"] = sum(
                submission["gpu_ms"] for submission in steps_run
            ) / len(steps_run)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--batches", default="2,8,32,128", help="comma-separated rows per step")
    parser.add_argument("--tokens", default="8,32", help="comma-separated sequence lengths")
    parser.add_argument("--sequences", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--json", type=Path, help="write the raw reports here")
    args = parser.parse_args(argv)

    profile = "release" if args.release else "debug"
    binary = CRATE / "target" / profile / (
        "training_step_probe.exe" if os.name == "nt" else "training_step_probe"
    )
    if not binary.exists():
        print(f"{binary} is missing; build it first", file=sys.stderr)
        return 1
    SCRATCH.mkdir(parents=True, exist_ok=True)

    batches = [int(value) for value in args.batches.split(",")]
    tokens_list = [int(value) for value in args.tokens.split(",")]

    reports = []
    for tokens in tokens_list:
        for batch in batches:
            report = run_probe(
                binary,
                batch,
                tokens,
                args.steps,
                args.warmup,
                args.sequences,
                args.device_index,
                profile=True,
            )
            reports.append(report)
            print(
                f"measured batch {batch} x {tokens} tokens "
                f"({report['batch_tokens_per_step']} tokens/step)",
                file=sys.stderr,
            )

    baseline = reports[0]
    print()
    print(
        f"{'batch':>6} {'tokens':>7} {'tok/step':>9} {'wall_ms':>9} {'gpu_ms':>8} "
        f"{'gpu_share':>9} {'tokens/s':>9} {'vs_first':>9} {'disp':>6} {'barriers':>8} {'loss':>10}"
    )
    for report in reports:
        tokens_per_step = report["batch_tokens_per_step"]
        throughput = tokens_per_step / (report["median_step_ms"] / 1000.0)
        baseline_throughput = baseline["batch_tokens_per_step"] / (
            baseline["median_step_ms"] / 1000.0
        )
        gpu = report["gpu_ms_per_step"]
        share = f"{gpu / report['median_step_ms'] * 100:.1f}%" if gpu else "n/a"
        print(
            f"{report['batch']:>6} {report['tokens_per_sequence']:>7} {tokens_per_step:>9} "
            f"{report['median_step_ms']:>9.2f} {gpu if gpu else float('nan'):>8.2f} {share:>9} "
            f"{throughput:>9.1f} {throughput / baseline_throughput:>8.2f}x "
            f"{report['dispatch_count']:>6} {report['shader_barrier_count']:>8} "
            f"{report['mean_loss']:>10.4f}"
        )

    best = max(
        reports,
        key=lambda report: report["batch_tokens_per_step"] / (report["median_step_ms"] / 1000.0),
    )
    best_throughput = best["batch_tokens_per_step"] / (best["median_step_ms"] / 1000.0)
    baseline_throughput = baseline["batch_tokens_per_step"] / (
        baseline["median_step_ms"] / 1000.0
    )
    print()
    print(
        f"best: {best['batch']} rows x {best['tokens_per_sequence']} tokens = "
        f"{best['batch_tokens_per_step']} tokens/step at {best['median_step_ms']:.2f} ms "
        f"({best_throughput:.1f} tokens/s), {best_throughput / baseline_throughput:.2f}x the "
        f"{baseline['batch_tokens_per_step']}-token baseline"
    )
    print(
        f"dispatches per step: {baseline['dispatch_count']} at "
        f"{baseline['batch_tokens_per_step']} tokens vs {best['dispatch_count']} at "
        f"{best['batch_tokens_per_step']} tokens"
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
