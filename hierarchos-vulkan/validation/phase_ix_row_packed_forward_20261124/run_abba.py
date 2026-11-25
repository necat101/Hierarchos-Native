#!/usr/bin/env python3
"""Phase IX complete-step ABBA: generic forward (A) vs row-packed forward (B).

Phase III/VII protocol, unchanged: three warmup steps, twenty-six measured steps
per process, five paired ABBA rounds, one arm per process, strictly serial.

  round r: A B B A    (A = production generic attention forward,
                       B = the row-packed candidate, both with the identical
                           production two-pass GQA backward)

Each process is `phase_ix_row_packed_forward_full_step` with
`HIERARCHOS_PHASE_IX_STEP_ARM` selecting the arm; the harness drives the same
training surface as `hierarchos-vulkan-training-qualification` (same fixture
model, same LoRA config, same AdamW hyper-parameters, same dataset and step
order) and swaps only `kernels.attention_forward`.

Digests are off in the timing runs: hashing gradients host-side inside the
measured interval would add work that neither arm of Phase VII paid.

Every number printed here comes from a `[phase-ix-step]` line the harness
emitted; a process that fails, or that yields the wrong number of measured
steps, raises instead of contributing a sample.
"""
from __future__ import annotations

import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

CRATE = Path(__file__).resolve().parents[2]
RECEIPT = CRATE / "validation/phase_ix_row_packed_forward_20261124"
RAW = RECEIPT / "raw"
STEPS = 29
WARMUP = 3
ROUNDS = 5

STEP = re.compile(
    r"\[phase-ix-step\] arm=(\S+) step=(\d+) epoch=(\d+) batch=(\d+) "
    r"measured=(\w+) step_ms=([0-9.]+) loss_bits=([0-9a-f]{8}) loss=(\S+)"
)


def t_critical(df: int) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
             8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
             15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086}
    return table.get(df, table[min(table, key=lambda key: abs(key - df))])


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    mean = statistics.fmean(values)
    if n < 2:
        return mean, float("nan"), float("nan")
    half = t_critical(n - 1) * statistics.stdev(values) / math.sqrt(n)
    return mean, mean - half, mean + half


def harness() -> Path:
    if os.environ.get("HIERARCHOS_PHASE_IX_ABBA_RELEASE") == "1":
        found = sorted((CRATE / "target/release/deps").glob("hierarchos_vulkan-*.exe"))
    else:
        found = sorted(
            (CRATE / "target/debug/deps").glob("hierarchos_vulkan-*.exe"),
            key=lambda path: path.stat().st_mtime,
        )
        found = found[-1:]
    if not found:
        raise SystemExit("no harness binary; run `cargo test --lib phase_ix --no-run` first")
    return found[-1]


def main() -> int:
    exe = harness()
    RAW.mkdir(parents=True, exist_ok=True)
    samples_path = RAW / "abba-samples.jsonl"
    print(f"harness: {exe}", file=sys.stderr)
    print(f"protocol: rounds={ROUNDS} steps={STEPS} warmup={WARMUP} "
          f"order=A,B,B,A", file=sys.stderr)

    arm_samples: dict[str, list[float]] = {"A": [], "B": []}
    per_round: list[dict] = []
    with samples_path.open("w", encoding="utf-8") as samples:
        for round_index in range(ROUNDS):
            per_arm: dict[str, list[float]] = {"A": [], "B": []}
            for position, arm in enumerate(["A", "B", "B", "A"]):
                step_arm = "generic" if arm == "A" else "rowpacked-cached"
                tag = f"abba-r{round_index}-p{position}-{arm}"
                env = {key: value for key, value in os.environ.items()
                       if not key.startswith("HIERARCHOS_VULKAN_PROFILE")
                       and key not in ("HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS",
                                       "VK_ICD_FILENAMES", "VK_DRIVER_FILES")}
                env["HIERARCHOS_WMMA_MODE"] = "off"
                env["HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS"] = "1"
                env["HIERARCHOS_PHASE_IX_STEP_ARM"] = step_arm
                env["HIERARCHOS_PHASE_IX_STEP_STEPS"] = str(STEPS)
                log = RAW / f"abba-{tag}.log"
                started = time.monotonic()
                with log.open("w", encoding="utf-8") as handle:
                    completed = subprocess.run(
                        [str(exe), "phase_ix_row_packed_forward_full_step",
                         "--nocapture", "--test-threads=1"],
                        cwd=CRATE, env=env, stdout=handle, stderr=subprocess.STDOUT,
                        timeout=3600,
                    )
                elapsed_ms = (time.monotonic() - started) * 1000.0
                if completed.returncode != 0:
                    raise SystemExit(f"{tag} failed rc={completed.returncode}; see {log}")
                rows = []
                for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                    match = STEP.search(line)
                    if match:
                        rows.append(match.groups())
                measured = [float(row[5]) for row in rows if row[4] == "true"]
                if len(measured) != STEPS - WARMUP:
                    raise SystemExit(
                        f"{tag}: expected {STEPS - WARMUP} measured samples, got {len(measured)}"
                    )
                per_arm[arm].extend(measured)
                arm_samples[arm].extend(measured)
                samples.write(json.dumps({
                    "round": round_index,
                    "position": position,
                    "arm": arm,
                    "step_arm": step_arm,
                    "process_wall_ms": round(elapsed_ms, 3),
                    "step_ms": [round(value, 6) for value in measured],
                    "loss_bits": [row[6] for row in rows],
                }) + "\n")
                samples.flush()
                print(f"  round {round_index} position {position} arm {arm}: "
                      f"median {statistics.median(measured):.3f} ms", file=sys.stderr)
            medians = {arm: statistics.median(values) for arm, values in per_arm.items()}
            ratio = medians["A"] / medians["B"]
            per_round.append({
                "round": round_index,
                "a_median_ms": round(medians["A"], 4),
                "b_median_ms": round(medians["B"], 4),
                "ratio": round(ratio, 5),
            })

    ratios = [entry["ratio"] for entry in per_round]
    mean, low, high = mean_ci(ratios)

    def describe(values: list[float]) -> dict:
        ordered = sorted(values)
        return {
            "samples": len(values),
            "median_ms": round(statistics.median(values), 4),
            "mean_ms": round(statistics.fmean(values), 4),
            "stdev_ms": round(statistics.stdev(values), 4),
            "p25_ms": round(ordered[len(ordered) // 4], 4),
            "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 4),
            "min_ms": round(min(values), 4),
            "max_ms": round(max(values), 4),
        }

    summary = {
        "harness": str(exe),
        "protocol": {
            "rounds": ROUNDS,
            "steps_per_process": STEPS,
            "warmup_per_process": WARMUP,
            "measured_per_process": STEPS - WARMUP,
            "order": ["A", "B", "B", "A"],
            "arm_a": "generic attention forward + production two-pass GQA backward",
            "arm_b": "row-packed attention forward + production two-pass GQA backward",
            "processes": "one arm per process, strictly serial",
            "wmma_mode": "off",
            "sample_file": str(samples_path),
        },
        "per_round": per_round,
        "paired_ratios": ratios,
        "mean_paired_ratio": round(mean, 4),
        "geomean_paired_ratio": round(statistics.geometric_mean(ratios), 4),
        "ci95": {
            "low": round(low, 4),
            "high": round(high, 4),
            "df": len(ratios) - 1,
            "t_critical": t_critical(len(ratios) - 1),
        },
        "ratio_of_medians": round(
            statistics.median(arm_samples["A"]) / statistics.median(arm_samples["B"]), 4
        ),
        "arm_a": describe(arm_samples["A"]),
        "arm_b": describe(arm_samples["B"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    (RAW / "abba-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
