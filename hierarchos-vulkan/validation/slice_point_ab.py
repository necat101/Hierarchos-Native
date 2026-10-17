"""A/B the watchdog slice point on this adapter, and hash the trajectories it writes.

The Gen9 policy retires a submission when the time one is measured to take reaches
a target (see `watchdog_slice_budget_after_retirement` in `src/vulkan.rs`), so two
things have to be shown on the machine: that the measured bound costs less than the
fixed dispatch count it replaced, and that no arm changes a single number.

Timing is interleaved -- one run of every arm per round, medians of the rounds --
because this laptop drifts several percent between back-to-back sessions, and a
block-per-arm comparison would charge that drift to whichever arm ran second.
Trajectory runs use the golden's own three cases and assert that every arm produces
the same dump byte for byte, which is the exactness half of the claim.

    python validation/slice_point_ab.py                       # timing only
    python validation/slice_point_ab.py --rounds 5            # more rounds
    python validation/slice_point_ab.py --trajectory          # + hashes (default)
    python validation/slice_point_ab.py --no-trajectory
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
GOLDEN = CRATE / "validation" / "kernel_golden" / "training_state.json"
SCRATCH = CRATE / "target" / "slice_point_ab"

# Each arm is the environment the probe runs under; the shipped arm is the empty
# one, so an accidental default change is visible as a moved row rather than as a
# row that quietly stops being measured.
ARMS = {
    "legacy": {"HIERARCHOS_VULKAN_WATCHDOG_SUBMISSIONS": "0"},
    "pinned1024": {"HIERARCHOS_VULKAN_WATCHDOG_SLICE_DISPATCHES": "1024"},
    "pinned4096": {"HIERARCHOS_VULKAN_WATCHDOG_SLICE_DISPATCHES": "4096"},
    "pinned8192": {"HIERARCHOS_VULKAN_WATCHDOG_SLICE_DISPATCHES": "8192"},
    "shipped": {},
}
TIMING_ARGS = ["--batch", "2", "--tokens", "8"]


def probe_env(arm):
    env = dict(os.environ)
    for key in ("HIERARCHOS_VULKAN_WATCHDOG_SUBMISSIONS",
                "HIERARCHOS_VULKAN_WATCHDOG_SLICE_DISPATCHES"):
        env.pop(key, None)
    env.update(ARMS[arm])
    return env


def run_probe(probe, arm, args, tag):
    directory = SCRATCH / tag
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    command = [str(probe), "--model-dir", str(directory / "model"), *args]
    result = subprocess.run(command, cwd=CRATE, text=True, capture_output=True,
                            env=probe_env(arm))
    if result.returncode:
        raise RuntimeError(f"{arm} {tag} probe exit {result.returncode}: {result.stderr[-2000:]}")
    return json.loads(result.stdout[result.stdout.index("{"):])


def time_arms(probe, arms, rounds, steps):
    samples = {arm: [] for arm in arms}
    dispatches = set()
    for round_index in range(1, rounds + 1):
        for arm in arms:
            output = run_probe(probe, arm, [*TIMING_ARGS, "--steps", str(steps)],
                               f"{arm}-r{round_index}")
            samples[arm].append(output["median_step_ms"])
            dispatches.add(output["dispatch_count"])
    return samples, dispatches


def trajectory_hashes(probe, arms):
    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"]
    hashes = {}
    for arm in arms:
        for index, case in enumerate(cases):
            snapshot = SCRATCH / f"traj_{arm}_{index}.json"
            run_probe(probe, arm, [*case["args"], "--state-output", str(snapshot)],
                      f"traj_{arm}_{index}")
            digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            hashes.setdefault(case["name"], {})[arm] = digest
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", type=Path, default=CRATE / "target" / "debug" /
                        ("training_step_probe.exe" if os.name == "nt" else "training_step_probe"))
    parser.add_argument("--release", action="store_true",
                        help="use target/release instead of --probe's default")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--arms", default="legacy,pinned1024,shipped,pinned4096")
    parser.add_argument("--trajectory", dest="trajectory", action="store_true", default=True)
    parser.add_argument("--no-trajectory", dest="trajectory", action="store_false")
    args = parser.parse_args()

    probe = args.probe
    if args.release:
        probe = CRATE / "target" / "release" / probe.name
    if not probe.exists():
        raise SystemExit(f"{probe} is missing; build it first (do not go looking for it)")
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; known: {', '.join(ARMS)}")
    SCRATCH.mkdir(parents=True, exist_ok=True)

    samples, dispatches = time_arms(probe, arms, args.rounds, args.steps)
    baseline = statistics.median(samples["legacy"]) if "legacy" in samples else None
    print(f"probe: {probe}")
    print(f"fixture: {' '.join([*TIMING_ARGS, '--steps', str(args.steps)])} "
          f"({','.join(str(value) for value in sorted(dispatches))} dispatches), "
          f"{args.rounds} interleaved round(s)\n")
    header = f"{'arm':<11}{'per-round ms':>34}{'median ms':>11}{'delta':>10}"
    print(header)
    for arm in arms:
        median = statistics.median(samples[arm])
        delta = "" if baseline is None else f"{(median / baseline - 1) * 100:+.1f}%"
        per_round = " ".join(f"{value:.1f}" for value in samples[arm])
        print(f"{arm:<11}{per_round:>34}{median:>11.1f}{delta:>10}")

    if not args.trajectory:
        return 0
    hashes = trajectory_hashes(probe, arms)
    print("\ntrajectory dumps by arm (sha256, first 16, every arm agreeing):")
    disagreements = []
    for case, per_arm in hashes.items():
        values = sorted(set(per_arm.values()))
        agrees = len(values) == 1
        if not agrees:
            disagreements.append(case)
        print(f"{case:<20}{values[0][:16]}  "
              f"{f'{len(per_arm)} arm(s)' if agrees else 'DISAGREES across arms'}")
    if disagreements:
        for case in disagreements:
            for arm, digest in sorted(hashes[case].items()):
                print(f"  {case} {arm}: {digest}")
        print(f"\nFAIL: {len(disagreements)} case(s) differ between arms")
        return 1
    print(f"\nPASS: all {len(arms)} arms write byte-identical trajectories "
          f"for {len(hashes)} case(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
