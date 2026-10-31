#!/usr/bin/env python3
"""Rank the kernels of a Hierarchos step by GPU time.

The runtime emits, for every submission, one ``hierarchos_vulkan_kernel_profile
device=...`` line followed by per-category and per-shader lines:

    hierarchos_vulkan_kernel_profile device="AMD Radeon Graphics" dispatches=6912 gpu_ms=87.215080 timestamp_period_ns=10.000000
    hierarchos_vulkan_kernel_profile_category category=normalization.layer-norm dispatches=536 gpu_ms=37.534640 pct=43.037
    hierarchos_vulkan_kernel_profile_shader shader=linear_forward category=projection.linear dispatches=105 gpu_ms=2.209880 pct=2.534

Those lines are written to stderr while ``HIERARCHOS_VULKAN_PROFILE_KERNELS=1``.
Selecting the submissions that carry the training step (the only ones with a
non-trivial dispatch count) and aggregating them gives the true hot-spot ranking
for one step, instead of the cost we assumed from benchmarks of each kernel in
isolation.

Two properties of the emitted data matter for reading the output correctly:

* The per-category lines are complete, so their dispatches sum to the
  submission's dispatch count and their milliseconds sum to its ``gpu_ms``.
  The per-shader lines are the runtime's top-N subset, so they cover most but
  not all of a submission's time; the shader table therefore reports the share
  of the *step* total and does not need to add up to 100%.
* A per-dispatch bracket only covers that dispatch, so ``gpu_ms`` is execution and
  nothing else. The ``_span`` line reports the submission's whole GPU span from a query
  pair that is written whether or not the dispatch around it was sampled, and the
  difference between the two is time no kernel owns: barrier flush, launch latency and
  idle between dispatches.
* Dispatches are *sampled*: every ``sample_stride``-th dispatch of each shader is
  timestamped and the first one always is, so no shader drops out of the ranking. Each
  shader's ``gpu_ms`` is then its sampled mean scaled by its exact dispatch count, and
  the ``samples`` column is the sample size that estimate rests on. Set
  ``HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1`` for exact timing, at the cost of
  distorting the step: on this machine stride 1 adds 82% to the wall clock and
  stride 17 adds 8%.
* The ``_placement`` lines split each shader's dispatches by whether they were
  recorded directly after a barrier (``after_barrier``) or back to back with another
  dispatch (``after_dispatch``). Both populations on a line share the same ``grid``
  and therefore the same grid, push constants and binding count, so the comparison is
  of placement and not of workload. A kernel that costs the same both ways is paying
  for its own work; one that costs far more cold is carrying the barrier it follows.
  Small sample sizes still make single lines noisy, and the counts are printed for
  that reason.
* Submissions with no dispatches in between are pool cycling, not steps.

Usage:
    HIERARCHOS_VULKAN_PROFILE_KERNELS=1 ./target/debug/training_step_probe --steps 3 2> profile.log
    python validation/rank_kernel_profile.py profile.log [--min-dispatches 512] [--top 25]
"""

import re
import sys

DEVICE = re.compile(r"hierarchos_vulkan_kernel_profile device=.* dispatches=(\d+) gpu_ms=([\d.]+)")
CATEGORY = re.compile(
    r"hierarchos_vulkan_kernel_profile_category category=(\S+) "
    r"dispatches=(\d+) gpu_ms=([\d.]+) pct=([\d.]+)"
)
SHADER = re.compile(
    r"hierarchos_vulkan_kernel_profile_shader shader=(\S+) category=(\S+) "
    r"dispatches=(\d+) gpu_ms=([\d.]+) pct=([\d.]+)"
)
SPAN = re.compile(
    r"hierarchos_vulkan_kernel_profile_span span_ms=([\d.]+) dispatch_ms=([\d.]+) "
    r"unattributed_ms=([\d.]+) unattributed_pct=([\d.]+) dispatches=(\d+)"
)
PLACEMENT = re.compile(
    r"hierarchos_vulkan_kernel_profile_placement shader=(\S+) category=(\S+) grid=(\S+) "
    r"after_barrier_dispatches=(\d+) after_barrier_us_each=([\d.]+) "
    r"after_dispatch_dispatches=(\d+) after_dispatch_us_each=([\d.]+)"
)
SHADER_SAMPLES = re.compile(r"samples=(\d+)\s*$")
# Mirrors KERNEL_PROFILE_TOP_SHADERS in src/vulkan.rs; only used for the note
# that explains why the shader table cannot cover the whole step.
KERNEL_PROFILE_TOP_SHADERS = 24


def parse(path):
    """Return one record per submission: (dispatches, gpu_ms, categories, shaders)."""
    submissions = []
    current = None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            device = DEVICE.search(line)
            if device:
                current = {
                    "dispatches": int(device.group(1)),
                    "gpu_ms": float(device.group(2)),
                    "categories": [],
                    "shaders": [],
                    "span": None,
                    "placements": [],
                }
                submissions.append(current)
                continue
            if current is None:
                continue
            category = CATEGORY.search(line)
            if category:
                current["categories"].append(
                    (category.group(1), int(category.group(2)), float(category.group(3)))
                )
                continue
            shader = SHADER.search(line)
            if shader:
                samples = SHADER_SAMPLES.search(line)
                current["shaders"].append(
                    (
                        shader.group(1),
                        shader.group(2),
                        int(shader.group(3)),
                        float(shader.group(4)),
                        int(samples.group(1)) if samples else int(shader.group(3)),
                    )
                )
                continue
            span = SPAN.search(line)
            if span:
                current["span"] = (float(span.group(1)), float(span.group(2)), float(span.group(3)))
                continue
            placement = PLACEMENT.search(line)
            if placement:
                current["placements"].append(
                    (
                        placement.group(1),
                        placement.group(2),
                        placement.group(3),
                        int(placement.group(4)),
                        float(placement.group(5)),
                        int(placement.group(6)),
                        float(placement.group(7)),
                    )
                )
    return submissions


def aggregate_categories(steps):
    totals = {}
    for step in steps:
        for name, dispatches, ms in step["categories"]:
            entry = totals.setdefault(name, [0, 0.0])
            entry[0] += dispatches
            entry[1] += ms
    return totals


def aggregate_shaders(steps):
    totals = {}
    for step in steps:
        for name, category, dispatches, ms, samples in step["shaders"]:
            entry = totals.setdefault(name, [category, 0, 0.0, 0])
            entry[1] += dispatches
            entry[2] += ms
            entry[3] += samples
    return totals


def aggregate_placements(steps):
    """Sum the cold/warm counts and microseconds per (shader, grid) pair.

    The grid is part of the key because the runtime only prints a placement line when
    both populations share a shape, and keeping it here means the report cannot merge
    two different shapes back into one misleading row.
    """
    totals = {}
    for step in steps:
        for name, category, grid, cold_n, cold_us, warm_n, warm_us in step["placements"]:
            entry = totals.setdefault((name, grid), [category, 0, 0.0, 0, 0.0])
            entry[1] += cold_n
            entry[2] += cold_us * cold_n
            entry[3] += warm_n
            entry[4] += warm_us * warm_n
    return totals


def report(path, steps, top):
    count = len(steps)
    total_ms = sum(step["gpu_ms"] for step in steps)
    total_dispatches = sum(step["dispatches"] for step in steps)
    print(
        f"{path}: {count} training step(s), {total_dispatches // count} dispatches and "
        f"{total_ms / count:.3f} ms GPU per step"
    )

    spans = [step["span"] for step in steps if step["span"] is not None]
    if spans:
        span_ms = sum(entry[0] for entry in spans) / len(spans)
        dispatch_ms = sum(entry[1] for entry in spans) / len(spans)
        gap_ms = sum(entry[2] for entry in spans) / len(spans)
        print(
            f"GPU span {span_ms:.3f} ms/step = {dispatch_ms:.3f} ms of dispatch brackets "
            f"+ {gap_ms:.3f} ms ({gap_ms * 100.0 / span_ms:.1f}%) that no bracket covers"
        )
        print(
            "  that unattributed share is barrier flush, launch latency and idle between "
            "dispatches;\n  host time is wall clock minus the span, so `gpu_ms` alone "
            "understates a step"
        )

    categories = aggregate_categories(steps)
    covered_ms = sum(values[1] for values in categories.values())
    print()
    print(f"{'category':34} {'dispatches':>10} {'gpu_ms/step':>12} {'us/disp':>9} {'pct':>7}")
    for name, (dispatches, ms) in sorted(categories.items(), key=lambda kv: -kv[1][1]):
        per_dispatch = ms * 1000.0 / dispatches if dispatches else 0.0
        print(
            f"{name:34} {dispatches // count:>10} {ms / count:>12.3f} {per_dispatch:>9.1f} "
            f"{ms * 100.0 / total_ms:>6.2f}%"
        )
    print(
        f"{'':34} {'':>10} {covered_ms / count:>12.3f} {'':>9} "
        f"{covered_ms * 100.0 / total_ms:>6.2f}%  (category lines)"
    )

    shaders = aggregate_shaders(steps)
    ranked = sorted(shaders.items(), key=lambda kv: -kv[1][2])
    shown = ranked[:top]
    all_ms = sum(values[2] for _, values in ranked)
    shown_ms = sum(values[2] for _, values in shown)
    print()
    print(
        f"{'shader':64} {'category':26} {'disp':>6} {'gpu_ms/step':>12} "
        f"{'us/disp':>9} {'pct':>7} {'samples':>8}"
    )
    for name, (category, dispatches, ms, samples) in shown:
        per_dispatch = ms * 1000.0 / dispatches if dispatches else 0.0
        print(
            f"{name:64} {category:26} {dispatches // count:>6} {ms / count:>12.3f} "
            f"{per_dispatch:>9.1f} {ms * 100.0 / total_ms:>6.2f}% "
            f"{samples // max(count, 1):>8}"
        )
    print(
        f"listed {len(shown)} shader(s) covering {shown_ms * 100.0 / total_ms:.2f}% of the step; "
        f"all {len(shaders)} printed shader(s) cover {all_ms * 100.0 / total_ms:.2f}% "
        f"(the runtime prints the top {KERNEL_PROFILE_TOP_SHADERS} per submission)"
    )

    placements = aggregate_placements(steps)
    if placements:
        ranked = sorted(
            placements.items(), key=lambda kv: -(kv[1][2] / kv[1][1] + kv[1][4] / kv[1][3])
        )
        print()
        print(
            "placement: does a dispatch pay for itself, or for the barrier before it?"
        )
        print(
            "  both populations share a grid and therefore the same shape and push constants"
        )
        print(
            f"{'shader':52} {'grid':>10} {'cold us':>9} {'warm us':>9} {'cold-warm':>10} "
            f"{'cold n':>7} {'warm n':>7}"
        )
        for (name, grid), (_, cold_n, cold_total, warm_n, warm_total) in ranked[:top]:
            cold_each = cold_total / cold_n
            warm_each = warm_total / warm_n
            print(
                f"{name:52} {grid:>10} {cold_each:>9.3f} {warm_each:>9.3f} "
                f"{cold_each - warm_each:>10.3f} {cold_n // count:>7} {warm_n // count:>7}"
            )


def main(argv):
    if not argv:
        print("usage: rank_kernel_profile.py <profile.log> [--min-dispatches N] [--top N]", file=sys.stderr)
        return 2
    path = argv[0]
    min_dispatches = 512
    top = 25
    rest = argv[1:]
    while rest:
        flag = rest.pop(0)
        if flag == "--min-dispatches":
            min_dispatches = int(rest.pop(0))
        elif flag == "--top":
            top = int(rest.pop(0))
        else:
            print(f"unknown argument {flag}", file=sys.stderr)
            return 2

    try:
        submissions = parse(path)
    except OSError as err:
        print(f"cannot read {path}: {err.strerror}", file=sys.stderr)
        return 1
    if not submissions:
        print(f"no kernel profile lines in {path}", file=sys.stderr)
        return 1
    steps = [step for step in submissions if step["dispatches"] >= min_dispatches]
    if not steps:
        print(
            f"no submission with >= {min_dispatches} dispatches in {path}; "
            f"largest was {max(step['dispatches'] for step in submissions)}",
            file=sys.stderr,
        )
        return 1

    report(path, steps, top)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
