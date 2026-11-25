"""Median GPU time per (geometry, arm) from a paired two-pass timing log.

The test prints one marker line per submission and the timestamp profiler prints
the GPU total immediately after it, so the pairing is positional. `-sample`
markers are measurements; `-warmup` markers are not. A log with no profiler block
yields no number rather than an invented one.
"""
import re
import statistics
import sys

MARKER = re.compile(
    r"^\[(?:gqa|two-pass)-bench(?P<kind>-\w+)\] geometry=(?P<geometry>\S+) arm=(?P<arm>\S+)"
)
GPU = re.compile(r"hierarchos_vulkan_kernel_profile device=.*gpu_ms=(?P<gpu>[0-9.]+)")


def main(paths):
    for path in paths:
        current = None
        samples = {}
        order = []
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                marker = MARKER.match(line)
                if marker:
                    current = (marker["geometry"], marker["arm"], marker["kind"])
                    if current[2] == "-sample" and current[:2] not in order:
                        order.append(current[:2])
                    continue
                gpu = GPU.search(line)
                if gpu and current is not None:
                    if current[2] == "-sample":
                        samples.setdefault(current[:2], []).append(float(gpu["gpu"]))
                    current = None
        print(f"# {path}")
        for key in order:
            values = samples.get(key, [])
            if not values:
                print(f"{key[0]}\t{key[1]}\tno_samples")
                continue
            values.sort()
            print(
                f"{key[0]}\t{key[1]}\tn={len(values)}"
                f"\tmedian_ms={statistics.median(values):.6f}"
                f"\tmin_ms={values[0]:.6f}\tmax_ms={values[-1]:.6f}"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
