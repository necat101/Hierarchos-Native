#!/usr/bin/env python3
"""Phase LXXXVI counterbalanced (ABBA/BAAB) campaign driver.

This is the Phase LXXVI driver (`phase_lxxxi_evidence/run_lxxxi_balanced.py`)
verbatim in its schedule, its measurement environment and its statistics, with
two changes and no others:

1. It imports `parse_gpu_profile` from the preserved Phase LXXXI evidence
   directory rather than carrying a second copy of the profile parser, so the
   GPU-timestamp parsing rule stays single-sourced.
2. It does not pin the cooperative shader name against that module's historical
   `COOP_SHADERS` tuple. A Phase LXXXVI candidate is by definition a module that
   did not exist in Phase LXXXI, so the check would reject the candidate it is
   supposed to measure. The module name is still read from the benchmark
   report's own `full_fp16_pipeline` identity and recorded per run; the runner
   only requires that both arms report a `wmma_gemm_*` module.

Protocol, unchanged: strictly serial one-process-at-a-time execution, the same
`HIERARCHOS_VULKAN_PROFILE_KERNELS=1` / `PROFILE_SAMPLE_STRIDE=1` GPU timestamp
stream, `--rounds 1 --iterations 16`, three warm-up dispatches dropped per
shape, and a predeclared ABBA, BAAB, ABBA, ... quad schedule.

Usage:
  run_balanced.py <out-dir> <base-exe> <base-variant> <cand-exe> <cand-variant> \
                  [pairs] [iterations] [rounds] [--dry-run]
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VALIDATION = HERE.parent
# hierarchos-vulkan/validation/<phase> -> hierarchos-vulkan -> repository root
REPO_ROOT = HERE.parents[2]
LXXXI_EVIDENCE = REPO_ROOT / "phase_lxxxi_evidence"
sys.path.insert(0, str(LXXXI_EVIDENCE))

from parse_gpu_profile import blocks, parse_stream, steady_state_gpu_ms  # noqa: E402

ARM_LABEL = "wmma-full-f16acc"
WARMUP = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schedule(pairs: int) -> list:
    """Predeclared counterbalanced order: ABBA, BAAB, ABBA, BAAB, ..."""
    if pairs % 2 != 0:
        raise SystemExit("pairs must be even: a quad is two paired blocks")
    quads = pairs // 2
    if quads % 2 != 0:
        raise SystemExit("quads must be even so ABBA and BAAB are balanced")
    out = []
    for quad in range(1, quads + 1):
        pattern = "ABBA" if quad % 2 == 1 else "BAAB"
        for offset, label in enumerate(pattern):
            pair = 2 * (quad - 1) + 1 + offset // 2
            arm = "base" if label == "A" else "cand"
            out.append((pair, arm, offset + 1, quad, pattern))
    return out


def run_arm(exe: Path, variant: str, stem: Path, iterations: int, rounds: int) -> int:
    env = dict(os.environ)
    env["HIERARCHOS_VULKAN_PROFILE_KERNELS"] = "1"
    env["HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE"] = "1"
    if variant:
        env["HIERARCHOS_WMMA_FULL_FP16_VARIANT"] = variant
    else:
        env.pop("HIERARCHOS_WMMA_FULL_FP16_VARIANT", None)
    with open(stem.with_suffix(".json"), "wb") as out, open(
        stem.parent / (stem.name + "_err.txt"), "wb"
    ) as err:
        completed = subprocess.run(
            [
                str(exe),
                "--wmma-mode",
                "full-fp16",
                "--rounds",
                str(rounds),
                "--iterations",
                str(iterations),
                "--json",
            ],
            stdout=out,
            stderr=err,
            env=env,
            check=False,
        )
    return completed.returncode


def convert(stem: Path, shader: str, iterations: int) -> dict:
    lines = (stem.parent / (stem.name + "_err.txt")).read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    batches, period = parse_stream(lines)
    per_shape = blocks(batches)
    rows = {}
    for shape, block in per_shape.items():
        gpu_ms = steady_state_gpu_ms(block, shader, warmup=WARMUP)
        if len(gpu_ms) != iterations:
            raise SystemExit(
                f"{stem.name} {shape}: {len(gpu_ms)} steady-state dispatches, "
                f"expected {iterations}"
            )
        rows[shape] = [value / 1.0e3 for value in gpu_ms]
    return {"rows": rows, "period_ns": period}


def main() -> int:
    args = [arg for arg in sys.argv[1:] if arg != "--dry-run"]
    dry_run = "--dry-run" in sys.argv[1:]

    out = Path(args[0])
    base = Path(args[1]).resolve()
    base_variant = args[2]
    cand = Path(args[3]).resolve()
    cand_variant = args[4]
    pairs = int(args[5]) if len(args) > 5 else 24
    iterations = int(args[6]) if len(args) > 6 else 16
    rounds = int(args[7]) if len(args) > 7 else 1

    order = schedule(pairs)
    if dry_run:
        for pair, arm, position, quad, pattern in order:
            print(f"pair {pair:>2} quad {quad:>2} {pattern} position {position} arm {arm}")
        counts = {}
        for _pair, arm, position, _quad, _pattern in order:
            counts[(arm, position)] = counts.get((arm, position), 0) + 1
        print("position balance:", dict(sorted(counts.items())))
        return 0

    for exe in (base, cand):
        if not exe.is_file():
            raise SystemExit(f"missing executable: {exe}")

    out.mkdir(parents=True, exist_ok=True)
    hashes = {base: sha256(base), cand: sha256(cand)}
    index = out / "index.tsv"
    index.write_text(
        "pair\tarm\trc\texe_sha256\trun\tvariant\texe\tquad\tschedule\tposition\n",
        encoding="utf-8",
    )

    for pair, arm, position, quad, pattern in order:
        exe = base if arm == "base" else cand
        variant = base_variant if arm == "base" else cand_variant
        stem = out / f"p{pair}_{arm}"
        code = run_arm(exe, variant, stem, iterations, rounds)
        record = {
            "pair": pair,
            "arm": arm,
            "rc": code,
            "sha256": hashes[exe],
            "variant": variant or "<unset>",
            "exe": str(exe),
            "quad": quad,
            "schedule": pattern,
            "position": position,
        }
        (out / f"p{pair}_{arm}_meta.json").write_text(
            json.dumps(record, indent=1), encoding="utf-8"
        )
        if code != 0:
            print(f"RUN FAILED pair={pair} arm={arm}", file=sys.stderr)
            continue
        report = json.loads((stem.with_suffix(".json")).read_text(encoding="utf-8"))
        shader = report.get("full_fp16_pipeline", "").split("/")[0]
        if not shader.startswith("wmma_gemm_"):
            raise SystemExit(f"unexpected cooperative shader {shader!r}")
        record["shader"] = shader
        record["pipeline"] = report.get("full_fp16_pipeline", "")
        (out / f"p{pair}_{arm}_meta.json").write_text(
            json.dumps(record, indent=1), encoding="utf-8"
        )
        converted = convert(stem, shader, iterations)
        with open(out / (stem.name + "_samples.jsonl"), "w", encoding="utf-8") as handle:
            for shape, samples in converted["rows"].items():
                handle.write(
                    json.dumps(
                        {
                            "shape": shape,
                            "arm": ARM_LABEL,
                            "round": 0,
                            "samples_s": [round(float(value), 9) for value in samples],
                            "gpu_timestamp": True,
                            "timestamp_period_ns": converted["period_ns"],
                            "shader": shader,
                            "variant": report.get("full_fp16_variant", "n/a"),
                        }
                    )
                    + "\n"
                )
        with open(index, "a", encoding="utf-8") as handle:
            handle.write(
                f"{pair}\t{arm}\t{code}\t{hashes[exe]}\t{stem.name}\t"
                f"{variant or '<unset>'}\t{exe}\t{quad}\t{pattern}\t{position}\n"
            )
        print(f"pair {pair} {pattern} position {position} {arm} done", flush=True)

    (out / "SCHEDULE.json").write_text(
        json.dumps(
            {
                "patterns": [
                    "ABBA" if quad % 2 == 1 else "BAAB"
                    for quad in range(1, pairs // 2 + 1)
                ],
                "order": [
                    {"pair": p, "arm": a, "position": pos, "quad": q, "pattern": pat}
                    for p, a, pos, q, pat in order
                ],
                "pairs": pairs,
                "iterations": iterations,
                "rounds": rounds,
                "warmup_dropped": WARMUP,
                "exe_hashes": {str(k): v for k, v in hashes.items()},
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"ABBA/BAAB CAMPAIGN COMPLETE: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
