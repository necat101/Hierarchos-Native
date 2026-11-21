#!/usr/bin/env python3
"""Phase LXXXI serial paired-ABBA runner over GPU timestamp brackets.

Each run is one process of a preserved `wmma_bench` executable with
`HIERARCHOS_VULKAN_PROFILE_KERNELS=1` and `HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1`,
which makes the Vulkan layer bracket every compute dispatch with two timestamp
queries and print the per-shader GPU time per submitted batch on stderr.

The driver writes, for every run, the untouched stderr and JSON, plus a
`<run>_samples.jsonl` in exactly the layout `hierarchos-vulkan/validation/abba_stats.py`
consumes, so the canonical statistics implementation is reused without
modification. Samples are GPU seconds.

Usage:
  run_lxxiii_abba.py <out-dir> <base-exe> <base-variant> <cand-exe> \
                     <cand-variant> [pairs] [iterations] [rounds]
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from parse_gpu_profile import (  # noqa: E402
    COOP_SHADERS,
    ProfileError,
    blocks,
    parse_stream,
    steady_state_gpu_ms,
)

ARM_LABEL = "wmma-full-f16acc"
WARMUP = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    """Turn one run's stderr into abba_stats sample rows keyed by shape."""
    lines = (stem.parent / (stem.name + "_err.txt")).read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    batches, period = parse_stream(lines)
    per_shape = blocks(batches)
    rows = {}
    for shape, block in per_shape.items():
        gpu_ms = steady_state_gpu_ms(block, shader, warmup=WARMUP)
        if len(gpu_ms) != iterations:
            raise ProfileError(
                f"{stem.name} {shape}: {len(gpu_ms)} steady-state dispatches, "
                f"expected {iterations}"
            )
        rows[shape] = [value / 1.0e3 for value in gpu_ms]
    return {"rows": rows, "period_ns": period}


def main() -> int:
    out = Path(sys.argv[1])
    base = Path(sys.argv[2]).resolve()
    base_variant = sys.argv[3]
    cand = Path(sys.argv[4]).resolve()
    cand_variant = sys.argv[5]
    pairs = int(sys.argv[6]) if len(sys.argv) > 6 else 24
    iterations = int(sys.argv[7]) if len(sys.argv) > 7 else 16
    rounds = int(sys.argv[8]) if len(sys.argv) > 8 else 1

    for exe in (base, cand):
        if not exe.is_file():
            raise SystemExit(f"missing executable: {exe}")

    out.mkdir(parents=True, exist_ok=True)
    hashes = {base: sha256(base), cand: sha256(cand)}
    index = out / "index.tsv"
    index.write_text("pair\tarm\trc\texe_sha256\trun\tvariant\texe\n", encoding="utf-8")

    for pair in range(1, pairs + 1):
        order = ("base", "cand") if pair % 2 == 1 else ("cand", "base")
        for arm in order:
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
            }
            (out / f"p{pair}_{arm}_meta.json").write_text(
                json.dumps(record, indent=1), encoding="utf-8"
            )
            if code != 0:
                print(f"RUN FAILED pair={pair} arm={arm}", file=sys.stderr)
                continue
            report = json.loads(
                (stem.with_suffix(".json")).read_text(encoding="utf-8")
            )
            shader = report.get("full_fp16_pipeline", "").split("/")[0]
            if not shader:
                # The historical Main binary has no selector and reports no
                # pipeline identity; its cooperative module is the split8 one.
                shader = "wmma_gemm_f16_f16acc"
            if shader not in COOP_SHADERS:
                raise SystemExit(f"unexpected cooperative shader {shader!r}")
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
                    f"{variant or '<unset>'}\t{exe}\n"
                )
        print(f"pair {pair} done", flush=True)
    print(f"ABBA COMPLETE: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
