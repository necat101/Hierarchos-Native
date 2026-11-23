#!/usr/bin/env python3
"""Phase LXXXVII counterbalanced campaign driver.

This is the Phase LXXXVI driver's protocol with the smallest possible surface
change, so the schedule and the statistics stay single-sourced:

* the schedule comes from `phase_lxxxvi_wmma_optimization/run_balanced.py`
  (`schedule()`), i.e. the predeclared ABBA / BAAB quad order, unchanged;
* the interval comes from `validation/abba_stats.py`, unchanged;
* the GPU-timestamp parsing comes from
  `phase_lxxxi_evidence/parse_gpu_profile.py`, unchanged, via that same driver.

Two additions, both measurement-only and both recorded per run in
`SCHEDULE.json`:

* `--shapes MxNxK,...` forwards an explicit geometry list, so a boundary sweep
  is measured by the same code path as the protected shapes;
* `--skip-reference` forwards `--skip-reference`, so a wide sweep does not
  spend most of its wall clock in the host FP32 reference GEMM. Runs that
  report a drift are the ones without it; a `--skip-reference` sweep makes no
  numerical claim and says so.

Both arms run the **same executable**, differing only in
`HIERARCHOS_WMMA_FULL_FP16_VARIANT`, which is the repository's established
comparison mechanism. That is a tighter control than Phase LXXXVI's separate
executables: the host code, allocator, dispatch path and binary are identical,
so a ratio can only come from the SPIR-V module.

Usage:
  run_arm_campaign.py --out DIR --exe EXE --base VARIANT --cand VARIANT \
      [--pairs 24] [--iterations 16] [--rounds 1] [--shapes LIST] \
      [--skip-reference]
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
REPO_ROOT = HERE.parents[2]
LXXXI = REPO_ROOT / "phase_lxxxi_evidence"
LXXXVI = VALIDATION / "phase_lxxxvi_wmma_optimization"
sys.path.insert(0, str(LXXXI))
sys.path.insert(0, str(LXXXVI))
sys.path.insert(0, str(VALIDATION))

from parse_gpu_profile import blocks, parse_stream, steady_state_gpu_ms  # noqa: E402
from run_balanced import ARM_LABEL, WARMUP, schedule  # noqa: E402

import abba_stats  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str]) -> dict:
    options = {
        "pairs": 24,
        "iterations": 16,
        "rounds": 1,
        "shapes": None,
        "skip_reference": False,
    }
    positional: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--skip-reference":
            options["skip_reference"] = True
            index += 1
        elif arg in ("--out", "--exe", "--base", "--cand", "--pairs", "--iterations", "--rounds", "--shapes"):
            value = argv[index + 1]
            key = arg[2:].replace("-", "_")
            if key in ("pairs", "iterations", "rounds"):
                options[key] = int(value)
            else:
                options[key] = value
            index += 2
        else:
            positional.append(arg)
            index += 1
    for key in ("out", "exe", "base", "cand"):
        if key not in options:
            raise SystemExit(f"missing --{key}")
    return options


def run_arm(
    exe: Path,
    variant: str,
    stem: Path,
    iterations: int,
    rounds: int,
    shapes: str | None,
    skip_reference: bool,
) -> int:
    env = dict(os.environ)
    env["HIERARCHOS_VULKAN_PROFILE_KERNELS"] = "1"
    env["HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE"] = "1"
    if variant:
        env["HIERARCHOS_WMMA_FULL_FP16_VARIANT"] = variant
    else:
        env.pop("HIERARCHOS_WMMA_FULL_FP16_VARIANT", None)
    command = [
        str(exe),
        "--wmma-mode",
        "full-fp16",
        "--rounds",
        str(rounds),
        "--iterations",
        str(iterations),
        "--json",
    ]
    if shapes:
        command.extend(["--shapes", shapes])
    if skip_reference:
        command.append("--skip-reference")
    with open(stem.with_suffix(".json"), "wb") as out, open(
        stem.parent / (stem.name + "_err.txt"), "wb"
    ) as err:
        completed = subprocess.run(command, stdout=out, stderr=err, env=env, check=False)
    return completed.returncode


def convert(stem: Path, shader: str, iterations: int, shape_names: tuple[str, ...]) -> dict:
    lines = (stem.parent / (stem.name + "_err.txt")).read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    batches, period = parse_stream(lines)
    # The preserved Phase LXXXI parser segments positionally against a shape
    # label tuple. A `--shapes` run must hand it the tuple that was actually
    # requested, or every block is labelled with the built-in six-shape names.
    # The segmentation check (equal block count, identical shader signature per
    # block) is unchanged, so a mis-sized list still raises instead of
    # producing a measurement of the wrong geometry.
    per_shape = blocks(batches, shapes=shape_names) if shape_names else blocks(batches)
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
    options = parse_args(sys.argv[1:])
    out = Path(options["out"])
    exe = Path(options["exe"]).resolve()
    base_variant = options["base"]
    cand_variant = options["cand"]
    pairs = options["pairs"]
    iterations = options["iterations"]
    rounds = options["rounds"]
    shapes = options["shapes"]
    skip_reference = options["skip_reference"]
    shape_names: tuple[str, ...] = ()
    if shapes:
        shape_names = tuple(item.strip() for item in shapes.split(",") if item.strip())

    order = schedule(pairs)
    if not exe.is_file():
        raise SystemExit(f"missing executable: {exe}")

    out.mkdir(parents=True, exist_ok=True)
    exe_hash = sha256(exe)
    index = out / "index.tsv"
    index.write_text(
        "pair\tarm\trc\texe_sha256\trun\tvariant\texe\tquad\tschedule\tposition\tpipeline\n",
        encoding="utf-8",
    )

    for pair, arm, position, quad, pattern in order:
        variant = base_variant if arm == "base" else cand_variant
        stem = out / f"p{pair}_{arm}"
        code = run_arm(
            exe, variant, stem, iterations, rounds, shapes, skip_reference
        )
        record = {
            "pair": pair,
            "arm": arm,
            "rc": code,
            "sha256": exe_hash,
            "variant": variant,
            "exe": str(exe),
            "quad": quad,
            "schedule": pattern,
            "position": position,
        }
        if code != 0:
            (out / f"p{pair}_{arm}_meta.json").write_text(
                json.dumps(record, indent=1), encoding="utf-8"
            )
            print(f"RUN FAILED pair={pair} arm={arm}", file=sys.stderr)
            continue
        report = json.loads((stem.with_suffix(".json")).read_text(encoding="utf-8"))
        pipeline = report.get("full_fp16_pipeline", "")
        shader = pipeline.split("/")[0]
        if not shader.startswith("wmma_gemm_"):
            raise SystemExit(f"unexpected cooperative shader {shader!r} in {pipeline!r}")
        record["shader"] = shader
        record["pipeline"] = pipeline
        (out / f"p{pair}_{arm}_meta.json").write_text(
            json.dumps(record, indent=1), encoding="utf-8"
        )
        converted = convert(stem, shader, iterations, shape_names)
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
                f"{pair}\t{arm}\t{code}\t{exe_hash}\t{stem.name}\t{variant}\t{exe}\t"
                f"{quad}\t{pattern}\t{position}\t{pipeline}\n"
            )
        print(f"pair {pair:>2} {pattern} position {position} {arm} done", flush=True)

    (out / "SCHEDULE.json").write_text(
        json.dumps(
            {
                "patterns": [
                    "ABBA" if quad % 2 == 1 else "BAAB" for quad in range(1, pairs // 2 + 1)
                ],
                "order": [
                    {"pair": p, "arm": a, "position": pos, "quad": q, "pattern": pat}
                    for p, a, pos, q, pat in order
                ],
                "pairs": pairs,
                "iterations": iterations,
                "rounds": rounds,
                "warmup_dropped": WARMUP,
                "shapes": shapes or "<built-in six>",
                "skip_reference": skip_reference,
                "base_variant": base_variant,
                "cand_variant": cand_variant,
                "exe": str(exe),
                "exe_sha256": exe_hash,
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    summary = abba_stats.summarize(str(out), ARM_LABEL)
    print(
        f"\n{'shape':<16}{'pairs':>6}{'base ms':>10}{'cand ms':>10}{'ratio':>10}"
        f"{'95% t CI':>24}  verdict"
    )
    report = {}
    for shape, row in summary.items():
        low, high = row["ci95_low"], row["ci95_high"]
        verdict = (
            "FASTER (lower > 1.0)"
            if low >= 1.0
            else "SLOWER (CI excludes 1.0)"
            if high < 1.0
            else "INCONCLUSIVE"
        )
        print(
            f"{shape:<16}{row['pairs']:>6}{row['median_reference_ms']:>10.4f}"
            f"{row['median_candidate_ms']:>10.4f}{row['mean_ratio']:>10.4f}"
            f"  [{low:.4f}, {high:.4f}]  {verdict}"
        )
        row["verdict"] = verdict
        report[shape] = row
    (out / "SUMMARY.json").write_text(
        json.dumps(
            {
                "base_variant": base_variant,
                "cand_variant": cand_variant,
                "shapes": shapes or "<built-in six>",
                "skip_reference": skip_reference,
                "ratio_convention": "reference_ms / candidate_ms; > 1.0 means candidate faster",
                "interval_rule": "two-sided 95 % Student-t over per-pair ratios (abba_stats.py)",
                "results": report,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nCAMPAIGN COMPLETE: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
