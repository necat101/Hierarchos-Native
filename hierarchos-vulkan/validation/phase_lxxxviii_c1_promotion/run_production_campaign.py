#!/usr/bin/env python3
"""Phase LXXXVIII paired campaign: v0.2.5 P3 (baseline) vs the v0.2.6 selector.

Phase LXXXVII compared two *modules* inside one executable by overriding
`HIERARCHOS_WMMA_FULL_FP16_VARIANT`. That mechanism is deliberately gone: the
v0.2.6 production selector picks the staging module from the dispatch geometry,
and there is no environment variable that can force C1 onto a geometry the
contract does not prove. A paired measurement therefore compares two
**executables**:

* `--base-exe` — built from the v0.2.5 release commit (`d0f48c9`), whose
  production route is the original P3 module;
* `--cand-exe` — built from the v0.2.6 tree, whose production route is the
  selector: C1 for geometries inside the alignment/bounds contract, P3
  otherwise.

Both binaries are driven with exactly the same arguments, so the only
difference between the arms is the executable.

Everything that makes the numbers comparable is single-sourced elsewhere and
unchanged:

* the ABBA/BAAB quad schedule comes from
  `phase_lxxxvi_wmma_optimization/run_balanced.py:schedule`;
* the GPU-timestamp parsing comes from `phase_lxxxi_evidence/parse_gpu_profile.py`;
* the corrected interval comes from `validation/abba_stats.py`.

This driver adds one thing to the LXXXVII driver: because the candidate's
staging module is chosen per shape, a run over the built-in six-shape list
produces two different cooperative shaders. The per-shape block is therefore
segmented positionally with a *checked* split — equal batch counts and an
identical per-batch dispatch-count sequence in every block — and the shader
timed inside each block is whichever single cooperative shader that block
actually dispatched. A block that dispatched two cooperative shaders, or none,
raises instead of being averaged.

Usage:
  run_production_campaign.py --out DIR --base-exe EXE --cand-exe EXE \
      [--pairs 24] [--iterations 16] [--rounds 1] [--shapes LIST]
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

from parse_gpu_profile import SHAPE_ORDER, parse_stream  # noqa: E402
from run_balanced import ARM_LABEL, WARMUP, schedule  # noqa: E402

import abba_stats  # noqa: E402

COOPERATIVE_PREFIX = "wmma_gemm_f16_f16acc_twowave_p3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str]) -> dict:
    options = {"pairs": 24, "iterations": 16, "rounds": 1, "shapes": None}
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in ("--out", "--base-exe", "--cand-exe", "--pairs", "--iterations",
                   "--rounds", "--shapes"):
            value = argv[index + 1]
            key = arg[2:].replace("-", "_")
            options[key] = int(value) if key in ("pairs", "iterations", "rounds") else value
            index += 2
        else:
            raise SystemExit(f"unsupported argument {arg!r}")
    for key in ("out", "base_exe", "cand_exe"):
        if key not in options:
            raise SystemExit(f"missing --{key.replace('_', '-')}")
    return options


def run_arm(exe: Path, stem: Path, iterations: int, rounds: int, shapes: str | None) -> int:
    env = dict(os.environ)
    env["HIERARCHOS_VULKAN_PROFILE_KERNELS"] = "1"
    env["HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE"] = "1"
    # Nothing may force a staging module: the selector decides.
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
    with open(stem.with_suffix(".json"), "wb") as out, open(
        stem.parent / (stem.name + "_err.txt"), "wb"
    ) as err:
        completed = subprocess.run(command, stdout=out, stderr=err, env=env, check=False)
    return completed.returncode


def segment(batches: list[dict], shape_names: tuple[str, ...]) -> dict[str, list[dict]]:
    """Split a profile stream into one block per shape, positionally, checked.

    The LXXXI parser demands an identical *shader* signature in every block,
    which a run whose staging module changes per shape cannot satisfy. The
    segmentation itself is still legitimate because every shape contributes the
    same number of batches in the same order; that is what is checked here. The
    shader is then read per block instead of assumed.
    """
    count = len(batches)
    if count % len(shape_names) != 0:
        raise SystemExit(
            f"{count} batches do not divide into {len(shape_names)} shape blocks"
        )
    size = count // len(shape_names)
    blocks = [batches[index * size:(index + 1) * size] for index in range(len(shape_names))]
    reference = [batch["dispatches"] for batch in blocks[0]]
    for shape, block in zip(shape_names[1:], blocks[1:]):
        observed = [batch["dispatches"] for batch in block]
        if observed != reference:
            raise SystemExit(
                f"block {shape} does not mirror the first block's dispatch counts: "
                f"{observed} != {reference}"
            )
    return dict(zip(shape_names, blocks))


def timed_dispatches(block: list[dict], warmup: int) -> tuple[str, list[float]]:
    """The one cooperative shader this block timed, and its steady-state GPU ms."""
    solo = [batch for batch in block if len(batch["shaders"]) == 1]
    names = {batch["shaders"][0][0] for batch in solo}
    cooperative = {name for name in names if name.startswith(COOPERATIVE_PREFIX)}
    if len(cooperative) != 1:
        raise SystemExit(
            f"expected exactly one cooperative shader in a block, saw {sorted(cooperative)}"
        )
    shader = cooperative.pop()
    timings = [
        batch["gpu_ms"]
        for batch in solo
        if batch["shaders"][0][0] == shader
    ]
    if len(timings) <= warmup:
        raise SystemExit(
            f"only {len(timings)} solo dispatches of {shader}; expected more than {warmup}"
        )
    return shader, timings[warmup:]


def main() -> int:
    options = parse_args(sys.argv[1:])
    out = Path(options["out"]).resolve()
    base_exe = Path(options["base_exe"]).resolve()
    cand_exe = Path(options["cand_exe"]).resolve()
    pairs = options["pairs"]
    iterations = options["iterations"]
    rounds = options["rounds"]
    shapes = options["shapes"]
    shape_names: tuple[str, ...] = (
        tuple(item.strip() for item in shapes.split(",") if item.strip())
        if shapes
        else SHAPE_ORDER
    )

    for path in (base_exe, cand_exe):
        if not path.is_file():
            raise SystemExit(f"missing executable: {path}")

    out.mkdir(parents=True, exist_ok=True)
    exe_hashes = {"base": sha256(base_exe), "cand": sha256(cand_exe)}
    index = out / "index.tsv"
    index.write_text(
        "pair\tarm\trc\texe_sha256\trun\tvariant\texe\tquad\tschedule\tposition\t"
        "pipeline\tstaging\n",
        encoding="utf-8",
    )

    for pair, arm, position, quad, pattern in schedule(pairs):
        exe = base_exe if arm == "base" else cand_exe
        stem = out / f"p{pair}_{arm}"
        code = run_arm(exe, stem, iterations, rounds, shapes)
        record = {
            "pair": pair,
            "arm": arm,
            "rc": code,
            "sha256": exe_hashes[arm],
            "exe": str(exe),
            "quad": quad,
            "schedule": pattern,
            "position": position,
        }
        if code != 0:
            (out / f"p{pair}_{arm}_meta.json").write_text(
                json.dumps(record, indent=1), encoding="utf-8"
            )
            raise SystemExit(f"RUN FAILED pair={pair} arm={arm}; see {stem}_err.txt")
        report = json.loads((stem.with_suffix(".json")).read_text(encoding="utf-8"))
        record["pipeline"] = report.get("full_fp16_pipeline", "")
        record["staging_pipeline"] = report.get("full_fp16_staging_pipeline", "")
        record["staging_counters"] = report.get("staging_counters", {})
        lines = (stem.parent / (stem.name + "_err.txt")).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        batches, period = parse_stream(lines)
        per_shape = segment(batches, shape_names)
        rows = {}
        shaders = {}
        for shape, block in per_shape.items():
            shader, gpu_ms = timed_dispatches(block, WARMUP)
            if len(gpu_ms) != iterations:
                raise SystemExit(
                    f"{stem.name} {shape}: {len(gpu_ms)} steady-state dispatches, "
                    f"expected {iterations}"
                )
            rows[shape] = [value / 1.0e3 for value in gpu_ms]
            shaders[shape] = shader
        record["shader"] = shaders
        (out / f"p{pair}_{arm}_meta.json").write_text(
            json.dumps(record, indent=1), encoding="utf-8"
        )
        with open(out / (stem.name + "_samples.jsonl"), "w", encoding="utf-8") as handle:
            for shape, samples in rows.items():
                handle.write(
                    json.dumps(
                        {
                            "shape": shape,
                            "arm": ARM_LABEL,
                            "round": 0,
                            "samples_s": [round(float(value), 9) for value in samples],
                            "gpu_timestamp": True,
                            "timestamp_period_ns": period,
                            "shader": shaders[shape],
                            "variant": report.get("full_fp16_variant", "n/a"),
                        }
                    )
                    + "\n"
                )
        with open(index, "a", encoding="utf-8") as handle:
            handle.write(
                f"{pair}\t{arm}\t{code}\t{exe_hashes[arm]}\t{stem.name}\t"
                f"{report.get('full_fp16_variant', 'n/a')}\t{exe}\t{quad}\t{pattern}\t"
                f"{position}\t{record['pipeline']}\t{json.dumps(shaders)}\n"
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
                    for p, a, pos, q, pat in schedule(pairs)
                ],
                "pairs": pairs,
                "iterations": iterations,
                "rounds": rounds,
                "warmup_dropped": WARMUP,
                "shapes": list(shape_names),
                "base_exe": str(base_exe),
                "base_exe_sha256": exe_hashes["base"],
                "cand_exe": str(cand_exe),
                "cand_exe_sha256": exe_hashes["cand"],
                "ratio_convention": "baseline_ms / candidate_ms; > 1.0 means faster",
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
            if low > 1.0
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
                "shapes": list(shape_names),
                "base_exe_sha256": exe_hashes["base"],
                "cand_exe_sha256": exe_hashes["cand"],
                "ratio_convention": "baseline_ms / candidate_ms; > 1.0 means candidate faster",
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
