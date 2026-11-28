#!/usr/bin/env python3
"""Write the v0.2.8 GitHub release body from the built package manifest.

    python make_release_body.py --manifest <BUILD_MANIFEST.json> --out <body.md>

Asset sizes and SHA-256 values come from the manifest the assembler wrote, so
the published notes cannot disagree with the published files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HEADER = """Windows x64 portable release of **Hierarchos Native v0.2.8**. It promotes two
qualified query-pass changes into the default RDNA3 training route and leaves the
WMMA precision policy unchanged: global WMMA stays **off by default** and Mixed
WMMA stays **opt-in**.

## Promoted

- **Phase XVIII literal-index dQ accumulator.** On the geometry the existing
  query selector already admitted (QK and value head dimension 128), the query
  pass indexes its dQ accumulator with literals instead of a runtime chain.
  Eight independent geometry/mode series improved in **5/5 ABBA rounds each**,
  **1.042×–1.124×** on the complete optimizer step, with the 95 % CI excluding
  1.0 in every series. The original query shader remains the overall fallback.
- **Phase XX scratch reuse is the default on that qualified path.** The query
  pass reuses its softmax/dS scratch instead of rewriting it. The Phase XIX
  five-round, 160-process campaign measured the complete step at **1.088×** FP32
  (26.799 → 24.643 ms, 95 % CI 0.9176–0.9210) and **1.120×** Mixed (21.386 →
  19.111 ms, 95 % CI 0.8874–0.8977), 5/5 rounds improved in both. The six
  additional regression series improved 1.071×–1.148×.
- **Phase XXII output-tiled FP32 input gradient.** For shapes whose rows, input
  dimension and output dimension are all multiples of 32, a 32 x 32 output tile
  computes four outputs per thread while preserving each output's ascending-`k`
  single-`fma` reduction. The five-round ABBA measured the complete step at
  **1.1462×** (24.875 → 21.716 ms, 95 % CI 0.8706–0.8742, 5/5 rounds), with
  272,038/272,038 output words bitwise equal over six shapes. The original
  16 x 16 kernel remains the fallback and the control arm
  (`HIERARCHOS_LINEAR_INPUT_GRAD_OUTPUT_TILE=0`).

Pass 1 selects `query_dqunroll_scratchreuse` by default on the qualified path,
`query_dqunroll` when scratch reuse is disabled or its pipeline is refused, and
the original query shader when the dQ-unroll path is disabled or the geometry is
not admitted. `HIERARCHOS_ATTENTION_BACKWARD_QUERY_SCRATCH_REUSE=0` selects the
Phase XVIII fallback; `HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL=0` selects
the original query shader.

Numerical evidence: Phase XIX compared 140/140 tensors bitwise across 28 cases
(`dQ`, `scratch_probability`, `scratch_ds`, `dK`, `dV`), with zero non-finite
values. Phase XX compared 29 complete steps in-graph: 87 comparisons, zero
mismatches, 29/29 loss bits, LoRA gradient digests and parameter digests equal,
and the identical `parameter_end_digest = d79a12d8…9c4655`. The Phase XVIII
fallback matrix was re-read with scratch reuse pinned off: 30 cases, 150 tensor
comparisons, zero failures.

The Phase XXII full-step ABBA also produced an identical 29-step
`parameter_end_digest = d79a12d8…9c4655` in both arms. Every ratio is an
independent paired result on one adapter and is never multiplied into an
overall training speedup.

## Unchanged

Mixed WMMA remains opt-in and default-off; its thresholds, full-parameter
learning-rate caveat and adapter elementwise drift observations are unchanged.
`FULL_FP16_MAX_REDUCTION = 64`, `WMMA_MIN_PROFITABLE_MACS = 16_000_000`, the hard
numerical gate `0.03125`, the Phase XII four-channel key pass, the row-packed
forward and the GQA grouped key pass keep their artifacts, gates and control
arms. No precision threshold or eligibility allowlist moved.

## Scope

One measured adapter: integrated AMD RDNA3, PCI `1002:15bf`, AMD Radeon
Graphics, driver `32.0.32015.14001`, Windows 11 x64. Query-pass eligibility is
head dimension 128 inside the existing two-pass gate. **No universal AMD,
NVIDIA, Intel or cross-architecture qualification is implied.** Timings are
per-step milliseconds on one machine: no throughput, power or multi-GPU claim.

## Assets
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    package = manifest["package"]
    lines = [HEADER]
    lines.append("")
    lines.append(
        f"- `{package['file']}` — {package['bytes']:,} bytes, "
        f"{package['entries']} entries, SHA-256 `{package['sha256']}`"
    )
    lines.append(
        "- `BUILD_MANIFEST.json` — packaging commit, toolchain, flags, per-executable "
        "hashes, the two promoted query module identities and the retained attention "
        "and WMMA module identities"
    )
    lines.append("- `SHA256SUMS` — SHA-256 of the archive and the manifest")
    lines.append("")
    lines.append("Seven executables, all x64 PE32+, statically linked against the MSVC C")
    lines.append("runtime: no Visual C++ Redistributable, Rust, Cargo, Python or shader")
    lines.append("compiler is needed to run them.")
    lines.append("")
    lines.append("```text")
    for name, spec in manifest["executables"].items():
        lines.append(f"{name:44s} {spec['bytes']:>10d}  {spec['sha256']}")
    lines.append("```")
    lines.append("")
    lines.append(
        f"Built from source commit `{manifest['source_commit']}` "
        f"({manifest['target_triple']}, {manifest['build_timestamp_utc']}), "
        f"source state `{manifest['source_state_at_packaging']}`."
    )
    lines.append("")
    args.out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
