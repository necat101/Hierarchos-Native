#!/usr/bin/env python3
"""Write the v0.2.7 GitHub release body from the built package manifest.

    python make_release_body.py --manifest <BUILD_MANIFEST.json> --out <body.md>

The prose is fixed; the asset sizes, SHA-256 values, module identities and
build provenance come from the manifest the assembler just wrote, so the
published notes cannot disagree with the published files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HEADER = """Windows x64 portable release of **Hierarchos Native v0.2.7**. It promotes
four qualified attention routes into the default training path and requalifies the
opt-in Mixed WMMA arm without promoting it.

## Promoted

- **Two-pass MHA attention backward** — the default for the gate it already had
  (`unfused_qk` 0/9). A query pass writes softmax and `dS` scratch; a key pass
  reduces it. Attention backward fell from 435.28 ms to 44.43 ms per step and the
  complete optimizer step from 560.78 ms to 171.10 ms, a paired **3.2492×**
  (95 % CI 3.1146–3.3838, five ABBA rounds, 230 samples per arm). Gradients are
  bit-exact against the scalar route over four geometries: 6,316,128 of 6,316,128
  FP32 elements, zero ULP. Scratch is sized exactly to
  `batch * heads * q_seq * kv_seq` and allocated only for eligible geometry.
- **GQA grouped backward key pass** — admitted for `8q/2kv`, `4q/2kv`, `8q/4kv`
  and `16q/4kv` at head dimension 128, `q_seq = kv_seq = 64`, causal, and for
  nothing else. Bit-exact for the measured groupings in both arithmetic modes;
  the 8q/2kv complete step measured **19.76×** (95 % CI 19.64–19.89). MHA is
  unchanged bit for bit. Every other grouped geometry keeps the scalar kernel.
- **Row-packed cached attention forward** — one lane per `(batch, head, query)`
  with its scores in registers and the reused `V` rows cached, for
  `0 < q_seq == kv_seq <= 64`, equal QK/value widths, `unfused_qk ∈ {0,1,9}` and
  the plain path only. Forward attention fell from 16.42 ms to 4.38 ms per step;
  the shipped selector measured **1.2239×** on the complete step. Outputs are
  bit-exact: 1.85 M of 1.85 M words over ten geometries and three modes.
- **Four-channel backward key pass** — one lane owns a four-channel chunk of a
  destination key row, 32 lanes to a row's 128 channels. Limited to
  `head_dim == value_head_dim == 128`; head dimensions 16, 32 and 64 keep the
  original key shader. The complete step measured **1.5297×** (43.53 ms →
  28.46 ms, 5/5 rounds) with 20 parity cases, 60 tensors and 5,505,024 elements
  at zero failures, including an MHA control.

Every promoted route keeps its previous fallback, and the per-dispatch counters in
a receipt report which route actually ran. `HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=0`
forces the scalar backward; `HIERARCHOS_ATTENTION_BACKWARD_KEY_CHANNEL4=0`
restores the original key shader. The rejected Phase VIII key-range partitioning
is not in the production path: a split backward submission is refused the
two-pass pair and keeps the scalar kernel.

## Mixed WMMA: requalified, still opt-in

| Measurement | Result |
| --- | ---: |
| LoRA complete step, Phase XII harness | 28.481 ms → 23.001 ms = **1.2385×** |
| LoRA complete step, independent counter harness | **1.2288×** |
| Full-parameter complete step (timing evidence only) | **1.2792×** |

Mixed stays **default-off and opt-in**. The policy switch is process-wide and
cannot distinguish LoRA from full-parameter training, so the measured LoRA result
does not justify a default change. The full-parameter result at the fixture
learning rate `1e-2` is **timing evidence only**: the FP32 reference itself
diverges there (7.13 → 19.56), so full-parameter numerics at that rate are not
qualified. The four LoRA adapter tensors show deterministic elementwise drift
(maxima 0.023203, 0.022338, 0.021156 and 0.025953 — nine elements of 32,768 above
`0.02`) while every tensor's RMS stays inside the established `0.003` limit. **No
per-element adapter-weight parity is claimed.** Every assertion the documented
Mixed contract does make passes.

## Unchanged

`FULL_FP16_MAX_REDUCTION = 64`, `WMMA_MIN_PROFITABLE_MACS = 16_000_000` and every
numerical threshold; the explicit WMMA off mode; all FP32 fallbacks and
fail-closed admission; and the separation of Full FP16 from Mixed. `full-fp16`
still runs the C1 batched-staging module in front of the unchanged P3 contract,
with P3 as the automatic fallback.

## Scope

One measured adapter: integrated AMD RDNA3, PCI `1002:15bf`, `AMD Radeon
Graphics`, driver `32.0.32015.14001`, Windows 11 x64. GQA eligibility is limited
to the four recorded head-count groupings at head dimension 128; channel4 is
limited to head dimension 128; row-packed forward to square sequences ≤ 64.
**No universal AMD, NVIDIA, Intel or cross-architecture qualification is
implied.** The ratios above are independent paired results and are never
multiplied into an overall speedup.

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
        "hashes, the four promoted attention module identities and the two WMMA "
        "staging module identities"
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
