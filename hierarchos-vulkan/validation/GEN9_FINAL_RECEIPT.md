# Final Intel Gen9 optimization pass: 2026-11-01

The reported Llama-wide streaming regression is closed on the i5-6200U /
Intel HD Graphics 520 (`8086:1916`, driver `31.0.101.2115`). The focused
five-round comparison reduces its median from **11.016 to 4.597 ms (58.3%)**
against the preceding optimization pass. All measured logits remain identical
to that pass, with the unchanged **`max_abs < 2e-7`, `rtol=0`** CPU gate.
This completes the Gen9 optimization work. Physical RDNA3 regression remains
required before release. The regression branch and device commands are described
in the [RDNA3 runbook](../../RDNA3_REGRESSION.md).

The [machine-readable receipt](release_parity/gen9_final/20261101/qualification.json)
and [evidence archive](release_parity/gen9_final/20261101/evidence.zip) contain
the frozen CPU references, raw reports, test logs, implementation snapshots and
executable hashes. They supersede the open Llama performance exception in the
[preceding receipt](GEN9_OPTIMIZATION_RECEIPT.md); its historical JSON and ZIP
remain unchanged.

## Final changes

Ordinary streamed projections previously submitted and waited after every
weight tile. Gen9 now lets small independent projections share a submission,
with at most **1 MiB of coalesced weight/bias payloads**. The next eligible
projection retires the window before exceeding that cap. Large, multi-tile
and aliased projections retain their existing retirement path; a following
fallback tile can temporarily coexist with the bounded small window. Upload
arena allocation/padding is additional and is measured separately.

The runtime's existing staged copies, read/write barriers and bound-buffer
retention preserve ordering and lifetime until completion. This changes
submission timing, without changing shader instructions, projection shapes or
floating-point operations. Parameters still come from the package and are
uploaded for each use; there is no model-sized resident parameter cache.
Other GPU families retain their previous scheduling. Setting
`HIERARCHOS_VULKAN_GEN9_STREAM_BATCH=0` restores per-tile retirement on Gen9.

The V4.1 expert batch reader also detects split weight/scale shards from the
loaded index before building batch plans. Such layouts go directly to the
ordinary reader, retaining its validation and payload identity checks.

## Llama closure measurements

Five alternating rounds use five warmups and 31 measured forwards per arm.
The original architecture-port binary, preceding optimization binary and
final binary consume the same frozen packages. A fourth arm disables batching
in the **same final executable**. Every invocation checks CPU parity, exact
logit words and physical device identity. No build or other GPU test runs
during measurement.

| Streamed control | Original, ms | Previous pass, ms | Final, ms | Same final binary, batching off, ms | Reduction vs previous pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama wide | 11.760 | 11.016 | 4.597 | 10.582 | 58.3% |
| Llama | 10.159 | 9.494 | 2.867 | 7.259 | 69.8% |
| Mistral | 10.088 | 9.028 | 2.841 | 7.219 | 68.5% |
| Qwen2 | 9.438 | 7.310 | 2.955 | 9.697 | 59.6% |

For Llama wide, submissions per forward fall from **19 to four** against the
previous pass and the unbatched final binary; the original used 20. The final
binary is 56.6% faster with batching enabled than disabled in this comparison,
so the result is not dependent on different executable builds alone.

All 80 focused invocations preserve their logit words. The 20 final-arm
invocations also require equal device allocation totals before and after the
31 warmed forwards. Llama-wide peak buffer allocation increases from
5,060,780 to 5,552,300 bytes (**491,520 bytes / 0.469 MiB**) as transient
weights share a submission. That allocation total stabilizes after warmup.
The payload cap bounds coalescing independently of model depth; the original
tile and allocation-arena bounds still apply to larger projections.

## Broader performance check

These are observations on retained scaled checkpoints, not production
throughput. Three alternating rounds use five warmups and 31 measured forwards
for both binaries and both storage modes. The baseline is the architecture
port before the preceding eight optimizations, so this table measures the
combined optimization result. It is separate from the focused closure above.

| Control | Original resident, ms | Final resident, ms | Original streaming, ms | Final streaming, ms |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 wide | 2.727 | 2.145 | 12.783 | 4.375 |
| Llama wide | 2.938 | 2.379 | 11.770 | 4.158 |
| Qwen3-Next | 2.137 | 1.662 | 9.058 | 2.278 |
| Qwen3.5 mixed | 3.037 | 2.444 | 18.758 | 4.284 |
| Qwen4-Exp mixed PLE | 5.658 | 4.066 | 34.698 | 15.411 |
| MiniMax-M3 | 2.767 | 2.671 | 15.006 | 8.545 |
| DBRX | 2.682 | 1.821 | 15.806 | 9.107 |
| GPT-OSS | 2.228 | 1.891 | 18.926 | 4.094 |

Streaming medians improve 42.4–78.4%; resident medians improve 3.5–32.1%
in this run. Every timed invocation checks CPU parity and exact words across
binaries and storage modes. Small host/driver timings vary between experiments;
the raw measurements are retained without discarding slower runs.

V4.1's separate three-pair release probe preserves all five frozen digests
and the staged-expert fallback. Against the original binary, resident prefill
and decode medians improve 3.6% and 4.5%; default streamed prefill and decode
are 0.65% and 5.41% slower in that comparison. A matched three-round follow-up
against the immediately preceding pass measures +1.04% prefill and +0.69%
decode, with the reader-disabled final binary close to the preceding pass.
These measurements show variation between runs and do not establish a new
V4.1 streamed speedup. Construction medians and all follow-up arms are retained
in the JSON receipt. The Llama closure is supported separately by its much
larger reduction and deterministic submission-count change.

## Final qualification

| Gate | Result |
| --- | --- |
| Distinct Rust tests | 269 passed, zero failed |
| V4.1 default suite | 190 passed; 25 explicit opt-in tests ignored |
| V4.1 explicit gates | Default package, joint resident, joint streamed vocabulary and uninterrupted LoRA/rsLoRA trajectory pass |
| Streaming | Seven tests pass, including bounded upload retirement, ordered weight lifetimes, alias handling and two-step LoRA/AdamW identity |
| Other native checks | 17 offload, 12 cache, one Qwen3-Next backward and 38 Vulkan tests pass |
| Forward CPU parity | 166/166 runs, 43 profiles / 26 native identifiers; worst `1.7881393432617188e-7` |
| Generation CPU parity | 72/72 requests, 18 profiles / four steps; worst `8.940696716308594e-8`; tokens match |
| Prior Gen9 output identity | 77,772 forward words and 9,216 generation words match exactly |
| Storage identity | Every resident/streaming word matches, including padded queries |
| Frozen V4.1 files | All 68 original hashes unchanged |
| Shaders | All 585 SPIR-V files and shader sources unchanged from the preceding pass; its 569-artifact byte-rebuild proof still applies |
| Push-constant ABI | 429 shader blocks / 108 host structs agree |
| Workspace | Four crates / 65 binaries, all-target checks pass |
| Qualification negative controls | 16 passed |

The new GPU regression alternates distinct weight payloads across the 1 MiB
retirement boundary, asserts submission and retention behavior, and compares
every output bit against resident projections. Existing tests exercise
multi-tile and aliased paths. All tests here use the final library test binary.

CPU forward parity covers 66,474 observable values and excludes 11,298 padded
query values. Storage and prior-binary identity include those values.
Generation compares 9,216 values. SmolLM3 cache requests retain the existing
full-prefix fallback. The remaining 120 registry identifiers and all-profile
CPU PEFT training are outside this certificate. Fixture provenance remains
17 verified pins and 54 historical unverified toolchains.

## RDNA3 handoff

Use the final working-tree overlay or this complete checkout. Extract the
final evidence ZIP to obtain the portable CPU references:

```powershell
python -m zipfile -e hierarchos-vulkan/validation/release_parity/gen9_final/20261101/evidence.zip target/rdna-final-reference
```

Set `RDNA_DEVICE_ID` to the known adapter's PCI device ID, then run:

```powershell
python hierarchos-vulkan/validation/run_rdna3_final_regression.py --forward-fixtures target/rdna-final-reference/cpu/forward --generation-fixtures target/rdna-final-reference/cpu/generation --generation-packages target/rdna-final-reference/cpu/generation-packages --expected-device-id $env:RDNA_DEVICE_ID --output target/rdna-final-gen9-closure-20261101
```

This standard-library Python runner rebuilds locally, consumes the frozen CPU
expectations, verifies AMD vendor and the requested device in every forward and
generation report, and runs GPU gates sequentially. Its native gate includes
the new batching test and Qwen3-Next backward regression. Choose a fresh output
directory for each attempt. Gen9 batching is disabled by physical device
selection on RDNA3. No new physical RDNA3 result is claimed here. The
qualification archives capture the Gen9 state before branch publication.
