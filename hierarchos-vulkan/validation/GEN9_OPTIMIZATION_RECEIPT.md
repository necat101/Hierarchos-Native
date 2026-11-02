# Intel Gen9 optimization pass: 2026-11-01

The subsequent [final Gen9 pass](GEN9_FINAL_RECEIPT.md) closes the Llama-wide
streaming exception recorded below. This document retains the earlier results.

The i5-6200U / Intel HD Graphics 520 (`8086:1916`, driver `31.0.101.2115`)
passes the retained CPU gates at **`max_abs < 2e-7`, `rtol=0`** after eight
optimization changes. Resident forward medians improve across all eight timed
architecture controls. Streaming measurements have a remaining Llama-wide
performance exception described below. Physical RDNA3 regression is pending.

The [machine-readable receipt](release_parity/gen9_optimizations/20261101/qualification.json)
and [evidence archive](release_parity/gen9_optimizations/20261101/evidence.zip)
retain frozen CPU inputs, raw native outputs, timing runs, implementation
snapshots, executable hashes and the scripts used for this pass. Historical
V4.1 and architecture-port receipts remain separate.

## Changes

1. Package readers retain payload handles and validated tensor locations.
   Every payload cache miss still checks the shard's file identity; changed
   handles reopen, and missing/truncated sources fail.
2. FP32 tile decoding appends directly to the destination, avoiding a temporary
   vector for each tile. Finiteness checks retain the requested-value scope.
3. Dense transposed tiles and consecutive mapped rows use bounded reads of up
   to 8 MiB. Partial/interleaved slices retain exact reads, including exclusion
   of NaNs outside the requested slice. Mapped row order and duplicates remain
   intact.
4. Logits-only inference omits the unused final hidden-state download. The
   public API requesting hidden states still returns the same values.
5. Gen9 QSA caches visible keys, block scores and query normalization, and
   computes independent block scores across lanes. It preserves score sums,
   lower-index ties, incomplete tails, padding and long-context fallbacks.
   The host also stops launching X workgroups that immediately returned.
6. Gen9 HC computes independent row/column statistics across lanes while
   retaining each sum's original order and the existing barriers.
7. Gen9 inference fuses SiLU and multiplication by the up projection into one
   dispatch, storing both activation and product. Training retains its existing
   path. Other adapters retain their existing shader selection.
8. Packed expert reads sharing one shard validate identity at both boundaries
   of a bounded six-payload batch. Metadata checks and adapter attachment remain
   in force; cross-shard or ordinary-weight layouts use the original loader.

Three new Gen9 artifacts are selected by physical PCI identity. All **582
pre-existing SPIR-V files match the prior architecture receipt**, and rebuilding
all **569 active artifacts** reproduces their embedded bytes. The native
macro-disabled HC, QSA and SiLU artifacts also rebuild unchanged.

## Qualification

| Gate | Result |
| --- | --- |
| Forward CPU parity | 166/166 runs, 43 profiles / 26 native architecture identifiers; worst `1.7881393432617188e-7` |
| Generation CPU parity | 72/72 cache-on/off requests across 18 profiles, four steps; worst `8.940696716308594e-8`; tokens match |
| Previous Gen9 output identity | All 77,772 forward words and 9,216 generation words match the previous binaries exactly |
| Storage identity | Every resident/streaming word matches, including padded query rows |
| V4.1 default suite | 190 passed, zero failed; 25 explicit opt-in tests ignored |
| V4.1 explicit release gates | Default package, joint resident, joint streamed vocabulary, uninterrupted LoRA/rsLoRA trajectory pass |
| Additional native regressions | 17 offload, six streaming/hidden-state, 12 cache, one Qwen3-Next adjoint and 38 Vulkan tests pass |
| Distinct Rust tests | 268; two payload-identity tests were also rerun separately |
| Qualification negative controls | 16 passed |
| Standalone shader controls | 148 QSA, 135 HC and 13 SiLU cases; every compared output word identical |
| SiLU extreme inputs | All 65,280 finite BF16 values included, with activation and product checked |
| Shader ABI | 429 shader push blocks / 108 host structs agree |
| Workspace | Four crates / 65 binaries, including all-target checks |
| Frozen V4.1 inputs | All 68 original file hashes unchanged |

The V4.1 suite uses a frozen test executable built before the independent
generic SiLU integration; its V4.1 implementation is the final implementation.
The later test executable covers the final generic paths and explicit gates.
Both executables' hashes are retained. No CPU oracle values were regenerated.

Forward CPU comparison covers 66,474 observable values and excludes 11,298
padded-query values. Storage and previous-binary identity include those padded
values. Generation compares 9,216 values. SmolLM3 cache requests exercise its
existing full-prefix fallback. The remaining 120 registry identifiers and
all-profile CPU PEFT training are outside this certificate. Fixture provenance
remains 17 genuine pins and 54 historical unverified toolchains.

## Idle performance observations

These are small retained checkpoints, not production throughput. Three
alternating rounds per control use three warmups and eleven measured forwards;
both binaries execute both storage modes. Every invocation checks CPU parity
and exact baseline/candidate/storage words. Builds and other GPU tests have
finished before timing.

| Control | Baseline resident, ms | Candidate resident, ms | Resident latency reduction | Initial streaming latency change |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 wide | 2.676 | 2.313 | 13.5% | -49.4% |
| Llama wide | 2.920 | 2.468 | 15.5% | +11.0% |
| Qwen3-Next | 2.194 | 1.594 | 27.4% | -6.7% |
| Qwen3.5 mixed | 3.099 | 2.483 | 19.9% | -2.9% |
| Qwen4-Exp mixed PLE | 6.104 | 4.152 | 32.0% | -29.7% |
| MiniMax-M3 | 3.278 | 2.730 | 16.7% | +15.6% |
| DBRX | 2.701 | 1.944 | 28.0% | +7.3% |
| GPT-OSS | 2.341 | 1.853 | 20.9% | -10.8% |

A longer five-round repeat uses five warmups and 31 measured iterations for
Llama, MiniMax and DBRX. Streaming medians improve 3.1% for MiniMax and 28.3%
for DBRX, but Llama is 16.7% slower. A further five-round Llama baseline/fused/
unfused diagnostic gives 10.860 / 11.077 / 11.062 ms: about 2% slower for both
candidate arms. Disabling fusion does not recover the difference. These runs
show substantial host/driver timing variation and a **remaining possible
Llama-wide streaming regression**; this receipt makes no streaming speedup
claim for that control. All raw runs are retained.

V4.1 uses three alternating pairs, two measured steps and one warmup. Resident
prefill falls from 196.392 to 188.320 ms (**4.1%**); resident decode falls from
175.109 to 166.342 ms/token (**5.0%**). The default streamed policy improves
1.35% and 0.80% respectively. Five frozen digest guards and the staged expert
fallback pass. Original two-shard resident construction is 6.1% slower in this
measurement; default construction is 7.1% faster. This is a load/runtime
tradeoff, not a universal loading-speed claim.

An exact-byte single-shard repack separately exercises expert batching. Every
tensor's dtype, shape and payload SHA-256 is checked against the frozen joint
checkpoint. Three paired runs preserve the resident/default frozen digests.
Resident load median falls from 10,649.624 to 6,343.022 ms (**40.4%**), with
93,338 payload reads / 29,447,344 bytes unchanged and identity checks reduced
to 31,738. Default-policy load changes by +0.5%. The original two-shard
checkpoint deliberately remains unchanged and uses the fallback.

The isolated uncached-package probe improves from 599.453 to 336.575 ms
(**1.78x**, three alternating pairs), with per-miss identity checks enabled.
Standalone median shader ratios are 17.86x for QSA, 1.12x for HC and 2.00x
for fused SiLU/product. These isolate operators and do not predict model
throughput. In the mixed-QSA resident profile, selector time falls from
1.330 to 0.111 ms/dispatch and its GPU-time share from 44.2% to 6.6%.
Profiled timings are excluded from the unprofiled model medians.

## RDNA3 final regression

Transfer the working-tree handoff overlay or the complete current checkout.
Extract this receipt's `evidence.zip` under `target/rdna-final-reference`:

```powershell
python -m zipfile -e hierarchos-vulkan/validation/release_parity/gen9_optimizations/20261101/evidence.zip target/rdna-final-reference
```

Set `RDNA_DEVICE_ID` to the known RDNA3 adapter's PCI device ID, then run:

```powershell
python hierarchos-vulkan/validation/run_rdna3_final_regression.py --forward-fixtures target/rdna-final-reference/cpu/forward --generation-fixtures target/rdna-final-reference/cpu/generation --generation-packages target/rdna-final-reference/cpu/generation-packages --expected-device-id $env:RDNA_DEVICE_ID --output target/rdna-final-20261101
```

The runner uses standard-library Python, rebuilds the release probes and
library tests, verifies AMD vendor and the requested PCI device in every
forward/generation report, and runs GPU gates sequentially. It consumes the
frozen expectations directly, including relocation of the original generation
source manifest. Choose a fresh output directory for each attempt. This pass
does not contain a new physical RDNA3 result. That regression remains required
before push.

For Gen9 diagnosis, setting any of the following to `0` selects the retained
fallback for that optimization:

* `HIERARCHOS_VULKAN_GEN9_PARALLEL_QSA`
* `HIERARCHOS_VULKAN_GEN9_PARALLEL_HC`
* `HIERARCHOS_VULKAN_GEN9_FUSED_SILU`
* `HIERARCHOS_VULKAN_V41_BATCH_EXPERT_READS`

The HC switch preserves the earlier Gen9 exact-FMA shader selections.
