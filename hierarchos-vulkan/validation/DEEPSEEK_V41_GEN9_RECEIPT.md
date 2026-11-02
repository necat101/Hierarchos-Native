# DeepSeek V4.1 Flash: Gen9 port and qualification

The subsequent [architecture-port qualification](GEN9_ARCHITECTURES_RECEIPT.md)
adds 166 passing forward runs and 72 generation runs on Gen9. It reruns the
V4.1 model/training filter with 53 passed / zero failed / 24 ignored and confirms
all 172 implementation hashes and 68 original frozen files from this receipt
remain unchanged. The historical receipts below retain their original scope.

## Current qualification: 2026-11-01

The remaining strict Gen9 failures are closed on the physical i5-6200U / Intel
HD Graphics 520, driver 31.0.101.2115. The default V4.1 suite now reports
**189 passed, zero failed, 25 ignored by default**. Four ignored release/training
gates were executed separately. Together with Vulkan, parameter streaming and
weight offload, the selected Gen9 gate contains **250 distinct passing tests**.
The original logit limit remains `max_abs < 2e-7`, `rtol=0`.

| Check | Current result |
| --- | --- |
| Mixed-MoE block128 dRoute | `4.65661287307739258e-10`, previously `7.15e-7` |
| Expert 2 down input VJP | `2.98023223876953125e-8`, previously `2.38e-7` |
| Compressor decode | All 8,684 checked operands and 6,808 gradients bit-exact |
| Original structure fixtures | All six storage modes pass with zero logit/routing drift |
| Wider YaRN | All three seeds / 22 steps / 158,514 logits bit-exact, including the 99,328-logit prefill |
| Four release/training gates | 11,184 logits bit-exact, including six uninterrupted LoRA/rsLoRA updates |
| Vulkan / streaming / offload | 38 / 3 / 16 passing tests |
| Four-crate workspace | All targets pass; 65 binaries |
| Shader provenance | 566 artifacts rebuild byte for byte |
| Push ABI / Python checks | 429 blocks / 109 host structs agree; 64 Python tests pass |

The numerical changes preserve the measured operator schedules:

- Gen9 variants emulate the full FP32 FMA where the driver loses product bits.
  The 24-case projection probe improves from 1,398 differing words to zero out
  of 2,304. The 17 native counterpart artifacts retain their original bytes;
  other devices keep their existing shader selection. Exactly representable
  BF16/packed products retain the native fast path.
- Short, wide projection input adjoints use the frozen adjacent-four-product
  groups independently of host ISA. The narrow LoRA-B policy is preserved.
- Compressor forward and backward retain sixteen-channel softmax bodies and
  scalar tails. Selecting eight from the serving CPU changed a pooled value
  by `2.98e-8`, which normalization amplified past the gate. All ten retained
  decode chains are now bit-exact; no state-write or mask behavior changed.
- The eight-lane CE operator uses independently generated CPU expectations on
  the original training logits and targets. All 31 complete inputs are bound
  by SHA-256; both row losses and final scalars reproduce CPU PyTorch bit for
  bit. Original projection/model gradients and optimizer expectations remain
  in use. The full BF16 training gate selects its genuine AVX2 sibling.

All **68 original frozen fixture/checkpoint files remain byte-identical**.
The five missing structure/wide packages were reconstructed without their CPU
forward trajectories, accepting all 18 files only after their frozen hashes
matched. They are retained for fresh-checkout Rust gates; the adapter optimizer
test reconstructs its small package directly from the immutable initial bits.

The new handoff's claim of 78 verified pins is absent from this checkout.
The actual audit has **17 verified pins and 54 hash-bound historical fixtures**.
The compatibility audit passes; `check-all --strict` still rejects those 54
unverified toolchains. They were not assigned speculative AVX-512 provenance.
See [fixture provenance](V41_FIXTURE_PROVENANCE.md).

### Performance after the correctness fixes

Two alternating pairs compare the saved phase-1 optimized binary with the
current binary, with one warmup and two measured steps per process. All five
frozen digests, including the staged fallback, remain unchanged. Profiling and
short-prefill guards are excluded from the latency medians.

| Mode / phase | Phase 1 | Current | Latency change |
| --- | ---: | ---: | ---: |
| Resident prefill 8 | 190.644 ms | 197.780 ms | +3.7% |
| Resident decode per token | 173.529 ms | 175.985 ms | +1.4% |
| Streamed loader prefill 8 | 2,344.482 ms | 2,319.957 ms | -1.0% |
| Streamed loader decode per token | 963.472 ms | 900.976 ms | -6.5% |

The resident profile keeps 24,714 dispatches over nine calls; GPU time changes
from 953.915 to 972.181 ms. These paired samples quantify the correctness cost;
they do not establish a throughput gain from the arithmetic changes.

Machine-readable results and all 58 hashed evidence files are in
[qualification.json](release_parity/deepseek_v41/gen9-20261101/qualification.json)
and [evidence.zip](release_parity/deepseek_v41/gen9-20261101/evidence.zip).
Archive SHA-256: `892cec46eb70c6059dd441014e24f6176608bf8bd7a46af81da8bbb698297d34`.
The existing PowerShell Gen9 gate reproduces the selected test matrix. The
checkpoint restoration command is documented in the provenance guide.

The remaining 21 ignored tests, full production weights, and a fresh physical
RDNA3 run are outside this result. The section below preserves the earlier
optimization measurements and its then-open failure set.

## Historical optimization record: 2026-10-31

Target: Intel Core i5-6200U / HD Graphics 520 (`8086:1916`), Windows driver
31.0.101.2115, 32 KiB compute shared memory. Base: `2695c93`, the published
`codex/deepseek-v41-gen9-handoff`. The later Freebuff edits described in the
pasted session were absent from Git; this change reconstructs the missing work
on `codex/gen9-v41-completion`.

This is **relative port regression and performance evidence**, not a claim that
every original CPU qualification gate passes on Gen9. All 68 original frozen
fixture/checkpoint files still match `HANDOFF-MANIFEST.json`. No original CPU
values or numerical tolerances were changed. Three strict whole-model inference
gates still fail on both the unchanged handoff and the optimized tree.

Machine-readable results are in
[`port.json`](release_parity/deepseek_v41/gen9-20261031/port.json), with hashed
raw logs and reproduction inputs in
[`evidence.zip`](release_parity/deepseek_v41/gen9-20261031/evidence.zip).

## Measured release latency

Quiet, unprofiled alternating pairs; one warmup and two measured steps per
process. Medians pool the raw measured phases. These are the retained scaled
40-layer/384-expert checkpoint, not production-width throughput estimates.

| Mode / phase | Base `2695c93` | Gen9 port | Latency reduction |
| --- | ---: | ---: | ---: |
| Resident, prefill 8 | 873.9 ms | 185.1 ms | 78.8% |
| Resident, decode per token | 401.9 ms | 166.8 ms | 58.5% |
| Streamed loader, prefill 8 | 3,196.7 ms | 3,062.1 ms | 4.2% |
| Streamed loader, decode per token | 1,293.4 ms | 1,143.3 ms | 11.6% |

Resident calls drop from **170 to 50 queue submissions**. Full-stride profiles
over the same nine calls record 54,324 -> 24,714 dispatches and
2,807.6 -> 959.5 ms of GPU dispatch brackets (65.8% lower). Profiled runs are
excluded from latency medians. HC split (27.1%), MoE gate (17.7%) and quantize
(14.0%) are now the largest GPU costs in that profile.

A separate alternating same-binary comparison isolates the fused expert batch
path, including its GPU routing worklist, against the optimized staged expert
loop: prefill **592.8 -> 184.6 ms**, decode **271.1 -> 166.5 ms/token**. All
digests match in both arms. The staged arm uses the same immutable weight arena.

Frozen guards passed for resident 8/8 (`90e096ebc5bf0659`), streamed-loader 8/4
(`2d2f8e2d5914a0d2`), and resident prefill 1/2/3 with one decode
(`8e61e9bd28665ff1`, `95e693dcb9e76429`, `17cc1ba6404ff8d6`). These digest
guards supplement the component CPU references; they do not replace the failing
strict whole-model CPU gates.

## Shipped schedules

- **Quantization:** correct the host grid to `ceil(blocks / 64)` workgroups.
  The old grid launched 64 times too many groups. A guarded exact reciprocal
  multiply handles normal power-of-two scale division. Non-power-of-two,
  subnormal and exceptional operands/results retain `fp32_div`.
- **Packed GEMM:** stage each activation K block once per 64-column workgroup,
  including partial output tiles. Each column retains its serial ascending
  FMA chain, per-group reset, `dot * sa`, then `* sb`, and ordered block sum.
  No split-K reassociation or parallel partial-dot reduction is introduced.
- **MoE packing:** compute selection and the compact-row prefix once per row;
  lanes copy the hidden columns. Pack, forward scatter and gradient scatter
  retain separate grids, including both gradient gathers.
- **HC and norm adjoints:** distribute the 32 independent CPU accumulation
  banks across 32 lanes, retaining the cascaded folding and final ordered sum.
  HC sigmoid/exp, independent Sinkhorn divisions and elementwise adjoints run
  across those lanes. Ordered maxima, sums, reciprocals and adjoint tail sums
  remain on lane zero with the original expressions and exact division.
- **Sparse attention:** score slots across 64 lanes; lane zero retains the
  ordered softmax. Index validity remains distinct from the later score-mask
  threshold, including extreme finite logits. A regression test covers that
  scalar-path edge behavior.
- **Selected expert fusion:** one grid executes routed gate/up, their BF16
  boundaries, SwiGLU/route multiplication, intermediate FP8 quantization and
  the down projection. Each workgroup owns one selected token/expert slot.
  It computes the original compact-row ordinal for shape-dependent unaries.
  Routing indices/counts stay on the GPU; this path removes the per-layer
  counts readback. One shared input quantization precedes the fused grid.

Packed resident experts share **one immutable weight arena**, not a second copy
of the weights. The staged packed GEMM and weight decoder accept byte offsets
into that same arena, so retained training tapes and the staged comparison
remain available. Shared experts retain the staged path.

Fusion requires resident, adapter-free packed routed projections, a common
gate/up activation block, intermediate width at most 2,304, an arena at most
128 MiB, and queried support for its shared memory and 64-lane workgroup.
Other storage/widths and streamed experts retain their staged paths. Fusion
defaults on for Intel Gen9; `HIERARCHOS_VULKAN_V41_FUSED_EXPERTS=0` selects
staged grids. `=1` is a capability-checked qualification override.

## Numerical and build evidence

| Check | Result |
| --- | --- |
| Original handoff V4.1 suite | 153 passed / 33 failed / 25 ignored |
| Final V4.1 suite | 157 passed / 32 failed / 25 ignored; no added failures |
| Previously failing schedule-width attention contraction | Now passes |
| Fused/staged/pinned-CPU MoE | 2,656 output words bit-exact, including block128 |
| Standalone block128 linears | 642 output words bit-exact |
| Routing | Indices/counts exact; weights retain the existing strict `<2e-7` CPU gate |
| Terminal AVX2 CPU-autograd oracle | Loss and all compared adjoints bit-exact |
| Uninterrupted LoRA + rsLoRA trajectory | 6 updates, 24 logit vectors / 1,536 values, all bit-exact |
| Vulkan infrastructure, fresh process | 38 passed |
| Other library tests | 845 passed, 9 ignored, with the isolation described below |
| Four-crate workspace, all targets | Passed |
| Architecture inventory | 146 canonical + 83 aliases, generated document current |
| Shader source/artifact provenance | All 549 artifacts reproduce with their pinned compilers |
| Push-constant ABI | 429 shader blocks / 109 host structs agree |
| Python helper/provenance/layout unit tests | 64 passed; 60 validator/helper modules compile |

The long non-V4.1 library process reported 30 `ERROR_OUT_OF_DEVICE_MEMORY`
failures after 815 passes. The unchanged baseline produced the identical failure
set. All 30 passed in separate fresh processes, giving 845 distinct passing
non-V4.1 tests. The standalone Vulkan suite also passes in a fresh process.
This records device-allocation exhaustion separately from numerical failures.

The uninterrupted training gate previously completed its numerical comparisons
and failed writing its receipt into a missing directory. Its test and the
PowerShell runner now create their evidence directories. Optimizer math is
unchanged. The strict package gate still has exactly the baseline drift
`2.56420373916625977e-2`; this is not hidden by the passing trajectory or by
the final-row performance digests.

## CPU fixture provenance

The new block128 oracle uses the existing SHA-256-checked released model and
kernel sources at `1bae0e1088993b9b82de08f59de5be5a34e42170`. It covers nine
MoE rows, hidden 128/intermediate 256, four experts/top-2, both scale-grid
orientations and two down-projection K groups. Separate linears cover two K
groups and a partial 129-column output tile.

The terminal test selects a separately generated AVX2 sibling for the host's
eight-lane ATen policy. The original fixture is preserved for the sixteen-lane
policy. All validators enforce torch/numpy/CPU pins and protect frozen values.
Eight fixtures have verified local pins under torch 2.14.1+cpu / numpy 2.5.3 /
AVX2. **55 historical fixtures remain unverified**, with original hashes in a
legacy ratchet; they were not assigned speculative AVX-512 provenance.
See [`V41_FIXTURE_PROVENANCE.md`](V41_FIXTURE_PROVENANCE.md).

## Reproduction

Build the probe from base `2695c93` separately, then build the current tree:

```powershell
cargo build --locked --release --manifest-path hierarchos-vulkan/Cargo.toml --bin hierarchos-vulkan-deepseek-v41-forward-probe -j 1
python hierarchos-vulkan/validation/benchmark_deepseek_v41_gen9.py --baseline PATH_TO_BASELINE_PROBE.exe --candidate hierarchos-vulkan/target/release/hierarchos-vulkan-deepseek-v41-forward-probe.exe --output target/gen9-ab-fresh --profile
```

Run on a quiet adapter after builds/tests finish. The runner alternates arm
order over two pairs, warms up each process, stores raw phase/load/I/O records
and rejects any frozen digest mismatch. Profiling runs and short-prefill guards
do not enter latency medians. It checks resident 8/8, streamed-loader 8/4,
resident prefill 1/2/3 with one decode, and the staged fallback.

Focused numerical checks:

```powershell
cargo test --locked --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_fused_expert_chain_matches_staged_grids_and_cpu_references -- --nocapture --test-threads=1
cargo test --locked --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_mixed_linear_block128_matches_pinned_cpu_reference -- --test-threads=1
cargo test --locked --manifest-path hierarchos-vulkan/Cargo.toml --lib terminal_adjoints_match_the_pinned_cpu_fixture -- --nocapture --test-threads=1
cargo test --locked --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_production_uninterrupted_training_logit_trajectory -- --ignored --nocapture --test-threads=1
```

The original strict Gen9 gate remains strict and still stops on the existing
V4.1 failures. These results concern the retained scaled checkpoints, not full
production weights, million-token contexts, or a fresh RDNA3 qualification.
