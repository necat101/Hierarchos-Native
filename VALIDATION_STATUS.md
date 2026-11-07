# Current validation scope

The [November 2 RDNA3 closure](hierarchos-vulkan/validation/RDNA3_GPT2_CLOSURE.md)
passes **166/166 forward runs** across 43 profiles / 26 architecture identifiers
and **72/72 generation requests**. The padded GPT-2-wide gap is closed at the
unchanged strict `<2e-7`, `rtol=0` gate with the original CPU words. V4.1 native
PEFT, resident/streamed GPT-2 CPU PEFT, offload and streaming regressions pass.
The new math is selected only for RDNA3 GPT-2; all 585 existing shader binaries
are unchanged and Gen9 routing is guard-tested. This host has no physical Gen9
adapter, so no new physical Intel run is claimed. This remains a scaled-checkpoint
certificate, not complete architecture-registry or all-profile CPU training coverage.

The [2026-11-06 RDNA3 profile completion receipt](hierarchos-vulkan/validation/release_parity/multimodal_training/20261106-rdna3-profile-completion/README.md)
records the physical replay of the six previously missing shared profiles
(9/9 cases) and the native V4.1 bridge (2/2 cases, 1,360 logits). The 44-profile
inventory is 44/44 fully green; 85/85 image cases pass. The
`gpt2_wide/unmasked` residual is closed at `1.7881393432617188e-07` (final),
with `gpt2_wide/padded` at `1.9371509552001953e-07`, so
`rdna3_multimodal_qualified` is set. The forward stage tape is bit-exact
in the historical operation-local eager diagnostic. The retained CPU model
actually uses SDPA; the subsequent
[expanded GPT2 qualification](hierarchos-vulkan/validation/release_parity/multimodal_training/20261106-rdna3-gpt2-shapes/README.md)
corrects that diagnostic, matches the SDPA and unary/optimizer schedules, and
passes all 16 new shape/mask cases. It also repeats the existing shared and
native image gates. The
`2e-7` gate and CPU expectations are unchanged. Three earlier candidate
association changes (K=128 block gating, small-Welford off, AVX2 row GEMM at
width 16) were measured and rejected; the backward alignment closed the gate.
The
[receipt guard](hierarchos-vulkan/validation/check_rdna3_multimodal_receipts.py)
re-derives 44/44 and 85/85 from the per-profile rows, verifies the receipt
hashes, and requires the GPT-2-wide closure evidence.

The [full Gen9-green RDNA3 replay](hierarchos-vulkan/validation/release_parity/multimodal_training/20261106-rdna3-full-gen9-replay/README.md)
replays all nine published stages on `1002:15bf`: 77 shared-image cases, 6
seq2seq, 1 legacy Gemma4, 1 RGB Gemma4, 4 Gemma3, 4 LLaVA, 166
resident/streaming text runs, 215 active DeepSeek tests (26 opt-in ignored),
and 30 native V4.1 tests. All nine stages pass at the unchanged strict gate;
the 2,748-file frozen manifest is unchanged. A four-round same-binary ABBA
benchmark of the native V4.1 hardware-FMA candidate was word-exact but
performance-neutral (`0.996x` and `1.000x` on the two native geometries), so no
speedup is claimed for it. No Gen9 execution is claimed.

The final [native V4.1 submission and routing receipt](hierarchos-vulkan/validation/release_parity/multimodal_training/20261106-rdna3-v41-final/README.md)
records the measured optimization: model construction and the retained graph
share one compute session, component uploads publish into it, and forward
routing counts are reused by the reverse recompute and adapter filter. Native
submissions fall from 970 to 509 per step. Four ABBA rounds with seven measured
updates give 1.295x at 4x4 and 1.217x at 6x4, word-exact, with shared controls
unchanged. The 30-test native qualifier passes with `max_abs_logit = 0.0`, and
the full nine-stage replay passes on the same binary with all 2,748 frozen files
unchanged.

The later [session-wide upload receipt](hierarchos-vulkan/validation/release_parity/multimodal_training/20261106-rdna3-v41-uploads/README.md)
records the next increment: uploads, metadata copies and tape snapshots record
into the open session instead of retiring it, and attention norm gains were briefly
shared instead of duplicated; that sharing was reverted. Native submissions fall to 241 per step, with four-round
ABBA gains of 1.464x at 4x4 and 1.366x at 6x4, word-exact and with shared
controls unchanged. The 30-test native qualifier passes with
`max_abs_logit = 0.0`; the nine-stage replay was not rerun for this increment.

The [resident-expert measurement](hierarchos-vulkan/validation/release_parity/multimodal_training/20261106-rdna3-v41-resident-experts/README.md)
shows that the remaining routing-count reads cannot be batched into one transfer
per step: each layer's gate counts select its streamed expert payloads, and the
fused packed batch refuses on this RDNA3 adapter. Resident experts remove 72 of
the 241 submissions instead, with four-round ABBA gains of 1.175x at 4x4 and
1.125x at 6x4, bit-exact and with the 30-test qualifier green at
`max_abs_logit = 0.0`. Residency is a benchmark-only switch; the production
default remains streamed.

An [adapter-readback batching trial](hierarchos-vulkan/validation/release_parity/multimodal_training/20261106-rdna3-v41-adapter-read-batching/README.md)
collapsed the 1,008 per-group reads to 28 with one transfer per step, but every
submission cadence was slower (best 6x4 resident 296.6 ms vs 273.4 ms), so the
change was reverted. The reads pace host recording against device execution
rather than limiting bandwidth.

The [batch-4 state-chain receipt](hierarchos-vulkan/validation/release_parity/multimodal_training/20261107-rdna3-v41-batch4-state-chain/README.md)
binds binary `2e9cfaf1...`: all nine Gen9-frozen replay stages pass,
including 166 text runs, 217/217 active DeepSeek tests (243 enumerated, 26 original
ignored), and 30/30 native V4.1 tests with `max_abs_logit = 0.0`. The separate
native AVX512 track passes all 95 retained image cases and 16 expanded GPT2
executions. The decode-tape snapshot test now covers batches 1, 2 and 4; batch-2
and batch-4 KV/score state gradients are chained through completed and pending
tapes, with both token gradients and all four state gradients remaining
word-identical after live gain/cache overwrite. Every row is activated
independently at batches 2 and 4, so no gradient may cross into another row, and
overwritten state slots are checked. All 2,748 frozen files, 1,341 protected Gen9
files and 672 original recipes remain unchanged. The only refreshed source
binding is the batch-4 test; the prior
[state-chain receipt](hierarchos-vulkan/validation/release_parity/multimodal_training/20261106-rdna3-v41-state-chain/README.md)
and [preceding batch-2 snapshot receipt](hierarchos-vulkan/validation/release_parity/multimodal_training/20261107-rdna3-v41-batch2-decode-snapshot/README.md)
remain sealed for their original binaries. No new performance claim is made.

The Gen9
reproduction pack is
[here](hierarchos-vulkan/validation/intel-handoff/MULTIMODAL_GEN9.md); its
[status record](hierarchos-vulkan/validation/intel-handoff/multimodal-gen9-status.json)
marks the physical run blocked on this AMD-only host with no receipt issued.

The [final Gen9 receipt](hierarchos-vulkan/validation/GEN9_FINAL_RECEIPT.md)
closes the Llama-wide streaming regression: 11.016 to 4.597 ms against the
preceding pass, with 19 submissions reduced to four. Bounded Gen9 upload
batching preserves every measured logit word. All 269 selected Rust tests,
166 forward runs and 72 generation runs pass at `max_abs < 2e-7`, `rtol=0`.
The broader eight-control run shows lower resident and streaming medians.
This is the final Gen9 optimization pass. The frozen-reference RDNA3 runner is
available; the physical closure above records the complete measured matrix. The
[earlier optimization receipt](hierarchos-vulkan/validation/GEN9_OPTIMIZATION_RECEIPT.md)
retains the original measurements and the exception this final pass closes.

The Gen9 changes are published for device testing on
`codex/gen9-final-rdna3-qualification`. Follow the
[RDNA3 regression runbook](RDNA3_REGRESSION.md) for a fresh AMD run; the scoped
physical result is linked above. Qualification archives are immutable
snapshots; only the new closure receipt claims the new physical RDNA3 pass.

The next Gen9 architecture-port phase is recorded in the
[architecture receipt](hierarchos-vulkan/validation/GEN9_ARCHITECTURES_RECEIPT.md):
166/166 resident/streamed forward runs across 43 profiles / 26 native architectures
pass at `max_abs < 2e-7`, `rtol=0`, worst `1.7881393432617188e-7`.
All 32 earlier RDNA profiles are covered for inference. Another 72 cached/full-prefix
generation requests pass, worst `8.940696716308594e-8` (SmolLM3 retains its
full-prefix fallback when caching is requested). Storage modes match every logit
word. The fresh V4.1 model gate passes 53 tests; six native regressions and 14
qualification negative controls pass. This does not qualify the remaining 120
architecture identifiers or extend every profile's CPU PEFT training certificate.

The Intel Gen9 optimization follow-up is recorded in the
[Gen9 receipt](hierarchos-vulkan/validation/DEEPSEEK_V41_GEN9_RECEIPT.md).
The 2026-11-01 Gen9 continuation passes all 189 default V4.1 tests with zero
failures (25 ignored by default), at the unchanged strict CPU logit limit.
The four explicit release/training gates compare 11,184 logits bit-exactly.
Vulkan, parameter streaming and offload bring the selected Gen9 gate to 250
passing tests. All five performance digests and 68 original frozen files remain
unchanged. Historical toolchain provenance still has 54 unverified entries;
see the [fixture audit](hierarchos-vulkan/validation/V41_FIXTURE_PROVENANCE.md).

**The published RDNA3 validation below binds the original handoff, before the
Gen9 shader changes.** It covers V4.1 Flash and recorded parameter/MoE streaming
checks, not the complete architecture registry or all precision policies.
Those historical receipts do not add current-revision physical qualification;
the November 2 closure above supplies the fresh RDNA3 result.

| Device / revision | Validated scope |
| --- | --- |
| AMD Radeon Graphics / Ryzen Z1 Extreme, 2026-11-02 GPT-2 closure | 166/166 forward runs, 72/72 generation requests; V4.1 native PEFT and resident/streamed GPT-2 CPU PEFT; 269 selected native tests, 50 GPT-2 tests and unchanged Gen9 shader artifacts |
| AMD Radeon Graphics / Ryzen Z1 Extreme, published `2695c93` handoff | V4.1 Flash native resident/streamed inference, LoRA/rsLoRA training, save/resume/export, and the recorded parameter/MoE streaming matrix |
| Intel HD Graphics 520, 2026-11-01 Gen9 follow-up | Retained scaled-checkpoint inference, resident/streamed execution, wide YaRN and native PEFT gates; 250 passing tests, plus all-target workspace and shader checks |
| Intel HD Graphics 520, 2026-11-01 architecture continuation | 166 forward runs, 72 generation runs, 26 native architectures; resident/streamed word identity; 53 fresh V4.1 model/training tests and six native regression tests |
| Earlier Intel and ROG Ally revisions | Historical evidence in their existing reports; not current-revision certification |

The final RDNA3 pass has 288 passing tests. Parsed default-gate logits cover
201,714 values, worst absolute drift `1.4901161193847656e-7`; the strict gate is
`max_abs < 2e-7`, `rtol=0`, against immutable CPU PyTorch expectations. The four
explicit package/resident/streamed/PEFT gates compare 11,184 logits bit-exactly.
Streaming stays opt-in through `--stream-parameters` or the unchecked GUI toggle.

Qualification uses the accepted scalable-checkpoint criterion. The retained
44.8 MB inference package exercises the 40-layer schedule, 384 experts/top-6,
HC4, compression, shared KV/index state and Engram with reduced widths/tables.
PEFT training uses six reduced-width layers with the production expert ratio.
Full production weights and CUDA are not required or claimed.

See the [final receipt](hierarchos-vulkan/validation/release_parity/deepseek_v41/rdna3-final-push.json)
and [streaming evidence](hierarchos-vulkan/validation/release_parity/parameter_streaming/).
The [published final-pass logs and profiles](hierarchos-vulkan/validation/release_parity/deepseek_v41/receipts/final-push/)
preserve the measured runs in compressed form. `HANDOFF-MANIFEST.json` binds the
published payloads; Git attributes preserve their qualified bytes on Windows.
Machine-specific paths in historical receipts are provenance; the Git checkout
contains the Rust sources, shaders, frozen CPU fixtures and retained joint
checkpoint needed for a fresh Gen9 run.

Pull the handoff branch on the Gen9 machine:

```powershell
git fetch origin
git switch --track origin/codex/deepseek-v41-gen9-handoff
powershell -ExecutionPolicy Bypass -File hierarchos-vulkan/validation/run_deepseek_v41_gen9_gate.ps1
```

For a new clone:

```powershell
git clone --branch codex/deepseek-v41-gen9-handoff https://github.com/necat101/Hierarchos-Native.git
```

Keep frozen expectations unchanged. Record a separate Gen9 receipt and profile
that device before choosing kernel tiles, workgroup sizes or fusion changes.
The [Gen9 guide](hierarchos-vulkan/validation/intel-handoff/DEEPSEEK_V41.md)
contains the gate, profiling commands and arithmetic contracts.
