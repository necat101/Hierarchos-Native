# RDNA3 AVX-512 multimodal port receipt

Date: 2026-11-06
Target: `AMD Radeon Graphics`, PCI `1002:15bf` (RDNA3 / gfx11)
Host CPU: AMD Ryzen Z1 Extreme, AVX-512, `aten_vector_width() == 16`
Driver: `26.10.07.06 (LLPC)`, Windows `32.0.31007.6002`

## GPT2 shape expansion, 2026-11-06

The [expanded receipt](hierarchos-vulkan/validation/release_parity/multimodal_training/20261106-rdna3-gpt2-shapes/README.md)
adds 16 GPT2 image-training cases: batch 1/2/3, sequence 6/8/10/16,
and 2/4/8 heads at hidden width 64, with masked and unmasked inputs.
These are eight measured geometries, not every combination of those dimensions.
All pass the unchanged strict `max_abs < 2e-7`, `rtol=0` gates through two
AdamW updates, RGB gradients, parameter export and reload. Both control cases
reproduce the original CPU logits word-for-word before GPU qualification.

The retained GPT2 CPU oracle uses **SDPA**. Earlier operation-local eager
diagnostics did not establish that backend contract. The corrected diagnostics
include padding masks and require frozen CPU replay before interpreting stages.
Separate RDNA3 modules now match CPU flash SDPA, CRT scalar log, MKL tanh and
AdamW square root, and the sum operator's Vec8 cascade schedule. CPU capability
`AVX512` does not imply that every operator uses sixteen lanes.

Nine primitive tests check independent CPU words, including 8,192 NewGELU
values, 80 bias-reduction geometries and 36 optimizer updates. The instruction
seed table covers all mantissas in its two normalized exponent bins; it is not
derived from model inputs. Native FMA, padded LDS attention storage and measured
workgroup sizes are confined to the qualified Windows Z1 Extreme / driver
profile. Other adapters and unmeasured attention geometries retain their fallback.

Forward workgroup sizes were swept at 1/2/4/8/16/32/64 threads. The measured
batch-2 / sequence-8 / four-head path uses eight threads; other forward shapes
use serial rows. Backward shares probabilities and score gradients in LDS.
The receipt records paired ABBA GPU timings and exact output identity; those
isolated kernel results do not establish production-size training throughput.

The final replay also retains 83 shared/Kimi/seq2seq cases, ten native
Gemma/LLaVA cases and the native V4.1 tests and bridge. The original 44-profile
inventory and its V4.1 frozen-language LoRA contract remain intact. All 1,341
upstream shader/fixture files and 672 original recipes remain unchanged.
The receipt guard is wired into the existing hook and Windows CI.

## Full Gen9-green replay, 2026-11-06

The [full replay receipt](hierarchos-vulkan/validation/release_parity/multimodal_training/20261106-rdna3-full-gen9-replay/README.md)
replays every published Gen9 multimodal stage on the physical adapter:
77 shared-image cases, 6 seq2seq cases, 1 legacy Gemma4, 1 RGB Gemma4,
4 Gemma3, 4 LLaVA, 166 resident/streaming text runs, 215 active DeepSeek
tests and 30 native V4.1 tests. All nine stages pass at the unchanged strict
gate. The 2,748-file frozen manifest is unchanged, and no physical Gen9 run is
claimed. The native V4.1 hardware-FMA candidate is word-exact but
performance-neutral in a four-round ABBA test (`0.996x`/`1.000x`), so it is not
recorded as a speedup. The final
[submission and routing receipt](hierarchos-vulkan/validation/release_parity/multimodal_training/20261106-rdna3-v41-final/README.md)
records the measured native optimization: 970 to 509 submissions per step and
1.295x/1.217x throughput at 4x4/6x4, word-exact, with shared controls
unchanged. Forward routing counts are reused instead of re-read, and the full
nine-stage replay passes on the same binary with all 2,748 frozen files
unchanged. Its residual profile lists the next upload/readback targets.

The later [session-wide upload receipt](hierarchos-vulkan/validation/release_parity/multimodal_training/20261106-rdna3-v41-uploads/README.md)
removes that funnel: uploads, metadata copies and tape snapshots record into the
open session. Attention norm gains were briefly shared rather than
duplicated; that sharing was reverted. Submissions fall to 241 per step at 1.464x/1.366x, word-exact, with the 30-test
qualifier green and the shared controls unchanged.

The [resident-expert measurement](hierarchos-vulkan/validation/release_parity/multimodal_training/20261106-rdna3-v41-resident-experts/README.md)
records the routing-readback limit: per-layer gate counts select streamed
expert payloads, so they cannot collapse into one transfer per step, and the
fused packed batch refuses on this adapter. Resident experts instead remove 72
of 241 submissions at 1.175x/1.125x, bit-exact and with the 30-test qualifier
at `max_abs_logit = 0.0`; the production default stays streamed. The
[adapter-readback batching trial](hierarchos-vulkan/validation/release_parity/multimodal_training/20261106-rdna3-v41-adapter-read-batching/README.md)
cut the 1,008 reads to 28 but was slower at every submission cadence (best 6x4
resident 296.6 ms vs 273.4 ms), so it was reverted; the reads pace host
recording against device execution.

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

## Result

The shared image-text training path is green for all four Gemma3 SigLIP cases
under the unchanged acceptance policy `max_abs < 2e-7`, `rtol = 0`:

| case | step-1 logits | step-2 logits | final logits | gate |
| --- | ---: | ---: | ---: | --- |
| `siglip_default` | 8.196e-8 | 6.706e-8 | 1.378e-7 | PASS |
| `siglip_batch_padded` | 5.600e-8 | 9.686e-8 | 8.941e-8 | PASS |
| `siglip_full_grid` | 0.0 | 1.267e-7 | 1.304e-7 | PASS |
| `siglip_wider` | 1.416e-7 | 1.714e-7 | 1.527e-7 | PASS |

The qualification command is:

```
python validation/qualify_gemma3_pipeline.py run \
  --fixtures target/rdna3-port/cpu-gemma3 \
  --output target/rdna3-port/cpu-gemma3/siglip-eager2 \
  --binary target/debug/gemma3_vision_probe.exe \
  --target rdna3 --device-id 0x15bf
```

The non-required `vision_features` diagnostics after the two updates remain
between 0.6e-6 and 1.7e-6. They are retained as diagnostics in the receipt, but
they are not the logit gate requested for this port.

## What changed

The work is additive and selected only on an RDNA3 adapter with the AVX-512
host width. The Gen9 shaders, routes, fixtures and tests were not edited.

- SigLIP attention has a dedicated AVX-512/RDNA3 module for sequence lengths
  `4`, `5`, `16` and `17`. It uses the measured query-key order: materialized
  products below 16 tokens and serial hardware FMA from 16 tokens upward.
- The Gemma3 projector RMSNorm has a dedicated module for the two measured
  pooled-row topologies: 16 rows use four interleaved reduction streams, 8 rows
  use sixteen. The weight offset is `(1 + weight)` and the reciprocal square
  root follows the measured CPU path.
- The RDNA3 RoPE VML trig path no longer falls through to SLEEF above eight
  radians. The existing VML polynomial matches the CPU words over the tested
  0-32 radian range, including position 17 / frequency 0.
- Gemma3's eager text-attention admission now covers two KV heads with an
  8-wide head, sequence lengths `8`, `10` and `20`, and the model's 64-token
  sliding window when that window is a no-op for the sequence. This was the
  final full-grid fix: the final logit error moved from `2.235e-7` to
  `1.304e-7` without changing the gate.

The new modules are gated by `vendor::rdna3_avx512_variants`, separate from
`rdna3_vision_variants` and the Gen9 routes. Other adapters, including Gen9,
keep the committed kernels.

## Shared-backbone regression

The 74-case `cpu-shared-v2` replay completed with all 74 cases passing. Both
GPT-2-wide cases are green: unmasked final `1.7881393432617188e-07`, padded
final `1.7881393432617188e-07`. The forward stage tape is bit-exact against
eager ATen; the last training drift came from the backward attention seed and
was closed by matching the same QK muladd, SLEEF-u10 exp, exact reciprocal and
serial exp sum in the GPT-2 backward kernel. No tolerance or CPU expectation
was changed.

## Six-profile replay and native V4.1 bridge

The six shared profiles that were absent from the earlier retained fixture set
were replayed on the physical adapter and all pass:

| profile | cases | worst visible logit | gate |
| --- | ---: | ---: | --- |
| `bart` | 2 | 6.705522537231445e-08 | PASS |
| `t5` | 2 | 4.470348358154297e-08 | PASS |
| `switch_transformers` | 2 | 5.960464477539063e-08 | PASS |
| `kimi_k3_kda` | 1 | 2.9802322387695312e-08 | PASS |
| `kimi_k3_mla` | 1 | 2.2351741790771484e-08 | PASS |
| `kimi_k3_text` | 1 | 3.166496753692627e-08 | PASS |

The seventh inventory profile, `deepseek_v41_scalable`, is not a CLIP text
backbone. Its native V4.1 fixture is bound to the shared inventory by
`validation/qualify_deepseek_v41_shared_image.py`; it is not aliased to the
generic CLIP adapter. The dedicated `DeepseekV41MultimodalTrainer` joint test
passes 2/2 cases, 4 optimizer updates, 1,360 logits, and 1,866 comparisons over
155,444 values (155,179 bit-exact; worst non-exact `1.4901161193847656e-08`).

The inventory now has a case source for all 44 profiles: all 44 are fully green
and 85/85 image cases pass. `rdna3_multimodal_qualified` is true. The GPT-2-wide
closure, including the bit-exact forward tape and the aligned backward seed, is
recorded in `gpt2-wide-residual.json`. The full receipt is
[`validation/release_parity/multimodal_training/20261106-rdna3-profile-completion/`](hierarchos-vulkan/validation/release_parity/multimodal_training/20261106-rdna3-profile-completion/README.md).

## Isolation

`validation/verify_rdna3_multimodal_isolation.py` was run against baseline
`528e3f722bc609cb2d0bdc5dcb032141c662dc5b`:

- 1,341 upstream Gen9 shader and fixture files checked;
- 672 pre-existing shader recipes checked;
- zero changed files or recipes.

This is a static preservation check. No physical Intel Gen9 adapter is present
on this host, so Gen9 execution was not repeated.

## Tests and provenance

The new raw-word test is:

```
cargo test --locked --lib rdna3_avx512_vision_attention_projector_cpu_words \
  -- --nocapture --test-threads=1
```

It passed on the physical adapter. The fixture
`validation/fixtures/rdna3_vision_avx512.json` contains the actual Windows
Torch 2.6.0+cpu AVX-512 stage words for the full-grid and batch-padded SigLIP
attention, plus the 16-row and 8-row projector RMSNorm. The test checks raw
`to_bits()` equality, not a tolerance.

At the SigLIP stage, `validation/training_shader_manifest.json` recorded 27 new
artifacts for the AVX-512/RDNA3 family. All 27 reproduced from their recorded sources with
`glslangValidator 15.3.0` and target `vulkan1.0`; the residual RMS artifact
needs both `-DRDNA3_RMS_BACKWARD` and `-DRDNA3_RMS_RESIDUAL`. The new
attention and projector artifacts are:

```
vision_attention_avx512_amd_rdna3.spv
vision_projector_rms_avx512_amd_rdna3.spv
```

The existing text-attention test remains in
`src/transformer/rdna3_text_attention_tests.rs`; the new vision test is in the
same module.

## Limits

- No commit was created. The worktree contains the additive source, SPIR-V,
  fixtures and receipts.
- No Gen9 execution was performed; preservation is established statically.
- The shared suite is all-green: 44/44 profiles and 85/85 image cases pass.
  The six previously missing shared profiles, native V4.1 and the GPT-2-wide
  closure are recorded in the 2026-11-06 receipt; Gen9 zero-drift GPT-2 values
  are not copied into RDNA3.
- No end-to-end speedup claim is made for the new path.
