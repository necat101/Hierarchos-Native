# Multimodal image-training handoff to the Ally

All 44 previously green text profiles have joint image training implemented and
qualified on Gen9 at the frozen FP32 geometries with `max_abs < 2e-7`, `rtol=0`.
The final source passes 95 image cases and 166 text cases. Native towers are
preserved; other covered backbones use the authorized trainable shared image
encoder/projector. Native V4.1 trains its ViT, aligner and delimiters with the
established frozen-language-base LoRA contract. This handoff covers image and
text. The 2026-11-06 physical RDNA3 replay closes the six shared-profile gaps
and adds the native V4.1 bridge. The subsequent
[GPT2 shape expansion](hierarchos-vulkan/validation/release_parity/multimodal_training/20261106-rdna3-gpt2-shapes/README.md)
passes all 16 masked/unmasked GPT2 cases and retains the original 44-profile
inventory at the unchanged strict gate. Gen9 files remain read-only. Audio/video
and production-size throughput remain pending.

The branch is `codex/gen9-multimodal-training`. The GitHub prerelease
[`gen9-multimodal-20261105`](https://github.com/necat101/Hierarchos-Native/releases/tag/gen9-multimodal-20261105)
contains the frozen oracle archive and a source snapshot with SHA-256 manifests.
The qualification was recorded before publication; its historical notes saying
that no commit/push had been made describe that earlier run. GitHub publication
does not imply that the Ally's GPU has passed its own tests.

## Bring the source and expectations over

In the Ally checkout, fetch and switch to the branch without discarding local
changes. If this local branch does not already exist:

```powershell
git fetch origin
git switch --track origin/codex/gen9-multimodal-training
```

For an existing local branch, switch to it and use `git pull --ff-only`. Reconcile
any local work before proceeding. Download the oracle bundle without rerunning
the CPU freezers. From the repository root:

```powershell
gh release download gen9-multimodal-20261105 --repo necat101/Hierarchos-Native --pattern gen9-full-image-training-20261104-oracles.zip --dir target/multimodal-gen9
$multimodalOracleArchive = 'target/multimodal-gen9/gen9-full-image-training-20261104-oracles.zip'
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $multimodalOracleArchive).Hash -ne 'f52a62d55790728e232f45feac3ebce1f42d0d42844a34bcab9dfa3aa62b519a') { throw 'Oracle archive SHA-256 mismatch' }
$multimodalOracleStaging = 'target/multimodal-gen9/frozen-oracle-unpack-20261105'
Expand-Archive -LiteralPath $multimodalOracleArchive -DestinationPath $multimodalOracleStaging
Copy-Item -LiteralPath "$multimodalOracleStaging/target" -Destination . -Recurse -Force
Copy-Item -LiteralPath "$multimodalOracleStaging/hierarchos-vulkan" -Destination . -Recurse -Force
$env:HIERARCHOS_ATEN_VECTOR_WIDTH = '8'
```

Use a fresh staging directory. Copying only the two oracle-root folders preserves
the checkout's existing root `HANDOFF-MANIFEST.json`; the oracle archive's own
per-file manifest remains inside the staging directory.

The archive contains 2,748 unchanged generated files across seven frozen oracle
roots, plus `HANDOFF-MANIFEST.json` with per-file hashes. The native V4.1 fixture,
reference sources, shipped SPIR-V, implementation, maintained validators,
qualification JSON and failure/recovery logs are committed on the branch.
Build all binaries locally; laptop executables and local Python environments
are not part of the source distribution.

## Build and replay serially

Use the locked dependency set. From `hierarchos-vulkan/`:

```text
cargo test --locked --lib --no-run
cargo build --locked --bin image_text_pipeline_probe --bin gemma4_vision_probe --bin gemma3_vision_probe --bin llava_pipeline_probe --bin hierarchos-vulkan-transformer-logits
```

On a new machine, Cargo may need network access to obtain the locked packages.
For tests that write traces, create `target/deepseek-v41-reference` next to the
compiled crate directory. Pass the newly built libtest executable explicitly;
do not select an older binary by a broad filename glob.

From the repository root, run the native qualifier, then the paired throughput
comparison with fresh output directories and no concurrent GPU jobs:

```text
python hierarchos-vulkan/validation/qualify_deepseek_v41_native_pipeline.py --binary PATH_TO_NEW_LIBTEST_EXE --output NEW_RDNA3_NATIVE_DIRECTORY --target rdna3 --device-id 0x15bf
python hierarchos-vulkan/validation/benchmark_deepseek_v41_training.py --binary PATH_TO_NEW_LIBTEST_EXE --fixtures target/multimodal-gen9 --output NEW_RDNA3_BENCHMARK_DIRECTORY --rounds 2 --steps 7
```

The native qualifier uses the local Vulkan GPU and pins its private AVX2 math
contract. Run all 77 shared-image, 6 encoder-decoder, 10 separate native-tower
and 166 text cases using the qualifiers and frozen-root table in the
[baseline reproduction guide](hierarchos-vulkan/validation/release_parity/multimodal_training/20261104-gen9-full-image-training/README.md#reproduction).
Use `--target rdna3` for those matrix qualifiers, retain complete receipts and
preserve their strict limits. A successful build or laptop receipt alone does
not qualify the physical Ally.

## RDNA3 completion receipt

The physical replay now covers the six previously missing shared profiles and
the seventh native V4.1 profile. The six shared profiles pass 9/9 cases; native
V4.1 passes 2/2 cases and 1,360 logits. The inventory is 44/44 fully green and
85/85 image cases pass; `rdna3_multimodal_qualified` is set. The GPT-2-wide
forward tape is bit-exact and the backward attention seed now matches the
forward SLEEF/muladd reconstruction, closing the former residual. The complete
receipt, including the 30-execution native V4.1 qualifier, is in
[`validation/release_parity/multimodal_training/20261106-rdna3-profile-completion/`](hierarchos-vulkan/validation/release_parity/multimodal_training/20261106-rdna3-profile-completion/README.md).
The [full Gen9-green replay](hierarchos-vulkan/validation/release_parity/multimodal_training/20261106-rdna3-full-gen9-replay/README.md)
adds the complete published matrix: 77 shared-image, 6 seq2seq, 2 Gemma4,
4 Gemma3, 4 LLaVA, 166 text runs, 215 active DeepSeek tests and 30 native V4.1
tests all pass on `1002:15bf` at the unchanged gate. The 2,748-file frozen
manifest is unchanged; no physical Gen9 execution is claimed.

## Gen9 results to preserve

The [latest qualification and replay notes](hierarchos-vulkan/validation/release_parity/multimodal_training/20261105-gen9-v41-upload-batching/README.md)
and [44-profile inventory](hierarchos-vulkan/validation/multimodal_training_inventory.json)
bind the implemented coverage. The final native qualifier executes 30 tests;
all 1,360 native logits are word-exact. Its broader suite verifies 215 tests,
with 26 opt-in tests ignored. One initial trace-write failure was recovered by
creating a missing output directory and rerunning the sole case; both logs are
retained. All 672 shader recipes rebuild/validate, 81 historical V4.1 SPVs and
127 CPU fixtures are unchanged, and all generated oracle files remain unchanged.

Native V4.1's upload batching reduces submissions per timed step from 1,102 to
982. Paired same-binary ABBA measurements show 1.052x and 1.067x throughput at
the two frozen Gen9 geometries, with identical trained digests and logit words
after nine updates. These unoptimized libtest measurements do not predict
production-size or RDNA3 throughput. The Ally replay recorded its own results
in the full receipt; its native hardware-FMA candidate was word-exact but
performance-neutral, so no further optimization was selected on that basis. On the same machine, model construction and the retained V4.1 graph now share
one compute session, component uploads publish into it, and forward routing
counts are reused by reverse. Submissions fall from 970 to 509 per step, with
four-round ABBA gains of 1.295x at 4x4 and 1.217x at 6x4, word-exact and with
unchanged shared controls. Routing-count readbacks fall from 100 to 64 per
step. The [final receipt](hierarchos-vulkan/validation/release_parity/multimodal_training/20261106-rdna3-v41-final/README.md)
binds the 30-test native qualifier (`max_abs_logit = 0.0`), the full nine-stage
replay (2,748 frozen files unchanged), and the remaining upload/readback
profile. A later
[session-wide upload receipt](hierarchos-vulkan/validation/release_parity/multimodal_training/20261106-rdna3-v41-uploads/README.md)
records the next increment: uploads, metadata copies and tape snapshots record
into the open session, and submissions fall to 241 per step at
1.464x/1.366x, with the 30-test qualifier green. Its attention norm-gain
sharing was later reverted because it broke the retained tape's snapshot
contract. The
[resident-expert measurement](hierarchos-vulkan/validation/release_parity/multimodal_training/20261106-rdna3-v41-resident-experts/README.md)
shows the remaining routing-count reads are per-layer dependencies under
streamed experts, and that resident experts remove 72 of 241 submissions at
1.175x/1.125x, bit-exact with the 30-test qualifier at
`max_abs_logit = 0.0`. Residency stays a benchmark-only switch. An
[adapter-readback batching trial](hierarchos-vulkan/validation/release_parity/multimodal_training/20261106-rdna3-v41-adapter-read-batching/README.md)
reduced the 1,008 adapter reads to 28 but regressed wall time at every tested
submission cadence (best 6x4 resident 296.6 ms vs 273.4 ms), so it was reverted.
Treat the per-group reads as pacing points, not a bandwidth bottleneck.
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