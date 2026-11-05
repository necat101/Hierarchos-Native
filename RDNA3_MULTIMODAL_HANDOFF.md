# Multimodal image-training handoff to the Ally

All 44 previously green text profiles have joint image training implemented and
qualified on Gen9 at the frozen FP32 geometries with `max_abs < 2e-7`, `rtol=0`.
The final source passes 95 image cases and 166 text cases. Native towers are
preserved; other covered backbones use the authorized trainable shared image
encoder/projector. Native V4.1 trains its ViT, aligner and delimiters with the
established frozen-language-base LoRA contract. This handoff covers image and
text. Audio/video, production-size throughput and physical RDNA3 qualification
remain pending.

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
python hierarchos-vulkan/validation/qualify_deepseek_v41_native_pipeline.py --binary PATH_TO_NEW_LIBTEST_EXE --output NEW_RDNA3_NATIVE_DIRECTORY
python hierarchos-vulkan/validation/benchmark_deepseek_v41_training.py --binary PATH_TO_NEW_LIBTEST_EXE --fixtures target/multimodal-gen9 --output NEW_RDNA3_BENCHMARK_DIRECTORY --rounds 2 --steps 7
```

The native qualifier uses the local Vulkan GPU and pins its private AVX2 math
contract. Run all 77 shared-image, 6 encoder-decoder, 10 separate native-tower
and 166 text cases using the qualifiers and frozen-root table in the
[baseline reproduction guide](hierarchos-vulkan/validation/release_parity/multimodal_training/20261104-gen9-full-image-training/README.md#reproduction).
Use `--target rdna3` for those matrix qualifiers, retain complete receipts and
preserve their strict limits. A successful build or laptop receipt alone does
not qualify the physical Ally.

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
production-size or RDNA3 throughput. The Ally replay should record its own
results before selecting further optimizations.
