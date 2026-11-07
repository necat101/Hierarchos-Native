# Batch-2 compressor state-gradient chain

The rebuilt test binary `e4384546d16e6bfb1c6fa97d9eb42917b68e8f157367188c2d98f15d5bf3e964` passes both complete qualification tracks
on the Ryzen Z1 Extreme / Radeon `1002:15bf` host. The strict acceptance gate
remains `max_abs < 2e-7`, `rtol = 0`.

## Added coverage

The existing decode snapshot test still covers batches 1 and 2. Batch 2 now
supplies distinct nonzero KV and score state gradients after the completed
position-3 step, then passes the returned state gradients into the pending
position-2 tape's backward call. Both token input gradients must respond to
the terminal state gradient.

Each batch row is also activated independently with no latent loss. Gradients
must remain zero in the other row, and neither overwritten slot may leak a
gradient into the prefill state. After the live norm gain and live pool cache
are overwritten, both token gradients and all four returned state-gradient
buffers must remain word-identical. Replaying forward after the gain overwrite
must change both token gradients, proving sensitivity to the captured gain.

The two existing compressor snapshot tests remain the same test identities:
243 enumerated DeepSeek tests, 217 active and 26 original opt-in tests. This
increment extends the test coverage and qualification tooling. The full native
replay also exposed a SigLIP full-grid regression (`2.1606683731079102e-7`)
caused by admitting the added GELU-backward rounding module to the native
AVX512 graph. Its gate now limits that addition to the frozen AVX2 track and
preserves the established native AVX512 graph selection. Both SigLIP tracks
pass after the correction; expectations and thresholds remain unchanged.
`siglip-dispatch-correction.json` retains the failing run and targeted replays.
`siglip-dispatch-scope.patch` reconstructs the failing source byte for byte
against its recorded SHA-256 and shows the admission-gate correction.

## Fresh validation

| Track | Result |
| --- | --- |
| Original frozen replay | All nine stages pass |
| Frozen image pipelines | 93 cases: shared 77, seq2seq 6, Gemma4 2, SigLIP 4, LLaVA 4 |
| Standalone text | 166 resident/streaming runs pass |
| DeepSeek | 217/217 active tests, 243 enumerated, 26 original ignored |
| Native V4.1 | 30/30 executions; logit maximum absolute error 0.0 |
| Separate native AVX512 image track | 95 retained cases, including the two private-AVX2 V4.1 bridge cases |
| Expanded GPT2 | 16 masked/unmasked executions at eight geometries |
| GPT2 primitives | 9 passed, 1 opt-in benchmark ignored |
| RDNA3 GELU/SigLIP primitive extensions | 2 passed |
| Compressor snapshot regressions | 2 passed, including the extended batch-2 chain |

The two image tracks retain their independent CPU contracts and expectations.
The native V4.1 host process uses AVX512 while its pinned private math scope
continues to use AVX2. Counts above are test executions and overlapping scopes,
not a sum of unique test cases.

## Gen9 and provenance

All 2,748 original frozen files remain unchanged. The frozen manifest is
`a9e519008e712105ceb1b135227398785617db3eb03c65e7518bb7f225966636`. The original 1,341 shader/fixture files and
672 shader recipes also match the upstream Gen9 baseline exactly. Physical
Intel execution is outside this RDNA3 run.

`source-sha256.json` binds 1531 source and artifact files. The replay
binds all six executables and verifies that they stay unchanged. `evidence.zip`
retains 2740 raw JSON/log files, including all bounded DeepSeek shard
logs and all per-case comparison receipts. Large exported SafeTensors remain
in the local work directory, with their hashes in `export-sha256.json`.

Five inherited expanded-GPT2 source bindings were stale. All 95 retained native
cases, 16 expanded shapes and nine primitive tests were rerun before refreshing
those bindings and the per-suite receipts. The entire previous expanded receipt
is preserved byte for byte in `previous-expanded-receipt/`;
`expanded-receipt-refresh.json` records the old/new bindings.

## Performance scope

This correctness extension introduces no new performance measurement or speedup claim.
`performance-reference.json` binds the prior streamed/resident ABBA benchmark
to its original binary in the [preceding receipt](../20261107-rdna3-v41-batch2-decode-snapshot/README.md).
That receipt's tiny-geometry measurements remain historical evidence for that
binary. Production-size throughput remains unqualified.

## Reproduce

Build serially before starting GPU jobs:

```powershell
cargo test --locked --lib --no-run
cargo build --locked --bin image_text_pipeline_probe --bin gemma4_vision_probe --bin gemma3_vision_probe --bin llava_pipeline_probe --bin hierarchos-vulkan-transformer-logits
python validation/qualify_rdna3_full_multimodal.py --frozen target/rdna3-port/gen9-frozen --test-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe --output NEW_FROZEN_REPLAY
python validation/qualify_rdna3_native_multimodal.py --test-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe --output NEW_NATIVE_REPLAY
```

Use fresh output directories and one GPU job at a time. The retained local run
is `target/rdna3-port/state-chain-final-20261106/`. Changes remain uncommitted.
