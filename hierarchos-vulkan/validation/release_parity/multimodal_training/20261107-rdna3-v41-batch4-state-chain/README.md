# Batch-4 compressor state-gradient chain

The rebuilt test binary `2e9cfaf1e7bc9c1242977f0e6d0f14aa00672560bc3574ced2a836737d98aedd` passes both complete qualification tracks
on the Ryzen Z1 Extreme / Radeon `1002:15bf` host. The strict acceptance gate
remains `max_abs < 2e-7`, `rtol = 0`.

## Added coverage

The existing `deepseek_v41_compressor_decode_tape_snapshots_norm_gain` test now
runs the same snapshot identity at batches 1, 2 and 4. For every batch above 1,
distinct nonzero KV and score state gradients are chained through the completed
position-3 tape and the pending position-2 tape. Both token input gradients must
respond to the terminal state gradient; a dropped state argument fails the test.

Each row is then activated independently with no latent loss. At batch 4 this
checks all four rows: the active row must carry a state VJP, every other row must
remain exactly zero, and neither overwritten pool slot may leak into prefill.
After the live norm gain and live pool cache are overwritten, both token VJPs
and all four state VJPs must remain word-identical, and a restarted forward pass
must show that both token VJPs still respond to the changed gain. The exact
batch-4 observation is:

```
batch 4 state chain: both token VJPs and four state VJPs word-exact after live gain/cache overwrite; isolated rows and overwritten slots verified
```

No oracle, expectation or threshold was changed. The two compressor snapshot
tests remain the same test identities: 243 enumerated DeepSeek tests, 217 active
and 26 original opt-in tests.

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
| Compressor snapshot regressions | 2 passed, including batch 2 and batch 4 state chains |

The two image tracks retain their independent CPU contracts and expectations.
The native V4.1 host process uses AVX512 while its pinned private math scope
continues to use AVX2. Counts above are test executions and overlapping scopes,
not a sum of unique test cases.

## Gen9 and provenance

All 2,748 original frozen files remain unchanged. The frozen manifest is
`a9e519008e712105ceb1b135227398785617db3eb03c65e7518bb7f225966636`. The original 1,341 shader/fixture files and
672 shader recipes also match the upstream Gen9 baseline exactly. Physical Intel
execution is outside this RDNA3 run.

`source-sha256.json` binds 1531 source and artifact files. The only
refreshed binding is the batch-4 test source; `source-changes.json` records its
previous and current SHA-256. The replay binds all six executables and verifies
that they stay unchanged. `evidence.zip` retains 2751 raw JSON/log
files, including all bounded DeepSeek shard logs and all per-case comparison
receipts. Large exported SafeTensors remain in the local work directory, with
their hashes in `export-sha256.json`.

The prior [state-chain receipt](../20261106-rdna3-v41-state-chain/README.md)
remains sealed for its original binary and is not rewritten. This receipt is the
current batch-4 evidence; `previous-receipt.json` records the prior manifest hash.

## Performance scope

This correctness extension introduces no new performance measurement or speedup
claim. `performance-reference.json` binds the prior paired streamed/resident
benchmark to its original binary in the
[preceding batch-2 snapshot receipt](../20261107-rdna3-v41-batch2-decode-snapshot/README.md).
Production-size throughput remains unqualified.

## Reproduce

Build serially before starting GPU jobs:

```powershell
cargo test --locked --lib --no-run
cargo build --locked --bin image_text_pipeline_probe --bin gemma4_vision_probe --bin gemma3_vision_probe --bin llava_pipeline_probe --bin hierarchos-vulkan-transformer-logits
python validation/qualify_rdna3_full_multimodal.py --frozen target/rdna3-port/gen9-frozen --test-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe --output NEW_FROZEN_REPLAY
python validation/qualify_rdna3_native_multimodal.py --test-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe --output NEW_NATIVE_REPLAY
```

Use fresh output directories and one GPU job at a time. The retained local run is
`target/rdna3-port/state-chain-batch4-20261107/`.
