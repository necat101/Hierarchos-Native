# Phase XIX-Z packed-cell recovery - AMD RDNA3

Source revision: `b03bf0919ea0b3bfb4fcbc6964674131bbc4464d`
Device: `AMD Radeon Graphics [pci=1002:15bf;driver=00800184]` (adapter `amd-rdna3-lds64`)
Result: **13/13 applicable routes PASS** (benchmark exit 0 and route guard exit 0).

The original partial receipt is preserved unchanged at `../20261112-phase19z-current-source/packed-cell/status.json` (sha256 `bad4795ba44d8ff17b31c9f06bf29c5de5a0073362f2d0e2a202043ca138bde8`).

## Route results

| Route | Selected at batches | Numeric worst | Bench / guard |
|---|---|---:|---|
| `time-mix-projection` | 1:fused, 4:fused, 8:fused, 16:fused, 32:fused | 0 | 0 / 0 |
| `channel-mix` | 1:fused, 4:fused, 8:composed, 16:composed, 32:fused | 0 | 0 / 0 |
| `low-rank-forward` | 1:fused, 4:fused, 8:fused, 16:fused, 32:fused | 1.29e-05 | 0 / 0 |
| `low-rank-forward-fp16` | 1:fused, 4:fused, 8:fused, 16:fused, 32:fused | 2.38e-07 | 0 / 0 |
| `key-state-forward` | 1:narrow-fused, 4:narrow-fused, 8:narrow-fused, 16:narrow-fused, 32:narrow-fused | 0 | 0 / 0 |
| `tbptt-packed-forward` | 1:legacy, 4:packed, 8:packed, 16:packed, 32:packed | 0 | 0 / 0 |
| `tbptt-packed-rematerialization` | 1:legacy, 4:legacy, 8:legacy, 16:legacy, 32:legacy | 0 | 0 / 0 |
| `tbptt-packed-forward-fp16` | 1:packed, 4:packed, 8:packed, 16:packed, 32:packed | 0 | 0 / 0 |
| `tbptt-packed-rematerialization-fp16` | 1:legacy, 4:legacy, 8:legacy, 16:legacy, 32:legacy | 0 | 0 / 0 |
| `tbptt-backward-fusion` | 1:folded, 4:staged, 8:folded, 16:folded, 32:fused | 0 | 0 / 0 |
| `tbptt-backward-fusion-fp16` | 1:folded, 4:folded, 8:folded, 16:folded, 32:fused | 0 | 0 / 0 |
| `tbptt-time-mix` | 1:fused, 4:fused, 8:fused, 16:fused | 0 | 0 / 0 |
| `packed-cell-composition` | batch 8 qualified amd | 0 | 0 / 0 |

All rows report `worst_max_abs = 0.0` where the producer promises exact parity; the low-rank and FP16 routes retain their stated non-zero tolerances.

## Repairs

- **Key/state exact parity:** the wide fused kernel now reads the transform outputs from global buffers after `memoryBarrierBuffer` instead of staging all state operands in shared memory. That removes the RDNA3 one-ulp contraction difference; `new_state`, `tmix`, `scaled_k`, `kk` and `saved_sa` all match the two-dispatch chain exactly.
- **Time-mix projection exact parity:** the same key/state path feeds the composed arm, so the previous 1.144e-05 gradient difference is gone (`worst_max_abs = 0.0` on the full sweep).
- **Channel-mix routing:** a recorded `staged` verdict may now build the portable composition even where the vendor matmul route did not win; the vendor gate still decides unrecorded geometries. The stale AMD batch-32 `staged` entry was replaced by fresh 30-sample measurements (batch 8/16 staged, batch 1/4/32 fused).
- **TBPTT routing:** the guard now reads explicit `fp32` scopes exactly as the runtime does, filters consumer chain entries by the receipt policy, and does not compare a consumer window's timings against the producer record. TBPTT forward/rematerialization/FP16 records were re-recorded from the controlled sweeps; rematerialization now records `legacy` at every measured geometry.
- **Windows Safetensors:** shared fixture replacement now materializes tensors, writes a verified sibling temporary file, releases the mmap, and atomically replaces the destination. Regression tests cover a repeated package write and a rewrite after loading.

## Verification

- `python -m unittest validation.test_measured_producer_routes`: 51/51 PASS.
- `python -m unittest validation.test_safetensors_atomic_replace`: 2/2 PASS.
- All 13 benchmark/guard pairs above: PASS.

## Remaining blockers

- Source-bound Phase XIX-Z qualification is **FAIL**: the recovery changed `shaders/rwkv_key_state_forward_fused_wide.comp`, so the old PASS receipt is not valid for this source. Training/stability/held-out/drift/activation were not regenerated in this window.
- `verify_training_shaders.py` is **FAIL** because the installed glslang/glslc versions (15.3.0 / 2025.3) differ from the committed inventory versions (16.6.0 / 2023.8). The recovery shader itself reproduces byte-for-byte with the installed glslc.
- GitHub Actions run 43130590148 is still **queued**: no self-hosted Windows Vulkan AMD runner is registered. No CI success is claimed.
- The batch-1 TBPTT packed-forward crossover is not stable outside the 2% band across paired runs; the current record follows the latest controlled 100-sample measurement.

## Artifacts

- `packed-cell/status.json` - machine-readable route matrix, hashes, exits and limitations.
- `source-bindings.json` - revision, changed-source hashes, shader/binary hashes and preserved baseline hash.
- `packed-cell/*.json` / `*.guard.log` - raw receipts and guards.
- `packed-cell/logs/` - benchmark transcripts.
- `negative-controls.log`, `windows-safetensors.log`, `check_phase19z_current.log`, `verify_training_shaders.log`.
