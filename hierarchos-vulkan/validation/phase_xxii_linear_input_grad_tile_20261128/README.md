# Phase XXII — FP32 `linear_input_grad` output tiling

**Status: GO — promoted in v0.2.8.**

This receipt records one bounded experiment and the production integration that
followed it. The experiment started from the committed Phase XX tree,
`468a7ceada598dd613162323604f4240f1b411c6`. Nothing else was promoted, no other
kernel was changed, and no profiling campaign was re-run.

---

## 1. Target verification

Phase XXI attributes **6.453 ms/step over its 29 profiled batches** (5.948 ms
over the 26 measured batches) to the shader identity `linear_input_grad`, with
17 dispatches per step. The production source is
`transformer.rs::Linear::record_backward_impl`, whose ordinary path dispatches
`shaders/linear_input_grad.spv` with push constants
`rows / input_dim / output_dim` and grid
`[ceil(input_dim / 16), ceil(rows / 16), 1]`.

The profiler's grid reading of `[64, 8, 1]` fixes `input_dim = 1024` and
`rows = 128`; those are exactly the 8q/2kv fixture's batch 2 × sequence 64 rows
and its 1024-wide hidden state. The candidate therefore targets the verified
128 × 1024 × 1024 operation. The same kernel also serves the 1024 → 256
projections, so the promoted eligibility covers every 32-aligned shape and each
output keeps the original reduction.

## 2. Candidate

`linear_input_grad_output_tile32.comp`:

* one workgroup owns a **32-row × 32-input-column** output tile;
* 256 threads, four outputs each — rows `ly` and `ly + 16`, columns `lx` and
  `lx + 16`;
* both shared operand tiles are padded by one float per row, so the rows a wave
  reads for one reduction column land on distinct banks;
* every output still performs
  `sum = fma(grad_output[row][k], weight[k][col], sum)` for `k` ascending, in
  FP32, with one accumulator and no reassociation. The candidate changes the
  tile and the operand staging only.

The original 16 × 16 kernel is retained as the control and fallback. During the
experiment the candidate was reachable only through a `#[cfg(test)]` seam; it is
now the default for shapes where `rows`, `input_dim` and `output_dim` are all
multiples of 32. `HIERARCHOS_LINEAR_INPUT_GRAD_OUTPUT_TILE=0` restores the
original kernel for every shape. A failure to load the candidate artifact also
falls back to the original kernel.

## 3. Numerical parity

`raw/parity.log` — the shipped kernel and the candidate are dispatched on
identical operands and every output word is compared bitwise:

| case | rows × input × output | elements | bitwise | sentinel | non-finite |
| --- | --- | ---: | ---: | ---: | ---: |
| measured | 128 × 1024 × 1024 | 131,072 | 131,072 | 0 | 0 |
| projection | 128 × 1024 × 256 | 131,072 | 131,072 | 0 | 0 |
| square | 64 × 128 × 128 | 8,192 | 8,192 | 0 | 0 |
| boundary | 33 × 37 × 45 | 1,221 | 1,221 | 0 | 0 |
| narrow | 5 × 96 × 160 | 480 | 480 | 0 | 0 |
| degenerate | 1 × 1 × 1 | 1 | 1 | 0 | 0 |
| **total** | | **272,038** | **272,038** | **0** | **0** |

The full-step comparison is the 29-step `parameter_end_digest`
`d79a12d882fd780deaada83a940374ab753b0a3bc5586a7ca348735ed29c4655`, identical in
every arm and every round below.

## 4. Predeclared acceptance rule

`00_acceptance_criterion.md`, written before measurement: bitwise parity, the
identical end digest, five rounds of `A B B A`, 3 warmup + 26 measured steps per
process, and the Phase XIX gate — 95 % CI entirely below 1.0, at least 4/5 rounds
improved, median ratio ≤ 0.990. No post-hoc threshold was used.

## 5. Complete-step ABBA

`raw/experiment/` preserves the pre-promotion seam measurement; `raw/` holds the
promoted-selector measurement. Both are 5 rounds × 4 processes, unprofiled,
`HIERARCHOS_WMMA_MODE=off`, same fixture and seed.

| run | arm A | arm B | A (ms) | B (ms) | B/A | speedup | rounds | 95 % CI |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| experiment (test seam) | original kernel | candidate via seam | 24.882 | 21.711 | 0.872166 | **1.1466×** | 5/5 | 0.869681–0.873909 |
| promoted | `…OUTPUT_TILE=0` | production default | 24.875 | 21.716 | 0.872420 | **1.1462×** | 5/5 | 0.870612–0.874241 |

Per-round ratios, promoted run:

```text
0.872420 0.870514 0.874338 0.871645 0.873216
```

Every arm-B process logs `[phase-xxii-tile] dispatch rows=128 input_dim=1024
output_dim=1024` exactly once; no arm-A process logs it. All 20 processes exit 0.
The parameter-end digest is identical across both arms and all five rounds.

## 6. Production integration

| file | change |
| --- | --- |
| `shaders/linear_input_grad_output_tile32.comp` / `.spv` | the promoted artifact (`fb948297…` / `9316501e…`) |
| `src/transformer.rs` | embedded artifact, lazy loader, 32-aligned selector, opt-out, fallback, dispatch |
| `src/phase_xxii_linear_input_grad_tile_tests.rs` | bitwise parity and selector tests |
| `validation/training_shader_manifest.json` | artifact, compiler flags, target environment |
| `validation/phase_xxiii_v028_release_20261128/` | package manifest pins the artifact |

The promotion was not combined with WMMA, attention, RMSNorm, activation
conversion or any other change.

## 7. Reproduction

```text
cd hierarchos-vulkan
cargo test --release --lib --no-run
BIN=$(ls -t target/release/deps/hierarchos_vulkan-*.exe | head -1)

env -u HIERARCHOS_LINEAR_INPUT_GRAD_OUTPUT_TILE \
  "$BIN" phase_xxii_linear_input_grad_output_tile_parity --nocapture
PHASE_XXII_TEST_BIN="$BIN" \
  bash validation/phase_xxii_linear_input_grad_tile_20261128/run_abba.sh
```

Device: AMD Radeon Graphics, integrated RDNA3, PCI `1002:15bf`, driver
`32.0.32015.14001`, Windows 11 x64. Compiler: `rustc 1.93.0`, `glslc` shaderc
2023.8. No profiler was active during any ABBA process.
