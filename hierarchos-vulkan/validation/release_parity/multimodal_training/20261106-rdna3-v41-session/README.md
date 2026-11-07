# Native V4.1 reverse-graph submission session

## Result

The retained native V4.1 reverse graph now records into one compute session.
The same binary, software-FMA arms and seven measured updates after two warmups
produce identical final logit words, image digests and adapter digests in every
run:

| Case | Baseline, ms | Session, ms | Ratio | Submissions/step |
| --- | ---: | ---: | ---: | ---: |
| `native_vit_4x4` | 369.3835 | 315.5736 | 1.171x | 982 -> 650 |
| `native_vit_6x4` | 449.6063 | 371.8268 | 1.209x | 982 -> 650 |
| `llama` control | 13.1841 | 13.3238 | 0.990x | 5 -> 5 |
| `gpt2` control | 10.6933 | 10.6168 | 1.007x | 5 -> 5 |
| `t5` control | 31.3479 | 31.1574 | 1.006x | 7 -> 7 |

Four ABBA rounds were used. The baseline arm is the same binary with
`HIERARCHOS_VULKAN_LAYER_SESSION=0`; the session arm is the default. Both arms
set `HIERARCHOS_VULKAN_RDNA3_V41=0` so the previously measured neutral
hardware-FMA candidate does not confound this ablation. The shared controls are
unchanged within noise and keep their original submission counts. These are
tiny frozen training geometries, not production-size throughput claims.

## Source change

`DeepseekV41Backbone::backward_retained` now opens one
`VulkanDevice::begin_compute_session` around the retained reverse graph and
lets `ComputeSessionScope::finish` retire it. The native adjoint stages obtain
their commands through `VulkanDevice::compute_batch`, so their existing
`submit` calls become publication barriers inside the session instead of
separate queue submissions and fence waits. No arithmetic, kernel selection,
dispatch order or buffer content changed.

Touched native adjoint paths:

- `src/deepseek_v41/backbone_tape.rs` (session wrapper)
- `src/deepseek_v41/backbone.rs`
- `src/deepseek_v41/attention_grad.rs`
- `src/deepseek_v41/attention_projection_grad.rs`
- `src/deepseek_v41/attention_tape.rs`
- `src/deepseek_v41/compressor_grad.rs`
- `src/deepseek_v41/engram_grad.rs`
- `src/deepseek_v41/linear_grad.rs`
- `src/deepseek_v41/mixed_moe_grad.rs`

## Verification

- Four ABBA rounds, seven measured steps, two warmups: every final logit,
  image digest and adapter digest matched across all eight runs.
- Native V4.1 qualifier: 30/30 tests pass, 1,360 logit values,
  `max_abs_logit = 0.0`, joint `max_abs = 1.4901161193847656e-08`,
  checkpoint/resume/export verified.
- `cargo check` and `git diff --check` pass.
- The nine-stage full replay was not rerun for this change; the 30-test native
  qualifier covers the modified V4.1 graph, and the benchmark rechecks the
  shared controls.

## Residual profile and next targets

`profile.log` records the session-on run. Per native step, after this change,
the remaining 650 submissions are dominated by 604.8 session retirements
(host-visible reads or uploads splitting a session). The largest attributed
funnels are:

| Funnel | Per step |
| --- | ---: |
| `upload_bytes` | 303.4 |
| `duplicate_f32` | 108.0 |
| `read_u32` (routing counts) | 100.0 |
| `read_f32` | 45.8 |
| `copy_buffer` at `vulkan.rs:4614` | 72.0 |

The next measured optimization targets, in order, are: batch or reuse the
parameter uploads so a step does not upload a freshly reconstructed model;
remove or batch the per-layer routing-count readback; avoid the per-layer norm
duplication or copy it inside the session; and group the remaining scalar
readbacks. A first attempt to make `copy_buffer` join the active session was
rejected: it failed the native projection-adjoint geometry gate with
`V4.1 projection adjoint has zero/overflowing geometry`, so that path needs an
explicit transfer-hazard design rather than a generic session join.

## Reproduce

```powershell
python validation/qualify_deepseek_v41_native_pipeline.py `
  --binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe `
  --output target/rdna3-port/full-multimodal-20261106/v41-session-native-qualifier `
  --target rdna3
```

```powershell
python validation/benchmark_rdna3_v41_session.py `
  --binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe `
  --fixtures target/rdna3-port/gen9-frozen/target/multimodal-gen9 `
  --output target/rdna3-port/full-multimodal-20261106/v41-session-receipt-r1 `
  --rounds 4 --steps 7
```

## Files and limits

- `benchmark.json`: four-round ABBA medians and submission counts.
- `runs/`: eight raw run JSON files and logs.
- `qualification.json`, `run-status.json`, `qualifier.log`: the 30-test native
  V4.1 gate.
- `profile.log`: session-on submission profile.
- `benchmark.log`: runner output.
- `receipt-hashes.json`: SHA-256 bindings for this receipt.

The receipt does not certify production-size throughput, a physical Gen9 run,
or the unimplemented residual optimizations above.
