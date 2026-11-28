# Phase XXI — post-promotion component profile and next-target discovery

**Status: COMPLETE — NEXT TARGET IDENTIFIED.**

No optimization was implemented, no production behaviour was changed, and no
shader, dispatch, precision policy or tolerance was modified. This receipt is a
measurement and a ranking, nothing else.

The Phase XX promotion is committed at
`468a7ceada598dd613162323604f4240f1b411c6` (`Promote RDNA3 query scratch-reuse
attention path`). This profile measures that committed tree, with the production
scratch-reuse route selected by the default environment.

---

## 1. Profiling question and methodology

**Question.** With query scratch reuse now on the production path, where does a
complete training step actually spend its GPU time, and which component is the
best-supported next optimization target?

**Method.**

* Fixture: `target/gqa-e2e/fixture-8q2kv` — 2 layers, hidden 1024, 8 query heads
  / 2 KV heads, head dim 128, intermediate 1024, vocab 1024, 32 sequences,
  batch 2, sequence 64, causal LoRA (rank 8 / alpha 16), lr `1e-2`, seed 7.
  Identical model, batch and workload to the Phase XIX and Phase XX evidence.
* Driver: the established complete-step harness
  `phase_xii_channel4_key_full_step`, which pins **no** query switch. Warmup 3,
  29 steps per process, `HIERARCHOS_PHASE_XII_STEP_DIGEST=0` (no per-step
  readback), `arm=channel4`.
* Profiler: `HIERARCHOS_VULKAN_PROFILE_KERNELS=1`,
  `HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1`,
  `HIERARCHOS_VULKAN_PROFILE_TOP_SHADERS=4096`. Three independent profiled
  processes per mode; medians reported. Each process logs 29 profiled step
  batches, of which 26 are measured steps and 3 are warmup steps. The component
  attribution tables below average all 29 profiled batches; the instrumented
  step-time median uses the 26 measured steps.
* Modes profiled separately: FP32 (`HIERARCHOS_WMMA_MODE` unset/off) and Mixed
  (`HIERARCHOS_WMMA_MODE=mixed`, still opt-in).
* `HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL` and
  `HIERARCHOS_ATTENTION_BACKWARD_QUERY_SCRATCH_REUSE` were **unset** for every
  process, so the production selector admitted the new route by default.
* One additional unprofiled process per mode gives the short sanity step timing.
  No ABBA campaign, no 160-process run.

> **Profiler output is attribution, not a performance claim.** Stride-1 sampling
> perturbs execution, and the profiled GPU total is not a step time. The
> unprofiled medians in §2 are separate measurements and are never combined with
> the profiled totals.

Files: `raw/profile_{fp32,mixed}_r{0,1,2}.log`, reduced to
`raw/attribution.json` by `reduce_profile.py`; `raw/plain_{fp32,mixed}.log` for
the unprofiled sanity timing; `run_profile.sh` reproduces all of it.

---

## 2. Production baseline

| measurement | FP32 | Mixed |
| --- | ---: | ---: |
| unprofiled step median (26 measured steps, 1 process) | **24.890 ms** | **19.551 ms** |
| unprofiled step min / max | 24.541 / 26.913 ms | 19.060 / 24.070 ms |
| profiler-instrumented step median (`[phase-xii-step]`, 26 measured steps, 3 processes) | 28.519 ms | 23.440 ms |
| profiled GPU total per step (29 profiled batches per process, median of 3) | 26.536 ms | 20.789 ms |
| GPU dispatches per profiled step | 98 | 143 |

The unprofiled medians land next to the Phase XIX arm-B medians for the same
promoted module (24.643 ms FP32, 19.111 ms Mixed), which is the expected
continuity check, not a new before/after claim.

### 2.1 Production dispatch identity and counters

From the run log (identical in all six profiled processes and both plain runs):

```text
two-pass backward: route=query_dqunroll_scratchreuse eligible=true reason=eligible \
  kernels=true scratch=true key_blocks=1 batch=2 heads=8 kv_heads=2 q_seq=64 kv_seq=64 \
  head_dim=128 causal=1 unfused_qk=9 scratch_elements=65536 scratch_bytes=524288
two-pass backward: route=key_channel4 ... kernels=true ...
```

| check | result |
| --- | --- |
| profiler identity of pass 1 | `transformer_attention_backward_query_rdna3_dqunroll_scratchreuse` |
| pass-1 dispatches per step | **2** (one per layer) |
| Phase XVIII `…_rdna3_dqunroll` dispatches per step | **0** |
| original `transformer_attention_backward_query_rdna3` dispatches per step | **0** |
| scalar backward dispatches | 0 |
| pass-2 route | `transformer_attention_backward_key_channel4_rdna3`, 2 per step |
| `shader=unknown` lines in selected steps | 0 in all six processes |
| parameter start / end digest | `d79a12d8…9c4655` in every process |

The scratch-reuse module is therefore the only query path that ran; no dQ-unroll
and no original-query fallback dispatch occurred.

### 2.2 Grid dimensions (analytic, from the dispatch sites)

| dispatch | grid | notes |
| --- | --- | --- |
| query backward (scratch reuse) | `[16, 1, 1]` | `ceil(batch·heads·seq / 64)` = 1024 rows / 64 |
| row-packed attention forward | `[16, 1, 1]` | same row count |
| channel-4 key backward | `[128, 1, 1]` | `ceil(key_rows·32 / 64)`, key_rows = 256 |
| FP32 `linear_forward` | `[64, 8, 1]` | `[ceil(output_dim/16), ceil(rows/16), 1]`, 1024/16 × 128/16 |
| FP32 `linear_input_grad` | `[64, 8, 1]` | same shape contract |
| activation conversion | `[512, 1, 1]` | `ceil(len/256)` for a 128×1024 activation |

The profiler emits per-shader counts and times but not grid extents, so no grid
value above is a profiler reading.

---

## 3. FP32 profile (median of three processes, 29 profiled batches each)

| component | ms/step | share | dispatches/step |
| --- | ---: | ---: | ---: |
| **dense projection** (`linear_forward`, `linear_input_grad`, `…_lane4`, `…_group4_muladd`, `linear_weight_grad`) | **14.012** | **52.8 %** | **42** |
| row-packed attention forward | 3.858 | 14.5 % | 2 |
| RMSNorm (forward + input grad) | 3.254 | 12.3 % | 10 |
| **query backward, scratch reuse** | **3.141** | **11.8 %** | **2** |
| cross-entropy (+ `ordered_f32_sum`) | 1.265 | 4.8 % | 2 |
| channel-4 key backward | 0.505 | 1.9 % | 2 |
| other | 0.524 | 2.0 % | 38 |
| **profiled GPU total** | **26.536** | 100 % | **98** |

Dense-projection breakdown (median ms/step):

| shader | ms/step | dispatches/step |
| --- | ---: | ---: |
| `linear_forward` | 7.206 | 17 |
| `linear_input_grad` | 6.453 | 17 |
| `linear_forward_lane4` | 0.173 | 2 |
| `linear_input_grad_group4_muladd` | 0.122 | 2 |
| `linear_weight_grad` | 0.075 | 4 |

`linear_forward` and `linear_input_grad` are therefore the two largest single
shaders in the default production mode, together 13.66 ms/step (51.5 % of the
profiled total).

---

## 4. Mixed profile (median of three processes, 29 profiled batches each)

Mixed WMMA stays explicitly opt-in; `HIERARCHOS_WMMA_MODE=mixed` was set only
for these processes, and the production default remains off.

| component | ms/step | share | dispatches/step |
| --- | ---: | ---: | ---: |
| **WMMA GEMM (`wmma_gemm_f16_f32acc`)** | **6.392** | **30.7 %** | **30** |
| row-packed attention forward | 4.009 | 19.3 % | 2 |
| RMSNorm (forward + input grad) | 3.321 | 16.0 % | 10 |
| **query backward, scratch reuse** | **3.301** | **15.9 %** | **2** |
| cross-entropy (+ `ordered_f32_sum`) | 1.315 | 6.3 % | 2 |
| **FP32→FP16 activation conversion** (`fp32_to_fp16_native`) | **1.209** | **5.8 %** | **45** |
| channel-4 key backward | 0.510 | 2.5 % | 2 |
| other | 0.465 | 2.2 % | 38 |
| dense projection (residual FP32: `linear_weight_grad` 4, `…_group4_muladd` 2, `linear_forward` 2, `…_lane4` 2, `linear_input_grad` 2) | 0.237 | 1.1 % | 12 |
| FP32→FP16 **weight mirrors** (`fp32_to_fp16_packed`) | **0.000** | **0.0 %** | **0** |
| **profiled GPU total** | **20.789** | 100 % | **143** |

### 4.1 Conversion accounting — measured, not assumed

* **45 activation-conversion dispatches per Mixed step**, 1.209 ms/step. This
  reproduces the pre-promotion count (45 / 1.20 ms) exactly; it was not assumed.
* **0 weight-mirror conversions per step.** `fp32_to_fp16_packed` does not appear
  in any selected batch, so there is no per-step mirror traffic on this fixture.
  This says nothing about load-time mirror construction (the profiler cannot see
  work before the first measured step).
* Activation and mirror conversions are separate shader identities and are never
  summed here.

**Call-site identity.** Every one of the 45 dispatches comes from the single
shader identity `fp32_to_fp16_native`, whose only production call sites in a step
are `LinearWmma`:

| call site | operands cast | GEMMs produced |
| --- | --- | --- |
| `wmma_linear.rs::LinearWmma::record_forward` | projection input activation | 1 |
| `LinearWmma::record_weight_grad` | `grad_output`, then the input activation **again** | 1 |
| `LinearWmma::record_input_grad` | `grad_output` **again** | 1 |

Projection association: the model's dense projections (`q`/`k`/`v`/`o`,
`gate`/`up`/`down`) plus the LM head, i.e. the shapes reached through
`TransformerKernels::wmma_linear(rows, input_dim, output_dim)`.

**What the profiler cannot resolve.** It reports shader identity, not call site,
so the 45 dispatches cannot be split per projection from these logs. What can be
stated from arithmetic is a bound: 30 WMMA GEMMs consumed at most 30 first-use
casts, so **at least 15 of the 45 conversions re-cast an operand that is already
in f16 form** (or are a second cast of the same tensor) within the same step.
Resolving the exact split needs a measurement-only call-site counter, which this
phase was not authorized to add.

---

## 5. Historical comparison (context, not a measured before/after)

Phase XIX, profiler-instrumented, same device and fixture, 9 selected steps:

| component | FP32 then | Mixed then | FP32 now | Mixed now |
| --- | ---: | ---: | ---: | ---: |
| dense projections | 15.894 ms | 0.303 ms | 14.012 ms | 0.237 ms |
| query backward | 6.465 ms (dQ-unroll) | 7.250 ms (dQ-unroll) | 3.141 ms (scratch reuse) | 3.301 ms (scratch reuse) |
| row-packed forward | 4.801 ms | 5.263 ms | 3.858 ms | 4.009 ms |
| WMMA GEMM | — | 7.686 ms | — | 6.392 ms |
| activation conversion | — | 1.201 ms / 45 | — | 1.209 ms / 45 |
| RMSNorm | 3.637 ms | 3.921 ms | 3.254 ms | 3.321 ms |
| cross-entropy | 1.541 ms | 1.684 ms | 1.265 ms | 1.315 ms |
| channel-4 key | 0.554 ms | 0.607 ms | 0.505 ms | 0.510 ms |
| profiled GPU total | 33.399 ms | 28.480 ms | 26.536 ms | 20.789 ms |

**Caveats, stated explicitly.**

1. The query row is a **different shader** in the two phases. The old timing
   belongs to the Phase XVIII dQ-unroll module and must not be subtracted from
   the new one as if it were the same code.
2. The driver differs: Phase XIX used its own 9-step complete-step driver;
   Phase XXI uses the Phase XII complete-step driver with 3 warmup steps and 26
   measured steps, run three times. Profiled totals are sensitive to warmup,
   clocks and selection, so **no column-to-column delta is claimed as an
   improvement**. Only the query identity change is a genuine module change.
3. The only defensible cross-phase comparison is the unprofiled step median:
   24.890 ms FP32 and 19.551 ms Mixed here, against the Phase XIX arm-B medians
   24.643 / 19.111 ms for the same promoted module — agreement within 1.0 % and
   2.3 %, which is consistency, not a new measurement of the promotion.
4. Profiler totals are attribution; they are 6.6 % (FP32) and 6.3 % (Mixed) above
   the unprofiled medians here because stride-1 sampling perturbs execution.

---

## 6. Candidate evaluation

No source, shader, test, dispatch, precision policy or production behaviour was
modified to produce this section.

### Candidate A — shared FP16 activation conversion

* Measured: 1.209 ms/step, 45 dispatches, 5.8 % of the Mixed profiled total,
  0 weight-mirror conversions.
* Duplicate work is real but bounded: 45 casts for 30 GEMMs means ≥15 casts are
  redundant by construction.
* Sharing opportunities visible in the code: `q`/`k`/`v` consume the same
  post-norm hidden state; `gate`/`up` consume the same post-norm MLP input;
  `grad_output` is cast separately for weight-grad and input-grad on the same
  projection. Executors are cached by `(rows, input_dim, output_dim)`, so
  same-shape projections already share one `input_f16`/`grad_output_f16` buffer
  and overwrite it per call — the casts are still re-issued.
* Safety requirement: a graph-level cache cannot key on allocation address.
  Activations are written by producer dispatches and pooled buffers are reused
  for different logical tensors, so a cache needs a per-buffer write generation
  (incremented by every dispatch that writes it) plus the tensor length, and a
  proof that no writer ran between the producer and the cast.
* Ceiling: the entire conversion cost is 5.8 % of the profiled Mixed step and
  ~6.2 % of the unprofiled Mixed step. Even a perfect cache saves at most
  ~1.2 ms; eliminating only the provably redundant casts is worth ~0.3–0.6 ms
  (1.5–3 % of a Mixed step). Synchronization, extra memory and graph-recording
  complexity would eat into that, and FP32 — the default mode — gains nothing.

### Candidate B — FP32 dense projection path

* Measured: 14.012 ms/step, 52.8 % of the FP32 profiled total, 42 dispatches.
  It remains the largest credible target after scratch reuse.
* Split: `linear_forward` 7.206 ms / 17, `linear_input_grad` 6.453 ms / 17,
  weight-grad and the vectorized variants 0.37 ms / 8.
* Dominant shapes: rows = 128 (`batch 2 × seq 64`), 1024→1024, 1024→256,
  1024→1024 per layer plus the LM head. Implied throughput is roughly
  500–600 GFLOP/s against a multi-TFLOP/s FP32 peak on this iGPU, so the kernels
  are not at an architectural limit.
* Why it is attractive: 17 dispatches each means this is compute-bound work, not
  launch overhead, and the same 64×8 grid is used for every projection
  regardless of shape.

### Candidate C — Mixed WMMA GEMM

* Measured: 6.392 ms/step, 30.7 % of the Mixed profiled total, 30 dispatches
  (0.213 ms each). Still the dominant Mixed component, but Mixed is opt-in and
  the FP32 default pays none of it.
* A 10 % shader improvement is worth ~0.64 ms (≈3.3 % of a Mixed step), smaller
  in absolute terms than Candidate B and confined to the non-default mode.
* Prior staging/tile work already exists (Phases LXXXVI–LXXXVIII, C1), which
  raises the chance that the easy wins are gone; FP16 accumulation also makes
  numerical risk non-trivial.

### Candidate D — row-packed attention forward

* Measured: 3.858 ms FP32 (14.5 %), 4.009 ms Mixed (19.3 %), 2 dispatches.
* Its share did rise in relative terms because everything else shrank, but the
  absolute cost is a quarter of Candidate B's. The Phase VIII streaming candidate
  was rejected on numerical grounds and must not be resurrected.

### Ranking

| rank | candidate | absolute headroom | expected recoverable | implementation risk | evidence confidence |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | **B — FP32 dense projections** | 14.01 ms FP32 | 1.0–1.5 ms (≈4–6 % of step) at a 7–10 % shader gain | medium: new tiling must keep each output's reduction order to preserve bitwise parity | high — direct profiler attribution, 3 processes |
| 2 | C — Mixed WMMA GEMM | 6.39 ms Mixed | 0.4–0.7 ms (≈2–3.5 %) | medium-high: prior art exhausted, FP16 accumulation | high for cost, medium for achievable gain |
| 3 | D — row-packed forward | 3.86–4.01 ms | 0.2–0.4 ms (≈1–2 %) | high: Phase VIII streaming already rejected | medium |
| 4 | A — shared activation conversion | 1.21 ms Mixed only | 0.3–0.6 ms (≈1.5–3 %), 0 in FP32 | high: generation/lifetime tracking, graph ordering | medium — duplicate count known, ownership split not |

---

## 7. Recommended follow-up experiment (one)

**FP32 dense input-gradient tiling experiment** — `linear_input_grad` at the
128×1024×1024 / 1024→256 geometry, with `linear_forward` as the second target if
the first generalizes.

* What: a shape-specialized workgroup/tile schedule for the existing FP32
  input-gradient kernel, selected behind a test seam, leaving each output
  element's reduction order and precision untouched. Vectorized operand loads and
  more outputs per workgroup are the levers; the 64×8 grid currently used for
  every shape suggests neither is saturated.
* Why this one: it is the second-largest single shader in the default production
  mode (6.453 ms, 17 dispatches), it has a simpler contract than the forward
  kernel (no bias epilogue, no fused variants to keep in step), and the implied
  throughput is far below the device's FP32 peak. Candidate C is bigger only in
  the opt-in mode; Candidate A's ceiling is a fraction of this one.
* Success criterion: an ABBA complete-step comparison on the established fixture
  with a 95 % confidence interval excluding 1.0, plus bitwise-identical outputs
  against the current kernel on the parity corpus.
* Falsifier: if the tiled kernel is within noise at stride-1 attribution and in
  the ABBA step, stop and re-rank toward Candidate C rather than widening the
  experiment.

---

## 8. What this phase did not do

* No optimization implemented, no shader or SPIR-V changed, no dispatch or
  selector changed, no precision policy or tolerance changed.
* Mixed WMMA default unchanged (off); it was enabled only per process for the
  Mixed profile.
* Three independent profiled processes per mode (29 profiled step batches
  each, 3 warmup + 26 measured) and one unprofiled process per mode were run.
  No five-round ABBA campaign and no 160-process benchmark were run.
* No file in the committed Phase XIX or Phase XX receipts was modified or
  duplicated. The sealed Phase XIX digest recomputed inside this phase is
  `078ef22eee85c82f34cbc0de62ee594d458464353fc7a2fb1cc84723af571392`
  (195 files / 2,091,594 bytes).
* This receipt is committed with the v0.2.8 release evidence; the raw logs are
  unchanged.

---

## 9. Reproduction

```text
cd hierarchos-vulkan
cargo test --release --lib --no-run
BIN=$(ls -t target/release/deps/hierarchos_vulkan-*.exe | head -1)

PHASE_XXI_TEST_BIN="$BIN" bash validation/phase_xxi_postpromotion_profile_20261128/identity.sh
PHASE_XXI_TEST_BIN="$BIN" bash validation/phase_xxi_postpromotion_profile_20261128/run_profile.sh
```

`run_profile.sh` unsets both query-route environment variables, runs three
profiled processes per mode plus one unprofiled sanity process, and reduces the
logs with `reduce_profile.py` into `raw/attribution.json`.

---

## 10. Artefacts

| file | contents |
| --- | --- |
| `raw/00_identity.txt` | HEAD `468a7ce…`, branch, refs, stash, status at profile time |
| `raw/01_device.txt` | `vulkaninfo --summary`: device, API version, driver version/info |
| `raw/02_environment.txt` | rustc/cargo identity, test-binary hash, shader hashes, Phase XIX digest, environment |
| `raw/profile_fp32_r{0,1,2}.log` | three profiled FP32 processes, 29 steps each |
| `raw/profile_mixed_r{0,1,2}.log` | three profiled Mixed processes, 29 steps each |
| `raw/plain_fp32.log`, `raw/plain_mixed.log` | unprofiled sanity timing, 26 measured steps |
| `raw/attribution.json` | per-process and median component/shader attribution |
| `reduce_profile.py`, `run_profile.sh`, `identity.sh` | reducer and drivers |

### Identity

| item | value |
| --- | --- |
| starting Phase XX commit | `468a7ceada598dd613162323604f4240f1b411c6` |
| branch | `research/v0.2.8-optimization-discovery` |
| device | AMD Radeon Graphics (integrated, `0x15bf`), driverInfo `26.20.15.14 (LLPC)`, driverVersion 2.0.416, Vulkan `1.4.359` |
| compiler | `rustc 1.93.0 (254b59607 2026-01-19)`, LLVM 21.1.8 |
| build | release `cargo test --release --lib --no-run`, binary sha256 `8c356f4f18c7e0ee029d8e43a94391fc551a5918ec4dd4e599ca563c3ae59e82` |
| promoted query SPIR-V | `a6b3d6c6d7bd2911e61f81181d79cb41a3c4be2762cd9a0b6ee636e57d6057ee` |
| environment | both query variables unset; `HIERARCHOS_WMMA_MODE` unset for FP32, `mixed` for Mixed only |

---

## 11. Amendment — 2026-11-28

No raw log was edited and no measurement was re-run. This amendment corrects the
reporting window and the receipt status; it does not change any conclusion.

1. The attribution reducer selects every profiled batch that contains the
   promoted pass-1 dispatch. Each process has 29 such batches, including the 3
   warmup steps, and `reduce_profile.py` divides their totals by 29. The figures
   in §3 and §4 are therefore **29-batch profiled attribution**, not a
   "26 measured step" statistic. The earlier §1 wording that implied otherwise
   has been corrected; the raw logs and `raw/attribution.json` are untouched.
2. For transparency, the same reducer restricted to the 26 measured batches
   gives these per-process means (median of the three processes):

   | FP32 component | 29-batch (§3) | 26-measured |
   | --- | ---: | ---: |
   | dense projection | 14.012 ms | 13.034 ms |
   | `linear_forward` | 7.206 ms | 6.751 ms |
   | `linear_input_grad` | 6.453 ms | 5.948 ms |
   | query backward, scratch reuse | 3.141 ms | 2.894 ms |
   | profiled GPU total | 26.536 ms | 24.471 ms |

   | Mixed component | 29-batch (§4) | 26-measured |
   | --- | ---: | ---: |
   | WMMA GEMM | 6.392 ms | 5.887 ms |
   | activation conversions | 1.209 ms / 45 | 1.192 ms / 45 |
   | query backward, scratch reuse | 3.301 ms | 2.967 ms |
   | profiled GPU total | 20.789 ms | 18.870 ms |

   The warmup batches inflate both windows' totals; the measured-step window is
   the one that should be compared with an unprofiled step. Neither window is a
   performance claim, and no column-to-column delta is introduced here.
3. The profile is three independent processes per mode, not one. Each process
   has its own log, its own step timings and its own route counters; the logs
   are distinct files with distinct SHA-256 values. Three observations are a
   median, not a confidence interval, and this receipt makes no inferential
   claim beyond the observed spread.
4. The recommendation in §7 stands on the measured-step attribution: FP32 dense
   projections remain the largest component and `linear_input_grad` the largest
   single kernel inside it. The Phase XXII experiment uses the 128 x 1024 x 1024
   production operation, which the profile identifies through the shader
   identity and the 64 x 8 dispatch grid.
