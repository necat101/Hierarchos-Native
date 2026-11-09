# Opt-in RDNA3 WMMA (cooperative matrix) GEMM

Status: **opt-in, experimental, default off.** The FP32 Vulkan path is unchanged
and no existing shader, tolerance, golden reference or checkpoint format was
modified. No published release (`v0.1.0-rc2` or any other tag) is touched.

This document records what was measured on the physical RDNA3 device, what is
implemented, what is **not** wired yet, and where the mode is unqualified.

---

## 1. Measured hardware contract

Probe: `hierarchos-vulkan-wmma-probe --wmma-mode mixed` on the Ryzen Z1 Extreme
(ASUS ROG Ally, `AMD Radeon Graphics`, PCI `1002:15bf`).

```
device: AMD Radeon Graphics
requested mode: mixed
cooperative matrix: extension_exposed=true feature_supported=true memory_model=true subgroup_size_control=true enabled=true
  storage_16bit=true shader_float16=true driver_subgroup_size=64 configurations=11
  mixed (f16xf16+fp32) tile: 16x16x16 A=f16 B=f16 C=f32 scope=subgroup
  full-fp16 (f16xf16+f16) tile: 16x16x16 A=f16 B=f16 C=f16 scope=subgroup
supported: mixed=true full-fp16=true
```

| Property | Measured |
| --- | --- |
| `VK_KHR_cooperative_matrix` | exposed (device extension, revision 2) |
| `VkPhysicalDeviceCooperativeMatrixFeaturesKHR::cooperativeMatrix` | true, **enabled** on the logical device |
| `vulkanMemoryModel` (required by the SPIR-V) | true, enabled |
| `VK_EXT_subgroup_size_control` | exposed; compute accepts 32 |
| Driver default subgroup size | **64** |
| Advertised cooperative configurations | 11 |
| Selected mixed tile | `16x16x16` f16 × f16 → **f32**, scope `SUBGROUP`, non-saturating |
| Selected full-FP16 tile | `16x16x16` f16 × f16 → **f16**, scope `SUBGROUP`, non-saturating |
| `storageBuffer16BitAccess`, `shaderFloat16` | true, enabled |

Two consequences drove the implementation:

1. **The default subgroup is 64, but 16x16x16 cooperative operations are wave32
   on RDNA3.** The WMMA pipelines therefore carry a per-pipeline required
   subgroup size of 32 (`VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`).
   A device-wide override was deliberately *not* used: it would have changed the
   execution geometry of every unrelated kernel.
2. **The SPIR-V needs the Vulkan memory model.** `glslc` emits
   `OpCapability VulkanMemoryModel` plus `MakePointerVisible`/`NonPrivatePointer`
   on the cooperative loads, so the feature is queried and enabled explicitly.

### Verified shader artifacts

`glslc -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O`:

| Artifact | Cooperative operations |
| --- | --- |
| `wmma_gemm_f16_f32acc.spv` | 4 × `OpCooperativeMatrixMulAddKHR` (+ loads/stores) |
| `wmma_gemm_f16_f16acc.spv` | 4 × `OpCooperativeMatrixMulAddKHR` (+ loads/stores) |
| `gemm_f16_f32_plain.spv` | **0** (control/fallback: FP16 storage, no matrix cores) |
| `fp32_to_fp16_native.comp` | 0 (IEEE RNE conversion into native `float16_t` storage) |

`OpCapability CooperativeMatrixKHR` and `OpExtension "SPV_KHR_cooperative_matrix"`
are present in both WMMA modules. That establishes that cooperative-matrix
*instructions* are in the module. Actual RDNA3 WMMA SASS/ISA generation is **not**
directly observable with the tooling available here; the evidence for hardware
execution is that the device advertises and enables the KHR feature, the pipeline
is created with wave32, and results are numerically correct (section 4). No claim
of specific ISA mnemonics is made.

### Fixed defect in the earlier draft shader

The pre-existing draft (`wmma_gemm_*.comp`, untracked) staged the B operand as
`[n][k]` and then loaded it as a row-major K×N cooperative fragment. For square
tiles that silently computes `A * Bᵀ`. The committed shaders stage **A as
`[m][k]` and B as `[k][n]`** and both load row-major; that is what makes
`trans_a`/`trans_b` mean what they say, and the transpose coverage in section 4
is what locks it down.

---

## 2. Modes

`--wmma-mode off|mixed|full-fp16` is orthogonal to `--amp`/`--precision`.
It is also read from `HIERARCHOS_WMMA_MODE` by the Vulkan device constructor.

| Mode | Operands | Accumulation | Parameters | Optimizer | Norm/softmax/loss |
| --- | --- | --- | --- | --- | --- |
| `off` (default) | FP32 | FP32 | FP32 | FP32 | FP32 |
| `mixed` | FP16 | **FP32** | FP32 master | FP32 | FP32 |
| `full-fp16` | FP16 | **FP16** (where advertised) | FP32 master | FP32 | FP32 |

An explicit request on a device that cannot satisfy it **fails** with the probed
reason, e.g.:

```
--wmma-mode mixed is not supported by this Vulkan device: device does not expose VK_KHR_cooperative_matrix
cooperative matrix: extension_exposed=false ...
```

A device that is unsupported never silently runs a fallback while claiming WMMA:
`VulkanWmmaCapabilities::enabled` is only true when every feature was actually
switched on, and the plain FP16 kernel is labelled `plain-f16-storage-f32acc`
and is never counted as WMMA.

---

## 3. Where the acceleration is actually used

Currently wired:

* `LinearProjectionTrainer` (the shared dense-linear primitive used by the
  native projection / multimodal projector training path): dense-linear forward
  GEMM, input-gradient GEMM and weight-gradient GEMM, with one FP16 execution
  mirror refreshed from the FP32 master after each AdamW step.
* `WmmaGemm` is public API for further wiring (`record_conversion`,
  `record_gemm`, `select_arm`, `dispatch_counts`).

**Not wired yet:** the per-architecture transformer graphs
(`transformer.rs`, `deepseek_v41`, MoE expert projections, LoRA A/B
projections, Qwen/Qwen3-Next graphs) still dispatch their own hand-tuned
kernels. Those graphs are numerically frozen against the `<2e-7` gate and each
would need its own re-qualification, so routing them is deliberately out of this
change. Consequently:

* `transformer-train --wmma-mode mixed` selects the policy and enables the
  device feature, but the transformer graph does not consume it yet; no
  transformer training speedup is claimed.
* Streamed/resident full-model training, full fine-tuning, LoRA/PEFT and
  checkpoint-resume integration with WMMA are **not** implemented.

This is a deliberate scope boundary, not a passing result.

---

## 4. Numerical qualification

Run: `hierarchos-vulkan-wmma-qualify --wmma-mode mixed`
(fixtures `64x48x40`, `33x17x19`, `128x128x96`, `16x1024x64`, `256x64x16`,
`37x8192x48`, plus all four transpose layouts on the first three).

Comparison groups are kept strictly separate:

1. FP32 reference (`reference_fp32_gemm`, f64 accumulation). The original FP32
   `2e-7` gate is **unchanged** and is *not* applied to reduced precision.
2. WMMA mixed vs FP32 — quantization drift.
3. WMMA mixed vs the matched FP16-operand/FP32-accumulate oracle — kernel
   correctness.
4. WMMA full-FP16 vs FP32.
5. WMMA full-FP16 vs its matched FP16 oracle.
6. Plain FP16 storage (no matrix cores) vs the matched oracle — the control.

### Mixed (FP16 × FP16 → FP32): **37/37 gates pass**

| Gate | Result | Limit |
| --- | --- | --- |
| vs matched oracle, max_abs | **2.1e-5** (worst of all layouts) | 1e-3 |
| vs FP32, max_abs | **9.89e-3** | 2e-2 |
| vs FP32, RMS | **2.09e-3** | 3e-3 |
| Non-finite outputs | 0 | 0 |
| Top-1 agreement | **100%** (all fixtures) | — |
| Mean top-5 overlap | **1.0000** | — |
| Softmax max abs divergence | 1.78e-3 | — |

The decisive point: *WMMA-mixed vs matched oracle* (2.1e-5) is four orders of
magnitude tighter than *matched oracle vs FP32* (9.9e-3). The drift is therefore
entirely expected FP16 operand quantization, not a kernel defect.

### Full-FP16 (FP16 × FP16 → FP16): **9/37 gates fail — partially unqualified**

| Fixture | vs FP32 max_abs | vs matched oracle | Verdict |
| --- | --- | --- | --- |
| `64x48x40` (k≤40) | 1.7e-2 – 2.6e-2 (limit 5e-2) | within derived bound | **passes** |
| `33x17x19` | 7.6e-3 – 1.2e-2 | within derived bound | **passes** |
| `256x64x16` | 8.2e-3 | 7.8e-3 | **passes** |
| `16x1024x64` | 3.9e-2 | exceeds derived bound | **unqualified** |
| `37x8192x48` | 3.7e-2 | exceeds derived bound | **unqualified** |
| `128x128x96` (k=96) | **6.3e-2 – 7.3e-2** | exceeds derived bound | **unqualified** |

FP16 accumulation has no fixed absolute error bound: the error scales with the
accumulator magnitude, so the matched-oracle limit is derived per fixture as
*two binary16 ULPs at the largest reference magnitude* (floored at the mixed
constant). The residual failures are a few ULPs at magnitude 32–128 and grow with
K, which is the expected cost of an FP16 accumulator — but they **exceed the
initial `5e-2` FP32-drift target at K ≥ 64 wide-magnitude fixtures**, so those
configurations are reported as unqualified rather than excused. No tolerance was
relaxed to reach this table.

### Oracle semantics finding

A naive "round to FP16 after every multiply-add" oracle disagreed with the
hardware by up to **1.1e-1**, in exact multiples of 1.5625e-2 (one FP16 ULP at
accumulator magnitude ~16). Replacing it with a **one-rounding-per-16-wide-K-tile**
model (intra-tile dot in FP32, accumulator rounded to FP16 per `k0` step) reduced
the disagreement by roughly 10× and is what the table above uses. The oracle
model is documented in `OracleAccumulation::Fp16TileK`; this was a correction of
the reference to the observed semantics, not a loosening of a gate.

---

## 5. Performance on the physical RDNA3 device

`hierarchos-vulkan-wmma-bench --wmma-mode mixed --rounds 4 --iterations 20`,
release build, alternating ABBA order per round, 3 warm-up iterations per arm,
layout `A[rows,k] × Bᵀ[columns,k]` (the existing kernel's layout). Each timed
iteration is one submit + fence for exactly one GEMM; conversion and allocation
are excluded so the number is about the kernel. Speeds are **GEMM call latency**,
not end-to-end training-step latency.

| Shape | FP32 baseline | Plain FP16 (no WMMA) | WMMA mixed | WMMA speedup |
| --- | ---: | ---: | ---: | ---: |
| 512×768×768 (QKV) | 1.238 ms | 4.047 ms (0.334×) | **0.573 ms** | **2.170×** |
| 512×2048×768 (FFN up) | 3.595 ms | 10.617 ms (0.341×) | **1.704 ms** | **2.120×** |
| 2048×768×2048 (FFN down) | 15.176 ms | 33.027 ms (0.454×) | **6.447 ms** | **2.309×** |
| 128×8192×768 (vocab) | 3.678 ms | 10.727 ms (0.343×) | **1.987 ms** | **1.832×** |
| 64×64×64 (LoRA control) | 0.096 ms | 0.101 ms (0.975×) | 0.127 ms | **0.738×** |
| 16×128×32 (tiny control) | 0.086 ms | 0.093 ms (0.933×) | 0.111 ms | **0.758×** |

Observations:

* The gain is from **matrix cores, not from FP16**: plain FP16 storage without
  cooperative matrices is *slower* than the tuned FP32 kernel at every size.
* **Crossover is real.** Below roughly 64×64×64 (and for a single 32×32 output
  tile), staging plus wave32 launch overhead dominates and WMMA loses 25–27%.
  `wmma_shape_eligible` therefore refuses shapes under 16 rows/columns/K, under
  1024 outputs, or that fill fewer than four 32×32 tiles; those keep the caller's
  kernel and increment the shape-fallback counter.
* Drift measured in the same runs: mixed 7.0e-5 – 1.2e-4 absolute on these
  (small-magnitude) operands, vs 3.3e-7 – 6.9e-7 for FP32 self-consistency.
* Per-run dispatch accounting: `cooperative=504 plain_f16=504 shape_fallbacks=0`
  (cooperative dispatches are counted separately from the plain control, so a
  fallback can never be mistaken for WMMA execution).

No end-to-end training-step speedup is claimed: the transformer graph is not
wired (section 3), and kernel-level ratios must not be extrapolated.

---

## 6. Regression and default-path safety

* `--wmma-mode off` (the default) requests no cooperative-matrix extension and
  creates no WMMA pipeline; `WmmaMode::Off` is what an unset
  `HIERARCHOS_WMMA_MODE` produces.
* The WMMA shaders are new artifacts in `hierarchos-vulkan/shaders/`; no existing
  `.spv` was rebuilt or replaced, and no existing tolerance, golden file or
  frozen CPU expectation was edited.
* Multi-device: a Vulkan *device group* reports `VulkanWmmaCapabilities::default()`
  (feature off) because the feature is per physical device; an explicit WMMA
  request against a group fails at the per-device boundary instead of assuming
  every member has matrix cores.
* Existing `--amp`/`--no-amp` semantics are untouched.

Tests added (`cargo test --lib wmma`): mode parsing, IEEE RNE half conversion
(including ties-to-even, subnormal and 65520 → ∞), shape admission, drift and
agreement metrics, mixed vs matched oracle across all four layouts and irregular
shapes, full-FP16 vs matched oracle, plain-FP16 control vs oracle, and the
negative control that an unsupported explicit request produces a diagnostic.

---

## 7. Reproduce

```powershell
cargo build --release --manifest-path hierarchos-vulkan/Cargo.toml `
  --bin wmma-probe --bin wmma-qualify --bin wmma-bench

# Capability receipt (no feature enabled; purely informational)
.\hierarchos-vulkan\target\release\wmma_probe.exe --wmma-mode off --json

# Numerical gates: exit code is non-zero when any gate fails
.\hierarchos-vulkan\target\release\wmma_qualify.exe --wmma-mode mixed
.\hierarchos-vulkan\target\release\wmma_qualify.exe --wmma-mode full-fp16 --json

# Paired ABBA benchmark
.\hierarchos-vulkan\target\release\wmma_bench.exe --wmma-mode mixed --rounds 4 --iterations 20
```

---

## 8. Unsupported / unqualified configurations

| Configuration | Status |
| --- | --- |
| Any device without `VK_KHR_cooperative_matrix` | unsupported; diagnostic, FP32 unchanged |
| Device without a SUBGROUP-scope 16x16x16 f16/f16→f32 tile | `mixed` unsupported |
| Device without a SUBGROUP-scope 16x16x16 f16/f16→f16 tile | `full-fp16` unsupported |
| Device where `VK_EXT_subgroup_size_control` cannot force 32 for compute | unsupported (16x16x16 needs wave32 here) |
| Vulkan device group / multi-GPU | not negotiated; request fails |
| `full-fp16` at K ≥ 64 wide-magnitude fixtures | **unqualified** (exceeds 5e-2 drift target) |
| Transformer graph training/fine-tuning/LoRA/streamed paths | not wired; no WMMA claim |
| Actual RDNA3 SASS WMMA mnemonics | not observable with available tooling |

## Transformer graph integration (2026-11-09)

### Wired execution routes

The resident dense projection is the single choke point of the Transformer
graph, so the mixed arm is wired there and every attention/MLP projection that
uses it inherits the dispatch:

| Route | Implementation | Phases |
|---|---|---|
| Dense linear forward | `record_linear_projection` tries `LinearWmma::record_forward` before the qualified kernel | forward |
| Weight gradient | `VulkanLinear::record_backward_impl` tries `record_weight_grad` before `linear_weight_grad` | weight-grad |
| Input gradient | same function tries `record_input_grad` in the generic branch | input-grad |

Bias-bearing projections (GPT-2, Qwen2 QKV) are covered: the GEMM runs
unbiased and `bias_add` applies the bias afterwards, which is the same
two-dispatch order the tied LM head already used.

Executors are keyed by `(rows, input_dim, output_dim)` and shared inside
`TransformerKernels`, so layers of identical geometry reuse one set of FP16
mirrors and an `off` run allocates nothing. Mirrors are refreshed from the FP32
masters on every dispatch, so a checkpoint, AdamW update or weight swap can
never leave a stale FP16 copy feeding the graph.

### Refusals (explicit, not silent)

* `full-fp16` is rejected for every Transformer workflow. The arm accumulates in
  FP16 and the graph has no in-graph widening store, so a low-precision result
  could not reach the FP32 masters. The CLI refuses the flag before any I/O and
  the graph refuses the mode again during construction. The GUI lists the option
  disabled with the reason.
* A device without a qualifying cooperative-matrix tile fails construction with
  the capability diagnostic instead of falling back silently.
* Rows that cannot fill a cooperative tile (single-row cached decode, tiny
  batches) keep their tuned FP32 kernel and increment the shape-fallback
  counter; a requested-but-not-dispatched run is therefore visible in the
  receipt.

### Verification performed

| Check | Result |
|---|---|
| `cargo test --lib wmma` (primitive, unchanged) | 8 passed |
| `cargo test --lib wmma_linear` (graph entry points, real device) | 3 passed |
| `cargo test --lib transformer_wmma` (CLI contract) | 2 passed |
| `check-workspace.ps1` (the CI gate) | exit 0 |
| `cargo check` lib/bins, CLI, GUI | exit 0 |

`wmma_linear`'s three-phase test records forward, weight-gradient and
input-gradient through the same functions the graph calls and compares each
against a host FP32 reference under the mixed gates (2e-2 max, 3e-3 RMS). It is
the regression test for the transposed operand layouts, because a wrongly
transposed dispatch still produces finite, plausible numbers.

### Not qualified (do not read as support)

* Whole-model logit drift, loss trajectory, multi-step convergence, checkpoint
  resume, LoRA adapter training and end-to-end step time are **not** measured.
  The only cached checkpoint on this machine (`hf-internal-testing/tiny-random-gpt2`)
  has no weights and widths far below the cooperative tile contract, and the
  local Llama fixture is `hidden_size = 16`; both fall back on every projection.
* Attention SDPA, MoE expert projections, DeepSeek V4.1 / Qwen hybrid graphs,
  LoRA A/B projections and the streamed/tiered parameter paths are **not** wired.
* No claim of a training-step speedup is made from the GEMM microbenchmarks in
  the section above.

## Eligible-geometry training evidence (2026-11-09)

The frozen GPT-2 matrix and the local Llama package are `hidden_size = 16/32`,
far below the cooperative tile contract, so nothing in them dispatches. A
deterministic fixture with real projection widths was generated instead:

```
python validation/make_wmma_fixture.py --output target/wmma-fixture \
  --hidden 1024 --layers 2 --vocab 1024 --seq-len 64 --batch-size 8 --sequences 32
```

16,782,336 parameters (67.1 MB), 14 dense projections, 1024 columns each, 512
rows per step. Weights and dataset are scratch artifacts, not committed.

### Logits: FP32 vs mixed (same weights, same fixture)

| Measure | Value |
| --- | --- |
| Logit values compared | 524,288 (512 rows x 1024 vocab) |
| Max absolute drift | `2.56264e-3` (gate `2e-2`) |
| RMS drift | `5.02520e-4` (gate `3e-3`) |
| Softmax max absolute divergence | `1.97416e-8` |
| Top-1 agreement | 100% |
| Top-5 overlap | 99.92% |
| Finite / logit max magnitude | yes / `3.018` |

### Real optimizer steps

| Run | Steps | Loss first -> final | Cooperative dispatches | Fallbacks |
| --- | --- | --- | --- | --- |
| Full FT, `off` | 20 | 7.1135 -> 5.6597 | 0 | 0 |
| Full FT, `mixed` | 20 | 7.1135 -> 5.6593 | **180** (fwd 60 / wgrad 60 / igrad 60) | 0 |
| LoRA r=8, `off`, lr 1e-2 | 60 | 7.1135 -> 7.05874 | 0 | 0 |
| LoRA r=8, `mixed`, lr 1e-2 | 60 | 7.1135 -> 7.05889 | **120** (fwd 60 / igrad 60 / wgrad 0) | 0 |

The LoRA weight-gradient count is zero by construction: the base projection is
frozen, so only forward and input-gradient run through it, and the adapter
matrices (`r = 8` columns) are below the tile contract and keep their kernels.
That is the correct accounting, not a silent fallback: fallbacks are counted
separately and none occurred.

### End-to-end step time (ABBA, release, 512 rows/step)

Same checkpoint, seed, optimizer and data; 4-step and 48-step runs per arm, so
the fixed load/export cost cancels. Three alternating rounds:

| Arm | 4-step | 48-step | Per-step (derived) |
| --- | ---: | ---: | ---: |
| `off` | 3.27 / 3.41 / 3.41 s | 32.10 / 32.62 / 32.59 s | ~0.62 s |
| `mixed` | 3.10 / 3.11 / 3.21 s | 29.43 / 29.98 / 36.86 s | ~0.56 s |

Median-of-rounds ratio: **1.087x** whole-optimizer-step. The third mixed round
(36.86 s) is an outlier under a full disk and is not excluded from the table;
even so the median ratio stays under 1.1x. This is one two-layer model on an
integrated GPU, not a general training-speedup claim, and it is far from the
1.8-2.3x GEMM-only numbers above.

### Resume and disk

Continuing from the exported mixed package (`--model <package>`) loads the
trained weights and starts at loss `4.08` on the first batch, i.e. below the
`5.66` reached before export. Weight-level resume works; the transformer trainer
has no exact optimizer-state resume flag, so scheduler/scaler state is not
restored and no exact-resume claim is made.

That verification run then failed with `os error 112` ("not enough space on the
disk"): the machine's system volume is at 100% with ~1.4 GB free after deleting
the 1.4 GB of run packages. The failure is environmental, not numerical, and it
is recorded here because it also blocks the receipt regeneration below.

## CI status: qualification receipt NOT regenerated

`check_rdna3_gpt2_receipts.py` requires
`release_parity/multimodal_training/20261108-rdna3-phase15-qualification/current-qualification.json`
to bind every source byte of the checkout, including
`hierarchos-vulkan/src/transformer.rs`. This change edits that file, so the
step `Current-source RDNA3 GPT2 qualification` fails with
`ValueError: current qualification does not bind hierarchos-vulkan/src/transformer.rs`.

Regenerating it needs the full Phase XV chain (four debug probes, the frozen
shape matrix, six retained multimodal gates, 30 V4.1 native tests, the bridge,
shader provenance, SPIR-V validation and the 1,341-file isolation check) run
from a clean worktree. That chain was **not** completed here: with the system
volume at 100% the debug `libtest` binary alone (308 MB) plus the probe builds
and producer outputs could not be staged reliably. The historical receipt is
untouched, no hash was substituted and no gate was relaxed; the check therefore
fails closed, which is the intended behaviour for an unqualified revision.

`check-workspace.ps1` (the `One-pass workspace check` step) passes: exit 0.
