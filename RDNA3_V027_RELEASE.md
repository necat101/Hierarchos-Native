# RDNA3 attention optimizations — v0.2.7 release note

v0.2.7 promotes four attention-path changes into the default training route and
requalifies the opt-in Mixed WMMA arm without promoting it. Nothing else in the
precision policy moved: global WMMA is still **off by default**, `full-fp16`
still runs the C1 batched-staging module in front of the P3 contract, and every
rejected experiment from Phases VIII–XIII stays out of the production path.

Every number below is a paired measurement against the route it replaced, on the
same host, in one driver session. **The ratios are independent results and are
never multiplied into an overall training speedup.**

## 1. What changed

| # | Change | Production artifacts | Eligibility | Receipt |
| ---: | --- | --- | --- | --- |
| 1 | Two-pass MHA attention backward | `transformer_attention_backward_query_rdna3.{comp,spv}`, `transformer_attention_backward_key_rdna3.{comp,spv}`, `src/transformer.rs` | existing gate, `unfused_qk` 0/9 | `validation/mha_attention_backward_closeout_20261124/` |
| 2 | GQA grouped backward key pass | same key pair | `8q/2kv`, `4q/2kv`, `8q/4kv`, `16q/4kv`, head dim 128, `q = kv = 64`, causal | `validation/gqa_production_promotion_20261124/` |
| 3 | Row-packed cached attention forward | `transformer_attention_forward_rowpacked64_cached_rdna3.{comp,spv}` | `0 < q_seq == kv_seq <= 64`, equal QK/value widths, `unfused_qk ∈ {0,1,9}`, plain path | `validation/phase_x_row_packed_forward_promotion_20261125/` |
| 4 | Four-channel backward key pass | `transformer_attention_backward_key_channel4_rdna3.{comp,spv}` | `head_dim == value_head_dim == 128`, partition `32 × 4 = 128` | `validation/phase_xii_channel4_key_production_20261125/` |
| 5 | Selector, counters, scratch sizing, opt-outs | `src/transformer.rs`, `src/lib.rs`, `src/vulkan.rs`, `src/bin/transformer_train.rs` | — | all of the above |

The two-pass backward replaces a scalar kernel for the geometry the gate already
admitted. The query pass writes softmax and `dS` scratch; the key pass reduces
it. The GQA key pass is the same pair with a grouped reduction, admitted for four
measured head groupings and nothing else. The row-packed forward keeps one lane's
scores in registers and caches the `V` rows it re-reads. The four-channel key
pass splits a destination key row across 32 lanes, four channels each.

Both rejected candidates stay rejected and stay out of the production tree:

- **Phase VIII key-range partitioning** — partitioning cost **1.52×–4.57× more**
  because per-lane work, not workgroup count, is the binding constraint. Its
  shaders live only under `validation/phase_viii_key_blocks/`. A split backward
  submission is explicitly refused the two-pass pair (`key_blocks != 1` ⇒
  `key_range_split` ⇒ scalar) and keeps the scalar kernel.
- **Phase XI channel tilings other than 4** — every other `(chunk, unroll)`
  combination was measured and lost to the original key shader for its own
  geometry. Only `c=4, u=0` is promoted, and only at head dimension 128.

## 2. Numerical qualification

The promoted artifacts are the qualified artifacts, byte for byte. Their SHA-256
values are the ones their receipts record:

| Artifact | SHA-256 |
| --- | --- |
| `shaders/transformer_attention_forward_rowpacked64_cached_rdna3.comp` | `8f7bf06f65291793b12c81a9b0bd733a5ce656ffbd4a3689fea5fabf3439e6ed` |
| `shaders/transformer_attention_forward_rowpacked64_cached_rdna3.spv` | `c053503e267ede432cd174e887d800640ce854aad186f8b71390e8c4543764fd` |
| `shaders/transformer_attention_backward_query_rdna3.comp` | `2e3f5720da27ca6d31567f4734b140e8b9c768afa2e287333f617817fad7eb56` |
| `shaders/transformer_attention_backward_query_rdna3.spv` | `f9652f525cbc9256188d9c998fc602ad1eff43c68bbbc6b88650697584104256` |
| `shaders/transformer_attention_backward_key_rdna3.comp` | `e417935459e49dee1bc56db400f321496a2c04ad05392e8b14aef40d6c92ff78` |
| `shaders/transformer_attention_backward_key_rdna3.spv` | `ccf609843be2a37d33bb6df3b1ab490b7b3fa6a52e5816975c21524af71cd925` |
| `shaders/transformer_attention_backward_key_channel4_rdna3.comp` | `e1373db7ea1e7e0a1ac700ec7a9d464571f8b24f6065dbcc23c7d6df286e78f9` |
| `shaders/transformer_attention_backward_key_channel4_rdna3.spv` | `f5480ccedce8c2752f7b94b2fe478d9c9e4f496f284344c2140abc3367f7b2c8` |

- **Two-pass backward:** gradients bit-exact against the scalar route over four
  geometries — 6,316,128 / 6,316,128 FP32 elements, zero ULP. Mode 0 and mode 9
  are both covered.
- **GQA key pass:** bit-exact against FP32 and against the scalar route for
  `8q/2kv` and `4q/2kv` in both arithmetic modes, and the MHA four-geometry
  matrix is unchanged bit for bit after the swap.
- **Row-packed forward:** 1.85 M / 1.85 M words bit-exact over ten geometries and
  three arithmetic modes, plus exact step digests against the generic route.
- **Four-channel key pass:** 20 parity cases, 60 tensors, 5,505,024 elements,
  zero unwritten elements, zero failures, including an MHA control; the
  29-step end-to-end equivalence diff is 0 bytes.
- **Rejected paths:** the head-dimension-64 fallback control and the
  tiled-artifact-unavailable control both pass — an ineligible geometry keeps the
  shader it had.

## 3. Performance

| Change | Isolated | Complete step |
| --- | ---: | ---: |
| Two-pass MHA backward | 9.7965× (435.28 → 44.43 ms) | **3.2492×** (560.78 → 171.10 ms), 95 % CI 3.1146–3.3838, 5 ABBA rounds, 230 samples/arm |
| GQA grouped key pass, 8q/2kv | — | **19.76×**, 95 % CI 19.64–19.89 |
| Row-packed cached forward | 16.42 → 4.38 ms/step (Phase X attribution) | **1.2239×** on the shipped selector (1.2236× in Phase IX) |
| Four-channel key pass, head 128 | — | **1.5297×** (43.530 → 28.460 ms), 5/5 rounds |
| Mixed WMMA, LoRA | — | **1.2385×** (28.481 → 23.001 ms) |

These answer different questions. The isolated figures are kernel time; the
complete-step figures are whole optimizer steps including every unchanged kernel.
A 9.7965× backward is not a 9.7965× step, and the step figure is the one that
matters for a training claim.

## 4. Mixed WMMA: requalified, still opt-in

| Measurement | Result |
| --- | ---: |
| LoRA complete step, Phase XII harness | 28.481 ms → 23.001 ms = **1.2385×** |
| LoRA complete step, independent counter harness | **1.2288×** |
| Full-parameter complete step (timing evidence only) | **1.2792×** |

- **Default-off, unchanged.** `WmmaMode::Off` is the process default; a probe with
  no flag and no environment variable reports `mode: off`, `probed: false`,
  `enabled: false`, zero configurations.
- **The full-parameter result is timing only.** At the fixture learning rate
  `1e-2` the FP32 reference itself diverges (7.13 → 19.56), so full-parameter
  numerical behaviour at that rate is **not qualified** and the 1.2792× figure is
  not proof of it. The separate `1e-4` control (loss 6.8e-5, masters 8.0e-4) is
  the supportable claim and remains separate evidence.
- **The four LoRA adapter tensors drift elementwise and deterministically.** Their
  maxima are 0.023203, 0.022338, 0.021156 and 0.025953 — nine elements of 32,768
  above `0.02` — while every tensor's RMS stays inside the established `0.003`
  limit (worst 0.002012). **No per-element adapter-weight parity is claimed.**
- Every assertion the documented Mixed acceptance contract does make passes: LoRA
  loss trajectory, native LoRA gradient norm, adapter RMS, AdamW moments, frozen
  base parameters, rerun determinism (29/29 loss bits), exact checkpoint resume,
  the 37 `wmma_qualify` gates and the four graph-path tests.

The policy switch is process-wide and cannot distinguish LoRA from
full-parameter training, which is why a measured 1.24× LoRA step does not
justify a default change.

## 5. What stayed put

Unchanged and re-verified for this release:

- `FULL_FP16_MAX_REDUCTION = 64` (`src/precision_policy.rs`, unmodified).
- `WMMA_MIN_PROFITABLE_MACS = 16_000_000` (`src/wmma.rs`, unmodified).
- `MIXED_VS_FP32_MAX_ABS = 2.0e-2`, `MIXED_VS_FP32_RMS = 3.0e-3`,
  `MIXED_VS_MATCHED_ORACLE_MAX_ABS = 1.0e-3`, `FULL_VS_FP32_MAX_ABS = 5.0e-2`,
  `FULL_VS_FP32_RMS = 1.0e-2` — no threshold moved.
- The explicit WMMA off mode and every FP32 fallback.
- `full-fp16` and `mixed` remain distinct precision paths.
- Fail-closed admission: an unrecognized mode string is an error, not a fallback
  to a faster path.

Runtime opt-outs that behave as control arms:

| Variable | Effect |
| --- | --- |
| `HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=0` | forces the scalar attention backward, and disables the channel-tiled pipeline with it |
| `HIERARCHOS_ATTENTION_BACKWARD_KEY_CHANNEL4=0` | restores the original two-pass key shader for every geometry, touching no other kernel |
| `HIERARCHOS_ATTENTION_FORWARD_DIAGNOSTICS=1` | prints the forward decision once per distinct geometry; selects nothing |

## 6. Supported scope

- **One device.** Integrated AMD RDNA3, PCI `1002:15bf`, `AMD Radeon Graphics`,
  driver `32.0.32015.14001`, Windows 11 x64. **No universal AMD, NVIDIA, Intel or
  cross-architecture qualification is implied.**
- **GQA** covers exactly `8q/2kv`, `4q/2kv`, `8q/4kv`, `16q/4kv` at head dimension
  128 with `q_seq = kv_seq = 64` and causal masking. Other head dimensions keep
  the original key shader; `head_dim % 4 == 0` is deliberately not enough.
- **Channel4** is limited to head dimension 128 for both QK and value.
- **Row-packed forward** is limited to square sequences ≤ 64 on the plain path.
- **Mixed** keeps its full-parameter caveat and stays opt-in.
- Timings are per-step milliseconds on one machine: no throughput, power or
  multi-GPU claim.

## 7. Deferred to a later release

GQA backward query-pass optimisation, Mixed WMMA operand staging, new attention
geometries, additional head-dimension channel tiling and broad device
qualification are **not** in v0.2.7. The largest remaining kernel after this
release is the attention backward query pass (8.4 ms/step in the Phase XII
attribution).

## 8. Reproducing the record

Each receipt under `hierarchos-vulkan/validation/` ships its own `reproduce.sh`,
its raw measurements and its source/artifact hashes. The cumulative identity this
release was audited against is in
`hierarchos-vulkan/validation/phase_xiv_v027_closeout_20261125/`; the release
build itself is recorded in
`hierarchos-vulkan/validation/phase_xv_v027_release_20261125/`.
