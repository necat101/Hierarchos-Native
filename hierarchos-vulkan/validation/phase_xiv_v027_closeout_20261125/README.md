# Phase XIV — Mixed WMMA closeout and v0.2.7 release readiness

**Result: CONDITIONAL GO for v0.2.7.** The adapter-drift exception resolves as
**CONDITIONAL**: the `0.02` elementwise limit is not part of the documented Mixed
acceptance contract for adapter master weights, and every assertion the contract
does make about the LoRA arm passes. Mixed WMMA nevertheless stays **default-off
with explicit opt-in**, because the policy switch is process-wide and therefore
cannot exclude full-parameter training, whose numerics at the fixture learning
rate remain unqualified. **No production source, shader, threshold, default or
precision-policy file was changed by this phase.** Phase III–XIII are preserved
byte-for-byte.

This is a closeout, not an optimization pass. The GQA query pass and Mixed operand
staging were not touched.

---

## 1. Preservation

Base `560809eead186f832344c15d1a9b70795da3b75d`, branch `main`. Captured before any
work (`raw/00_identity.txt`) and again after this receipt was written
(`raw/09_final_identity.txt`).

| item | Phase XIII recorded | this phase | match |
| --- | --- | --- | --- |
| HEAD | `560809ee…b75d` | `560809ee…b75d` | yes |
| branch | `main` | `main` | yes |
| staged index tree | `4709d450…3760` | `4709d450…3760` | yes |
| staged diff sha256 | `e3b0c442…` (empty) | `e3b0c442…` (empty) | yes |
| full diff (`--binary`) | `3a671c75…853f` | `3a671c75…853f` | yes |
| binary-only diff (`*.spv`) | `cfa0d1c0…319a` | `cfa0d1c0…319a` | yes |
| tracked text diff | `1176e36d…6db4` | `1176e36d…6db4` | yes |
| `stash@{0}` | `831f8927…8f7e` | `831f8927…8f7e` | yes |
| stash count | 1 | 1 | yes |

The tracked-text digest is path-sensitive: it reproduces only with the
repository-root pathspec. `raw/00_identity.txt` records both variants
(`tracked_text_diff_sha256` root, `…_subdir` from `hierarchos-vulkan`) so the next
phase does not have to rediscover that. Nothing else differs.

Prior receipts, recursive sha256 over every file in sorted path order, hashed with
paths relative to `hierarchos-vulkan` exactly as Phase XIII did
(`raw/01_prior_receipt_hashes.txt`). Every value equals the Phase XIII record:

| directory | files | identity | matches Phase XIII |
| --- | ---: | --- | --- |
| `mha_attention_backward_closeout_20261124` | 36 | `bf2bde0f…a15e2` | yes |
| `gqa_attention_backward_experiment_20261124` | 13 | `ebde677a…8852d` | yes |
| `gqa_production_promotion_20261124` | 14 | `25e2a128…09107` | yes |
| `gqa_end_to_end_training_20261124` | 178 | `4c1b4d96…df21a` | yes |
| `phase_viii_key_blocks` | 26 | `b15767c3…fcd5f` | yes |
| `phase_ix_row_packed_forward_20261124` | 61 | `5d24674f…594f8b` | yes |
| `phase_x_row_packed_forward_promotion_20261125` | 66 | `b6efb85e…9ef49f` | yes |
| `phase_xi_channel_tiling_20261125` | 69 | `048ee479…3b6948` | yes |
| `phase_xii_channel4_key_production_20261125` | 59 | `92656e87…143c` | yes |
| `phase_xiii_mixed_wmma_requalification_20261125` | 172 | `78a5c8b0…ed44` | first recorded here |

All eight production attention artifacts are byte-identical to the Phase XIII
record (`raw/02_production_artifacts.txt`): forward row-packed
`8f7bf06f…3439e6ed` / `c053503e…43764fd`, backward query `2e3f5720…d7eb56` /
`f9652f52…104256`, backward key `e4179354…92ff78` / `ccf60984…1cd925`, channel4 key
`e1373db7…6e78f9` / `f5480cce…7f7b2c8`.

No reset, clean, rebase, discard, overwrite, commit, push, merge or tag was
performed. The only new untracked path is this receipt directory.

---

## 2. What the 0.02 / 0.003 limits actually govern

Two producers carry these numbers, and it matters which is which.

### 2.1 `src/wmma.rs` (code-level GEMM gates)

```text
MIXED_VS_FP32_MAX_ABS  = 2.0e-2      MIXED_VS_FP32_RMS       = 3.0e-3
MIXED_VS_MATCHED_ORACLE_MAX_ABS = 1.0e-3
FULL_VS_FP32_MAX_ABS   = 5.0e-2      FULL_VS_FP32_RMS        = 1.0e-2
WMMA_MIN_PROFITABLE_MACS = 16_000_000    FULL_FP16_MAX_REDUCTION = 64
```

`wmma.rs`, `wmma_linear.rs` and `precision_policy.rs` are **not modified in this
worktree** (`raw/03_gate_thresholds.txt`), so no threshold moved during Phase
III–XIII either. These constants are consumed by `wmma_qualify` (GEMM outputs
against FP32 and against a matched oracle) and by the `wmma_linear` graph tests
(forward / dW / dX outputs against host FP32). They are never applied to optimizer
masters or adapter weights.

### 2.2 `validation/qualify_precision_training.py` (the training contract)

The `(.02, .003)` pair is applied at exactly three places:

| assertion | what it bounds |
| --- | --- |
| `{kind}-{mode}/loss-vs-FP32` | the **loss trajectory**, for `full` and `lora` |
| `wide-{mode}/masters-vs-FP32` | **full-parameter master tensors** on the bounded 64-row / 512-wide fixture |
| `vision-{mode}/step-*` | vision forward logits and pixel/projected gradients |

LoRA master tensors are explicitly **reassigned rather than asserted**:

```python
checks[f'{kind}-{mode}/masters-reassigned']=dict(
    geometry='long-row',
    reason='numerical comparison assigned to the 64-row 512-wide fixture',
    redirect=f'wide-{mode}/masters-vs-FP32')
```

with the accompanying comment:

> Master-parameter drift is intentionally NOT asserted at this long-row geometry.
> Crossing WMMA_MIN_PROFITABLE_MACS on a hidden-64 model forces >=3907 rows,
> where the 20-step mixed trajectory leaves the sealed 0.02 envelope (0.0382
> measured at both 3936 and 4608 rows).

The producer's own precedent is therefore that `0.02` elementwise is **not** a
per-element master-weight requirement: it is an envelope tied to a bounded
trajectory, and it was **reassigned, not failed**, when a longer-row geometry left
it. The LoRA arm additionally has no FP32↔Mixed adapter-weight comparison at all;
its checks are the loss trajectory, exact resume at `2e-7` (asserted `== 0`), the
optimizer moments, and the scaler state.

**Conclusion for §2 of the brief: the `0.02` threshold does not apply to
individual adapter master-weight elements.** It applies to loss trajectories,
to full-parameter master tensors on the bounded fixture, and to vision tensors.
Phase XIII's `numerics.py` applied `MAX_ABS`/`RMS` generically to adapter weights —
a defensible *additional* observation, and the right thing to report, but not an
established gate. This phase does not reclassify it as a pass.

---

## 3. The adapter-drift exception, resolved

### 3.1 The exact quantity

FP32 arm (`--wmma-mode off`) versus Mixed, 29 optimizer steps, fixture seed 7,
batch 2, sequence 64, hidden 1024, LoRA rank 8 / alpha 16, lr 1e-2, AdamW
β 0.9/0.999, ε 1e-8. Compared tensor inventory: four q_proj adapters per layer.

| tensor | shape | max_abs | rms | elements > 0.02 | p99 \|Δ\| | rel. rms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `layers.0…q_proj.lora_A` | 8×1024 | 0.023203 | 0.001967 | 5 / 8192 | 0.006071 | 2.56 % |
| `layers.0…q_proj.lora_B` | 1024×8 | 0.022338 | 0.002012 | 1 / 8192 | 0.006500 | 2.67 % |
| `layers.1…q_proj.lora_A` | 8×1024 | 0.021156 | 0.001752 | 2 / 8192 | 0.004705 | 2.81 % |
| `layers.1…q_proj.lora_B` | 1024×8 | **0.025953** | 0.002009 | 1 / 8192 | 0.005719 | 3.20 % |

**Correction to the Phase XIII narrative:** it described the second layer's
`lora_B` as "the one number outside the envelope". All **four** tensors are above
`0.02` on `max_abs`; nine elements in total, out of 32,768. All four are inside
`0.003` on RMS. The maximum is reported, not hidden, and it is not reclassified.

### 3.2 The exact worst coordinate

| field | value |
| --- | --- |
| tensor | `base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight` |
| index | `[613][1]` |
| reference (FP32 arm) | `0.056015804409980774` |
| Mixed | `0.030062971636652946` |
| signed difference | `-0.025952832773327827` |
| absolute difference | `0.025952832773327827` |
| tensor RMS | `0.0020093538650263402` |
| tensor relative RMS | `3.20 %` |

### 3.3 Recurrence and trajectory shape

The preserved binary was reused unchanged (`training-stability.exe`,
`374e6235…55b1`, the exact executable Phase XIII recorded). Off and Mixed were
re-run at 6, 12 and 29 steps (`raw/driftsteps/`, `raw/10_adapter_drift.json`).

| steps | worst tensor `max_abs` | worst tensor `rms` | tensors over 0.02 |
| ---: | ---: | ---: | ---: |
| 6 | 0.019630 | 0.000697 | 0 |
| 12 | 0.019240 | 0.000843 | 0 |
| 29 | 0.025953 | 0.002012 | 4 |

* **RMS accumulates monotonically** in all four tensors across 6 → 12 → 29.
* **`max_abs` is flat** at 0.0168–0.0196 through 12 steps, then rises at 29. At
  most five elements of 8192 exceed 0.02 in any one tensor (one, in the worst
  tensor); p99 is 0.0047–0.0065. That is a heavy tail over a small systematic
  offset, not a whole-tensor shift.
* So the excursion is **not transient** — it accumulates — but it accumulates
  inside the RMS envelope at every measured point, and its elementwise maximum
  crosses 0.02 only at the longest step count tried.
* The 6, 12 and 29-step runs are **one trajectory**: each shorter arm's loss
  sequence is an exact prefix of the 29-step arm for both modes
  (`trajectory_prefix_consistent: {off: true, mixed: true}`), and the 29-step
  endpoints reproduce Phase XIII exactly (`off 6.552530`, `mixed 6.5513897`).
* **The 29-step rerun reproduces the reported value to 0.0**: Phase XIII worst
  `0.025952832773327827`, this phase worst `0.025952832773327827`. The exports are
  bit-identical, so the number is a deterministic property of the kernels, not
  run-to-run noise.

### 3.4 Disposition: **CONDITIONAL**

Not PASS: the elementwise maximum genuinely exceeds `0.02`, and saying otherwise
would be a retroactive loosening.

Not BLOCKED: no applicable threshold in the established contract is exceeded.
The `0.02` elementwise limit is not asserted against adapter master weights by the
producer that owns the Mixed training contract (§2.2), the producer's own documented
precedent reassigns master drift at `0.02` when a geometry leaves the bounded
fixture, and the code-level `MIXED_VS_FP32_MAX_ABS` constants gate GEMM outputs,
not optimizer masters.

Every applicable assertion passes, re-confirmed in Phase XIII and unchanged here:

| applicable assertion | measured | limit | verdict |
| --- | ---: | ---: | --- |
| LoRA loss trajectory `max_abs` | 4.02e-3 | 2.0e-2 | pass |
| LoRA loss trajectory `rms` | 1.37e-3 | 3.0e-3 | pass |
| native LoRA gradient norm `max_abs` | 7.00e-3 | 2.0e-2 | pass |
| adapter RMS (worst of 4 tensors) | 2.012e-3 | 3.0e-3 | pass |
| adapter AdamW moments | 4.39e-4 | 2.0e-2 | pass |
| base parameters (frozen) | identical | exact | pass |
| Mixed rerun determinism | 29/29 loss bits, grad digest, param digest | exact | pass |
| checkpoint save + exact resume | bit-identical, 12/12 | exact | pass |
| `wmma_qualify` Mixed, 37 gates | worst vs FP32 9.888e-3 / rms 2.092e-3 | 2.0e-2 / 3.0e-3 | pass |
| graph-path `wmma_linear`, 4/4 | — | — | pass |

The learning rate was **not** changed to produce any of this; the separate
full-parameter lr 1e-4 control remains separate evidence and is not averaged in.

**Direct evidence that the contract does not cover this metric:** §2.2 quotes the
producer's three `(.02, .003)` call sites and the LoRA `masters-reassigned`
branch with its own comment. There is no fourth site, and no adapter-weight
comparison anywhere in the file.

---

## 4. Mixed WMMA production policy: retain default-off

Decision: **`Mixed` stays opt-in. The default is unchanged at `off`.** No source
change is required to hold this, because that is already the shipped behaviour.

Selector as it stands, re-verified on this device:

```text
WmmaMode::from_env()                 HIERARCHOS_WMMA_MODE, absent/empty ⇒ Off
WmmaMode::parse()                    "off"|"none"|"fp32" ⇒ Off
wmma_shape_eligible(rows,cols,red)   rows>=16 && cols>=16 && red>=16
                                     && rows*cols>=1024 && ceil(rows/32)*ceil(cols/32)>=4
wmma_shape_profitable(...)           eligible && rows*cols*red >= 16_000_000
TransformerKernels::wmma_linear      None when Off, or when not profitable (counted fallback)
```

A probe with no `--wmma-mode` and no environment variable reports `mode: off`,
`probed: false`, `enabled: false`, `configuration_count: 0`
(`raw/04_wmma_probe_default.json`) — the default genuinely never requests the
cooperative feature chain.

Why not promote, given that the adapter disposition is not a contract violation:

1. **The switch is process-wide.** `HIERARCHOS_WMMA_MODE` / `--wmma-mode` cannot
   distinguish LoRA from full-parameter training. Phase XIII's verdict is explicit
   that full-parameter Mixed must not be promoted on its evidence: timing qualifies
   at 1.2792×, but numerics are **unqualified at the fixture lr 1e-2** because the
   FP32 oracle itself diverges there (7.13 → 19.56). Default-on would promote Mixed
   to exactly the workload whose numerical qualification is unresolved.
2. **The disposition is CONDITIONAL, not PASS.** Section 3 of the brief conditions
   promotion on a pass; a conditional disposition with a reported elementwise
   excursion is not the strongest possible footing for a default change.
3. **A default is a behaviour change for every eligible user**, and the benefit is
   workload-shaped: the wide projections win 1.89×, but the narrow K/V forward is
   0.93× in isolation and the four adapter GEMMs never accelerate at all.

What is preserved, unchanged: the explicit `off` override, every FP32 fallback,
`FULL_FP16_MAX_REDUCTION = 64`, `WMMA_MIN_PROFITABLE_MACS = 16_000_000`, the
Mixed tile, the staging schedule, all fail-closed admission, and the separation
between `full-fp16` and Mixed FP16-input/FP32-accumulation.

The performance result is documented for the release as an opt-in capability:
**1.2385×** LoRA complete step on the Phase XII harness, **1.2288×** on the
independent counter harness, **1.2792×** full-parameter timing, 5/5 ABBA rounds in
each, with no claim of clean numerical qualification for full-parameter training.

---

## 5. Minimal validation performed

No kernel, shader or policy changed, so the historical program was not repeated.
What was run:

| check | command | result |
| --- | --- | --- |
| default selection | `wmma_probe --json` (no mode, no env) | `mode=off, enabled=false, probed=false, configurations=0` |
| explicit `off` dispatch | `training-stability` at 6/12/29 steps | `wmma_*` all 0 cooperative, 0 fallbacks, at every step count |
| Mixed eligible counters | same, `HIERARCHOS_WMMA_MODE=mixed` | 15 fwd + 0 dW + 15 dX per step (90/180/435 totals), 0 fallbacks |
| ineligible adapters on FP32 | `wmma_linear::tests::ineligible_shapes_keep_their_own_kernel_and_are_counted` | pass (fresh) |
| numerical gates | `wmma_qualify --wmma-mode mixed --json` | `gates: 37, failed: 0`, `status: PASS` — output byte-identical to Phase XIII, sha256 `c17df140…3385` |
| graph-path gates | lib tests `wmma_linear` | `4 passed; 0 failed` |
| end-to-end comparison | off vs mixed, 29 steps | loss `6.552530` vs `6.5513897`; both reproduce Phase XIII exactly |
| thresholds | `raw/03_gate_thresholds.txt` | all unchanged; threshold-bearing files unmodified at HEAD |
| adapter trajectory | 6 / 12 / 29 steps | §3.3, monotone RMS, max flat then crossing at 29 |

Reused receipts: Phase XIII's 37/37 Mixed gates, 4/4 graph tests, three ABBA series
(5/5 rounds each), kernel attribution and numerics; Phase XII's channel4 promotion
and its ABBA; Phase X's selector matrix. None was invalidated, because no kernel
changed.

Not done, deliberately: no new device or architecture was qualified, no shader was
rebuilt, no ABBA series was re-run, and no broad numerical sweep was opened.

---

## 6. Release inventory (cumulative Phase III–XIII)

### 6.1 Promoted

| # | feature | production files | eligibility | numerics | receipt |
| ---: | --- | --- | --- | --- | --- |
| 1 | Two-pass attention backward (default for the existing gate) | `transformer_attention_backward_query_rdna3.{comp,spv}`, `…_key_rdna3.{comp,spv}`, `src/transformer.rs` | `unfused_qk` 0/9; existing gate unchanged | bit-exact, 4 geometries, 0 ULP | `mha_attention_backward_closeout_20261124` `bf2bde0f…` |
| 2 | GQA grouped key-pass reduction | same key pair | `8q/2kv`, `4q/2kv`, `8q/4kv`, `16q/4kv`, head_dim 128, q=kv=64, causal | bit-exact vs FP32 and vs scalar | `gqa_production_promotion_20261124` `25e2a128…` |
| 3 | GQA end-to-end training promotion | — | as above | 8q/2kv complete step 19.76× (CI 19.64–19.89) | `gqa_end_to_end_training_20261124` `4c1b4d96…` |
| 4 | Row-packed attention forward | `transformer_attention_forward_rowpacked64_cached_rdna3.{comp,spv}`, `src/transformer.rs` | `0 < q_seq == kv_seq <= 64`; equal qk/value head dim; `unfused_qk ∈ {0,1,9}`; plain path only | bit-exact 1.85M/1.85M words, 10 geometries, 3 modes; 1.2236× step (Phase IX), 1.2239× on shipped selector | `phase_x_row_packed_forward_promotion_20261125` `b6efb85e…` |
| 5 | Four-channel key pass | `transformer_attention_backward_key_channel4_rdna3.{comp,spv}`, `src/transformer.rs` | `head_dim == value_head_dim == 128`, `blocks*chunk == 128`; opt-out `HIERARCHOS_ATTENTION_BACKWARD_KEY_CHANNEL4=0` | bit-exact incl. MHA control; 1.528× step | `phase_xii_channel4_key_production_20261125` `92656e87…` |

Rejected and retained as evidence: Phase VIII key-block partitioning
(`phase_viii_key_blocks` `b15767c3…`) — partitioning cost 1.52×–4.57× more because
per-lane work, not workgroup count, is the binding constraint.

### 6.2 Opt-in, not promoted

| feature | state | evidence |
| --- | --- | --- |
| Mixed WMMA (`--wmma-mode mixed`) | opt-in, default `off` | 1.2385× / 1.2288× LoRA, 1.2792× full-parameter timing; adapter disposition CONDITIONAL (§3) |
| Full FP16 (`--wmma-mode full-fp16`) | opt-in, unchanged since v0.2.6 | Phase LXXXVIII C1 promotion; not touched here |

### 6.3 Supported scope — stated narrowly

* One device: `AMD Radeon Graphics`, vendor `0x1002`, device `0x15bf` (integrated
  RDNA3), one driver, one machine. No discrete RDNA3, no RDNA4, no NVIDIA/Intel
  claim is made.
* WMMA numbers: 2 layers, hidden 1024, intermediate 1024, vocab 1024, 8 query heads
  × 128, 2 KV heads × 128, batch 2 × sequence 64 = 128 GEMM rows, LoRA rank 8 /
  alpha 16.
* Attention paths: only the geometries listed in §6.1. MHA plus four measured
  grouped shapes; `head_dim == 128` for the channel-tiled key pass; square
  sequences ≤ 64 for the row-packed forward.

### 6.4 Limitations — do not overclaim

* **Not universal GQA coverage.** Four measured grouped shapes plus MHA at head
  dimension 128. Other head dimensions keep the original key shader; generalizing
  to `head_dim % 4 == 0` would need its own qualification.
* **Not universal WMMA acceleration.** Only RDNA3-or-newer passing the full
  cooperative feature contract, and only shapes above 16 MMAC. The narrow K/V
  forward (128×256×1024) is 0.93× — slower in isolation — and the adapter GEMMs are
  ineligible by geometry (out_dim 8, reduction 8).
* **No general full-parameter numerical guarantee** at the unstable 1e-2 fixture
  rate. The 1e-4 control (loss 6.8e-5, masters 8.0e-4) is the supportable claim.
* LoRA adapter elementwise drift reaches 0.0260 on one tensor, with four tensors
  above 0.02 and every RMS inside 0.003 (§3).
* Timings are per-step milliseconds on one machine; no throughput, power or
  multi-GPU claim.

### 6.5 Actual release blockers

**None for the Phase III–XII scope.** The four promoted optimizations are
byte-identical to their qualified artifacts, their receipts verify, and their
gates re-pass. The conditions below are conditions on *claims and defaults*, not
code defects:

1. Mixed must remain `off` by default (§4).
2. Release notes must scope performance to the tested device and geometries (§6.3).
3. Release notes must not present full-parameter Mixed as numerically qualified at
   lr 1e-2, and must record the adapter observation (§3).

### 6.6 Deferred, explicitly not blockers

* **GQA backward query pass** — 9.173 ms/step, 31.0 % of the Mixed step, now the
  largest single kernel. Its share grew only because the key pass left the
  distribution.
* **Mixed WMMA operand staging (C1-style batching)** — 7.411 ms GEMM plus 1.169 ms
  conversion per step for 3.22 GMAC of projection work.
* Narrow K/V forward shape (0.93×), `head_dim` 16/32/64 channel tiling, non-square
  and >64-key attention, and any device beyond the one tested.

---

## 7. Documentation defects found while verifying

Two, neither a threshold or artifact change, both recorded rather than edited,
because prior receipts must not be rewritten:

1. Phase XIII's README (§5) says "the one number outside the envelope is …
   `lora_B`" — actually **four** adapter tensors exceed 0.02 (§3.1 here).
2. Phase XIII's README (§5) cites `128x128x96 trans_a=true trans_b=true`,
   `max_abs 8.63e-3` as "the worst `vs FP32` row". The worst is
   `128x128x96 wmma-mixed vs FP32` at **9.888e-3** (rms 2.092e-3), still inside
   2.0e-2 / 3.0e-3. `raw/05_wmma_qualify_mixed.json` is byte-identical between the
   two phases, so this is a transcription slip, not a re-measurement.

Some earlier receipt READMEs are written with CRLF line endings while newer ones
are LF; this is cosmetic and was left alone.

---

## 8. Files added by this phase

| file | role |
| --- | --- |
| `README.md` | this receipt |
| `capture_identity.sh` | identity, preservation, artifact hashes, thresholds |
| `adapter_drift.py` | drift-vs-steps, worst coordinate, reproduction check |
| `summarize_drift.py` | readable reduction of `raw/10_adapter_drift.json` |
| `reproduce.sh` | re-runs the drift series and the gate checks |
| `raw/` | identity before/after, prior receipt hashes, artifact hashes, thresholds, probe, 37 gates, 4/4 graph tests, drift series and reductions |
