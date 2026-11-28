# Phase XVIII — v0.2.8 promotion of the dQ accumulator specialization

**Status: GO — SPECIALIZED PATH PROMOTED.**

The Phase XVII candidate is promoted into the production two-pass attention
backward as a distinct shader identity,
`transformer_attention_backward_query_rdna3_dqunroll`, selected only when

```text
qk_head_dim == 128 && value_head_dim == 128
```

Everything else keeps the shipped query shader,
`transformer_attention_backward_query_rdna3`, unchanged and byte-identical. The
promoted artifact is the Phase XVII SPIR-V byte for byte (`f0214b60…`), rebuilt
from its source in this tree (`430492fd…`) with the same `glslc` invocation, and
that invocation reproduces the shipped control module exactly (`f9652f52…`).

---

## 1. Baseline and repository state

| item | value |
| --- | --- |
| branch | `research/v0.2.8-optimization-discovery` |
| base commit / HEAD | `9b0091e02e976441e343d6c445101f9c29395bd6` |
| `main` | `9b0091e02e976441e343d6c445101f9c29395bd6` (unchanged) |
| `v0.2.7` tag object | `85b64a1eb9be1e925e791c1c29b69ced74e23404` |
| `v0.2.7^{commit}` | `9b0091e02e976441e343d6c445101f9c29395bd6` |
| `stash@{0}` | `831f892774619603c2dd68d0dd654d315f1a8f7e` (unchanged) |
| commits created | none |
| pushes / merges / tags / releases | none |

`git rev-parse v0.2.7` returns the tag object, not the commit; the dereferenced
commit is the published baseline. The research branch had not moved since Phase
XVII: HEAD is still the v0.2.7 merge, and the only tracked change was the 27-line
`#[cfg(test)]` seam Phase XVII added. See `raw/00_identity.txt`.

### 1.1 Preservation

`raw/02_preservation.txt` recomputes the Phase XII–XVII receipt digests with the
Phase XVII method (hash of the `sha256sum` lines of every file in sorted path
order). Every prior receipt reproduces its recorded value exactly, including

| receipt | digest | files | bytes |
| --- | --- | ---: | ---: |
| `phase_xii_channel4_key_production_20261125` | `92656e87…3143c` | 59 | 2,058,249 |
| `phase_xiii_mixed_wmma_requalification_20261125` | `78a5c8b0…8ed44` | 172 | 4,967,740 |
| `phase_xv_v027_release_20261125` | `53ec9443…e77ff6` | 9 | 44,501 |
| `phase_xvii_v028_optimization_discovery_20261126` | `a7daded2…59cf2` | 71 | 807,003 |

The Phase XVII directory is byte-identical to what it was when it was sealed
(its files still carry their 2026-11-26 01:42 timestamps); Phase XVIII only read
it, and copied the candidate source and artifact out of it.

---

## 2. Production change

### 2.1 Shader identity

| artifact | sha256 | role |
| --- | --- | --- |
| `shaders/transformer_attention_backward_query_rdna3.comp` | `2e3f5720…7eb56` | shipped source, **unchanged** |
| `shaders/transformer_attention_backward_query_rdna3.spv` | `f9652f52…04256` | shipped module, **unchanged** |
| `shaders/transformer_attention_backward_query_rdna3_dqunroll.comp` | `430492fd…8dffc` | promoted source (Phase XVII candidate, byte-identical) |
| `shaders/transformer_attention_backward_query_rdna3_dqunroll.spv` | `f0214b60…9d237` | promoted module (Phase XVII candidate, byte-identical) |

`src/transformer.rs` embeds the promoted module with `include_bytes!`
(`ATTENTION_BACKWARD_QUERY_DQUNROLL_SPV`), exactly like every other committed
shader. No binary is copied at run time and no validation directory is a build
input: `build.rs` registers every `shaders/*.spv` by FNV-1a signature, which is
what gives the new module its profiler identity.

### 2.2 Selector

```rust
fn dqunroll_query_attention_rejection(push: &AttentionPush) -> Option<&'static str> {
    if push.num_heads == 0 || push.num_key_value_heads == 0 {
        return Some("invalid_geometry");
    }
    let head_dim = push.query_hidden_size / push.num_heads;
    let value_head_dim = push.value_hidden_size / push.num_key_value_heads;
    if head_dim != DQUNROLL_QUERY_HEAD_DIM {          // 128
        return Some("dqunroll_query_head_dim_not_128");
    }
    if value_head_dim != DQUNROLL_QUERY_HEAD_DIM {    // 128
        return Some("dqunroll_query_value_head_dim_not_128");
    }
    None
}
```

That is the whole geometry condition. Arithmetic mode, causal range, masks,
sequence length and the MHA/grouped allowlist are *not* re-decided here: the
enclosing two-pass gate settles them and the shader handles both admitted modes.

`HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL=0` restores the shipped query
shader for every geometry (the control arm of the A/B). Unset, `1`, `true` and
`on` keep the eligible default. The switch inherits
`HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=0`, because the promoted pass is a
variant of the two-pass route, not a separate route.

### 2.3 Fallback

`record_two_pass_attention_backward_dispatch` keeps one pass-1 dispatch. It
dispatches the promoted module when the selector admits the geometry *and* the
optional artifact initialized; otherwise it dispatches `passes[0]`, the shipped
query kernel, with the same bindings, push block and grid
(`[ceil(rows/64),1,1]`). A pair installed by a harness (`reason != "ready"`,
i.e. an experimental arm) keeps its own kernel, so no experimental arm is
silently replaced by the production artifact.

The device-level fallbacks live in `DqUnrollQueryBackward::load`, not in the
selector: a non-qualified adapter family, a device that cannot launch a 64-wide
workgroup, or a pipeline the driver refuses all leave pass 1 on the shipped
shader and record a reason (`dqunroll_device_not_qualified`,
`dqunroll_shader_unavailable`). This mirrors the Phase XII channel-4 key pass.

### 2.4 What did not change

| invariant | evidence |
| --- | --- |
| two query-backward dispatches per step | `raw/23_step_smoke_*.log`: `backward_query_original + backward_query_dqunroll = 10` for 5 steps × 2 layers; `commands.dispatch_count() == 2` asserted on every parity case |
| dispatch grid | unchanged `[ceil(rows/64),1,1]`; parity log prints `grid_x` per case |
| push-constant layout | no field added or reordered; `size_of::<AttentionPush>() = 120` unchanged, printed by the selector test |
| descriptor bindings | still 9 for pass 1; the module is built with `binding_count = 9` |
| scratch layout and 1 MiB/step | unchanged `2 × 65,536` floats per layer; step harness reports `scratch_bytes=1048576` per step in both arms |
| fallback shader contents | `f9652f52…` before and after; control rebuild byte-identical |
| Mixed WMMA eligibility / defaults / policy | untouched |
| two softmax-dot reductions | still separate in the shader (`softmax_dot` plain, `kv_softmax_dot` precise) |
| Candidate A score/value caching | not introduced |
| arithmetic association, mode 0 and mode 9 | statement-for-statement identical source; the promoted module differs only in the literal-index expansion |

The Phase XVII test-only SPIR-V seam (`HIERARCHOS_PHASE_XVII_QUERY_SPV`) was
removed: it existed to time an artifact that had no production identity, and it
was `#[cfg(test)]`-only. The Phase XVII test module is self-contained (it embeds
its own `q_rebuild.spv` / `q_dqunroll.spv` copies) and still passes unchanged.
No production build ever compiled the seam, and none compiles anything from
`validation/`.

### 2.5 Build path

`validation/training_shader_manifest.json` gains the promoted module with the
exact flags, and `reproduce_shaders.sh` rebuilds it:

```text
glslc -fshader-stage=compute --target-env=vulkan1.2 --target-spv=spv1.4 -O -Ishaders \
  shaders/transformer_attention_backward_query_rdna3_dqunroll.comp \
  -o shaders/transformer_attention_backward_query_rdna3_dqunroll.spv
```

Toolchain: `shaderc v2023.8 v2025.3` / `glslang 11.1.0-1253-gefd24d75`,
`spirv-tools v2025.3` (VulkanSDK 1.4.321.1). The same invocation applied to the
shipped source reproduces `f9652f52…` byte for byte, so the promoted module's
delta is attributable to the source change and not to different flags.

---

## 3. Compiler and resource findings

### 3.1 Validation

`spirv-val --target-env vulkan1.2` passes on the promoted module, the shipped
module and the rebuilt control (`raw/12_spirv_val.txt`).

### 3.2 The accumulator in SPIR-V

`analyze_spirv.py` reads `spirv-dis` output and separates Function-storage
float-array variables from their access chains (`raw/15_spirv_analysis.txt`):

| module | Function-storage `float[128]` variables | constant-index chains | runtime-index chains |
| --- | ---: | ---: | ---: |
| shipped `f9652f52` | 1 (`%1297`) | 0 | 4 |
| promoted `f0214b60` | 1 (`%1294`) | 128 (128 distinct constants, 0…127) | 4 |

The promoted module is a superset: its single remaining array variable and its
four runtime-index chains belong to the untouched generic fallback body, which
still exists for every geometry the host never admits. The specialized body
addresses the accumulator only through 128 distinct constant indices, so **the
specialized path no longer requires dynamic function-storage indexing**. The
module grows from 48,492 to 109,300 bytes and from 1,806 to 4,430 instructions,
which is the expected cost of expanding the three accumulator statements.

### 3.3 Register residency

`spirv-dis` shows what the module asks the IHV compiler for; it cannot show what
LLVM/LLPC does with it. The AMD ISA toolchain (`RadeonGPUAnalyzer`, `amdllvm`) is
not installed on this host, so no ISA dump is available. Residency is therefore
taken from the driver's own build statistics via `VK_AMD_shader_info` §3.4
rather than inferred from the source form.

### 3.4 `VK_AMD_shader_info`

`raw/16_amd_shader_info.txt`, produced by `examples/amd_shader_resource_probe.rs`:
each module is built into a compute pipeline with the production descriptor set
(9 storage buffers), push block (120 bytes) and entry point, then queried for
`VK_SHADER_INFO_TYPE_STATISTICS` on the same device and driver as the training
run.

| module | VGPR | SGPR | scratch bytes | LDS bytes | workgroup |
| --- | ---: | ---: | ---: | ---: | --- |
| shipped query `f9652f52` | 27 | 61 | 528 | 0 | 64×1×1 |
| promoted query `f0214b60` | 36 | 62 | 528 | 0 | 64×1×1 |
| reference: key channel-4 `f5480cce` | 35 | 88 | 0 | 0 | 64×1×1 |

The promoted module costs **9 more VGPRs and 1 more SGPR**. That is what 128
constant-index accumulator accesses lower to: the addresses become compile-time
constants, but the values they touch stay live across the loop, and LLPC spends
registers to keep them there instead of spilling them. Crucially **scratch is
unchanged at 528 bytes**, so the expansion introduced no new spilling; the
528 bytes are the pre-existing private allocation both query modules carry.
Neither module uses LDS, and both require the same 64-wide workgroup, so the
device qualification in `load` (RDNA3 family, 64-wide workgroup support) is
unchanged by the promotion. The probe also reports `physical_vgpr=1024`,
`physical_sgpr=2048`, `available_vgpr=256`, `available_sgpr=106` and
trailing `numPhysicalDevices=0` for every module; the trailing field is
reported verbatim and is not used by any decision here.

---

## 4. Numerical coverage

### 4.1 Selector

`raw/20_selector.log`. The predicate is exercised without a device:
8q/2kv, 8q/8kv and 4q/2kv at 128/128 are admitted; head dimension 64 (either
dimension), 96, and mismatched pairs are rejected with distinct reasons;
degenerate head counts return `invalid_geometry` instead of dividing by zero.

Selector proof is not inferred from numerical parity: every parity case also
reports the production dispatch counter it took.

### 4.2 Dispatch-helper matrix

`raw/22_parity.log`, driven through
`record_two_pass_attention_backward_dispatch` (the function the optimizer step
calls) with the production-promoted artifact. Reference arm: the shipped query
module, installed as a harness pair. Both arms hold the shipped key module, so
the comparison is attributable to pass 1.

| geometry | sequence | mode | causal | masks |
| --- | --- | --- | --- | --- |
| 8q/2kv | 64, 128 | 0, 9 | causal, non-causal | all visible, dropped query + key |
| 8q/8kv MHA | 64, 128 | 0, 9 | causal, non-causal | all visible, dropped query + key |
| 4q/2kv | 64, 128 | 0, 9 | causal, non-causal | all visible, dropped query + key |
| 4q/2kv | 37 | 0, 9 | causal, non-causal | dropped / all visible (Phase XVII case preserved) |
| 8q/2kv head dim 64 | 64 | 0, 9 | causal | dropped / all visible (fallback control) |

Outputs compared bitwise: `dQ`, `scratch_probability`, `scratch_ds`, plus `dK`
and `dV` to prove the shipped key pass consumes the promoted pass's scratch.

```text
summary cases=30 specialized=28 fallback=2 tensor_comparisons=150 failed=0
```

All 150 comparisons are bitwise equal with zero non-finite values
(`raw/22_parity.log`). The two head-dimension-64 cases report
`candidate_query=1 candidate_dqunroll=0`, i.e. the shipped fallback actually
dispatched.

Write coverage is **not** independently measured. Both arms pre-fill every output
and scratch slot with the same sentinel, so the comparison proves the two shaders
agree everywhere it looks; it cannot distinguish a slot both shaders leave at the
sentinel from one both wrote identically. The `unwritten=0` field in the
historical `raw/22_parity.log` was a hardcoded constant, not a measurement, and
has been removed from the test output rather than restated as a result. The
bitwise assertions themselves are unchanged.

**Helper level, not the full production route.** These cases call
`record_two_pass_attention_backward_dispatch` — the function a step calls — with
the pair and both selectors presented as a step presents them, but without the
enclosing step's own gate. 13 of the 28 specialized cases are grouped
configurations that `grouped_query_attention_rejection` rejects before pass 1 in
a real step: the two non-causal 8q/2kv sequence-64 cases, the four 8q/2kv
sequence-128 cases, the non-causal 4q/2kv sequence-64 case, the four 4q/2kv
sequence-128 cases, and the two 4q/2kv sequence-37 cases. The matrix therefore
establishes shader/helper-level parity across the listed configurations; it does
not establish that every listed configuration is admitted by the full production
training route. The production eligibility rule and the grouped-sequence gate are
unchanged by this phase, and each complete-step measurement in §7 exercises only
the configurations its own per-process counters report. The 4q/2kv sequence-128
series remains excluded (§4.3, §7.3) and is not a specialized-path measurement.

### 4.3 Production route at sequence 128 for 4q/2kv

The dispatch-helper matrix exercises the promoted pass at 4q/2kv sequence 128,
but the *enclosing production route* does not reach it: the pre-existing
grouped-query qualification admits sequence 64 only
(`grouped_query_attention_unqualified_sequence`), so the real step uses the
scalar kernel there. This phase does not widen that gate — it is an unrelated
attention-configuration selector and outside the promotion's scope. The
step-level measurement in §7 therefore reports that series as a null result with
`backward_scalar > 0`, not as a candidate effect.

---

## 5. In-graph training comparison

LoRA, rank and alpha from the existing fixture config, learning rate `1e-2`,
29 complete optimizer steps (3 warmup + 26 measured), `arm A` =
`HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL=0`, `arm B` = promoted default.
Both processes run the full production path (`graph.train_step`) with all
parameter updates applied, so every step's loss is the loss of the model the
previous step produced, not a replayed forward.

```text
[phase-xviii-in-graph] series=in-graph steps=29 grad_compared=29 \
  loss_bits_equal=True param_end_a=d79a12d8…9c4655 param_end_b=d79a12d8…9c4655 \
  mismatches=0 verdict=PASS
```

| check | result |
| --- | --- |
| `loss_bits` equal on all 29 steps | yes (`raw/30_in_graph.json` carries the whole sequence) |
| LoRA gradient digest equal on all 29 steps | 29/29 |
| parameter digest equal after every step | 29/29 |
| parameter end digest | `d79a12d882fd780deaada83a940374ab753b0a3bc5586a7ca348735ed29c4655` in both arms |
| production dispatch counters | A: `query_original=58 query_dqunroll=0`; B: `query_dqunroll=58 query_original=0`; both `key_channel4=58`, `scalar=0` |

The first losses reproduce Phase XVII's in-graph sequence bit for bit
(`40e44309`, `40e43f10`, `40e30547`, …), so the promoted build is continuous
with the discovery measurements. Bitwise equality here is stronger than "no
numerical regression": the specialized accumulator is not merely close, it is
the same computation in the same association order, which is what §4.2's
statement-level identity predicted.

---

## 6. Full-parameter result

Full finetuning (`HIERARCHOS_PHASE_XVIII_LORA=0`) at the CLI's default
full-finetune learning rate `5e-5` (`transformer_train.rs`), same 29 steps, all
13,636,608 parameters updated in place.

```text
[phase-xviii-in-graph] series=full-parameter steps=29 grad_compared=0 \
  loss_bits_equal=True param_end_a=39b91a8f…277c37 param_end_b=39b91a8f…277c37 \
  mismatches=0 verdict=PASS
```

| check | result |
| --- | --- |
| `loss_bits` equal on all 29 steps | yes |
| parameter digest equal after every step | 29/29 (no per-step gradient readback exists on this path; the parameter digest is the observable) |
| parameter end digest | `39b91a8fd9d2952fe4c960e160694175a2b0db94e38efe96ec294e1d63277c37` in both arms |
| loss trajectory | `7.133183e0` → `6.8517265e0` → `5.6944585e0` → `5.8465433e0` (step 29) — descending and stable |
| dispatch counters | identical to §5 |

A six-step smoke at `lr = 1e-2` had already shown the same equality, alongside
the known aggressive-rate divergence (loss `7.13` → `16.8` → `12.6`, parameter
norm `108.7` → `147.4`). That instability is a property of the learning rate on
this fixture — both arms traverse it identically — which is why the reported
full-parameter run uses the default `5e-5` and reports a stable trajectory
instead of a diverging one.

---

## 7. Complete-step performance

Protocol per geometry and precision mode: **5 rounds × 4 separate OS processes**,
schedule `A B B A`, 3 warmup + 26 measured complete steps per process, no
profiler, no parameter readback (`HIERARCHOS_PHASE_XVIII_STEP_DIGEST=0`, so the
harness never reads 13.6 M parameters back per step), same fixture, seed and
batch for both arms. Arm A is `HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL=0`;
arm B is the promoted default. `A` = median of the two arm-A process medians,
`B` = median of the two arm-B process medians, ratio `B/A`, CI over the five
round ratios (two-sided t, df = 4).

### 7.1 Admitted geometries (milliseconds per complete step)

| mode | geometry | seq | A | B | B/A | speedup | rounds improved | 95 % CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP32 | 8q/2kv | 64 | 28.494 | 26.821 | 0.9411 | 1.063× | 5/5 | 0.9399–0.9427 |
| FP32 | 8q/8kv MHA | 64 | 32.279 | 30.704 | 0.9514 | 1.051× | 5/5 | 0.9477–0.9546 |
| FP32 | 8q/8kv MHA | 128 | 110.475 | 105.912 | 0.9601 | 1.042× | 5/5 | 0.9549–0.9647 |
| FP32 | 4q/2kv | 64 | 17.196 | 15.348 | 0.8922 | 1.121× | 5/5 | 0.8909–0.8930 |
| Mixed | 8q/2kv | 64 | 23.020 | 21.374 | 0.9282 | 1.077× | 5/5 | 0.9270–0.9293 |
| Mixed | 8q/8kv MHA | 64 | 24.829 | 23.238 | 0.9365 | 1.068× | 5/5 | 0.9352–0.9374 |
| Mixed | 8q/8kv MHA | 128 | 90.889 | 86.246 | 0.9490 | 1.054× | 5/5 | 0.9477–0.9498 |
| Mixed | 4q/2kv | 64 | 17.114 | 15.253 | 0.8897 | 1.124× | 5/5 | 0.8851–0.8994 |

Every entry is its own experiment; nothing is pooled across geometries or modes.
The smallest effect is MHA at sequence 128 (`1.042×` FP32, `1.054×` Mixed), the
largest is the 4q/2kv sequence-64 geometry (`1.12×`), which is also the fixture
with the smallest step, so its relative weight of the pass-1 kernel is highest.

Per-process dispatch counters (`raw/40_abba_*.json`, `counters`) show every A
process ran `query_original=58 query_dqunroll=0` and every B process
`query_dqunroll=58 query_original=0`, with `key_channel4=58`, `scalar=0`, and
`2 layers × 29 steps = 58` dispatches on each. The step count, not the shader,
is 58 in both arms, confirming the A/B difference cannot come from a changed
dispatch count.

### 7.2 Position, not arm

The schedule puts arm A at positions 0 and 3 and arm B at 1 and 2, so a clock or
warm-up artifact would appear as a position effect. For every admitted series
the position medians split by **arm**: 8q8kv-s128 Mixed, for example, is
`pos0=90.885 pos1=86.224 pos2=86.227 pos3=90.887` — the two A positions agree to
0.002 ms with each other and differ from both B positions by 4.7 ms. Across all
admitted runs the run-to-run spread is 5–13 % (`run_spread_ratio`), far below the
separation between arms, and the round ratios repeat to within 0.3 % (FP32
4q/2kv-s64: 0.8912, 0.8928, 0.8922, 0.8910, 0.8925).

### 7.3 Excluded geometry: 4q/2kv at sequence 128

Both arms run the same route here. Every one of the 40 processes reports
`backward_scalar=58`, `backward_query_original=0`, `backward_query_dqunroll=0`,
`backward_key_channel4=0`: the pre-existing grouped qualification rejects
sequence 128 before either pass-1 shader is considered. The series therefore
measures *nothing* about the promotion, and the two results show that plainly:

| mode | A (ms) | B (ms) | B/A | rounds "improved" | 95 % CI | run spread |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| FP32 | 4238.512 | 3703.199 | 0.8778 | 5/5 | 0.8722–0.8814 | 1.341 |
| Mixed | 4234.989 | 4232.341 | 0.9997 | 3/5 | 0.8698–1.1087 | 1.341 |

Identical code cannot improve; the difference is a bimodal clock state. One run
per round lands at ~3.17–3.24 s instead of ~4.24 s, at rotating positions
(rounds 0–4: pos 2, 1, 2, 2, 1 in FP32) — a 1.34× step change set by the
platform, not by the arm. In FP32 all five fast runs fell in B positions, in
Mixed there was one in each arm, which is the whole of the 12 % versus 0.03 %
"effect". This series establishes two things: the excluded geometry is unaffected
by the promotion (same kernel, same counters), and the median-of-two estimator is
only trustworthy when the run distribution is unimodal — which is the case for
every admitted series (§7.1, run spread ≤ 1.13 except this one).

8q/2kv at sequence 128 is not measured: that geometry is rejected by the same
grouped qualification, its scalar step costs ~8.7 s, and it is not one of the
brief's required geometries.

### 7.4 Sham control (identical shaders in all four positions)

To separate the schedule from the shader, the same driver was run with
`PHASE_XVIII_SHAM=1`, which exports `HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL=0`
in every position: the true effect is exactly zero. Same fixtures, same 5 rounds,
same `A B B A` layout, same reduction (`raw/41_sham.json`).

| mode | geometry | A (ms) | B (ms) | B/A | 95 % CI | run spread |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| FP32 | 8q/8kv MHA seq 128 | 109.070 | 108.993 | 0.9995 | 0.9984–1.0009 | 1.005 |
| Mixed | 8q/8kv MHA seq 128 | 90.851 | 90.804 | 0.9993 | 0.9985–1.0003 | 1.004 |
| FP32 | 4q/2kv seq 64 | 17.223 | 17.202 | 1.0010 | 0.9935–1.0049 | 1.015 |
| Mixed | 4q/2kv seq 64 | 17.096 | 17.116 | 1.0010 | 0.9981–1.0030 | 1.006 |
| FP32 | 8q/2kv seq 64 | 28.460 | 28.471 | 1.0002 | 0.9988–1.0020 | 1.007 |
| Mixed | 8q/2kv seq 64 | 23.002 | 22.999 | 0.9992 | 0.9970–1.0023 | 1.005 |

The schedule contributes **at most 0.1 %** on these geometries, and the sham's
absolute times agree with the real arm-A times (§7.1: 109.07 vs 110.48,
17.22 vs 17.20, 28.46 vs 28.49 ms), so the control is in the same regime rather
than a different one. The measured effects of 4.2–12.4 % are 40–120× the
control's spread, every round of every admitted series improved, and the
position medians separate by arm. The promotion is what moved them.

### 7.5 Bookkeeping

`logs/abba_driver.log` ends without the closing `PHASE_XVIII_ABBA_DONE` marker
because `run_abba_step.sh` was edited while bash was still reading it. The log
records the two resulting wrapper errors, and they are the only failures in it:

| log line | text |
| --- | --- |
| 201 | `run_abba_step.sh: line 85: out_mode: unbound variable` |
| 203–204 | `run_abba_step.sh: line 92: syntax error near unexpected token 'done'` |

Both are the wrapper failing on the partially re-read revision of the script, not
a test process: the line numbers refer to what bash had read at that byte offset,
not to the file now on disk.

The log contains **201 record lines**: the 200 unique `wmma=… exit=0` records of
the campaign — one per (mode, series, round, position) — plus one duplicated
reprint of the final record
(`wmma=mixed series=4q2kv-s128 round=4 pos=3 tag=A arm=original expect=scalar`),
which carries `exit=1` from the failed wrapper echo. That reprint is not a 201st
process: every one of the 200 unique records is a completed `exit=0` run, all 200
per-process logs are present, each passes the reducer's "exactly 26 measured
steps" check, and the reduction of the complete set is `raw/40_abba.json`. The
sham runs used the frozen script and each finished with its own
`PHASE_XVIII_ABBA_DONE`.

---

## 8. Profiler attribution

`raw/32_profile_A.log`, `raw/32_profile_B.log`, summarised in
`raw/32_attribution.txt`. Nine complete steps per arm with
`HIERARCHOS_VULKAN_PROFILE_KERNELS=1` and `SAMPLE_STRIDE=1`, so every dispatch
is attributed:

| query | arm A | arm B |
| --- | --- | --- |
| `shader=transformer_attention_backward_query_rdna3` | 2 dispatches per step, attributed | absent |
| `shader=transformer_attention_backward_query_rdna3_dqunroll` | absent | 2 dispatches per step, attributed (9/9 steps) |
| `shader=unknown` lines | 0 | 0 |

The promoted module has a registered identity through the manifest entry and
`include_bytes!`; the profiler resolves it and never falls back to
`shader=unknown` for either arm. The profiled GPU times are not used for any
performance claim — sampling at stride 1 perturbs execution, and §7's unprofiled
ABBA is the measurement. They are reported only to show attribution:
`query_rdna3` 6.69–16.22 ms per two dispatches in arm A,
`query_rdna3_dqunroll` 5.06–12.36 ms in arm B, each `pct` 18–24 % of its step.

---

## 9. Decision

**GO — SPECIALIZED PATH PROMOTED.**

The promotion ships as the default for exactly `qk_head_dim == 128 &&
value_head_dim == 128`; every other geometry keeps the shipped query shader,
byte-identical, through the same dispatch helper.

| requirement | evidence | verdict |
| --- | --- | --- |
| exactly the 128/128 pair, nothing else | selector test 3 admitted / 6 rejected; parity's head-dim-64 cases dispatch the shipped module | met |
| shipped shader unchanged | `f9652f52…` before and after; control rebuild byte-identical | met |
| dispatch grid, push block, bindings, scratch, two query dispatches/step | parity asserts `dispatch_count == 2` on every case; step counters 58 = 2 × 29; grid printed per case; `scratch_bytes=1048576` both arms | met |
| Mixed WMMA policy and both softmax-dot reductions untouched | source differs only in the literal-index expansion; Mixed series measured separately (§7.1) | met |
| bitwise equality `dQ`, `scratch_probability`, `scratch_ds` (+ `dK`, `dV`), zero non-finite | 150/150 tensor comparisons bitwise, `non_finite=0` (`raw/22_parity.log`); write coverage not independently measured (§4.2) | met |
| exact in-graph `loss_bits` | 29/29 equal, LoRA and full-parameter (§5, §6) | met |
| complete-step gain, unprofiled, per geometry and mode | 8 series, 5/5 rounds improved each, 1.042×–1.124×, CI excludes 1.0 in all eight (§7.1) | met |
| no regression elsewhere | excluded geometry runs the identical scalar route; sham control bounds schedule bias at 0.1 % (§7.3, §7.4) | met |
| profiler identity replaces `shader=unknown` | 9/9 steps attributed to `transformer_attention_backward_query_rdna3_dqunroll`; 0 `unknown` lines in either arm | met |
| device qualification | `DqUnrollQueryBackward::load` admits RDNA3 with 64-wide workgroups; probe reports VGPR 27→36, SGPR 61→62, scratch 528→528, workgroup 64×1×1 | met |

Limits stated rather than hidden: no AMD ISA dump exists on this host, so
residency is the driver's report, not a disassembly (§3.3); the parity matrix is
helper-level and does not show that every listed geometry is admitted by the full
training route (§4.2); write coverage of the outputs is not independently
measured (§4.2); the 4q/2kv sequence-128 geometry is outside the enclosing
two-pass qualification and was not touched; and the 12 % figure on that excluded
series is a measurement artifact documented in §7.3, not a benefit.

The Phase XVII candidate was a 1.6 ms kernel saving with no production identity.
It is now a 1.0–1.9 ms complete-step saving (4.2–12.4 %) with an identity, a
fallback, a switch, a control and a halfway honest error bar.

---

## 10. Changed files

| path | change |
| --- | --- |
| `src/transformer.rs` | promoted constant, selector, loader, counters, dispatch choice; Phase XVII test seam removed; `#[cfg(test)] mod` registrations for `phase_xvii_v028_optimization_discovery_tests` and `phase_xviii_dq_accumulator_promotion_tests` |
| `src/phase_xvii_v028_optimization_discovery_tests.rs` | new test-only module carried in this change set; it is a source file of the current development work, not part of the sealed Phase XVII receipt |
| `src/phase_xviii_dq_accumulator_promotion_tests.rs` | new test module (selector, parity, loading, step driver) |
| `shaders/transformer_attention_backward_query_rdna3_dqunroll.comp` | new production source (Phase XVII candidate, byte-identical) |
| `shaders/transformer_attention_backward_query_rdna3_dqunroll.spv` | new production artifact (Phase XVII candidate, byte-identical) |
| `validation/training_shader_manifest.json` | manifest entry for the promoted module |
| `examples/amd_shader_resource_probe.rs` | VGPR/SGPR/scratch probe via `VK_AMD_shader_info`; builds each module into a compute pipeline with the production descriptor/push shape and reads the driver's statistics |
| `validation/phase_xviii_v028_dq_accumulator_promotion_20261126/` | this receipt |

---

## 11. Reproduction

```text
cd hierarchos-vulkan
bash validation/phase_xviii_v028_dq_accumulator_promotion_20261126/reproduce_shaders.sh
cargo test --release --lib --no-run
./target/release/deps/hierarchos_vulkan-*.exe phase_xviii_dq_accumulator_selector --nocapture --test-threads=1
./target/release/deps/hierarchos_vulkan-*.exe phase_xviii_dq_accumulator_parity --nocapture --test-threads=1
./target/release/deps/hierarchos_vulkan-*.exe phase_xviii_dq_accumulator_optional_loading --nocapture --test-threads=1
# one campaign covers the five series in both modes (200 processes, ~70 min
# because the excluded series spends seconds per scalar step)
PHASE_XVIII_TEST_BIN=target/release/deps/hierarchos_vulkan-<hash>.exe \
  bash validation/phase_xviii_v028_dq_accumulator_promotion_20261126/run_abba_step.sh
python validation/phase_xviii_v028_dq_accumulator_promotion_20261126/reduce_step.py \
  validation/phase_xviii_v028_dq_accumulator_promotion_20261126/raw 5 \
  off:8q2kv-s64 off:8q8kv-s64 off:8q8kv-s128 off:4q2kv-s64 off:4q2kv-s128 \
  mixed:8q2kv-s64 mixed:8q8kv-s64 mixed:8q8kv-s128 mixed:4q2kv-s64 mixed:4q2kv-s128 \
  > validation/phase_xviii_v028_dq_accumulator_promotion_20261126/raw/40_abba.json
# sham control: identical shaders in all four positions, same schedule
for s in 8q8kv-s128 4q2kv-s64 8q2kv-s64; do
  PHASE_XVIII_SHAM=1 PHASE_XVIII_SERIES_FILTER=$s \
  PHASE_XVIII_TEST_BIN=target/release/deps/hierarchos_vulkan-<hash>.exe \
    bash validation/phase_xviii_v028_dq_accumulator_promotion_20261126/run_abba_step.sh
done
python validation/phase_xviii_v028_dq_accumulator_promotion_20261126/reduce_step.py \
  validation/phase_xviii_v028_dq_accumulator_promotion_20261126/raw 5 \
  sham-off:8q8kv-s128 sham-mixed:8q8kv-s128 sham-off:4q2kv-s64 sham-mixed:4q2kv-s64 \
  sham-off:8q2kv-s64 sham-mixed:8q2kv-s64 \
  > validation/phase_xviii_v028_dq_accumulator_promotion_20261126/raw/41_sham.json
# profiler attribution and driver residency
PHASE_XVIII_TEST_BIN=target/release/deps/hierarchos_vulkan-<hash>.exe \
  bash validation/phase_xviii_v028_dq_accumulator_promotion_20261126/profile_attribution.sh
cargo build --release --example amd_shader_resource_probe
./target/release/examples/amd_shader_resource_probe.exe \
  shaders/transformer_attention_backward_query_rdna3.spv \
  shaders/transformer_attention_backward_query_rdna3_dqunroll.spv \
  shaders/transformer_attention_backward_key_channel4_rdna3.spv
```

Every timing process is unprofiled (`HIERARCHOS_PHASE_XVIII_STEP_DIGEST=0`, no
`HIERARCHOS_VULKAN_PROFILE_*`), and the profiler runs are never used as
timings. `raw/40_abba.json`, `raw/41_sham.json`, `raw/30_in_graph.json` and
`raw/31_fullparam.json` carry the machine-readable results; the per-process logs
stay beside them in `raw/` for independent re-reduction.

Fixtures: `target/gqa-e2e/fixture-8q2kv` and `fixture-4q2kv` are the existing
Phase VII fixtures (sequence 64). The sequence-128 and MHA fixtures are built
with the same generator and seed:

```text
python validation/make_wmma_fixture.py --output target/phase-xviii/fixture-8q2kv-s128 \
  --hidden 1024 --layers 2 --vocab 1024 --heads 8 --kv-heads 2 --head-dim 128 \
  --seq-len 128 --batch-size 2 --sequences 16
python validation/make_wmma_fixture.py --output target/phase-xviii/fixture-8q8kv-s64 \
  --hidden 1024 --layers 2 --vocab 1024 --heads 8 --kv-heads 8 --head-dim 128 \
  --seq-len 64 --batch-size 2 --sequences 16
python validation/make_wmma_fixture.py --output target/phase-xviii/fixture-8q8kv-s128 \
  --hidden 1024 --layers 2 --vocab 1024 --heads 8 --kv-heads 8 --head-dim 128 \
  --seq-len 128 --batch-size 2 --sequences 16
python validation/make_wmma_fixture.py --output target/phase-xviii/fixture-4q2kv-s128 \
  --hidden 512 --layers 2 --vocab 1024 --heads 4 --kv-heads 2 --head-dim 128 \
  --seq-len 128 --batch-size 2 --sequences 16
```
