# Phase VII — end-to-end RDNA3 GQA Transformer training benchmark

**Date:** 2026-11-24
**Base commit:** `560809eead186f832344c15d1a9b70795da3b75d`
**Verdict:** **GO** for the current production GQA two-pass path.
**Measured:** 8q/2kv **19.76×** complete-step speedup (95 % CI [19.64, 19.89]),
4q/2kv **18.30×** (95 % CI [18.22, 18.37]), with bit-identical training.

This phase measures the promoted GQA attention-backward path in the actual native
Transformer training graph. It changes no shader, no eligibility predicate, no
source file and no existing receipt. The only new files are this receipt
directory and its harness.

---

## 1. Preserved Phase III–VI working tree

Recorded before any measurement, on the actual tree:

| Check | Value |
| --- | --- |
| Branch | `main` |
| `git rev-parse HEAD` | `560809eead186f832344c15d1a9b70795da3b75d` (expected) |
| `git status --short` | 17 entries: 13 modified tracked files + 4 untracked receipt directories |
| `git diff --stat` | 13 files changed, 2813 insertions(+), 202 deletions(−) |
| `git diff \| sha256sum` | `383296d7b58b0664563eda3b3b7ac1fa862beb9d7e804fb2715603a558243d77` |
| `git diff --cached \| sha256sum` | `e3b0c442…b855` (empty index) |
| `git stash list` | `stash@{0}: On main: pre-v0.2.6 untracked validation evidence (preserved, not deleted)` |
| `git rev-parse stash@{0}` | `831f892774619603c2dd68d0dd654d315f1a8f7e` |
| Untracked | `gqa_attention_backward_experiment_20261124/` (Phase V), `gqa_production_promotion_20261124/` (Phase VI), `mha_attention_backward_closeout_20261124/` (Phase III), `gqa_end_to_end_training_20261124/` (this receipt) |

No reset, rebase, clean, checkout, discard or commit was performed. `stash@{0}`
is untouched. No existing receipt was edited.

### Production shader identities

| File | sha256 | Matches Phase VI |
| --- | --- | --- |
| `shaders/transformer_attention_backward.comp` | `3c8e6267e320c1f187175209e8a42e6a2658494c6bb3762b1e009cb8b7810a34` | yes |
| `shaders/transformer_attention_backward.spv` | `e78cb2920e3253d7187c479c23bee40d4b4ab677fedeaa97f3109bf6a00dc5a8` | yes |
| `shaders/transformer_attention_backward_query_rdna3.comp` | `2e3f5720da27ca6d31567f4734b140e8b9c768afa2e287333f617817fad7eb56` | yes |
| `shaders/transformer_attention_backward_query_rdna3.spv` | `f9652f525cbc9256188d9c998fc602ad1eff43c68bbbc6b88650697584104256` | yes |
| `shaders/transformer_attention_backward_key_rdna3.comp` | `e417935459e49dee1bc56db400f321496a2c04ad05392e8b14aef40d6c92ff78` | yes |
| `shaders/transformer_attention_backward_key_rdna3.spv` | `ccf609843be2a37d33bb6df3b1ab490b7b3fa6a52e5816975c21524af71cd925` | yes |

### Prior receipt identities (read only)

| Receipt | sha256 |
| --- | --- |
| Phase III `mha_attention_backward_closeout_20261124/qualification.json` | `7127160ff8b498567ef3563b4e92c667c217b713ddc945465377150d23c5a851` |
| Phase V `gqa_attention_backward_experiment_20261124/README.md` | `14ebf32d3860e4e7ce5bf96368edbef396cd0d2c52fbe02efbe7ec97ba7c465c` |
| Phase VI `gqa_production_promotion_20261124/README.md` | `19a44800b6e589a5e011a53212545e8b26b43c69945276912e911da05850b3af` |

### Discrepancy check against the Phase VI report

Two recorded values legitimately differ and are explained rather than treated as
drift:

1. **`git diff --stat` is larger than Phase VI's start-of-phase value.**
   Phase VI recorded 11 files / 2437 insertions at *its* start. Phase VI then
   promoted the key pass and edited `transformer.rs`, `vulkan.rs`, `lib.rs`,
   `transformer_train.rs`, `RDNA3_WMMA.md` and four bins. The current
   13-file / 2813-insertion tree is Phase VI's **end** state, which is the state
   Phase VII is required to preserve. Nothing was added by Phase VII.
2. **The full-diff hash differs from Phase VI's `7f926469…`.** That value was
   the *pre-Phase-VI* tree. Phase VI did not record an end-of-phase diff hash.
   The current hash `383296d7…` is therefore recorded here as the Phase VII
   baseline, and the end-of-phase check below confirms it is unchanged.

The stronger identity check is the shader table above: all six production
artifacts hash to the exact post-promotion values Phase VI recorded, including
the promoted key SPIR-V `ccf60984…` that Phase V measured bit-exact.

`src/transformer.rs` hashes to
`3e4bde38cc835c1957458c791bdb2495ab0151af624521fa72df93eb3d670143`, which is the
Phase VI end state (Phase V's end value `fcca5595…` predates the promotion).
`src/vulkan.rs` (`e1ec6dc0…`) and `src/lib.rs` (`34caa583…`) match the Phase V
receipt's recorded values. The qualification binary source
(`53c075a4…`) is recorded here for the first time.

---

## 2. Workload and the geometry the native parser actually loaded

### Primary: 8 query heads / 2 KV heads

| Parameter | Requested | Loaded by `VulkanTransformerConfig::from_hf_config` |
| --- | ---: | ---: |
| Batch size | 2 | 2 (`--batch 2`) |
| Query heads | 8 | 8 |
| KV heads | 2 | 2 |
| Queries per KV head | 4 | 4 (derived) |
| Head dimension | 128 | 128 (explicit `head_dim`) |
| Query sequence | 64 | 64 (`--seq 64`) |
| KV sequence | 64 | 64 (self-attention) |
| Causal | 1 | 1 (Llama) |
| Hidden size | — | 1024 = 8 × 128 |
| Layers | — | 2 |
| Vocabulary | — | 1024 |
| Parameters | — | 13,636,608 |

The fixture is the established two-layer, hidden-size-1024 shape from Phase III,
generated by `validation/make_wmma_fixture.py` with `--heads 8 --kv-heads 2
--head-dim 128 --seq-len 64 --batch-size 2 --sequences 32`. The parser's Llama
branch (`from_llama_family_value`) reads `num_key_value_heads` explicitly
(`transformer.rs:23388`) and rejects a non-divisible pair, so the requested
geometry cannot be silently ignored. The fixture's `config.json` contains
`"num_key_value_heads": 2`, `"num_attention_heads": 8`, `"head_dim": 128`; its
hash is `f5f1703b…` (full list in `raw/fixture-hashes.txt`).

The model advertises GQA in its config **and** the runtime push constants agree
— the latter is the evidence that matters:

```
two-pass backward: route=two-pass eligible=true reason=eligible kernels=true
  scratch=true key_blocks=1 batch=2 heads=8 kv_heads=2 q_seq=64 kv_seq=64
  head_dim=128 causal=1 unfused_qk=9 scratch_elements=65536 scratch_bytes=524288
```

| Push constant / dispatch | Value |
| --- | ---: |
| `num_heads` | 8 |
| `num_key_value_heads` | 2 |
| `head_dim` | 128 |
| `query_seq_len` | 64 |
| `key_value_seq_len` | 64 |
| `causal` | 1 |
| `unfused_qk` | 9 (Llama materialised-SLEEF adjoint) |
| `key_blocks` | 1 |
| `batch_size` | 2 |
| Scratch elements / bytes per layer | 65,536 / 524,288 |

The production eligibility predicate is the Phase VI one, unchanged:

```
(num_heads, num_key_value_heads) ∈ {(8,2), (4,2), (8,4), (16,4)}
head_dim == 128; q_seq == kv_seq == 64; causal == 1; unfused_qk ∈ {0, 9}
plus every pre-existing MHA restriction and key_blocks == 1
```

Both layers are identical decoder layers, so the decision diagnostic — which is
keyed on the full decision and prints once per distinct geometry — emits one
line for both. Per-layer execution is proven by the counters below, not by the
deduplicated line.

### Secondary: 4 query heads / 2 KV heads

Same sequence (64), head dimension (128), batch (2), layers (2), vocabulary
(1024) and dataset; hidden size is 512 = 4 × 128, which is the only compatible
width for 4 heads at head dimension 128. Its runtime line is
`heads=4 kv_heads=2 head_dim=128 q_seq=64 kv_seq=64 causal=1 unfused_qk=9`,
`scratch_elements=32768 scratch_bytes=262144`, and its config hash is
`bb01d962…`.

---

## 3. Scalar versus accelerated dispatch proof

### Native CLI trainer, 2 steps, one process per arm

`hierarchos-vulkan-transformer-train`, `--lora-rank 8 --lora-alpha 16
--max-steps 2`, `HIERARCHOS_WMMA_MODE=off`:

| Arm | `HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS` | scalar | query | key | scratch allocations | scratch bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | `0` | 4 | 0 | 0 | 0 | 0 |
| B | `1` | 0 | 4 | 4 | 2 | 1,048,576 |

`4 = 2 steps × 2 layers` per shader. The accelerated arm never dispatched the
scalar backward; the scalar arm never dispatched either two-pass shader. The
accelerated arm allocated exactly one 524,288-byte scratch pair per layer
(`two-pass backward scratch: layer=1 … bytes=524288` and `layer=0 … bytes=524288`),
and the scalar arm allocated none. Both arms ended at loss
`7.132698059082031`.

### Profiled qualification harness, 12 steps

Per-shader dispatch counts from the shipped GPU timestamp profiler:

| Arm | scalar backward / step | query / step | key / step |
| --- | ---: | ---: | ---: |
| scalar | 2.0 | 0.0 | 0.0 |
| two-pass | 0.0 | 2.0 | 2.0 |

### Default environment

With `HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS` genuinely unset (removed from the
child environment, not blank):

```
two-pass backward: route=two-pass eligible=true reason=eligible …
two-pass backward counters: query=6 key=6 scalar=0   (3 steps × 2 layers)
```

The production selector chooses the GQA two-pass route automatically, without
the variable.

### Fail-closed behaviour of the explicit arm

`HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=1` is a hardware qualification, not a
hint. Two checks:

* **Kernel initialization** — the existing production-path test
  `optional_two_pass_load_failure_keeps_scalar_execution_and_gradients`
  (fault injected through the real loader, not a mock) passes:
  `status=PASS cases=8 default_query=1 default_key=1 default_scalar=0
  fallback_scalar=1 explicit_zero_loads=0 scratch_fallbacks=0`. It asserts that
  `=1` with an unavailable query shader returns a `BLOCKED …
  query_shader_unavailable` error instead of a silent scalar run, while the
  eligible default falls back to scalar with bit-identical gradients.
* **No device** — `=1` with the Vulkan loader pointed at a nonexistent ICD exits
  non-zero (`ERROR_INCOMPATIBLE_DRIVER`), so an accelerated arm cannot report a
  measurement without a GPU.

---

## 4. Numerical training parity

Two LoRA steps, `--digest-every 1`, identical fixture, dataset, batch order,
seed (7), optimizer (AdamW, lr 1e-2, β 0.9/0.999, ε 1e-8, no weight decay),
sequence length, precision (WMMA off) and dropout configuration. Each arm is a
separate process. Every field below is bit-identical between arms.

### 8q/2kv

| Check | Scalar | Two-pass |
| --- | --- | --- |
| Loss bits (step 1, step 2) | `[1088701193, 1088700176]` | `[1088701193, 1088700176]` |
| Losses | `[7.133183, 7.132698]` | `[7.133183, 7.132698]` |
| LoRA gradient digest (step 1) | `48819c3607764300d638af50b65d326173b15888029766711681e6a44673cfc2` | same |
| LoRA gradient digest (step 2) | `9093107b4bec860d5a41544408451f5988205ffdaaf88c0d17a154c0d5bb886e` | same |
| Adapter parameter digest (both steps) | `d79a12d882fd780deaada83a940374ab753b0a3bc5586a7ca348735ed29c4655` | same |
| Parameter-end digest | `d79a12d8…c4655` | same |
| Base parameters frozen | `true` | `true` |
| `adapter_model.safetensors` | `136989b6823f0e7dd94b36e0faa448d96fee3c0c7b6eae6d707c5db3c26c7a03` | same |
| `adapter_config.json` | `1fa6572209fb7a4c5fce8f4efcfaef8a90b11af831952a76dcb76b2a5da6d696` | same |
| AdamW optimizer state (`hierarchos_adapter_optimizer.safetensors`) | `54ea9a8c4bbf1df909fe4eb2173b7f845ebc1674609d94be6b3e7bf71b52905d` | same |

### 4q/2kv

| Check | Scalar | Two-pass |
| --- | --- | --- |
| Loss bits | `[1088458129, 1088450572]` | same |
| Losses | `[7.017281, 7.0136776]` | same |
| Gradient digests | `ed565c35…29ad2`, `d5ffcf67…27fd9` | same |
| Adapter parameter digest | `12a5187a7d9f3605210459e0388e33f1e9367a72d776b8cbe1627e3306aa6c1b` | same |
| `adapter_model.safetensors` | `9d2389b09015e371e1e5a9d34b9a029371079ddc14cad82b560ab2621ba1b6d3` | same |
| `adapter_config.json` | `1fa65722…6d696` | same |
| Optimizer state | `b5c95332d88362f185c8a5e1f5176d60d535b13fdab7c0be19bbf4c3b634e306` | same |

**No numerical gate was relaxed.** The attention gradients (the LoRA gradient
digest covers every adapted attention projection), the per-step losses, the
adapter weights, the adapter configuration and the AdamW moments are all
bit-identical. A timing difference therefore cannot come from a different
training trajectory or from missing work: the accelerated arm did the same
arithmetic and skipped only the scalar backward kernel.

---

## 5. Complete-step timing

### Protocol

* Device: `AMD Radeon Graphics` (integrated RDNA3, `deviceID 0x15bf`), Vulkan
  API 1.4.359, driver `2.0.416` / `26.20.15.14 (LLPC)`, `timestamp_period_ns=10`.
* Host: rustc 1.93.0; the trainer is `hierarchos-vulkan-training-qualification`
  (`0a66bc366d490e720beb8b067043f8721eca6c84f4f200e2f8ee2eb7c430dbe0`).
* 29 optimizer steps per process; first **3 warm-up**, next **26 measured**.
* 5 paired ABBA rounds on 8q/2kv, 3 on 4q/2kv. Order per round:
  scalar, two-pass, two-pass, scalar. Each run is a fresh process; runs are
  strictly serial, one at a time, with no other GPU work in flight.
* `HIERARCHOS_WMMA_MODE=off` in every arm; profiling off for timing.
* Identical fixture, dataset, batch order, seed, optimizer and learning rate in
  every run. The only per-arm difference is
  `HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=0|1`.
* The paired statistic is `median(scalar step_ms) / median(two-pass step_ms)`
  per round; the interval is the two-sided Student-t interval over the paired
  ratios (`validation/abba_stats.py`, df = rounds − 1).

### 8q/2kv — primary

| Metric | Scalar | Two-pass |
| --- | ---: | ---: |
| Samples per arm | 260 (5 × 2 × 26) | 260 |
| Median step | 1050.144 ms | 53.072 ms |
| Mean step | 1049.952 ms | 53.262 ms |
| Standard deviation | 3.006 ms | 0.584 ms |
| P95 | 1054.656 ms | 54.500 ms |
| Min / max | 1042.275 / 1057.306 ms | 52.400 / 55.578 ms |

Paired round ratios: `19.7866, 19.8105, 19.8025, 19.5850, 19.8226`.

* **Mean paired speedup: 19.7615×**
* **95 % CI: [19.6379, 19.8850]** (df = 4, t = 2.7764)
* Geometric mean 19.7613×; ratio of medians 19.787×
* Complete-step time eliminated: **997.07 ms of 1050.14 ms (94.95 %)**

Noise assessment: the scalar arm's standard deviation is 0.29 % of its mean and
the two-pass arm's is 1.1 %. No adjustment to the protocol was needed. Round 3's
two-pass median (53.63 ms) is the only visible excursion; it is retained, not
discarded, and it widens the interval honestly.

### 4q/2kv — secondary cross-check

| Metric | Scalar | Two-pass |
| --- | ---: | ---: |
| Samples per arm | 156 (3 × 2 × 26) | 156 |
| Median step | 525.399 ms | 28.712 ms |
| Mean step | 525.177 ms | 28.766 ms |
| Standard deviation | 1.559 ms | 0.298 ms |
| P95 | 527.345 ms | 29.293 ms |

Paired round ratios: `18.3292, 18.2740, 18.2844`.

* **Mean paired speedup: 18.2959×**
* **95 % CI: [18.2231, 18.3687]** (df = 2, t = 4.3027)
* Ratio of medians 18.299×; eliminated 496.69 ms of 525.40 ms (94.54 %)

This is a cross-check, not a second campaign: three rounds are enough to show the
second qualified grouping translates into a complete-step gain of the same
order, and the wide df = 2 interval is reported as such.

Raw samples: `samples/abba-8q2kv.jsonl` (20 run records) and
`samples/abba-4q2kv.jsonl` (12 run records). Each record carries the round,
position, arm, every one of the 29 per-step `step_ms` values, and the arm's route
line.

---

## 6. Per-kernel attribution

Twelve profiled steps per arm (`HIERARCHOS_VULKAN_PROFILE_KERNELS=1`,
`…SAMPLE_STRIDE=1`, `…TOP_SHADERS=4096`). Values are GPU timestamp brackets,
milliseconds **per step** (two layers). Shares are of dispatched GPU time.

### 8q/2kv

| Bucket | Scalar ms | Scalar share | Two-pass ms | Two-pass share |
| --- | ---: | ---: | ---: | ---: |
| Attention backward — query pass | 0.000 | 0.00 % | 7.629 | 13.20 % |
| Attention backward — key pass | 0.000 | 0.00 % | 15.956 | 27.61 % |
| Attention backward — scalar | 1016.981 | 96.67 % | 0.000 | 0.00 % |
| Other attention (forward, RoPE, elementwise) | 15.248 | 1.45 % | 15.747 | 27.25 % |
| Linear forward | 7.196 | 0.68 % | 7.354 | 12.72 % |
| Linear weight gradient | 1.032 | 0.10 % | 0.054 | 0.09 % |
| Linear input gradient | 6.629 | 0.63 % | 6.466 | 11.19 % |
| Normalization | 2.754 | 0.26 % | 2.871 | 4.97 % |
| MLP (SiLU fwd/bwd) | 0.070 | 0.01 % | 0.065 | 0.11 % |
| Optimizer (AdamW) | 0.009 | 0.00 % | 0.008 | 0.01 % |
| Other (loss, embedding, adds, LoRA scale) | 2.119 | 0.20 % | 1.646 | 2.85 % |
| **Total dispatched GPU** | **1052.039** | 100 % | **57.795** | 100 % |
| Dispatches per step | 98 | | 98 | |

Per-shader detail for the accelerated arm (ms/step, dispatches/step):
`transformer_attention_backward_key_rdna3` 15.956 (2.0),
`transformer_attention_forward` 15.599 (2.0),
`transformer_attention_backward_query_rdna3` 7.629 (2.0),
`linear_forward` 7.168 (17.0), `linear_input_grad` 6.344 (17.0),
`transformer_rms_norm_input_grad` 1.468 (5.0),
`transformer_rms_norm_forward` 1.403 (5.0),
`falcon_h1_cross_entropy_lanes16` 1.234 (1.0), then all remaining shaders below
0.35 ms/step.

### 4q/2kv

| Bucket | Scalar ms | Scalar share | Two-pass ms | Two-pass share |
| --- | ---: | ---: | ---: | ---: |
| Attention backward — query pass | 0.000 | 0.00 % | 7.818 | 23.02 % |
| Attention backward — key pass | 0.000 | 0.00 % | 8.912 | 26.24 % |
| Attention backward — scalar | 511.005 | 96.79 % | 0.000 | 0.00 % |
| Other attention | 9.057 | 1.72 % | 9.894 | 29.13 % |
| Linear forward | 1.887 | 0.36 % | 2.025 | 5.96 % |
| Linear weight gradient | 0.048 | 0.01 % | 0.042 | 0.12 % |
| Linear input gradient | 1.965 | 0.37 % | 2.088 | 6.15 % |
| Normalization | 1.586 | 0.30 % | 1.527 | 4.50 % |
| MLP | 0.041 | 0.01 % | 0.037 | 0.11 % |
| Optimizer | 0.007 | 0.00 % | 0.004 | 0.01 % |
| Other | 2.351 | 0.45 % | 1.622 | 4.77 % |
| **Total dispatched GPU** | **527.945** | 100 % | **33.968** | 100 % |

### GPU execution versus host submission and fence waiting

| Run | Profiled wall/step | Sum of GPU brackets | Outside brackets |
| --- | ---: | ---: | ---: |
| 8q/2kv scalar | 1058.728 ms | 1052.039 ms | 6.689 ms (0.63 %) |
| 8q/2kv two-pass | 62.544 ms | 57.795 ms | 4.749 ms (7.59 %) |
| 4q/2kv scalar | 533.638 ms | 527.945 ms | 5.693 ms (1.07 %) |
| 4q/2kv two-pass | 38.455 ms | 33.968 ms | 4.487 ms (11.67 %) |

Two honest caveats. The timestamp profiler itself costs wall time: the same
two-pass step measures 62.54 ms profiled versus 53.07 ms unprofiled, i.e. about
9.5 ms/step. Its brackets are therefore used for **relative** attribution, not as
a decomposition of the unprofiled step. What the table does establish is that
host submission and fence waiting are a single-digit-millisecond quantity: even
in the profiled runs, 92.4 % (8q/2kv) of wall time is inside dispatch brackets,
and the unprofiled step is *faster* than the profiled bracket sum. The step is
GPU-execution-bound, not submission-bound.

---

## 7. What the GQA path actually eliminated

| Quantity | 8q/2kv | 4q/2kv |
| --- | ---: | ---: |
| Complete step, scalar | 1050.144 ms | 525.399 ms |
| Complete step, two-pass | 53.072 ms | 28.712 ms |
| **Complete-step speedup** | **19.76×** | **18.30×** |
| Step time eliminated | 997.07 ms (94.95 %) | 496.69 ms (94.54 %) |
| Attention-backward GPU, scalar | 1016.981 ms | 511.005 ms |
| Attention-backward GPU, two-pass | 23.585 ms | 16.730 ms |
| **Backward GPU eliminated** | **993.40 ms (97.68 %)** | **494.28 ms (96.73 %)** |
| Isolated backward ratio at this geometry | 43.1× | 30.5× |

The end-to-end gain (19.76×) is **larger** than the isolated MHA kernel ratio
Phase III reported (9.80×) and **smaller** than the isolated GQA kernel ratio
Phase VI reported (47.9×). Neither isolated number predicts the other: the
complete step contains work the isolated harness does not, and the isolated
harness measures one dispatch in a warm loop rather than two layers inside a
training graph. This is exactly why the step-level measurement was required.

Amdahl, on the accelerated arm's own GPU attribution:

* Attention backward is now 23.59 ms of 57.80 ms (40.8 %), down from 96.7 %.
* All attention (forward + backward + RoPE + elementwise) is 39.33 ms (68.1 %).
* Dense projections (forward + weight grad + input grad) are 13.87 ms (24.0 %).
* Normalization is 2.87 ms (5.0 %), the loss projection/CE 1.23 ms (2.1 %),
  MLP 0.07 ms (0.1 %), AdamW 0.008 ms (0.01 %).
* If attention backward were eliminated **entirely**, the floor would be
  35.06 ms — a further 1.65× over the current 57.80 ms and 30.0× over the
  scalar GPU step. That is the entire remaining headroom in the step.

---

## 8. Dominant bottleneck after the GQA promotion

The bottleneck is **attention itself**, not the dense projections:

1. **Key pass — 15.96 ms/step (27.6 %)**, the single largest kernel. It runs
   `batch × kv_heads = 4` workgroups at 8q/2kv (2 at 4q/2kv), against 16 for the
   query pass. Phase V/VI already identified this parallelism starvation; this
   phase measures its end-to-end cost.
2. **Attention forward — 15.60 ms/step (27.0 %)**, essentially tied. No
   diagnosed structural defect; it is the same kernel in both arms.
3. **Query pass — 7.63 ms/step (13.2 %)**, already 2× faster than the key pass
   with 4× the workgroups, which is the direct evidence that the key pass is
   workgroup-starved rather than arithmetic-bound.
4. Dense projections — 13.87 ms (24.0 %) — are the next bucket, but they are
   spread over 42 dispatches and are the natural target of the existing WMMA
   path, not of new kernel work.

The optimizer (0.01 %), MLP elementwise (0.1 %) and normalization (5 %) are not
bottlenecks. Runtime overhead is not a bottleneck either (§6).

---

## 9. GO / NO-GO

**GO** for the current production GQA two-pass path, on this evidence:

* 19.76× complete-step speedup at 8q/2kv (CI lower bound 19.64×) and 18.30× at
  4q/2kv (CI lower bound 18.22×), measured with paired ABBA on separate
  processes, identical fixtures and identical training configuration;
* every measured field of the training trajectory is bit-identical between arms:
  losses, attention gradients, adapter weights, adapter config, parameter-end
  digest and AdamW moments;
* the accelerated arm dispatched the query and key passes once per layer per
  step and the scalar backward **zero** times; the scalar arm did the reverse;
* the default environment selects the GQA two-pass route with no variable set;
* `=1` fails closed on kernel-initialization failure and on a missing device;
* scratch is allocated once per eligible layer (2 × 524,288 bytes) and not at
  all in the scalar arm;
* the production predicate and the shaders were not modified.

**NO-GO** for any broader claim:

* two of the four qualified groupings were measured end to end (8q/2kv, 4q/2kv);
  8q/4kv and 16q/4kv remain qualified numerically but unmeasured at step level;
* one head dimension (128), one symmetric sequence shape (64/64), causal only,
  unmasked, unit value scale;
* one machine, one driver, one thermal state; WMMA off. The absolute times are
  not comparable to other hardware, and the interval covers run-to-run variation
  on this machine only.

---

## 10. Single highest-value next optimization

**Split the key pass across key blocks** — the `key_blocks` dimension the gate
already reserves and currently requires to be 1.

Why this one, from the measurements:

* it is the largest single kernel in the accelerated step (15.96 ms, 27.6 %);
* its defect is measured, not inferred: 4 workgroups at 8q/2kv versus 16 for the
  query pass, and the query pass does 4× the grouped work in half the time;
* the fix is **order-preserving per output element**: each `dK[key, dim]` /
  `dV[key, dim]` reduces over queries in ascending order, so partitioning key
  rows across workgroups changes which block owns a row, never the accumulation
  order inside it. Bit-exactness with the scalar oracle is therefore
  preservable — unlike splitting the reduction itself;
* expected effect: reaching the query pass's 7.63 ms would save 8.3 ms/step
  (53.07 → ~44.8 ms, a further **1.19×**); reaching 4 workgroups' worth of
  ideal scaling could save ~12 ms (**1.29×**). This is Amdahl-supported: the
  step is 68 % attention, and the key pass is its largest component.

Runner-up, if the key pass is not taken: attention forward (15.60 ms, 27.0 %).
It has no diagnosed defect and touches the softmax path, so it is higher risk
for a similar ceiling.

Orthogonal configuration lever, not a kernel rewrite: enable the existing WMMA
path for the dense projections. They are 13.87 ms (24.0 %) with WMMA off, and
the fixture (hidden 1024, 1024-column projections) is WMMA-eligible. This is a
measurement to take before it is a recommendation; it was deliberately held
constant in this phase to isolate attention backward.

**Not recommended:** further scalar-backward work (it is gone), optimizer work
(0.01 %), or MLP work (0.1 %).

---

## 11. Receipt contents

```
validation/gqa_end_to_end_training_20261124/
  README.md                     this document
  summary.json                  machine-readable results (all tables above)
  run_benchmark.py              the harness that produced every number
  reproduce.sh                  reruns the whole protocol
  samples/
    abba-8q2kv.jsonl            20 run records × 29 per-step samples
    abba-4q2kv.jsonl            12 run records × 29 per-step samples
  raw/
    git-identity.txt            branch, HEAD, stash, diff hashes, counts
    git-status.txt              full `git status --short`
    git-diffstat.txt            full `git diff --stat`
    git-diff.patch              full tracked diff (preservation copy)
    shader-hashes.txt           six production shader artifacts
    receipt-hashes.txt          Phase III/V/VI receipt identities
    source-hashes.txt           sources + trainer executable
    fixture-hashes.txt          both fixtures' config/weights/dataset
    smoke-{scalar,twopass}.*    route + per-shader proof, 2 steps
    numerics-*/                 2-step parity arms (reports, logs, adapters,
                                exact checkpoints with AdamW moments)
    profile-*/                  12-step profiled arms (reports, logs)
    abba-*/                     60 timing processes (reports, logs)
    default/                    unset-variable run
    failclosed/                 no-ICD run
    failclosed-kernel.log       production-path fail-closed test output
    trainer-*/                  native CLI trainer counters (scratch/layer)
```

`summary.json` records only measured values, and the raw reports and logs it was
built from are preserved above. `run_benchmark.py` refuses to emit a number for
a failed process or a missing sample; it never invents one.

## Reproduction

```bash
cd hierarchos-vulkan

# Build the harness binary the measurements use.
cargo build --release --offline \
  --bin hierarchos-vulkan-training-qualification \
  --bin hierarchos-vulkan-transformer-train

# Deterministic GQA fixtures (scratch artifacts, not committed).
python validation/make_wmma_fixture.py --output target/gqa-e2e/fixture-8q2kv \
  --hidden 1024 --layers 2 --vocab 1024 --heads 8 --kv-heads 2 --head-dim 128 \
  --seq-len 64 --batch-size 2 --sequences 32
python validation/make_wmma_fixture.py --output target/gqa-e2e/fixture-4q2kv \
  --hidden 512 --layers 2 --vocab 1024 --heads 4 --kv-heads 2 --head-dim 128 \
  --seq-len 64 --batch-size 2 --sequences 32

# Numerics, attribution, default selector, fail-closed, then ABBA.
python validation/gqa_end_to_end_training_20261124/run_benchmark.py --phase numerics
python validation/gqa_end_to_end_training_20261124/run_benchmark.py --phase profile
python validation/gqa_end_to_end_training_20261124/run_benchmark.py --phase default
python validation/gqa_end_to_end_training_20261124/run_benchmark.py --phase failclosed
python validation/gqa_end_to_end_training_20261124/run_benchmark.py --phase abba

# Kernel-initialization fail-closed contract (debug test binary is sufficient).
HIERARCHOS_WMMA_MODE=off HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=1 \
  target/debug/deps/hierarchos_vulkan-*.exe \
  optional_two_pass_load_failure_keeps_scalar_execution_and_gradients \
  --nocapture --test-threads=1
```

Or simply `bash validation/gqa_end_to_end_training_20261124/reproduce.sh`.

## Limitations

* One machine, one driver, one thermal state. The CI covers run-to-run variation
  on this machine, not other hardware.
* Two of four qualified groupings measured end to end; 8q/4kv and 16q/4kv are
  numerically qualified but have no step-level result.
* One head dimension (128), `q_seq = kv_seq = 64`, `causal = 1`, unmasked rows,
  unit value scale. Nothing here covers `kv_seq < q_seq`, cross-attention,
  `causal = 0`, masked GQA, paged/compressed KV, bf16/fp16, or other head
  dimensions.
* WMMA was off in every arm, deliberately. The attribution is therefore a
  WMMA-off attribution; enabling it would change the dense share.
* The profiled attribution is relative. The profiler costs ~9.5 ms/step of wall
  time and its bracket sum can exceed the unprofiled step, so the GPU totals are
  not interchangeable with the ABBA step times.
* The fail-closed kernel test was run from the current debug test binary
  (built after the last source edit) because a release test build exceeds the
  interactive window; the test exercises the production loader, not a mock.
* 4q/2kv used three paired rounds by design (a cross-check, not a campaign), so
  its interval has df = 2 and is correspondingly wide.
