# DeepSeek V4.1 forward profile and first submission-cost fix

This is the profile-first step of the Intel Gen9 handoff
([`intel-handoff/DEEPSEEK_V41.md`](intel-handoff/DEEPSEEK_V41.md)): the brief says
to profile before choosing the optimization order, so the order below is measured
on the host machine (AMD Ryzen Z1 Extreme / Radeon 780M, Windows) rather than
assumed. The Gen9 target still needs its own run and its own receipt.

## How to take the profile

```powershell
cd hierarchos-vulkan
cargo build --release --bin hierarchos-vulkan-deepseek-v41-forward-probe

# GPU-time ranking (privileged timestamps, every dispatch timed)
$env:HIERARCHOS_VULKAN_PROFILE_KERNELS = "1"
$env:HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE = "1"
$env:HIERARCHOS_VULKAN_PROFILE_TOP_SHADERS = "4096"
.\target\release\hierarchos-vulkan-deepseek-v41-forward-probe `
  --steps 2 --warmup 1 --prefill 8 --decode 8 --residency resident `
  2> ..\target\v41-gen9\profile-resident.log
python validation/rank_kernel_profile.py ..\target\v41-gen9\profile-resident.log

# Submission structure (every recorded batch, with its dispatch/upload counts)
$env:HIERARCHOS_VULKAN_SUBMIT_TRACE = "1"
```

`--residency resident` keeps every weight on the device so the ranking is compute,
not checkpoint streaming; `--residency default` mirrors the shipping loader.
The probe prints a digest of the last-position logits, so a kernel change can be
shown to leave the forward bit-identical while the profile shows what it bought.

## Measured baseline (before any change)

GPU-time shares over 27 forward calls (3 steps x (1 prefill + 8 decodes),
resident, sample stride 1, 9,568 ms of dispatch brackets in total):

| stage | dispatches | GPU ms | share | us/dispatch |
| --- | --- | --- | --- | --- |
| `v41.moe-packed` (gate, pack, quantize, packed GEMM, SwiGLU, scatter) | 103,311 | 7,275.6 | 76.0 % | 70 |
| `v41.hyper-connection` (pre/split/post) | 6,507 | 1,597.0 | 16.7 % | 245 |
| `v41.other` (BF16 casts) | 57,456 | 330.7 | 3.5 % | 5.8 |
| `v41.attention` (scores, values, indexer, rotary, cache, grouped output) | 8,577 | 211.5 | 2.2 % | 25 |
| `v41.normalization` (RMSNorm) | 4,491 | 106.3 | 1.1 % | 24 |
| `v41.projection` (dense linear) | 2,664 | 40.5 | 0.4 % | 15 |

Hottest single kernels: `moe_gate` 2,898 ms (30.3 %, 2.68 ms per dispatch),
`quantize` 2,489 ms (26.0 %, 66 us), `packed_linear` 1,618 ms (16.9 %, 46 us),
`hc_split` 1,564 ms (16.3 %, 724 us).

But the GPU only owned 9.6 s of dispatch brackets across all 27 calls - about
3.2 s of a 63 s step - so the wall clock was dominated by **host submission**. The graph created a command batch and submitted it with a
fence wait per stage, and every intermediate was a fresh `GpuBuffer::zeros_*`
that uploaded its zeros through a staging copy with its own submit + wait. The
structural trace showed 57,411 submissions for one prefill (8 tokens) plus one
decode, of which **35,783 had no dispatch at all** - they were zero-fill copies.

## Change: zero fills are recorded, not submitted

`GpuBuffer::zeros_f32` / `zeros_u32` no longer stage a zero vector and wait for a
copy. A host-visible allocation is memset through its mapping; a device-local
allocation enqueues a `vkCmdFillBuffer(0)` that the next command buffer records
before the dispatch or transfer that first binds the buffer, between the same
transfer/shader barriers the upload path already uses. Device-to-host readbacks
and full overwrites record or cancel the pending fill, so the observable contents
are unchanged.

| per prefill(8) + decode(1) | before | after |
| --- | --- | --- |
| submissions | 57,411 | 32,018 |
| submissions with no dispatch | 35,783 | 10,390 |

Unprofiled wall clock, same checkpoint, same tokens (medians of repeated runs):

| residency | prefill, 8 tokens | decode, per token |
| --- | --- | --- |
| default (shipping loader) | 5,318 ms -> **3,470 ms** | 503 ms -> **341 ms** |
| resident | 2,431 ms (after only) | 116 ms (after only) |

Only the shipping path has an unprofiled before/after pair; the resident numbers
are the post-change measurement, reported because the profile ranking above was
taken in that mode. The logit digest is unchanged in both modes
(`2d2f8e2d5914a0d2` for prefill 8 / decode 4 through the default loader,
`90e096ebc5bf0659` for the resident 8/8 run), so nothing in the arithmetic
moved.

## Verification after the change

| check | result |
| --- | --- |
| `vulkan::tests::` (core buffer/submission semantics, incl. three new deferred-zero tests) | 36 passed, 0 failed |
| `deepseek_v41::` suite | 181 passed, 0 failed, 24 ignored |
| `deepseek_v41_default_package_scalable_release_strict_all_logits` (`--ignored`) | passed (3,216 logits, zero drift) |
| 10 cross-architecture `kv_cache_matches_full_prefix_logits` tests (GPT-2, Llama, Mistral, Mixtral, Qwen, DeepSeek V4, ...) | 10 passed |

The deferred fill is consumed only by the command buffer that binds the buffer,
and only by the thread that allocated it, so concurrent graphs on one device
cannot steal or reorder each other's fills; a host mapping write or a full
upload cancels the fill instead.

## Change: a layer records into one command buffer

`Backbone::forward_layer` opens a `ComputeSession`: every stage in the layer
records into that session's command buffer and its `submit` becomes a
publication point instead of a queue submission. A host read, or a streamed
weight upload, retires the recorded prefix and keeps recording in the
replacement buffer, so the session is invisible to the stage that asked.

```powershell
# A/B: same binary, same checkpoint, same tokens; only the granularity changes.
$env:HIERARCHOS_VULKAN_LAYER_SESSION = "0"   # recording is the default
.\target\release\hierarchos-vulkan-deepseek-v41-forward-probe `
  --steps 2 --warmup 1 --prefill 8 --decode 8 --residency resident
```

The probe now prints the device's submission count per phase
(`submissions=` on each `v41_probe_phase` and `v41_probe_load` line), which is
the number this change moves; `HIERARCHOS_VULKAN_SUBMIT_TRACE` still prints the
structure of every batch.

| per forward call | session off | session on |
| --- | --- | --- |
| submissions, prefill(8), resident | 16,162 | **252** |
| submissions, decode, resident | 5,632 | **252** |
| prefill median (8 tokens), resident | 2,207-2,216 ms | **457-537 ms** |
| decode median, resident | 106.8-106.9 ms/token | **20.3-20.4 ms/token** |

Both arms produce the frozen digest `90e096ebc5bf0659` for resident 8/8 and
`2d2f8e2d5914a0d2` for the default loader at 8/4, and the default loader keeps
its digest too (prefill 3,352 -> 2,685 ms, decode 305 -> 234 ms/token, the
smaller gain because streamed weights retire the session on every upload).

### Two facts the session had to make explicit

A queue submission was doing two jobs that recording alone does not: it waited,
and it kept the resources it referenced alive. Both had to be restored.

1. **A stage publication point is a barrier.** `BatchHandle::submit` inside a
   session records a compute-to-compute memory barrier when the stage wrote
   anything, because the per-dispatch elision compares exact buffer keys while
two stages can alias one allocation through different views. Without it the
layer is not ordered the way its per-stage submissions were.
2. **A command buffer must hold its bound buffers until its wait completes.**
   `retain_bound_buffers` keeps every bound allocation on the batch, and the
   submission hands those extra references to the device's timeline retirement
   arena (`retire_bound_buffers_on_timeline`), so they are dropped only once the
   timeline value is observed. Previously the batch was dropped at submission,
   so an allocation released by the host after recording could be freed and
   handed to a later allocation while an unexecuted dispatch still read it. On
   the joint fixture that showed up as a nondeterministic change in every
   layer's output; with the lifetime fixed, layer digests match bit-for-bit.
   Retaining the whole batch instead looks equivalent but is not: a streamed
   upload needs its arena back as soon as the submission completes, and holding
   the batch past that point broke the streamed-expert logits.

`HIERARCHOS_VULKAN_LAYER_DIGEST=1` prints an FNV-1a digest of each named stage
(`HIERARCHOS_VULKAN_DIGEST_ONLY=<stage>` narrows it to one), which is how the
lifetime bug was localized and how a kernel change proves it left every layer
identical. Reading a buffer splits the session, so this is a diagnostic mode.

### Status

Recording is the default. With it on, `deepseek_v41::` reports 181 passed, 0
failed, 24 ignored; `vulkan::` 39 passed; the 10 cross-architecture
`kv_cache_matches_full_prefix_logits` tests pass; and the strict ignored release
gate (`deepseek_v41_default_package_scalable_release_strict_all_logits`, 3,216
logits over 40 layers and 384 experts) passes with zero drift. The probe's
frozen digests are identical in both arms. `HIERARCHOS_VULKAN_LAYER_SESSION=0`
remains the A/B arm and the escape hatch.

## What the profile says to do next

1. **The per-layer MoE `counts` readback.** Each layer still retires its session
   once to read the routing counts on the host, which is why a call costs 252
   submissions rather than one per layer (40). A GPU-side active-expert worklist
   removes that read; the session already proves the surrounding stages can
   share one command buffer.
2. **`moe_gate` (30 % of GPU time, 2.68 ms/dispatch).** One invocation per row
   computes all 384 expert dot products serially and then runs the pinned CPU
   top-k staging. Splitting the score stage across experts (one invocation per
   row/expert) keeps each dot's FMA order, hence bit-identical scores, and leaves
   the selection staging untouched.
3. **`hc_split` (16 %, 724 us/dispatch) and `quantize`/`packed_linear`
   (43 %, 46-66 us/dispatch).** Both run tiny grids per dispatch, so their cost
   is launch-bound; they want wider tiles or fusion, and each has an ordered
   reduction that the strict gate will check.

None of this weakens the contract: every step is gated at `max_abs < 2e-7`,
`rtol = 0`, against the frozen CPU fixtures.
