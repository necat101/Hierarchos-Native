# RDNA3 V4.1 Phase V: suite-order crash fix, retirement attribution, and qualification

Status: **crash fixed and stability-qualified; performance qualification FAILS.**
The full library suite now passes repeatedly under default parallelism, but the
final binary regresses both control workloads and, against the two most recent
baselines, native ViT. Nothing here is promoted to a release PASS.

## Binding

| Field | Value |
|---|---|
| Source commit | `4b659b3d681f6248f01750d32c497dcef32e98c5` |
| Crash-fix commit | `614077150996556bc930f6cfa9c40db3302611fd` |
| Measured binary (preserved) | `target/rdna3-port/final-phase5-bddd6568.exe` |
| Binary SHA-256 | `bddd65686955d13365fbb4cced95eb4591d0eca097d84277fc509046a3bcf4ad` |
| Device | `AMD Radeon Graphics` (1002:15bf, driver 26.10.07.06 LLPC) |
| Fixtures | `target/rdna3-port/gen9-frozen/target/multimodal-gen9` |
| Numeric policy | `max_abs < 2e-7`, `rtol = 0` (unchanged) |

A later rebuild of the same committed bytes produced
`5a9b4d08e619a995343761d29bb9d8e125a6cadb5de00b38ce9a76b35b9efa1e`, so this
incremental debug build is **not byte-reproducible**. The measured artifact is
preserved under the path above and every number in this receipt refers to its
recorded hash.

## Crash: cause, fix, and stability

`0xc0000409` in suite order was resource exhaustion, not the RWKV test named in
the abort. `DeviceInner.pending_clears` held `PendingClear { buffer: GpuBuffer }`
strongly, forming `DeviceInner -> GpuBufferAllocation -> DeviceInner`: every
zeroed intermediate dropped without being bound pinned a whole logical Vulkan
device. The suite eventually exhausted the driver and fail-fast aborted while
entering the next test.

The fix keeps a `Weak<GpuBufferAllocation>`, so an unreachable fill disappears
with its last user, and prunes an unbound fill whose allocation has no other
owner so the pending list stays bounded. Two tests cover the invariants:
`deferred_zero_fill_does_not_pin_its_device` and
`unreachable_pending_clear_is_not_retained`.

Stability on the measured binary (1168-test inventory):

| Run | Result | Time |
|---|---|---:|
| Default parallelism 1 | 1128 passed, 0 failed, 40 ignored | 71.21 s |
| Default parallelism 2 | 1128 passed, 0 failed, 40 ignored | 65.80 s |
| Default parallelism 3 | 1128 passed, 0 failed, 40 ignored | 65.42 s |
| Serial (`--test-threads=1`) | 1128 passed, 0 failed, 40 ignored | 386.20 s |

`rwkv_optimizer::tests` passes 22/22 and the previously reported test passes
five consecutive isolated runs. No crash recurred in any run.

## Retirement attribution: a 5x measurement error corrected

The Phase IV profile divided a run total covering **five benchmark cases** by
**one case's** step count. That is why it reported 243.2 retirements per step
while the same step reported only 151 native submissions — an internal
contradiction. Dividing by case-steps (25 here) and recording host rebuild
separately from the fence/timeline wait gives:

| Call site | Per case-step | Blocking ms/step | Host rebuild | GPU wait |
|---|---:|---:|---:|---:|
| `ComputeBatch::submit_and_restart` | 48.64 | 28.417 | 0.329 | 28.088 |
| in-session `read_many_f32` (`linear_grad.rs:157`) | 7.20 | 8.205 | 0.053 | 8.152 |

Phase IV's 142.107 ms and 39.446 ms correspond to 28.42 ms and 7.89 ms per case
step. Host churn is 1.2% of the retirement cost, so command-buffer or arena
coalescing cannot move step latency; the cost is the GPU completion the next
stage consumes. That is why no submission-coalescing change was adopted, and why
the previously rejected streamed-wait intervals were not retried.

## Performance qualification: FAIL

Four-round ABBA, seven measured updates, two warmups, batch 1, software FMA in
both arms, 10,000-draw paired bootstrap.

| Comparison | ViT 4x4 | ViT 6x4 | llama | gpt2 | t5 |
|---|---:|---:|---:|---:|---:|
| 372407b -> final | **1.4907x** | **1.4569x** | 0.8290x | 0.9156x | 0.8222x |
| 722acd0 -> final | 0.9989x | 0.9805x | 0.8358x | 0.8307x | 0.8393x |
| 61591bc -> final | 0.9695x | 0.9628x | 0.8298x | 0.8501x | 0.8323x |
| c7e096d -> final | 0.9676x | 0.9662x | 0.8271x | 0.8258x | 0.8334x |

Against the 372407b baseline the final binary is a clear native win
(1.46-1.49x, intervals entirely above 1.0). Against both 1d859d7-generation
baselines it is a small but statistically clear native **regression** (about
3-4%) and a **17-18% control regression** that no baseline escapes.

The control regression was chased rather than assumed:

- Restoring the pending-fill pruning rule did not move it (`rejected/pre-prune-b1.json`).
- A temporary probe that removed the `#[cfg(test)]` ATen-width TLS read did not improve it, and was slower on all three controls (`rejected/tls-probe-b1.json`).

Both candidate mechanisms were rejected on measurement. The regression therefore
remains **unexplained**, and it is a release blocker. The only production change
in range is the deferred-zero-fill ownership fix; it is required for stability
and is retained, with the regression documented rather than hidden.

Session ablation is a same-binary, word-exact comparison: 1.5307x (4x4) and
1.4794x (6x4) for the session arm, controls within 1% — the session path is not
the source of the control regression.

## Batch matrix

Native V4.1 rows expand real `batch * seq_len` geometry. Controls pin batch 1;
their requested-batch 2/4 rows are the same single-sample workload and are
reported as `effective_batch = 1`, not as native batching.

| Case | Batch | Median | Samples/s | Scaling | Peak live bytes |
|---|---:|---:|---:|---:|---:|
| ViT 4x4 | 1 | 231.458 ms | 4.320 | 1.000x | 9,094,168 |
| ViT 4x4 | 2 | 234.694 ms | 8.522 | 1.973x | 9,348,088 |
| ViT 4x4 | 4 | 238.379 ms | 16.780 | 3.884x | 9,855,928 |
| ViT 6x4 | 1 | 295.487 ms | 3.384 | 1.000x | 9,143,036 |
| ViT 6x4 | 2 | 293.849 ms | 6.806 | 2.011x | 9,428,516 |
| ViT 6x4 | 4 | 288.674 ms | 13.856 | 4.095x | 9,999,476 |

Outliers are retained (349.43 ms at batch-1 6x4, 350.44 ms at batch-2 4x4,
284.54 ms at batch-4 4x4). Submission count stays 151 for every native batch.

## Correctness and gates

- Full suite: 1128 passed, 0 failed, 40 ignored, three default-parallel runs plus
  one serial run on the measured binary.
- Compressor parity passes with worst absolute drift `1.788139343262e-7` under the
  unchanged strict gate; exact-word accumulator-order tests pass.
- GPT-2 module passes 59 tests and its focused parity cases return the fixture's
  AVX2 value `1.788139343262e-7`.
- `check_rdna3_multimodal_receipts.py` and `check_training_stack_receipt.py` pass;
  prior receipts were not modified.
- The ten performance-gate unit tests pass.
- **`check_rdna3_gpt2_receipts.py` fails closed** (`gate/gpt2-receipt-guard-direct.log`):
  its `receipt-hashes.json` binds the whole of `src/transformer.rs`, which the
  Phase V test-only width pin changed after that qualification. The sealed
  receipt was not edited and no replacement GPT-2 qualification was fabricated.
  The three-test negative-control suite does pass
  (`gate/gpt2-negative-control-tests.log`), but it only exercises the guard's
  acceptance function on synthetic receipts; it does not restore the binding.
- Physical Gen9 execution, the nine-stage Gen9 replay, the native AVX-512 track
  and the external shader-oracle checkout remain unavailable on this host.
- The RDNA3 workflow parses but has never executed: no
  `[self-hosted, Windows, X64, rdna3, vulkan]` runner is registered.

## Evidence map

- `crash/` — the Phase IV failure log that reproduced the abort.
- `stability/` — three default-parallel logs and the serial log.
- `profile/` — raw retirement/submission report and benchmark JSON.
- `batch/` — batch 1/2/4 raw JSON and logs.
- `abba/` — four paired comparisons with every round JSON and log.
- `session/` — session OFF/ON ablation.
- `gate/` — performance-gate unit tests, targeted regressions, receipt guards,
  workflow parse.
- `rejected/` — the measurements that rejected the two regression hypotheses.
- `manifest.json` — SHA-256 binding for every sealed file.
