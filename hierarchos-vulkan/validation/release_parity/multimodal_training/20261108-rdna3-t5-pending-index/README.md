# RDNA3 T5 pending-clear index: 2026-11-08

**Status: PASS for the same-profile T5 correction; historical elimination is
NOT established. The cross-profile `c7e096d` comparison is invalid.**

Source under test is `2671de0885813dadfabbf8bc81ec6b4c6a71bdb5`. No tolerance,
fixture, historical receipt or frozen reference was changed.

## Root cause

The deferred zero-fill list in `hierarchos-vulkan/src/vulkan.rs` was a `Vec` of
weak entries. `drain_pending_clears` walked the entire vector on every batch and
searched the batch's bound buffers for each entry. T5 keeps about 470 live
deferred fills and performs 445 drains per training step:

| Counter per T5 step | Before | After |
| --- | ---: | ---: |
| Pending entries | 470 | 480 |
| Drains | 445 | 445 |
| Entry inspections | **209,312** | **1,613** |
| Clears recorded | 3 | 3 |
| Suballocated buffers allocated / freed | 5 / 5 | 5 / 5 |
| Buffer-allocation reuse | 4 | 4 |
| Submissions | 7 | 7 |

The 209k inspections are the measured, removable cost. Allocation/free churn is
only five buffers per step, so the residual was not dominated by destruction.
The counter deltas are in `t5-telemetry.json` and `t5-candidate-telemetry.json`.

## Correction

`PendingClears::entries` is now keyed by allocation address. A drain probes only
the buffers the command batch actually binds, recording the same deferred fills
and preserving the allocating-thread ownership rule. Entries still hold
`Weak<GpuBufferAllocation>`, never a strong buffer. Unreachable entries are
pruned when the index reaches a bounded threshold, so the device-lifetime cycle
cannot return. The new regression test creates 64 live unbound fills, binds only
a two-buffer copy, and asserts that fewer than 64 entries are probed.

## Performance

Same-profile six-round ABBA, batch 1, two warmups, seven measured updates,
software-FMA configuration. Baseline is the same instrumented source before the
index; candidate is the indexed source:

| Case | Baseline median | Candidate median | Speedup | 95% interval | Result |
| --- | ---: | ---: | ---: | --- | --- |
| native ViT 4x4 | 258.148 ms | 122.362 ms | 2.1097 | 2.0885..2.1286 | faster |
| native ViT 6x4 | 269.541 ms | 125.542 ms | 2.1470 | 2.1000..2.1888 | faster |
| Llama | 13.284 ms | 6.917 ms | 1.9204 | 1.7705..2.0239 | faster |
| GPT-2 | 9.687 ms | 3.858 ms | 2.5110 | 2.3891..2.5894 | faster |
| **T5** | 11.667 ms | 10.974 ms | **1.0631** | **1.0121..1.1162** | **faster** |

All T5 signatures were word-identical inside and across the two arms under the
explicit cross-source allowance. Raw evidence: `abba-index/`,
`t5-abba-summary.json`.

Session same-binary OFF/ON:

| Case | OFF median | ON median | ON/OFF | 95% interval | Result |
| --- | ---: | ---: | ---: | --- | --- |
| native ViT 4x4 | 222.689 ms | 116.539 ms | 1.9109 | 1.8786..1.9484 | faster |
| native ViT 6x4 | 224.762 ms | 118.443 ms | 1.8976 | 1.8930..1.9193 | faster |
| Llama | 6.255 ms | 6.288 ms | 0.9946 | 0.9522..1.0597 | inconclusive |
| GPT-2 | 4.083 ms | 4.016 ms | 1.0167 | 0.9817..1.0846 | inconclusive |
| **T5** | 10.498 ms | 10.442 ms | **1.0053** | 0.9444..1.0246 | **inconclusive** |

The T5 session effect is not separated from noise; it is not claimed as a
regression or an improvement.

## GPU timestamp check

`HIERARCHOS_VULKAN_PROFILE_KERNELS=1` succeeded on the adapter. Timestamp period
is 10.0 ns, the profile used 64-bit query results and the existing
`timestamp_delta` wraparound test passes (`timestamp_delta_handles_wrapped_queue_timestamps`).
T5 profile totals, seven measured steps:

| Metric | Before | After |
| --- | ---: | ---: |
| Profile samples | 7 | 7 |
| Timestamped dispatches per step | 389 | 389 |
| GPU span per step | 11.453910 ms | 10.812430 ms |
| Full profiled step median | 22.422 ms | 21.269 ms |

Profiling adds query-pool results and synchronization, so these totals are not
used as the paired speedup. They show GPU work is essentially unchanged while
the host-side entry inspections disappear.

## Correctness

- Targeted `vulkan::tests::pending_clear_drain_probes_only_bound_buffers`: PASS.
- Existing pending-clear lifetime/readback tests: PASS.
- `vulkan::tests::` all 43: PASS.
- T5-prefixed tests 6/6: PASS.
- mT5-prefixed tests 4/4: PASS.
- Full serial library suite: **INCOMPLETE at sealing**. The attached
  `full-suite.log` reached more than 200 tests with no failure shown, but the
  suite had not reached its final result when this receipt was written. It is
  not reported as a pass.

The new test and the retained `deferred_zero_fill_does_not_pin_its_device` test
cover both sides of the lifetime contract: the weak entry does not retain the
device, and an unbound live fill is not enumerated by an unrelated bind.

## Historical comparison limitation

The stored `c7e096d.exe` is not a valid profile match for this release binary.
It is 308,437,504 bytes with a 51,456,036-byte `.text` section and runs the full
benchmark in 5.92 s; the indexed release binary is 254,914,048 bytes with a
35,780,773-byte `.text` section and runs it in 2.74 s. Its apparent 2.50x T5
ratio is therefore **not** a performance claim and is retained under
`invalid-c7e-profile/` only for traceability. The earlier 0.9453x historical
figure was measured on those older binaries; this pass does not claim that the
historical gate is now green. The valid claim is the same-profile 1.0631x
correction.

## Remaining limitations

- The exact final-source release test binary is being built separately; the
  candidate binary hash in this receipt is the instrumented pre-test-only
  binary. The production Rust source used for both is identical.
- Historical cross-profile elimination is not established.
- Gen9 hardware and an external shader oracle were unavailable.
- Debug and release test binaries are not byte-reproducible.
