# RDNA3 T5 regression bisection: 2026-11-08

**Status: REGRESSION. No source fix was landed.** The correct weak-reference
device-lifetime fix is preserved. The T5 regression is localized to its first bad
revision and two candidate mechanisms were measured and rejected; a residual
slowdown remains a release blocker.

Source under test is `89af6932ddb6e7a554d5bdce600906d3e45c395e`, whose working
tree was clean before this evidence-only commit. No tolerance, fixture, historical
receipt or source file was changed by this pass.

## Boundary

| Role | Revision | T5 vs prior |
| --- | --- | ---: |
| Last good | `c7e096d` | 0.9978 vs `372407b` (from the prior receipt) |
| First bad | `6140771` | **0.8562 [0.8427, 0.8709]** vs `c7e096d` |
| Scan-optimized | `42958d3` | **0.9554 [0.9101, 0.9679]** vs `c7e096d` |
| Current | `89af693` | **0.9453 [0.9320, 0.9538]** vs `372407b` (from the prior receipt) |

`6140771` is the first bad revision. Its only production change is that
`DeviceInner.pending_clears` stopped holding a strong `GpuBuffer` and started
holding a `Weak<GpuBufferAllocation>`; it also added a test-only ATen-width TLS
override. At `6140771` every workload is slower against `c7e096d`, not just T5:

| Case | Speedup | 95% interval | Conclusion |
| --- | ---: | --- | --- |
| native ViT 4x4 | 0.9701 | 0.9622..0.9770 | slower |
| native ViT 6x4 | 0.9657 | 0.9592..0.9753 | slower |
| Llama | 0.9185 | 0.8895..0.9282 | slower |
| GPT-2 | 0.9097 | 0.8870..0.9237 | slower |
| T5 | **0.8562** | **0.8427..0.8709** | **slower** |

The later `42958d3` scan change replaces the per-entry `Weak::upgrade()` with a
strong-count load and pointer comparison. It recovers most of the broad loss but
not all of it:

| Case | Speedup | 95% interval | Conclusion |
| --- | ---: | --- | --- |
| native ViT 4x4 | 0.9905 | 0.9861..0.9922 | slower |
| native ViT 6x4 | 0.9781 | 0.9432..0.9856 | slower |
| Llama | 0.9419 | 0.9219..0.9885 | slower |
| GPT-2 | 0.9584 | 0.9014..1.0032 | inconclusive |
| T5 | **0.9554** | **0.9101..0.9679** | **slower** |

T5 uses **seven submissions in every revision**. The regression is not extra
queue submissions.

## Rejected experiment 1: zeroed-allocation recycler

The previously reverted device-local recycler was re-applied on `89af693` and
qualified with two new tests (owner-guarded reuse, exact-size-only reuse) plus
the existing poison/reuse test. All three passed (2/2, 3/3, 1/1). Its six-round
result against `372407b` was:

| Case | Speedup | 95% interval | Conclusion |
| --- | ---: | --- | --- |
| native ViT 4x4 | 1.4212 | 1.4000..1.4523 | faster |
| native ViT 6x4 | 1.2786 | 1.2591..1.2946 | faster |
| Llama | 0.9724 | 0.9570..0.9955 | slower |
| GPT-2 | 1.0899 | 1.0380..1.1121 | faster |
| T5 | **0.9468** | **0.9336..0.9740** | **slower** |

T5 did not recover, Llama became a significant regression, and the native ViT
gains fell from ~1.50x/1.45x to 1.42x/1.28x. The recycler was reverted and is
**not** part of the current tree. The exact patch and every round are retained
under `rejected/`.

## Rejected experiment 2: remove the test-only ATen-width TLS read

`6140771` also added a thread-local check at the top of `aten_vector_width()`,
compiled into the test binary used by this benchmark. Removing only that check
from the `42958d3` tree produced:

| Case | Speedup | 95% interval | Conclusion |
| --- | ---: | --- | --- |
| native ViT 4x4 | 0.9898 | 0.9784..1.0350 | inconclusive |
| native ViT 6x4 | 0.9942 | 0.9873..1.0310 | inconclusive |
| Llama | 1.0052 | 0.9428..1.0248 | inconclusive |
| GPT-2 | 0.9680 | 0.9191..1.0082 | inconclusive |
| T5 | **0.9674** | **0.9225..1.0136** | **inconclusive** |

This recovers T5 from `0.9554` to `0.9674` and Llama to parity, so the TLS read is
a real contributor. It is not sufficient: the T5 interval still crosses 1.0, and
the focused GPT-2 width test depends on the override. No source change was
landed.

## Profile

Direct runs with `HIERARCHOS_VULKAN_SUBMISSION_REPORT=1` on `c7e096d` and
`42958d3`, seven measured steps per case:

| Metric | `c7e096d` | `42958d3` |
| --- | ---: | ---: |
| T5 median | 30.23 ms | 30.96 ms |
| T5 submissions | 7 | 7 |
| retirement host rebuild | not reported | 0.525 ms/step |
| blocking retirement wait | not reported | 29.621 ms/step |
| retirement total | 1304.4 ms | 1356.6 ms |

Host command-buffer churn is ~0.5 ms of a ~31 ms T5 step. The blocking wait is
GPU completion the next stage consumes, so the residual is a GPU/lifetime effect,
not removable host rebuild cost. The full logs and per-case JSON are in
`profile/`.

## Interpretation and safe path

The regression entered with the weak pending-clear lifetime change. The scan
cost was largely removed at `42958d3`; the remaining T5 gap is not allocation
reuse (recycler rejected) and is not fully the ATen-width TLS read (partial
recovery only). The remaining candidate is the changed lifetime of unbound
zeroed intermediates: a weak entry is pruned and the allocation destroyed
immediately, altering when the driver can retire/free memory relative to the
next T5 stage. The next safe step is instrumented allocation-destruction timing
and GPU timestamp queries around the T5 encoder/decoder boundary, not another
pool or a tolerance change.

No release PASS is claimed. The current tree remains the correct,
lifetime-safe implementation; the T5 gate remains red.

## Limits

- The final ~3% T5 residual is not fully attributed.
- Debug test binaries are not byte-reproducible; every binary hash and raw round
  is recorded, and the two-round screens are marked exploratory only.
- No physical Gen9 hardware or external shader oracle was available; no Gen9
  claim is made.
