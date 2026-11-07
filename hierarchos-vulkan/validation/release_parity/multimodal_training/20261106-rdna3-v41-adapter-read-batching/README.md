# Native V4.1 adapter-gradient readback batching: rejected

## Question

Can the 1,008 per-group adapter-gradient reads in a native training step be
collapsed into one device-to-host transfer per step, and does that improve
throughput?

## What was tried

Every adapter pair was kept on the device through the backward, then one
`GpuBuffer::read_many_f32` downloaded the whole set after the compute session
was submitted. Three submission cadences were measured on the same binary:

- no explicit flushes: one large end-of-backward session;
- one session flush per layer;
- session flushes at the old adapter-group boundaries.

All variants kept the frozen gate; the group-flush build passed the 30-test native qualifier.

## Measurements

Resident-arm medians over four ABBA rounds with seven measured updates; the
prior column is the accepted resident-expert receipt (`205.1`/`273.4`).

| Case | Prior streamed / resident, ms | No flush, ms | Per-layer flush, ms | Per-group flush, ms | Reverted build, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `native_vit_4x4` | 240.9 / 205.1 | 319.2 | 210.9 | 209.7 | 208.0 |
| `native_vit_6x4` | 307.7 / 273.4 | 453.5 | 303.8 | 296.6 | 278.1 |

All times are the resident arm except the prior pair. The prior streamed values
were 240.9 ms (4x4) and 307.7 ms (6x4); the batched streamed values were
248.3/340.8 (no flush), 246.0/330.8 (per-layer) and 252.5/337.3 (per-group).

The batching is numerically exact: `pass: true`, observed `max_abs = 0.0`, and
every image and adapter digest matched across all arms. The qualifier is
30/30 with `max_abs_logit = 0.0`.

Read and submission counts show what changed:

- adapter reads: 1,008 -> 28 total (4 per step, one per backward);
- submissions per step, streamed: 241 -> 124 (no flush), 130 (per-layer), 166 (per-group);
- submissions per step, resident: 169 -> 52 (no flush), 58 (per-layer), 94 (per-group).

## Result

Every cadence is slower than the accepted code. The best batched variant
(per-group flush) is 209.8 ms vs 208.0 ms reverted (205.1 ms prior) at 4x4 and
296.6 ms vs 278.1 ms reverted (273.4 ms prior) at 6x4. The reverted build
re-measured at 208.0 ms and 278.1 ms, within 1-2% of the prior receipt, so the
comparison is not thermal drift.

The no-flush variant is the clearest signal: resident collapses to 0.78x/0.75x
of streamed when the per-group reads no longer submit accumulating work. The
reads were acting as pacing points that kept host recording overlapped with
device execution. Removing 1,000 small transfers loses more overlap than it
saves; the readback is a synchronization mechanism, not a bandwidth bottleneck.

## Decision

Rejected and reverted. No batching types, flushes, or call-site changes remain
in the tree, and the accepted resident-expert receipt still describes the
current behavior. The measurement is recorded here so the 1,008-call profile
site is not retried without the overlap argument.

## Artifacts

- `benchmark-no-flush.json`, `benchmark-layer-flush.json`,
  `benchmark-group-flush.json`, `benchmark-reverted.json` - ABBA receipts.
- `native-qualification.json` - 30-test qualifier for the batched code.
- `profile-group-flush.log` - attribution showing 28 adapter reads and the
  192 routing reads.
- `profile-reverted.log` - streamed baseline from the reverted build.
- `receipt-hashes.json` - SHA-256 for every file above.
