# RDNA3 V4.1 Phase IV: retirement accounting, native batch scaling, and release-gate hardening

Status: **measured, but not release-qualified**. The native comparisons are
statistically inconclusive. This receipt does not promote them to PASS and does
not claim a speedup for the final source revision.

## Binding

| Field | Value |
|---|---|
| Source commit | `1d859d7b428567a2b8062a78007be6860cc568e8` |
| Measurement binary | `target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe` |
| Binary SHA-256 | `f1f597c7a87d73676e7f418eb27bc6435d0cee56f870280223a857e0286ab265` |
| Prior optimized baseline | `722acd09626c592ead883b8c7aa652dd0d6ceeac`, SHA-256 `7afa0d985a5304f9383b47d4deaf6119c982d50ab0ccee75db3715e60ab41642` |
| Immediate predecessor | `61591bc6b5187560e08f65b0158db2e711459815`, SHA-256 `9cc5a6504a2e74eb8cfbdce2b00e2d9419598ce3d364c6b45de4e780438a6c32` |
| Device | `AMD Radeon Graphics` (RDNA3 host; exact Vulkan driver record in `toolchain.txt`) |
| Fixtures | `target/rdna3-port/gen9-frozen/target/multimodal-gen9` |
| Numeric policy | `max_abs < 2e-7`, `rtol = 0` (unchanged) |

The measured binary was built before the final rustfmt-only whitespace change to
the test instrumentation module; the production execution path is unchanged.
This is recorded as a provenance limitation rather than hidden.

## Retirement accounting

The instrumentation now records the actual `ComputeBatch::submit_and_restart`
publication point, not only `retire_active_compute_session`. It records caller,
event count, dispatches, and blocking wall time only when
`HIERARCHOS_VULKAN_SUBMISSION_REPORT` is set. It is compiled out of production
builds.

Fresh three-step profile, batch 1, two warmups:

| Call site | Events/step | Dispatches | Blocking ms/step |
|---|---:|---:|---:|
| `src/vulkan.rs:5502` (`submit_and_restart`) | 243.20 | 17,534 | 142.107 |
| `src/deepseek_v41/linear_grad.rs:157` (in-session readback) | 36.00 | 5,090 | 39.446 |
| **Total** | **279.20** | **22,624** | **181.553** |

A retirement is a host publication point, not a unique native queue submission.
The observed benchmark step still reports 151 native submissions. The remaining
cost is dominated by the blocking streamed-expert publication path, not by the
36 direct gathered readbacks.

## Native batch scaling

Each row is the same final binary and frozen fixture with full
`batch * seq_len` geometry. Additional rows are text-only (`TEXT=-1`) samples;
the single released image span remains in row zero. Logits length grows from
128/144 at batch 1 to 512/576 at batch 4, proving that the extra rows execute;
this is not a loop of batch-1 calls.

| Case | Batch | Median latency | Throughput | Throughput scaling | Submissions | Peak live bytes |
|---|---:|---:|---:|---:|---:|---:|
| Native ViT 4x4 | 1 | 228.702 ms | 4.373 samples/s | 1.000x | 151 | 9,094,168 |
| Native ViT 4x4 | 2 | 227.391 ms | 8.795 samples/s | 2.012x | 151 | 9,348,088 |
| Native ViT 4x4 | 4 | 231.202 ms | 17.301 samples/s | 3.957x | 151 | 9,855,928 |
| Native ViT 6x4 | 1 | 293.034 ms | 3.413 samples/s | 1.000x | 151 | 9,143,036 |
| Native ViT 6x4 | 2 | 294.812 ms | 6.784 samples/s | 1.988x | 151 | 9,428,516 |
| Native ViT 6x4 | 4 | 291.992 ms | 13.699 samples/s | 4.014x | 151 | 9,999,476 |

The near-flat latency is consistent with fixed graph/upload cost dominating these
small fixtures. Memory grows modestly and monotonically. Batch 4 ViT 6x4 had one
337.331 ms outlier among seven samples (median 291.992 ms); raw samples are
retained. Controls were measured only at batch 1; the benchmark harness does not
claim multi-batch support for Llama, GPT-2, or T5.

## Controlled comparisons

All comparisons are six-round ABBA, seven measured optimizer updates, two
warmups. Older binaries predate the batch field, so batch-1 provenance is
explicitly marked; no batch-2/4 comparison is inferred from them.

| Comparison | Case | Paired median | 95% CI | Outcome |
|---|---|---:|---:|---|
| 722acd0 -> final | Native ViT 4x4 | 1.0381x | [0.9404, 1.0837] | inconclusive |
| 722acd0 -> final | Native ViT 6x4 | 1.0163x | [0.9530, 1.0965] | inconclusive |
| 61591bc -> final | Native ViT 4x4 | 1.0068x | [0.9914, 1.0860] | inconclusive |
| 61591bc -> final | Native ViT 6x4 | 1.0037x | [0.9891, 1.0763] | inconclusive |

All control intervals overlap the practical-equivalence margin or are
inconclusive; no control regression is established. Both comparisons preserve
cross-arm final words exactly. The gate state for each comparison is
`INCONCLUSIVE`, never `PASS`.

Historical `372407b` -> final and session OFF/ON were **not rerun** in this pass.
No preserved historical binary was available on the host, and a fresh build
would be a rebuilt-source comparison, not the original artifact. They remain
verification gaps.

## Rejected optimization

A bounded streamed-expert wait interval was implemented and measured on the same
binary. With interval 2, Native ViT 4x4 was 0.9966x [0.9856, 1.0108]; with
interval 4, 1.0201x [0.9969, 1.0465]. Neither arm established a gain, and the
wider intervals increase live staging. The code path was removed. Raw evidence is
in `rejected/streamed-wait-interval-2/` and `rejected/streamed-wait-interval-4/`.
The earlier whole-layer adapter-span deferral remains rejected: 0.9233x at 4x4
and 0.9057x at 6x4.

## Performance gate

`rdna3_performance_gate.py` now emits `PASS`, `REGRESSION`, `INCONCLUSIVE`, or
`INVALID`. Development mode exits zero for INCONCLUSIVE but reports it as
INCONCLUSIVE, not PASS. Release mode requires PASS for every workload at batches
1, 2, and 4, validates artifact hashes and fixture identity, and rejects missing
cases, duplicate batches, malformed samples, mismatched binaries/configuration,
and corrupted JSON. Equivalence bounds are derived from measured arm dispersion,
bounded by the 2% product cap; statistical non-significance is not treated as
equivalence.

Ten synthetic negative tests pass. The workflow uses isolated candidate/baseline
build directories, records source/toolchain/binary hashes, runs batches 1/2/4,
and uses the strict state gate. It has **not executed in GitHub Actions** because
no self-hosted `[self-hosted, Windows, X64, rdna3, vulkan]` runner is registered.
Workflow syntax parses successfully with PyYAML.

## Correctness and limitations

- The full native `cargo test --locked --lib` suite was started with
  `--test-threads=1`. It passed 411 tests and then aborted in suite order with
  Windows `STATUS_STACK_BUFFER_OVERRUN` (`0xc0000409`) while entering
  `rwkv_optimizer::tests::loss_unscale_and_adamw_share_one_command_stream`.
  That test passes in isolation and the complete `rwkv_optimizer` module passes
  (22/22), so this is recorded as an **incomplete/failed suite run**, not a pass.
  The raw log is `gate/full-lib-tests.log`.
- `batched_read_reuses_the_open_session_submission`, the gathered adapter-span
  word/order test, strict compressor prefill/decode parity (worst absolute drift
  `1.788139343262e-7`), and compressor tape norm-gain snapshots all pass against
  the measured binary.
- `check_rdna3_multimodal_receipts.py` passes, preserving the prior sealed
  receipts and the unchanged `2e-7`/`rtol=0` policy.
- `check_training_stack_receipt.py` passes against the committed 12-step ladder
  receipt; this verifies the recorded receipt, not a fresh nine-stage replay.
- Fresh nine-stage Gen9 replay, native AVX-512 execution, shader-oracle parity,
  and physical Intel Gen9 execution were not run. The Gen9 host is unavailable
  and the shader-oracle checkout is absent.
- No physical Gen9 result is inferred from RDNA3 replay.
- A dedicated RDNA3 CI runner is still required before the workflow can qualify a
  release.
- The final source formatting change is test-only; see the binding note above.

## Evidence map

- `retirement-profile/` — raw JSON and log from the three-step retirement run.
- `batch/b1`, `batch/b2`, `batch/b4` — raw native benchmark JSON and logs.
- `abba/final-vs-722acd0`, `abba/final-vs-61591bc` — raw round JSON, benchmark
  receipts, gate decisions, and logs.
- `rejected/` — raw evidence for the removed wait-interval experiments.
- `gate/` — gate unit tests, prior-receipt guard, and training-stack guard logs.
- `batch/summary.json`, `abba/summary.json`, `retirement-profile/summary.json` —
  machine-readable derived summaries.
- `manifest.json` — SHA-256 binding for every sealed file.
