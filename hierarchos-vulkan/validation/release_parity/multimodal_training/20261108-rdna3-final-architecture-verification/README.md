# RDNA3 final architecture verification — 2026-11-08

This is a new receipt. It does not replace or rewrite any earlier qualification,
performance, or hardware receipt.

## Verdict

The complete nine-stage Gen9 multimodal producer exited **0** in one fresh run:
**9/9 stages PASS**, **507 executed cases/tests**, and a worst non-native logit
drift of **`1.7881393432617188e-07`**, strictly below the unchanged `2e-7` absolute gate
with `rtol=0`. The native V4.1 stage reports `max_abs_logit = 0.0`.

The complete Rust library suite passed **1,139 / 0 failed / 40 ignored** in one
default-parallel run (65.37 s), and the GPT-2 primitive group passed **9 / 0
failed / 1 ignored**. The batch-1 matched-profile comparison was positive on all
five canonical workloads. Session OFF/ON was positive for both ViT workloads and
**inconclusive** for Llama, GPT-2 and T5; no textual session speedup is claimed.

## Source and executables

- Tested source: `3f0aaf541a5ed26b826515ca723bf25057dcea16`
- Parent: `b173653441b321e26f66dd9293c05947d817a39b`
- Branch at capture: `phase6-fix`; source tree clean
- Production numerical sources changed: **no**
- Frozen manifest: `a9e519008e712105ceb1b135227398785617db3eb03c65e7518bb7f225966636` (2,748 files)
- Debug harness: `17aa6e998ea5763e083429bed49757ac9fa86167918908d4563bdbdd119a2355`
- Release candidate: `11a83692282bf3e3d4122d4b7490c85fc3c29708837ce7578a12133460b9cf7d`
- Historical baseline: `981e1576415c6fc0a4e54979a8a021a462a9d97a70fb3514ca2e9e7bc88bfd51`

Device: **AMD Radeon Graphics** (RDNA3, `1002:15bf`), driver
`32.0.31007.6002`, Vulkan instance `1.4.341`, device API `1.4.344`,
AMD proprietary driver `26.10.07.06 (LLPC)`. OS: Windows 11 Enterprise
10.0.26300, 64-bit; Rust 1.93.0 / LLVM 21.1.8.

## Nine-stage producer

The aggregate ran all nine stages in one uninterrupted process. Each stage's
canonical `qualification.json` is retained under `aggregate/<stage>/`; the
aggregate manifest is `aggregate/qualification.json`, and the raw producer
console is `aggregate/console.log`.

| Stage | Cases/tests | Worst logit max_abs | Status |
| --- | ---: | ---: | --- |
| shared77 | 77 | 1.4901161193847656e-07 | PASS |
| seq2seq | 6 | 8.9406967163085938e-08 | PASS |
| gemma4-legacy | 1 | 1.4156103134155273e-07 | PASS |
| gemma4-rgb | 1 | 1.3783574104309082e-07 | PASS |
| gemma3 | 4 | 1.7881393432617188e-07 | PASS |
| llava | 4 | 8.9406967163085938e-08 | PASS |
| text166 | 166 | 1.7881393432617188e-07 | PASS |
| deepseek215 | 218 | n/a (exact test suite) | PASS |
| native-v41 | 30 | 0 | PASS |

The text stage has 166 records with `max_abs` per record; its worst record is
`1.7881393432617188e-07`. DeepSeek is a pass/fail exact test inventory (218 verified,
26 ignored, 28 shards), not a logit-metric stage. Native V4.1 completed all 30
tests with `0.0` logit error. The aggregate retains `absolute_limit = 2e-07`,
`rtol = 0`, and `comparison = strict_less_than`.

## Supplementary numerical checks

| Check | Result | Worst logit max_abs | Evidence |
| --- | --- | ---: | --- |
| GPT-2 frozen shapes | 16/16 | `5.9604644775390625e-08` | `numerical/gpt2-shapes-qualification.json` |
| Kimi canonical | 3/3 | `3.166496753692627e-08` | `numerical/kimi-qualification.json` |
| DeepSeek V4.1 shared-image bridge | 2/2, 10 logit checks | `0` | `numerical/v41-bridge-qualification.json` |

## Batch-1 paired performance

Protocol: six paired ABBA rounds, two warmups, seven measured updates, exact
historical baseline `372407b.exe` versus the fresh `3f0aaf5` release harness.
`updates/s` is `1000 / median_ms`. All five workload verdicts are
`candidate_faster`; the benchmark record has no gate failures.

| Workload | Baseline ms/update | Current ms/update | Baseline updates/s | Current updates/s | Speedup | 95% CI | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| native_vit_4x4 | 267.57595 | 132.91910 | 3.737 | 7.523 | **2.0131×** | 1.9921–2.0330 | candidate faster |
| native_vit_6x4 | 268.00080 | 135.37470 | 3.731 | 7.387 | **1.9797×** | 1.9457–2.0174 | candidate faster |
| llama | 8.19295 | 7.20875 | 122.056 | 138.720 | **1.1365×** | 1.0789–1.2449 | candidate faster |
| gpt2 | 7.55660 | 4.70255 | 132.335 | 212.651 | **1.6069×** | 1.4460–1.8597 | candidate faster |
| t5 | 14.82575 | 11.52065 | 67.450 | 86.801 | **1.2869×** | 1.1422–1.3827 | candidate faster |

The complete per-round records, logs, and `hashes.json` are under `benchmarks/b1/`.

## Session OFF/ON at batches 1, 2 and 4

These runs use the **same current binary** for both arms. OFF sets
`HIERARCHOS_VULKAN_LAYER_SESSION=0`; ON is the default. This is not a
historical-baseline comparison at batch 2/4, because `372407b` does not support
a valid baseline at those batches. A CI containing 1.0 remains inconclusive even
when the point estimate is above or below 1.0.

| Batch | Workload | OFF ms/update | ON ms/update | ON/OFF speedup | 95% CI | Verdict |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | native_vit_4x4 | 259.91610 | 134.01920 | 1.9394 | 1.9119–1.9803 | candidate_faster |
| 1 | native_vit_6x4 | 264.14200 | 135.68380 | 1.9467 | 1.8661–1.9799 | candidate_faster |
| 1 | llama | 7.06280 | 7.28175 | 0.9699 | 0.9111–1.0656 | inconclusive |
| 1 | gpt2 | 5.10365 | 5.09315 | 1.0021 | 0.9715–1.0399 | inconclusive |
| 1 | t5 | 11.98305 | 11.78570 | 1.0167 | 0.9444–1.0666 | inconclusive |
| 2 | native_vit_4x4 | 264.32565 | 134.65840 | 1.9629 | 1.9303–1.9861 | candidate_faster |
| 2 | native_vit_6x4 | 267.28475 | 136.72415 | 1.9549 | 1.9270–1.9797 | candidate_faster |
| 2 | llama | 6.75460 | 7.02160 | 0.9620 | 0.9084–1.0439 | inconclusive |
| 2 | gpt2 | 4.97350 | 4.79715 | 1.0368 | 0.9862–1.0652 | inconclusive |
| 2 | t5 | 11.77430 | 11.78765 | 0.9989 | 0.9465–1.0622 | inconclusive |
| 4 | native_vit_4x4 | 264.14420 | 136.87480 | 1.9298 | 1.9019–1.9681 | candidate_faster |
| 4 | native_vit_6x4 | 266.90970 | 138.26425 | 1.9304 | 1.9046–1.9438 | candidate_faster |
| 4 | llama | 7.00060 | 6.95605 | 1.0064 | 0.9159–1.1123 | inconclusive |
| 4 | gpt2 | 4.89615 | 4.94660 | 0.9898 | 0.9456–1.0410 | inconclusive |
| 4 | t5 | 11.63855 | 11.92655 | 0.9759 | 0.8893–1.0678 | inconclusive |

The raw records and hashes are under `benchmarks/session-b1/`,
`benchmarks/session-b2/`, and `benchmarks/session-b4/`.

## Rust checks

- Full library suite: **1,139 passed, 0 failed, 40 ignored**; 1,179 discovered;
  65.37 s; exit 0 (`rust/full-suite.log`).
- GPT-2 primitives: **9 passed, 0 failed, 1 ignored**; 1,169 filtered out;
  2.23 s; exit 0 (`rust/gpt2-primitives.log`).

## Failed setup attempt

The first GPT-2 shapes invocation failed before executing any case because the
driver pre-created its output directory while the canonical runner requires a
new path. The failure is preserved as
`attempt1/gpt2-shapes.precreate-failure.log` and `.exit` (exit 1). The corrected
run created its own output directory and passed 16/16. No output from the failed
attempt is included in the pass counts.

## Limitations

- Session OFF/ON is positive for both native ViT workloads but inconclusive for Llama, GPT-2 and T5 at batches 1, 2 and 4; no textual session speedup is promoted.
- The historical 372407b binary does not support a valid batch-2/4 baseline comparison; batch-2/4 measurements are same-binary session comparisons only.
- No cross-vendor or physical Intel Gen9 execution was performed in this pass; those earlier receipts retain their own device scope.
- The all-bins debug build failed in rustc with STATUS_STACK_BUFFER_OVERRUN; only the five probes, debug harness and release harness were built. This is recorded as an environment/resource limitation, not a source PASS.
- Torch and Transformers are unavailable on this host; the frozen GPT-2 runner was used and its prepare phase was not rerun.
- RDNA3 hardware CI has no registered self-hosted runner; this is a local source-bound qualification, not a hardware-CI PASS.
- Timings are fixture-scale matched-profile measurements, not production-model throughput claims.
- The first GPT-2 shapes attempt failed before execution because its output directory was pre-created; that log is retained under attempt1/ and no result from it is used.

## Reproduction

See `commands.md` for the exact commands and environment variables. Verify the
sealed directory with:

```bash
python verify.py
```

The verifier checks every file in `SHA256SUMS`, the source and binary bindings,
the numerical aggregate and stage counts, the benchmark verdicts, the session
outcomes, the Rust counts, and the preserved setup-failure record.
