# RDNA3 V4.1 in-session gathered readback receipt

Source commit: `e8d892418385bc99702ff77209c345abf094d312` (parent
`0cd37de85eda2cc084fe3be592ee3e81976a6868`).

This receipt binds the source change that records gathered `read_many_f32`
copies into the matching open recording session and retires that session once,
instead of retiring first and submitting a separate readback batch. It also
binds the regression test and the dedicated RDNA3 CI gate/workflow.

## Tested binary

- Candidate: `optimized-session-read-9cc5.exe`
  SHA-256 `9cc5a6504a2e74eb8cfbdce2b00e2d9419598ce3d364c6b45de4e780438a6c32`.
- Prior optimized source: `722acd09626c592ead883b8c7aa652dd0d6ceeac`, rebuilt
  into `prior-722acd0-2c73.exe`
  SHA-256 `7afa0d985a5304f9383b47d4deaf6119c982d50ab0ccee75db3715e60ab41642`.
  The earlier sealed `8ec108f1...` binary was overwritten by a rebuild and was
  not available; this prior binary was rebuilt from the exact prior source in a
  separate worktree. The binary hash therefore differs from the earlier
  receipt's binary hash even though the source commit is the same.

The benchmark is the frozen five-case batch-1 training test, two warmup updates
and seven measured updates per case, six ABBA rounds on the AMD Radeon Graphics
adapter. Both arms use `HIERARCHOS_ATEN_VECTOR_WIDTH=16` and software FMA.

## Performance result

| Case | Role | Prior median | Final median | Paired median speedup | 95% CI | Submission count |
|---|---|---:|---:|---:|---:|---:|
| native_vit_4x4 | native | 224.3813 ms | 218.2482 ms | 1.0280x | [0.9771, 1.0378] | 187 → 151 |
| native_vit_6x4 | native | 283.6814 ms | 276.8303 ms | 1.0305x | [1.0160, 1.0450] | 187 → 151 |
| llama | control | 12.8563 ms | 12.7841 ms | 1.0035x | [0.9863, 1.1826] | 5 → 5 |
| gpt2 | control | 11.9302 ms | 11.8092 ms | 1.0102x | [0.9935, 1.0884] | 5 → 5 |
| t5 | control | 30.7789 ms | 30.7690 ms | 1.0003x | [0.9898, 1.0161] | 7 → 7 |

The 36 fewer submissions per step are the removed second readback submission
for each gathered read. The `6x4` improvement is statistically established.
The `4x4` improvement is **inconclusive**: its interval crosses 1.0, with one
slower round in the six. The clear-regression-only CI gate passes because no
case has its entire interval below 1.0. The harness's stricter
`--require-native-improvement` flag did **not** pass, and that failure is kept
in `benchmark.json` rather than relabeled.

The previously attempted cross-layer deferral of every MoE span to one read at
the end of the reverse pass was rejected: it measured `0.9233x` on `4x4` and
`0.9057x` on `6x4` against the same prior binary. The oversized reverse-pass
command buffer is the suspected cause; that attempt is not in this commit.

Historical `372407b` versus final and session OFF versus ON were not rerun in
this pass. The prior sealed receipt remains the evidence for those comparisons;
this receipt does not transfer its numbers to the new binary. Session-retirement
counts were not separately re-profiled, so no retirement-count improvement is
claimed.

## Correctness

- Targeted Rust: 9/9 tests pass (in-session readback with exact values and one
  submission, six mixed-MoE retained/adjoint/adapter tests, adapter word/order,
  and pinned MoE adjoint).
- Receipt guards: multimodal 44/44 and 85/85; GPT-2 16 cases, 85 shared-image,
  10 native Gemma/LLaVA and 30 V4.1; training-stack 12 steps and 10 coverage
  floors.
- Native qualifier: 8/8 stages pass, 95 retained cases, 16 expanded shape cases,
  strict `<2e-7`, `rtol=0`.
- The paired harness reports `cross_arm_word_exact: true`; final logits, image
  digests and adapter digests are word-identical across arms.
- `cargo fmt --check` and `git diff --check` passed before commit.

## CI

`.github/workflows/rdna3-performance.yml` targets a dedicated
`[self-hosted, Windows, X64, rdna3, vulkan]` runner, builds the candidate and
the requested baseline revision, runs the paired harness and applies
`validation/rdna3_performance_gate.py`. The gate fails only when a full 95%
interval lies below 1.0; a release promotion can opt into the stricter
native-improvement policy. The workflow is manual or weekly and intentionally
has no `pull_request` trigger. Five gate unit tests pass.

## Limitations

- Batch-2/4 **throughput** is not supported by the frozen benchmark or fixture;
  only batch-1 throughput is measured. Batch-2/4 correctness remains covered by
  the existing state-gradient/decode tests, not by this receipt.
- The physical Intel Gen9 execution target is unavailable on this RDNA3 host.
- The trusted `Hierarchos-Native-stable` shader-oracle checkout is absent, so
  shader-oracle parity remains an explicit skip, not a pass.
- No production model or production batch throughput is claimed from this tiny
  frozen geometry.
- The 4x4 interval is inconclusive and one round is visibly slow; the CI gate
  must not be read as a significant 4x4 win.

## Verify

```powershell
python validation/release_parity/multimodal_training/20261107-rdna3-v41-session-read/verify_receipt.py
```

The verifier checks the source commit exists, recomputes every file hash,
re-reads the benchmark's binary bindings and exact-word flag, and requires the
clear-regression gate to pass. See `benchmark.json`, `gate.json`, `gate.md`,
`tests.log`, `guards.log`, `native_qualifier.log`,
`native-qualification.json`, `source.patch`, `source-commit.txt` and `raw/` for
the underlying evidence.
