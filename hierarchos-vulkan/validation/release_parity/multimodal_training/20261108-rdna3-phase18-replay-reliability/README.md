# RDNA3 Phase XVIII — checkpoint replay reliability

This receipt is new. It does not replace or rewrite the Phase XV, XVI, or earlier
receipts.

## Source and executable

- Tested source: `d799890861af26fa8b33f03a5c17f7a34059c6fc`
- Parent: `259ccd63da89f30212dc34ba795d6b8e950bb999`
- Last numerical qualification commit: `a2d68f9a46d82d23fbdeb471f3791b1ad8840d97`
- Test executable SHA-256:
  `cc558cc72012bc5bb0362093af6f9a3b5e8bdf07fdda06be3883d24d200dad62`
- Frozen manifest SHA-256:
  `a9e519008e712105ceb1b135227398785617db3eb03c65e7518bb7f225966636`
- Cargo manifest SHA-256:
  `431d2c1709e299b8afe077480df1a85e9e43abd44a333872272dec56228b51d2`
- Cargo lock SHA-256:
  `802782847273b8d7d4ff34991e2a5facfbc6954fff0e416eeb9108a2426bab02`

The six source changes are test/infrastructure changes. No production Vulkan or
model-kernel source changed. See `source.json` and `source.diff`.

## Root cause

The V4.1 test allocator built its package root from process ID and a
process-local atomic counter. The counter reset to zero in every new process.
An interrupted earlier run could therefore leave a stale directory such as
`%TEMP%\hierarchos-v41-lm-<pid>-0`. A later execution that received the same
reused Windows PID attempted `fs::create_dir` on that existing directory and
failed with Windows `os error 183` before checkpoint export/resume logic ran.
This was a test scratch-name collision, not a checkpoint format or kernel
failure.

Affected test infrastructure:

- `hierarchos-vulkan/src/deepseek_v41/model_tests.rs`
- `hierarchos-vulkan/src/deepseek_v41/checkpoint_tests.rs`
- the native checkpoint-resume coverage in
  `hierarchos-vulkan/src/deepseek_v41/native_vision_state_tests.rs`

## Repair and invariants

The package-root allocator now uses a UTC timestamp, process ID, an atomic
counter, and a bounded retry on `AlreadyExists`. The retry handles an unrelated
concurrent creator without suppressing any other filesystem error. Every
checkpoint, export, resume, and freshness path for one test invocation still
lives under one reserved package root.

The added tests cover concurrent reservation, repeated sequential reservation,
a deliberately stale marker directory, real checkpoint export/resume, and
cleanup restricted to test-owned paths. The checkpoint freshness assertions
were not weakened, and the test does not assign a fresh root to operations that
are supposed to share checkpoint state.

## Targeted results

All three serial checkpoint-resume runs and both parallel processes exited 0.
The new allocator/concurrency/stale/cleanup tests each passed under `--exact`.
See `ci/targeted/` for the raw logs.

The Windows path smoke test passed 5/5. It exercises real `cygpath -u/-w`
conversion, directory checks, build-log initialization, candidate/baseline
separation, and workflow path selectors through Git Bash. See
`ci/path-smoke.log`.

## Complete nine-stage producer

The initial aggregate (`full-4`) completed eight stages but failed the
DeepSeek stage before running it because the test inventory had changed from
244 to 248 discovered tests. The validator still required the old count. That
failed run is preserved under `aggregate/full-4-failed/` and is not used as a
pass.

The validator now excludes the four new infrastructure tests from the
numerical inventory. The resulting inventory is 248 discovered, 4 excluded,
244 numerical, 218 active, 26 ignored.

Two clean complete reruns then passed in one uninterrupted process each:

| Run | Exit | Stages | Shared | Seq2seq | Gemma4 legacy/RGB | Gemma3 | LLaVA | Text | DeepSeek | Native V4.1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full-5 | 0 | 9/9 | 77/77 | 6/6 | 1/1 + 1/1 | 4/4 | 4/4 | 166/166 | 218/218, 26 ignored, 28 shards | 30/30 |
| full-6 | 0 | 9/9 | 77/77 | 6/6 | 1/1 + 1/1 | 4/4 | 4/4 | 166/166 | 218/218, 26 ignored, 28 shards | 30/30 |

Both run manifests retain `absolute_limit = 2e-07`, `rtol = 0`, and
`comparison = strict_less_than`. The worst non-native stage logit in both runs
was `1.7881393432617188e-07`; native V4.1 reported `max_abs_logit = 0.0` and
completed all 30 tests. The raw stage logs and exit codes are in
`aggregate/full-5/` and `aggregate/full-6/`.

## Current-source qualification

- GPT-2 frozen shapes: 16/16, worst logit `5.960464477539063e-08`, bound to
  source `d799890861af`.
- Current-source receipt guard: PASS, including 129 current sources and four
  immutable historical bindings.
- V4.1 bridge replay: 2/2 cases, 10 logit checks, `max_abs = 0.0`.
- Phase XV receipt verifier: PASS.
- Multimodal receipt guard: PASS.
- Historical GPT-2 receipt guard: PASS.

## Checks that did not pass

`cargo fmt --check` still reports the repository's pre-existing formatting
differences (including `src/vulkan.rs` and previously noted test files). The
two allocator test files (`model_tests.rs` and `checkpoint_tests.rs`) pass
`rustfmt --check`; `native_vision_state_tests.rs` retains known pre-existing
differences. `git diff --check` passes. These are recorded as formatting
limitations, not converted into passes. The receipt verifier treats the
non-zero `cargo fmt` and native-file results as expected and checks the
narrower allocator-test result.

## Hardware limitation

The repository has no registered self-hosted RDNA3 runners
(`total_count = 0`). No RDNA3 hardware CI PASS is claimed. All numerical
evidence here is the existing local qualification path; remote hardware CI
remains unavailable until a runner is deliberately and securely registered.

## Verification

Run:

```bash
python verify.py
```

The verifier checks the SHA-256 manifest, source ancestry and changed-file
hashes, the failed and passing aggregate records, targeted/path logs, current
GPT-2 and V4.1 evidence, receipt guards, and the known formatting limitation.
