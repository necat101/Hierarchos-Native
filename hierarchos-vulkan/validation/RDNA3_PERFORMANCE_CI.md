# RDNA3 performance CI

`rdna3-performance.yml` runs the paired V4.1 benchmark on a dedicated Windows
runner labelled `rdna3` and `vulkan`. It is manual or weekly; it is deliberately
not attached to `pull_request`, because a self-hosted runner must never execute
untrusted pull-request code.

## Runner contract

- Windows host with the repository's normal Rust toolchain, Vulkan driver and
  Git Bash.
- One AMD RDNA3 adapter, idle during the run, on its normal power profile.
- Repository variable `RDNA3_V41_FIXTURES` points at the frozen Gen9 fixture
  root (`.../target/multimodal-gen9`).
- No other GPU job runs concurrently. The workflow has a single concurrency
  group, but the runner operator must also keep interactive GPU work away.
- The baseline binary is built from `baseline_ref` (default `722acd0`) in a
  separate worktree and target directory. The candidate is the checked-out
  commit. Both hashes are printed and retained in the raw artifact.

The workflow retains every round JSON, child log and the harness `hashes.json`
for 30 days. It does not upload build trees or secrets.

## Gate policy

The paired harness reports a 95% bootstrap interval over ABBA round medians.
The CI gate is `clear-regression-only`:

- fail when any native or control interval is wholly below `1.0`;
- pass an interval that crosses `1.0`, and label it **inconclusive**;
- report significant improvements separately.

This follows the repository rule that a performance job fails only when the
evidence clearly indicates an unacceptable regression. A release promotion can
opt into `require_native_improvement`, which instead requires every native
interval to be wholly above `1.0`. That stricter policy is not the scheduled
job's default, because a small true win can be inconclusive on a noisy host.

Numerical correctness remains independent of the performance gate: the harness
requires word-identical final logits, image digests and adapter digests within
each arm and across arms, and the existing receipt guards remain mandatory.

## Batch coverage

The frozen harness currently exercises batch 1 for Native ViT 4x4, Native ViT
6x4, Llama, GPT-2 and T5. The existing batch-2/4 state-gradient and decode
isolation tests are correctness coverage, not throughput coverage. This
workflow therefore does **not** claim batch-2/4 performance: the fixture and
benchmark CLI have no batch selector, and silently treating the five batch-1
cases as multi-batch evidence would be false. Extending the fixture and harness
to batch 2/4 is a separate, explicit prerequisite; until then the workflow and
receipt record this as an unsupported configuration.

## Reproducing locally

```powershell
cd hierarchos-vulkan
cargo test --locked --lib --no-run
python validation/benchmark_rdna3_v41_paired.py `
  --baseline-binary target/rdna3-port/prior-722acd0-2c73.exe `
  --candidate-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe `
  --fixtures target/rdna3-port/gen9-frozen/target/multimodal-gen9 `
  --output target/rdna3-port/abba-session-read `
  --rounds 6 --steps 7 `
  --forbid-control-regression
python validation/rdna3_performance_gate.py `
  target/rdna3-port/abba-session-read/benchmark.json `
  --json target/rdna3-port/abba-session-read/gate.json `
  --markdown target/rdna3-port/abba-session-read/gate.md
```
