# RDNA3 current-source GPT-2 qualification: 2026-11-08

Physical target: Ryzen Z1 Extreme, AMD Radeon Graphics `1002:15bf`, driver
`32.0.31007.6002`. This run executes the frozen GPT-2 shape matrix and the
retained numerical harnesses on the current source and binds every result to
the executable and fixture bytes that produced it.

Tested revision: `7f6d1d61540bd3eb273e19ca208c2fd07b7c9e2c`.

The receipt cannot be bound to the commit that contains it, so it records the
revision it measured. `check_rdna3_gpt2_receipts.py` accepts that revision as
an ancestor only while all 133 bound source bytes still match the checkout; a
source change after this run fails closed.

## Result

All sixteen frozen GPT-2 shape cases pass the unchanged policy
`max_abs < 2e-7`, `rtol = 0`, `strict_less_than`. The worst required logit
error is `5.960464477539063e-08`; the worst full-training loss delta is
`9.5367431640625e-07` against the existing `2e-5` loss gate.

| Gate | Result |
| --- | --- |
| Expanded GPT-2 shapes | **16/16** pass, 0 failed checks |
| Shared CLIP image training | **74/74** cases, worst logit `1.9371509552001953e-07` |
| Kimi shared image | **3/3** cases, worst logit `3.166496753692627e-08` |
| Seq2seq image training | **6/6** cases, worst logit `6.705522537231445e-08` |
| Native Gemma3 | **4/4** cases, worst logit `1.7136335372924805e-07` |
| Native Gemma4 | **2/2** cases, worst logit `1.6391277313232422e-07` |
| Native LLaVA | **4/4** cases, worst logit `7.82310962677002e-08` |
| Native DeepSeek V4.1 | **30/30** tests, max logit error `0.0` |
| V4.1 shared-image bridge | **2/2** cases, 10 logit checks, max `0.0` |
| GPT-2 primitives | **9 passed, 0 failed, 1 ignored** |
| RDNA3 shader provenance | **43/43** byte-identical rebuilds |
| SPIR-V validation | **43/43** accepted |
| Upstream preservation | **1,341** files / **672** recipes unchanged |

No physical Intel Gen9 execution was performed. The shader gate is a static
preservation check; the Gen9 receipt retains its own hardware scope.

## Frozen inputs and executables

| Input | SHA-256 |
| --- | --- |
| Frozen CPU manifest (`gpt2-shapes-final-cpu/manifest.json`) | `709427881624237c05dccf1e9bdd7593285819304af7d3778b885bc206fbad0c` |
| Frozen CPU oracle archive (`gpt2-shapes-avx512-oracles.zip`) | `a45396b752d62b01f75aa0f0cc8a83ee71640bf44a9f83b841e7476056b6ef14` |
| `image_text_pipeline_probe.exe` | `5f4026ee4ced0a1a07d69dd511d811a516167f4459557241a0c698c569a76652` |
| `gemma3_vision_probe.exe` | `335c93f258cce167c6c359b6c0f9fa109072662a01ca77758e0011c346cc8a15` |
| `gemma4_vision_probe.exe` | `227b0cff26873aeb662b696fb2b73613ae82047d9dc16871ece4a38439f5d407` |
| `llava_pipeline_probe.exe` | `995be1a9eefd3b5ec8b9cd995c63f6f908bbe52446f35d11da104e5a14d3aeae` |
| libtest (`hierarchos_vulkan-2f708da7cb986ebb.exe`) | `cfae56cf9cb63aa207381fa13002bf62470154f654a4548ac0a7263d45bd081d` |

The frozen CPU oracles are reused unchanged; no reference value, tolerance or
frozen file was regenerated or widened. The CPU contract is Windows Torch
`2.6.0+cpu`, SDPA text attention, AVX512, one thread.

## Reproduce

From `hierarchos-vulkan`:

```powershell
cargo build --bin image_text_pipeline_probe --bin gemma3_vision_probe `
  --bin gemma4_vision_probe --bin llava_pipeline_probe
cargo test --lib --no-run

.venv-torch/Scripts/python.exe validation/qualify_rdna3_gpt2_shapes.py run `
  --fixtures target/rdna3-port/gpt2-shapes-final-cpu `
  --binary target/debug/image_text_pipeline_probe.exe --output target/current/shapes
cargo test --lib rdna3_gpt2_ -- --test-threads=1
.venv-torch/Scripts/python.exe validation/qualify_rdna3_gpt2_current.py `
  --run target/current --cpu-fixtures target/rdna3-port/gpt2-shapes-final-cpu `
  --libtest-binary target/debug/deps/hierarchos_vulkan-<hash>.exe
```

The producer refuses a dirty worktree and re-verifies every gate before it
writes a passing receipt. See `run-status.json` for per-gate exit codes and
`current-qualification.json` for the complete source, fixture, executable and
evidence bindings.

## Limits

- Certifies eight measured GPT-2 geometries, not their Cartesian product.
- This receipt is correctness evidence only. Paired training throughput is a
  separate measurement and is not claimed here.
- The expanded matrix uses the existing frozen CPU words; it does not rerun
  CPU oracle generation.
- Native V4.1 retains its frozen-language LoRA contract.
