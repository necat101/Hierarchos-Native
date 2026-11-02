# Intel Gen9 architecture port: 2026-11-01

The i5-6200U / Intel HD Graphics 520 (`8086:1916`, driver `31.0.101.2115`)
passes the recorded inference matrix at **`max_abs < 2e-7`, `rtol=0`**.
This continues the verified RDNA handoff on Gen9. The numerical coverage is
43 profiles representing 26 native architecture identifiers, plus the retained
DeepSeek V4.1 model gates. The 146-entry registry has 120 other identifiers
outside this new numerical matrix.

The [machine-readable receipt](release_parity/gen9_architectures/20261101/qualification.json)
and [evidence archive](release_parity/gen9_architectures/20261101/evidence.zip)
retain the independent CPU packages, raw reports, logs, source snapshots,
negative controls and executable hashes. Earlier V4.1 receipts remain intact.

| Check | Result |
| --- | --- |
| Forward inference | 166/166 runs: 83 cases, resident and streaming |
| CPU forward logits | 66,474 observable logits; worst `1.7881393432617188e-7` |
| Storage identity | Every forward logit word matches, including padded queries |
| Earlier RDNA profile coverage | All 32 recorded profiles pass Gen9 inference |
| Gen9 tiled GEMM coverage | All eight wide GPT-2/Llama mode/mask controls dispatch `intel-gen9-lds` |
| Cache-on/off generation requests | 72/72 runs across 18 profiles, four steps each; SmolLM3 uses its existing full-prefix fallback |
| CPU generation logits | 9,216 logits; worst `8.940696716308594e-8`; generated tokens match |
| Generation storage identity | Every resident/streamed logit word matches |
| Fresh V4.1 model/training regressions | 53 passed, zero failed, 24 explicit opt-in tests ignored |
| Native streaming/T5/Qwen3-Next regressions | Six passed, including two-step LoRA/AdamW identity |
| Qualification negative controls | 14 passed; included in workspace CI |
| Original V4.1 frozen files | All 68 SHA-256 values unchanged |
| CPU fixture reproduction | All 410 package/input/expectation files match the first frozen preparation |
| Previous V4.1 implementation | All 172 previously hashed files unchanged |

## Completed port paths

The baseline passed 156 of 166 forward runs. The ten failing runs are retained
in the archive alongside the passing candidate results.

* **Qwen3-Next padding:** the recurrent projection caller now passes the padding
  mask for multi-token forwards and its adjoint. It shares the existing
  pre-projection masking used by Qwen3.5. Single-token decode follows the CPU
  reference's mask omission. Padded logit drift falls from `8.881911635398865e-4`
  to `2.9802322387695312e-8` in both storage modes.
* **DBRX packed experts:** streaming recognizes each transposed down-projection
  block in `[experts * input, output]` storage. Every candidate source is still
  proved word for word. The regression checks both expert blocks, partial row
  tiles and rejection when only the first row matches.
* **Encoder-decoder streaming:** the explicit seq2seq constructor streams
  ordinary encoder/decoder projections, decoder cross-attention and the output
  head. T5 encoder traversal no longer depends on a decoder PEFT schema. Reader
  counters are deduplicated across both graphs. T5, BART and Switch pass both
  unmasked and padded cases. Ambiguous head sources remain rejected.

Default model constructors keep resident parameters. Streaming remains explicit.
The logits and generation probes report raw FP32 words and PCI identity; the
benchmark additionally accepts `resident` or `streaming` as its final argument.

## Resident-mode measurements

Three interleaved rounds use three warmups and eleven measured forwards per
leg. Each invocation checks CPU parity and identical baseline/candidate/storage
logit words. No build or second GPU workload runs during timing.

| Retained control | Candidate resident median, ms | Candidate streaming median, ms | Resident speedup |
| --- | ---: | ---: | ---: |
| GPT-2 wide | 2.5523 | 14.1387 | 5.54x |
| Llama wide | 2.9171 | 8.8634 | 3.04x |
| Qwen4-Exp mixed PLE | 5.8386 | 25.1100 | 4.30x |
| MiniMax-M3 | 3.2118 | 10.6700 | 3.32x |

These are observations on small retained checkpoints. Baseline streaming
measurements are archived too; their run-to-run variation does not establish a
new shader speedup. This phase preserves the existing shader arithmetic.

## Reference and reproduction

The independent CPU oracle uses the clean adjacent Transformers checkout at
`469230357aab0f2b303b0d638c1f8d06edb14184`, Torch `2.14.1+cpu` revision
`5c4886908584029761b579af026dcfb627c84070`, AVX2, one CPU thread. CPU preparation
runs no native executable. Preparation and execution use separate commands and
new output directories. Source adapters and Kimi remote-code inputs are retained.

From the repository root, with the validation environment and local Transformers
checkout available:

```powershell
cargo build --locked --release --manifest-path hierarchos-vulkan/Cargo.toml -j1 --bin hierarchos-vulkan-transformer-logits --bin transformer_generation_parity --bin parameter_streaming_bench
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/qualify_gen9_architectures.py prepare --output target/gen9-repeat/cpu
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/qualify_gen9_architectures.py run --fixtures target/gen9-repeat/cpu --binary hierarchos-vulkan/target/release/hierarchos-vulkan-transformer-logits.exe --output target/gen9-repeat/forward
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/qualify_gen9_generation.py prepare --fixtures target/gen9-repeat/cpu --output target/gen9-repeat/cpu-generation
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/qualify_gen9_generation.py run --fixtures target/gen9-repeat/cpu-generation --binary hierarchos-vulkan/target/release/transformer_generation_parity.exe --output target/gen9-repeat/generation
```

CPU padding comparisons inherit the existing observable-query-row contract:
11,298 padded-query values across both modes are excluded from the CPU forward
comparison. Storage identity includes those values. The Kimi-K3 primitive oracle
covers dense unpadded inference; its three profiles have no padded or cached
generation certificate here.
SmolLM3 does not advertise native KV-cache support in the current graph. Its
cache-enabled request therefore tests the full-prefix fallback against the CPU
cached reference; this receipt does not certify native SmolLM3 cached decoding.

Source-first body loading is measured for Llama, Llama wide, Mistral, Mixtral,
Qwen2 and both Qwen2.5 profiles. Other ordinary-linear conversions, including
seq2seq, follow resident loading and do not certify bounded loading memory.
Specialized packed parameter operators retain their existing residency policy.
The 32-profile inference result does not extend the independent CPU PEFT training
certificate to every profile. The earlier [V4.1 qualification](DEEPSEEK_V41_GEN9_RECEIPT.md)
continues to describe its full 250-test gate and fixture-provenance limits.
