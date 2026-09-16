# Hierarchos pure-Rust inference

This crate is the native FP32 inference path for the current Hierarchos
`coherent-v9` architecture. It intentionally has no Python, PyTorch, BLAS,
CUDA, or C/C++ runtime dependency.

The first milestone implements:

- FP32 safetensors loading with strict architecture/shape checks.
- Portable `architecture_contract_sha256` identity carried from the Python/
  Vulkan package and validated as a SHA-256 identifier before inference; the
  debug CLI reports the same identity beside its logits.
- Tied token embedding / language head.
- Shared-factorized DeepEmbed and ROSA adapters.
- Incremental ROSA suffix-automaton prediction with bounded coherent-v9 context.
- Read-only Titans LTM top-k retrieval, fast-value overlay, and token router.
- Matrix-state RWKV-v8 cells using coherent-v9 explicit-output state.
- Hard-masked manager ACT with hard-selected recurrent state.
- Worker refinement / state-derived drift recurrence.
- Autoregressive state, prefill, sampling, and token-ID generation.
- A small `hierarchos-infer` parity/debug CLI.

The core inference API operates on token IDs. The Windows native preview layers
the same tokenizer and chat prompt format used by the Python path on top of that
API, while keeping tokenizer behavior outside the learned-function core.

## Export an existing checkpoint

The old `.pt` file is a Python/PyTorch serialization format, so it is converted
once during model packaging. Python is not used by the resulting runtime:

```powershell
python tools/export_rust_inference.py path\to\model_or_checkpoint path\to\rust_model
cargo run --release --manifest-path hierarchos-inference\Cargo.toml --bin hierarchos-infer -- --model path\to\rust_model --tokens 1,2,3
```

The export contains `model.safetensors` (FP32 weights),
`hierarchos_rust_config.json` (the resolved learned-function contract), and the
portable tokenizer assets needed by native text frontends. If a direct `.pt`
file is outside its tokenizer directory, pass `--tokenizer-source DIR`.
Packages produced after the architecture-contract identity was introduced also
carry `architecture_contract_sha256`; native inference preserves that identity
so PyTorch, Vulkan training, and Rust inference can name the same learned
function/package contract without reinterpreting optimizer-side files.

End-to-end Python/Rust logit parity can be checked with:

```powershell
python tools\verify_rust_inference_parity.py
python tools\verify_rust_inference_parity.py --width 192
```

The second command exercises multiple 64-wide RWKV matrix heads, matching the
multi-head state layout used by production-sized Hierarchos models.

## Windows native GUI preview

The repository includes an egui frontend that links the Rust inference runtime
directly and does not start Python for native chat:

```powershell
cargo build --release --manifest-path hierarchos-gui\Cargo.toml --bin hierarchos-native
```

Run `hierarchos-gui\target\release\hierarchos-native.exe`, choose an exported
Rust model directory, and chat through the in-process FP32 runtime. The main GUI
can also launch the separate `hierarchos-native-cli`/`hierarchos-vulkan` training
path when **Vulkan (native)** is selected; that training worker remains Rust +
Vulkan and does not route through PyTorch. Framework-only controls remain on
their legacy path or fail closed when requested from the native backend.

## Android direction

The crate is deliberately filesystem + CPU only. The model core does not spawn
processes, call Python, or depend on platform GPU libraries. A future Android
wrapper can expose this crate through JNI while keeping the exact same recurrent
state and model package format. SIMD/scratch-buffer specialization can then be
added under architecture-specific Rust modules without changing model semantics.
