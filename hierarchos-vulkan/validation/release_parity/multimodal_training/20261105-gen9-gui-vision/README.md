# Gen9 GUI and CLI vision layer (2026-11-05)

This receipt covers the frontend layer that exposes the already-qualified Gen9
image backends: five native CLI commands and three desktop workflows. It adds no
Vulkan kernel and changes no frozen fixture. The backend parity record remains
`20261104-gen9-full-image-training` and `20261105-gen9-v41-upload-batching`.

## What is qualified

Frontend parity replayed the existing frozen CPU oracles through the new CLI
routes. Each case trains the frozen two-step schedule at the recorded hyper
parameters and compares the exported parameters against the unchanged
`cpu-step-2.safetensors` or the native V4.1 joint expectation. No expectation was
regenerated.

| route | max_abs | exact |
| --- | ---: | :---: |
| shared-gpt2 | 2.9802322387695312e-08 | yes |
| shared-llama-wide | 0.0 | yes |
| seq2seq-t5 | 2.9802322387695312e-08 | yes |
| native-llava | 1.4901161193847656e-08 | yes |
| native-gemma3 | 2.9802322387695312e-08 | yes |
| native-gemma4 | 5.960464477539063e-08 | yes |
| native_vit_4x4 | 0.0 | yes |
| native_vit_6x4 | 1.1641532182693481e-10 | yes |

All eight are inside the established `< 2e-7` gate. Raw output is
`gui-cli-parity1.log`.

Unit tests: 13 passed, 2 ignored (`gui-unit-tests4.log`). The ignored tests need a
local model root and are the Falcon integration test plus the worker smoke below.
`gui-worker-smoke1.log` records the ignored worker smoke as `pass: true`,
`returncode: 0`, one expected test. It drives the same `NativeApp::start`,
supervision and `drain_output` path as the UI: a real vision training run, a
reload-and-generate step from the exported package, and a real 64x64 single-step
text-to-image through the pinned engine, asserting the written PNG's dimensions
and its receipt's backend.

## Source under test

The smoke binary was built from these sources; the hashes below are the ones the
recorded qualification ran against.

| file | sha256 |
| --- | --- |
| `hierarchos-native-cli/src/vision.rs` | `6f9deb6c42bb148bc073bada15aa6c165d397a4500a8351c28277058ba5cf577` |
| `hierarchos-native-cli/src/vision/data.rs` | `4f69c0f02cea9dbda6b7597df4f33ef4314e1c238e94a43743f5c8228cb402d6` |
| `hierarchos-native-cli/src/vision/package.rs` | `7edf8b29b77009066ace686477c9346a7684f6e77edc296e013acf1516360c27` |
| `hierarchos-native-cli/src/vision/runtime.rs` | `9d36b480fa432c90f4fb1486eb0ce46df38581ebadcf94d6ac287ab8666a336d` |
| `hierarchos-native-cli/src/vision/tests.rs` | `ef4928ee4c3544d150f51f0a70596f6a38101d99c1d579a18727080bcf37dc32` |
| `hierarchos-native-cli/src/image_generation.rs` | `0587891a18f1b63a24f2bdc2c85a15b87a0a633cdcca6f70920b92e4485956f8` |
| `hierarchos-native-cli/src/lib.rs` | `b8c7aca60cb7af0ebd3d097c54e9c8d393ec61e6798f78cf7196cb150071eeb0` |
| `hierarchos-native-cli/install-image-engine.ps1` | `8bf855b9d95a26118eee0549f3daa087117f690c12f9b5f79594acd0de64d22a` |
| `hierarchos-gui/src/main.rs` | `54bff33ba913f58f103c1cbe1705c6d38adeb3e3fc3903ef32a3920e9de5c917` |
| `hierarchos-gui/src/multimodal_ui.rs` | `1cd7e33728453cc49f663b787d7a08841be810acd0e8b86b051ad983c58f6624` |
| `hierarchos-gui/src/process_tree.rs` | `8ea80c5876ec7d20b324fb533cf56e5ca7161c8374b6c5c35d38c029c5c948af` |

## Text-to-image boundary

`image-generate` is not a second implementation of the image backends. It invokes
a pinned stable-diffusion.cpp Vulkan build and passes the requested checkpoint,
prompt, dimensions, steps, CFG scale and seed, forcing every backend flag to
Vulkan with automatic fitting off. The engine is installed separately by
`install-image-engine.ps1`, which verifies the archive SHA-256 before extraction;
no model weights are committed. Generated images therefore carry the engine's
numerics, not the `< 2e-7` text contract, and that receipt says so explicitly.

## Limits

Qualified on one Intel HD Graphics 520 (Gen9) with FP32 frozen geometries. No
RDNA3 host has run this layer, production-size throughput was not measured, and
the smoke exercised a 64x64 one-step generation rather than a useful image. Image
models, engines and generated outputs are local artifacts and are not part of the
handoff.

## Reproduce

From the repository root, with the frozen oracle roots restored as described in
`RDNA3_MULTIMODAL_HANDOFF.md`:

```powershell
cargo build --locked --manifest-path hierarchos-native-cli/Cargo.toml
cargo test  --locked --manifest-path hierarchos-native-cli/Cargo.toml
cargo test  --locked --manifest-path hierarchos-gui/Cargo.toml
$env:HIERARCHOS_MULTIMODAL_GUI_SMOKE = '<manifest with cli, model, image, tokenizer, dataset, training_output, image_model, engine, generated_png>'
cargo test  --locked --manifest-path hierarchos-gui/Cargo.toml multimodal_start -- --ignored --nocapture
```

The smoke manifest is local: it names the installed engine and checkpoints, which
are not distributed.
