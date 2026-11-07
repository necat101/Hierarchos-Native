# Intel HD 520 Gen9 multimodal reproduction pack

Target: Intel HD Graphics 520, PCI `8086:1916` (Gen9), Windows x64.

This is a reproduction pack for the current Gen9 image-training gate. It is not
a new physical Gen9 run: this host has an AMD RDNA3 adapter only. The expected
values below are copied from the retained Gen9 receipts, not inferred from the
RDNA3 port.

## Status on this host

`multimodal-gen9-status.json` records the 2026-11-06 preflight: no Intel
adapter is present, so the pack was not replayed and no receipt was issued. The
recorded adapters contain only `1002:15bf`; `8086:1916` is absent. The status
record is a blocker, not a qualification, and it must stay `status: blocked`,
`physical_run: false`, `receipt_issued: false` until a real Intel host runs the
gates below.

Use the current receipts. The 2026-11-02 SigLIP receipt was 3/4 and the old
GPT-2-wide promotion receipt was red; both are superseded by the 2026-11-03
SigLIP tail and the 2026-11-05 full-image qualification. Do not use the old
values as pass criteria.

## Prerequisites

- A checkout containing this pack and the locked Cargo dependencies.
- Vulkan runtime and an Intel driver exposing the HD 520 as `0x8086:0x1916`.
- Python 3 for the standard-library replay runners; no PyTorch is needed to
  replay a frozen fixture.
- Run from the repository root unless a command says otherwise.

Set the AVX2 oracle policy explicitly:

```powershell
$env:HIERARCHOS_ATEN_VECTOR_WIDTH = '8'
```

Build the two probes and the native test library:

```powershell
cargo test --locked --manifest-path hierarchos-vulkan/Cargo.toml --lib --no-run
cargo build --locked --manifest-path hierarchos-vulkan/Cargo.toml -j1 --bin gemma3_vision_probe --bin image_text_pipeline_probe
```

The runner rejects a non-Intel vendor or a device ID other than `0x1916` for
`--target gen9`. If your HD 520 enumerates differently, stop and record the
actual PCI identity; do not add a bypass or lower the gate.

## Current SigLIP gate

Extract the retained tail fixture (the repository already contains it):

```powershell
python -m zipfile -e hierarchos-vulkan/validation/release_parity/multimodal_training/20261103-gen9-gemma3-siglip-tail/cpu-fixtures.zip target/gen9-siglip-current
$expectedZipHash = '8f4cb6fb260d0e159d2a538ba58bc0582bcf6f2b754b109faa034893f03be758'
if ((Get-FileHash -Algorithm SHA256 -LiteralPath 'hierarchos-vulkan/validation/release_parity/multimodal_training/20261103-gen9-gemma3-siglip-tail/cpu-fixtures.zip').Hash.ToLowerInvariant() -ne $expectedZipHash) { throw 'SigLIP fixture zip hash mismatch' }
```

Run into a fresh directory:

```powershell
python hierarchos-vulkan/validation/qualify_gemma3_pipeline.py run `
  --fixtures target/gen9-siglip-current/gemma3-cpu `
  --output target/gen9-siglip-current-run `
  --binary hierarchos-vulkan/target/debug/gemma3_vision_probe.exe `
  --target gen9 --device-id 0x1916
```

Expected receipt: 4/4 cases pass. The fixture manifest SHA-256 is
`62c8b0d16992da35397d83914113f2edb9caeee08b73876d55df1f2f3129a5f8`.
The exact visible-logit values are:

| case | alternate | step-1 | step-2 | final | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `siglip_default` | 6.705522537231445e-08 | 5.960464477539063e-08 | 7.450580596923828e-08 | 7.450580596923828e-08 | PASS |
| `siglip_batch_padded` | 7.450580596923828e-08 | 5.960464477539063e-08 | 1.6391277313232422e-07 | 1.0803341865539551e-07 | PASS |
| `siglip_full_grid` | 5.960464477539063e-08 | 7.82310962677002e-08 | 1.9371509552001953e-07 | 1.9650906324386597e-07 | PASS |
| `siglip_wider` | 1.7881393432617188e-07 | 4.470348358154297e-08 | 8.940696716308594e-08 | 8.009374141693115e-08 | PASS |

Worst visible logit: `1.9650906324386597e-07`.

## Current GPT-2-wide and shared-77 gate

The current full shared root is the November 4 oracle bundle. Download and
verify it rather than using the older `matrix-cpu-fixtures.zip` transfer:

```powershell
New-Item -ItemType Directory -Force target/multimodal-gen9 | Out-Null
gh release download gen9-multimodal-20261105 --repo necat101/Hierarchos-Native --pattern gen9-full-image-training-20261104-oracles.zip --dir target/multimodal-gen9
$archive = 'target/multimodal-gen9/gen9-full-image-training-20261104-oracles.zip'
$expectedArchiveHash = 'f52a62d55790728e232f45feac3ebce1f42d0d42844a34bcab9dfa3aa62b519a'
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant() -ne $expectedArchiveHash) { throw 'Oracle archive SHA-256 mismatch' }
Expand-Archive -LiteralPath $archive -DestinationPath target/multimodal-gen9/frozen-oracle-unpack
Copy-Item -LiteralPath 'target/multimodal-gen9/frozen-oracle-unpack/target' -Destination . -Recurse -Force
Copy-Item -LiteralPath 'target/multimodal-gen9/frozen-oracle-unpack/hierarchos-vulkan' -Destination . -Recurse -Force
```

The shared root is `target/multimodal-gen9/shared-image-cpu-v3`. Confirm its
manifest before replay:

```powershell
$sharedManifest = 'target/multimodal-gen9/shared-image-cpu-v3/manifest.json'
$expectedManifestHash = 'a98dc5c32f4ba044b59a7a1e03fcac2b32d234ae12fde7af8ca3b1ff00cad1be'
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $sharedManifest).Hash.ToLowerInvariant() -ne $expectedManifestHash) { throw 'Shared fixture manifest hash mismatch' }
```

Replay the two GPT-2-wide cases:

```powershell
python hierarchos-vulkan/validation/qualify_shared_image_pipeline.py run `
  --fixtures target/multimodal-gen9/shared-image-cpu-v3 `
  --output target/gen9-gpt2-current-run `
  --binary hierarchos-vulkan/target/debug/image_text_pipeline_probe.exe `
  --target gen9 --device-id 0x1916 --profiles gpt2_wide
```

Expected receipt: 2/2 cases pass with **zero** visible-logit difference. The
current full shared-77 receipt is also green, worst
`1.7136335372924805e-07`; see
`validation/release_parity/multimodal_training/20261105-gen9-v41-upload-batching/shared77-qualification.json`.

| case | visible values | alternate | step-1 | step-2 | final | gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `gpt2_wide/unmasked` | 5,120 | 0.0 | 0.0 | 0.0 | 0.0 | PASS |
| `gpt2_wide/padded` | 4,480 | 0.0 | 0.0 | 0.0 | 0.0 | PASS |

The old `20261102-gen9-gpt2-promotion` values (`2.384185791015625e-07` final and
red) are historical. The old `20261102-gen9-gemma3-siglip` values are also
historical; the current SigLIP tail is 4/4.

## RDNA3 values are not Gen9 expectations

The physical RDNA3 shared suite still has the existing `gpt2_wide/unmasked`
residual (`2.384185791015625e-07` step-2/final over 5,120 values), while Gen9
now passes the same family with zero drift. Do not copy RDNA3 values into a
Gen9 receipt, and do not use a Gen9 zero-drift result to claim RDNA3 44/44.
The RDNA3 six-profile and native V4.1 receipts are in
`validation/release_parity/multimodal_training/20261106-rdna3-profile-completion/`.

## Machine-readable expectations

`multimodal-gen9-expected-values.json` in this directory contains the same
fixture hashes, case values and acceptance policy for automated comparison.
