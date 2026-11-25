# Hierarchos Native — Windows x64 Portable v0.2.7 Acceptance Report

Status: **same-host extracted-artifact acceptance.** This report is written
against the v0.2.7 archive and does not re-use or replace the v0.2.6 report:
v0.2.7 changes the production attention routes, so the route evidence below was
collected again from the new package.

## Artifact and provenance

- Artifact: `Hierarchos-Native-Windows-x64-Portable-v0.2.7.zip`
- Packaging source commit: the exact checked-out HEAD at packaging, recorded in
  the archive's `BUILD_INFO.txt` and in `BUILD_MANIFEST.json` beside the archive.
- Qualified source: the `release/v0.2.7` candidate. The attention artifacts it
  embeds are the ones their receipts qualify byte for byte; the WMMA policy and
  both staging modules are unchanged from v0.2.6.
- Target: `x86_64-pc-windows-msvc` (default target; **not** `target-cpu=native`)
- Profile: release (`opt-level=3`, thin LTO, `codegen-units=1`)
- CRT linkage: static MSVC C runtime (`-C target-feature=+crt-static`)
- Standalone `Cargo.lock` SHA-256:
  `f4360adfc4a2127c0be817c71c08659c181f7ff04813b5f8bc882a23ca7132d7` (unchanged
  from v0.2.5 — no dependency was added or upgraded for this release)
- Packager: `hierarchos-vulkan/package-standalone.ps1 -StaticCrt` with the
  dependency scan **enabled** (no `-SkipDependencyScan`)
- The archive SHA-256 is recorded in `BUILD_MANIFEST.json` and `SHA256SUMS` beside
  the ZIP, not inside it.

## Package inventory

```text
bin/hierarchos-native.exe                     (GUI)
bin/hierarchos-native-cli.exe
bin/hierarchos-infer.exe
bin/hierarchos-vulkan-train.exe
bin/hierarchos-vulkan-devices.exe
bin/hierarchos-vulkan-transformer-train.exe
bin/hierarchos-vulkan-transformer-logits.exe
LICENSE   THIRD_PARTY_NOTICES.md   licenses/
README.md   README_ARCHITECTURES.md   RELEASE_NOTES.md
CLEAN_MACHINE_TEST.md   BUILD_INFO.txt   DEPENDENCIES.txt
```

Exactly seven executables. No source tree, Cargo files, shader sources, PDBs,
fixtures, validation tree, test harness or model weights.

## Embedded routes

v0.2.7 embeds the four promoted attention artifacts and both unchanged WMMA
staging modules. The attention identities are the qualified ones:

| Module | SPIR-V SHA-256 |
| --- | --- |
| `transformer_attention_forward_rowpacked64_cached_rdna3` | `c053503e267ede432cd174e887d800640ce854aad186f8b71390e8c4543764fd` |
| `transformer_attention_backward_query_rdna3` | `f9652f525cbc9256188d9c998fc602ad1eff43c68bbbc6b88650697584104256` |
| `transformer_attention_backward_key_rdna3` | `ccf609843be2a37d33bb6df3b1ab490b7b3fa6a52e5816975c21524af71cd925` |
| `transformer_attention_backward_key_channel4_rdna3` | `f5480ccedce8c2752f7b94b2fe478d9c9e4f496f284344c2140abc3367f7b2c8` |

| WMMA module | Bytes | SPIR-V SHA-256 |
| --- | ---: | --- |
| `wmma_gemm_f16_f16acc_twowave_p3` (contract, fallback) | 23,280 | `2aff4123544757b0d3aaf8eb1fe967a0c00c692eb73a74e49af802d7b5c517d2` |
| `wmma_gemm_f16_f16acc_twowave_p3_c1` (batched staging) | 36,912 | `b072242034ba7bd0f53690c38c03b11ed587b538480c04ed13ecf19fe2a8b27d` |

## Runtime acceptance

The extracted archive is exercised from a relocated path containing spaces, and
the route evidence is recorded per command. The full command log, the extracted
inventory and the checksum verification are stored in
`hierarchos-vulkan/validation/phase_xv_v027_release_20261125/`.

The dispatch check uses a disposable Llama fixture (2 layers, `hidden_size` 1024,
8 query heads, 8 KV heads, `head_dim` 128, `seq_len` 64) that is simultaneously
eligible for all four promoted routes: the two-pass backward, the four-channel
key pass at head dimension 128, the row-packed cached forward at a square
sequence of 64, and — with WMMA requested — the unchanged C1 staging module. The
production selector is the only chooser; no variant override, threshold change
or test flag is used.

## Blocked or unverified

- Independent clean Windows installation. **BLOCKED.**
- A second physical CPU/GPU, or any non-AMD adapter. **BLOCKED.**
- Physical network isolation and different-drive relocation. **BLOCKED.**
- GUI interaction and automated GUI "Check backend" activation. **BLOCKED.**
- Code signing. The binaries are **unsigned**.
- `hierarchos-vulkan-train` and `hierarchos-infer` functional paths: both need a
  native V4.1 model package that this environment does not have. **BLOCKED.**
- RGP, ISA, VGPR or occupancy counters. None were obtained; none are claimed.

## Qualification record for these binaries

The numerical and performance qualification is the receipt set under
`hierarchos-vulkan/validation/`, unchanged by this packaging:

- Two-pass MHA backward: bit-exact gradients over four geometries,
  6,316,128 / 6,316,128 elements at zero ULP; 3.2492× complete step
  (95 % CI 3.1146–3.3838).
- GQA grouped key pass: bit-exact for the four measured groupings; 19.76×
  complete step at 8q/2kv.
- Row-packed cached forward: 1.85 M / 1.85 M words bit-exact over ten geometries;
  1.2239× complete step on the shipped selector.
- Four-channel key pass: 20 parity cases, 60 tensors, 5,505,024 elements, zero
  failures; 1.5297× complete step.
- Mixed WMMA: 1.2385× / 1.2288× LoRA, 1.2792× full-parameter timing only, still
  opt-in and default-off; the adapter elementwise observation and the
  full-parameter learning-rate caveat are in `RELEASE_NOTES.md`.

These are independent paired measurements, not multiplicative factors. No
overall or end-to-end speedup is claimed.

## Release status

Published as `v0.2.7`. Published contents are immutable; any further change is a
new version.
