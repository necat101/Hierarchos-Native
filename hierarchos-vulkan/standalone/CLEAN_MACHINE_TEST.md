# Hierarchos Native — Windows x64 Portable v0.2.7 Acceptance Report

Status: **same-host extracted-artifact acceptance PASS, with explicitly blocked
items.** The archive was extracted from a path containing spaces and every
promoted attention route was exercised from the extracted executables with the
unchanged production selector. This report does not re-use or replace the v0.2.6
report: v0.2.7 changes the production attention routes, so the route evidence
below was collected again from the new package.

## Artifact and provenance

- Artifact: `Hierarchos-Native-Windows-x64-Portable-v0.2.7.zip`
- Packaging source commit: `c05131e3f141abc94cbf3fb01485800e4ad92f26`
  (`release/v0.2.7`), recorded in `BUILD_INFO.txt` and in `BUILD_MANIFEST.json`
  beside the archive. The tree was **clean** at packaging.
- Build timestamp (UTC): `2026-11-25T10:32:38Z`
- Target: `x86_64-pc-windows-msvc` (default target; **not** `target-cpu=native`)
- Profile: release (`opt-level=3`, thin LTO, `codegen-units=1`)
- CRT linkage: static MSVC C runtime (`-C target-feature=+crt-static`)
- Toolchain: `rustc 1.93.0 (254b59607 2026-01-19)`,
  `cargo 1.93.0 (083ac5135 2025-12-15)`
- Standalone `Cargo.lock` SHA-256:
  `f4360adfc4a2127c0be817c71c08659c181f7ff04813b5f8bc882a23ca7132d7` (unchanged
  from v0.2.5 — no dependency was added or upgraded for this release)
- Packager: `hierarchos-vulkan/package-standalone.ps1 -StaticCrt` with the
  dependency scan **enabled** (no `-SkipDependencyScan`); `PASS (DEPENDENCIES.txt)`
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

Exactly seven executables and sixteen entries. No source tree, Cargo files,
shader sources, PDBs, fixtures, validation tree, test harness or model weights.

## Embedded routes

The four promoted attention artifacts are the qualified bytes:

| Module | SPIR-V SHA-256 |
| --- | --- |
| `transformer_attention_forward_rowpacked64_cached_rdna3` | `c053503e267ede432cd174e887d800640ce854aad186f8b71390e8c4543764fd` |
| `transformer_attention_backward_query_rdna3` | `f9652f525cbc9256188d9c998fc602ad1eff43c68bbbcb6b88650697584104256` |
| `transformer_attention_backward_key_rdna3` | `ccf609843be2a37d33bb6df3b1ab490b7b3fa6a52e5816975c21524af71cd925` |
| `transformer_attention_backward_key_channel4_rdna3` | `f5480ccedce8c2752f7b94b2fe478d9c9e4f496f284344c2140abc3367f7b2c8` |

Both WMMA staging modules are unchanged from v0.2.6: the 23,280-byte P3 contract
module (`2aff4123…`) and the 36,912-byte C1 batched-staging module (`b0722420…`).

## Startup and diagnostics

| Command | Exit | Result |
| --- | ---: | --- |
| `bin\hierarchos-native-cli.exe --help` | 0 | usage rendered |
| `bin\hierarchos-native-cli.exe doctor` | 0 | `Vulkan compute devices: 1`; companions discovered beside `bin\` |
| `bin\hierarchos-vulkan-devices.exe` | 0 | `AMD Radeon Graphics`, `INTEGRATED_GPU`, index 0 |
| `bin\hierarchos-native-cli.exe precision-info --device-index 0` | 0 | `full_fp16_supported=true`, `mixed_supported=true`, `full_fp16_max_reduction=64` |
| `bin\hierarchos-native-cli.exe architectures` | 0 | `146 canonical + 83 package aliases` |
| `sha256sum -c SHA256SUMS` | 0 | archive and manifest both `OK` |

`precision-info` reports `full_fp16_qualified=false` on this host, exactly as in
v0.2.6: that field is the CLI's driver-key allowlist, which still names the
v0.2.5 driver key while this host reports `…;driver=008001a0`. No threshold,
predicate or allowlist was edited for v0.2.7.

## Promoted attention routes: PASS

Workload — the disposable deterministic Llama fixture (2 layers, `hidden_size`
1024, 8 query heads, 8 KV heads, `head_dim` 128), `batch 1 x seq 64`, one
optimizer step. That geometry is simultaneously eligible for all four promoted
routes. The production selector was the only chooser; no variant override, floor
change or test flag was used.

| Arm | Two-pass counters | Kernels the profile recorded |
| --- | --- | --- |
| default | `effective=true scalar=0 query=2 key=0 scratch_allocations=2 scratch_bytes=524288` | `…_forward_rowpacked64_cached_rdna3`, `…_backward_query_rdna3`, `…_backward_key_channel4_rdna3` |
| `HIERARCHOS_ATTENTION_BACKWARD_KEY_CHANNEL4=0` | `effective=true scalar=0 query=2 key=2 scratch_allocations=2 scratch_bytes=524288` | `…_forward_rowpacked64_cached_rdna3`, `…_backward_query_rdna3`, `…_backward_key_rdna3` |
| `HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=0` | `effective=false scalar=2 query=0 key=0 scratch_allocations=0 scratch_bytes=0` | `…_forward_rowpacked64_cached_rdna3`, `transformer_attention_backward` |

Two dispatches per kernel is one per layer. The scratch figure is exactly the
documented size: `1 batch * 8 heads * 64 queries * 64 keys = 32,768` floats per
buffer, two buffers per layer, two layers, `524,288` bytes total — no padding, no
over-allocation. Both control arms behave as documented: the channel-4 opt-out
restores the original two-pass key shader while leaving the query pass and the
forward untouched, and the two-pass opt-out returns to the scalar backward with
no scratch at all.

`hierarchos-vulkan-transformer-train.exe` wrote `training_report.json` in every
arm (`architecture: llama`, `optimizer_step: 1`, `final_loss: 6.937828063964844`
in all three).

## PE closure

All seven executables are x64 PE32+, statically linked against the MSVC C
runtime, with no `VCRUNTIME*`, `MSVCP*`, `MSVCR*`, `concrt*` or `ucrtbase` import.
The packager's portable dependency scan passed with the scan enabled and wrote the
full per-executable import closure to `DEPENDENCIES.txt`. `vulkan-1.dll` remains
an external runtime requirement supplied by the graphics environment; it is not
bundled and is not a substitute for the driver.

## Blocked or unverified

- Independent clean Windows installation. **BLOCKED.**
- A second physical CPU/GPU, or any non-AMD adapter. **BLOCKED.**
- Physical network isolation and different-drive relocation. **BLOCKED.**
- GUI interaction and automated GUI "Check backend" activation. **BLOCKED.**
- Code signing. The binaries are **unsigned**.
- `hierarchos-vulkan-train` and `hierarchos-infer` functional paths: both need a
  native V4.1 model package that this environment does not have. **BLOCKED.**
- WMMA cooperative-dispatch re-verification from this package: not repeated here,
  because no WMMA source, shader or policy changed in v0.2.7. The v0.2.6 archive
  evidence for C1 and P3 remains the applicable record.
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

This archive is the verified **v0.2.7 release candidate**. Its publication state
is recorded in the release receipt
(`hierarchos-vulkan/validation/phase_xv_v027_release_20261125/README.md`), not in
the package. Published contents are immutable; any further change is a new
version.
