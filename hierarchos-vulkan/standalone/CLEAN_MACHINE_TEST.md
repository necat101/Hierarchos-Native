# Hierarchos Native — Windows x64 Portable v0.2.8 Acceptance Report

Status: **same-host extracted-artifact acceptance PASS, with explicitly blocked
items.** The archive was extracted from a path containing spaces and the
production selectors were exercised from the extracted executables. This report
does not re-use the v0.2.7 route evidence: v0.2.8 changes the default query pass
and the FP32 input-gradient kernel, so those routes were checked again from the
new package.

## Artifact and provenance

- Artifact: `Hierarchos-Native-Windows-x64-Portable-v0.2.8.zip`
- Packaging source commit and clean/dirty state: recorded in `BUILD_INFO.txt`
  and `BUILD_MANIFEST.json` beside the archive.
- Target: `x86_64-pc-windows-msvc` (default target; **not** `target-cpu=native`)
- Profile: release (`opt-level=3`, thin LTO, `codegen-units=1`)
- CRT linkage: static MSVC C runtime (`-C target-feature=+crt-static`)
- Packager: `hierarchos-vulkan/package-standalone.ps1 -StaticCrt` with the
  dependency scan **enabled** (no `-SkipDependencyScan`)
- The archive SHA-256 is recorded in `BUILD_MANIFEST.json` and `SHA256SUMS`
  beside the ZIP, not inside it.

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

## Embedded promoted modules

| Module | SPIR-V SHA-256 |
| --- | --- |
| `transformer_attention_backward_query_rdna3_dqunroll` | `f0214b60fac0b5fe420b846f89f43fd3cb3229f4a12d30fc2e5ad80e25c9d237` |
| `transformer_attention_backward_query_rdna3_dqunroll_scratchreuse` | `a6b3d6c6d7bd2911e61f81181d79cb41a3c4be2762cd9a0b6ee636e57d6057ee` |
| `transformer_attention_backward_query_rdna3` (fallback) | `f9652f525cbc9256188d9c998fc602ad1eff43c68bbbc6b88650697584104256` |
| `linear_input_grad_output_tile32` | `9316501e88a97a41da7a30c21e68eb31e80fd328cc6c8626790a7c8bbbb12c45` |
| `linear_input_grad` (fallback) | embedded production artifact |
| `transformer_attention_forward_rowpacked64_cached_rdna3` | `c053503e267ede432cd174e887d800640ce854aad186f8b71390e8c4543764fd` |
| `transformer_attention_backward_key_rdna3` | `ccf609843be2a37d33bb6df3b1ab490b7b3fa6a52e5816975c21524af71cd925` |
| `transformer_attention_backward_key_channel4_rdna3` | `f5480ccedce8c2752f7b94b2fe478d9c9e4f496f284344c2140abc3367f7b2c8` |

Both WMMA staging modules are unchanged from v0.2.6: the 23,280-byte P3 contract
module and the 36,912-byte C1 batched-staging module.

## Startup and diagnostics

| Command | Exit | Result |
| --- | ---: | --- |
| `bin\hierarchos-native-cli.exe --help` | 0 | usage rendered |
| `bin\hierarchos-native-cli.exe doctor` | 0 | Vulkan compute device visible; companions discovered beside `bin\` |
| `bin\hierarchos-vulkan-devices.exe` | 0 | `AMD Radeon Graphics`, `INTEGRATED_GPU`, index 0 |
| `bin\hierarchos-native-cli.exe precision-info --device-index 0` | 0 | `full_fp16_supported=true`, `mixed_supported=true`, `full_fp16_max_reduction=64` |
| `bin\hierarchos-native-cli.exe architectures` | 0 | architecture inventory rendered |
| `sha256sum -c SHA256SUMS` | 0 | archive and manifest both `OK` |

`precision-info` reports `full_fp16_qualified=false` on this host, as in v0.2.6:
that field is the CLI's driver-key allowlist, not a numerical result. No
threshold, predicate or allowlist was edited for v0.2.8.

## Production routes: PASS

Workload — the disposable deterministic Llama fixture (2 layers, `hidden_size`
1024, 8 query heads, 8 KV heads, `head_dim` 128), `batch 1 x seq 64`, one
optimizer step. That geometry is eligible for the query, key and forward
promotions. The production selector was the only chooser.

| Arm | Observed route evidence |
| --- | --- |
| default | two-pass backward effective, query pass reports `query_dqunroll_scratchreuse`, scalar dispatches 0 |
| `HIERARCHOS_ATTENTION_BACKWARD_QUERY_SCRATCH_REUSE=0` | query pass reports `query_dqunroll` |
| `HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL=0` | query pass reports the original `query_rdna3` shader |
| `HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=0` | scalar attention backward, no query scratch |
| `HIERARCHOS_ATTENTION_BACKWARD_KEY_CHANNEL4=0` | original two-pass key shader |

The FP32 input-gradient promotion was checked from the same extracted package by
running the one-step workload twice: once with the promoted default and once with
`HIERARCHOS_LINEAR_INPUT_GRAD_OUTPUT_TILE=0`. Both arms exit 0, and their
`training_report.json` loss bits are identical. The 29-step complete-step
equivalence is recorded in
`validation/phase_xxii_linear_input_grad_tile_20261128/`: identical
`parameter_end_digest = d79a12d8…9c4655` in every ABBA arm and round, with
272,038/272,038 output words bitwise equal over six kernel shapes.

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
- RGP, ISA, VGPR or occupancy counters. None were obtained; none are claimed.

## Qualification record for these binaries

The numerical and performance qualification is the receipt set under
`hierarchos-vulkan/validation/`:

- Phase XVIII dQ-unroll: eight geometry/mode series, 5/5 ABBA rounds each,
  1.042×–1.124× complete step, CI excluding 1.0.
- Phase XIX/XX scratch reuse: 140/140 bitwise tensor comparisons over 28 cases;
  29/29 in-graph loss, gradient and parameter digests; 1.088× FP32 and 1.120×
  Mixed complete step (5/5 rounds).
- Phase XXII input gradient: 272,038/272,038 words bitwise equal over six
  shapes; 1.1462× complete step (24.875 → 21.716 ms, 95 % CI 0.8706–0.8742,
  5/5 rounds).
- Retained attention routes: the Phase III–XIV receipts remain the applicable
  record for the two-pass pair, the four GQA groupings, the row-packed forward
  and the four-channel key pass.
- Mixed WMMA: unchanged, opt-in and default-off; the adapter elementwise
  observation and the full-parameter learning-rate caveat are in
  `RELEASE_NOTES.md`.

These are independent paired measurements, not multiplicative factors. No
overall or end-to-end speedup is claimed.

## Release status

This archive is the verified **v0.2.8 release candidate**. Its publication state
is recorded in the release receipt
(`hierarchos-vulkan/validation/phase_xxiii_v028_release_20261128/README.md`), not
in the package. Published contents are immutable; any further change is a new
version.
