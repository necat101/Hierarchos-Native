# Phase XV — v0.2.7 release record

Status: **CANDIDATE BUILT AND VERIFIED — NOT PUBLISHED.**

The release candidate is complete: audited source, re-verified shaders, a clean
provenance build, an assembled and checksummed portable archive, and route
acceptance from the extracted archive. Publication was withheld for one concrete,
documented gate: the repository's `workspace check` fails on the sealed unified
release receipt, whose Phase XIX-W component pins `src/transformer.rs` to the
v0.2.6 digest. Fixing that is the full reseal procedure, which this release is
explicitly not allowed to reopen. Details and the exact remediation are below.

## Identity

| Item | Value |
| --- | --- |
| Base | `560809eead186f832344c15d1a9b70795da3b75d` (`main`, v0.2.6 merge) |
| Branch | `release/v0.2.7` |
| Candidate release commit | `63a46f3` — "Release v0.2.7" (476 files) |
| Tool fix | `15d0fbc` — push-constant checker accepts `%` in `#if` |
| Verification record | `c05131e` — shader/layout verification |
| Packaging source commit | `c05131e3f141abc94cbf3fb01485800e4ad92f26`, tree clean |
| Pull request | https://github.com/necat101/Hierarchos-Native/pull/12 |
| Tag | none created |
| GitHub release | none created |
| `main` | unchanged at `560809e` |

## Package assets

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| `Hierarchos-Native-Windows-x64-Portable-v0.2.7.zip` | 34,136,635 | `82a7baedadf620b026b1a3ca90170fea980c783c203a74b55b0ad462a1537f66` |
| `BUILD_MANIFEST.json` | 8,415 | `a558bcaeaa8bf13a6d519b35566bd2aa65d46664e8beb68f78e9ce79ccd5a2f2` |

`SHA256SUMS` is LF-only and verifies both sidecars beside the archive. As in
v0.2.5 and v0.2.6, the manifest and checksum sidecar are committed under
`phase_lxxxiv_evidence/release_assets_v027/`; the 34 MB archive is not committed
and is rebuilt by the assembler from the recorded stage.

- Build: `hierarchos-vulkan/package-standalone.ps1 -StaticCrt`, default target
  `x86_64-pc-windows-msvc`, release profile (`opt-level=3`, thin LTO,
  `codegen-units=1`), static MSVC CRT, dependency scan enabled → `PASS`.
- Toolchain: `rustc 1.93.0 (254b59607 2026-01-19)`,
  `cargo 1.93.0 (083ac5135 2025-12-15)`.
- Two builds were made. The first recorded `Source state at packaging: dirty`
  (three untracked evidence files); those were committed and the archive above
  comes from the second, clean build at `c05131e`.
- Seven executables, sixteen archive entries, executable hashes recorded in
  `BUILD_INFO.txt` and in the manifest.

## Shader audit

All eight promoted GLSL/SPIR-V artifacts match their receipts byte-for-byte. The
four SPIR-V modules that carry the new attention routes pass `spirv-val` and
rebuild byte-identically from the recorded source with the manifest's flags; the
record is `raw/01_shader-validation.txt`. Push-constant layouts:
`PASS: 531 shader push blocks, 109 host structs` (`raw/02_push_constant_layouts.txt`),
after the checker was taught to evaluate `%` in `#if` so the channel-4 shader's
`#if 128 % XI_CHANNEL_CHUNK != 0` guard is counted rather than skipped.

Mixed WMMA remains **default-off**; the scalar backward remains the fallback;
`FULL_FP16_MAX_REDUCTION = 64`, `WMMA_MIN_PROFITABLE_MACS = 16_000_000` and the
`HIERARCHOS_ATTENTION_BACKWARD_KEY_CHANNEL4=0` opt-out are unchanged.

## Acceptance

From the archive extracted to a path containing spaces, on the packaging host
(AMD Radeon Graphics, `pci=1002:15bf`), at `batch 1 x seq 64`:

| Arm | Counters | Routes |
| --- | --- | --- |
| default | `effective=true scalar=0 query=2 key=0 scratch_allocations=2 scratch_bytes=524288` | forward row-packed cached, backward query, backward key channel-4 |
| channel-4 off | `effective=true scalar=0 query=2 key=2 …` | forward row-packed cached, backward query, backward key |
| two-pass off | `effective=false scalar=2 … scratch_allocations=0` | forward row-packed cached, backward (scalar) |

`doctor`, `devices`, `precision-info`, `architectures`, `--help` and
`sha256sum -c SHA256SUMS` all exit 0. Full transcript:
`raw/03_package_acceptance.txt`; the shipped `CLEAN_MACHINE_TEST.md` carries the
same results plus the blocked items (no second machine, no second GPU, no GUI
interaction, unsigned binaries).

Rust: `cargo check --lib --bins --tests` passes.
`cargo check --all-targets` fails only on the pre-existing
`examples/rdna3_shader_probe.rs` (broken at `main` as well; CI does not build
examples).

## CI

| Workflow | Result |
| --- | --- |
| candidate a | success |
| evidence integrity | success |
| phase lxv | success |
| workspace check | **failure** |
| packed-cell producer A/B | queued (self-hosted runner) |

Failing step, `workspace-check` step 19 "Unified FP32 Mixed and scoped Full FP16
release":

```text
ValueError: matrix source contract changed: transformer_sha256
##[error]Process completed with exit code 1.
```

`phase19w_matrix.py::_validate_source_contract` compares the sealed component

```text
release_parity/multimodal_training/20261117-phase42-current-source/evidence/unified/phase19w.json
```

against the working tree. It pins `transformer_sha256 = 0d8e2ba3…` (v0.2.6's
`src/transformer.rs`); v0.2.7's is `770fb20c…`. The same check passes at `main`
for exactly that reason. It is not a flake, and it is not caused by this host.

`RELEASE_RESEALING_PROCEDURE.md` allows only one remedy: re-execute the
producers against the qualified source. Rewriting the digest in place, copying an
older receipt into a newer root or relaxing a threshold is not resealing, and was
not done. The compliant fix is the full Phase XIX-W/X/Y/Z reseal of all 24
components, which needs the scoped precision campaign, the wide-matrix families,
the 166-record text campaign, the pinned transformers checkouts, the Phase XIX-Y
oracle archives, the pinned torch build and the runtime/GUI evidence — none of
which exist on this host, and reopening completed qualification is out of scope
for v0.2.7.

## Preservation

- `v0.2.6` release: intact — `tag=v0.2.6`, published `2026-11-23T05:18:24Z`,
  3 assets, annotated tag object `678868f7` → commit `f9b683e`.
- `main` and `origin/main`: unchanged at `560809e`.
- Pre-v0.2.6 untracked evidence stash: intact at
  `831f892774619603c2dd68d0dd654d315f1a8f7e`.
- 2,860 committed validation receipt files remain untouched; no historic receipt
  was edited, regenerated or deleted, and no `.gitignore`d `.log` or
  `.safetensors` was force-added.
- No force-push, no history rewrite, no `git reset --hard`, no `git clean`.

## How to finish

1. Run the sanctioned reseal for the Phase XIX-W matrix and the unified receipt
   against `c05131e` on a host that has the campaign fixtures and pinned
   interpreters, then commit the regenerated receipt.
2. `gh run watch` until `workspace check` is green on `release/v0.2.7`.
3. `gh pr merge 12 --merge` (or `git checkout main && git merge --no-ff
   release/v0.2.7`), then `git tag -a v0.2.7 -m "Hierarchos Native v0.2.7"` and
   push the tag.
4. `python hierarchos-vulkan/validation/phase_xv_v027_release_20261125/make_release_body.py`
   then `gh release create v0.2.7 --notes-file <body> <zip> BUILD_MANIFEST.json SHA256SUMS`
   from `target/v027-dist`.
5. Verify: download the published assets and check them against `SHA256SUMS`.
