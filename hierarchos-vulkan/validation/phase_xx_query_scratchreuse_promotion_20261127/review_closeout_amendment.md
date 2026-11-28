# Phase XX review closeout amendment

**Documentation-only amendment. No executable change.**

The independent Phase XX review returned **APPROVED — READY FOR LOCAL COMMIT**
with two wording ambiguities (F1, F2) and no code findings. This file records the
corrections made before the promotion was committed, so the receipt explains why
two documentation files no longer match the hashes captured in
`raw/01_production_hashes.txt`.

## 1. F1 — the two `raw/00_identity.txt` snapshots are now distinguished

The receipt previously said the starting working tree was "captured in
`raw/00_identity.txt`", which a reader could take to mean the Phase XX ending
snapshot. §1 of `README.md` now states explicitly that there are two different
files and that neither is the other:

* **starting** state — `validation/phase_xix_query_scratch_reuse_20261127/raw/00_identity.txt`,
  captured before any Phase XIX or Phase XX edit;
* **ending** state — `validation/phase_xx_query_scratchreuse_promotion_20261127/raw/00_identity.txt`,
  captured after Phase XX validation with all nine promoted paths present.

The §10 raw-artefact table row for the Phase XX file now also labels it as the
ending-state snapshot and points at the Phase XIX file for the starting state.
Neither file was rewritten, deleted or regenerated.

## 2. F2 — the A/B switch is described as a test-hook injection

The receipt described arm A as "selected through the new scratch-reuse opt-out"
and arm B as "at default environment state", which implies the paired test read
its per-arm setting from the process environment. It does not.

`README.md` §4.1, the module documentation of
`src/phase_xx_query_scratchreuse_promotion_tests.rs`, the `phase_xx_run_arm`
documentation and comment, and the `phase_xx_production_route_ab` documentation
now state the accurate description:

* the A/B test exercises the real production dispatch function, loader, device
  qualification, selector, buffers, dispatch path and production counters;
* the production route is admitted with `reason == "ready"`;
* the test does not use `experimental_pair_override`;
* for deterministic arm selection the scratch-reuse switch value is injected
  through the existing `#[cfg(test)]` `scope_scratch_reuse` hook;
* the paired A/B comparison does not obtain its per-arm setting from the process
  environment;
* actual environment-variable parsing, default behaviour, opt-out and precedence
  are verified separately by the real-process runs recorded in `raw/27_env_precedence.log`.

No test logic changed. No `scope_scratch_reuse` call was added, removed, moved or
retargeted; only prose changed.

## 3. What this amendment did not touch

* All original raw evidence is byte-identical. Nothing in `raw/` was regenerated,
  appended to or rewritten, including `raw/00_identity.txt`,
  `raw/01_production_hashes.txt`, `raw/10_compile.txt`, `raw/12_spirv_val.txt`,
  the process/test logs and the benchmark data. Selected pre-amendment digests:

  | raw file | sha256 |
  | --- | --- |
  | `raw/00_identity.txt` | `89d700f04883b8ed84746d38fb3f33d71370f3ad71cd11870f3d722daca1ee6c` |
  | `raw/01_production_hashes.txt` | `a51924a3743c7df2038d769e666bfe09df0332801a6eeecda61f151f53a63fdc` |
  | `raw/10_compile.txt` | `6ac6fd51083185276883e4e18c7a7f161490a46487137bd3f8158201f6ac63c2` |
  | `raw/12_spirv_val.txt` | `4aeb3d118f9c4bc73f76e4d6a809e15a9ef7993fcaec36296aebcfe6dc191a67` |
  | `raw/21_route_tests.log` | `f0f2e9016064e746be4fc739f579801ba3fe76ba760e98fa6de08e30d2460607` |
  | `raw/22_route_ab.log` | `8a8d7c88028682678409fc91030d05b8b3ed04d7af732136430dd785db3a10cb` |
  | `raw/23_phase_xix_parity.log` | `a01aa021361c170ed8585aa57d28d16b4ce77e1b436491851b0dd4df492e4ff1` |
  | `raw/24_phase_xviii_fallback.log` | `91c5c15c25e47e48877531df404ffa8ed2fe69e9bb5c359faaabd9b7d39fb5ce` |
  | `raw/27_env_precedence.log` | `87da53363bb920a299938a644ab619e60b7023ea3b50427391b3464cf770aaf6` |

* The sealed Phase XIX receipt is untouched and still digests to
  `078ef22eee85c82f34cbc0de62ee594d458464353fc7a2fb1cc84723af571392`,
  195 files / 2,091,594 bytes.
* The original captured source hashes in `raw/01_production_hashes.txt` remain the
  historical record of the tree that was actually built and tested. They were not
  updated to match the editorial tree.

## 4. Hash provenance: tested tree vs. amended tree

Only one file covered by `raw/01_production_hashes.txt` changed, and only in
comments. Historical and current values:

| file | sha256 in `raw/01` (tested tree) | sha256 after this amendment | change |
| --- | --- | --- | --- |
| `src/phase_xx_query_scratchreuse_promotion_tests.rs` | `bc15fffb1b497bf392e3c4cb6e95cf008bcb066d1e2bb653a8ca4d467d4f467f` | `1e290a8e9abd7901a634708881ca92142b874d64e28e1bd70df7ca9c6c4c6b9a` | module docs, `phase_xx_run_arm` docs/comment, A/B test doc — comments only |

The Phase XX `README.md` is a receipt document, not an input to the test binary;
it was not covered by `raw/01`. Its current hash is
`a7db653eb2443d3a16933ef21ebc4c40a312e982527251a16c9abcc49462b15a`.

Every other tested identity still reproduces exactly:

| artifact | sha256 |
| --- | --- |
| promoted scratch-reuse source | `219167927214d96a16f72d0ba755ee879a55eaf4b1e054233ce924d7b0ccef2a` |
| promoted scratch-reuse SPIR-V | `a6b3d6c6d7bd2911e61f81181d79cb41a3c4be2762cd9a0b6ee636e57d6057ee` |
| Phase XVIII source | `430492fd8ab9a1955fc89c5c37c4cc4a145bf03c009b00fedb3cd05e4a88dffc` |
| Phase XVIII SPIR-V | `f0214b60fac0b5fe420b846f89f43fd3cb3229f4a12d30fc2e5ad80e25c9d237` |
| original query source | `2e3f5720da27ca6d31567f4734b140e8b9c768afa2e287333f617817fad7eb56` |
| original query SPIR-V | `f9652f525cbc9256188d9c998fc602ad1eff43c68bbbc6b88650697584104256` |
| `src/transformer.rs` | `b64780474c0b7d6878cbe94acb7b9c0916c64647d3167790fbde1f4ca1886b39` |
| `src/phase_xix_query_scratch_reuse_tests.rs` | `fd2072651eab420d80735c8079f3f11a7879816b3c04bf04a62c79fec4fce8f1` |
| `src/phase_xviii_dq_accumulator_promotion_tests.rs` | `523ef7883d40e25f1e81d07dd909de4d4a08b9e384c771bca501559480ed0b08` |
| `validation/training_shader_manifest.json` | `b89b36bad953d02567ed0a1b5640caf81c11f0da3f3a97d2d92a433b2bd8d1ec` |

## 5. Confirmation

This amendment changed **no** production code, shader source, SPIR-V binary,
selector, environment parsing, loader, dispatch path, fallback behaviour, test
logic, assertion, tolerance, precision policy or numerical expression. Every Rust
statement in the touched test module is unchanged; only `//!`, `///` and `//`
text was edited. The 29-step A/B, the parity matrix and the 160-process campaign
were not re-run, because the executable tree that produced them is unchanged.
