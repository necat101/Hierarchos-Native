# Physical ROG Ally qualification

Status: **qualified on the tested device**, with the limitations listed below.
Every numerical gate was re-run on the final tuned route tables and again on
the final GPT-2 module; the shipping bundle, independent mathematical oracles
and paired performance measurements are recorded under `target/ally-results/`
(ignored working evidence, not source).

## Reference and physical baseline

- Candidate starts at `8f4eaead6ed6ce78042d20fda92661dbe246c1f0`.
- Work branch: `codex/rdna3-ally-qualification`.
- Clean numerical reference: `2e7893d4e3db17370304a8d67adacab963b22bbc`,
  isolated at `../Hierarchos-windows-oracle`. Its parent is stable `b519361`;
  its only changes are the three Windows build adapter files.
- The user-supplied `C:/Users/User/Downloads/Hierarchos-Native` remains at
  clean stable `b519361`.
- ASUS ROG Ally RC71L_RC71L; Ryzen Z1 Extreme, 8 cores / 16 threads.
- AMD Radeon Graphics, PCI `1002:15bf`, integrated GPU; Windows driver
  `32.0.31007.6002`; Vulkan driver `26.10.07.06 (LLPC)`, API 1.4.344.
- Vulkan reports 32 KiB shared memory, subgroup size 64, 1024 maximum
  workgroup invocations, FP16 arithmetic and 16-bit storage-buffer access.
- Four 4 GB Micron LPDDR5 packages at 6400 MT/s; 11,494,346,752 usable bytes.
  Turbo Windows power scheme, AC connected, battery 100%. A Windows power-scheme
  name does not establish the firmware's sustained APU power limit.
- Fresh local MSVC builds with Rust/Cargo 1.93.0; no Intel executable reuse.
- Validation uses isolated CPython 3.12.13 with the Intel evidence's pinned
  packages (torch 2.14.1, numpy 2.5.3, safetensors 0.8.0, accelerate 1.15.0,
  huggingface-hub 1.33.0, tokenizers 0.23.2). Transformers is a separate clean
  checkout at `469230357aab0f2b303b0d638c1f8d06edb14184`. Kimi K3 source is
  independently downloaded at `f831ab66814297da540d832a5235f8e904f29d06`.

## Final qualification

The final run is `target/ally-results/qualification-small-welford`. It repeats
the complete 13-stage suite on the tuned route tables after the GPT-2
short-width module described below and passed all 13 stages in 2,314 s:

| Stage | Result |
| --- | --- |
| Python contracts | 375 tests pass, the same 12 opt-in skips |
| All binaries | `cargo check --all-targets` passes |
| Library and trainer | 813 library tests pass (9 ignored), 40 trainer tests pass |
| Two-rank TCP | both modes, all 290 exported model/optimizer tensors identical |
| Inference / native CLI / GUI crates | all pass |
| Stable recurrent | 34 cases, 38,930 exact digests, 0 differences |
| Stable logits | 75 comparisons, 36,096 logits, 0 differing bits |
| Vendor composition | all four RDNA3 modules dispatched in the composed graph |
| CPU bridge | pass |
| Release training ladder | 12/12: 6,464 dispatches, 3,076 barriers, loss 5.543217, 32,162 digests |
| Debug training ladder | 12/12, same count, barrier bound and digests |

This run repeats the same tuned route tables and 13 stages as
`qualification-ship-posttune`; only the GPT-2 module, the vendor trace flag and
the shader manifest changed, so the reference executables were rebuilt and the
run took longer. The tuned-route result remains in that earlier directory.

The same 13 stages had already passed before tuning. Tuning then measured the
five producer families and the six packed-chain/policy pairs, and every
numerical post-tuning gate passed in that automated run as well. Its
`python-contracts-after-tuning` stage stopped on a stale test assumption (every
chain sweep geometry now has a measured row, so a control that required an
unmeasured batch raised `StopIteration`). That is not a numerical failure: the
control now removes one verdict itself and still requires the guard to reject
the unlisted win. The runner then restored the pre-tuning tables byte-for-byte,
which is why the tuned tables were re-applied from
`target/ally-results/tuned-routes-ship/` and requalified from scratch. The
failed run and its rollback remain in `qualification-ship/` as evidence.

Tuning added 27 AMD rows across seven decision files; all 35 Intel/other-adapter
rows are preserved unchanged (`optimization-inventory-final.json`,
`intel-routes-preserved-final.json`). Measured and withheld AMD rows:

| Family | Measured | Withheld |
| --- | --- | --- |
| Time-mix projection | 5 (3 fused, 2 staged) | 2 composed wins after eight-timestep drift |
| Channel mix | 5 (4 fused, 1 staged) | 0 |
| Low-rank full forward | 10 (9 staged, 1 fused) | 9 staged rows that change FP32/FP16 bits |
| Key/state | 5 wide-fused | 0 |
| Packed forward | 5 (3 legacy, 2 packed) | 0 |
| Packed rematerialization | 10 (9 legacy, 1 packed) | 0 |
| Packed backward fusion | 8 (6 folded, 1 staged, 1 fused) | 0 |

A faster row is recorded only after the driver's own A/B shows zero worst
absolute difference and the exact trajectory accepts it. The low-rank FP32
driver previously recorded a faster staged graph whose values differed; it now
withholds exactly like the FP16 driver. The time-mix composition that passes the
single-cell probe but drifts over eight timesteps is withheld at the measured
device/driver, so the original fused whole producer ships.

## Second-update drift: resolved

The earlier blocker was reproduced against the clean oracle: the first optimizer
update was bit-identical, the second diverged at batches 2-5. Controlled replays
showed **two valid FP32 backward programs** on driver `00800184`:
materialized time-mix adjoints followed by vector additions, and the original
fused fan-in modules. The clean oracle's own timing search can select either, so
an automatic-versus-automatic comparison is not a reproducible reference on this
driver; disabling vendor modules did not remove it.

The committed AMD goldens independently fix the required association: batch 2
and the width-256 odd/multi-sequence fixture use fused arithmetic, batch 32 uses
materialized arithmetic, under both storage policies. Clean stable replays match
every existing digest with the corresponding association and fail with the
opposite one; no golden was replaced. `rwkv_fan_in_reference_profiles.json`
pins that association before timing, and only the registered width-128,
ranks-32/32/64, maximum-batch-32 geometry may use the AMD fused modules that
reproduce it. Forced state/projection/fan-in choices and the numerical class are
now part of the backward autotune cache identity, so a forced experiment cannot
overwrite the unconstrained winner; constrained experiments stay in memory and
never enter the shared v1 cache file.

The structural ladder check had the same root cause: timing could legitimately
select 6,368, 6,416, 6,464 or 6,512 dispatches with identical tensors. The
adapter/driver-scoped probe now reproduces the recorded recipe explicitly
(RkvAdd3, split-input-mix then weight gradient, FusedOuter fan-in) and still
asserts the unchanged 6,464 dispatches, 3,076 barriers and exact loss. Normal
shipping loss is checked separately through the complete trajectory gate.

Finally, the low-rank topology choice could flip within a measured 13.7% spread.
For AMD `1002:15bf` driver `00800184` width-128 ranks-32/32/64, adoption now
requires a 15% win in all seven rounds; Intel and other geometries keep the 2%
margin. Forty debug replays produced 160/160 one-row selections with zero flips,
and 119 contended decisions are retained as diagnostics instead of promotions.

## Independent oracles and applications

`target/ally-results/independent-final-qualified` records nine serial stages,
all exit zero, with `inputs_unchanged: true`:

- 19 additional small-rank recurrent cases (both policies, 31,914 digests),
  including a raw masked accumulation, checkpoint resume and the shipping-default
  FP16 width-256 case.
- Full Hugging Face training over the registered families.
- Cached generation for Qwen2.5, Qwen3.5, SmolLM3, GPT-OSS and Qwen4.
- The independent OLMo beta-scale diagnostic; the documented beta-scale-2 repair
  is the only old-stable parity exception besides the GPT-2 repair.
- The complete 32-family PEFT matrix (LoRA, saved modules and adapter
  switching): zero failures.

The shipping bundle contains seven executables built into
`target/ally-results/shipping-qualified/app` with pinned source/executable
identities. `smoke_shipping_cli.py` passed ten application commands, including
recurrent train, mid-epoch checkpoint resume with all 290 tensors bit-identical,
chat after both exports, Transformer training, package reload and cached
generation. `gui-qualified/receipt.json` records the GUI crate tests and the
Falcon H1 workflow test, which drives the real `NativeApp` command builder and
the native CLI and reproduces the CLI weights bit-for-bit. No interactive window
session was automated; the test covers the application's command path, not a
human clicking through every screen.

## GPT-2 inference recovery

The previous qualification reported the tiny repaired GPT-2 fixture at 1.15x
stable (0.667 ms vs 0.581 ms, +0.086 ms) as the accepted cost of the
Welford/addmm numerical repair. The cost was the general Welford module's
per-row cascade: at widths up to 128 the reduction is a single 16-vector chunk,
but the module still carries the cascade stack and recomputes `1/n` by division
for every vector.

`transformer_layer_norm_welford_small_forward.comp` keeps the same association
without that overhead. One workgroup of 64 invocations owns eight rows, eight
lanes per row, and merges the eight per-lane moment streams in the general
module's order. Integer counts use a table of correctly rounded 1..16 values;
the scalar tail and the final `1/dim` keep `fp32_div`. Admission is exactly
`AMD Radeon Graphics [pci=1002:15bf;driver=00800184]` at hidden widths 1-128;
every other adapter, driver and width keeps the general Welford module, and
non-GPT-2 architectures keep the legacy LayerNorm module.

- All 512 geometries (widths 1-128 x rows 1/6/17/64) produce bit-identical
  mean, rstd and affine output to the general module.
- The unchanged independent CPU fixture passes all 15 widths: the ten widths
  up to 128 exercise the small module bit for bit, and 129-768 keep the general
  module.
- A fresh `glslc` v2023.8 build with the manifest's flags reproduces the
  committed SPIR-V byte for byte (sha256 `fd9acd8b...`).
- The admission test pins the boundary: widths 0 and 129+, any other adapter or
  driver string, are rejected.

The same follow-up moved the diagnostic `HIERARCHOS_VULKAN_VARIANT_TRACE`
lookup from every dispatch to kernel construction. It changes only the
diagnostic path, no arithmetic or route.

Complete-model paired medians (five alternating pairs, candidate/stable):

| Case | Before | After |
| --- | --- | --- |
| Tiny GPT-2 unmasked (0.58 ms) | 1.150 | 1.004 (0.997-1.007) |
| Wide GPT-2 unmasked | 1.017 | 0.915 (0.911-0.918) |
| Wide GPT-2 padded | 1.034 | 0.912 (0.901-0.919) |
| Wide GPT-2 decode | 1.032 | 0.848 (0.843-0.854) |
| Wide GPT-2 batch-32 | 1.023 | 0.952 (0.924-0.997) |
| Wide Llama unmasked | 1.017 | 0.997 (0.994-1.006) |
| Wide Llama padded | 1.009 | 1.000 (0.998-1.005) |

The accepted 13-stage harness repeats three runs per case and is noisier on the
0.58 ms fixture: tiny GPT-2 is 1.032 (0.982-1.060), wide 0.914/0.923, decode
0.856, batch-32 0.952, Llama 1.002/1.009. The old 15% penalty is gone; the tiny
case is now inside the noise band rather than measurably slower.

## Performance

Quiet, paired, serial measurements (`training-performance-qualified`,
`transformer-performance-qualified`, three repeats, three legs including
untouched automatic stable):

- 16 whole-RNN training cases: controlled median ratios 0.964-1.020 in the
  accepted quiet run; the largest is FP16-storage batch-32 at +2.0%, inside the
  observed run-to-run spread, and several masking/odd-sequence cases are 1-4%
  faster. A first run immediately after the long build measured one FP16 masked
  case at 1.078. That run is retained as
  `training-performance-interference`: its own unpatched `stable-auto` control
  ran at 248.5 ms against the candidate's 248.9 ms on that case, one
  transformer stable leg stalled to a 0.337 ratio, and the immediate rerun on
  the same binaries produced 1.006-1.014 for the case. The accepted receipt is the
  rerun; neither run is hidden.
- 7 Transformer inference cases: wide GPT-2 0.914/0.923, decode 0.856, batch-32
  0.952, Llama 1.002/1.009, tiny GPT-2 1.032 (0.982-1.060 at 0.58 ms). See the
  recovery section for the five-pair medians. No case is materially slower; the
  old 1.15x tiny-GPT-2 cost is gone.
- Dispatch profiles record 240 nonzero shader artifacts across 24 workload rows
  (`profiles-qualified`); instrumentation timings are excluded from acceptance.

## Workspace check

`check-workspace.ps1` builds a synthetic four-crate workspace without a
lockfile, so cargo resolved the newest `sysinfo` (0.39.6, requiring rustc 1.95)
against the installed 1.93 toolchain and the check failed with exit 101. The
committed failure is reproduced in
`target/ally-results/workspace-check-before-lock-fix.log` using the pre-fix
script. The check now seeds the generated workspace with
`hierarchos-vulkan/Cargo.lock`, matching the dependency graph of the qualified
binaries, and passes; cargo still resolves members that are absent from that
lock. No product source changed for this.

## Numerical contract

No tolerance, existing numerical golden or validation skip has been changed.
The short-width GPT-2 module is not a new tolerance: it reproduces the general
repaired module's bits on all 512 measured geometries and the independent CPU
fixture, and is admitted only on the measured adapter/driver and width range.
Missing device goldens were established from the isolated stable executable with
its source and executable identity recorded; candidate output never supplied an
expected value. The existing OLMo Hybrid beta-scale-2 repair and the GPT-2
Welford/addmm repair remain the scoped old-stable parity exceptions. Beta scale 1
still matches stable exactly, and repaired scale 2 is checked against the
intended Transformers implementation. GPT-2 matches the minimal repaired
reference exactly and passes the unchanged independent `2e-7` ceiling; the
legacy comparison remains visible in every receipt.

## Scope and limitations

- One physical adapter and driver (`1002:15bf`, `00800184`). Route keys include
  the device and driver, so these measurements cannot select routes on other
  silicon or drivers.
- The TCP tests are two processes on this machine. They do not certify two
  physical GPUs, a real network, or Windows RDMA (unavailable; TCP only).
- The GUI evidence is the application's own workflow test, not an automated
  interactive desktop session.
- Sustained firmware power limits and Armoury Crate operating modes were not
  changed or certified; measurements used the Turbo Windows scheme on AC.
- Evidence archives are working data. `complete-optimization-ledger.json` maps
  every reviewed source to its evidence and explicitly separates observed
  dispatches from host-only wiring and inactive experiments.

## Reproduce

```powershell
# From the candidate root, with the isolated oracle and validation environment:
. .\target\ally-results\environment.ps1
.\\.venv-vulkan\Scripts\python.exe hierarchos-vulkan\validation\qualify_handoff.py `
  --stable ..\Hierarchos-windows-oracle --expected-vendor amd `
  --output target\ally-results\qualification-small-welford
```

`--tune` measures and merges only this adapter's rows before repeating all 13
stages; a failure restores the previous tables and rebuilds their binaries.
Every stage writes its own log, and `qualification.json` is green only when all
13 commands exit zero.

## Evidence index

| Path | Contents |
| --- | --- |
| `qualification-small-welford/` | final 13-stage run after the GPT-2 module |
| `qualification-ship-posttune/` | tuned-route 13-stage run before it |
| `qualification-ship/` | pre-tuning run, tuning receipts, stale-test failure and rollback |
| `tuned-routes-ship/` | tuned route tables and tuning receipts |
| `independent-final-qualified/` | nine independent oracle stages and source identity |
| `additional-recurrent-qualified/` | 19 small-rank cases with exact digests |
| `shipping-qualified/` | seven-executable bundle and release library run |
| `application-qualified/` | ten application commands and exact resume comparison |
| `gui-qualified/` | GUI crate tests and Falcon workflow parity |
| `training-performance-qualified/`, `transformer-performance-qualified/` | accepted paired timings |
| `training-performance-interference/`, `transformer-performance-interference/` | excluded run, its `NOTE.json` and the reason |
| `gpt2-speed-recovery/` | prototypes, 512-geometry parity and five-pair model timings |
| `pre-small-welford/` | prior performance receipts and final review |
| `profiles-qualified/` | actual shipping dispatch profiles |
| `complete-optimization-ledger.json` | 156 reviewed sources across 19 groups |
| `workspace-check-qualified.log`, `workspace-check-before-lock-fix.log` | corrected and reproduced workspace checks |
