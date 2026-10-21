# Physical ROG Ally qualification

Status: **qualified on the tested device**, with the limitations listed below.
Every numerical gate was re-run on the final tuned route tables; the shipping
bundle, independent mathematical oracles and paired performance measurements are
recorded under `target/ally-results/` (ignored working evidence, not source).

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

The final run is `target/ally-results/qualification-ship-posttune`. It repeats
the complete 13-stage suite on the tuned route tables and passed all 13 stages
in 1,524 s:

| Stage | Result |
| --- | --- |
| Python contracts | 375 tests pass, the same 12 opt-in skips |
| All binaries | `cargo check --all-targets` passes |
| Library and trainer | 811 library tests pass (9 ignored), 40 trainer tests pass |
| Two-rank TCP | both modes, all 290 exported model/optimizer tensors identical |
| Inference / native CLI / GUI crates | all pass |
| Stable recurrent | 34 cases, 38,930 exact digests, 0 differences |
| Stable logits | 75 comparisons, 36,096 logits, 0 differing bits |
| Vendor composition | all four RDNA3 modules dispatched in the composed graph |
| CPU bridge | pass |
| Release training ladder | 12/12: 6,464 dispatches, 3,076 barriers, loss 5.543217, 32,162 digests |
| Debug training ladder | 12/12, same count, barrier bound and digests |

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

## Performance

Quiet, paired, serial measurements (`training-performance-qualified`,
`transformer-performance-qualified`, three repeats, three legs including
untouched automatic stable):

- 16 whole-RNN training cases: controlled median ratios 0.970-1.015 against
  clean stable; the largest slowdown is the FP16-storage long-128 case at 1.5%,
  inside the observed run-to-run spread. No case regressed materially, and
  several masking/odd-sequence cases are 1-3% faster.
- 7 Transformer inference cases: 1.009-1.033 for the wide Llama/GPT-2 fixtures,
  decode and batch-32. The tiny GPT-2 unmasked case is 1.15x (0.667 ms vs
  0.581 ms); the absolute difference is 0.086 ms and is consistent with the
  scoped Welford/addmm numerical repair that makes its logits match the pinned
  CPU association. It is reported as a known cost, not hidden.
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
  --output target\ally-results\qualification-ship-posttune
```

`--tune` measures and merges only this adapter's rows before repeating all 13
stages; a failure restores the previous tables and rebuilds their binaries.
Every stage writes its own log, and `qualification.json` is green only when all
13 commands exit zero.

## Evidence index

| Path | Contents |
| --- | --- |
| `qualification-ship-posttune/` | final 13-stage run on tuned routes |
| `qualification-ship/` | pre-tuning run, tuning receipts, stale-test failure and rollback |
| `tuned-routes-ship/` | tuned route tables and tuning receipts |
| `independent-final-qualified/` | nine independent oracle stages and source identity |
| `additional-recurrent-qualified/` | 19 small-rank cases with exact digests |
| `shipping-qualified/` | seven-executable bundle and release library run |
| `application-qualified/` | ten application commands and exact resume comparison |
| `gui-qualified/` | GUI crate tests and Falcon workflow parity |
| `training-performance-qualified/`, `transformer-performance-qualified/` | paired timings |
| `profiles-qualified/` | actual shipping dispatch profiles |
| `complete-optimization-ledger.json` | 156 reviewed sources across 19 groups |
| `workspace-check-qualified.log`, `workspace-check-before-lock-fix.log` | corrected and reproduced workspace checks |
