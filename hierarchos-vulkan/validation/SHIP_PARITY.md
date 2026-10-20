# Training optimization numerical sign-off — 2026-10-15

All numerical gates passed on the local **AMD Radeon Graphics** device in debug
and release builds. The optimizations were retained. No tolerances were widened
and the existing kernel goldens were not re-recorded in this sign-off.

| Check | Result |
| --- | --- |
| Direct comparison against original embedded shaders | 144 dense + 183 reduction + 79 attention invocations; all bit-identical |
| Full training trajectory against the pre-optimization implementation | 3,817 exact tensor digests over 13 optimizer steps; zero differences |
| Shader source/artifact provenance and SPIR-V validity | 86/86 affected variants reproduce byte-for-byte and validate |
| Debug and release verification ladder | All eight checks pass in both profiles |
| Library tests | 706 passed, zero failed, nine ignored |
| Trainer tests | 40 passed, zero failed |
| Eight-way gradient accumulation | Exact match to both four-way and scalar fallbacks across 12 tensor counts, tails and 256-element group boundaries |
| Verification negative controls | Parameter/moment drift, missing tensors/steps and wrong optimizer clocks are rejected |
| All binary targets | `cargo check --bins` passes |
| Standalone source package | All 62 Cargo target source paths exist after staging |

## Baseline and trajectory coverage

The baseline restores `vulkan.rs`, `rwkv_optimizer.rs` and 50 optimization-touched
shader artifacts from commit `5004a8a2923aa0ac89deeffa241f5d7d7f65edc1` in an
isolated source copy. It retains the CE/objective changes that existed before
this optimization task, along with the other current host code. Baseline and
candidate consume the same initialized model package for each direct comparison.
The original artifact hashes are in `release_parity/baseline_manifest.json`.

The trajectory fixtures cover batch 2 for six updates, batch 32 for four updates,
and batch 9 with two independent sequences and width 256 for three updates.
After every update, checks include every trainable parameter, both Adam moments,
global/per-slot optimizer clocks, token losses, final H/L recurrent states,
initial-state adjoints, and context carriers. Hashing uses the full raw FP32 bit
patterns and vector lengths; non-finite snapshots fail. State goldens were frozen
from the verified **baseline**, not from candidate outputs.

## Repairs to verification and packaging

The reduction harness's optional A/B mode compared baseline outputs with candidate
buffers before dispatching the candidate. All three affected paths now dispatch
first. The verification ladder exercises both A/B and frozen-golden comparison
on every reduction run, so this ordering regression cannot hide behind golden-only
CI runs. The corrected harness passes all 183 direct comparisons with HEAD.

The ladder formerly checked only the 32 CE variants for provenance. It now uses
`training_shader_manifest.json` to check all 86 affected variants with their actual
compiler/flags, including the latest low-rank producer and `gradient_accumulate8`.
The manifest records attention's glslang build, reduction optimization flags, and
the native FP16 specialization separately, and it records the release of each
compiler per artifact, so an artifact built by a different release of the same
compiler is compiled by that release instead of being reported as drift.

`training_step_probe --state-output FILE` adds untimed full-state snapshots.
`verify_training_state.py` compares those snapshots against the immutable baseline
goldens, and is now a mandatory ladder check. The standalone packager also includes
`validation/`, which Cargo's new attention harness target requires.

The ladder's two fixture records are both device-keyed, because the vendor layer
selects different kernels on different parts and neither a trajectory nor a
dispatch count transfers between them: `kernel_golden/training_state.json` carries
one trajectory per device, and `training_step_probe/step_expectation.json` now
carries `mean_loss` and `dispatch_count` per device with `probe_args` and the
barrier bound shared (the bound only widens). Adding a device with
`--record-expectation`, or a trajectory with `record_training_state.py`, merges
that device's numbers and leaves every other device's record alone; a device with
no entry fails rather than passing by default.

## Reproduce

From `hierarchos-vulkan`:

```powershell
python validation/verify_training_stack.py --release --receipt target/release_parity.json
python validation/verify_training_stack.py --receipt target/debug_parity.json
python validation/verify_training_state.py --record-coverage   # add this device's arm/case coverage
cargo test --lib --bin hierarchos-vulkan-train -j 1 -- --test-threads=1
python -m unittest discover -s validation -p test_training_state.py -v
python -m unittest discover -s validation -p test_training_stack.py -v
cargo check --bins
```

Run the two ladder profiles sequentially. New hardware requires independently
qualified device goldens; an unrecorded device fails the gate. Do not re-record
expectations to make an unintended numerical change pass.

Machine-readable receipts, direct comparison logs, full test output and a hash
inventory of the tested sources/artifacts are under `release_parity/`.

The subsequent [attention-backward performance fix](ATTENTION_BACKWARD_PERFORMANCE.md)
retains the same output goldens and numerical trajectory: invocation-private
caches remove the unnecessary 14,400-byte shared-memory reservation and close the
measured slowdown at larger batches. The shader provenance gate now also checks
that this kernel keeps its single-invocation ABI and zero workgroup variables.

This sign-off establishes numerical parity for the tested fixtures on this AMD
device. Cross-vendor parity remains unverified. Long strict-backward benchmark
jobs can still hit this driver's watchdog with either artifact. The nine
explicitly ignored library tests were not run.

## Variant equivalence and coverage (2026-10-16)

The LN1/low-rank producer's topology is scheduling and not numerics, and the gate now
covers both kernel families. The FP32 suite is unchanged: 3,817 exact tensor digests
over 13 optimizer steps, run once for the shipped choice and once for each forced
variant (`one-row`, `two-rows`, `column-slices`) against the frozen pre-optimization
baseline.

The packed-FP16 producer is the kernel the FP16-storage policy actually dispatches, and
no arm of the FP32 suite can reach it. `kernel_golden/training_state_fp16.json` covers
it: three cases (batch 2 and batch 32 at width 128, plus a width-32 fixture), 14
optimizer steps and 4,088 exact tensor digests, recorded under
`HIERARCHOS_VULKAN_TRAINING_PRECISION=fp16-storage-fp32-compute` -- the policy the
golden names in its own `env` block, which both the recorder and the verifier read, so
a trajectory cannot be recorded under one policy and replayed under another. It is not
a pre-optimization artifact: it pins the six packed variants to each other, while the
FP32 golden keeps pinning the trajectory to the earlier baseline. Measured on the Intel
Gen9 target, `one-row` and `one-row-cached` run in all three cases, `two-rows-wide` in
both width-128 cases, `two-rows` and `four-rows` in the width-32 case, and
`one-row-rank128` -- gated on a rank above 64, which this part's shapes do not reach --
in none. That last one is a note and not a pass, and the record says so: its verdict is
`unavailable` in the case its canary re-checks and `asserted` in the two the canary
stands in for, so the record still documents all three.

`validation/training_state_coverage.json` records exactly that per device and suite,
along with what each golden case compares, and a run fails when any of it shrinks: an
arm that stops running where it ran, a case that is no longer measured, or a case that
compares fewer steps or tensors than it did. A device with no entry fails rather than
passing by default, and `--record-coverage` merges this machine's evidence in, refusing
to shrink an existing entry unless `--allow-coverage-shrink` says the loss is intended.
Both halves were proven by making them fail first: the run before the record existed
reported `no coverage recorded for this device` for both suites, and a doctored record
claiming `column-slices` also ran in `odd_multi_sequence` plus a `ghost_case` failed
with exactly those two messages while all 15,268 digests still matched.

The packed family is deliberately **not** in `topology_decisions.json`, and that is a
fact about how the record is written rather than about the family: the replay that writes
it runs the probe at the default FP32 precision policy, where the packed producer is
never dispatched, so the producer emits no `ln-low-rank-fp16` decision line for the
record to hold (the replay's own scratch record is eight `ln-low-rank-f32` lines and
nothing else). Its choice used to be a coin flip under the median-only rule -- five
consecutive width-128 runs selected `one-row-cached` three times and `one-row` twice,
with the cached arm's lead between 0.6% and 3.6% against the 2% material margin -- and
under the unanimous-round rule every run since has kept `one-row`, so what ships is the
structural default with every variant proven bit-identical. The end-to-end effect of the
choice is below this measurement's resolution anyway: shipped 361.2 / 376.3 ms per
step at batch 2 against `one-row` forced 367.6 / 381.0 ms, and 923.0 / 926.2 against
930.2 / 941.4 at batch 32 -- never slower, and smaller than the repeat-to-repeat spread
of the step itself.

Verified on the Gen9 target: `validation/verify_training_stack.py` **10/10**, with the
state step reporting `2 suite(s), 29 run(s), 32162 exact tensor digests compared;
coverage audited against validation/training_state_coverage.json`; `--topologies` with
no values passes both suites with `coverage not audited`; 109 validation unit tests OK;
`cargo test --lib` 736 passed / 0 failed / 9 ignored.

## The trajectory step without the runs that cannot show anything (2026-10-16)

Every arm used to replay every case of both goldens, including the cases where the
variant is not there on this device: `one-row-rank128` was forced in all three FP16
cases and reported unavailable in each, `two-rows` and `four-rows` were forced in the
width-128 cases where only the wide pairing exists, and all three FP32 arms were forced
in the width-256 `odd_multi_sequence` case, where the FP32 producer is not in the shape
at all and the override changed nothing. Those four full replays could only re-confirm a
verdict the record already holds -- one probe run of a case is 10-26 s of wall time, most
of it model load and pipeline setup that has nothing to do with the arm.

The record now holds two absence verdicts per (arm, case) -- `unavailable` (the producer
rejected the request) and `not_dispatched` (the producer never saw it at that shape) --
and a forced arm runs: every case the record has it verified in, every case the record
says nothing about, and one **canary** per arm. The canary is the widest absent case,
replayed for a single update, because existence is settled by the first dispatch and
there is no trajectory to compare in a case the variant never ran in; the remaining
absent cases are written to the record as `asserted`, carried rather than re-measured,
with the canary as their re-check. A canary that disagrees (the variant runs where the
record says it is absent) makes the run measure every carried case of that arm in full
instead of trusting the record, so a device or a gate that changes restores the coverage
it invalidated in the same run.

Each case's model package is also shared by every arm now instead of being re-created
per arm: the probe initializes a package when it is missing and only ever reads it, which
was verified directly -- the package's content hash is identical before and after two
consecutive six-update runs, and the two snapshots are byte-identical files. Initializing
it is measurable: the same case took 10.85 s from a fresh directory and 9.64 s from the
one that already existed.

Measured on the Intel Gen9 target, debug probe, both suites, the two runs back to back in
one session so the machine's regime is shared: **33 runs / 346.97 s against 29 runs /
281.02 s (-19.0%)**, with the same trajectories, the same coverage audit and every digest
matching. The pre-change side is measured the same way the change is, by pointing the
verifier at an all-absent-verdicts record, which makes every arm replay every case
exactly as it used to; it reports the old `33 run(s), 43884 exact tensor digests`.

The row-level digest count drops 43,884 to 32,162 because a canary compares one update:
the fallback trajectory it stands in for is still compared in full by the measured arm
and by every arm that actually dispatches in that case, so no unique evidence is lost,
and the record's per-case counts still come from the full measured run.

The FP16 decision that used to be a coin flip is now unanimous or nothing: thirty-five
consecutive width-128 runs over seven interleaved rounds each kept the structural default
`one-row` (thirty, then five), where the last five measured `one-row-cached` at 0.27299 /
0.27492 / 0.27496 / 0.27356 / 0.27642 ms against `one-row` 0.28326 / 0.27960 / 0.28728 /
0.28150 / 0.28495 -- a 1.7-4.3% median lead that nevertheless won only 4, 3, 3, 5 and 4
of the seven rounds, so no candidate was ever unanimous. Before the rule, three of five
identical runs had selected the cached arm. `validation/topology_decisions.json` records
the rounds and the narrowest per-round lead for the decisions that are stable, and
`verify_topology_decisions.py` audits that evidence as well as the choice.

The kernel-golden coverage record was re-proven on the harnesses after the wiring landed.
The reduction harness exits 1 on a record claiming `adamw len=1 step=ghost` -- `Error:
reduction on Intel(R) HD Graphics 520: invocation adamw len=1 step=ghost was covered and
this run did not exercise it`, printed after its 183 GPU cases still reported bit-identical
outputs -- and exits 1 on a record with this device's entry removed -- `no coverage
recorded for this device; run with --record-coverage on this machine to add its entry`.
The dense and attention harnesses fail the same way on a doctored record (`Error: dense on
Intel(R) HD Graphics 520: invocation gemm rows=8 ghost was covered and this run did not
exercise it`, `Error: attention on Intel(R) HD Graphics 520: invocation forward_cache_1024
backward=ghost was covered and this run did not exercise it`), so the rule is wired into
every harness rather than into one, and all three exit 0 against the checked-in record,
reporting 144, 183 and 79 invocations audited.

## A new device bootstraps instead of failing, and both records answer from one implementation (2026-10-16)

The coverage rule is only evidence while a device with no entry is a failure, so the
default stays fail-closed. `--bootstrap-coverage` is the deliberate door: it creates only
the entries this run is missing, prints the diff it wrote, never rewrites an entry that
exists, and cannot be combined with `--record-coverage` or `--allow-coverage-shrink`.
Proven on the GPU: `matmul_kernel_validation --golden validation/kernel_golden/dense.json
--coverage <scratch> --bootstrap-coverage` printed `bootstrap added 1 coverage entry (0
already recorded)`, `+ dense on 'Intel(R) HD Graphics 520': 144 invocation(s)`, passed, and
the file it wrote audited clean on the next run; the same run without the flag exited 1
with `... is missing; create it with --record-coverage on this machine, or
--bootstrap-coverage to create only the entries this run is missing`. The trajectory
driver did the same for one suite: `--suite fp32 --bootstrap-coverage` added
`fp32 3 case(s), 3 arm(s)`, then passed with `1 suite(s), 12 run(s), 15268 exact tensor
digests compared`.

Both records now share `src/coverage_record.rs` for the envelope, the audit, the merge,
the shrink refusal and the bootstrap; each contributes only `diff` (what a loss is for one
of its entries) and `summary`. The Python driver no longer has an audit or a merge: it
asks the engine through the `coverage_record` tool built beside `training_step_probe`, and
`validation/test_training_state.py` lost the 20 tests that pinned the Python copy.
`cargo test --lib coverage_record` runs the single suite: 26 tests over the kernel rules,
the trajectory rules, the merge, the refusal wording, the envelope and the bootstrap.
Ladder: **10/10**, state step `2 suite(s), 29 run(s), 32162 exact tensor digests compared;
coverage audited against validation/training_state_coverage.json`.

## Final push pass (2026-10-17)

Re-run on the Intel Gen9 target against the uncommitted tree, with the stable oracle at
`..\Hierarchos-Native-stable\Hierarchos-Native` (commit `58632d3`, clean). This is the
before-push check, so it re-runs the gates rather than citing the records above.

| Check | Result |
| --- | --- |
| `verify_training_stack.py` (debug) | **12/12** steps PASS |
| `verify_training_stack.py --release --skip-build` (release bins built first, exit 0) | **11/11** steps PASS, the same numbers as debug |
| `cargo test --lib` (debug) / `cargo test --release --lib` | 762 passed / 0 failed / 9 ignored in both |
| `cargo test --release --lib vendor::` | 23 passed / 0 failed |
| `cargo check --bins` | exit 0 |
| Python validation suite | 129 tests, 1 error (`test_peft_oracle_provenance`, `torch` not installed), 1 skip |
| `check_oracle_parity.py` | 810 portable files compared, 3 documented divergences, 0 missing, 20 only here, 19 vendor files excluded |
| `check_shader_provenance.py` / `check_push_constant_layouts.py` | 467/467 artifacts; 356 push blocks / 109 host structs, PASS |
| Kernel goldens | 144 dense + 183 reduction + 79 attention verified, coverage audited per device |
| `training_step_probe` (ladder args) | `mean_loss 5.5432177`, 6368 dispatches, 2980 barriers (bound 3076) |
| Training trajectories | `2 suite(s), 29 run(s), 32162 exact tensor digests`, both profiles |

**The other device's record is unchanged.** Comparing the modified goldens against
`HEAD` value-by-value: `dense.json`, `reduction.json` and `attention.json` each add
exactly one entry per invocation for `Intel(R) HD Graphics 520` (144/183/79 added) and
change or remove none; `training_state.json` adds 5,078 values and changes or removes
none. Zero AMD digests moved, which is the regression statement the other machine's
checklist is looking for.

**One defect found and fixed by this pass.** The budget test
(`attention_backward_budget_matches_the_measured_boundary`) carried a duplicated
`#[test]` attribute, because the doc comment for
`non_intel_devices_keep_the_portable_modules` had been stranded above it. The harness
therefore registered the budget test twice: `cargo test` reported 763 and `vendor::`
reported 24, one higher than the 762 / 23 unique tests that exist. The duplicate
attribute and the two needless `let mut` closures beside it are gone, and the stranded
doc comment is back on the test it describes. That cleanup also exposed the only other
warnings this tree produced -- two unread `vendor.rs` items -- and both are now
`cfg(test)`-only, because nothing outside the tests reads them. The eight warnings
`cargo test --lib` still prints (one `unused_mut` in the Falcon H1 config parser, seven
`dead_code` items) and the extra `dead_code` `cargo check --bins` sees when `cfg(test)`
is off are all in code this tree does not touch.

**Not re-measured here.** The AMD adapter is not present, so its timings and its own
goldens stay as recorded; `verify_vendor_parity.py` and the headline-logits runs are
carried from the qualification above rather than repeated. Every runtime gate that *can*
run on this machine was re-run rather than cited.

## Transformers logit parity re-run (2026-10-17, checkout `4692303`)

Re-run after the Gen9 tuning against the local Transformers checkout at
`C:\Users\User\transformers` (`5.19.0.dev0`, commit `4692303`, clean), so the registry
was measured against the current reference rather than the earlier `5.16.0.dev0`. This
supersedes the carried headline-logits note above.

| Check | Result |
| --- | --- |
| `verify_hf_logits.py` (full registry) | 75/75 comparisons, 0 failing values, worst `1.788139343e-07`, gate `atol=2e-7`, `rtol=0` |
| ... of which the 37 original families | 71 rows unchanged at worst `5.960464e-08` |
| `verify_falcon_h1.py` (forward) | 27/27 comparisons over 3 variants, 0 failing, worst `8.940696716e-08`, gate `2e-7` |
| `verify_hf_logits.py` with `HIERARCHOS_VULKAN_FORCE_VENDOR=amd` | 75/75 comparisons, every `(max_abs, failing_values)` row identical to the resolved run |
| `verify_vendor_parity.py` (full registry, both paths) | exit 0, 75 resolved / 75 forced-portable rows identical, worst `1.788139343e-07` |
| `verify_hf_training.py` (all 34 non-encoder-decoder fixtures) | 34/34 at two AdamW steps, worst `1.192092896e-07`, gate `2e-7` |
| `verify_falcon_h1.py --training --generation --keep-going` | exit 0, 3 variants: forward logits `4.470348e-08` worst, adamw `1.192093e-07`, dual `1.220033e-07`, generation `4.470348e-08` |
| ... cached decode at prefill 1/4/7/9 | cached vs full-prefix identical (`0.0`) at every token; cached vs HF worst `5.960464e-08` |
| `python -m unittest discover -s hierarchos-vulkan/validation -p "test_*.py"` | 130 tests, OK (1 skipped), exit 0 |
| `cargo test --release --lib vendor::` | 23 passed / 0 failed on this tree |

## RWKV cell CPU bridge (2026-10-17)

Hierarchos is its own architecture, so the coherent-v9 RWKV cell has no Transformers
oracle. `validation/verify_rwkv_cpu_bridge.py` builds that oracle out of the other
native implementation: it generates one deterministic tiny coherent-v9 package
(`width=128`, `head_size=16`, `batch=4`, neutralised DeepEmbed adapter), then steps the
same package through `hierarchos-vulkan-rwkv-packed-cell-step` and through
`hierarchos-inference`'s own loader and cell (`rwkv-cell-probe`). Each leg feeds its own
packed state into the next of three steps, so the check covers the recurrent carry and
the `[width, 4 + head_size]` explicit-output state ABI, not just one dispatch.

| Check | Result |
| --- | --- |
| `verify_rwkv_cpu_bridge.py`, `state_clamp=0.0` (disabled) | 3 steps, 6 comparisons, exit 0, worst `7.6e-6` output / `7.6e-6` packed state |
| ... `state_clamp=50.0` (production `recurrent_state_clamp`) | identical rows, worst `7.6e-6`; the clamp is asserted not to bind |
| ... `state_clamp=0.25` (active) | worst `7.6e-6` output / `3.01e-6` packed state; both legs asserted to peak exactly on `0.25` |
| Negative control (CPU input scaled by `1.01`) | rejected at `8.054e-3` against a `1.21e-4` bound, so the comparison is not vacuous |
| `cargo test --manifest-path hierarchos-inference/Cargo.toml` | 13 passed / 0 failed, including the new non-positive-clamp test |

Gate: `atol=1e-4` + `rtol=1e-5`, loose enough for fp32 accumulation-order differences and
still four orders of magnitude below any structural disagreement (the first run of this
harness, against a CPU clamp that read `0.0` as "clamp to zero", reported `4.25`).

**What this does not cover.** The packed-cell runner traces the resolved plan and its
real dispatches, so the harness below qualifies the tuned Intel Gen9 and AMD RDNA3
`linear*` modules on the packed cell itself.

## RWKV packed-cell vendor-linear qualification (2026-10-17)

`RwkvPackedCellOp` composes its channel mix from the vendor
`linear-forward`/`linear-residual-forward` modules and, where the tuned module is
admitted, from `linear3-forward`. The fused packed-cell shader that used to bypass
the channel-mix modules was removed after it lost both the training pass and the
TBPTT forward-only transition. The fused *time-mix projection* was kept in part:
an A/B found the `linear3-forward` composition `0.960-0.978x` the fused kernel at
batches 8/16/32, where the tuned module is admitted, and `1.006-1.009x` at
batches 1/2/4, where it would fall back to the portable module. The packed cell
now composes exactly where the module's own admission gate (the predicate its
dispatch already uses) admits it and keeps the fused kernel elsewhere; both
graphs are bit-identical, so the change is scheduling. Three switches force each
graph in the qualification legs:
`HIERARCHOS_VULKAN_DISABLE_TIME_MIX_FORWARD_FUSION` drops the fused time-mix
projection, `HIERARCHOS_RWKV_TIME_MIX_FORCE_FORWARD_FUSION` restores it at every
batch, and `HIERARCHOS_VULKAN_DISABLE_CHANNEL_MIX_FORWARD_FUSION` reaches
`linear-forward`/`linear-residual-forward`.

`validation/verify_rwkv_packed_cell_composition.py` runs the tuned and portable
modules over the same generated package (`width=128`, `head_size=16`, `batch=8`,
live DeepEmbed adapter), steps them three times with each leg consuming its own
packed state, and compares `output`, `packed_new_state`, `grad_x`,
`grad_packed_state` and `token_feature_grad` at every step. Batch 8 is required: at
the bridge's four rows the Gen9 low-row gate (`output_dim >= 192`) leaves `value_cm`
(128 outputs) on the portable module, and the harness would not be comparing the
vendor graph. The portable leg must dispatch nothing and the vendor leg must
dispatch all three families, so the two legs cannot silently collapse onto one
graph.

| Check | Result |
| --- | --- |
| `verify_rwkv_packed_cell_composition.py` | 3 steps, 15 comparisons, exit 0, worst `0.0` on every tensor |
| ... vendor dispatch | `linear-forward=1 linear-residual-forward=1 linear3-forward=1` per step; portable leg empty |
| ... vendor plan | `linear-forward=intel-gen9-lds linear-residual-forward=intel-gen9-lds linear3-forward=intel-gen9-lds` on this host |
| ... vendor qualification (`--expected-vendor intel`) | exit 0; the receipt records `expected_vendor=intel`, `qualified_vendor=intel` and the three resolved variants |
| ... wrong-vendor guard (`--expected-vendor amd` on this host) | exit 1: `linear-forward resolved 'intel-gen9-lds', not an amd module; this host cannot qualify the amd path` |
| `test_rwkv_composition_receipt.py` | 7 tests OK, 1 skipped (live GPU leg gated by `HIERARCHOS_RUN_GPU_VALIDATION=1`) |
| Negative control (`x` scaled by `1.01` on the vendor leg) | rejected at `1.586200e-4` against a `1.038859e-4` bound |
| Removed packed-cell fused A/B | fused lost by 4% on `forward_backward` and 11% on the TBPTT forward-only transition at batch 8 |

The comparison is bit-exact, not merely within tolerance: the vendor modules keep
the portable modules' lane-serial accumulation order, so no reordering drift
appears. That is the strongest form of the agreement, but it also means this check
cannot tell a genuine `0.0` from two bit-identical implementations of the same bug.
The CPU bridge remains the independent oracle for correctness; this check closes the
**vendor-linear coverage** gap, not the numerical-oracle gap.

The recorded evidence is also a test: `test_rwkv_composition_receipt.py` fails the
ordinary `unittest discover` run if the receipt loses a vendor dispatch family, a
vendor-specific plan, a forward-fusion switch or the negative control. It reads
evidence rather than re-running the GPU; `HIERARCHOS_RUN_GPU_VALIDATION=1` enables
the live leg, which is what catches a code regression rather than a stale record.

The removal decision came from one process building both packed-cell graphs and
interleaving them at batch 8: `forward_backward` medians were `4.039 ms` fused vs
`3.871 ms` composed, and the TBPTT forward-only transition was `1.870 ms` vs
`1.664 ms`, bit-identical in both runs. The benchmark binary and receipt were
retired with the kernel; the table and caveats are in
[PACKED_CELL_COMPOSITION.md](PACKED_CELL_COMPOSITION.md).

**What this does not cover.** The adapter and all backward dispatches are shared
between the two legs, so their gradients agree by construction once the forward
activations agree. This host resolves the Intel Gen9 modules only:
`--expected-vendor amd` is implemented and fails closed here, but the RDNA3
`linear*` modules still need the same run on an AMD adapter (and
`HIERARCHOS_VULKAN_FORCE_VENDOR` is rejected precisely because it would name those
modules without the hardware).

All 14 headline architectures are covered: the 13 the registry carries
(`deepseek_v4`, `phi4_multimodal_text`, `phi3`, `kimi_k25_text`, `kimi_k3_text`,
`gpt_oss`, `smollm3`, `qwen3_5_*`, `qwen4_exp_*`, `mistral4`, `minimax_m3`, `gemma4`,
`minimax_m2`) plus Falcon H1 through its own harness. Two-step AdamW parameter parity
passes for every one of them.

**Gate-width fixtures added this pass.** `gpt2_wide` and `llama_wide`
(`hidden_size` 64, `intermediate_size` 256, `vocab_size` 320, 2x6 token batch) clear the
Intel Gen9 gate (`rows >= 4`, `input_dim >= 32`, `output_dim >= 128`), so their attention,
MLP and LM-head projections dispatch the tuned modules and are compared against
Transformers there. Bias (`gpt2_wide`, `linear-bias-forward`) and no-bias
(`llama_wide`, `linear-forward`) are both represented. Worst drift is `1.788139343e-07`
on these four rows; the other 71 rows are unchanged.

**Dispatch is now asserted at runtime, not inferred from the plan.** The geometry gate
can reject a dispatch the resolved plan selected, so `src/vendor.rs` keeps its
per-family dispatch counter in release builds and `transformer_logits` prints it when
`HIERARCHOS_VULKAN_VARIANT_TRACE` is set. The harness reads both the plan and the
counters for the gate-width fixtures and enforces the pair: a family the plan resolves to
a vendor module must be dispatched, and a family the plan resolves to `portable` must be
dispatched zero times. That second half is what the forced-portable run checks, and it
also keeps the assertion valid on adapters that have no module for the family. Recorded
counts: `gpt2_wide` = `linear-forward` 1 / `linear-bias-forward` 4, `llama_wide` =
`linear-forward` 5, on both cases; the forced run plans `portable` and dispatches
nothing. `cargo test --release --lib vendor::` still passes 23/23 with the counter always
compiled.

`linear3-forward` and `linear-residual-forward` remain reachable only from the RWKV
graphs, so those two tuned families are still covered by the bit-exact A/B test and the
`vendor_kernels_match_portable` dispatch assertions rather than by HF logits.

**Falcon's dual-AdamW fixture was re-conditioned.** That trajectory previously used a
mid-sequence mask gap (`[[1,1,0,1,1,1,1]]`) and failed on `gated_after` at
`model.layers.0.mamba.out_proj.weight` (`8.67992639541626e-07`). The failure reproduced
with `HIERARCHOS_VULKAN_FORCE_VENDOR=amd`, so no tuned module was involved, and the
reference could not measure it either: HF's own fp32 and fp64 backward passes differ by
`4.209578e-07` on that parameter, while the native fp32 result is `4.470348e-07` from the
fp64 answer -- both implementations the same distance from the fp64 value in opposite
directions, which is what produced the `8.68e-07` mutual difference. The fixture now
uses trailing padding (`[[1,1,1,1,1,0,0]]`) with the padded labels masked, so it still
trains through a masked batch; the same trajectory now measures `1.192093e-07` on
`gated_after` and at most `1.220033e-07` across the three variants, and the full
`--training --keep-going` run exits 0. The mid-gap numbers are recorded in the harness
comment: that case is no longer scored because the 2e-7 gate cannot resolve it, not
because it was made to pass.

**What the wide fixtures do not cover.** `linear3-forward` and
`linear-residual-forward` are only dispatched from the RWKV graphs, which the
Transformer harness does not build; they remain covered by
`intel_variants_are_bit_exact_with_portable_kernels`, which passed here at
`(8,448,512)` and `(6,96,320)` and at geometries the gate rejects.

## Packed TBPTT chains under FP16 storage (2026-10-19)

The two packed TBPTT chain A/Bs are now runnable under the training graph's
parameter-storage policy: `bench_rwkv_tbptt_packed_chains.py
--parameter-storage fp16-storage` installs the cell's packed FP16 execution
mirrors on all three legs, refuses to report if they did not activate, writes
the policy's own receipt, and records its verdicts under that scope. The
forward A/B stays bit-identical under the mirrors (`worst_max_abs` exactly zero
over all five returned tensors at every batch) and repeats the FP32 crossover --
`legacy` ahead by 3.8%/5.3% at batches 16/32 -- so the `fp16-storage` policy
now adopts its own measured `legacy` verdicts there and neither chain table
carries a `withheld` entry. The refusal mechanism remains for a policy whose
trajectory golden pins the structural graph, and the contract suite keeps it
exercised by withdrawing the FP16 measurements in a fixture.

| Check | Result |
| --- | --- |
| `verify_training_stack.py --receipt validation/training_stack_receipt.json` (debug, current bins) | **12/12** steps PASS |
| Kernel goldens | 144 dense + 183 reduction + 79 attention, unchanged |
| Shader provenance | 464 variants match their sources and flags |
| `training_step_probe` | `mean_loss 5.5432177`, 6512 dispatches, 3021 barriers (bound 3076) |
| Topology decisions | 4 measured decisions match the record |
| Training trajectories | 2 suites, 29 runs, 32162 exact tensor digests, including the FP16-storage golden |
| Chain receipts | FP32 and FP16 A/Bs both pass the route guard; FP16 forward scoped entries at batches 1, 16 and 32 |
| Backward-fusion A/B | all five batches bit-identical; the fully staged three-dispatch chain, the autotuner's two-dispatch one and the fully folded rung all measured at every batch; `fused` recorded where it led (batches 4 and 32), 1/8/16 withdrawn as three-way ties inside the band |
| Receipt guard | `check_training_stack_receipt.py` holds the committed 12 steps, the recorded adapter, the probe loss and 10 coverage floors |

The FP16 trajectory suite passing after the route change is the confirmation
that the lift is safe rather than merely measured: the golden was recorded on
the structural packed-first-forward route, and the FP16 receipt shows the
`legacy` replay reproduces that graph bit-for-bit at the geometries where the
policy now adopts it.

The reverse pass's state-backward fusion is the third chain on the same driver.
It holds the projection order and low-rank fan-in fixed and times the autotuner's
schedule against the most decomposed executable chain. The topology labels name
stages: `+` still dispatches, `-` is folded into the kernel, so
`rkv-add3+key+reduce` is fully staged and compiles at every batch; so does
`rkv-add3-key-reduce` now that the guarded cross-batch fold exists, whose direct
parameter-gradient writes otherwise need one row per channel. The selection
table had been ordered most-fused-first, so the staged arm resolved to the most
folded kernel instead of the shipped schedule, and while that kernel was
batch-1-only the sweep reported batches 4-32 `unmeasured`. Ordering it stage-first makes the A/B measure
the three-dispatch chain against the two-dispatch one at every batch, with
`worst_max_abs` exactly zero. Fused holds at batches 1 and 4 (1.1%/0.5%); 8-32
are inside the 2% band, and `tbptt_backward_fusion_decisions.json` records the
fused verdict at all five without moving the shipped schedule. The FP16 pass
measures fused ahead at 1 and 4 and staged ahead by at most 0.6% at 8-32, all
inside the band, so the scoped table keeps the shared verdict.

The sweep also times the fully folded rung at every batch, so all three fusion
depths are measured rather than inferred. That rung used to be batch-1-only: its
direct `grad_k_k`/`grad_k_a` writes assume one row per channel. The guarded
`FUSE_KEY_PARAM_CROSS_BATCH` specialization stores each workgroup's partials,
publishes an arrival through an atomic per-head counter, and lets the workgroup
that completes a head's last batch reduce that head's partials in ascending
batch order -- the order `rwkv_key_transform_param_reduce.comp` uses -- before
clearing the counter. The driver pins it for the duration of the sweep, and the
cell compiles those kernels either for that opt-in
(`HIERARCHOS_RWKV_CROSS_BATCH_KEY_PARAM_REDUCE`) or for a record that can
actually select `folded` on this adapter and width, so the kernel set follows
the record.

The folded rung is a routing candidate, and its acceptance gate is the fold's
own output. `grad_x` never consumes `grad_k_k`/`grad_k_a`, so the driver reads
the reduction back after the sequence and requires it to equal the two-dispatch
chain exactly: the folded rung's `grad_k_k`/`grad_k_a` are bit-identical at all
five batches, and the staged arm's gradient magnitude (`9.4e-3` at batch 4) is
recorded beside the zero so the agreement cannot be zero-vs-zero. Only a
geometry that clears it lists `folded` in its entry; the returned tensors are a
separate bound, because the contract associates one FP32 product differently
and moves `grad_x` by at most `2.4e-7` (a couple of ulps), so the receipt holds
them to a `1e-6` allowance while the routing pair must stay exactly
bit-identical. One identical AdamW step on each leg then diffs the parameter
snapshots as a second gate; the delta is at most `7.5e-8` against a `1e-6`
window, but that gate is the weaker of the two: AdamW's first update is
sign-normalized, so a fold that drops a batch row changes a gradient's
magnitude without changing the update, and only the direct readback covers the
reduction itself.

Recorded with all three candidates, the depths are a three-way tie on this
adapter: in the committed receipts the shipped schedule is fastest at six of the
ten measured geometries, the staged chain at two and the folded rung at two, and
no winner is more than `0.4%` ahead. So
`tbptt_backward_fusion_decisions.json` keeps `fused` at batches 4 and 32 -- where
the autotuner's schedule led -- and withdraws 1, 8 and 16 as ties inside the
band. The FP16-scoped table keeps one entry, batch 32. No entry names `folded`:
the route is recordable, and the measurement says there is nothing to gain by
recording it here.

**The promoted route was exercised, not only recorded.** A route that no
measurement on this adapter selects would otherwise be dead code with a passing
guard, so the committed record was temporarily patched to name `folded` at batch
4 and the driver re-run against it. The receipt then reported
`default_route folded`, `default_schedule` equal to the folded leg's
`rkv-add3-key-reduce` label, and the key-parameter readback at `0.0` again, so
the record's verdict reached the graph through `folded_backward_schedule`,
`apply_folded_backward_fusion` and the cell's schedule setter. The patch was then
reverted and the record re-verified (`check_measured_producer_routes.py` and the
297-test suite below).

That check found one real defect: the recorded verdict is applied at the start of
*every* sequence, and the first implementation only looked for a rung strictly
more fused than the current one. The second sequence therefore found nothing
above the folded depth it had just installed and failed closed, so a recorded
`folded` route would have broken on its second run. `folded_backward_schedule`
now returns the installed schedule when it is already the folded rung and only
searches otherwise.

`--negative-control` proves the gates reject such
 a fold on both publication paths. Above batch 1 it seeds one phantom arrival per
 batch row, so no arrival can complete a head; at batch 1 the rung writes the
 gradients directly, so the control redirects those writes into a scratch buffer
 instead. Either way it builds clean and corrupted pairs in one process and
 requires the clean pair to pass both gates and the corrupted pair to fail both
 (key-parameter delta `9.4e-3` at batch 4 and `8.1e-3` at batch 1, post-step
 delta `1.0e-4`). The driver applies the same two gate
functions to the control's rows, so the artifact under test is the receipt's
own guard.

The committed ladder receipt is now a checked artifact rather than a log.
`check_training_stack_receipt.py` runs on the CPU in `workspace-check` and holds
`training_stack_receipt.json` to the ladder's twelve steps in order, every one
`PASS`, the adapter both device-aware steps name, the probe's loss value and
the coverage floors -- matched shader variants, comparison modules, golden
invocations (144/183/79), backward topology decisions and the training-state
suites, runs and exact tensor digests. A dropped or renamed step, a non-PASS
row, a receipt re-recorded on another adapter, a changed loss, a detail line
that stops reporting a count, or coverage below a floor all fail. `--compare`
applies the same rules to a fresh ladder run and additionally refuses any count
below the committed one; `--record` adopts a fresh run only after those checks
pass. The floors live in the guard, so hand-editing the receipt cannot lower
them.

## Time-mix promotion ladder (2026-10-18)

The time-mix projection is now promoted per geometry from a recorded per-device
measurement (see `PACKED_CELL_COMPOSITION.md`), so the full ladder was re-run on
the promoted default rather than trusting the bench alone.

| Check | Result |
| --- | --- |
| `verify_training_stack.py` (debug, current bins) | **12/12** steps PASS |
| `verify_training_stack.py --release --skip-build` (release bins built first, exit 0) | **11/11** steps PASS |
| Kernel goldens | 144 dense + 183 reduction + 79 attention, unchanged |
| Shader provenance | 463 variants match their sources and flags (was 467) |
| Oracle parity | 805 portable files, 3 documented divergences, 5 retired, 20 only here |
| `training_step_probe` | `mean_loss 5.5432177`, 6512 dispatches, 3020 barriers (bound 3076) |
| Topology decisions | 4 measured decisions match the record |
| Training trajectories | 2 suites, 29 runs, 32162 exact tensor digests |

**Every receipt that moved, and why.**

* `training_step_probe/step_expectation.json`, Intel entry: 6368 -> 6512
  dispatches and 2980 -> 3020 barriers, with `mean_loss` unchanged at
  5.5432177. The composition dispatches `linear3-forward` plus the key/state
  kernels where the fused projection was a single dispatch; 144 extra dispatches
  over 72 layer-timesteps is exactly that difference. The barrier bound (3076)
  did not move.
* `oracle_divergences.json`: five `state: "removed"` entries for the retired
  packed-cell channel-mix/state kernel and its four artifacts.
  `check_oracle_parity.py` now distinguishes removals from modifications and
  rejects a stale entry of either kind, so a retirement is documented and a
  reappearing file still fails.
* Shader provenance count: 467 -> 463, the four artifacts retired above.

Nothing else moved: the kernel goldens, the 32162 state digests, the four
topology decisions, the push-constant audit and the binding-access audit are
identical to their recorded values.

**One timing flake, recorded rather than hidden.** The first release pass
reported `ln-low-rank-f32` at `width=128 ranks=32/32/64` moving from
`column-slices` to `one-row`. That pass overlapped an abandoned background ladder
on the same adapter. Re-running only the topology step passed, and the clean
second pass passed with all four decisions matching, so the recorded decision did
not change; the autotuner's 2% margin can react to GPU contention. A future
failure that reproduces on an idle adapter is a real move and still fails.

## Stable-checkout probe parity, as a script (2026-10-20)

The trajectory comparison against the pre-optimization tree used to be a manual
run: build the probe in both checkouts, point both at the same package and diff
`--state-output` snapshots by hand. It is now
`validation/check_stable_probe_parity.py --stable <checkout>`, and it fails on a
single differing digest.

The script takes the repository root of the stable tree (the directory that holds
`hierarchos-vulkan/`, not the crate), builds `training_step_probe` from *both*
trees with the same profile, and runs every geometry the golden records
(`batch2`, `batch32`, `odd_multi_sequence`). One model package per geometry is
initialized by the newer probe and handed to both legs, which is safe because the
probe only ever reads a package it did not create: both legs therefore read
byte-identical initial weights, and the comparison is the graph rather than the
initialization. The gate compares raw FP32 bit patterns through the same
`compare_snapshots` `verify_training_state.py` uses against its goldens -- every
trainable parameter, both Adam moments, the per-slot clocks, the losses, the final
recurrent states, the initial-state adjoints and the context carriers after every
update. Dispatch counts and losses are printed beside the result but are not part
of the gate: a scheduling change may move them, and no value may move.

| Check | Result |
| --- | --- |
| `check_stable_probe_parity.py --stable ../Hierarchos-Native-stable/Hierarchos-Native` | 3 geometries, 3,817 exact tensor digests, **zero differences** |
| `test_stable_probe_parity.py` | 10 tests OK (1 skipped: the live run is opt-in) |

The test module drives the comparison with fabricated snapshots so its strictness
is checked without an adapter: one differing bit, a missing tensor, a moved
optimizer clock and a short run each fail. The live run is
`HIERARCHOS_RUN_STABLE_PARITY=1 HIERARCHOS_STABLE_CHECKOUT=<checkout>`.

**What it does not cover.** The stable checkout is whatever is on disk, so the
script states the tree it compared against (`stable_crate` in its receipt) rather
than pinning a commit; pass a checkout of the commit you intend to compare with.
It compares the training step, not inference logits or kernel goldens, which have
their own oracles.
