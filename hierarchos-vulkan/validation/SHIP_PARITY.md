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

**What this does not cover.** The packed-cell runner resolves its own fused packed-cell
shaders, so the harness prints `variant-plan:` but expects an empty
`variant-dispatch:` -- it exercises the portable fused cell, not the tuned Intel Gen9
`linear*` modules. Those are covered by `verify_vendor_parity.py` and the gate-width
fixtures above, and by the fused-vs-composition check below.

## RWKV fused packed cell vs vendor-linear composition (2026-10-17)

`HIERARCHOS_VULKAN_DISABLE_CHANNEL_MIX_FORWARD_FUSION` selects the staged
channel-mix graph instead of the single fused packed-cell shader. On that path
`key_cm` and `value_cm` dispatch the vendor `linear-forward` and
`linear-residual-forward` modules, so the same recurrent cell can be run once
through the fused shader and once through the composition the rest of the training
graph uses.

`validation/verify_rwkv_packed_cell_composition.py` runs both graphs over the same
generated package (`width=128`, `head_size=16`, `batch=8`, live DeepEmbed adapter),
steps them three times with each leg consuming its own packed state, and compares
`output`, `packed_new_state`, `grad_x`, `grad_packed_state` and
`token_feature_grad` at every step. Batch 8 is required: at the bridge's four rows
the Gen9 low-row gate (`output_dim >= 192`) leaves `value_cm` (128 outputs) on the
portable module, and the harness would not be comparing the vendor graph. The
runner reports which packed forward kernel it built, so the two legs cannot
silently collapse onto one graph.

| Check | Result |
| --- | --- |
| `verify_rwkv_packed_cell_composition.py` | 3 steps, 15 comparisons, exit 0, worst `0.0` on every tensor |
| ... kernel selection | fused leg `packed-cell-forward-kernel: fused` with empty `variant-dispatch`; composed leg `composed` with `linear-forward=1 linear-residual-forward=1` per step |
| ... vendor plan | `linear-forward=intel-gen9-lds linear-residual-forward=intel-gen9-lds` on this host |
| ... vendor qualification (`--expected-vendor intel`) | exit 0; the receipt records `expected_vendor=intel`, `qualified_vendor=intel` and the two resolved variants |
| ... wrong-vendor guard (`--expected-vendor amd` on this host) | exit 1: `linear-forward resolved 'intel-gen9-lds', not an amd module; this host cannot qualify the amd path` |
| `test_rwkv_composition_receipt.py` | 7 tests OK, 1 skipped (live GPU leg gated by `HIERARCHOS_RUN_GPU_VALIDATION=1`) |
| Negative control (`x` scaled by `1.01` on the composed leg) | rejected at `1.586200e-4` against a `1.038859e-4` bound |
| `bench_rwkv_packed_cell_composition.py`, batch 8 / 16 / 32 | composed median `0.946x` / `0.882x` / `0.872x` of fused, bit-identical results |

The comparison is bit-exact, not merely within tolerance: the fused shader keeps the
same lane-serial accumulation order as the portable linear modules and the Gen9
module preserves it, so no reordering drift appears. That is the strongest form of
the agreement -- a composed-cell only user can trust the fused accelerator and vice
versa -- but it also means this check cannot tell a genuine `0.0` from two
bit-identical implementations of the same bug. The CPU bridge remains the
independent oracle for correctness; this check closes the **vendor-linear coverage**
gap, not the numerical-oracle gap.

The recorded evidence is also a test: `test_rwkv_composition_receipt.py` fails the
ordinary `unittest discover` run if the receipt loses the per-leg kernel marker, a
vendor dispatch family, a vendor-specific plan or the negative control. It reads
evidence rather than re-running the GPU; `HIERARCHOS_RUN_GPU_VALIDATION=1` enables
the live leg, which is what catches a code regression rather than a stale record.

Throughput (release build, HD 520, width 128, 30 interleaved samples per graph):
the composed graph is `0.946x` the fused median at batch 8 and `0.872x` at batch 32,
with `0.0` difference in output and packed state. The fused packed shader is a
single-dispatch convenience, not a throughput feature on this part; the full table
and the caveats before deleting it are in
[PACKED_CELL_COMPOSITION.md](PACKED_CELL_COMPOSITION.md).

**What this does not cover.** Only the channel-mix/state forward is swapped; the
time-mix, adapter and all backward dispatches are shared between the two legs, so
their gradients agree by construction once the forward activations agree. This host
resolves the Intel Gen9 modules only: `--expected-vendor amd` is implemented and
fails closed here, but the RDNA3 `linear*` modules still need the same run on an AMD
adapter (and `HIERARCHOS_VULKAN_FORCE_VENDOR` is rejected precisely because it would
name those modules without the hardware).

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
