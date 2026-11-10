# Phase XIX-S: Instrumented Transformer Current-Source Qualification

## Purpose

The instrumented branch exists to prove that the current Rust/Vulkan
Transformer still satisfies the RDNA3 numerical contract after the Phase XIX-R
evaluation-lag repair. It binds the activation tracer, training harnesses,
attention-backward evidence, WMMA drift evidence and shader provenance to one
receipt produced from the source that actually ran on the GPU.

The frozen Phase XIX-S receipt is:

`validation/release_parity/multimodal_training/20261110-phase19s-current-source/current-qualification.json`

Its numerical source revision is `60063fd3b0a3a963b05e7d5c472cf9b27a4a1cab`.
The receipt was committed at `9a9a8c7755a0e58553aa3e664926d392f17a2d25` and
qualified by CI on branch `codex/rdna3-phase19-instrumented`.

## Why Phase XIX-R Failed Current-Source Checks

Phase XIX-R produced genuine activation evidence, but its receipt bound source
bytes from an earlier revision. Later validation and production changes meant
that a checkout of the instrumented branch could not honestly reuse the old
receipt as current-source evidence. The sealed Phase XIX-R receipt was kept
unchanged; Phase XIX-S reran the affected GPU producers and assembled a new
source-bound receipt instead of editing historical hashes.

## Qualified Evidence

The Phase XIX-S receipt binds:

- 165 source paths bound by the frozen numerical receipt.
- 178 complete source paths after Phase XIX-T folds in the 13 training,
  stability, held-out and drift harness/producer/test sources that the
  sub-receipts bind.
- 338 committed evidence artifacts.
- 12 direct attention-backward gradient tensors.
- 45 shader artifacts: 43 historical provenance artifacts and 2 Phase XVI
  artifacts, all reported as identical.
- Eight training arms, including three exact LoRA resume arms and eight
  performance arms.
- Four full-parameter training-stability arms at the selected learning rate
  `0.001`.
- Four held-out evaluation arms.
- Six mixed-WMMA drift runs across three training-window orders.
- Five activation checkpoints, ten activation traces and 27 tensors per trace.

### Attention Backward

The direct gradient comparison covers four cases and twelve tensors. The
receipt requires bitwise equality: `max_abs = 0.0`, `max_ulp = 0`, and every
element equal. The two-pass shader arithmetic was not changed by this phase.

### Training and Exact LoRA Resume

The eight-arm matrix covers FP32/mixed precision, scalar/two-pass backward,
full-parameter/LoRA modes and the corresponding comparisons. Three LoRA arms
have exact interrupted/resumed state evidence. Full-parameter exact resume is
still unsupported and remains an explicit limitation.

### Stability

The stability sweep selected `0.001` and retained four arms at that rate. Its
receipt binds the adapter and optimizer state artifacts used for the resume
comparisons; the committed `.safetensors` bytes are part of the evidence
manifest.

### Held-Out Generalization

The held-out receipt records validation and test evaluations under the frozen
split:

- Validation final: A `2.831961`, D `2.927568`.
- Test final: A `2.822131`, D `2.914524`.

These are measurements from the qualified source, not a claim that mixed-WMMA
matches FP32 exactly.

### WMMA Drift

The drift receipt contains six runs (three window orders, FP32 and mixed).
Same-weight forward comparisons remain small; the final-step mixed FP32 gap is
larger and is reported rather than hidden. The current-source checker accepts a
zero-conversion training step only when the immediately preceding evaluation
refreshed the mirror epoch, and still requires a conversion in every run.

### Activation Tracing and the Mirror-Epoch Repair

Tracing is opt-in through `HIERARCHOS_ACTIVATION_TRACE=1`; it is observation
only. The Phase XIX-R stale-mirror anomaly was:

- pre-fix in-process step-1 loss: `7.060197`
- post-fix in-process step-1 loss: `6.757523`
- fresh step-1 mixed-process loss: `6.757523`

The repair publishes the WMMA master epoch after the submitted optimizer step.
A between-step evaluation therefore converts mirrors from the post-step
masters. The long-run mixed/FP32 difference remains measurable: at step 100 the
mixed fresh process reports `2.927568` against FP32 `2.831961`. The repair
removes the evaluation-lag artifact; it does not eliminate precision drift.

## Shader Provenance

The current 45-artifact inventory is derived from the attention receipt. The
historical 43-artifact gate remains sealed and unchanged. The two Phase XVI
artifacts are checked in the same inventory. The checker rejects a changed
inventory, changed status counts, a changed historical artifact count, or a
changed sealed-gate digest.

## Verification Architecture

The frozen Phase XIX-S checker (`check_phase19s_current.py`) verifies the
receipt and its six stage receipts. It is preserved unchanged because the
receipt binds its bytes.

Phase XIX-T adds a separate policy overlay. The overlay does not replace the
numerical receipt and does not claim a new GPU run. It adds:

- an exact 178-source inventory: the frozen 165-path numerical subset plus 13
  sub-receipt harness/producer/test sources, with source-group counts;
- an exact 338-entry evidence inventory with family counts;
- the 36 required training/stability adapter and optimizer `.safetensors`
  artifacts;
- tracked-file checks, so a file present only in a working tree cannot qualify;
- canonical relative-path validation, traversal and symlink rejection;
- sub-receipt lineage, fixture, device, protocol and executable cross-binds;
- binary-digest checks, including isolated fake-executable negative controls;
- non-destructive negative controls that never rewrite committed receipts.

The policy overlay is:

`validation/phase19t-policy-overlay.json`

The historical-test audit is:

`validation/PHASE19T_SEALED_TEST_AUDIT.md`

The checker is:

`validation/check_phase19t_current.py`

Its negative controls are in:

`validation/test_phase19t_current.py`

## CPU versus GPU Boundary

CPU CI verifies committed bytes, source ancestry, evidence manifests,
cross-stage identities and numerical derivations from retained evidence. It
does not execute the RDNA3 kernels and does not rehash GPU binaries that are not
present in the checkout.

The SHA-256 bindings establish integrity relative to the committed receipt and
evidence. They do not prove that an independent author ran the GPU producer.
That distinction remains explicit: committed evidence integrity is not
cryptographic proof of hardware provenance.

If a supplied binary root contains an executable, the checker hashes it and
rejects a mismatch. If no binary is available, the recorded digest and its
cross-stage identity are still checked.

## Reproduce the Checks

From `Hierarchos-Native-phase19`:

```bash
python hierarchos-vulkan/validation/check_phase19s_current.py
python hierarchos-vulkan/validation/check_phase19t_current.py
python -m unittest discover -s hierarchos-vulkan/validation -p "test_phase19t_current.py"
```

The Phase XIX-T checker also accepts `--binary-root`; supplied executable
bytes are hashed and a mismatch is rejected. Without a supplied binary root,
the recorded digests and cross-stage identities are still checked.

```bash
python hierarchos-vulkan/validation/check_phase19t_current.py --binary-root target/release
```

The Phase XIX-S controls still exist as historical controls. They mutate the
real receipt in place and should only be run in a disposable checkout. Current
CI uses the isolated Phase XIX-T controls instead.

## Update Procedure

1. If production numerical code changes, rerun the affected GPU producers and
   create a new source-bound numerical receipt. Do not edit the frozen Phase
   XIX-S receipt.
2. If only verification policy changes, update the policy checker, tests and
   documentation, then regenerate `phase19t-policy-overlay.json` with
   `qualify_phase19t_policy.py`.
3. The producer refuses to freeze an overlay unless the frozen Phase XIX-S
   checker still passes and the exact source/evidence manifests match the
   policy constants.
4. A changed source, evidence byte, sub-receipt identity or binary binding must
   fail closed; do not update a digest merely to make CI green.
5. If a checked-in evidence file is removed from Git, the tracked-file check
   fails even if a local copy still exists.

## Remaining Limitations

- Full-parameter exact resume remains unsupported; only LoRA resume is
  qualified.
- Raw GPU captures, parameter dumps and five activation weight dumps remain
  under `target/` or outside the committed tree and are bound by SHA-256 only.
- The mixed-WMMA long-run difference is real and is not erased by the
  mirror-epoch repair.
- Historical tests are executed against their original revisions when those
  revisions are compatible with the checkout's path/toolchain constraints.
  They are not evidence for the current instrumented source.
