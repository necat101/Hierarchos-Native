# Phase XVIII follow-up — Windows path smoke repair

This is a new, narrow receipt. It does not modify the main Phase XVIII
numerical receipt at:

`validation/release_parity/multimodal_training/20261108-rdna3-phase18-replay-reliability`

## Source

- Source: `ce1757d3c2e362d811f32f9467ff065fb7dcf0b8`
- Parent: `58ab4fbc7374bc44c64bc65e183155f277db9ea6`
- Changed file: `hierarchos-vulkan/validation/test_rdna3_workflow_paths.py`
- Test binary: `cc558cc72012bc5bb0362093af6f9a3b5e8bdf07fdda06be3883d24d200dad62`
- No numerical or production Vulkan source changed.

## Remote failure and repair

The first post-seal remote run, `42525900516`, passed `workflow-validation`
and reached the Windows job. The new runtime probe failed because it invoked
bare `bash -lc`; on the hosted runner that did not reliably resolve the Git
Bash runtime and the probe returned 1 with no stderr. The failed remote log is
preserved verbatim in `ci/initial-remote-failure.log`.

The repair:

- prefers the standard Git Bash executable instead of an arbitrary `bash`
  (including WSL);
- skips login profiles with `--noprofile --norc`;
- converts the running Python executable through `cygpath -u`; and
- invokes the selector with the converted Python path, while still passing the
  converted build-log path through Git Bash.

The local Windows test now passes 5/5. The replacement remote run,
`42526223339`, reports `workflow-validation: success` and
`RDNA3 Windows path smoke test: success`. Its broader `workspace-check` was
still in progress when this receipt was sealed; no overall remote workspace
PASS is claimed. The captured status JSON is in `ci/followup-remote-status.json`.

## Current-source qualification

The current-source GPT-2 guard was regenerated after the test-only commit:

- source head: `ce1757d3c2e362d811f32f9467ff065fb7dcf0b8`
- 16/16 frozen shapes
- worst logit: `5.960464477539063e-08`
- `CURRENT SOURCE QUALIFICATION: PASS (ce1757d3c2e3)`

This confirms the test-only edit did not stale the numerical qualification.
The main receipt's `full-5` and `full-6` nine-stage runs remain the numerical
evidence; this follow-up does not rerun or relabel them.

## Verification

Run `python verify.py`. It checks the SHA manifest, source and diff hashes,
the 5/5 local path log, the initial remote failure, the successful replacement
remote path step, and the current-source GPT-2 guard.

## Hardware limitation

RDNA3 self-hosted runners remain at zero. No hardware CI PASS is claimed.
