# Phase XIX-U Historical Qualification Audit

## Scope

This audit records the sealed historical suites that Phase XIX-T could not
execute because their receipt commits were absent from the local clone. It is
historical evidence only. It does not qualify the current instrumented source
and does not replace the frozen Phase XIX-S numerical receipt.

Machine-readable audit:

`validation/historical_audit/phase19u-historical-audit.json`

Independent checker:

`validation/check_phase19u_audit.py`

## Recovered Revisions

The two missing commits were fetched by exact SHA into the existing repository
and executed one at a time in disposable detached worktrees:

- Phase XVI stability: `3fb0639ff47117a27949616f0dbfeee2f29080e3`
- Phase XVII held-out: `27f853fb477d6904183a91f77f5dabc7e7d4d3ed`

No additional repository clone was created.

## Results

| Qualification | Receipt | Source | Original result | Exit |
| --- | --- | --- | --- | ---: |
| Phase XVI training | `b34e51d` | `8a12af2` | 24 passed, 0 failed, 1 error | 1 |
| Phase XVI stability | `3fb0639` | `21027bf` | 29 passed, 0 failed, 2 errors | 1 |
| Phase XVII held-out | `27f853f` | `5cd0c0c` | 22 passed, 0 failed, 0 errors | 0 |
| Phase XVIII WMMA drift | `1e77b40` | `87a0095` | 26 passed, 0 failed, 0 errors | 0 |
| Phase XIX-R activation | `e69f7f9` | `3ac63b4` | 21 passed, 0 failed, 1 skipped | 0 |

All runs used `python -m unittest discover -s validation -p <suite>` in the
receipt’s own worktree. Full logs and process exit codes are retained by the
audit.

## Missing Historical Artifacts

The original stability receipt lacks:

- `evidence/adapter-E-u.safetensors`
- `evidence/states/F/u/hierarchos_adapter_optimizer.safetensors`

The original training receipt lacks:

- `evidence/states/F/u/hierarchos_adapter_optimizer.safetensors`

These caused the historical errors above. They are evidence-completeness gaps,
not current numerical failures.

## Exact Recovery

The missing byte sequences were recovered from unreachable repository commits
that were created immediately after the sealed receipts:

- `830c0089eef189da9e20065ce0d1597e2719c121`
- `f6024ee0c61d60d293d029b02e204d4a6b96a794`

The recovered files match the historical receipt hashes exactly:

| Artifact | SHA-256 | Git blob |
| --- | --- | --- |
| Stability `adapter-E-u.safetensors` | `cff27588f3e15e32435992e36e649787ee0bdaefe135161c2182928880ebd2ce` | `0d770f4` |
| Stability `states/F/u/hierarchos_adapter_optimizer.safetensors` | `575810a7aa43b3165451867671786896fbda45840fb650846e3850fbc4abf45d` | `0210512` |
| Training `states/F/u/hierarchos_adapter_optimizer.safetensors` | `075e6e1f0a3c2f35a09352e32633d0ef3570041574e0c9cd2109db8f1f91188e` | `2fe0bd4` |

No Phase XIX-S artifact was substituted. The exact bytes are retained under
`validation/historical_audit/recovered/`.

Recovery runs with those exact bytes inserted into disposable copies:

- Stability: 31 passed, 0 failed, 0 errors, exit 0.
- Training: 25 passed, 0 failed, 0 errors, exit 0.

The original error records remain in the audit; recovery success does not
rewrite the sealed historical result.

## Current-Source Relationship

The current numerical authority remains the Phase XIX-S receipt at source
`60063fd3b0a3a963b05e7d5c472cf9b27a4a1cab`, together with the Phase XIX-T
policy overlay. The current controls are:

- `check_phase19s_current.py`
- `check_phase19t_current.py`
- `test_phase19t_current.py`
- `test_activation_phase19s_current.py`

Historical tests are never treated as current-source passes.

## Verification

`check_phase19u_audit.py` independently checks:

- Exact receipt and source commit identities, and source ancestry.
- Test-source and receipt hashes from Git objects.
- Original and recovery log hashes and unittest counts.
- Exact exit codes and missing-artifact lists.
- Recovered artifact SHA-256, Git blob identity, size and receipt binding.
- The current Phase XIX-S source and Phase XIX-T overlay identity.

`test_phase19u_audit.py` rejects changed source identities, receipts, test
hashes, log hashes, counts, recovered bytes and missing recovery records.

## Phase XIX-V CI Reproduction

Phase XIX-V adds `run_phase19u_historical_ci.py`, which replays all five
historical suites at their exact receipt revisions in disposable detached
worktrees. It runs each suite once without recovery bytes and, for the two
Phase XVI receipts, once with the exact retained recovery artifacts. The
original runs and recovery runs are separate entries in a new CI report.

The runner parses `unittest` totals and requires the documented error test
identities and missing-artifact messages. A generic exit code 1, import error,
truncated summary, unexpected assertion failure or changed skip is rejected.
The activation suite must retain its documented `test_changed_binary` skip
when the executable is absent.

The historical job runs on `windows-latest`. This matches the sealed execution
environment: the Phase XVII `vocab.json` is CRLF, while its historical
generator uses platform-native newlines. Linux correctly rejects the
regenerated LF fixture; Windows reproduces the committed bytes without
modifying the historical test.

CI report artifact name: `phase19u-historical-ci`

Report file: `phase19u-historical-ci-results.json`

Retention: 14 days. The report and CI logs are uploaded by the historical job
and are not committed to the repository; the retained Phase XIX-U Windows
logs remain immutable.

The runner is covered by `test_phase19u_ci_runner.py`, including wrong
revision, wrong test identity, truncated output, import error, unexpected
skip, wrong artifact, recovery error and scope controls.
