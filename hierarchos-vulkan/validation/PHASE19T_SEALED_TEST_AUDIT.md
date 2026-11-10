# Phase XIX-T Sealed-Test Audit

## Purpose

Sealed receipts bind the source bytes and test files of the revision that
produced them. A test written for one revision must therefore be executed in an
isolated checkout of that revision, not against the later instrumented source.
This audit records that distinction and the current-source controls that cover
the same contracts.

## Historical Executions

| Qualification | Receipt commit | Production source revision | Suite | Result |
| --- | --- | --- | --- | --- |
| Phase XVIII WMMA drift | `1e77b401489c7c59a95661461fbaaf5a91f3b77c` | `87a0095b83012f3b5fc3efa1085447516c45b4a7` | `test_wmma_drift.py` | **26/26 passed** in a clean worktree |
| Phase XIX-R activation | `e69f7f978101f29e8ad6e09b0a88ce5fb1b767ad` | `3ac63b4d6315f1940688765c17c3e0751567c0ce` | `test_activation_phase19r.py` | **22 tests OK**, 1 binary-dependent skip |
| Phase XVI training | `b34e51d062f5b9667770af57fdca408eab560b47` | `8a12af2d907cec715675c70399d8e3afe70b404c` | `test_training_phase16.py` | **24 passed, 1 error** |
| Phase XVI stability | `3fb0639ff47117a27949616f0dbfeee2f29080e3` | `21027bf9699757859e7f2a9c07c437402d5aa2ae` | `test_training_stability.py` | Blocked: commit is not present in the local clone |
| Phase XVII held-out | `27f853fb477d6904183a91f77f5dabc7e7d4d3ed` | `5cd0c0c6eb7f9a8d0c9ef0bdc214186a8df76fb6` | `test_heldout_phase17.py` | Blocked: commit is not present in the local clone |

The Phase XVIII suite was first run at `87a0095`, where it correctly skipped
because that commit predates the receipt installation. Running it at the
receipt commit `1e77b40` executed all 26 tests and passed.

### Phase XVI training finding

The historical training suite reached 24 passing tests and then errored in
`test_missing_checkpoint_state` because its receipt did not ship:

`states/F/u/hierarchos_adapter_optimizer.safetensors`

The historical test assumed the optimizer-state artifact existed. The current
Phase XIX-S evidence set does contain the corresponding 36 adapter/optimizer
artifacts, and Phase XIX-T requires their exact paths, digests, sizes and Git
tracking. This is a real historical coverage gap, not a current numerical
failure.

## Current-Source Equivalents

The historical tests are not evidence for the instrumented source. The current
contract is covered by:

- `check_phase19s_current.py` and its frozen controls for the Phase XIX-S
  receipt, including the optimizer-step, shader, stale-mirror and evidence
  checks.
- `check_phase19t_current.py` and `test_phase19t_current.py` for exact source
  and evidence inventories, tracked-file presence, path confinement,
  sub-receipt lineage, fixtures, protocols, executable identities and the 36
  adapter/optimizer artifacts.
- `test_activation_phase19s_current.py` for the current activation receipt;
  the sealed `test_activation_phase19r.py` remains bound to its historical
  receipt.

## Control Gaps Closed by Phase XIX-T

- Source completeness can no longer pass on an aggregate count alone; the
  exact 165-source inventory and group counts are enforced.
- Evidence completeness can no longer pass on `len(evidence) > 100`; all 338
  entries and every family count are enforced.
- Missing adapter/optimizer safetensors are rejected even if the remaining
  evidence count is high.
- Evidence that exists only in an untracked working tree is rejected.
- Absolute paths, backslashes, traversal, symlinks, non-canonical names and
  case-normalized duplicate identities are rejected.
- Individually valid sub-receipts with incompatible source lineage, device,
  fixture, split, protocol or executable identities cannot be combined.
- Binary-hash controls run against a deterministic fake executable without an
  RDNA3 GPU.

## Remaining Limitations

- The stability and held-out historical commits are unavailable in this local
  clone. Their exact hashes are recorded above rather than being replaced with
  a different revision.
- Historical suites that mutate receipts are only safe in disposable
  worktrees. Current CI uses the isolated Phase XIX-T controls.
- A historical test passing at its own revision does not prove that the current
  instrumented source passes the corresponding numerical contract. The
  source-bound current receipt remains the authority for that claim.
