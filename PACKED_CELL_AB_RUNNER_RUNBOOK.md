# Packed-cell A/B self-hosted runner — resume runbook

How to bring up a compatible runner, run `.github/workflows/time-mix-ab.yml` against the
qualified source, and verify its output. Nothing here has been executed: as of this
note the repository has **zero** registered runners
(`GET /repos/necat101/Hierarchos-Native/actions/runners` → `{"total_count":0,"runners":[]}`),
so the A/B is **pending**, not passed.

This job is a **cross-check**. The scoped RDNA3 release already contains a locally
measured packed-cell matrix — 13/13 routes on `amd-rdna3-lds64` — produced by
`validation/qualify_packed_current.py` and sealed in
`20261115-phase25-current-source/phase19z/packed-cell/status.json`. The GitHub job
re-measures the same producers on a **self-hosted** adapter and fails if a shipped
default stops building the graph its decision record names. A green run does not
replace the local matrix, and the absence of a run does not weaken it.

## 1. What the workflow actually requires

Read from the workflow file, not assumed:

| Item | Requirement |
|---|---|
| Workflow | `packed-cell producer A/B` — `.github/workflows/time-mix-ab.yml` |
| Triggers | `workflow_dispatch` (**no inputs**), path-filtered `push`, weekly `schedule` (`23 4 * * 1`) |
| Job `producer-ab` | `runs-on: [self-hosted, windows, vulkan]` |
| Job `cross-adapter-ab` | `runs-on: [self-hosted, windows, vulkan, amd]` |
| Runner OS | Windows (`shell: pwsh` steps, `.exe` toolchain assumptions) |
| GPU | Vulkan adapter with a tuned `linear3-forward` **and** `linear-residual-forward` module (AMD RDNA or Intel Gen9) |
| Software | Rust stable (`dtolnay/rust-toolchain@stable`), Python 3.12, `torch` (CPU wheel), `safetensors`, `numpy` |
| Disk | A release build of the workspace plus benchmark receipts: budget several GB free; the repo's own `target/` tree is not in Git |
| Network | Access to crates.io, PyPI, and the Vulkan SDK/runtime already installed on the host |

The harnesses refuse to report a vacuous A/B when the tuned module resolves to
`portable`. A CPU-only host, a different GPU class, or a machine without the tuned
modules will not produce a valid result — do not register one and call it equivalent.

## 2. Runner registration (operator-side; not performed here)

1. Register a **repository-scoped** self-hosted runner on a Windows host with the AMD
   Vulkan driver installed.
2. Labels: `self-hosted`, `windows`, `vulkan` are mandatory for `producer-ab`; add `amd`
   for `cross-adapter-ab`. Labels are matched exactly by `runs-on`.
3. Run the runner as a dedicated, non-administrator service account. Do not run it on a
   workstation holding personal or production data: the workflow checks out and builds
   arbitrary repository content.
4. Grant only repository `read`-level access. The workflow needs no secrets; do not add
   any, and do not enable workflows from forks on this runner.
5. Keep `producer-ab` and `cross-adapter-ab` on the same host only if that host genuinely
   has a qualified AMD adapter with the tuned modules.

Verify registration and labels before dispatching:

```bash
gh api repos/necat101/Hierarchos-Native/actions/runners \
  --jq '.runners[] | {name, status, busy, labels: [.labels[].name]}'
```

A job that stays `queued` is the signature of a label mismatch — the runner is up but
does not match `runs-on`.

## 3. Dispatch

`workflow_dispatch` declares no inputs, so no `-f` flags are required:

```bash
gh workflow run time-mix-ab.yml --ref codex/rdna3-phase19-instrumented
```

The ref is the only control: the job measures whatever commit that ref points at. Record
the resulting `headSha` and confirm it is the commit you intended to measure — the
release qualification source is `306c7db9afbf5de3840c0e6c63876c6c67ab18fc`; later
commits on this branch are documentation-only and add no code to measure.

## 4. Monitor

```bash
gh run list --workflow=time-mix-ab.yml --limit 5 \
  --json databaseId,status,conclusion,headSha,createdAt
gh run watch <run-id>
gh run view <run-id> --log-failed
```

`status: queued` means no runner matched or all matching runners were busy;
`in_progress` means a runner picked the job up. A run that goes back to `cancelled`
after ~24 h never executed.

## 5. Artifacts

Both jobs upload receipts even on failure (`if: always()`), and the upload step uses
`if-no-files-found: warn` — so *an empty artifact is not a failed step*. Check the file
list, not just the step conclusion.

```bash
gh run download <run-id> -n producer-ab-receipts -D target/ab-run/producer
gh run download <run-id> -n cross-adapter-receipts -D target/ab-run/cross
```

Artifact contents: the 13 producer receipts (`packed_cell_time_mix_bench.json`,
`packed_cell_channel_mix_bench.json`, `packed_cell_low_rank_forward_bench.json`,
`low_rank_fp16_forward_bench.json`, `key_state_forward_bench.json`,
`tbptt_packed_forward_bench.json`, `tbptt_packed_remat_bench.json`,
`tbptt_packed_forward_fp16_bench.json`, `tbptt_packed_remat_fp16_bench.json`,
`tbptt_backward_fusion_bench.json`, `tbptt_backward_fusion_fp16_bench.json`,
`packed_cell_tbptt_time_mix_bench.json`, `packed_cell_composition.json`) plus
`key_state_forward_profile.json`.

> **Do not commit the downloaded receipts into
> `20261115-phase25-current-source/phase19z/packed-cell/`.** The workflow's `--record`
> flags point at that sealed directory, so a run rewrites those files **in the working
> tree**. They are measurements for review, not a re-seal. Adopting them requires a new
> dated root and the procedure in `RELEASE_RESEALING_PROCEDURE.md`.

## 6. What a genuine pass looks like

All of the following, from the actual run:

1. `headSha` is the intended commit and `conclusion: success`.
2. The step `Training-stack ladder matches the committed receipt` passed — it re-runs
   `verify_training_stack.py --receipt target/stack_receipt.json` and
   `check_training_stack_receipt.py --compare target/stack_receipt.json`.
3. Every `bench_*` step produced a receipt, and every
   `check_measured_producer_routes.py --producer <name>` guard passed for all 13
   producers: `time-mix-projection`, `channel-mix`, `low-rank-forward`,
   `low-rank-forward-fp16`, `key-state-forward`, `tbptt-packed-forward`,
   `tbptt-packed-rematerialization`, `tbptt-packed-forward-fp16`,
   `tbptt-packed-rematerialization-fp16`, `tbptt-backward-fusion`,
   `tbptt-backward-fusion-fp16`, `tbptt-time-mix`, `packed-cell-composition`.
4. Each receipt names the adapter it was measured on, and it is the real GPU — not
   `portable`.
5. The 10 `unittest` modules in the generated test block pass, including
   `test_ab_workflow_rendering.py` and `test_measured_producer_routes.py`.

A pass means "the shipped default still builds the graph the decision record names on
this adapter". It does **not** mean the packed-cell 13/13 qualification passed — that
status is already sealed from the local matrix.

## 7. After a failure

- **Queued forever / label mismatch** — fix the runner labels, then re-dispatch; the queue
  entry never executes on its own.
- **Build failure** — reproduce the failing step locally with the same commands; do not
  loosen a benchmark's iterations, batch sweep or guard to get past it.
- **Guard failure** — a real route regression: investigate
  `check_measured_producer_routes.py --producer <name>` output and the corresponding
  decision record. Never edit the record to match the run without a controlled
  re-recording.
- **Transient host failure** — `gh run rerun <run-id> --failed` on the same commit; keep
  the failed logs as evidence of the attempt.
- Never hand-edit a receipt, a decision record, or a status field to turn a failure into
  a pass.

## 8. Status

**Pending.** No compatible runner exists, no A/B run has executed against `306c7db` or
any later commit, and no A/B receipt exists for the current release. The scoped RDNA3
release rests on the local 13/13 packed-cell matrix, which is a different piece of
evidence on a different execution path.
