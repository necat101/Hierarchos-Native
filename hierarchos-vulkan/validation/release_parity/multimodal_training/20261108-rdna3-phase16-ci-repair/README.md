# Phase XVI — RDNA3 CI repair and replay isolation

This receipt records a CI and test-infrastructure repair, not a new numerical
optimization result. The qualified numerical implementation remains the one
covered by the Phase XV receipt.

## Bindings

- Code source: `4c973dd0e2b13d28b439f644f9cb74d65dfde329`
- Parent / observed remote main: `da71b8894378f3096a8e82595cc2007f63dfedca`
- Numerically qualified source: `a2d68f96802f278fbf79f019d58454270989ade5`
- Phase XV receipt: `../20261108-rdna3-phase15-qualification/`
- Frozen multimodal oracle: `hierarchos-vulkan/target/rdna3-port/gen9-frozen`
- Local test toolchain: Cargo/Rustc `1.93.0`

The only source-file change outside workflow and validation infrastructure is
`hierarchos-vulkan/src/deepseek_v41/model_tests.rs`, which is `cfg(test)` code.
No numerical implementation, shader, `Cargo.toml`, or `Cargo.lock` content was
changed. The Phase XV source binding therefore remains valid; the current-source
GPT-2 guard and the Phase XV receipt verifier are both recorded below.

## Workflow repair

`.github/workflows/rdna3-performance.yml` no longer uses `runner.temp` in the
job-level `env` block. The initialization step reads `RUNNER_TEMP`, converts
Windows paths with `pathlib.PureWindowsPath(...).as_posix()`, creates the run
root, and writes these absolute paths to `$GITHUB_ENV`:

- `CANDIDATE_TARGET`
- `BASELINE_TARGET`
- `BENCHMARK_ROOT`
- `BASELINE_SRC`

The benchmark root is therefore one absolute location shared by provenance,
batch-1/2/4 outputs, the performance gate, and artifact upload. Cargo and Rustc
are checked separately and both versions are recorded. Candidate and baseline
test binaries are built with `cargo test --release --locked --lib --no-run
--message-format=json`; `cargo_test_binary.py` selects the one linked test
executable from the Cargo messages instead of scanning a directory by mtime.
The release candidate profile was observed as `opt_level=3`,
`debug_assertions=false`, `overflow_checks=false`, `test=true`; the baseline
`Cargo.toml` is byte-identical to the candidate manifest (see
`ci/profile-parity.log`).

`actions/checkout@v5` and `actions/cache@v5` replace the Node-20 actions in
`workspace-check.yml`; `fetch-depth: 0` and `persist-credentials: false` are
preserved. Artifact upload uses `actions/upload-artifact@v5`. The workflow keeps
the hardware execution restricted to the trusted self-hosted RDNA3 runner and
adds no pull-request trigger.

Actionlint 1.7.12 is recorded as passing all workflows. Its negative control
against the pre-fix workflow exits 1 and reports that the `runner` context is
not allowed in the job environment, so the validation is exercising the actual
failure mode rather than only parsing YAML.

## Scratch-state repair

`model_tests.rs` now reserves every package root with a timestamp, atomic
counter, PID, and bounded `AlreadyExists` retry. A stale directory left by an
interrupted process is harmless because it cannot receive the same generated
name. The regression test
`deepseek_v41::model::tests::temporary_test_roots_do_not_reuse_stale_directories`
passes, and the checkpoint-resume test
`deepseek_v41::model::tests::training::native_vision::native_state::deepseek_v41_native_joint_checkpoint_resume_export_and_freshness`
passes in 7.40 s when run exactly. The checkpoint freshness and resume logic was
not weakened or skipped.

The DeepSeek validator excludes only the infrastructure regression test from
the numerical inventory. Enumeration remains 245 discovered / 244 numerical /
1 excluded, preserving the 218 active and 26 ignored numerical tests.

## Aggregate replay

Two consecutive complete nine-stage producer runs were executed from the same
debug test binary:

| Run | Exit | Stages | Aggregate |
| --- | ---: | ---: | --- |
| `full-2` | 0 | 9/9 | `pass=true`, `complete=true` |
| `full-3` | 0 | 9/9 | `pass=true`, `complete=true` |

Per-stage counts in both runs:

- Shared multimodal: 77/77
- Seq2seq: 6/6
- Gemma4 legacy: 1/1
- Gemma4 RGB: 1/1
- Gemma3: 4/4
- LLaVA: 4/4
- Text resident/streamed: 166/166
- DeepSeek V4.1: 218/218 verified, 26 ignored, 28 shards
- Native V4.1: 30/30

Both aggregate manifests retain `absolute_limit=2e-7`, `rtol=0`, and
`comparison=strict_less_than`. The frozen manifest and debug test-binary hashes
are recorded in the aggregate `qualification.json` files. The original failed
nine-stage run (`full-1`, caused by the then-unclassified regression test) has
not been relabeled or rewritten; its historical evidence remains in the Phase XV
record/locally preserved output.

## Verification and integrity

`ci/` contains the raw command logs, exit statuses, selector controls, path
simulation, worktree smoke test, receipt guards, and Phase XV verifier output.
Run:

```text
python verify.py
```

The verifier checks the SHA-256 manifest, both aggregate manifests, stage
counts, exact tolerances, actionlint results and negative control, release
profile, selector controls, scratch/checkpoint tests, receipt guards, and source
hashes. `cargo fmt --check` remains a known pre-existing failure in
`vulkan.rs`, `native_vision_state_tests.rs`, and `image_text.rs`; the changed
test file passes `rustfmt --check --config skip_children=true`. That distinction
is recorded rather than hidden.

## Hardware status

No RDNA3 self-hosted GitHub Actions runner is registered in this environment.
The performance workflow was statically valid and locally checked, including
the release build and deterministic binary selection, but it was not executed
on a physical RDNA3 runner here. No hardware performance PASS is claimed. The
existing performance acceptance thresholds are unchanged.
