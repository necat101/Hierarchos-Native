Hierarchos-Native PEFT Audit

## COMPLETE QUALIFICATION 2026-09-30

This section supersedes historical incomplete/red checkpoints below. All **20
inventory rows / 32 independent text surfaces** pass ordinary LoRA, named-adapter
switching, and registered `modules_to_save` at absolute `max_abs <= 2e-7`.
Qualification does not extend to arbitrary model sizes/dtypes/devices/dropout
trajectories, unimplemented targets/methods, or vision/audio towers.

### Fresh matrix and source of truth

- Command (from `hierarchos-vulkan`):
  `HIERARCHOS_PEFT_USE_PREBUILT=1 py -3 validation/verify_peft_green_matrix.py --stage all --jobs 1`.
- Finished `2026-09-30T12:26:27.868118+00:00`: **32 PASS / 0 FAIL**.
  Aggregate `.peft-oracle-fixtures/green-matrix-report.json`; log
  `.peft-final-matrix-20260930.log`. All stages were rerun, not collected.
- Provenance report confirms `inputs_unchanged=true`: native Rust, GLSL,
  embedded SPIR-V, validation executables, oracle scripts, and local Transformers
  Python sources were identical at entry/exit. Registry-only promotion and
  documentation updates followed qualification; numerical kernels are unchanged.
- Oracle imports `C:\Users\User\transformers\src\transformers\__init__.py`,
  `5.16.0.dev0`. The harness now rejects imports outside this requested clone.
  Direct local Gemma3 source inspection confirms embedding scaling, FP32
  `(1 + weight)` RMSNorm, Q/K norms before RoPE, eager attention and MLP order.
- All ordinary and saved-module frozen-base drift is exactly `0.0`; native
  resume is exactly `0.0`; saved banks are isolated; trainable key sets match.
- Worst ordinary PEFT logits `5.34034729215982e-8` (Mixtral), gradients
  `2.60770320892334e-8` (Gemma4), two-step AdamW `2.70083546638489e-8`
  (MiniMax M2). Worst saved-module gradient `1.9371509552002e-7`
  (MiniMax M3), still below the unchanged ceiling.
- Gemma3 saved embedding A/B gradients: `1.1920928955078125e-7` /
  `1.7881393432617188e-7`; Gemma4: `7.450580596923828e-9` /
  `2.9802322387695312e-8`. Linear/norm/head/embedding, both adapters,
  switching/disable, two-step AdamW, save/reload, and bank isolation all PASS.
- Gemma3/Gemma4 `modules_to_save_validated` are promoted. Registry saved classes
  now correctly include canonical LayerNorm/RMSNorm for all qualified families;
  arbitrary internal/per-head norms remain outside that graph.

### Regression and user workflow evidence

- `cargo test --lib -j 1`: **670 passed, 0 failed, 8 ignored** after promotion.
- Native CLI tests: **33 passed**; inference tests: **12 passed**;
  GUI tests: **7 passed, 1 fixture-dependent ignored**. Rust check/build gates PASS.
- Python registry/oracle-identity regressions: **6 passed** against the clone.
- Fresh base logits (28 common surfaces plus Gemma3 tied/softcap regression)
  PASS at `atol=2e-7, rtol=0`. Full two-step training across 28 common surfaces
  PASS. Qwen3-Next base/PEFT is separately exercised in the complete PEFT matrix.
- Falcon H1's three variants PASS dedicated strict base, cache/generation,
  full-training/dual-AdamW, and reload checks. A fresh shipping check of the
  CLI/GUI full-parameter frontend initially failed layer-1 `down_proj` at
  `4.470348358154297e-7` under the frontend's AdamW `eps=1e-8` schedule.
  Operand-matched tracing localized it to the outer Falcon RMSNorms: the local
  Transformers/PyTorch oracle reduces the FP32 mean through an eight-lane
  vector topology, while Falcon's norms used the scalar path. Selecting the
  existing vector CPU-rsqrt reduction for Falcon's final/per-layer outer norms
  (`src/transformer/falcon_h1.rs`, no shader or tolerance change) makes the
  matched-operand down-projection GEMM exact (`0.0`) and the complete
  `verify_falcon_h1_frontends.py --gui` training/fine-tuning/reload suite PASS.
  All three Falcon PEFT surfaces were requalified after that repair. Gemma3's
  ordinary and tied-softcap cache/train/reload lifecycle PASS independently.
- Shader source/artifact reproducibility PASS: 13 audited attention/RMS/RoPE/
  GELU/linear/order kernels compile identically with default `glslc`; the
  LoRA-A eight-lane kernel compiles identically with `glslc -O`. The audit now
  records explicit compiler flags; no production artifacts changed here.
- Optimized release CLI build PASS. Debug **and release** CLI smoke tests PASS:
  automatic merge max-abs `3.00000000119649e-8`, exact resumed adapter/moment
  payloads, merge alias, matching generation, and actual Gemma3/Gemma4
  saved-norm+embedding training and adapter generation through production gates.
- Usage: [root PEFT quickstart](README.md#native-peft--lora-fine-tuning) and
  [complete CLI guide](hierarchos-native-cli/README.md#peft-fine-tuning-guide).
  Covers model/data preparation, target selection, saved modules, HF config,
  budget rank selection, adapter files, native resume vs weights-only warm start,
  unmerged inference, merge/export, memory costs, and explicit exclusions.

No tolerance changes, gradient compensation, or architecture-wide higher-precision
workarounds were used. These results validate the supported implementation and
release CLI workflow, not universal production-checkpoint certification.


### Active verification 2026-09-30: fresh local-source qualification

- Rebuilt all four validation executables from the current checkout.
- Fresh `--stage all --jobs 1 --families gemma3` exits 0: LoRA, saved
  modules, and switching all PASS. Ordinary PEFT logits `4.6521644592179e-8`,
  gradients `1.4901161193847656e-8`, two-step AdamW
  `1.7986167222261429e-8`. Saved embedding A/B gradients are
  `1.1920928955078125e-7` / `1.7881393432617188e-7`; frozen base and native
  resume remain exactly `0.0`. Linear, norm, head, and embedding banks all PASS.
- Confirmed the imported mathematical oracle is
  `C:\Users\User\transformers\src\transformers\__init__.py`, version
  `5.16.0.dev0`. Read the local Gemma3 implementation directly; retained repair
  follows its FP32 RMS, attention, and MLP graph instead of compensating final
  gradients or changing tolerance.
- Validation now rejects Transformers imports outside the requested clone,
  fingerprints its Python sources alongside native sources/shaders/binaries,
  and rejects matrix promotion when fingerprinted inputs change during a run.
- Full 32-surface serialized all-stage requalification is running, log
  `hierarchos-vulkan/.peft-final-matrix-20260930.log`. Selected Gemma3 evidence
  is `.peft-final-gemma3-20260930.log`; do not conflate selected/full reports.
- Registry/docs promotion remains pending full-matrix and regression evidence.
  Absolute `max_abs <= 2e-7`, exact frozen base, and bank/lifecycle checks remain
  unchanged. Historical checkpoints below are not current completion claims.

### Active continuation 2026-09-30: Gemma3 PEFT arithmetic repair

The September 29 saved-stage matrix has 31/32 passing surfaces; Gemma3 is
the remaining numerical failure. This checkpoint is not full qualification.

- Fresh entry saved-embedding A probe reproduces `3.8743019104003906e-7`.
- Found that PEFT attachment overwrote Gemma3's final RMSNorm vector reduction
  (mode 3) with scalar mode 1. Preserve the reduction selected by the base graph.
  That correction alone measures `3.5762786865234375e-7`, still above the gate.
- Operand-matched replay confirms the CPU attention arithmetic already used by
  Gemma4 makes Gemma3's final hidden states and final RMS output exactly equal
  to HF in the captured embedding-A fixture. Apply that order in both forward
  and backward, including separately rounded BMM products and CPU softmax.
- Gemma3 MLP and output-projection LoRA-B input adjoints require separately
  rounded products; its MLP join must retain the independent gate LoRA/base
  branches. With the scoped fix, all replayed MLP/output-projection and Q/K norm
  input adjoints match HF on identical operands (`0.0`).
- Combined narrow saved-embedding A probe passes at `1.1920928955078125e-7`.
  Evidence: `.peft-gemma3-matched-attention-adjoints-20260930b.log` and
  `.peft-gemma3-mlp-adjoint-replay-20260930b.json` under `hierarchos-vulkan`.
- Full Gemma3 `--stage all --jobs 1 --families gemma3` exits 0 with all three
  stages passing. Saved linear/norm/head/embedding pass for both A and B;
  embedding gradients are `1.1920928955078125e-7` / `1.7881393432617188e-7`.
  Frozen base and resume are exactly `0.0`. Full log:
  `.peft-gemma3-full-repair-20260930b.log`.
- The complete 32-surface `--stage all --jobs 1` matrix is now running on the
  rebuilt binaries (`HIERARCHOS_PEFT_USE_PREBUILT=1`), with fresh per-stage
  reports and Rust/source/shader/binary provenance. Do not promote registry/docs
  before that matrix and regression gates pass. The absolute ceiling remains
  `2e-7`; frozen-base drift must be `0.0`.

### Active continuation 2026-09-29: Gemma4 operand-matched backward localization

This is an experimental checkpoint, not architecture qualification. Keep the
absolute `max_abs <= 2e-7` and frozen-base `0.0` requirements unchanged.

- Fresh entry probe (`.peft-gemma4-resume-baseline-20260929.log`) reproduces
  saved input-layernorm adapter-A gradient `2.384185791015625e-7`.
- Attention backward already matches PyTorch when replayed with identical native
  operands. Follow the upstream errors rather than changing its reductions.
- The residual-aware pow-RMS derivative used `rstd / mean` while ordinary
  pow-RMS backward independently evaluates `pow(mean, -1.5)`. Aligned the two
  paths and guarded rsqrt-specific modes; the narrow norm gate remains red.
- Operand-matched layer-1 MLP down backward initially differs by
  `1.4901161193847656e-8`. CPU replay identifies LoRA-B's separately rounded
  product accumulation; a scoped Gemma4 kernel makes this boundary exact.
- GELU-tanh backward's staged CPU arithmetic also matches exactly while the
  previous shader differs by `1.862645149230957e-9`. After staging it, the
  layer-1 gate output adjoint is exact but the final saved-norm gradient is
  `6.258487701416016e-7`; this intermediate candidate is NOT qualified.
- Gate/up LoRA-B adjoints have the same operand-matched reduction mismatch.
  Their scoped repair is under test. Do not promote any registry flags or
  describe full PEFT as complete until the complete saved and regression gates
  pass. Rollback source/SPIR-V copies for existing shader edits are under
  `.peft-recovery-20260927/resume-20260929-*`.
- Gemma4 Q/K/V ordinary-LoRA input adjoints now remain separate through the
  graph join, matching eager autograd's reverse projection visitation order:
  V base+LoRA, then K LoRA/base, then Q LoRA/base. On the current binary
  (`transformer_parity.exe` SHA-256 `9bbf24f4...`), the narrow saved
  `model.layers.0.input_layernorm` adapter-A gradient improves from the resumed
  `2.384185791015625e-7` blocker to `3.725290298461914e-9`.
- The fresh complete Gemma4 `modules_to_save` fixture still exits 1, so Gemma4
  is NOT qualified. `model.layers.0.mlp.down_proj` and `lm_head` pass. Input
  norm adapter A passes, while adapter B remains red at
  `3.5762786865234375e-7`. Saved embedding remains red at adapter A
  `3.8743019104003906e-7` and adapter B `2.384185791015625e-7`; resume drift
  and frozen-base drift remain `0.0`, and bank isolation remains true.
- The current embedding-A diagnostic proves the sparse saved-embedding scatter
  is not the source: HF source-order weight-gradient reconstruction is exact
  (`0.0`), while native-vs-HF gradient presented to the embedding already has
  `3.8743019104003906e-7` max-abs drift. Continue upstream graph-association
  localization. Evidence logs:
  `.peft-gemma4-post-qkv-order-20260929.log`,
  `.peft-gemma4-saved-post-qkv-order-20260929.log`,
  `.peft-gemma4-inputnorm-b-post-qkv-order-20260929.log`, and
  `.peft-gemma4-embed-a-post-qkv-order-20260929.log`.
- Gemma4 per-head Q/K/V `Gemma4RMSNorm` backward is now routed through the
  pow-RMS eight-lane CPU reduction mode used by the local PyTorch oracle. A
  fresh current-checkout serialized run,
  `py -3 validation\\verify_peft_green_matrix.py --stage saved --jobs 1 --families gemma4`,
  exits `0` with `lora=true`, `saved=true`, and `switch=true`. The earlier
  adapter-B embedding probe under this candidate measured only
  `2.9802322387695312e-8` gradient max-abs. Gemma4 is therefore recovered at
  the unchanged `2e-7` gate; keep the full shared-kernel regression matrix as
  the next promotion guard before changing registry/docs.

Last updated: 2026-09-27

### Fresh Gemma saved-module checkpoint: lane4 K/V experiment is not sufficient

- Recompiled the live `linear_forward_lane4.comp` with Vulkan SDK `glslc`; the
  generated SHA-256 exactly matches the embedded live SPIR-V at
  `5fe9fadd2cea031a49d332105efebe991ddd283f4f32dca59582499dd0e73da2`.
  The source SHA-256 is
  `e0f6976fcd83fe225c6137d8d0b3a9446abb983387282be2135f12e971aa459a`.
- Fresh serialized qualification command:
  `py -3 validation\\verify_peft_green_matrix.py --stage saved --jobs 1 --families gemma3 gemma4`
  exits `1`; both selected rows remain `saved=false`. This is a real fixture
  rerun, not `--collect-only` or inherited report state.
- Gemma3 now has only one failing saved-module row: `model.embed_tokens`
  adapter-A gradient `5.066394805908203e-7`. Its saved linear, input norm and
  `lm_head` rows pass; the `lm_head` adapter-B gradient is
  `1.7881393432617188e-7`. Frozen-base and native resume drift remain `0.0`.
- Gemma4 saved linear passes, but `model.layers.0.input_layernorm` gradients are
  A `1.341104507446289e-6` / B `4.76837158203125e-7`, `lm_head` adapter-B is
  `2.561137080192566e-7`, and saved-embedding gradients are A
  `6.556510925292969e-7` / B `1.1920928955078125e-6`. Frozen-base and native
  resume drift remain `0.0`.
- The live Gemma4 autograd trace localizes the large amplification to Q/K/V
  RMSNorm arithmetic: the top input-norm mean-square can match HF exactly while
  native rstd differs by `9.5367431640625e-7`, and Q/K/V norm input adjoints
  differ by roughly `4.8e-6` to `6.2e-6`. The replacement bank is therefore not
  the current root cause. Verify the exact local Transformers/PyTorch RMSNorm
  operation and its backward before retaining further K/V GEMM changes.
- The lane4 experiment is **not qualified** from these results. Do not promote
  Gemma3/Gemma4 registry flags until complete saved fixtures and the shared
  regression matrix pass at the unchanged `2e-7` absolute gate.

### Resume checkpoint: MiniMax M2 forward/backward interaction localized

- The direct preserved/current trace comparison supersedes the earlier claim
  that entry forward tensors were identical. With scalar RMS forward, the first
  divergence is layer-0 RMS output (`1.1920928955078125e-7`), before attention.
- Comparing the preserved passing trace `gradient-probe-m_39_kfv` with the
  rejected eight-lane RMS candidate `gradient-probe-n4go458e` proves **all 66
  common forward fields identical**. Its remaining 37 differing fields are
  backward. Evidence: `.peft-recovery-20260927/minimax-prior-vs-vector.jsonl`;
  `compare_minimax_traces.py --forward-only` now separates these categories.
- Combining vector RMS with the recovered SLEEF/exact-reciprocal attention
  arithmetic improves MiniMax M2 embedding A to `2.086162567138672e-7`, still
  FAIL; MiniMax M3 is `2.384185791015625e-7` (regression), GPT-OSS
  `1.1920928955078125e-7`. This combined candidate is not qualified.
- Rebuilding the full recovered attention source produces SPIR-V `2f0f6e04...`,
  not the recorded passing `450bd468...`, and gives the same MiniMax M2 failure.
  Source identity alone is insufficient to recover the previous arithmetic.
- Extended `validation/diagnose_attention_rounding.py` to replay value adjoints
  from identical native operands. Combined candidate matches SLEEF probabilities
  with separate multiply/add exactly; the preserved passing trace is much closer
  to explicit FMA (3 differing values, max `9.313225746154785e-10`, versus 10
  values with separate multiply/add). This is localization evidence, not a gate.
- Testing explicit attention-adjoint FMA next. Entry source/SPIR-V rollback
  files are `.peft-recovery-20260927/resume-entry-{rms,attention}.{comp,spv}`.
  Do not promote or retain any candidate without full saved fixtures and
  regression gates. Overall PEFT remains **INCOMPLETE**.

## ACTIVE RECOVERY 2026-09-27: experiment regression audit in progress

This checkpoint supersedes older completion claims where fresh results disagree.
User request: inspect the last few days of logs, repair experimental regressions,
finish every documented native-green PEFT surface, and checkpoint here frequently.

- Live COMPATIBILITY inventory is **20 families / 32 surfaces** (not the historical 28).
- Entry source/shader snapshot, SHA-256 manifest and complete tracked diff are
  preserved under `hierarchos-vulkan/.peft-recovery-20260927/`. Do not reset this dirty tree.
- Historical log inventory is being generated by
  `python validation/audit_peft_experiment_logs.py --output .peft-recovery-20260927/historical-logs.json`.
  It retains every recent log's hash, timestamp, failure/metric evidence and tail;
  timestamps and old PASS reports are not proof of current binary qualification.
- Entry saved-module reports: 27/32 PASS, 5 FAIL. Failures: Gemma3 (head/embedding),
  Gemma4 (norm/head/embedding), GPT-OSS embedding A `2.384185791015625e-7`,
  MiniMax M2 embedding A `2.384185791015625e-7`, MiniMax M3 MoE embedding A
  `2.086162567138672e-7`. The last three conflict with earlier passing checkpoints
  and are regression candidates, requiring controlled source/binary A/B tests.
- Entry ordinary-LoRA report files say 32/32 PASS, but no per-surface
  `multi-adapter/report.json` survives in the active fixture directories. Switch
  reports in old aggregate files must be refreshed, not silently inherited.
- Shared RMSNorm routing was broadened to CPU sqrt+reciprocal on multiple families;
  RMS forward/backward, RoPE, and attention experimental changes require exact
  source/SPIR-V provenance. Do not assume all failures have the same cause.
- Pre-change full native gate running: `cargo test --lib -j 1`, log
  `.peft-recovery-baseline-lib-20260927.log`. No production change made yet.
- Next: finish log classification, establish fresh baseline, localize/revert only
  demonstrated regressions, then rerun all independent gates. Fixed absolute
  ceiling remains `2e-7`, frozen base must be exactly zero. PEFT is **INCOMPLETE**.

### Recovery checkpoint: RoPE regression reproduced and repaired

- Fresh entry-state saved-embedding A diagnostics reproduced GPT-OSS
  `2.384185791015625e-7`, MiniMax M3 `2.086162567138672e-7`, MiniMax M2
  `2.384185791015625e-7` (logs under `.peft-recovery-20260927/baseline-*`).
- Recovered **exact** Sept 25 passing RoPE SPIR-V hashes by restoring SLEEF
  sine/cosine and separately rounded forward products:
  forward `10bbfa2187a5e144b099c03da0c1d412446ed7d2738958461633a01f56c4a0c8`,
  backward `9309d4fce244fdb17a1e62db911ac1f13a2176ca5a8c1d923e3c8871c4ccc455`.
  The Sept 26 19:52 session had replaced both with ordinary GLSL trig and
  removed the forward `precise` intermediates, despite the qualified baseline.
- With recovered RoPE, GPT-OSS and MiniMax M3 probes both improve to
  `1.7881393432617188e-7`; their **complete saved-module fixtures now PASS**,
  including A/B gradients, two-step AdamW, bank isolation, exact frozen base,
  save/reload and resume. Full all-stage/shared-kernel regression remains due.
- MiniMax M2 remains `2.384185791015625e-7`; Gemma3 embedding probe is
  `7.152557373046875e-7`. Gemma4 standalone diagnostic found missing old adapter
  scratch rather than a numeric result; full fixture regenerates its inputs.
- Recovered exact Sept 25 RMS forward source `9af68eb9...` and compiled artifact
  `f5b7e6d7...` from historical session output. It used an **eight-lane variance
  reduction** for unfused RMS families. Current source incorrectly claims the
  passing MiniMax M2/Qwen2 fixture used scalar reduction. Controlled RMS A/B next.
- Matrix runner now preserves prior aggregate reports, records shader/binary
  SHA-256 provenance, checkpoints each completed row, and supports explicitly
  separate selected-surface reports. Partial runs never replace the full matrix.

### Recovery checkpoint: RMS candidate rejected; attention backward regression localized

- The eight-lane RMS-forward candidate alone did not improve MiniMax M2
  (`2.384185791015625e-7`) and regressed restored MiniMax M3 from
  `1.7881393432617188e-7` to `2.384185791015625e-7`. It is **REVERTED**.
  Candidate source/SPIR-V and four probe logs remain in `.peft-recovery-20260927`.
  Do not retain it merely because an older qualified source used the same order.
- Compared MiniMax M2 current inputs against the preserved passing
  `minimax_m2-strict.prior-1790325368652683700/gradient-probe-m_39_kfv`:
  base weights, fixture, training inputs and saved adapter payload are identical.
  Native forward norm means/rstd/outputs, final prefix, logits adjoint and final
  norm input adjoint are exactly equal in the serialized traces. Divergence is
  downstream in backward, including attention output adjoint (~7e-9).
- Recovered exact historical attention-backward source `225b1823...` from Sept 25
  session output. Its softmax used SLEEF exp, exact reciprocal, and precise sum/dot
  unconditionally. A narrow restoration preserving the current Gemma-specific
  QK/repeated-head branches was tested and **rejected**: MiniMax M2 remained
  `2.384185791015625e-7`; GPT-OSS measured `1.7881393432617188e-7`, MiniMax M3
  `1.8998980522155762e-7`, Gemma3 `5.960464477539062e-7`, and Gemma4
  `1.7881393432617188e-6`. Probe logs are `attention-restore-*.log` under
  `.peft-recovery-20260927/`.
- The rejected attention-backward candidate was reverted to the preserved entry
  source. Recompiling with Vulkan SDK `glslc 1.4.321.1` reproduces the entry
  SPIR-V SHA-256 exactly at `fe24b5983596ba525fa54f4b5257ee7deea1bc01aad22c9c08a548321ba499c6`.
  A rebuilt validator and fresh MiniMax M2 probe
  `.peft-oracle-fixtures/minimax_m2-strict/gradient-probe-bsdnw0j7/` again
  measure `2.384185791015625e-7`, confirming the rollback and ruling out that
  softmax-arithmetic change as the MiniMax M2 repair.
- `diagnose_peft_saved_gradient.py` now regenerates the deterministic A/B payload
  in its own probe directory. It no longer depends on stale `multi-saved-*`
  scratch that the full fixture stopped creating. This repairs the Gemma4
  diagnostic setup failure without weakening or altering the oracle.
- Full restored-RoPE saved fixtures: GPT-OSS and MiniMax M3 PASS; Gemma3 still
  fails head/embedding; Gemma4 still fails norm/head/embedding. These are not
  all-stage promotions; fresh switch and broad base/full-training gates remain due.

### Recovery checkpoint: MiniMax M2 regression repaired and all-stage green

- The later scoped MiniMax-M2 attention-backward candidate is now validated on
  the live source/artifact pair rather than inferred from an older saved run.
  Current `transformer_attention_backward.comp` / `.spv` SHA-256 values are
  `372dfe8f7993c9ea0461406d9191b5ba4ca8a6c276e881b4b920d0a02af2f0d1` /
  `1490182b3cc83ba674049013b6fccc8c846b21223b25f2dc3215d7988234b326`.
- Fresh deterministic saved-embedding probes on that exact binary pass both
  adapters at the unchanged `2e-7` ceiling: adapter A
  `1.341104507446289e-7`, adapter B `1.7881393432617188e-7`.
- Fresh architecture row:
  `py -3 validation\verify_peft_green_matrix.py --stage all --jobs 1 --families minimax_m2`
  exits 0 with `lora=true`, `saved=true`, `switch=true` and no failures.
  This includes the complete saved-module lifecycle/bank gate, not just the
  narrow embedding diagnostic.
- MiniMax M2 is therefore restored to current-tree qualification. GPT-OSS and
  MiniMax M3 saved-module recovery remain green from the restored RoPE gate;
  Gemma3 and Gemma4 are the remaining saved-module numerical blockers before a
  fresh full 32-surface matrix can be used as completion evidence.

### Recovery log classification and entry matrix (2026-09-27)

Full scan: **404 logs**, **56 containing failure text** since 2026-09-23 UTC.
The JSON evidence inventory includes matching line numbers and hashes, including
rejected experiments, build failures, driver errors and completed passing runs.
Session patch evidence additionally preserved in `session-shader-evidence.txt`.

| Defect / experiment | Evidence | Entry-state interpretation |
| --- | --- | --- |
| Frozen BERT bias gradient state and Llama4 packed target discovery | `.peft-regression-current.log`, `.peft-regression-cli-cache-20260923.log`: 664 pass / 3 fail | Previously repaired; fresh full suite 667 PASS includes all three tests |
| Concurrent/incomplete Cargo state (`peft_rs` missing), misplaced `self` edit | `.peft-regression-20260923.log`, `.peft-switch-gpt_oss.log`, `.peft-switch-smollm3.log` | Build failures historical; current three validation binaries build successfully |
| Checkpoint/bank I/O Access denied and path length/collision | SmolLM3 rope-entry, Qwen4 bank and GPT-OSS rope-restore logs | Short unique scratch/checkpoint directories already implemented; full lifecycle reruns still required |
| K2.5 embedding target rejected | `.peft-saved-k25-live.err.log` | Historical validator gap; active saved report passes, needs current regression |
| K3 saved gradients and bank | `.peft-saved-k3-*`, `.peft-situ-mixed-saved.log` | Active K3 reports pass, need current regression |
| Narrow GEMM / materialized RMS / small-angle RoPE | `.peft-narrow-*`, `.peft-materialized-*`, `.peft-rope-small-cos-*` | Rejected numerical candidates; not proof of a bank bug |
| DeepSeek bank device allocation failure | `.peft-deepseek-saved-retry-20260923.log`: ERROR_OUT_OF_DEVICE_MEMORY | Infrastructure/allocation failure; do not call it adapter leakage without evidence |
| Vulkan incompatible driver | `.peft-restored-qwen4_exp_mixed_ple-20260924.log` | Infrastructure failure, not numerical evidence |
| Missing Gemma trace field | `.peft-gemma4-current-diagnostic-20260926-rerun.log` | Diagnostic/binary mismatch, not a passing fixture |
| Sept 26 RoPE restore removed SLEEF and separately rounded forward products | session 19:52 patch, baseline diagnostics | Reconstructed precise SLEEF forward SPIR-V `10bbfa21...`, backward `9309d4fc...` match Sept 25 passing artifacts; A/B underway |
| GPT-OSS / MiniMax M2 / MiniMax M3 saved embeddings | latest rerun logs plus fresh baseline diagnostics | Reproduced 2.384e-7 / 2.384e-7 / 2.086e-7; regression repair required |
| Gemma3 / Gemma4 saved modules | current saved reports and older experiments | Unresolved qualification failures; do not claim newly introduced regressions without a passing baseline |

Below are **entry-state historical report values**, not requalification of the
current shaders. Every row still needs fresh switch evidence and independent full
training/base regression; frozen-base, resume, merge and bank remain separate gates.

| Surface | Base logits max | PEFT logits max | Two-step adapter AdamW max | Saved modules | Switch report |
| --- | --- | --- | --- | --- | --- |
| deepseek_v4 | 4.385856627875384e-08 | 3.55264663659538e-08 | 1.0244548320770264e-08 | historical PASS | missing |
| falcon_h1_gated_after | 2.344215488392365e-08 | 3.1445350656689186e-08 | 3.725290298461914e-09 | historical PASS | missing |
| falcon_h1_gated_before | 3.0151649468845854e-08 | 2.4064426423775487e-08 | 3.725290298461914e-09 | historical PASS | missing |
| falcon_h1_silu | 2.5884132393017012e-08 | 3.446510316029716e-08 | 3.725290298461914e-09 | historical PASS | missing |
| gemma3 | 4.5099603655784914e-08 | 4.233936309883113e-08 | 1.862645149230957e-08 | FAIL: lm_head, model.embed_tokens | missing |
| gemma4 | 5.672987937743734e-08 | 7.433939361312092e-08 | 2.514570951461792e-08 | FAIL: model.layers.0.input_layernorm, lm_head, model.embed_tokens | missing |
| gpt2 | 4.8249282846857255e-08 | 3.3757095341258037e-08 | 1.3969838619232178e-08 | historical PASS | missing |
| gpt_oss | 2.954738236027321e-08 | 2.8133010865349917e-08 | 1.3969838619232178e-08 | FAIL: model.embed_tokens | missing |
| kimi_k25_text | 4.189449309688342e-08 | 3.179298399547115e-08 | 1.0244548320770264e-08 | historical PASS | missing |
| kimi_k3_kda | 1.5733871455703152e-08 | 2.0394535060752617e-08 | 9.778887033462524e-09 | historical PASS | missing |
| kimi_k3_mla | 1.1064491267642751e-08 | 1.2964668283377279e-08 | 1.862645149230957e-09 | historical PASS | missing |
| kimi_k3_text | 1.473552894287744e-08 | 2.9924331665376336e-08 | 6.51925802230835e-09 | historical PASS | missing |
| llama | 3.022907257843599e-08 | 3.310787199950482e-08 | 1.5832483768463135e-08 | historical PASS | missing |
| minimax_m2 | 2.9609413143560914e-08 | 3.448200225353304e-08 | 1.955777406692505e-08 | FAIL: model.embed_tokens | missing |
| minimax_m3 | 3.30995178210447e-08 | 2.4840621948296615e-08 | 1.6763806343078613e-08 | FAIL: model.embed_tokens | missing |
| minimax_m3_dense | 3.806545258167038e-08 | 3.182159424852671e-08 | 1.955777406692505e-08 | historical PASS | missing |
| mistral4 | 2.8792800893517878e-08 | 2.5062885281035285e-08 | 1.4901161193847656e-08 | historical PASS | missing |
| mixtral | 1.7110061645286834e-08 | 2.98225402706187e-08 | 1.210719347000122e-08 | historical PASS | missing |
| phi3 | 3.4201889043128375e-08 | 2.8720550548921864e-08 | 1.862645149230957e-08 | historical PASS | missing |
| phi4_multimodal_text | 2.7132301322341235e-08 | 4.937858580356114e-08 | 2.2817403078079224e-08 | historical PASS | missing |
| qwen2_5_gqa | 2.6086883542220107e-08 | 2.0720558172038217e-08 | 1.4901161193847656e-08 | historical PASS | missing |
| qwen2_5_sliding_tied | 3.214177131394269e-08 | 3.4485473643375286e-08 | 1.792795956134796e-08 | historical PASS | missing |
| qwen3_5_full | 3.168264389730169e-08 | 2.3336162571774288e-08 | 8.381903171539307e-09 | historical PASS | missing |
| qwen3_5_linear | 2.8601608276224155e-08 | 3.060451508885009e-08 | 3.725290298461914e-09 | historical PASS | missing |
| qwen3_5_mixed | 4.556251526066646e-08 | 2.982803343209994e-08 | 8.381903171539307e-09 | historical PASS | missing |
| qwen3_5_moe | 2.6312408446527158e-08 | 1.8502470968545515e-08 | 9.313225746154785e-09 | historical PASS | missing |
| qwen3_next | 3.7380218531612996e-08 | 4.467144013031721e-08 | 1.0244548320770264e-08 | historical PASS | missing |
| qwen4_exp_linear | 7.071361537991372e-09 | 1.5197053906956803e-08 | 9.313225746154785e-10 | historical PASS | missing |
| qwen4_exp_mixed_ple | 1.0295066826815358e-08 | 1.4831523897917975e-08 | 3.725290298461914e-09 | historical PASS | missing |
| qwen4_exp_ple | 1.5942430493798287e-08 | 1.6200523381137977e-08 | 1.862645149230957e-09 | historical PASS | missing |
| qwen4_exp_qsa | 7.929553031554848e-09 | 1.3205089563483874e-08 | 9.604264050722122e-09 | historical PASS | missing |
| smollm3 | 3.379756927102129e-08 | 3.1756629936818825e-08 | 1.955777406692505e-08 | historical PASS | missing |

## ACTIVE RESUME 2026-09-25: Mixtral modules_to_save qualified

Mixtral now clears the fixed `2e-7` saved-module gate. This checkpoint
supersedes the older Mixtral saved-embedding exclusion below while preserving
the historical `2.086162567138672e-7` measurement as localization evidence.

- The complete Mixtral `modules_to_save` fixture passes saved
  `self_attn.o_proj`, `input_layernorm`, `lm_head`, and
  `model.embed_tokens`. The formerly failing saved embedding now measures
  adapter-A gradient max-abs `1.7881393432617188e-7` and adapter-B
  `5.960464477539063e-8`.
- Both embedding adapters keep frozen-base drift `0.0`, resume drift `0.0`,
  and replacement-bank isolation PASS. Their AdamW max-abs values are
  `2.0489096641540527e-8` and `1.862645149230957e-8`; HF reload drift
  remains below `4.08e-8`.
- Saved `lm_head` gradients are adapter-A `8.940696716308594e-8` and
  adapter-B `5.960464477539063e-8`; saved Linear and norm rows are also
  fully green at the unchanged gate.
- A fresh ordinary-LoRA regression remains PASS: gradient max-abs
  `1.3969838619232178e-9`, two-step AdamW max-abs
  `1.210719347000122e-8`, frozen-base drift `0.0`, uninterrupted resume
  drift `0.0`, matching trainable keys, and every exercised lifecycle value
  below `7.1e-8`.
- The passing saved-module result comes from the retained shared RMSNorm
  execution-topology work already in the tree; no Mixtral-specific
  replacement-bank workaround, tolerance change, or correction term was
  introduced for promotion.
- `VulkanTransformerArchitecture::Mixtral` is promoted to
  `modules_to_save_validated=true`. Gemma4 and Gemma3 remain independently
  fail-closed until their own complete saved-module gates pass.

## ACTIVE RESUME 2026-09-25: Qwen2.5 sliding+tied modules_to_save qualified

The Qwen2.5 sliding+tied surface now clears the fixed `2e-7` saved-module gate
without loosening tolerance or changing the recovered attention executable.
This checkpoint supersedes the older Qwen2 saved-module exclusion below while
preserving it as historical localization evidence.

- The retained fix extends the already-qualified residual-aware RMSNorm backward
  association to Qwen2 only while the base is frozen for PEFT. For the ordinary
  pre-norm, non-parallel, non-cross-attention residual graph, native now
  materializes the residual and RMSNorm backward contributions with the same FP32
  association used by the PyTorch oracle. MiniMax M2 continues to use the same
  helper; unrelated architectures are unchanged.
- The complete `qwen2_5_sliding_tied` `modules_to_save` fixture exits PASS for
  saved Linear, norm, `lm_head`, and `model.embed_tokens`. The formerly failing
  saved embedding now measures adapter-A gradient max-abs
  `1.862645149230957e-7` and adapter-B `8.940696716308594e-8`. Both adapters keep
  frozen-base drift `0.0`, resume drift `0.0`, and replacement-bank isolation
  PASS. Saved `lm_head` gradients are adapter-A `1.1920928955078125e-7` and
  adapter-B `8.940696716308594e-8`; all exercised AdamW/reload values remain
  below the unchanged gate.
- The recovered attention executable baseline is still unchanged:
  `transformer_attention_forward.spv` remains SHA-256
  `e54a4a1636a581caa77fb9385d9f31af4b0b5493e520fc9bf65b1066442fad04`.
  The surviving source remains the reconstructed `f0031767...` text, not the
  unavailable historical `04d45643...` source snapshot.
- Fresh ordinary-LoRA regression on `qwen2_5_sliding_tied` passes with gradient
  max-abs `3.725290298461914e-9`, two-step AdamW max-abs
  `1.792795956134796e-8`, frozen-base drift `0.0`, uninterrupted resume drift
  `0.0`, matching trainable keys, and every exercised lifecycle value below
  `5.82e-8`.
- Fresh ordinary-LoRA regression on the independent `qwen2_5_gqa` surface also
  passes: gradient max-abs `1.6298145055770874e-9`, two-step AdamW max-abs
  `1.4901161193847656e-8`, frozen-base drift `0.0`, uninterrupted resume drift
  `0.0`, and every exercised lifecycle value below `4.0e-8`.
- Shared Rust regressions remain green before promotion:
  `cargo test modules_to_save --lib -j 1 -- --nocapture` is 14/14 PASS and
  `cargo test lora_ --lib -j 1 -- --nocapture` is 32/32 PASS.
- `VulkanTransformerArchitecture::Qwen2` is promoted to
  `modules_to_save_validated=true`; the stale `2.980232238769531e-7`
  sliding+tied exclusion is removed. Mixtral, Gemma4, and Gemma3 remain
  independently fail-closed until their own complete saved-module gates pass.

## ACTIVE RESUME 2026-09-25: MiniMax M2 modules_to_save qualified

MiniMax M2 now clears the fixed `2e-7` saved-module gate. This checkpoint
supersedes the older MiniMax M2 fail-closed conclusions below while preserving
them as historical localization evidence.

- The retained fix preserves PyTorch's FP32 residual/RMSNorm backward
  association for MiniMax M2 PEFT by materializing the combined RMS input
  adjoint in `(residual + direct) + through` order. The residual-aware helper
  belongs to `VulkanLayerNorm`; moving the interrupted copy out of
  `VulkanDeepseekV4Attention` restored a clean
  `cargo check --lib -j 1`.
- The recovered attention executable baseline is unchanged:
  `transformer_attention_forward.spv` remains SHA-256
  `e54a4a1636a581caa77fb9385d9f31af4b0b5493e520fc9bf65b1066442fad04`.
  The dedicated residual RMS backward source/SPIR-V pair hashes to
  `eb4ff22df741cc3a8c44062d76247bac09300448dc1e49890337fb6a1d2891ae` /
  `b353ad4a716d2513303294d32ecb3b858ed0c7afe8f8ec07291f04395f4e6aed`;
  Vulkan SDK `glslc 1.4.321.1` with the repository's default compile
  invocation reproduces that SPIR-V exactly.
- Fresh narrow saved-embedding adapter-A diagnostic
  `.peft-oracle-fixtures/minimax_m2-strict/gradient-probe-m_39_kfv/`
  passes at gradient max-abs `1.7881393432617188e-7`, down from the prior
  `2.384185791015625e-7`. The previously worst element is now
  `-1.0610789060592651` native versus `-1.0610790252685547` HF,
  exactly one FP32 ULP (`1.1920928955078125e-7`). The rebuilt validator
  recorded SHA-256
  `5cd280935e0892b2bc748519b45d7f1902dec488bc7308c4f3be99830ad403c3`.
- The complete MiniMax M2 `modules_to_save` fixture exits PASS for saved
  Linear, norm, `lm_head`, and `model.embed_tokens`. Saved embedding
  gradients are `1.7881393432617188e-7` for both adapter A and adapter B;
  both adapters keep frozen-base drift `0.0`, resume drift `0.0`, and
  replacement-bank isolation PASS. Embedding AdamW max-abs is
  `2.421438694000244e-8` for both adapters.
- The architecture's ordinary strict LoRA regression remains PASS:
  gradient max-abs `7.450580596923828e-9`, two-step AdamW max-abs
  `1.955777406692505e-8`, frozen-base drift `0.0`, uninterrupted resume
  drift `0.0`, matching trainable key sets, and all exercised
  save/reload/merge/unmerge lifecycle maxima below `5.3e-8`.
- Focused Rust regressions pass after promotion:
  `cargo test modules_to_save --lib -j 1 -- --nocapture` is 14/14 PASS and
  `cargo test lora_ --lib -j 1 -- --nocapture` is 32/32 PASS.
- `VulkanTransformerArchitecture::MiniMaxM2` is promoted to
  `modules_to_save_validated=true`. The user-facing
  `architectures --peft --json` output reports that value as true and no
  longer emits the stale MiniMax M2 numerical exclusion. Mixtral, Qwen2,
  Gemma4, and Gemma3 remain independently fail-closed on their documented
  saved-module failures.

## ACTIVE RESUME 2026-09-25: attention artifact reconstruction before MiniMax M2

This is the rolling continuation checkpoint after the interrupted attention
experiment. It does not supersede the completed Qwen4 qualification or the
32-surface ordinary-LoRA refresh below.

- The interrupted attention experiment has now been restored to the exact
  previously recorded executable baseline. Recompiling
  `transformer_attention_forward.comp` with Vulkan SDK `glslc 1.4.321.1`
  produces SPIR-V SHA-256
  `e54a4a1636a581caa77fb9385d9f31af4b0b5493e520fc9bf65b1066442fad04`.
  The surviving source text hashes to
  `f0031767bef841801e9ffa8e33b5a3481091cef219e8aba298bc206936ca7e19`;
  an exhaustive checkout search found no surviving copy of the historical
  `04d45643...` source bytes. Treat the SPIR-V identity, not source-text hash
  equality, as the recovered executable baseline.
- `cargo build -q --bin transformer_parity -j 1` passes on that rebuilt
  artifact (warnings only). The rebuilt validator SHA-256 is
  `fd5957179eefb8533974685e8c26d0bcb6420a7adb138e99223c41088688e55e`.
- A fresh isolated MiniMax M2 saved-embedding adapter-A diagnostic at
  `.peft-oracle-fixtures/minimax_m2-strict/gradient-probe-lsjymr5n/`
  reproduces the recorded known-good failure exactly:
  gradient max-abs `2.384185791015625e-7`, worst tensor
  `base_model.model.model.embed_tokens.weight`, with the worst element at
  the same `-1.0610787868499756` native versus `-1.0610790252685547`
  HF pair. The diagnostic records the rebuilt validator and exact
  `e54a4a16...` attention SPIR-V.
- Fresh attention/source replay from that trace localizes one upstream
  operation-order difference: sequential shader FMA score reduction differs
  from torch CPU matmul by at most `1.7881393432617188e-7` on visible
  multi-key rows, while materializing FP32 products before addition matches the
  torch scores exactly. On identical scores, the replayed softmax matches torch
  exactly; end-to-end attention forward differs by only
  `1.4901161193847656e-8`.
- The complete MiniMax replay further shows layer-0 Q/K/V projection-backward
  and RoPE-backward local replays at only roughly `1e-10` to `1e-9`, while
  the layer-0 attention/norm branch is already `1.341104507446289e-7` from
  HF and the final layer-0 input total reaches
  `2.384185791015625e-7`. Residual addition ordering can independently move
  individual elements by up to `1.1920928955078125e-7`. These are
  localization observations only; no new arithmetic candidate is qualified.
- A targeted attention-forward candidate replaced the QK and probability-value
  FMA reductions with `precise` separately rounded FP32 multiply/add
  operations, matching the reduction form that the CPU replay showed for torch
  matmul. Candidate source/SPIR-V hashes were
  `76cbcd00539faf1648392c713fc1843ede0bbf7189b9e4ff74e3570c5cf4b798` /
  `789bd8f4aee55eb172c333c9543c3ccbb4a590062142f919517309ded44418c3`.
  The isolated MiniMax M2 adapter-A probe
  `.peft-oracle-fixtures/minimax_m2-strict/gradient-probe-4tjybek1/`
  remained red at exactly `2.384185791015625e-7` on the same worst tensor and
  `-1.0610787868499756` versus `-1.0610790252685547` worst element. The
  candidate therefore did not qualify and was reverted.
- A second narrow A/B removed only `MiniMaxM2` from
  `peft_uses_unfused_rms_backward`, selecting the existing materialized-autograd
  PEFT RMS backward while keeping the recovered `e54a4a16...` attention
  executable. The fresh probe
  `.peft-oracle-fixtures/minimax_m2-strict/gradient-probe-hgk0tkwb/`
  again measured exactly `2.384185791015625e-7` at the identical worst
  saved-embedding element. This selector change therefore does not fix MiniMax
  M2 and was reverted.
- Qwen4 Experimental saved-module qualification is already complete and promoted,
  and the restored-shader ordinary-LoRA refresh is already 32/32 PASS. Do not
  rerun or relabel those completed gates unless a retained shared-kernel change
  requires regression evidence.
- The last fresh MiniMax M2 saved-embedding adapter-A probe on the exact recorded
  known-good attention pair
  `04d45643a0551a428e4db312260831b200c87107b845e3852ee4c1cd823c66e9` /
  `e54a4a1636a581caa77fb9385d9f31af4b0b5493e520fc9bf65b1066442fad04`
  remains just red at `2.384185791015625e-7`.
- A later attention-forward candidate produced source/SPIR-V hashes
  `5945fbc611a88895a9799a2ac6d1094af052a63b1ee94b6e5cc89d39e805d98d` /
  `4401830aa1864168140876d09997a34dae7eb4000f3893462578008b70324089`.
  Its preserved MiniMax M2 probe
  `.peft-oracle-fixtures/minimax_m2-strict/gradient-probe-_f1vl3cj/diagnostic.json`
  also measured `2.384185791015625e-7`; this candidate did not qualify the row.
- The subsequent interrupted edit/recompile left the live attention-forward pair
  at source `075144ed67cb5b129d3ec99b5bde774a1f0422b5ebf037296d782594e638c701`
  and SPIR-V `ff76d88d5657d9ee28b9ef6e1883ba991d6d790125fa3c9f1aec300f3f7af952`.
  That pair is not the recorded known-good executable baseline and must not be
  used to promote any saved-module row merely because its source looks restored.
- Continue MiniMax M2 from first-divergent-operation localization. Any retained
  attention/RoPE/RMSNorm arithmetic change must first improve the narrow
  saved-gradient probe below `2e-7`, then pass the complete MiniMax M2
  saved-module fixture and the affected shared-kernel ordinary-LoRA regressions
  before registry metadata changes.

## ROOT-CAUSE CHECKPOINT 2026-09-25: remaining saved-gradient failures are FP32 execution-topology mismatches

The remaining `modules_to_save` numerical failures are no longer best classified
as replacement-bank, embedding-scatter, optimizer-state, or adapter-lifecycle
bugs. The accumulated evidence identifies the cause class as strict FP32
execution-topology mismatch between the Hugging Face/PyTorch CPU oracle and the
native Vulkan shader path.

- Tiny upstream differences are introduced by mathematically equivalent but
  differently ordered FP32 operations, including RoPE/trigonometric evaluation,
  projection/reduction order, fused versus separately-rounded multiply/add,
  reciprocal/sqrt/rsqrt evaluation, and multi-branch gradient accumulation.
- These seed differences can be only one or a few ULPs and remain harmless in
  forward inference, but sensitive backward operations can amplify them. Gemma4
  currently provides the clearest measured example: raw Q/K/V attention
  adjoints differ only around `1e-8`; post-RoPE Q/K already show roughly
  `1e-7`-scale differences; Q/K/V RMSNorm backward then amplifies the incoming
  mismatch into approximately `1e-5` internal input-gradient differences; after
  projection/residual accumulation, the layer-input discrepancy returns to the
  `1e-6` scale observed at the saved embedding gradient.
- The embedding replacement/scatter is therefore frequently only the final
  observable surface. Where the incoming embedding/model adjoint already
  differs, do not rewrite the embedding bank or scatter unless new evidence
  proves that boundary introduces an independent mismatch.
- The Qwen4/MiniMax M3 A/B experiment strongly supports this diagnosis. A
  small-angle RoPE candidate improved a narrow Qwen4 probe but regressed
  MiniMax M3. Restoring the entry-state helper returned MiniMax M3 saved-
  embedding gradient parity to `1.7881393432617188e-7`; with the restored path,
  Qwen4 later passed complete saved-module qualification, including mixed+PLE
  saved-embedding adapter-A gradient `5.960464477539063e-8` and adapter-B
  `2.9802322387695312e-8`.
- This means there may not be one universal algebraic bug shared by every
  remaining family. The common problem is exact FP32 operation ordering; the
  instruction-level seed/amplifier can differ by architecture. Gemma4's most
  strongly localized amplifier is Q/K/V RMSNorm backward. Gemma3 and MiniMax M2
  still require first-divergent-operation localization rather than assuming the
  identical instruction is responsible.

### Critical source/binary reproducibility warning

`build.rs` embeds existing `.spv` artifacts and does not compile edited GLSL.
Source-equivalent-looking experiments can therefore execute different numerical
programs if the embedded SPIR-V differs. Qualification must record and compare
the exact shader source hash, SPIR-V hash, validator binary, and trace input.

During the current investigation the known-good attention replay recorded:

- `transformer_attention_forward.comp` SHA-256
  `04d45643a0551a428e4db312260831b200c87107b845e3852ee4c1cd823c66e9`
- `transformer_attention_forward.spv` SHA-256
  `e54a4a1636a581caa77fb9385d9f31af4b0b5493e520fc9bf65b1066442fad04`

Do not treat a GLSL revert alone as restoration of this numerical baseline.
Recompile explicitly, verify the resulting artifact identity, rebuild the
validator, and only then compare oracle results. A changed SPIR-V hash is a
different executable floating-point program even when the source looks
algebraically equivalent.

### Guidance for the next agent

Treat the remaining work as instruction-level backward reproducibility work:

1. Find the first native/HF intermediate that diverges, rather than compensating
   at the final saved embedding gradient.
2. Compare exact operation and reduction order around RoPE, projection backward,
   RMSNorm backward, residual/hyper-state additions, and other converging
   gradient branches.
3. Preserve the absolute `max_abs <= 2e-7` gate and exact frozen-base `0.0`.
4. Do not promote higher precision, tolerance changes, architecture-specific
   fudge factors, or a fix that merely moves another architecture above the
   ceiling.
5. For every candidate, first run the narrow failing saved-gradient probe, then
   the complete architecture saved-module fixture, then the affected ordinary-
   LoRA/shared-kernel regressions before updating registry metadata.

This root-cause classification supersedes older notes that treated the remaining
saved-embedding failures as potentially originating in replacement-state or
embedding-scatter mechanics, except where newer direct evidence proves otherwise.

## CURRENT CHECKPOINT 2026-09-25: Qwen4 saved-module qualification restored and promoted

This checkpoint supersedes the older Qwen4 mixed+PLE failure recorded below.
The fixed gate remains absolute `max_abs <= 2e-7`, with frozen-base drift
exactly `0.0`.

- The small-angle RoPE candidate was rejected after an A/B comparison showed it
  regressed MiniMax M3 saved-embedding gradient parity. Restoring the entry-state
  helper returned MiniMax M3 to `1.7881393432617188e-7`; the candidate remains
  reverted.
- On the restored shader, complete Qwen4 Experimental `modules_to_save` runs are
  fresh and PASS for all four documented native-green surfaces: linear/DeltaNet,
  sparse QSA, PLE, and mixed+PLE. Each exercised saved hyper-connection Linear,
  `lm_head`, and `model.embed_tokens`; every reported gradient and two-step AdamW
  max-abs is below `2e-7`, both adapters have resume drift `0.0`, frozen-base drift
  `0.0`, and bank isolation is true.
- Qwen4 mixed+PLE's saved embedding, the former blocker, now measures adapter-A
  gradient `5.960464477539063e-8` and adapter-B gradient
  `2.9802322387695312e-8`. The other Qwen4 saved-embedding adapter-A maxima are
  linear `2.9802322387695312e-8`, QSA `5.960464477539063e-8`, and PLE
  `5.960464477539063e-8`.
- The restored-shader ordinary-LoRA regression refresh completed across all 32
  required native-green PEFT surfaces. Every per-surface
  `.peft-rope-entry-lora-*-20260924.log` contains a final `pass=true`; this includes
  all four Qwen4 surfaces and the MiniMax M3 surfaces affected by the rejected
  candidate.
- `VulkanTransformerArchitecture::Qwen4Exp` is therefore promoted to
  `modules_to_save_validated=true`. This promotion is Qwen4-specific evidence;
  Mixtral, Qwen2 sliding+tied, Gemma4, MiniMax M2, and Gemma3 remain independently
  fail-closed until their own complete saved-module gates pass.
- The one-step saved-gradient diagnostic now records artifact hashes, and the
  MiniMax replay accepts an explicit trace-file path so future A/B comparisons can
  identify the exact validator binary, shader artifacts, and trace input used.

## CURRENT CHECKPOINT 2026-09-24: rebuilt baseline, CLI verified, registry synchronized

This checkpoint supersedes earlier same-day candidate/source-binary notes where
they conflict with the live tree. The fixed gate remains absolute
`max_abs <= 2e-7`, with frozen-base drift exactly `0.0`.

- Rejected narrow-GEMM and SLEEF RoPE runtime candidates were removed. The
  three affected SPIR-V files were explicitly rebuilt with Vulkan SDK
  `glslc 1.4.321.1`, then `transformer_parity` was rebuilt. Focused source
  diff for `linear_forward.comp` and both RoPE shaders is empty.
- Fresh rebuilt-baseline saved-embedding adapter-A probes: MiniMax M2
  `2.384185791015625e-7`, Qwen4 mixed+PLE `2.384185791015625e-7`, Gemma3
  `9.5367431640625e-7`, Gemma4 `2.6226043701171875e-6`.
- MiniMax M2/Qwen4 diagnostics localize the one-ULP failures upstream of the
  embedding scatter: the mismatch is already present in the model/layer-input
  adjoint. Their optimizer/resume/frozen-base/bank lifecycle checks remain
  green in the completed selected reports.
- Final PEFT CLI revalidation on the live tree is green. `cargo check
  --manifest-path hierarchos-native-cli/Cargo.toml -j 1` exited 0; the focused
  `peft_cli` test passed; `architectures --peft --json` exited 0; and
  `validation/verify_peft_cli.py` against the freshly built CLI returned
  `pass=true`, `auto_merge_max_abs=2.000000000335067e-08`, exact resumed
  weights/moments, `model_merge_alias=true`, and merged generation matching the
  unmerged adapter path. The user-facing controls are `--peft`,
  `--peft-config`, HF-style direct LoRA flags, `--peft-parameter-percent`, and
  `--lora-merge` / `--lora-model-merge`.

## ACTIVE RESUME 2026-09-24: explicit shader compilation and CLI revalidation

This checkpoint supersedes the source/binary assumptions below. Work is still
in progress; no candidate or stale full-matrix report is a completion claim.

- `build.rs` embeds existing `.spv` files and does **not** compile GLSL. Both
  live `linear_forward.comp` and its disassembled `.spv` contained the rejected
  lane-staged branch at entry. Rebuilt all four validator binaries successfully.
- Current-source narrow-branch selected embedding A/B gradient maxima:
  Gemma3 `1.1920928955078125e-6` / `2.980232238769531e-7`;
  Gemma4 `1.0728836059570312e-6` / `8.344650268554688e-7`;
  MiniMax M2 `2.384185791015625e-7` / `1.4901161193847656e-7`.
  These three complete selected runs have bank isolation PASS. Qwen4 mixed-PLE
  stopped during bank checkpointing with Windows `Access is denied` (not a
  measured bank-isolation mismatch). Logs: `.peft-resume-narrow-*-20260924.log`.
- Removed the branch, explicitly ran `glslc`, then rebuilt the validators.
  Sequential-FMA selected A/B gradients: Gemma3 `8.344650268554688e-7` /
  `3.3527612686157227e-7`; Gemma4 `3.2186508178710938e-6` /
  `1.0728836059570312e-6`; MiniMax M2 `2.384185791015625e-7` /
  `1.7881393432617188e-7`. All three banks passed. Qwen4 mixed-PLE aborted with
  transient `ERROR_INCOMPATIBLE_DRIVER` while four processes ran concurrently.
  Use at most two concurrent Vulkan oracles for the next runs. Logs:
  `.peft-restored-*-20260924.log`. The historical GEMM comparison is therefore
  not reproducible verbatim with the rest of today's source.
- Added a SLEEF 3.6.1 scalar small-argument FP32 trig candidate with its Boost
  license, and compiled forward/backward RoPE. With sequential FMA it does NOT
  qualify Gemma: embedding A/B are Gemma3 `8.344650268554688e-7` /
  `4.172325134277344e-7`, Gemma4 `2.5033950805664062e-6` /
  `1.6093254089355469e-6`. Banks pass. A detailed Gemma4 norm trace shows
  post-RoPE Q improved to `3.725290298461914e-9`, but K is
  `1.4901161193847656e-7` and projection rounding remains different.
- **Candidate currently under test:** trig plus narrow GEMM restricted to
  complete reduction tiles and output dimensions 2/3/4/8, the shapes matched
  by the independent small-GEMM diagnostic. Do not retain this as a runtime
  fix without fresh qualification/regression evidence. Build log:
  `.peft-trig-tiles-build-20260924.log`.
- Added `validation/diagnose_peft_saved_gradient.py` for one-step iteration
  without overwriting full qualification reports. Direct saved-module oracle
  runs now remove their own stale report/progress before starting; selected
  runs still never replace a full saved-module report.
- Added established GPT-2, Llama, Mixtral and Qwen3-Next baseline rows to the
  documentation-driven inventory. The complete matrix now has **32** required
  surfaces (28 newer surfaces plus those four established native-green ones).
- Existing requested CLI flags were preserved: `--peft`, `--peft-config`,
  direct HF LoRA controls, `--peft-parameter-percent`, and `--lora-merge` /
  `--lora-model-merge`. Current CLI tests/build are running, with logs at the
  repository root `.peft-cli-tests-20260924.log` and `.peft-cli-build-20260924.log`.
  Previous CLI smoke evidence remains historical until the new smoke runs.

## LATEST SYNC 2026-09-24: live saved-module baselines, source/binary mismatch, and Gemma backward localization

This is the newest continuation checkpoint. It records findings that were not
present in the audit before the 2026-09-24 resume. The fixed qualification
boundary is unchanged: every independently exercised numerical surface must
satisfy absolute `max_abs <= 2e-7`, with frozen-base drift exactly `0.0`.

### Important source/binary synchronization warning

- The earlier handoff below says the rejected lane-staged narrow-GEMM
  experiment in `shaders/linear_forward.comp` was reverted and that the file
  was back to the original sequential
  `fma(x, weight, sum)` reduction.
- That is **not true of the live working tree currently on disk**. The current
  `linear_forward.comp` still contains the experimental branch for
  `rows > 1`, `input_dim >= 8`, and `1 < output_dim <= 8`, with 4- or
  8-lane staged accumulation before falling through to the original FMA path.
- Therefore do not assume that the current shader source and the existing
  prebuilt validation binaries are synchronized. The fresh selected-row runs
  below intentionally used `HIERARCHOS_PEFT_USE_PREBUILT=1`; they establish
  the behavior of those binaries, not a rebuilt-current-source qualification.
  Before treating any shader edit as causal, rebuild the relevant binaries and
  rerun the same narrow rows.
- The rejected global lane-staged reduction still must not be promoted as a
  fix merely because it remains in the dirty tree. Earlier measured evidence
  showed that it failed to qualify Gemma3 and materially regressed Gemma4.

### Fresh live selected saved-embedding rows from the current prebuilt binaries

These runs were executed directly on 2026-09-24 with
`HIERARCHOS_PEFT_USE_PREBUILT=1` and the strict external
`validation/verify_peft_saved_modules.py` oracle.

- **Gemma3 / `model.embed_tokens`** remains numerically unqualified:
  - base logits max-abs: `5.97198581697711e-08`
  - adapter-A logits max-abs: `1.6528148651107077e-07`
  - adapter-B logits max-abs: `6.024295806650315e-08`
  - adapter-A gradient max-abs:
    `1.1920928955078125e-06`, worst tensor
    `base_model.model.model.embed_tokens.weight`
  - adapter-B gradient max-abs:
    `2.980232238769531e-07`, same worst tensor
  - adapter-A / adapter-B AdamW max-abs:
    `3.5390257835388184e-08` / `3.259629011154175e-08`
  - resume drift: `0.0` for both adapters
  - frozen-base drift: `0.0` for both adapters
  - HF reload max-abs:
    `7.38726806670531e-08` / `4.136446953018069e-08`
  - interleaved adapter-bank isolation: **PASS**
- **Gemma4 / `model.embed_tokens`** remains numerically unqualified:
  - base logits max-abs: `3.5842838287669654e-08`
  - adapter-A logits max-abs: `6.664863586025582e-08`
  - adapter-B logits max-abs: `1.0005918503475097e-07`
  - adapter-A gradient max-abs:
    `1.0728836059570312e-06`, worst tensor
    `base_model.model.model.embed_tokens.weight`
  - adapter-B gradient max-abs:
    `8.344650268554688e-07`, same worst tensor
  - adapter-A / adapter-B AdamW max-abs:
    `3.3527612686157227e-08` / `3.725290298461914e-08`
  - resume drift: `0.0` for both adapters
  - frozen-base drift: `0.0` for both adapters
  - HF reload max-abs:
    `1.0321326637287298e-07` / `5.93976554885689e-08`
  - interleaved adapter-bank isolation: **PASS**

These fresh rows strengthen the existing diagnosis: Gemma3/Gemma4 replacement
state, optimizer stepping, resume, frozen-base isolation, adapter switching and
reload behavior are healthy. The failing gate is specifically the first-step
native-vs-HF backward gradient.

### Gemma4: first large amplification is inside Q/K/V RMSNorm backward

The existing narrow Gemma4 trace was re-read in detail. Its useful localization
was not previously captured in this audit:

- Layer input and the saved input-layernorm **forward** operands are exact:
  native input, norm output, rstd, and mean-squared all show `0.0` drift.
  The input-layernorm incoming gradient is already different by
  `1.0728836059570312e-06`.
- The raw attention adjoints are still close to HF:
  - grad-Q max-abs `1.862645149230957e-08`
  - grad-K max-abs `1.30385160446167e-08`
  - grad-V max-abs `2.7939677238464355e-08`
  - replaying the native operands through torch gives only
    `1.862645149230957e-09` max-abs for each Q/K/V attention adjoint.
- Q/K/V RMSNorm **forward** inputs, rstd values, and outputs are all exact
  against HF, while their incoming output-gradients differ only slightly:
  about `2.05e-08`, `1.49e-08`, and `2.79e-08` respectively.
- Those tiny incoming differences are strongly amplified by the RMSNorm
  backward at the near-epsilon Q/K/V norms:
  - Q-norm input-gradient max-abs:
    `1.52587890625e-05`
  - K-norm input-gradient max-abs:
    `1.0013580322265625e-05`
  - V-norm input-gradient max-abs:
    `2.0503997802734375e-05`
- The native shader reconstruction reproduces the native RMSNorm input
  gradients exactly, so this is not a trace/export artifact. Substituting the
  exact `pow(-1.5)` form also does not remove the mismatch because the
  incoming adjoint operands already differ.
- After projection-backward accumulation, Q+K reaches
  `1.3709068298339844e-06` and Q+K+V reaches
  `1.0728836059570312e-06`, matching the scale of the saved embedding /
  layer-input failure.
- The remaining upstream arithmetic difference is already visible before this
  amplification: post-RoPE Q is
  `2.384185791015625e-07` from HF and post-RoPE K is
  `1.1920928955078125e-07` from HF. This makes the rotary/trigonometric path
  a legitimate next diagnostic target, but **not yet a qualified fix**.

### CPU torch trigonometric reference path inspected: SLEEF, not generic GLSL trig

The local CPU reference implementation used by torch was inspected to avoid
guessing about the post-RoPE one-ULP differences:

- The relevant SLEEF scalar functions are `xsinf_u1` and `xcosf_u1`.
  For the normal argument range they do compensated argument reduction using
  the split `PI_A2f`, `PI_B2f`, and `PI_C2f` constants, followed by a
  polynomial evaluated with SLEEF's two-float helpers.
- The helpers include compensated `dfadd2`, `dfadd`, `dfsqu`, and
  `dfmul` operations; the scalar implementation splits FP32 values with
  `upperf(x) = bits(x) & 0xfffff000` for the error terms.
- By contrast, the current native rotary shaders still obtain
  `c = cos(angle)` and `s = sin(angle)` directly from GLSL before applying
  the otherwise explicit FP32 rotation.
- This establishes a concrete reference-order difference that can plausibly
  explain a one-ULP post-RoPE discrepancy on AMD. It does **not** establish
  causality yet. Any SLEEF-equivalent Vulkan/GLSL trig candidate must be tested
  narrowly on Gemma3/Gemma4 and then against already-green PEFT/base surfaces
  before it can be retained.

### Current math-helper source state relevant to the next experiment

- `shaders/fp32_sqrt_recip.glsl` already contains integer-assisted FP32
  helpers for directly rounded divide / reciprocal-square-root and
  separately-rounded sqrt-then-reciprocal behavior. The current RMSNorm
  forward/backward shaders use these helpers for architecture-specific
  rounding paths.
- `shaders/fp32_tanh.glsl` already contains a small-input compensated
  polynomial helper, while `tanh_forward.comp` and `tanh_backward.comp`
  currently embed their own narrower small-input reference series. No new
  tanh change was qualified during this 2026-09-24 continuation.

### Registry metadata is stale relative to the newest saved-module evidence

The live `src/transformer/peft_registry.rs` was checked against this audit.
Ordinary LoRA qualification is already advertised for the current native-green
families, but several `modules_to_save` fields/exclusion strings still reflect
older numerical runs:

- `Qwen35` and `Qwen35Moe` are still absent from
  `modules_to_save_validated`, and their exclusion strings still cite the old
  `2.384185791015625e-7` saved-embedding failures. The newer audit evidence
  below records full saved-module PASS after selecting the materialized
  autograd RMS backward, with Qwen3.5 full/MoE saved-embedding gradients at
  most `1.1920928955078125e-7`.
- `MiniMaxM3VLText` is still absent from `modules_to_save_validated` and its
  exclusion still cites `2.086162567138672e-7`. The newer audit evidence
  records the complete saved-module suite passing, with sparse/indexer
  embedding A/B gradients `1.8440186977386475e-7` /
  `1.1920928955078125e-7`.
- The still-failing Gemma3, Gemma4, and MiniMax M2 registry exclusion strings
  also contain older failure magnitudes rather than the fresh/restored-kernel
  measurements recorded in the newest checkpoints. Their fail-closed status
  remains correct; only the evidence text is stale.

Do not infer capability from the stale strings in either direction. Registry
promotion/text refresh must follow the measured audit state, and any promotion
should happen only after the corresponding complete saved-module fixture is
confirmed green on rebuilt current-source binaries.

The requested PEFT CLI controls are already recorded below and were not
duplicated here: HF-style `--peft-config` / direct LoRA controls,
`--peft-parameter-percent`, and `--lora-merge` /
`--lora-model-merge` are present with the previously recorded CLI smoke.

## HANDOFF NOTE 2026-09-23: narrow-GEMM experiment rejected and reverted

This is a historical arithmetic-diagnosis checkpoint and is superseded by the
2026-09-24 sync above wherever the live working-tree/source state conflicts.
It still supersedes the earlier "narrow GEMM candidate is under test" text
below for the measured experiment outcome.

- The shared `linear_forward.comp` narrow-projection reduction experiment was
  tested against the remaining saved-module failures and **did not qualify**.
  The experiment replaced the normal sequential FP32 FMA accumulation for
  small output dimensions with lane-staged reductions intended to mimic a CPU
  narrow-GEMM reduction order.
- Gemma3 remained outside the fixed `2e-7` gradient ceiling. In the saved
  embedding case, adapter A reached
  `8.344650268554688e-7` gradient max-abs and adapter B reached
  `2.384185791015625e-7`; frozen-base drift and resume drift remained exactly
  `0.0`, so this is still a numerical backward-parity issue rather than state
  contamination or optimizer-resume failure.
- Gemma4 regressed materially under the same shared narrow-GEMM candidate.
  The saved embedding case reached
  `1.6689300537109375e-6` (adapter A) and
  `1.0132789611816406e-6` (adapter B) gradient max-abs. Another saved-module
  case recorded `4.172325134277344e-7` on
  `model.layers.0.self_attn.q_proj.lora_B.weight`. Frozen-base and resume drift
  remained `0.0` in the recorded cases.
- Because the shared forward-kernel change worsened the relevant saved-module
  gradients, it was **rejected and reverted**. `linear_forward.comp` is back to
  the original sequential `fma(x, weight, sum)` reduction. Do not reintroduce
  the lane-staged narrow-GEMM branch as a global fix based on the current
  evidence.
- The useful diagnostic finding is narrower: the CPU reference can use a
  balanced/non-fused reduction order for some narrow projections, but simply
  applying that order globally in native forward execution is not the source
  of the remaining PEFT saved-module gradient mismatch. Continue tracing the
  architecture-specific backward/adjoint path instead.

The already-qualified Qwen3.5 and MiniMax M3 saved-module fixes remain the
current good changes: Qwen3.5 full/MoE saved embedding gradients are at most
`1.1920928955078125e-7`, and MiniMax M3 sparse/indexer reaches
`1.8440186977386475e-7` / `1.1920928955078125e-7` for adapters A/B, all with
exact `0.0` frozen-base and resume drift and passing adapter-bank isolation.

Fresh restored-kernel saved-module reports:

- MiniMax M2: saved linear, input norm, and `lm_head` all pass. Only
  `model.embed_tokens` remains outside tolerance: adapter A gradient
  `2.384185791015625e-7`, adapter B `1.7881393432617188e-7`. The failing
  tensor is the saved embedding weight itself. Both adapters keep exact
  `0.0` frozen-base and resume drift and pass bank isolation.
- Qwen4 Experimental mixed+PLE: the saved hyper-connection mixer projection
  and `lm_head` pass. Only `model.embed_tokens` remains outside tolerance:
  adapter A gradient `2.384185791015625e-7`, adapter B
  `5.960464477539063e-8`. Both adapters keep exact `0.0` frozen-base and
  resume drift and pass bank isolation.

These two failures are now both one-ULP-over-ceiling-style saved-embedding
gradient mismatches on adapter A, with all lifecycle/state checks green. Do
not broaden them into ordinary-LoRA, optimizer-state, or bank-isolation bugs.

## LIVE CONTINUATION: Qwen3.5 backward candidate and narrow GEMM diagnosis

The pasted Qwen3.5 continuation has been resumed. The existing CLI percentage,
HF-config controls and automatic merge implementation were preserved.

- Removing Qwen35/Qwen35Moe from `peft_uses_unfused_rms_backward` selects the
  existing materialized-autograd RMS backward. It does NOT remove architecture
  support. Full saved-module runs now pass on both surfaces, with embedding
  gradient A/B maxima `1.1920928955078125e-7` on each.
- The same change for MiniMaxM3VLText passes the complete saved-module suite;
  embedding A/B gradient maxima are `1.8440186977386475e-7` /
  `1.1920928955078125e-7`. All three runs have exact zero frozen-base and resume
  drift and passing interleaved A/B bank isolation.
- Experiments applying this backward selection to Qwen4/MiniMaxM2 did not
  improve their saved embedding failure and were reverted. Gemma3's additional
  norm experiment did not qualify the surface and was also reverted.
- New diagnostic `validation/diagnose_peft_linear_order.py` identifies exact
  reference reduction order for Gemma4's narrow K/V and LoRA-A projections.
  `validation/diagnose_small_gemm.py` independently reproduces the arithmetic
  on random matrices. A shared `linear_forward` narrow-GEMM candidate is now
  under test; do NOT treat this candidate as qualified until fresh matrix and
  base-regression evidence is recorded below.

Current full-matrix evidence still needs refreshing after these changes.
Remaining saved-module failures before the GEMM candidate: Qwen4 mixed-PLE,
MiniMaxM2, Gemma3 and Gemma4. Do not count numerical failures as exclusions.

## ACTIVE CONTINUATION 2026-09-23: CLI controls and fresh full matrix

The previous interrupted SmolLM3 trace has now been consumed. Fresh
`validation/verify_peft_saved_modules.py smollm3` passed Linear, norm, lm_head
and embedding. Saved-embedding A/B gradient maxima are both
`1.7881393432617188e-7`; frozen base and uninterrupted-vs-resumed drift are
exactly zero. `_tmp_smollm3_embed_trace.py` confirms layer-0/1/2 input adjoints
at `1.7881393433e-7`, layer 3 at `1.1920928955e-7`; the embedding gather does
not add drift. Current source already contained the successful SmolLM3
unfused-RMS-products candidate; it was preserved, not redesigned.

New CLI implementation:

- `--peft-config FILE` reads HF adapter configuration for new adapters.
- Direct rank/alpha/dropout, target list/regex, exclude list, layer selection,
  rank/alpha patterns, and existing saved-module controls.
- `--peft-parameter-percent P` selects a uniform rank for a trainable-adapter
  budget, per the user's explicit choice. Counts native modules without
  downloading weights; verifies actual attached count before training.
  Denominator: serialized floating base tensor elements, duplicate tied head
  counted once. Unsupported nonuniform/saved-module budgeting fails explicitly.
- `--lora-merge` / `--lora-model-merge` saves adapter plus exact continuation
  state, then invokes the existing Rust merge into `OUT/merged-model`.
  Unsharded base required; existing destination rejected before training.
- CLI `cargo test --lib -j 1`: **33 passed**.
- `validation/verify_peft_cli.py` passed new-config training, direct config
  controls, percentage training, resume, automatic merge, and full fine-tuning.
  GPT-2 budget: rank 4, 512/5120 elements = exactly 10%; merged vs unmerged
  native logits `2.000000000335067e-8`. Evidence:
  `hierarchos-vulkan/.peft-oracle-fixtures/peft-cli-y1nqm6t5/report.json`.

The saved-module oracle now runs `peft_bank_parity` for EVERY saved module on
EVERY surface, checking interleaved A/B weights, gradients, optimizer moments,
disabled-base output and reload/switch/resume exactly. Frozen-base checks now
explicitly require `0.0` instead of accepting the general tolerance.

The full 28-surface matrix (all stages) is being rerun from compiled binaries:
`HIERARCHOS_PEFT_USE_PREBUILT=1 py -3 validation/verify_peft_green_matrix.py --stage all --jobs 4`.
Log: `hierarchos-vulkan/.peft-all-matrix-20260923.log`. Do not infer completion
from older saved reports while this run is in progress. `--jobs` parallelizes
independent fixture directories; each surface's stages remain sequential.
An earlier saved-only run was interrupted during CLI compilation and is not a
qualification result; use the all-stage run above.

## NEWEST CHECKPOINT: 28-surface architecture-wide PEFT matrix

This checkpoint supersedes the older 23-surface / 14-PASS matrix checkpoint
below. Future handoffs must use this section first when deciding what remains.

Fresh command/result preserved from the current tree:

- `py -3 validation\\verify_peft_green_matrix.py`
- 28 fixture surfaces were exercised.
- **Ordinary LoRA stage: 28/28 PASS.**
- **Named-adapter switch/disable stage: 28/28 PASS.**
- **`modules_to_save` / saved-module stage: 14/28 PASS, 14/28 FAIL or not-run.**
- Therefore the remaining matrix failures are **saved-module qualification only**.
  They must not be described as ordinary-LoRA failures.

Current per-surface state:

| Surface | Ordinary LoRA | `modules_to_save` / saved | switch/disable | Current status |
| --- | --- | --- | --- | --- |
| DeepSeek V4 | PASS | PASS | PASS | fully green for exercised matrix stages |
| Phi-4 Multimodal text | PASS | PASS | PASS | fully green for exercised matrix stages |
| Phi-3 | PASS | PASS | PASS | fully green for exercised matrix stages |
| Kimi K2.5 text | PASS | PASS | PASS | fully green for exercised matrix stages |
| Kimi K3 KDA | PASS | FAIL | PASS | saved-module-only failure |
| Kimi K3 MLA | PASS | FAIL | PASS | saved-module-only failure |
| Kimi K3 mixed text | PASS | FAIL | PASS | saved-module-only failure |
| GPT-OSS | PASS | PASS | PASS | fully green for exercised matrix stages |
| SmolLM3 | PASS | FAIL | PASS | saved-module-only failure |
| Qwen2.5 GQA | PASS | PASS | PASS | fully green for exercised matrix stages |
| Qwen2.5 sliding+tied | PASS | NOT RUN | PASS | saved-module stage still needs execution |
| Qwen3.5 full attention | PASS | FAIL | PASS | saved-module-only failure |
| Qwen3.5 pure linear | PASS | PASS | PASS | fully green for exercised matrix stages |
| Qwen3.5 mixed/hybrid | PASS | PASS | PASS | fully green for exercised matrix stages |
| Qwen3.5 MoE | PASS | PASS | PASS | fully green for exercised matrix stages |
| Qwen4 Experimental linear/DeltaNet | PASS | NOT RUN | PASS | saved-module stage still needs execution |
| Qwen4 Experimental sparse QSA | PASS | NOT RUN | PASS | saved-module stage still needs execution |
| Qwen4 Experimental PLE | PASS | NOT RUN | PASS | saved-module stage still needs execution |
| Qwen4 Experimental mixed+PLE | PASS | NOT RUN | PASS | saved-module stage still needs execution |
| Mistral 4 | PASS | PASS | PASS | fully green for exercised matrix stages |
| MiniMax M3 dense | PASS | PASS | PASS | fully green for exercised matrix stages |
| MiniMax M3 sparse/indexer | PASS | FAIL | PASS | saved-module-only failure |
| Gemma 4 | PASS | FAIL | PASS | saved-module-only failure |
| MiniMax M2 | PASS | FAIL | PASS | saved-module-only failure |
| Gemma 3 text | PASS | FAIL | PASS | saved-module-only failure |
| Falcon H1 gated-after | PASS | PASS | PASS | fully green for exercised matrix stages |
| Falcon H1 gated-before | PASS | PASS | PASS | fully green for exercised matrix stages |
| Falcon H1 SiLU | PASS | PASS | PASS | fully green for exercised matrix stages |

The persisted report is
`hierarchos-vulkan/.peft-oracle-fixtures/green-matrix-report.json`. It confirms
that every one of the 28 surfaces has `lora.pass=true` and `switch.pass=true`.
The 14 current aggregate failures correspond exactly to saved-module results.

Important saved-module failure details already measured in that report:

- Kimi K3 KDA: linear, norm, and embedding saved-module checks pass; `lm_head`
  gradient drift fails (`2.980232238769531e-7` adapter A,
  `4.172325134277344e-7` adapter B).
- Kimi K3 MLA: linear, norm, and embedding checks pass; `lm_head` gradient drift
  fails (`2.5331974029541016e-7` A, `2.980232238769531e-7` B).
- Kimi K3 mixed text: linear, norm, and embedding checks pass; `lm_head`
  gradient drift fails (`2.980232238769531e-7` for both adapters).
- SmolLM3: saved linear, norm, and `lm_head` pass; `model.embed_tokens` fails on
  adapter-A gradient drift `2.5331974029541016e-7` (adapter B is green at
  `1.1920928955078125e-7`).
- Qwen3.5 full attention: saved linear, norm, and `lm_head` pass;
  `model.embed_tokens` adapter-B gradient drift is `2.384185791015625e-7`
  (adapter A is green at `1.1920928955078125e-7`).
- MiniMax M3 sparse/indexer: saved linear, norm, and `lm_head` pass;
  `model.embed_tokens` adapter-A gradient drift is
  `2.086162567138672e-7` (adapter B is green at `5.960464477539063e-8`).
- Gemma 4: saved linear passes, but saved `input_layernorm`, `lm_head`, and
  `model.embed_tokens` gradient parity fail. The largest recorded failure is
  `input_layernorm` adapter-A gradient drift `3.129243850708008e-6`;
  embedding A/B are `2.1457672119140625e-6` / `1.6689300537109375e-6`.
- MiniMax M2: saved linear and norm pass. `lm_head` adapter-A gradient drift is
  `2.086162567138672e-7`; embedding A/B are
  `3.5762786865234375e-7` / `1.7881393432617188e-7`.
- Gemma 3: saved linear, norm, and `lm_head` pass; `model.embed_tokens`
  gradient drift is `4.76837158203125e-7` for both adapters.
- Qwen2.5 sliding+tied and all four Qwen4 Experimental surfaces currently have
  saved-module status `not run`; their ordinary LoRA and switch/disable stages
  are already green and must not be re-debugged as LoRA failures.

Immediate continuation target after this checkpoint:

1. Keep all 28 ordinary-LoRA and switch/disable results green as regression
   gates; do not redo their architecture integration unless a later saved-module
   fix regresses them.
2. Finish the 14 remaining saved-module rows only: run the five not-run saved
   fixtures (Qwen2.5 sliding+tied + four Qwen4 surfaces), then fix the nine
   measured saved-module failures above.
3. Preserve the unchanged strict `max_abs <= 2e-7` gate and exact frozen-base
   invariance; do not special-case architecture tolerances.
4. Promote per-architecture registry metadata only from measured evidence. A
   surface being LoRA-green does not automatically qualify its
   `modules_to_save` capability.

This newer result also supersedes earlier text below that described Falcon H1 as
PEFT-unimplemented or Kimi/Qwen4/Gemma/MiniMax ordinary LoRA as still pending.
Those statements are historical and must not be used as current handoff state.

## Current continuation: complete discovery, qualification still INCOMPLETE

The root README headline table and COMPATIBILITY.md headline table enumerate
13 families. The root README additionally strictly qualifies Qwen2.5 and Falcon
H1/H1R; COMPATIBILITY.md additionally records Gemma 3 (whose dedicated
PROGRESS_GEMMA3_AUDIT.md confirms strict forward/training). Therefore the current
required set is **16 families**, not the previous 14-family/23-surface matrix.
GPT-2, Llama, Mixtral and Qwen3-Next remain existing regression coverage.
No implementation-only exclusion counts as completed PEFT qualification.

| Architecture | Base <=2e-7 | PEFT registered | LoRA forward / AdamW | save/load | switch/disable | merge/unmerge | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek V4 | documented | pending | prior PASS | prior PASS | pending | prior PASS | incomplete |
| Phi-4 Multimodal text | documented | pending | prior PASS | prior PASS | pending | prior PASS | incomplete |
| Phi-3 | documented | pending | prior PASS | prior PASS | pending | prior PASS | incomplete |
| Kimi K2.5 text | documented | pending | saved report PASS | PASS | pending | PASS | incomplete |
| Kimi K3 / KimiLinear text | documented | pending | KDA PASS; MLA/mixed running | KDA PASS | pending | KDA PASS | incomplete |
| GPT-OSS | documented | pending | prior PASS | prior PASS | pending | prior PASS | incomplete |
| SmolLM3 | documented | pending | prior PASS | prior PASS | pending | prior PASS | incomplete |
| Qwen2.5 text | documented | pending | GQA/sliding+tied prior PASS | prior PASS | pending | prior PASS | incomplete |
| Qwen3.5 | documented | pending | full/hybrid/MoE prior PASS; pure linear pending | prior PASS | pending | prior PASS | incomplete |
| Qwen4 Experimental | documented | pending | linear/QSA prior PASS; PLE blocked | partial | pending | partial | incomplete |
| Mistral 4 | documented | pending | prior PASS | prior PASS | pending | prior PASS | incomplete |
| MiniMax M3 text | documented | pending | dense prior PASS; indexer target mismatch | partial | pending | partial | incomplete |
| Gemma 4 | documented | pending | frozen export drops layer_scalar buffers | pending | pending | pending | incomplete |
| MiniMax M2 | documented | pending | oracle uses unmatched random router jitter | prior PASS | pending | prior PASS | incomplete |
| Gemma 3 text | documented | pending | pending | pending | pending | pending | newly discovered |
| Falcon H1/H1R | documented | explicitly unsupported | pending implementation | pending | pending | pending | newly discovered |

`modules_to_save` remains independently pending for all 16 required families;
the existing GPT-2/Llama/Qwen3-Next bank evidence does not qualify them.

Recovered handoff evidence: the existing K2.5 report passes with forward
`3.179298399547115e-8`, gradient `1.862645149230957e-9`, AdamW
`1.4901161193847656e-8`, zero frozen-base and native resume drift. Fresh K3 KDA
run: forward `3.109840393186736e-8`, gradient `4.656612873077393e-10`, AdamW
`6.51925802230835e-9`, zero frozen-base and native resume drift. K3 currently
qualifies only the tested `o_proj` subset, not its whole module graph.

2026-09-22 all-native-green PEFT matrix checkpoint — INCOMPLETE

This checkpoint supersedes any earlier statement that the requested green-architecture PEFT scope is complete. The repository-wide completion gate now follows the README/compatibility-derived strict native-green set rather than representative GPT-2/Llama/Mixtral/Qwen3-Next coverage. `hierarchos-vulkan/validation/verify_peft_green_matrix.py` currently expands that set into 23 concrete native graph surfaces, including the additional strict-qualified Qwen2.5 text surfaces. The fixed absolute qualification ceiling remains `2e-7`; no tolerance was loosened.

Fresh matrix command actually run:
- `py -3 validation\\verify_peft_green_matrix.py`
- Result: 23 surfaces exercised; 14 PASS, 9 FAIL. The aggregate command exited 1 because every row must pass independently.

Fresh passing surfaces and strict maxima:

| Surface | LoRA forward max abs | Gradient max abs | 2-step AdamW max abs | Frozen base | Resume-vs-HF max abs | Lifecycle max abs | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DeepSeek V4 | `2.9526466369111937e-08` | `7.275957614183426e-10` | `9.313225746154785e-09` | `0.0` | `9.313225746154785e-09` | `4.6859474189808736e-08` | PASS |
| Phi-4 Multimodal text | `4.937858580356114e-08` | `1.3969838619232178e-09` | `1.6763806343078613e-08` | `0.0` | `1.6763806343078613e-08` | `5.000000000143778e-08` | PASS |
| Phi-3 | `2.8720550548921864e-08` | `1.3969838619232178e-09` | `1.862645149230957e-08` | `0.0` | `1.862645149230957e-08` | `4.428688049618046e-08` | PASS |
| GPT-OSS | `2.8133010865349917e-08` | `5.587935447692871e-09` | `1.30385160446167e-08` | `0.0` | `1.30385160446167e-08` | `3.257244110543844e-08` | PASS |
| SmolLM3 | `4.572349548981336e-08` | `3.1723175197839737e-09` | `2.60770320892334e-08` | `0.0` | `2.60770320892334e-08` | `7.999999998564711e-08` | PASS |
| Qwen2.5 GQA | `5.809974668036233e-08` | `4.190951585769653e-09` | `1.6763806343078613e-08` | `0.0` | `1.6763806343078613e-08` | `6.00000000239298e-08` | PASS |
| Qwen2.5 sliding+tied | `4.431552885764134e-08` | `3.725290298461914e-09` | `1.7695128917694092e-08` | `0.0` | `1.7695128917694092e-08` | `6.268951413535007e-08` | PASS |
| Qwen3.5 full attention | `3.2869949345704086e-08` | `9.313225746154785e-10` | `1.0244548320770264e-08` | `0.0` | `1.0244548320770264e-08` | `5.000000000143778e-08` | PASS |
| Qwen3.5 mixed/hybrid | `2.982803343209994e-08` | `1.862645149230957e-09` | `9.313225746154785e-09` | `0.0` | `9.313225746154785e-09` | `4.831390382209477e-08` | PASS |
| Qwen3.5 MoE | `1.5722084051317253e-08` | `1.1641532182693481e-09` | `1.1175870895385742e-08` | `0.0` | `1.1175870895385742e-08` | `3.309040069732205e-08` | PASS |
| Qwen4 Experimental DeltaNet/linear | `1.1762962340289285e-08` | `2.000888343900442e-11` | `9.313225746154785e-10` | `0.0` | `9.313225746154785e-10` | `1.1762962340289285e-08` | PASS |
| Qwen4 Experimental sparse QSA | `1.3205089563483874e-08` | `7.203198038041592e-10` | `9.42964106798172e-09` | `0.0` | `9.42964106798172e-09` | `1.491712760953723e-08` | PASS |
| Mistral 4 | `1.959629059022916e-08` | `1.862645149230957e-09` | `1.4901161193847656e-08` | `0.0` | `1.4901161193847656e-08` | `4.7639007572097825e-08` | PASS |
| MiniMax M3 dense | `3.490657807247999e-08` | `2.0954757928848267e-09` | `1.862645149230957e-08` | `0.0` | `1.862645149230957e-08` | `6.327934265470958e-08` | PASS |

Current failing surfaces and observed blockers:

| Surface | Failure / measured violation | Classification / next action |
| --- | --- | --- |
| Kimi K2.5 text | Native adapter load rejects the HF regex because the adapter targets `model.language_model.layers.*...` while the current DeepseekV3 PEFT traversal advertises `model.layers.*...`: `LoRA target_modules regex ... does not match a supported DeepseekV3 linear module`. | PEFT canonical-name/wrapper-prefix integration gap. Preserve the actual wrapper module path for target matching and adapter tensor lookup/export instead of treating K2.5 exactly like a bare DeepSeek-V3 checkpoint. |
| Kimi K3 KDA | Reference fixture fails before native execution in local Transformers `save_pretrained`: `_get_tied_weight_keys()` receives a `list` and calls `.keys()`. | Oracle/fixture serialization blocker, not yet a native PEFT numerical failure. Save the deterministic K3 base without triggering the broken tied-weight helper or normalize the local custom model metadata before serialization; then run the full strict PEFT suite. |
| Kimi K3 MLA | Same local Transformers `save_pretrained` tied-weight metadata failure (`AttributeError: 'list' object has no attribute 'keys'`). | Same oracle serialization blocker; no pass may be claimed until the strict suite runs end-to-end. |
| Kimi K3 mixed text | Same local Transformers `save_pretrained` tied-weight metadata failure (`AttributeError: 'list' object has no attribute 'keys'`). | Same oracle serialization blocker; then independently qualify the mixed KDA/MLA text surface. |
| Qwen4 Experimental PLE | Native training reaches checkpoint/base-output export and fails: `Qwen4 parameter_values cannot materialize the file-backed PLE embedding; use export_hf_package for a complete checkpoint`. | Harness/export integration gap. Frozen-base and training checks must use the complete package export path for file-backed PLE state rather than `parameter_values()`. |
| Qwen4 Experimental mixed+PLE | Same file-backed PLE checkpoint/base-output export failure. | Same harness/export integration gap; retest the mixed DeltaNet/QSA/PLE path independently after repair. |
| MiniMax M3 sparse/indexer | HF PEFT marks `base_model.model.model.layers.1.self_attn.indexer.q_proj.lora_A.default.weight` trainable but it receives no gradient. The native MiniMax-M3 indexer is a discrete routing side path and current backward intentionally does not manufacture an adjoint for the selector. | Target-discovery mismatch. The strict fixture must exclude non-differentiable indexer linears from ordinary LoRA target selection unless a legitimate differentiable PEFT path is implemented; do not fake a gradient. |
| Gemma 4 | Native PEFT training executes two steps, but frozen-base verification aborts on `tensor key mismatch` comparing the HF serialized base with the native `--base-output` package. | Checkpoint-key/alias normalization problem in the validation path until proven otherwise. Compare normalized canonical base tensors and verify actual value drift before classifying as a base mutation. |
| MiniMax M2 | Forward/lifecycle are green but strict training parity fails badly: gradient max `7.597776129841805e-05`, step-1 adapter drift `1.716800034046173e-05`, two-step AdamW drift `3.663916140794754e-05`, resume-vs-HF `3.663916140794754e-05`. Frozen-base drift is `0.0`, trainable key sets match, native uninterrupted resume drift is `0.0`, and lifecycle maxima remain `<=5.999999999617422e-08`. | Real architecture-specific PEFT backward/training parity failure. Diagnose attention LoRA gradients first, especially `model.layers.0.self_attn.o_proj.lora_B`, rather than hiding it behind successful inference/lifecycle checks. |

Important continuation state:
- Passing 14/23 matrix rows must not be reclassified from compilation alone; the numeric evidence above is the current strict result.
- Failing nine rows remain unqualified. Do not promote them in `peft_registry.rs` until their strict fixtures pass `<=2e-7` or a precise legitimate PEFT exclusion is documented.
- The current `peft_registry.rs` is still stale relative to this matrix: it advertises only GPT-2, Llama, Mixtral and Qwen3-Next. Once each README-native-green family is independently qualified, promote registry metadata per architecture rather than through a broad alias assumption.
- `modules_to_save` remains a separate gate. Existing qualified replacement-bank results for GPT-2/Llama/Qwen3-Next remain valid; Mixtral replacement modules remain fail-closed at the previously recorded `2.086162567138672e-7` saved-embedding gradient.
- The complete matrix artifact is `hierarchos-vulkan/.peft-oracle-fixtures/green-matrix-report.json`.

Immediate next debugging order: (1) Kimi K2.5 wrapper-aware canonical PEFT names; (2) K3 reference serialization so KDA/MLA/mixed can actually run; (3) Qwen4 PLE complete-package frozen-base export; (4) MiniMax M3 legitimate indexer exclusion/target discovery; (5) Gemma4 canonical frozen-base key comparison; (6) MiniMax M2 real LoRA backward parity. After each fix, rerun that family fixture before rerunning the whole 23-surface matrix.

2026-09-22 requested core-LoRA completion checkpoint

The requested green-architecture core LoRA/PEFT scope is now complete within the explicit qualification boundaries below. This is not a claim of universal PEFT feature coverage: non-core multi-adapter `trainable_token_indices`, repeated `layer_replication`, independently base-mutating OLoRA/PiSSA combinations, and any architecture that has not separately passed the base `<=2e-7` gate remain fail-closed.

Qualification surfaces:
- Ordinary LoRA is qualified on GPT-2, Llama, Mixtral, and Qwen3-Next for frozen-base training, trainable A/B, rank/alpha/scaling/dropout, target/suffix/regex/`all-linear` selection, layer filters, bias modes, named/multiple adapters, set/enable/disable, adapter-local optimizer progress, HF-compatible save/load, exact native resume, merge/unmerge, merge/export, and native inference/generation.
- Adapter-local `modules_to_save` is separately qualified on GPT-2, Llama, and Qwen3-Next. Linear, LayerNorm/RMSNorm where exposed, `lm_head`, and Embedding replacements preserve independent weights, gradients, AdamW moments, save/reload/resume state, switching, and disabled canonical-base restoration. Mixtral remains deliberately unqualified for `modules_to_save` because its saved-embedding adapter-A gradient measured `2.086162567138672e-7`, above the unchanged absolute gate.
- Full-parameter fine-tuning remains independently green on the exercised GPT-2/Llama/Mixtral regression set.
- Future architectures use two gates: first independent base parity `<=2e-7`, then explicit PEFT registry/module exposure plus the common PEFT fixture suite. Base-green never implies PEFT-green automatically.

Fresh generation evidence after the previous current-tree qualification refresh:
- Llama native-trained adapter: direct `transformer-generate` load/generate PASS; 14 LoRA modules loaded, deterministic one-token smoke output `hello`.
- Mixtral native-trained adapter: direct `transformer-generate` load/generate PASS; 8 LoRA modules loaded, deterministic one-token smoke output `hello`.
- Qwen3-Next native-trained adapter: direct `transformer-generate` load/generate PASS; 13 LoRA modules loaded, deterministic one-token smoke output `hello`.
- GPT-2 direct unmerged adapter generation and merged-package standalone generation were already PASS in the CLI lifecycle checkpoint below.

These generation smokes used copies of the existing 32-token tokenizer/generation metadata in the separate Codex scratch root; the repository fixtures and model math were not modified. The strict numerical evidence remains the fixed-gate oracle results recorded immediately below: all qualified ordinary-LoRA logits/gradients/two-step AdamW/lifecycle values are `<=2e-7`, the qualified saved-module fixtures are `<=2e-7`, frozen-base drift is `0.0`, and the full-training regressions remain below the same ceiling.

Final focused regression after this checkpoint documentation update:
- `cargo test modules_to_save --lib -j 1 -- --nocapture`: PASS, 14 passed / 0 failed.
- `cargo test lora_ --lib -j 1 -- --nocapture`: PASS, 32 passed / 0 failed.
- `cargo test transformer_generate_cli -- --nocapture`: PASS, 5 passed / 0 failed.
- `cargo run -q -- architectures --peft --json`: PASS; advertised ordinary-LoRA families remain GPT-2, Llama, Mixtral, and Qwen3-Next, with `modules_to_save_validated=true` only for GPT-2, Llama, and Qwen3-Next.

2026-09-22 current-tree green-only PEFT qualification refresh

This checkpoint refreshes the implemented PEFT lifecycle after the adapter-local `modules_to_save` bank and CLI lifecycle work below. The absolute qualification gate remains `max_abs <= 2e-7`; no tolerance was loosened and no non-green architecture implementation was modified. The dirty worktree and unrelated generated/architecture state were preserved.

Current qualification boundary:
- Ordinary native LoRA remains qualified on GPT-2, Llama, Mixtral, and Qwen3-Next.
- Adapter-local `modules_to_save` replacement state remains qualified only on GPT-2, Llama, and Qwen3-Next.
- Mixtral `modules_to_save` remains deliberately fail-closed because the previously measured saved-embedding adapter-A gradient was `2.086162567138672e-7`, above the unchanged `2e-7` gate. This continuation did not debug or special-case that path.
- The common replacement bank covers Linear, LayerNorm/RMSNorm, `lm_head`, and Embedding state where the architecture registry advertises those saved-module classes. Adapter weights, gradients, AdamW moments, switching, disabled-base restoration, save/reload, and resumed training remain adapter-local.

Fresh native evidence actually run on the current tree:
- `cargo test modules_to_save --lib -j 1 -- --nocapture`: PASS, 14 passed / 0 failed / 661 filtered. The named A/B isolation test passed for `attn.c_proj`, `ln_1`, `lm_head`, and `transformer.wte`, including independent training state, switching, disabled-base restoration, save/reload, and resumed training. The same filtered run also passed generic merged `modules_to_save` export and the untied-`lm_head` round-trip test.
- `cargo test lora_ --lib -j 1 -- --nocapture`: PASS, 32 passed / 0 failed / 643 filtered. Merge/unmerge, HF-style adapter export/reload, target/exclusion semantics, dropout, rank/alpha patterns, bias modes, optimizer resume, named-adapter switching, and frozen-base state release remain green. The tiny GPT-2 memory fixture reported `live_before=210104`, `live_after=174232`, `saved=35872`, `base_projection_training_state_before=3264`, and `adapter_training_state=768` bytes.
- `py -3 validation\\verify_hf_training.py --families gpt2 llama mixtral`: PASS against local Transformers `5.16.0.dev0`. Two-step full-fine-tuning parameter drift stayed below the PEFT gate as an independent regression: GPT-2 `3.166496753692627e-08`, Llama `5.960464477539063e-08`, Mixtral `1.1920928955078125e-07`.

Fresh strict external ordinary-LoRA oracles actually run on the current tree:
- GPT-2: PASS. `base_max_abs=4.8249282846857255e-08`, `peft_max_abs=3.3757095341258037e-08`, `gradient_max_abs=9.499490261077881e-08`, `two_step_adamw_max_abs=1.7462298274040222e-08`, frozen-base drift `0.0`, uninterrupted-resume drift `0.0`, and all reported save/reload/merge/unmerge lifecycle values remained below `5.28e-08`.
- Llama: PASS. `base_max_abs=3.022907257843599e-08`, `peft_max_abs=3.310787199950482e-08`, `gradient_max_abs=3.91155481338501e-08`, `two_step_adamw_max_abs=2.0489096641540527e-08`, frozen-base drift `0.0`, uninterrupted-resume drift `0.0`, and all reported lifecycle values remained below `4.91e-08`.
- Mixtral ordinary LoRA only: PASS. `base_max_abs=3.4411811816736915e-08`, `peft_max_abs=3.429634093388145e-08`, `gradient_max_abs=2.514570951461792e-08`, `two_step_adamw_max_abs=1.30385160446167e-08`, frozen-base drift `0.0`, uninterrupted-resume drift `0.0`, and all reported lifecycle values remained below `4.72e-08`. This does not promote Mixtral `modules_to_save`.
- Qwen3-Next: PASS. `base_max_abs=5.8302764893714e-08`, `peft_max_abs=3.4671440135580767e-08`, `gradient_max_abs=6.868503987789154e-09`, `two_step_adamw_max_abs=7.799826562404633e-09`, frozen-base drift `0.0`, uninterrupted-resume drift `0.0`, and all reported lifecycle values remained below `6.43e-08`.

Fresh strict external saved-module oracles actually run on the current tree:
- `py -3 validation\\verify_peft_saved_modules.py` (GPT-2): PASS, exit 0. Visible `lm_head` A/B gradient maxima were each `5.960464477539063e-08`; visible `transformer.wte` A/B gradient maxima were each `1.1920928955078125e-07`; adapter AdamW/reload values shown were below the gate; resume and frozen-base drift were `0.0`.
- `py -3 validation\\verify_peft_saved_modules.py llama`: PASS, exit 0. Visible `lm_head` gradients were A `1.1920928955078125e-07`, B `1.7881393432617188e-07`; visible `model.embed_tokens` gradients were A `1.7881393432617188e-07`, B `8.940696716308594e-08`; resume and frozen-base drift were `0.0`.
- `py -3 validation\\verify_peft_saved_modules.py qwen3_next`: PASS, exit 0. Visible `lm_head` gradients were A `1.1920928955078125e-07`, B `5.960464477539063e-08`; visible `model.embed_tokens` gradients were A `1.1920928955078125e-07`, B `8.940696716308594e-08`; resume and frozen-base drift were `0.0`.

Result: the immediate adapter-local replacement-state gap is closed and re-verified for the green saved-module families, ordinary LoRA remains green across the four qualified families, and full fine-tuning remains unchanged on its exercised regression set. Keep Mixtral `modules_to_save` unqualified until that independent saved-embedding gradient reaches `<=2e-7` without a Mixtral-specific PEFT workaround. Future architecture promotion still requires base parity first and then the common PEFT suite; registry presence alone is not qualification.

2026-09-22 native CLI lifecycle continuation (implemented and smoke-verified)

This section supersedes the older CLI-pending statements below. The strict `max_abs <= 2e-7` qualification gate was not changed, and no non-green architecture implementation was modified.

Implemented this continuation:
- `transformer-generate` now accepts `--lora-adapter-path` / `--peft-adapter` plus `--adapter-name`, loads the saved HF-compatible adapter directly into the native Vulkan graph, and runs generation without a merge step.
- Unmerged adapter generation is restricted to the independently PEFT-qualified registry. Adapter configs containing `modules_to_save` also pass through the separate `modules_to_save_validated` gate, so ordinary Mixtral LoRA remains available while Mixtral replacement-module adapters fail closed.
- CLI help/tests and `hierarchos-native-cli/README.md` now cover exact resume, direct unmerged inference, and native merge/export -> standalone generation.
- Added tiny deterministic CLI smoke assets `hierarchos-native-cli/tests/fixtures/peft_tokenizer.json` and `peft_generation_config.json` for the existing synthetic GPT-2 fixture only; these do not alter model math or parity tolerances.

Fresh CLI evidence actually run:
- `cargo check -q` in `hierarchos-native-cli`: PASS.
- `cargo test transformer_generate_cli -- --nocapture`: PASS, 5 passed / 0 failed. This includes named `--peft-adapter` parsing and fail-closed `--adapter-name` without an adapter.
- `cargo run -q -- architectures --peft --json`: PASS. Advertised ordinary LoRA families are GPT-2, Llama, Mixtral, and Qwen3-Next; `modules_to_save_validated=true` only for GPT-2, Llama, and Qwen3-Next, with Mixtral still false.
- Real GPT-2 CLI fine-tune smoke on the pinned local strict base: PASS, one native Vulkan step, mean loss `3.488848`; attached 4 LoRA modules and reported the Gpt2 base frozen. Output contained the adapter plus default `training-state`.
- Exact CLI resume from that `training-state`: PASS, resumed 4 LoRA modules, restored AdamW/RNG at step 1 with the saved optimizer settings, ran one further native Vulkan step with mean loss `3.478442`, and wrote a new adapter/checkpoint.
- The resumed adapter config contains `modules_to_save=["transformer.wte"]`; therefore the following inference/export smoke exercises a qualified saved Embedding replacement as well as LoRA.
- Direct unmerged CLI generation from the frozen GPT-2 base plus resumed adapter: PASS after loading 4 LoRA modules natively as adapter `smoke`; deterministic smoke output was `hello`.
- Native `merge-lora` on the same resumed adapter: PASS; `merged_modules=4`, `saved_module_tensors=1`. The output package was validated by the native Vulkan Transformer loader before publication.
- Standalone generation from the merged package: PASS; deterministic smoke output was `hello`.
- The first synthetic generation attempt hit only fixture metadata (`bos_token_id=50256` on a 32-token oracle vocabulary) after the adapter had already loaded successfully. A test-only generation config with in-range special-token ids resolved that fixture mismatch; no runtime architecture or numerical workaround was applied.

Current CLI lifecycle status for the exercised qualified GPT-2 fixture is now `base -> attach/train -> save -> reload/resume -> unmerged generate -> merge/export -> standalone generate`. Named adapter switching/disable isolation and strict save/reload/merge parity remain covered by the native/external PEFT tests recorded below rather than inferred from this text-generation smoke.

2026-09-22 adapter-local replacement continuation (implemented; strict qualification recorded below)

This section supersedes the older "immediate gap" description below. Dirty and generated state preserved. No non-green architecture implementation was changed or debugged.

Implemented this continuation:
- Wired existing peft_state.rs canonical base and adapter-local Vulkan handle banks into configure/set/disable/enable.
- Independent fresh Linear, LayerNorm, lm_head and Embedding clones; tied head restoration and input frontend rebinding.
- Adapter-local AdamW step/dropout progress, replacement moments and embedding-step checkpoint/resume; active adapter checkpoint works with multiple loaded adapters.
- HF GPT-2 saved Conv1D tensor orientation corrected at import/export.
- Direct native gradient readback for strict validation; export rejects disabled replacement state.
- Shared PEFT capability metadata attached to VulkanTransformerArchitecture, with separate runtime availability and validation fields.
- CLI work underway: --peft, --architecture, --adapter-name, --peft-resume, automatic training-state checkpoint and architectures --peft. cargo check of CLI passed before the latest RMS backward precision adjustment. End-to-end CLI validation is still pending.

Fresh evidence:
- cargo check --lib -j 1: PASS on the current tree.
- cargo test modules_to_save --lib -j 1 -- --nocapture: 13 passed, 0 failed. This includes named_modules_to_save_isolate_weights_gradients_moments_and_base, which exercised independent adapter-A/adapter-B weights, gradients, AdamW moments, switching, disabled-base restoration, and reload across Linear, LayerNorm, lm_head, and Embedding replacements.
- GPT-2 pinned-local external saved-module oracle (py -3 validation\verify_peft_saved_modules.py): PASS, exit 0. Worst observed saved-module gradient = 1.7881393432617188e-7; worst listed HF reload/logit drift = 7.223014832080654e-8; resume and frozen-base drift = 0.0.
- Llama pinned-local external saved-module oracle (py -3 validation\verify_peft_saved_modules.py llama): PASS, exit 0. The previously failing saved embedding is now green: adapter-A embedding gradient = 1.7881393432617188e-7, adapter-B = 8.940696716308594e-8; embedding AdamW <= 2.7939677238464355e-8; resume and frozen-base drift = 0.0.
- Qwen3-Next pinned-local external saved-module oracle (py -3 validation\verify_peft_saved_modules.py qwen3_next): PASS, exit 0. Saved embedding gradient <= 1.1920928955078125e-7; resume and frozen-base drift = 0.0.
- Mixtral pinned-local external saved-module oracle (py -3 validation\verify_peft_saved_modules.py mixtral): FAIL, exit 1, at the unchanged absolute 2e-7 gate. The remaining failure is adapter-A model.embed_tokens gradient = 2.086162567138672e-7. Its embedding logits, AdamW, reload, resume, frozen base, adapter-B gradient, and lm_head replacement checks are green. The deterministic embedding scatter was inspected and is not the source of this excess drift; no Mixtral-only architecture or tolerance workaround was applied.
- peft_registry.rs now promotes modules_to_save_validated only for GPT-2, Llama, and Qwen3-Next. Mixtral keeps replacement_bank_available = true for implementation/testing but remains modules_to_save_validated = false.
- Initial gradient inversion from AdamW updates was ill-conditioned for large replacement gradients; replaced with actual native Vulkan gradient readback. First direct GPT-2 bias discrepancy 2.384185791015625e-7 was resolved by using existing materialized log-softmax backward for PEFT (shared across architectures). Full-parameter CE route unchanged.
- A preliminary Mixtral oracle accidentally loaded installed Transformers against local-checkout packed expert weights, failed its base gate, and skipped PEFT. Harness now pins the same local Transformers and dependency paths as verify_peft_lora_strict.py. No base architecture workaround was made.

Current files touched this continuation:
hierarchos-vulkan/src/transformer.rs, transformer/peft_state.rs, transformer/peft_registry.rs (new), shared_lm_head.rs, lib.rs, src/bin/transformer_parity.rs; validation/verify_peft_multi_adapter.py, validation/verify_peft_saved_modules.py (new); hierarchos-native-cli/src/lib.rs. Existing shader files not modified by this continuation.

Next concrete tasks:
Keep Mixtral modules_to_save unqualified until the upstream saved-embedding gradient reaches <=2e-7 without an architecture-specific PEFT hack. The common adapter-local replacement bank itself is implemented and green on GPT-2, Llama, and Qwen3-Next. Next, finish CLI docs/tests and real fine-tune/resume/infer/merge/export smoke on those qualified families; then run focused green-only regressions and full-fine-tuning checks. Merged replacement export, untied head, and generation still need explicit current-turn evidence. Do not claim PEFT ready yet.


2026-09-22 strict architecture/lifecycle qualification update

This section supersedes older handoff statements below wherever they conflict. The current dirty worktree was preserved; no unrelated architecture, generated fixture, shader, or target state was cleaned or reset.

Fresh native packed-parameter adapter round-trip:

cargo test llama4_packed_target_parameters_round_trip_through_peft_adapter_export --lib -j 1 -- --nocapture

Result: 1 passed; 0 failed. The current Llama 4 PEFT `target_parameters` bridge round-trips packed `gate_up_proj` and `down_proj` expert adapters through native export -> native reload -> native re-export byte-identically on the exercised fixture.

Fresh strict external Hugging Face PEFT MoE qualification:

$env:HIERARCHOS_PEFT_ORACLE_FAMILY='mixtral'
python validation\\verify_peft_lora_strict.py

Result: PASS at the unchanged absolute `2e-7` gate.

base_max_abs = 3.4411811816736915e-08
peft_max_abs = 3.429634093388145e-08
lora_delta_max_abs = 4.6598129271879785e-08
gradient_max_abs = 2.514570951461792e-08
step1_adapter_max_abs = 1.862645149230957e-09
two_step_adamw_max_abs = 1.4260876923799515e-08
frozen_base_max_abs = 0.0
trainable_key_set_match = true
resume_uninterrupted_max_abs = 0.0
resume_hf_max_abs = 1.4260876923799515e-08

Mixtral lifecycle maxima:

hf-created_reload_hf_max_abs = 3.429634093388145e-08
hf-created_merged_unmerged_max_abs = 2.9999999984209325e-08
hf-created_merged_hf_max_abs = 3.429634093388145e-08
hf-created_unmerge_base_max_abs = 3.00000000119649e-08
native-created_reload_hf_max_abs = 3.6709451672889415e-08
native-created_save_reload_max_abs = 0.0
native-created_merged_unmerged_max_abs = 2.200000000229796e-08
native-created_merged_hf_max_abs = 3.4617004374126736e-08
native-created_unmerge_base_max_abs = 1.9999999989472883e-08
native-trained_reload_hf_max_abs = 4.7133789066178267e-08
native-trained_save_reload_max_abs = 0.0
native-trained_merged_unmerged_max_abs = 4.000000003445692e-08
native-trained_merged_hf_max_abs = 6.006782532486099e-08
native-trained_unmerge_base_max_abs = 3.999999997894577e-09

Fresh strict external Hugging Face PEFT hybrid/recurrent qualification:

Set-Item Env:HIERARCHOS_PEFT_ORACLE_FAMILY qwen3_next
python validation\\verify_peft_lora_strict.py

Result: PASS at the unchanged absolute `2e-7` gate.

base_max_abs = 5.8302764893714e-08
peft_max_abs = 3.4671440135580767e-08
lora_delta_max_abs = 5.829277038849057e-08
gradient_max_abs = 6.868503987789154e-09
step1_adapter_max_abs = 1.862645149230957e-09
two_step_adamw_max_abs = 9.313225746154785e-09
frozen_base_max_abs = 0.0
trainable_key_set_match = true
resume_uninterrupted_max_abs = 0.0
resume_hf_max_abs = 9.313225746154785e-09

Qwen3-Next lifecycle maxima:

hf-created_reload_hf_max_abs = 3.4671440135580767e-08
hf-created_merged_unmerged_max_abs = 2.000000001722846e-08
hf-created_merged_hf_max_abs = 4.20907974307827e-08
hf-created_unmerge_base_max_abs = 4.000000003445692e-08
native-created_reload_hf_max_abs = 4.4257354725463216e-08
native-created_save_reload_max_abs = 0.0
native-created_merged_unmerged_max_abs = 2.000000000335067e-08
native-created_merged_hf_max_abs = 4.4257354725463216e-08
native-created_unmerge_base_max_abs = 8.000000002728047e-09
native-trained_reload_hf_max_abs = 4.573051451406407e-08
native-trained_save_reload_max_abs = 0.0
native-trained_merged_unmerged_max_abs = 2.9999999984209325e-08
native-trained_merged_hf_max_abs = 6.420867920686746e-08
native-trained_unmerge_base_max_abs = 1.9999999989472883e-08

Fresh external named/multiple-adapter inference qualification:

python validation\\verify_peft_multi_adapter.py

Result: PASS.

adapter_a_max_abs = 3.5264015207925326e-08
adapter_b_max_abs = 3.350925445455566e-08
disabled_base_max_abs = 4.8249282846857255e-08
loaded_adapters = [adapter_a, adapter_b]
adapter_a_selected = true
adapter_b_selected = true
disabled_state = true

The external multi-adapter oracle verifies Hugging Face PEFT output parity for both selected adapters and disabled/base execution. Native trainability switching remains additionally covered by `named_lora_only_active_adapter_receives_adamw_updates`, which proves that the active adapter updates, the inactive adapter remains byte-identical, and switching transfers trainability without changing the frozen base.

Fresh full-parameter-training regression after the optional optimizer-state/PEFT refactor:

python validation\\verify_hf_training.py --families gpt2 llama mixtral

Result: PASS against the local Transformers checkout (`5.16.0.dev0`) for strict two-step native AdamW.

GPT-2 max parameter drift = 3.166496753692627e-08
Llama max parameter drift = 5.960464477539063e-08
Mixtral max parameter drift = 1.1920928955078125e-07

All three remain below `2e-7`. An attempted `--families gpt2 llama mixtral qwen3_next` invocation stopped before training because the legacy full-training validator does not enumerate `qwen3_next`; this is a validation-harness limitation, not a native-runtime failure. Qwen3-Next is independently covered above by the strict PEFT hybrid/recurrent oracle.

Fresh broad LoRA regression:

cargo test lora_ --lib -j 1 -- --nocapture

Result: 32 passed; 0 failed.

The run again measured the tiny GPT-2 PEFT memory reduction:

live_before = 210104 bytes
live_after = 174232 bytes
saved = 35872 bytes (~17.07%)
base_projection_training_state_before = 3264 bytes
adapter_training_state = 768 bytes

Current architecture qualification now covers a dense family (Llama), a routed-MoE family (Mixtral), and a hybrid/recurrent family (Qwen3-Next), in addition to the existing GPT-2 baseline. All measured deterministic logits, gradients, adapter updates, two-step AdamW updates, resume paths, merge/unmerge paths, native-created adapter interoperability, and frozen-base checks remain within the fixed `2e-7` gate on their exercised fixtures.

Core-LoRA completion caveat still open on 2026-09-22:

Do not yet claim exact PEFT semantics for every multi-adapter combination. Ordinary multiple LoRA adapters are externally qualified, but multiple adapters that each carry `modules_to_save` replacement modules still fail closed because the current graph stores those replacements directly rather than in an adapter-local replacement bank. Multiple adapters with independent `trainable_token_indices`, repeated `layer_replication`, and independently base-mutating OLoRA/PiSSA initialization also intentionally fail closed for the same lossless-state reason. The first item is a core-LoRA `modules_to_save` + multiple-adapter semantics gap and is the immediate implementation target; the other replacement/topology-mutating combinations must remain explicit failures until they have independent reversible state.

Advanced PEFT status:

The native tree already contains rsLoRA scaling and native DoRA forward/backward/AdamW plus adapter persistence tests. Single-adapter trainable-token/embedding state and extensive `modules_to_save` coverage are also present. IA3 and quantized-base/QLoRA are not yet implemented as production-native PEFT paths. Those remain follow-on work after the remaining core multi-adapter replacement-state gap is closed.

Immediate next task:

Implement adapter-local `modules_to_save` state so two or more named adapters can load independent replacement module tensors, switch trainability/state with `set_adapter`, disable back to the untouched base representation, save the selected adapter in normal PEFT layout, and preserve inactive adapter bytes. Add deterministic native and external Hugging Face PEFT parity coverage without weakening the `2e-7` gate.

Current-turn architecture qualification is in progress.

Continuation verification update - 2026-09-21

The current tree has advanced beyond the older handoff text later in this file. Fresh verification from this exact dirty worktree now supersedes older statements that merge/unmerge, native optimizer resume, native-created adapter interoperability, and dense-family qualification were still unverified.

Fresh focused regressions:

cargo test hf_merge_unmerge --lib -j 1 -- --nocapture

Result: 1 passed; 0 failed

cargo test lora_optimizer_resume --lib -j 1 -- --nocapture

Result: 1 passed; 0 failed

The merge/unmerge test verifies ordinary HF PEFT SafeTensors merge plus subtraction-based FP32 unmerge, while rejecting non-invertible replacement/base-mutating cases. The optimizer-resume test verifies adapter AdamW moments, optimizer step, dropout RNG state, saved hyperparameters, exact uninterrupted-vs-resumed continuation, and fail-closed mismatch rejection.

Fresh strict GPT-2 external oracle:

python validation\\verify_peft_lora_strict.py

Result: PASS at the unchanged <= 2e-7 absolute gate.

base_max_abs = 4.8249282846857255e-08
peft_max_abs = 3.3757095341258037e-08
lora_delta_max_abs = 4.8248100292802576e-08
gradient_max_abs = 9.499490261077881e-08
step1_adapter_max_abs = 1.862645149230957e-09
two_step_adamw_max_abs = 1.792795956134796e-08
frozen_base_max_abs = 0.0
trainable_key_set_match = true
resume_uninterrupted_max_abs = 0.0
resume_hf_max_abs = 1.792795956134796e-08

GPT-2 lifecycle/interoperability values:

native-created_reload_hf_max_abs = 5.279102324728946e-08
native-created_save_reload_max_abs = 0.0
native-trained_reload_hf_max_abs = 3.384792326976971e-08
native-trained_save_reload_max_abs = 0.0
hf-created_merged_unmerged_max_abs = 2.999999999808711e-08
hf-created_merged_hf_max_abs = 4.19536209106397e-08
hf-created_unmerge_base_max_abs = 3.00000000119649e-08
native-created_merged_unmerged_max_abs = 2.000000001722846e-08
native-created_merged_hf_max_abs = 3.78898620534418e-08
native-created_unmerge_base_max_abs = 3.00000000119649e-08
native-trained_merged_unmerged_max_abs = 2.999999999808711e-08
native-trained_merged_hf_max_abs = 3.071182252956106e-08
native-trained_unmerge_base_max_abs = 2.000000001722846e-08

Fresh dense-family Llama all-linear oracle:

$env:HIERARCHOS_PEFT_ORACLE_FAMILY='llama'
python validation\\verify_peft_lora_strict.py

Result: PASS at the unchanged <= 2e-7 absolute gate.

base_max_abs = 3.022907257843599e-08
peft_max_abs = 3.310787199950482e-08
lora_delta_max_abs = 5.100752449582746e-08
gradient_max_abs = 3.91155481338501e-08
step1_adapter_max_abs = 1.862645149230957e-09
two_step_adamw_max_abs = 1.7695128917694092e-08
frozen_base_max_abs = 0.0
trainable_key_set_match = true
resume_uninterrupted_max_abs = 0.0
resume_hf_max_abs = 1.7695128917694092e-08

Llama lifecycle/interoperability values:

native-created_reload_hf_max_abs = 3.434160232573813e-08
native-created_save_reload_max_abs = 0.0
native-created_merged_unmerged_max_abs = 3.00000000119649e-08
native-created_merged_hf_max_abs = 3.18003463839478e-08
native-created_unmerge_base_max_abs = 2.9999999984209325e-08
native-trained_reload_hf_max_abs = 4.384815216473825e-08
native-trained_save_reload_max_abs = 0.0
native-trained_merged_unmerged_max_abs = 3.00000000119649e-08
native-trained_merged_hf_max_abs = 3.093908310103366e-08
native-trained_unmerge_base_max_abs = 3.00000000119649e-08

This Llama rerun confirms the current all-linear selector fix in hierarchos-vulkan/src/lora.rs. An older saved Llama log failed while merging a native-created all-linear adapter; the current tree now recognizes PEFT all-linear targeting and the complete strict lifecycle passes.

Updated status: GPT-2 now strictly qualifies HF-created -> native load, native-created -> HF load, save/reload, optimizer-state resume, resumed training, merge, merged-vs-unmerged parity, HF merged parity, and reversible FP32 ordinary-LoRA unmerge. Llama now strictly qualifies the same core lifecycle for a dense all-linear family. Remaining work includes current-tree MoE and hybrid/recurrent strict qualification, external named/multiple-adapter lifecycle parity, adapter-local multi-adapter replacement/token state, graph-wide categorized memory accounting plus a representative larger-model comparison, and full-parameter-training regressions after the optimizer-state refactor.

Goal

Implement real Hugging Face PEFT-compatible native fine-tuning in Hierarchos-Native without regressing full-parameter training. Production execution must remain pure Rust/Vulkan; Python, Transformers, and Hugging Face PEFT are validation/oracle-only.

Strict qualification gate:

deterministic inference: max_abs(native_logits - PEFT_logits) <= 2e-7

LoRA gradients: absolute drift <= 2e-7

strict two-step AdamW parameter drift: <= 2e-7

frozen base parameters must remain unchanged by adapter-only optimization

do not loosen the 2e-7 tolerance

Do not treat compilation, component tests, or successful adapter serialization alone as PEFT parity.

Handoff snapshot for the next agent

This document is the continuation handoff for the in-progress native PEFT work. Do not restart PEFT/LoRA design from scratch.

The current dirty tree already contains the shared/native LoRA implementation, named and multiple ordinary LoRA adapters, PEFT-style configuration/target resolution, Vulkan forward/backward/AdamW, adapter load/export, merge logic, optional optimizer-state storage for frozen parameters, and extensive PEFT regression coverage.

Freshest verified state from the exact inherited tree on 2026-09-21:

cargo check --lib -j 1

Result:

passed

Fresh post-refactor lifecycle rerun:

cargo test named_lora_adapters_switch_disable_and_restore_without_overwriting_state --lib -j 1 -- --nocapture

Result:

1 passed; 0 failed

Fresh broad LoRA regression rerun:

cargo test lora_ --lib -j 1 -- --nocapture

Result:

30 passed; 0 failed

The broad run includes the optimizer/memory test:

lora_releases_frozen_base_training_state_and_reduces_live_buffer_bytes

Observed tiny-GPT2 Vulkan memory numbers:

live_before = 210104 bytes

live_after = 174232 bytes

saved = 35872 bytes

live-buffer reduction = approximately 17.07%

base_projection_training_state_before = 3264 bytes

adapter_training_state = 768 bytes

This is now real measured evidence that PEFT attachment can reduce live Vulkan buffer allocation on the exercised tiny fixture. It is not yet a complete graph-level breakdown and is not yet a representative larger-model memory qualification.

Fresh strict external Hugging Face PEFT 0.18.0 oracle:

python validation\verify_peft_lora_strict.py

Result:

PASS

Measured strict-oracle values:

base_max_abs = 4.8249282846857255e-08

peft_max_abs = 3.3757095341258037e-08

lora_delta_max_abs = 4.8248100292802576e-08

gradient_max_abs = 9.499490261077881e-08

step1_adapter_max_abs = 1.862645149230957e-09

two_step_adamw_max_abs = 1.792795956134796e-08

frozen_base_max_abs = 0.0

trainable_key_set_match = true

All exercised deterministic inference, gradient, and two-step AdamW drifts are below the required absolute 2e-7 gate without loosening tolerance.

The strict fixture is HF-created-adapter -> native-load coverage: PEFT 0.18.0 creates and saves the GPT-2 LoRA adapter, the native runtime loads it for inference/training, and native results are compared back to the HF oracle.

Do not repeat the already completed baseline work above unless later edits require a regression rerun.

The remaining core-LoRA completion work is now concentrated in:

native-created adapter -> Hugging Face PEFT load/interoperability validation

strict save -> load logits parity

save -> reload -> continued-training parity

adapter optimizer-state checkpoint/resume, not merely adapter tensor reload

merged vs unmerged logits under the <= 2e-7 gate

a mathematically reversible unmerge path for variants where subtraction is valid

standalone merged-export parity

external lifecycle parity for named/multiple adapters, enable/disable, and trainability switching

graph-wide frozen-state accounting and a representative larger-fixture memory measurement

representative dense, MoE, and hybrid/recurrent PEFT qualification

full-parameter training regression after the PEFT optimizer-state changes

Last inspected merge state before the interrupted session:

hierarchos-vulkan/src/lora.rs already exposes merge_hierarchos_lora_safetensors(...)

hierarchos-native-cli already exposes merge-lora

no native reversible unmerge surface was found during the final inspection

No unmerge implementation was completed before the session was stopped. The next agent should resume there after checking the current diff.

Worktree safety

The worktree is heavily dirty from prior architecture/parity work. Existing modified and generated files must be preserved. PEFT work should use new/common-path files and surgical edits only; do not clean/reset unrelated state.

Verified dirty state includes extensive prior Falcon H1, Gemma 3, Qwen 2.5, Qwen 3.5/Qwen 4, SmolLM3, Kimi K3, shader, validation, GUI, CLI, and generated target/fixture work.

The 2026-09-21 PEFT lifecycle work made surgical edits primarily in:

hierarchos-vulkan/src/transformer.rs

The later optimizer-state refactor also touched:

hierarchos-vulkan/src/transformer/falcon_h1.rs

Do not discard or rewrite unrelated existing changes.

Existing native LoRA / PEFT implementation state

This checkout is not starting PEFT from zero.

hierarchos-vulkan/src/lora.rs

Already contains a substantial native PEFT-LoRA checkpoint/merge implementation. Verified features present in the source include:

PEFT adapter filenames adapter_config.json and adapter_model.safetensors

native Rust SafeTensors parsing

LoRA A/B tensor discovery

LoRA B-bias handling

modules_to_save replacement handling

trainable-token replacement handling

rank_pattern

alpha_pattern

target_modules

exclude_modules

target_parameters

PEFT bias mode validation

detection/rejection paths for unsupported or non-mergeable PEFT variants

base/adapter SHA-256 binding through the Hierarchos adapter manifest

target resolution against canonical base tensor names

rank/shape validation

native merge into a standalone canonical checkpoint

explicit failure if no mergeable LoRA tensors are present

The public merge path is:

merge_hierarchos_lora_safetensors(...)

The existing file comments explicitly describe it as a package/checkpoint operation rather than the training primitive. Reuse it for adapter persistence and merge semantics instead of duplicating PEFT file handling.

hierarchos-vulkan/src/transformer.rs

The current working tree contains substantial native PEFT training/runtime functionality.

Verified existing or newly completed functionality includes:

PEFT/LoRA config parsing and validation

native LoRA wrapped-linear execution

LoRA A/B trainable Vulkan parameters

native LoRA forward/backward

LoRA parameter AdamW stepping

PEFT trainable-token support

modules_to_save support

adapter loading

adapter export

rank_pattern

alpha_pattern

regex targeting

suffix targeting

all-linear

layers_to_transform

layers_pattern

exclude_modules

target_parameters

layer_replication

lora_bias

DoRA-related state

multiple PEFT initialization modes

PEFT-aware frozen-base training branches

explicit fail-closed paths for unsupported architectures/features

separation of full-parameter trainability and PEFT trainability

The earlier single-adapter lifecycle limitation has now been substantially changed.

Named and multiple adapter lifecycle

Implemented on 2026-09-21 in the shared Transformer/linear PEFT path.

Native linear adapter bank

VulkanLinear no longer conceptually relies on only one active LoRA state.

A graph-shared adapter-selection mechanism and per-linear adapter bank were introduced so multiple LoRA parameter sets can coexist without materializing ?W or overwriting one another.

The adapter bank is keyed internally by graph-scoped numeric adapter IDs.

The active adapter is selected through shared runtime state, allowing adapter switching without copying LoRA tensors.

The disabled state uses an explicit no-active-adapter runtime ID.

Graph-level adapter registry

The Transformer now tracks native PEFT adapter lifecycle state including:

loaded adapter configs

adapter-name to runtime-ID mappings

graph-wide active adapter selection

active adapter name

adapters-enabled state

monotonically allocated runtime adapter IDs

The old graph-level lora_config path remains as a compatibility/current-active-config view for existing PEFT training/export code.

Do not regress existing callers that use the original default adapter APIs.

Public lifecycle behavior added

The native runtime now has behavior corresponding to:

default adapter loading

named adapter loading

default adapter creation

named adapter creation

list loaded adapter names

inspect active adapter name

set active adapter

disable adapter

re-enable selected adapter

retain multiple independent LoRA states simultaneously

The original single-adapter APIs continue to route through the PEFT default adapter name:

default

Confirmed switching behavior

A deterministic native test now demonstrates:

construct a tiny GPT-2 graph

record base logits

attach adapter_a

record adapter A tensors/logits

attach independently initialized adapter_b

verify both adapters remain resident

verify A/B adapter tensors differ

switch back to adapter_a

verify the original adapter A tensors are restored exactly

verify adapter A logits are restored exactly

disable adapters

verify exact base-model logits are restored

re-enable the selected adapter

verify adapter A logits are restored exactly again

reject switching to a nonexistent adapter

Test:

transformer::tests::named_lora_adapters_switch_disable_and_restore_without_overwriting_state

Result on 2026-09-21:

1 passed; 0 failed

This proves native lifecycle/state isolation for the exercised ordinary LoRA fixture. It does not yet prove full Hugging Face multi-adapter parity.

Hugging Face PEFT oracle

Installed local oracle inspected on 2026-09-21:

peft 0.18.0

Local Transformers oracle remains:

C:\Users\User\transformers

The installed PEFT source was inspected directly for multi-adapter semantics.

Verified PEFT 0.18.0 behavior:

Adapter switching

PEFT BaseTunerLayer.set_adapter(...) changes both:

the active adapter selection

which adapter parameter set has requires_grad=True

Inactive adapter layers are frozen.

Native validation must therefore verify not only inference switching but optimizer/trainability switching.

Adapter disabling

PEFT's enable_adapters(false) disables adapter-layer gradients and bypasses adapter execution.

The native implementation now supports ordinary LoRA disable/re-enable behavior.

Multiple adapter bias restriction

PEFT 0.18.0 does not require every loaded adapter to use bias="none".

Its rule is effectively:

at most one loaded LoRA adapter may use a non-none bias setting

The native compatibility validation was updated to mirror this rule rather than incorrectly rejecting every multi-adapter configuration containing one bias-bearing adapter.

Multi-adapter fail-closed boundaries

Some PEFT features modify state outside ordinary independent LoRA A/B tensors and cannot yet be switched losslessly by the adapter bank.

These combinations must continue to fail explicitly until adapter-local state is implemented.

modules_to_save

Multiple loaded adapters using modules_to_save are not yet supported losslessly.

Reason:

modules_to_save requires adapter-local trainable module clones, while the current native path can replace/promote underlying native module parameters directly.

Do not silently share one replacement module among multiple adapters.

trainable_token_indices

Multiple loaded adapters with independent trainable-token rows are not yet supported losslessly.

Independent adapter-local embedding delta state is still required.

layer_replication

Multiple loaded adapters combined with PEFT layer replication are not yet supported.

Do not mutate graph topology repeatedly or ambiguously when multiple adapters coexist.

OLoRA / PiSSA

OLoRA and PiSSA initialization can mutate/subtract from the underlying base weight during adapter injection.

Multiple independently initialized OLoRA/PiSSA adapters therefore cannot safely share one already-mutated native base under the ordinary adapter-bank design.

These combinations are explicitly rejected until a reversible/original-base representation is implemented.

Disable with adapter-local replacement state

Disabling an adapter that uses modules_to_save or trainable-token replacement state is not currently guaranteed to restore the untouched base representation losslessly.

The native runtime fails closed rather than pretending these wrappers are ordinary LoRA branches.

Architecture independence

The intended PEFT design is routed primarily through shared Transformer linear/module abstractions rather than being implemented solely as per-architecture checkpoint hacks.

The new multi-adapter bank lives at the shared VulkanLinear level, allowing existing architecture-specific PEFT routing to keep using common LoRA attachment/forward/backward machinery.

Existing architecture-specific target naming/layout logic remains necessary where Hugging Face model structures differ.

Verified fail-closed behavior exists for unsupported architecture routes.

Example:

Falcon H1 PEFT is not implemented

Unsupported targets or architectures must continue to fail explicitly rather than silently falling back to a numerically different path.

Existing training infrastructure relevant to PEFT

hierarchos-vulkan/src/full_training_graph.rs already exposes common native optimizer/trainable machinery used by full training, including:

NamedTrainable

RwkvTrainableRef

RwkvPersistentAdamW

optimizer state serialization

pending gradient state serialization

native Vulkan gradient transport

memory working-set planning

Transformer LoRA already uses native Vulkan TrainableParameter / AdamW paths.

Do not introduce a Python optimizer into production execution.

Important optimizer-memory status

The earlier eager-allocation problem has been partially fixed in the current dirty tree.

TrainableParameter now separates parameter values from optional training state.

Implemented behavior includes:

optional gradient storage

optional AdamW exp_avg storage

optional AdamW exp_avg_sq storage

checked grad_buffer() access

ensure_training_state() for fail-closed lazy recreation when a parameter becomes trainable

release_training_state() for dropping gradient, AdamW moments, and shared gradient-accumulation state

training_state_bytes() as a local accounting primitive

optimizer submission that fails if required state is absent instead of dispatching against dummy buffers

PEFT freeze/re-enable wiring in shared VulkanLinear/layer-normalization paths and several direct parameter owners

The current tree compiles after these changes.

The fresh lora_ regression run also proved a real live Vulkan buffer reduction on the tiny GPT-2 fixture:

live_before = 210104 bytes

live_after = 174232 bytes

saved = 35872 bytes

approximately 17.07% fewer live buffer bytes after PEFT attachment/frozen-base state release on that fixture

base_projection_training_state_before = 3264 bytes

adapter_training_state = 768 bytes

This supersedes the old statement that no memory reduction had yet been measured.

Do not overgeneralize the tiny-fixture result. The following memory work remains:

audit every direct TrainableParameter owner outside the shared VulkanLinear/layer-norm paths

prove every frozen base training state is released graph-wide

prove every PEFT exception recreates valid state before backward/step

expose a graph-level breakdown of parameter-value bytes vs gradient bytes vs AdamW first/second moments vs shared/accumulation bytes vs adapter-local bytes

compare full fine-tuning and PEFT on the same representative larger model/fixture

report exact bytes and ratios

verify full-parameter training still allocates and updates its complete expected trainable set

When fixing remaining memory gaps, propagate allocation failures. Do not use let _ = ..., unwrap(), or dummy buffers simply to obtain green compilation.

Base parameter values must remain resident. The memory target is removal of unnecessary frozen-base training/optimizer state, not removal of the base model itself.

Required LoRA semantics

Core LoRA:

y = Wx + scale * B(A(dropout(x)))

Required PEFT semantics:

r

lora_alpha

scaling

lora_dropout

target_modules

regex/suffix targeting

all-linear

layers_to_transform

layers_pattern

bias

modules_to_save

adapter names

enable/disable adapter

set active adapter

multiple loaded adapters

inactive adapter freezing

frozen base parameters

trainable active adapter parameters only

Required checkpoint/runtime behavior:

PEFT-compatible adapter_config.json

PEFT-compatible adapter_model.safetensors

independent adapter load/save

load existing HF PEFT LoRA adapters

resume adapter training

unmerged adapter inference

merge

mathematically reversible unmerge where applicable

merge-and-export standalone model

Verification status

Confirmed on the exact inherited tree on 2026-09-21:

cargo check --lib -j 1 passed

named_lora_adapters_switch_disable_and_restore_without_overwriting_state passed: 1 passed; 0 failed

named_lora_only_active_adapter_receives_adamw_updates remains green after the optional optimizer-state refactor: 1 passed; 0 failed

cargo test lora_ --lib -j 1 -- --nocapture passed: 30 passed; 0 failed

the broad LoRA suite includes the tiny-fixture memory reduction test and reports 35872 fewer live Vulkan buffer bytes, from 210104 to 174232

the strict external Hugging Face PEFT 0.18.0 oracle passed

strict oracle measurements:

base/no-adapter logits max_abs = 4.8249282846857255e-08

unmerged PEFT logits max_abs = 3.3757095341258037e-08

LoRA delta max_abs = 4.8248100292802576e-08

LoRA gradient max_abs = 9.499490261077881e-08

one-step adapter parameter max_abs = 1.862645149230957e-09

two-step AdamW adapter parameter max_abs = 1.792795956134796e-08

frozen base max_abs = 0.0

HF/native trainable LoRA key set match = true

Every measured strict value is within the required <= 2e-7 absolute gate.

The strict fixture also validates HF PEFT-created adapter -> native load for inference and training on the exercised GPT-2 configuration.

Internal lifecycle evidence confirms:

multiple ordinary LoRA parameter sets coexist without overwriting one another

switching adapters restores the selected adapter state

disable restores exact base logits in the tiny deterministic fixture

re-enable restores the selected adapter logits

only the active adapter receives native AdamW updates in the exercised two-adapter fixture

the inactive adapter remains byte-identical while another adapter trains

the frozen base c_proj.weight remains byte-identical during adapter-only optimization

Known PEFT 0.18.0 reference behavior already incorporated:

set_adapter changes active selection and trainability

inactive adapters are frozen

enable_adapters(false) disables adapter execution/gradients

at most one loaded LoRA adapter may use a non-none PEFT bias setting

unsupported multi-adapter replacement/topology-mutating combinations fail closed

Still not qualified / still to complete

Do not mark core LoRA complete yet.

Remaining items:

native-created adapter -> HF PEFT load and logits interoperability

strict adapter save -> reload logits reproduction

save -> reload -> continued-training parity

adapter optimizer-state serialization/resume; reloading adapter A/B tensors alone is not optimizer-resume parity

merged vs unmerged logits <= 2e-7 on a strict external fixture

reversible unmerge for mathematically reversible ordinary LoRA/rsLoRA-style additive deltas; fail closed for non-reversible replacement/base-mutating cases

standalone merge-and-export parity

external named/multiple-adapter output parity

external enable/disable parity

external active/inactive adapter trainability-switch parity

strict external coverage for all-linear, regex/suffix targeting, layers_to_transform, layers_pattern, bias modes, modules_to_save, trainable-token rows, rank 1 and higher ranks, and controlled nonzero dropout

adapter-local modules_to_save state for multiple adapters

adapter-local trainable-token switching for multiple adapters

complete graph-wide frozen-training-state elimination/accounting

representative larger-model PEFT-vs-full-training memory measurement

representative dense architecture qualification

representative MoE architecture qualification

representative hybrid/recurrent architecture qualification

full-parameter-training regression after the PEFT optimizer-memory refactor

The installed validation oracle remains:

local Transformers checkout: C:\Users\User\transformers

Hugging Face peft==0.18.0

Python/Transformers/PEFT remain validation-only and must not enter production execution.

PEFT oracle

Required oracle/reference for subsequent validation:

local Transformers checkout: C:\Users\User\transformers

installed Hugging Face peft==0.18.0

Python/Transformers/PEFT remain validation-only.

They must not enter the production execution path.

Next task

Resume from the current in-tree PEFT implementation. Do not rebuild LoRA, the adapter bank, targeting, or the strict GPT-2 oracle from scratch.

Priority order for the next agent:

Inspect git status/diff first. Preserve the heavily dirty worktree and all unrelated architecture/parity/generated state.

Read PROGRESS_PEFT_AUDIT.md and the current diffs in:
hierarchos-vulkan/src/transformer.rs
hierarchos-vulkan/src/transformer/falcon_h1.rs
hierarchos-vulkan/src/lora.rs
hierarchos-vulkan/validation/verify_peft_lora_strict.py
hierarchos-native-cli/src/lib.rs

Treat the following as already-green baselines:
cargo check --lib -j 1
named_lora_adapters_switch_disable_and_restore_without_overwriting_state: 1/1
named_lora_only_active_adapter_receives_adamw_updates: 1/1
cargo test lora_ --lib -j 1 -- --nocapture: 30/30
python validation\verify_peft_lora_strict.py: PASS at <= 2e-7

Finish merge/unmerge qualification next:
use the existing merge_hierarchos_lora_safetensors(...) path
validate strict merged-vs-unmerged logits <= 2e-7
validate the existing CLI merge-lora standalone export
add reversible unmerge for ordinary additive LoRA where mathematically valid
make unmerge fail explicitly for replacement/base-mutating/non-invertible variants rather than silently approximating

Finish adapter persistence/resume:
native save -> native reload -> logits <= 2e-7
native-created adapter -> HF PEFT load -> logits <= 2e-7
HF-created adapter -> native load is already covered by the passing strict oracle; extend rather than duplicate it
save -> reload -> continue training and compare expected parameter updates
add adapter optimizer-state serialization/resume so continuation does not reset AdamW moments/step

Extend the strict oracle for lifecycle semantics:
named adapters
multiple loaded adapters
set_adapter
enable/disable
active/inactive requires_grad/trainability equivalence
frozen-base invariance throughout

Complete graph-wide PEFT memory accounting:
audit all direct TrainableParameter owners
expose exact training-state bytes by category
retain the current tiny-fixture measurement as a baseline
add a representative larger fixture and compare full fine-tuning vs PEFT

Run architecture-independent qualification:
at least one dense family
at least one MoE family
at least one hybrid/recurrent family that actually has a supported common linear PEFT route
cover q/k/v/o projections, MLP gate/up/down, MoE expert projections, applicable recurrent/hybrid linear projections, and all-linear where representable
unsupported targets/architectures must continue to fail closed

Re-run full-parameter training regressions after the optimizer-state changes. The optional-state refactor must not leave ordinary full training missing gradients or AdamW state.

Update this audit only with commands and measurements actually run.

Do not spend another session re-proving the already-green GPT-2 strict inference/gradient/two-step AdamW baseline unless subsequent changes touch those paths. Use it as a regression gate after substantive edits.

Immediate handoff warnings

The next agent should not restart LoRA/PEFT design work from scratch.

Current known-good baseline:

native Rust/Vulkan LoRA forward/backward/AdamW is operational

PEFT 0.18.0-created GPT-2 adapter loads and trains natively

strict deterministic base logits, PEFT logits, LoRA delta, gradients, and two-step AdamW all pass the <= 2e-7 gate

frozen base drift is exactly 0.0 in the strict external fixture

HF/native trainable LoRA key sets match in that strict fixture

named/multiple ordinary adapter lifecycle is implemented internally

active/inactive adapter optimizer isolation is green internally

fresh broad LoRA regression suite is 30 passed; 0 failed

tiny-fixture live Vulkan buffer usage drops from 210104 to 174232 bytes after PEFT attachment/frozen-state release, saving 35872 bytes (~17.07%)

Still not complete:

native-created adapter -> HF load has not been strictly qualified

save/reload/continued-training parity is not complete

adapter optimizer-state resume is not complete

strict merged/unmerged parity has not been run

native reversible unmerge has not been implemented

graph-wide training-state accounting and a representative larger memory comparison remain incomplete

external multi-adapter lifecycle/trainability parity remains incomplete

representative dense/MoE/hybrid PEFT qualification remains incomplete

full-parameter training regression after the optimizer-state refactor remains required

The final interrupted session ended while inspecting hierarchos-vulkan/src/lora.rs and the CLI merge-lora path. No unmerge code was completed after that inspection.

Preserve existing fail-closed boundaries for unsupported multi-adapter modules_to_save, trainable-token replacement, layer replication, and base-mutating OLoRA/PiSSA until independent/reversible adapter-local state exists.

Do not loosen the 2e-7 gate.

Do not introduce Python/PyTorch/Transformers/PEFT into production execution.

Do not clean/reset the dirty worktree.

Completion rule

Do not mark PEFT/LoRA complete until the full native flow is demonstrated under the strict parity gates:

load base -> attach/load adapter -> train natively -> save adapter -> reload -> continue training -> generate -> optionally merge/unmerge/export

Completion requires evidence for:

native Rust/Vulkan execution

HF PEFT-compatible adapter files

<= 2e-7 deterministic inference drift

<= 2e-7 LoRA gradient drift

<= 2e-7 two-step AdamW parameter drift

frozen-base invariance

actual optimizer-state/memory reduction

adapter lifecycle parity

save/load/resume parity

full-training regression safety

Compilation or green component tests alone are not sufficient.
