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

## CURRENT CHECKPOINT 2026-10-05: full-matrix rerun against the current oracle

Environment and provenance:

- Oracle checkout: `C:\Users\User\transformers`, `5.19.0.dev0`, commit
  `469230357aab0f2b303b0d638c1f8d06edb14184`, clean worktree.
- Oracle stack: `peft==0.18.0`, torch `2.14.1+cpu`, CPython 3.12.15 in
  `.venv-vulkan`.
- `peft==0.21.2` was tried first and is not usable for this matrix: newer PEFT
  emits expert `target_parameters` adapters, and the native LoRA ABI rejects
  them (`LoRA target_parameters ... Mixtral/Qwen3Next does not yet expose a
  lossless native parameter adapter ABI`). The run was repeated with the
  documented oracle pin.
- Matrix command: `verify_peft_green_matrix.py --jobs 2`; report
  `.peft-oracle-fixtures/green-matrix-report.json`; provenance
  `.peft-oracle-fixtures/green-matrix-report-provenance.json` with
  `inputs_unchanged=true` (source, shader, binary, and oracle hashes unchanged
  for the whole run).

Result: 32/32 fixtures pass ordinary LoRA and named-adapter switching; 23/32
pass the saved-module stage. Nine saved rows are red and are now listed
explicitly in COMPATIBILITY.md ("Saved-module stage status"), with the exact
adapter A/B values per module: llama, deepseek_v4, phi4_multimodal_text,
smollm3, qwen3_5_full, minimax_m3, minimax_m2, gemma3, gemma4.

Classification and evidence (gate unchanged at absolute `2e-7`; no PEFT fixture
was re-conditioned):

- llama `model.embed_tokens` A/B `2.384185791015625e-07`. Triple comparison on
  the same tensor: native vs HF fp32 max `2.384185791015625e-07`, HF
  fp32-vs-fp64 max `2.682209014892578e-07`, native vs HF fp64 max
  `2.086162567138672e-07`. The worst native-vs-fp32 element (flat index 52,
  value `-0.6227507591247559`) sits between the fp32 and fp64 oracle values;
  at the tensor's largest element (index 113, `0.7870487570762634`) native
  equals fp64 exactly and differs from the fp32 oracle by `1.7881393432617188e-07`.
  The oracle's own fp32 rounding on this tensor therefore exceeds the gate.
- The same llama gradients reproduce bit-for-bit with
  `HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS=1`
  (`diagnose_peft_saved_gradient.py llama`), and the uncommitted Intel vendor
  replacements cover only `linear_forward`/`linear_bias_forward` with raw-bit
  portable-equivalence tests, so that work is excluded as a cause.
- The previously recorded saved numbers were measured against the `5.16.0.dev0`
  oracle (llama saved embedding A `1.7881393432617188e-07`, Gemma3 saved
  embedding A `5.066394805908203e-07`, Gemma4 saved input-norm A
  `2.384185791015625e-07`). The current checkout moves the same surfaces above
  the gate. Gemma3/Gemma4 remain the large amplifiers already documented above
  (Q/K/V RMSNorm backward); their scoped fixes were aligned to the older
  oracle's CPU reduction topology and need re-localization against the current
  torch build before those saved rows can return to green.
- The llama embedding scatter itself remains exact: the probe's
  `hf_source_order_reconstruction_max_abs` is `0.0`, so the divergence is in
  the adjoint presented to the embedding, matching the earlier
  execution-topology classification.

Decision for the merge window: the `2e-7` gate stays unchanged, no PEFT fixture
is re-conditioned, and the documented claim is scoped to the measured state
(32/32 lora/switch, 23/32 saved). The nine saved rows above are explicitly not
claimed as saved-module-qualified.

## CURRENT CHECKPOINT 2026-10-05 (later): saved-module drift was native-side, not oracle-side

The previous checkpoint classified llama's saved-module drift as oracle-side FP32
execution topology. That classification was wrong. Four native kernel bugs were
localized with on-device candidate probes and fixed; the same fixture then
measures `5.960464478e-08` for adapter B (gate unchanged at absolute `2e-7`, no
PEFT fixture re-conditioned, no tolerance relaxed). The oracle did not change.

### 1. Plain-RoPE inverse frequencies

`uses_precomputed_frequencies` is now true for every `scaling_type == None`
layer, not only Gemma3/Gemma4, so those layers consume host-materialized FP32
inverse frequencies (`layer_kernel_type() == 7`) instead of the in-shader
`pow(theta, -exponent)` path, whose device trig is 1-3 ulp off the oracle for
frequencies 2 and 3. `k_rotary_output` becomes bit-exact.

The branch must stay gated on `rotary_dim > 0`: without that guard a non-RoPE
layer (GPT-2, BERT) built an empty factor table and every graph construction
failed with `Vulkan storage buffer size must be positive` (69 library tests plus
the `hierarchos-vulkan-transformer-logits` binary).

### 2. Attention softmax reciprocal

The device's raw `1.0 / exp_sum` is one ulp high. Both attention shaders now use
the correctly rounded `fp32_div` for the reciprocal instead of the relaxed
divider. Forward attention is bit-exact against
`layers.*.self_attn.o_proj.input`.

### 3. SiLU (`shaders/silu_forward.comp`) was not the ATen formula

ATen/HF compute `silu(x) = x / (1 + exp(-x))`. The kernel computed
`x * (1 / (1 + exp(-x)))` with the driver's relaxed `exp()`. Measured on the 128
real layer-0 gate values, against `layers.0.mlp.act_fn.value`:

| module | differing elements |
| --- | ---: |
| shipped (`exp` + reciprocal form) | 44/128 |
| `exp` + correctly rounded `/` | 32/128 |
| SLEEF `exp` + bare `/` | 27/128 |
| SLEEF `exp` + `fp32_div` | **0/128 (bit-exact)** |

With this fix the whole layer-0 MLP chain is bit-exact (gate/up serial-FMA GEMM,
SiLU, elementwise product, down_proj serial-FMA), and every forward boundary of
both layers matches the oracle exactly (`layers.1.value`, `model.norm.value`,
`lm_head.input`, `layers.0.mlp.down_proj.value` all `d = 0.0`).

### 4. Cross-entropy input adjacency: the lane fold

The PEFT path dispatches the materialized-log-softmax cross entropy
(`falcon_h1_cross_entropy.spv`, built from `transformer_cross_entropy.comp` with
`HIERARCHOS_LOG_SOFTMAX_GRAD=1`). It accumulated the AVX512-sized 16-float lanes
correctly but folded them as a lane-order sequential sum. PyTorch reduces those
lane totals as a balanced binary tree (`vec_reduce_all`). Sequential fold: 31 of
128 reference adjoint elements one ulp off; tree fold: 0/128 bit-exact. Verified
on the executing device against `lm_head.grad_output`, plus six randomized
cases. `transformer_cross_entropy.spv` (macro 0, non-PEFT path) is behaviorally
unchanged.

### Measured result on the llama fixture

| quantity | before | after |
| --- | ---: | ---: |
| adapter A embedding gradient | `2.384185791e-7` | `1.192092896e-7` |
| adapter B embedding gradient | `2.384185791e-7` | `5.960464478e-8` |
| frozen base | `0.0` | `0.0` |
| `verify_peft_saved_modules.py llama` | fail | `pass: true` |

The residual `5.960464478e-8` (2^-24) is the last non-bit-exact backward
boundary in the layer-1 block (layer input gradient); it is 3.3x inside the gate
and its first non-exact step is inside the layer-1 backward, not at the CE head,
the final norm, the embedding scatter, or either layer's forward.

### Verification discipline

Every candidate module was executed on the reference device (Intel HD Graphics
520) by a temporary probe test that reads SPIR-V files from `target/` and
dispatches them through the production dispatch contract, then compared
bit-for-bit against the oracle capture; only the oracle-exact module was
promoted into `shaders/`. All temporary probe tests were removed from
`src/transformer.rs` before the final build.

## CURRENT CHECKPOINT 2026-10-05 (regression question): the current tree is
better than the last commit, not worse

Claim under test: the 2026-09-30 ROG Ally run was 32/32 all-stage, so this work
item may have introduced a saved-module regression. Every measurement below was
taken on the Intel HD Graphics 520 reference device.

1. This work item changed no saved-stage number. `green-matrix-report.json`
   (16:37) and its pre-work-item predecessor
   `.prior-1791239954259869000.json` (14:21) agree on every family verdict and
   every module value.
2. The last commit is worse, not better. A scratch worktree at `HEAD`
   (`d006459`), built separately (13m28s, own `target/`) and run against the
   byte-identical Gemma3 fixture, measures `model.embed_tokens` gradient A
   `1.1920928955078125e-06` / B `2.9802322387695312e-07` (red). The current tree
   measures A `4.76837158203125e-07` / B `2.384185791015625e-07` (still red, but
   ~2.5x closer). Gemma3's other saved rows (`mlp.down_proj`,
   `input_layernorm`, `lm_head`) pass on both. The scratch worktree was removed
   after the probe.
3. The oracle checkout is excluded by measurement, not by argument.
   `verify_peft_lora_strict.py` now accepts `HIERARCHOS_TRANSFORMERS_ROOT`
   (default unchanged, still the local checkout; the provenance guard still
   refuses a site-packages import). With the 2026-09-30 tag `v5.16.0`
   extracted outside the live checkout, the Gemma3 fixture regenerates
   **byte-identically** (base
   `65b0ef7de0f486103bf9cad191c874d98321428b628872ddbb05aa68f8bdd7e2`, adapter
   `2ec93d78a2aec5ffd419bba0856a281eb1c5a5eb097b74dbc7cf08c99d8cfa67`,
   `training.json` `ce42a7ca5fbf66921e89c650a2a94a7f4b56704329db9588c84c3dadcd052e01`)
   and the saved-embedding gradient is bit-identical to the `5.19.0.dev0`
   result. The "FP32 execution-topology drift between oracle checkouts"
   sentence in COMPATIBILITY.md was therefore stale and now records the measured
   native-side classification.
4. The Intel vendor layer is excluded:
   `HIERARCHOS_VULKAN_FORCE_VENDOR=amd` reproduces the same seven red rows with
   bit-identical numbers.

Unverified and left open: the Ally-era numbers quoted in COMPATIBILITY.md
(Gemma3 saved embedding A `1.1920928955078125e-7` / B `1.7881393432617188e-7`,
Gemma4 A `7.450580596923828e-9` / B `2.9802322387695312e-8`) exist in no artifact
on this disk, and `HEAD` reproduces `1.1920928955078125e-06` here, so they are
AMD-device measurements that cannot be reproduced without the Ally. Gemma3's
RMSNorm path is mode 3 (`fp32_sqrt_recip`, eight interleaved FP32 lanes folded in
order), so it is device-independent and is not a cross-vendor suspect.

Next: localize the Gemma3/Gemma4 first divergent backward boundary with
`validation/diagnose_peft_backward_boundary.py` (proven for llama) before
touching any shader.

### Gemma3 saved-embedding localization (same day, operand-matched)

`diagnose_peft_saved_gradient.py gemma3 --module model.embed_tokens --adapter a`
reproduces `4.76837158203125e-07` on `model.embed_tokens.weight` with
`hf_source_order_reconstruction_max_abs = 0.0`, so the scatter is exact and the
divergence is the adjoint handed to the embedding.

Boundary chain (`diagnose_peft_backward_boundary.py gemma3 --trace ...`), HF
boundary vs native key, max-abs:

| link | delta |
| --- | ---: |
| `lm_head.grad_output` vs `grad_logits` (CE adjoint) | `3.725e-09` |
| `model.norm.grad_input` vs `grad_final_norm_input` | `1.863e-09` |
| `layers.1.grad_output` vs `grad_final_norm_input` | `1.863e-09` |
| `layers.0.grad_output` vs `layers.1.grad_input` | `6.519e-09` |
| `layers.0.grad_input` vs `layers.0.grad_input` (= embedding adjoint) | `1.192e-07` |
| `lm_head.input` vs `final_norm_output` (forward) | `4.768e-07` |
| `layers.0.value` vs `layers.0.output` (forward) | `2.384e-07` |
| `o_proj.base_layer.input` vs `attention_output` (forward) | `7.451e-09` |

The chain is near-exact everywhere except layer 0, and the first material
divergence is in the FORWARD: layer-0 attention output, one ulp. Everything
downstream (layer output, final norm, adjoint, embedding gradient) inherits and
amplifies it.

Operand-matched evidence collected this round (scripts in `.venv-vulkan/scratch`
plus `validation/diagnose_gemma3_rope_operands.py`):

1. Q/K/V projections and norms are bit-exact, and Gemma3 layer-0 RoPE is
   bit-exact against HF's own captured post-rope operands (`q delta 0.0`,
   `k delta 0.0`), so the precomputed-frequency change is correct here. Layer 1's
   rope differs by ~1.5 ulp only because layer 1's inputs already differ.
2. Replaying HF eager attention with torch's own operators (matmul + softmax,
   not element-wise loops) reproduces the oracle capture exactly
   (`delta = 0.0`), and feeding the NATIVE operands through those same operators
   also lands on the oracle exactly. The native's inputs are therefore perfect;
   only its accumulation order differs: the native attention output is
   `7.451e-09` from the oracle given identical operands.
3. HF's softmax semantics are `exp` then multiply by the reciprocal: that form
   matches `torch.nn.functional.softmax` bit-exactly, while dividing each
   exponential by the sum differs on exactly one element by `7.451e-09`.
4. For the tiny shapes involved, none of sequential products, adjacent-pair tree
   products, or lane-FMA-plus-tree reproduces `torch.matmul` bit-exactly
   (QK best `2.384e-07`, P@V best `7.451e-09`), so the shader must be matched
   against ATen's blocked CPU micro-kernel order rather than a naive loop.
5. `HEAD` already routed `Gemma3 | Gemma4` to attention mode 1 in both the
   forward and backward dispatch, identical to the working tree, so today's work
   changed no Gemma3 attention mode; the only new arm is Llama's mode 9.

Next: pin ATen's exact reduction order for these shapes (the attention
accumulation), then re-run `verify_peft_saved_modules.py gemma3` and the
porcelain matrix rows for gemma4/minimax_m2/minimax_m3/mistral4/qwen3_5_full/
smollm3.

## CURRENT CHECKPOINT 2026-10-05 (Intel Gen9 FP32 fidelity): four root causes found, one family flipped green

The AMD-green / Intel-red split is a **FP32 fidelity** split, not a math split.
Every kernel order in the tree is already the oracle's order; what differs is how
a Gen9 driver evaluates a handful of expressions that a double-capable,
IEEE-FMA vendor evaluates exactly. Four independent mechanisms were isolated and
fixed this round. All are in shared fp32 primitives, so they apply to every
architecture.

### 1. Bare double literals silently promoted fp32 expressions to FP64

`sleef_expf_u10` (and the log, tanh and pow siblings) were written with
unsuffixed literals such as `-0.00139304355252534151077271`. GLSL then promotes
the whole expression to `double`, so the "SLEEF fp32 polynomial" actually
executed in FP64 and only rounded to FP32 at the very end.

Measured on this host (HD Graphics 520, driver 1656899): input
`x = -1.001569151878357` gave device/torch `0.36730265617370605`
(`0x3ebc0f18`, SLEEF) versus the old shader's `0.36730262637138367`
(`0x3ebc0f17`, the correctly rounded exp). One ulp, on roughly 1% of arguments,
and never visible in a reduction denominator -- only in a value that is later
reconstructed by a backward pass.

Fix: every literal carries an explicit `f` suffix, and every multiply-add that
SLEEF performs fused goes through the new `shaders/fp32_fma_exact.glsl`
(Dekker two-product with split `4097.0` plus Knuth two-sum, `precise`
throughout, no `shaderFloat64`). Validated offline against true `fmaf` over
900 000 samples: 0 mismatches. The new SPIR-V declares only capability
`Shader` -- no `Float64`.

### 2. `precise` + `fma()` is a separate multiply and add on Gen9

`lora_a_forward.comp` accumulates sixteen `precise` lanes with `fma(...)`. The
device result equals a *materialized product* model, not the fused model: for
Gemma3's down-projection LoRA-A (`32 -> 2`) the native output is reproduced
exactly (`0/64`) by "products rounded, eight or sixteen independent lanes,
halving tree", and never by the fused variant. `linear_forward.comp`, whose
accumulator is deliberately *not* `precise`, stays fused -- verified by editing
it to `fp32_fma_exact` and observing a bit-identical trace, then reverting the
edit so the shipped kernel keeps its cheaper `fma` chain.

### 3. The wrong lane topology for the wide LoRA-A reduction

`diagnose_small_gemm.py`-style modelling over the real captured operands
(`.venv-vulkan/scratch`), comparing HF's `lora_A.default.value` per shape:

| LoRA-A shape | topologies that reproduce HF |
| --- | --- |
| `(2, 16)` (q/k/v/o/gate/up) | `l8_nf`, `l16_fused`, `l16_nf` -- all agree at K=16 |
| `(2, 32)` (down projection) | **only `l8_nf`** |

So PyTorch's CPU small-output GEMM walks the reduction with AVX2-width (eight
float) lanes and folds them in halves; the sixteen-lane kernel only coincides
with it while the reduction is exactly sixteen wide. `src/transformer.rs` had
routed the MLP down projection to `lora_a_forward_lane8_muladd` for `Gemma4`
only. It now keys off the reduction size (`c_mlp_proj.input_dim >= 32`), which
covers Gemma3 and every other wide-rank down projection.

### 4. The GELU-tanh constant and the CE softmax path

`transformer_gelu_tanh_{forward,backward}.comp` computed `0.044715 * x3` and
`3.0 * 0.044715` with double literals, and `sleef_tanhf_u10` had both the FP64
promotion and GLSL `fma` problems plus two bare `/`. All are now suffixed,
exact-fused and correctly divided (`fp32_div`).

`falcon_h1_cross_entropy.spv` is the `-DHIERARCHOS_LOG_SOFTMAX_GRAD=1` build of
`transformer_cross_entropy.comp` (byte-identical to compiling that source with
the define; the plain build differs), and it is what the LoRA / PEFT loss path
dispatches -- see `transformer.rs` "if self.lora_config.is_some() || ...
FalconH1". It was regenerated with the fixed primitives.

### Measured effect (saved-module probe, `model.embed_tokens`, adapter a, TOL 2e-7)

| family | before | after |
| --- | --- | --- |
| gemma3 | 4.768e-07 | **2.086e-07** |
| minimax_m3 | 2.086e-07 | **1.788e-07 (PASS)** |
| smollm3 | 2.980e-07 | 2.980e-07 |
| qwen3_5_full | 2.384e-07 | 2.384e-07 |
| mistral4 | 2.384e-07 | 2.384e-07 |
| minimax_m2 | 3.576e-07 | 3.576e-07 |
| gemma4 | 4.768e-07 | 4.768e-07 |

For Gemma3 the **forward pass is now bit-exact end to end** (every remaining
non-zero row in `diagnose_peft_backward_boundary.py` is a numel-collision
mis-pairing: deltas 8e-05 .. 2e0 against a correctly matched neighbour at 0.0).
The residual 2.086e-07 is entirely in the backward chain: `layers.0.grad_input`
sits `5.215e-08` (about half an ulp) from HF, and the `modules_to_save`
reduction scales it by `4.0`, giving `2.086e-07` against a `2e-7` gate. Bit
exactness upstream is therefore required, not merely accuracy.
`layers.1.*` and `layers.0.*` gradients all show ~1 ulp relative error, first
appearing in the layer-1 MLP backward (`grad_post_mlp_norm` is exact,
`grad_mlp_norm_output` is `4.657e-10` off), so the next target is the backward
GEMM / LoRA-adjoint topology for those shapes.

### Instrumentation caveat found this round

`diagnose_peft_backward_boundary.py` registers a `register_hook` on the tensor
object handed to each module. PEFT hands the *same* dropout output tensor to
both the base layer and the LoRA branch, so an HF `*.grad_input` entry can be a
partial gradient from whichever branch backpropagates first. Concretely,
`layers.1.mlp.down_proj.base_layer.grad_input` disagrees with
`torch.mm(grad_output, weight)` on 96 of 128 elements at up to `6.1e-02`. Treat
those captured `grad_input`s as capture artifacts and recompute targets with
`torch.mm` / autograd instead of using them as ground truth.

### Provenance note

The worktree `shaders/transformer_cross_entropy.spv` is the plain build (GLSL
`exp()`, no SLEEF). That was confirmed by extracting the embedded SPIR-V blobs
from pre-change binaries and by compiling both variants; the tree's `.spv` was
already the plain build before this round, so no behavior was lost when it was
regenerated.

Next: localize the remaining backward ulp for Gemma3 (the layer-1 MLP backward
entry point above), then re-check gemma4, smollm3, qwen3_5_full, mistral4,
minimax_m2, and finally re-run the full matrix plus the cargo ladder.

## Round: CE reduction topology + RMSNorm vector-reduction topology (Intel Gen9)

### Root cause 1: the LoRA cross-entropy sum was folded as an AVX512 tree

`transformer_cross_entropy.comp` (`HIERARCHOS_LOG_SOFTMAX_GRAD`, used by the LoRA
path and Falcon H1) reproduced torch's `log_softmax` sum with a 16-lane
accumulator and a balanced tree. This torch build is **MSVC**, so ATen's
`VecReduceAllSIMD` specialization is compiled out and `vec_reduce_all` takes the
slow path: an **8-wide** accumulator seeded with the first chunk, later chunks
adding into the same lanes, then the eight lane totals folded **sequentially**
into lane 0. Rewriting the kernel to that topology makes
`native grad_logits == HF lm_head.grad_output` **bit-exact (0/128, was 30/128 at
3.7e-09)** and flips Gemma3's `saved` stage green.

Evidence: on 300 synthetic rows the MSVC slow-path sum + SLEEF exp + a
correctly-rounded log reproduces `torch.log_softmax` **300/300**; the 16-lane
tree does not. On the real Gemma4 fixture, `exp=sleef fold=8seq` is
**0/128** for every log/probability variant, while `fold=16tree` with the
emulated SLEEF exp is the only combination that fails.

### Root cause 2: the RMSNorm reduction topology was per-family instead of global

Every remaining red surface (`smollm3`, `qwen3_5_full`, `mistral4`,
`minimax_m2`, `minimax_m3`, `gemma4`) failed **only** on the token-embedding
weight gradient (Gemma4 additionally on `layers.0.input_layernorm.weight`).
That gradient is `token_embedding_scale * layers.0.grad_input`, so the whole
gate reduces to the accuracy of the layer-0 input gradient (2-6 ulp).

Differential diagnosis against HF captures:

* Gemma3's backward is **bit-exact at every boundary** (`grad_input`,
  `grad_norm_output`, `grad_mlp_output`, `grad_mlp_gate_activation`, ... all
  `0.000e+00`). Its forward is bit-exact too.
* Gemma4's identical-geometry fixture is 1-3 ulp off at every boundary, and its
  first divergence is the final RMSNorm *input* gradient.
* Gemma3 and Gemma4 differ only in `query_pre_attn_scalar` and in the RMSNorm
  class (Gemma4: `x * pow(mean,-0.5) * w`; Gemma3: `x * rsqrt(mean) * (1+w)`).

The decisive experiment (SmolLM3 final norm, HF capture, dim 16):

```
forward  8-lane reduction + sqrt-then-reciprocal rsqrt  -> 0/64 exact
forward  source order / 16-lane tree / AVX chunk order  -> 31-47/64 wrong
backward 8-lane dot + materialized autograd form        -> 0/64 exact
backward fused closed form or source-order dot          -> 11-35/64 wrong
```

So the reference (and the correct emulation) is the same MSVC/AVX2 topology as
root cause 1: **8 interleaved lanes + a sequential lane fold, in both the
forward mean square and the backward `(grad*w)*x` dot**.

### Fix

The shader already contains that model as "vector CPU-rsqrt mode"
(`extra_rsqrt_refinement == 3`, which selects the 8-lane reduction in the
forward and the materialized autograd backward form), but the backward push
selector sent mode 3 to the 16-lane tree whenever `materialized_backward` was
set. `RmsNormBackwardPush`/`RmsNormInputGradResidualPush` now map mode 3 to
`unfused_products == 2` (8-lane dot + autograd) ahead of the materialized
selector, and `SmolLm3`, `Mistral4`, `Qwen35` and `MiniMaxM3VLText` now select
`use_vector_cpu_rsqrt` at all three norm construction sites.

Result (`verify_peft_green_matrix.py --stage saved`):

| family | saved before | saved after | embed_tokens gradient |
| --- | --- | --- | --- |
| smollm3 | fail 2.980e-07 | **pass** | 1.490e-07 |
| mistral4 | fail 2.384e-07 | **pass** | 5.960e-08 |
| qwen3_5_full | fail 2.384e-07 | **pass** | 5.960e-08 |
| minimax_m3 | fail 2.086e-07 | **pass** | 5.960e-08 |
| gemma4 | fail 7.153e-07 | fail 7.153e-07 | input_layernorm 7.153e-07 |
| minimax_m2 | fail 3.576e-07 | fail 3.576e-07 | 3.576e-07 |

SmolLM3's `lora` stage re-verified after the change (`base_max_abs`
4.587e-08, `peft_max_abs` 2.790e-08, `gradient_max_abs` 2.095e-09), so the
forward did not regress.

### Re-verified gates after both fixes

* `cargo test --lib` (debug) 689 passed / 0 failed / 9 ignored.
* `verify_vendor_parity.py` PASS (21 resolved + 21 forced-portable, worst
  5.960e-08).
* `verify_hf_training.py --headline-strict` result `pass` (worst 5.960e-08).
* `verify_peft_green_matrix.py --stage all --jobs 2`: in progress; the first
  families (gpt2, llama, mixtral, qwen3_next, phi4_multimodal_text,
  deepseek_v4, ...) are green on all three stages, confirming the CE change did
  not regress the family that the 16-lane tree had previously satisfied.

### Remaining red (localized)

* `gemma4`: `extra_rsqrt_refinement == 0` with `pow_rstd != 0`, so its norm
  still uses the source-order materialized reduction in the forward and a
  source-order dot in the backward. Giving Gemma4 mode 3 would apply the 8-lane
  model to the pow-RMS norms; the forward would then change too, so it must be
  A/B tested against the base logits matrix in the same run.
* `minimax_m2`: already mode 3 for every norm (8-lane forward and backward), so
  its remaining 3.576e-07 comes from outside the norm path (MoE/router or
  attention adjoint topology).

### Gemma4 follow-up: the norm reduction is not the remaining source

Gemma4 was then given the same `use_vector_cpu_rsqrt` treatment on its layer
norms (it already had it on `v_norm`). Its forward norms stayed bit-exact
(`layers.{0,1}.norm_output` and `.mlp_norm_output` == HF `*.value`, 0/64) and
its `saved` metrics did not move at all
(`a_gradient_max_abs` 7.153e-07, `input_layernorm` 7.153e-07), so the norm
reduction topology is already correct for that fixture through the
`pow_rstd` path.

What the trace does show is why Gemma4 is the hardest surface: its RMSNorm
sandwich amplifies the backward to `|grad| ~ 8.9` (layer 1) and `~ 15.9`
(layer 0), so **one ulp is 1.07e-06 and even 0.45 ulp exceeds the absolute 2e-7
gate**. Observed relative errors are still only 1.5e-07 .. 4.3e-07 (1.3-3.6
ulp), first appearing at the attention-core adjoints
(`grad_v_norm_output` 3.18e-07, `grad_q_norm_output` 1.60e-07) and then
accumulating down to `layers.0.grad_input` 3.98e-07 (1.788e-07 absolute), which
`token_embedding_scale = 4.0` turns into the reported 7.153e-07. Gemma4
therefore needs genuinely bit-exact attention/MLP adjoints, not ulp-level
accuracy; `minimax_m2` is in the same class (already mode 3 everywhere, still
3.576e-07).

The relative-error map script used for this localization is
`hier_rel4.py <trace.json> [tensors.pt]` with the HF capture produced by
`diagnose_peft_backward_boundary.py <family> --module model.embed_tokens
--adapter a --trace <trace> --tensors <out.pt>`.

## Round: ATen tiny-fixture attention/linear oracles (qwen2_5_sliding_tied green)

This round localised and removed the two remaining sources behind
`qwen2_5_sliding_tied`'s red `saved` stage, then re-ran the whole 32-family
matrix. Nothing about the tolerance moved: the gate is still the absolute
`2e-7` from `verify_peft_saved_modules.py`, and the vendor bit-exactness
contract is untouched.

### Finding 1: ATen's CPU GEMM uses a two-lane topology when N <= 8

Probing `torch.nn.functional.linear` directly on the pinned MSVC build with
synthetic operands and comparing every plausible accumulation order against
ATen's own output (via `ctypes` `fmaf` for exact fused steps) gives a clean
kernel-selection rule for `M = 4` (the fixture geometry):

| N | K | exact-match accumulation | score |
|---|---|---| --- |
| 8 | 16 / 24 / 32 / 64 | two interleaved FMA lanes keyed by `i % 2`, folded by a single add, then `+ bias[n]` | 100% (192/192 at K=16, 768/768 at rows=16) |
| 4 | 16 | same two-lane fold | 100% (96/96) |
| 16 | 16 | serial FMA, then `+ bias[n]` | 100% (384/384) |
| 24 | 16 | serial FMA | 79.7% (459/576) |
| 32 | 16 | serial FMA | 100% (768/768) |

The bias must be applied **after** the fold: the bias-seeded forms score
120/192 and 87/192, i.e. they are simply the wrong oracle.

The backend already had `linear_forward_lane2.comp` for this topology, but its
selector only fired on `!has_bias && output_dim == 8`. Attention-bias models
(Qwen2/Qwen2.5, whose `Linear(hidden, 2 * head_dim) + bias` K/V projections are
exactly N = 8) therefore went through the serial-FMA `linear_bias_forward`
module. Measured against the HF capture: the `q_proj` base output (N = 16) was
bit-exact while `k_proj` and `v_proj` base outputs (N = 8) were one ulp away on
23/32 and 19/32 elements, and the parent projection (base + LoRA) inherited
exactly that difference.

Fix: new `shaders/linear_bias_forward_lane2.comp` (same two-lane body as
`linear_forward_lane2.comp`, plus the bias binding and an epilogue
`result + bias[out_col]`), compiled with glslang 16.6.0, wired as a plain
`vulkan::ComputeKernel` and selected when `has_bias && output_dim == 8` -
mirroring the existing biasless selector so the vendor slot is not involved.
Result: `k_proj` and `v_proj` base outputs became **bit-exact (0/32)**.

### Finding 2: ATen's tiny CPU softmax/BMM oracle is mode 9, for any model

The Qwen2 fixture's layer-0 chain was then traced stage by stage. Rope and
Q/K/V were bit-exact; the first divergence was the attention core: native
`attention_output` (the `o_proj` input) was one ulp away on 20/64 elements,
with rows 0 and 1 exact and rows 2/3 (3 and 4 visible keys) wrong.

Enumerating attention arithmetic offline (exact-FMA `ctypes`, a faithful Python
port of `fp32_sleef_exp.glsl`, correctly rounded divides) against HF's own
captured `o_proj.input` isolates the oracle exactly:

| QK | P@V | exp | normaliser | diff vs HF |
|---|---|---|---|---|
| any | materialised products + sequential add | SLEEF u10 | multiply by `1/sum` | **0 / 128** |
| any | serial FMA | SLEEF u10 | multiply by `1/sum` | 1 / 128 |
| any | materialised | SLEEF u10 | per-element divide | 16 / 128 |
| any | serial FMA | correctly rounded exp | multiply by `1/sum` | >= 1 / 128 |

That is precisely the existing "mode 9" enumeration (materialised QK and P@V
products, SLEEF u10 exp, correctly rounded FP32 reciprocal), which had only
been wired to `Llama` and `Gemma3/Gemma4`. Extending it to the `Qwen2`
architecture fixes the whole forward: `post_attention_layernorm` output,
`final_prefix` and `final_norm_output` all became **bit-exact (0/64)**, the
residual-chain drift disappeared, and `qwen2_5_sliding_tied`'s `saved` stage
went green (`model.embed_tokens` `a_gradient_max_abs` 2.384e-07 -> 1.192e-07,
all four modules `pass: true`).

`qwen2_5_gqa` (the other Qwen2 fixture) was re-verified separately: `lora`,
`saved` and `switch` all green after the change.

### Full 32-family matrix re-run

`verify_peft_green_matrix.py --stage all --jobs 2` -> 32 families,
`failures = [gemma4, minimax_m2]`. Every other family is green on all three
stages, including `qwen2_5_sliding_tied` (fixed here) and the previously
fixed `gemma3` / `smollm3` / `mistral4` / `qwen3_5_full` / `minimax_m3`.
No regression was introduced by either fix.

### minimax_m2: forward improved, gate unchanged

`MiniMaxM2` also runs the generic mode-0 attention forward. Giving it mode 9
strictly improved the forward trace (`layers.0.attention_output` 44/128 ->
29/128 one-ulp differences, `layers.1.input_layernorm` 2.384e-07 -> 1.192e-07,
`layers.1.attention_output` 2.980e-08 -> 2.235e-08) and kept `lora`/`switch`
green, but the `saved` metric did not move
(`model.embed_tokens` `a_gradient_max_abs` stayed 3.576e-07). The off-line
oracle reproduction for this fixture is exact (feeding **both** HF's rope and
native's rope into the materialised-product/SLEEF/reciprocal pipeline returns
HF's `o_proj.input` bit-for-bit, 0/128), and a least-squares recovery of the
per-row probability vector from native's own `attention_output` agrees with
HF's probabilities to within the fit noise, so the residual difference is a
device-side rounding detail inside the mode-9 softmax path rather than a
topology choice. The forward change is kept (it is a strict parity
improvement) and `minimax_m2` remains the second red surface.

### Regression ladder after both fixes (all green)

* `cargo test --lib` (debug): **689 passed / 0 failed / 9 ignored**.
* `cargo test --release --lib`: **689 passed / 0 failed / 9 ignored**.
* `verify_vendor_parity.py`: **PASS** (21 resolved + 21 forced-portable,
  worst `5.960e-08` on both paths).
* `verify_hf_training.py --headline-strict`: **result `pass`** (`max_abs`
  1.192e-07 against the harness's own 2.0e-7 bound; the same 1.192e-07 is the
  worst for `smollm3`, whose attention path this round did not touch, so the
  bound is not sensitive to these changes).

New artifact inventory for this round:
`shaders/linear_bias_forward_lane2.comp` / `.spv` (new, 3964 bytes, compiled
with glslang 16.6.0 so `build.rs`'s FNV-1a registry picks it up), the
`LINEAR_BIAS_FORWARD_LANE2_SPV` const + `linear_bias_forward_lane2` kernel and
its `has_bias && output_dim == 8` dispatch arm, and the `VulkanTransformerArchitecture::Qwen2
=> 9` / `MiniMaxM2 => 9` arms in both the forward and the backward attention
push-constant builders in `src/transformer.rs`.

## Round 3: gemma4 `saved` green — correctly rounded `pow(mean, -1.5)`

### gemma4 root cause (measured, not inferred)

Gemma4's forward was already fully bit-exact (every `attention_output`,
`norm_output`, `mlp_output`, `final_norm_output` boundary matched HF at
`0.000e+00`). The whole `saved` gap came from the **RMSNorm backward
derivative factor**: Autograd differentiates the forward's
`torch.pow(mean_squared, -0.5)` and evaluates `torch.pow(mean_squared, -1.5)`
*independently*, and on these tiny strict-fixture tensors ATen's CPU `pow`
takes the **scalar** path, i.e. the correctly rounded double-precision result.
The device used the SLEEF u10 log/exp paraphrase
(`sleef_powf_neg_three_halves_u10`), which is one ulp high/low on a few percent
of rows. Gemma4's RMSNorm "sandwich" amplifies that seed at every lower norm
(|grad| reaches 8.9 at layer 1 and ~15.9 at layer 0, so **one ulp is
1.07e-06** — five times the 2e-7 gate). The module-level `*.grad_input` keys in
the boundary capture are PEFT module grads (base + LoRA branch):
`down_proj.grad_input == base_layer.grad_input == lora_A.grad_input` exactly,
and `g@W` can never reproduce them, so they are not usable as a dgrad oracle —
the base-layer dgrad path itself was verified consistent.

### The fix

* `shaders/fp32_sqrt_recip.glsl`: new `fp32_pow_neg_three_halves(x)`, a
  **correctly rounded** FP32 `x^(-3/2)` that needs no `shaderFloat64` and no
  fused multiply-add (Intel Gen9 lowers `fma()` to mul+add, which is exactly
  what breaks SLEEF's compensated two-float staging). It works from the exact
  integer predicate `candidate <= x^(-3/2) <=> candidate^2 * x^3 <= 1`, building
  both sides as 128-bit integer mantissas with 32-bit products and resolving
  round-to-even against the exact midpoint. Validated offline against exact
  rational arithmetic on 40013 samples: **0 mismatches**.
* `shaders/transformer_rms_norm_input_grad.comp` and
  `..._residual.comp`: new push field `exact_pow_backward`; when set, the
  backward uses `fp32_pow_neg_three_halves(mean_squared)` instead of the SLEEF
  paraphrase. The residual variant is edited in lockstep — it shares
  `RmsNormBackwardPush`, and a stale 6-field push block there aborted the
  pipeline with `kernel expected 24 push-constant bytes, got 28` until both
  `.spv` files were regenerated.
* `src/transformer.rs`: `exact_pow_backward = pow_rstd && extra_rsqrt_refinement == 3`
  on both backward dispatch sites, and the two hard-coded push sizes are now
  `size_of::<RmsNormBackwardPush>()` so the struct and the kernel can never
  drift apart again.
* `src/vulkan.rs`: the push-constant mismatch error now names the shader
  (`kernel transformer_rms_norm_input_grad expected ...`), which is how the
  stale `.spv` was localised.

Result: the worst `a_gradient_max_abs` across gemma4's four strict modules
(`model.layers.0.mlp.down_proj`, `model.layers.0.input_layernorm`, `lm_head`,
`model.embed_tokens`) drops **7.153e-07 -> 7.45e-09**, every module reports
`pass: true` with `a_frozen_base_max_abs == 0.0`, and the family is green on
`lora`, `switch` **and** `saved`.

Rejected alternatives, for the record: re-capturing with a LoRA-free oracle is
not needed once the base-layer quantity is computed directly, and the 0.065
`|grad_input - g@W|` gap on `down_proj` is entirely the LoRA branch
(`t1 + 2*t2` with `lora_alpha/r = 4/2 = 2.0` reproduces HF's module grad
bit-exactly, 0/64).

### Full 32-family matrix after this round

`verify_peft_green_matrix.py --stage all --jobs 2` -> `families: 32`,
`failures: [minimax_m2]`. **31/32 families green on all three stages.**

### minimax_m2: where the remaining 3 ulps come from

The `saved` gate fails only on `model.embed_tokens`
(`a_gradient_max_abs` 3.576e-07 = 3 ulp, `b` 1.192e-07); the other three
modules pass. The RMSNorm derivative formula is **not** the cause here, and
this was verified rather than assumed: a direct ATen probe
(`t.rsqrt(t).backward(1.0)` over 11 magnitudes) shows ATen's rsqrt backward is
`-0.5 * rstd^3` with the *rounded* `rstd` (11/11), **not** the correctly rounded
`x^(-1.5)` (9/11 differ). MiniMax-M2's norm is `x * torch.rsqrt(variance + eps)`,
so the existing `rstd*rstd*rstd` device branch is already the right formula —
the gemma4 `pow` fix must not (and does not) apply to it.

The remaining divergence is upstream of the norms, in the rope elementary
functions. Measured against `torch.cos`/`torch.sin` (ATen, the oracle):

| implementation | rope angles (64) | random 30k, cos / sin |
|---|---|---|
| `shaders/fp32_sleef_trig.glsl` (SLEEF scalar u1, non-FMA) | 4 / 64 cos, 0 / 64 sin | 661 / 712 |
| faithful AVX2 `xcosf_u1`/`xsinf_u1` transcription (exact FMA) | 4 / 64 cos, 0 / 64 sin | 669 / 711 |
| SLEEF `xcosf`/`xsinf` (u35) | — | 9462 / 8564 |
| MSVC `cosf` / `sinf` (ucrtbase) | 2 / 64 cos, 0 / 64 sin | 1446 / 1431 |

So ATen's CPU float32 `cos`/`sin` is a 1-ULP (SLEEF-u10-like) kernel that is
*not* any of the four transcriptions above: it is off the correctly rounded
result on 4.9% of random arguments, yet on the four rope angles that currently
mismatch it returns the **correctly rounded** value while both SLEEF u10
transcriptions are one ulp high (and both agree with each other bit-for-bit).
The residual is a 1-ulp difference in `cos` at 2 of the 16 rope frequencies,
which is why the family stays red. Closing it needs a bit-exact transcription
of the specific CPU trig kernel this torch build dispatches; that kernel has
not been identified yet, and no shader change was made on a guess.
### Regression ladder after the gemma4 fix (all green)

* `cargo test --lib` (debug): **689 passed / 0 failed / 9 ignored** (27.96s).
* `cargo test --release --lib`: **689 passed / 0 failed / 9 ignored** (21.07s).
* `verify_vendor_parity.py`: **PASS** - 21 resolved + 21 forced-portable
  comparisons, worst `max_abs` `5.960e-08` on both paths, i.e. the tuned Gen9
  modules and the portable AMD-target modules are still identical.
* `verify_hf_training.py --headline-strict`: **`result: pass`** (it raises on any
  loss mismatch or any parameter above the harness's own 2.0e-7 bound, so
  reaching the final payload is the pass condition), worst `max_abs`
  `1.192e-07` at `model.layers.1.self_attn.q_norm.weight`. The same
  `1.192e-07` was already the worst value before this round's changes in a
  family this round did not touch and the bound is 2.0e-7, so the shift noted
  in the previous round (5.96e-08 -> 1.19e-07) stands as a documented
  observation rather than a regression introduced here.

Files touched this round: `shaders/fp32_sqrt_recip.glsl` (new
`fp32_pow_neg_three_halves` plus the 128-bit integer comparison helpers),
`shaders/transformer_rms_norm_input_grad.comp`,
`shaders/transformer_rms_norm_input_grad_residual.comp` (both recompiled to
`.spv` with glslang 16.6.0; the FNV-1a signature registry in `build.rs` accepts
them), `src/transformer.rs` (push field, gate, `size_of` push sizes) and
`src/vulkan.rs` (shader-named push-constant error). Nothing is committed or
staged: the working tree remains uncommitted by request.

Independent confirmation of the ATen trig characterisation (no mpmath involved):
`torch.cos` vs the double-precision libm result rounded to float32 (equal to the
correctly rounded float32 except for a vanishing double-rounding fraction)
differs on **1958 / 40000** random arguments in `[-60, 60]` (`sin`: 1993),
always by exactly one ulp in either direction. So ATen's CPU float32 `cos`/`sin`
is a genuine 1-ULP SIMD kernel, the four candidate transcriptions above are all
1-ULP kernels that are bit-different from it, and matching it bit-for-bit is the
remaining prerequisite for `minimax_m2`.
## Round 4: minimax_m2 narrowed to one elementary-function call (gemma4 held green)

Objective this round: fix `minimax_m2` without disturbing the gemma4 fix.
Outcome: not fixed, but the blocker is now a single, exactly characterised
elementary-function call rather than a chain, and two plausible repairs were
measured and rejected rather than guessed at.

### The device trig is verified, not inferred

Replaying native's rope from the fresh trace
(`gradient-probe-n7gwabyr/trace.json`) reproduces native's own
`layers.0.q_rotary_output` / `k_rotary_output` **exactly (0/128, 0/64)** when fed
`fp32_cos`/`fp32_sin` from `shaders/fp32_sleef_trig.glsl`, with the standard
`rotate_half` partner mapping. The same replay fed ATen's `torch.cos`/`torch.sin`
leaves 6/128 q and 6/64 k elements one ulp away. So the rope topology (partner
map, mul/add order, unit attention/query scaling) is bit-exact, and the whole
forward difference is the trig values themselves.

### Backward attention mode is not the cause

`MiniMaxM2`'s backward arm was left at mode 2 by the previous round. Moving it to
the forward's mode 9 (as Llama and Qwen2 have) was **measured neutral**: the
gate reported byte-identical values (`model.embed_tokens` 3.576e-07 / 1.192e-07,
the other three modules also unchanged). The change was reverted so the tree
stays at the exact state that produced the 31/32 matrix and the green ladder.
This also corroborates the boundary map, which shows the forward already
diverging before the adjoint runs: `layers.0.input` bit-exact, `layers.1.input`
(forward) 1.86e-09 over 18/64, and the backward then amplifying that to
`layers.1.grad_input` 1.79e-07 and `layers.0.grad_input` 3.58e-07.

### Why the trig cannot be patched without the real kernel

A full offline fingerprint was built from **every fixture's actual rope angles**
(config-derived base/head_dim/partial_rotary_factor, positions `0..seq-1`):
32 fixtures, 456 entries, **61 distinct angles**, with ATen's `cos`/`sin` as the
oracle. Against that fingerprint:

| candidate | distinct angles wrong |
|---|---|
| `fp32_sleef_trig.glsl` (current port) | **3** (7 entries) |
| 24 variants searched (3- vs 4-constant reduction, FMA vs Dekker poly, FMA vs Dekker vs exactly-rounded product, half-even vs half-away `rint`) | 3, none better |
| exactly rounded (double libm rounded, mpmath-verified) | many more |
| exact rounding of SLEEF's own two-float product | 2.0% of random inputs |
| SLEEF u35 (`xcosf`) | 31% of random inputs |
| MSVC `cosf` (ucrtbase) | 4.8% of random inputs |

Two consequences matter:

1. **A correctly rounded shader trig would be a regression.** Across the 61
   fingerprint angles the current port disagrees with ATen on 3; the correctly
   rounded value disagrees on far more, and it would *introduce* errors in ~19
   families - including `gemma4` itself (`gemma4` has a cr-bad angle and a
   port-good one). Since the round's constraint is to keep gemma4 green, this
   route is rejected on measurement.
2. **No SLEEF structural variant reproduces ATen.** The search floor is the
   current port's 3 angles; ATen's CPU float32 `cos`/`sin` is a 1-ULP kernel
   (independently confirmed: 1958/40000 random arguments differ from the
   correctly rounded value by exactly one ulp) that is none of the above.
   `torch/lib/sleef.lib` is linked statically and its symbols are not exported
   from `torch_cpu.dll`, so the kernel cannot be called directly either.

Net position: **31/32 families green** (only `minimax_m2`'s `saved` stage red, at
3 ulp of one embedding-gradient row), gemma4 green on all three stages, ladder
green. The remaining blocker is a bit-exact transcription of one unidentified
CPU trig kernel; no shader change was made on a guess.

## Rope tiny-argument repair: the reference kernel's truncation-free region

The previous round concluded that `minimax_m2`'s residual came from one elementary
function and could not be reproduced by any SLEEF transcription. This round found the
missing piece by changing the *oracle*: every earlier fingerprint compared against
`torch.cos` applied to a tensor I had built myself, which is **not** what the fixtures'
reference computes. The reference runs the family's HF rotary module, so the target is
that module's own table.

### The correct oracle

For each `*-strict` fixture the reference cosine table was rebuilt exactly as HF builds
it -- `inv_freq = 1.0 / base ** (arange(0, dim, 2).float() / dim)` in float32,
`freqs = outer(positions, inv_freq)`, `emb = cat((freqs, freqs), -1)`, `emb.cos()` --
giving **848 entries over 32 families, 52 distinct angles**. (A live run of the probe
captures the same table as `model.rotary_emb.value`, which is how the on-device check
below is done.)

Against that oracle:

| candidate | cos mismatches | sin mismatches |
|---|---|---|
| shader port pre-repair (SLEEF u1 path) | **14** | 0 |
| correctly rounded (`math.cos`) | 114 | -- |
| `Sleef_cosf8_u35` / `cosf4_u35` / scalar `u35` | 284 | -- |
| SLEEF u10 with FMA / u35 with FMA | 284 | -- |
| `Sleef_cosf8_u10` (all four u10 symbols) | 14 | -- |
| MSVC `cosf` (= correctly rounded here) | 114 | -- |

The 14 mismatches sit on **3 distinct angles**: `4.47213621e-4` and `8.94427241e-4`
(`minimax_m2`, `minimax_m3`, `minimax_m3_dense`; `1/sqrt(5e6)` and `2/sqrt(5e6)`) and
`1.41421345e-3` (`smollm3`). At those three the reference returns the **correctly rounded**
value while the reduction-based path returns its neighbour. `sin` was already exact
everywhere -- for arguments this small `sin(x)` is `x` to within half an ulp.

The reference kernel is not correctly rounded in general (it deviates by exactly one ulp
*upward* on ~9% of random arguments in this band, never downward), so a wholesale switch
to correct rounding is not the fix: that was measured at 114/848, breaking `gemma3`,
`gemma4`, `deepseek_v4`, `falcon_h1` and `gpt2`/`kimi` angles.

### The repair

`shaders/fp32_sleef_trig.glsl`: `fp32_cos` gains a truncation-free region.

```
if (abs(d) < 9.0e-4) return 1.0 - 0.5 * d * d;
```

`0.5 * d * d` evaluates as `(0.5*d)*d`: the multiply by `0.5` is exact, `d*d` is
correctly rounded by one multiply, and the subtraction rounds to nearest, so the result
is the correctly rounded cosine to within `1.4e-14` -- four orders of magnitude inside
the half-ulp boundary of `2.98e-8` at that magnitude. The next series term is
`d^4/24 < 3.4e-14`. No `fma`, no double-float staging, so Gen9's mul+add lowering of
`fma` cannot perturb it.

The cutoff is chosen from measurement, and the measurement is exhaustive for this suite:
across all 848 reference entries there is **no angle in `[0, 9.0e-4)` other than the
twelve `minimax_m*` entries the repair fixes** (the next smallest, `gpt2`/`kimi`'s
`9.48683359e-4`, and `smollm3`'s `1.41421345e-3` and `gemma3`'s `2.00000009e-3`, all lie
above it and all already match). Everything above the cutoff keeps the existing
reduction path bit-for-bit.

### Verification

* Offline, against the 848-entry oracle: cos mismatches **14 -> 2**; `minimax_m2`,
  `minimax_m3`, `minimax_m3_dense` **4 -> 0** each; the remaining 2 are `smollm3`'s
  `1.41421345e-3` entries, above the cutoff, unchanged from before. `sin` stays at 0.
  No other family's table changes at all.
* On device: the HF capture's own cosine table
  (`model.rotary_emb.value`, 4 positions x 16 frequencies) is reproduced by the patched
  port with **0 mismatches of 64**, and `sin` **0 of 64**. Replaying the device's rope
  from the fresh trace -- `q_rot = q_norm*cos + rotate_half(q_norm)*sin` per head -- with
  the *reference* tables reproduces the device's `q_rotary_output`/`k_rotary_output`
  **bit-exactly (0 of 256)**. Pre-repair the same replay against the reference tables left
  6 of 128 entries one ulp out.
* Boundary map, fresh trace vs fresh HF capture at f32 resolution: `layers.0.input`,
  `norm_output`, `q/k/v_linear_output`, `q/k_norm_output` are now **bit-exact (0 of 64,
  0 of 128)**. The previous round's first forward divergence (`layers.1.input` off by
  1.86e-09 over 18 of 64 entries) is gone; the first non-zero difference in the forward
  is now at the layer-0 attention output and is smaller than one f32 ulp for 61 of 64
  entries (2.3e-10 over 3 of 64). `final_prefix` went 1.863e-09 (20/64) from a
  pre-repair divergence one layer upstream.

### Why the saved gate is still red

`model.embed_tokens` still reports `a_gradient_max_abs` `3.5762786865234375e-07`
(= 3 ulp of its `1.776` element) and `b_gradient_max_abs` `1.1920928955078125e-07`
(= 1 ulp), byte-identical to before the repair, and the other three saved modules pass
with values from `1.1e-08` to `8.9e-08`.

The fresh boundary map shows why this is not a rope defect: the *forward* is bit-exact to
the last f32 bit through every stage the rope feeds, while the *backward* differences are
1-3 ulp **of the gradient values' own magnitude** -- `layers.1.grad_input`
`1.192e-07` on values of `1.478` (exactly 1 ulp), `layers.0.grad_input` `3.576e-07` on
values of `1.776` (exactly 3 ulp) -- while the top of the backward is nearly exact
(`grad_logits` `2.8e-09` on values of `0.324`, `grad_final_norm_output` `9.3e-10`). That
is the signature of a backward that agrees up to accumulation order, not of a wrong
derivative: the 2e-7 absolute tolerance is only ~1.7 ulp at this fixture's `O(1.78)`
gradient magnitude, which is why 30 other families sit at 1.19e-07-1.79e-07 and pass.
Closing it needs the remaining forward and backward boundaries to agree to the last bit,
not another trig repair.

### State

Both `transformer_rope_forward.spv` and `transformer_rope_backward.spv` were regenerated
with glslang 16.6.0 and both debug binaries rebuilt, so the embedded shader signatures
match the sources. The reverted `MiniMaxM2` backward-mode experiment stays reverted.

## Round 5: minimax_m2 `saved` green -- attention LoRA-A topology + MoE router sigmoid

The last red cell of the 32-family saved matrix is green. Two defects, both in the
*fp32 elementary/topology* class this audit has been chasing, and both proven by exact
round-to-nearest replays of the reference's own captured intermediates rather than by
tolerance games.

### The saved gate was riding on the forward, not on the backward

`model.embed_tokens` failed with `a_gradient_max_abs = 3.5762786865234375e-07`
(= 3 ulp of its `1.776` element, and there is no accumulation in the embedding gather:
`input_ids = [1,7,3,11]` are distinct, so each gradient row is a straight copy of the
layer-0 input gradient row). The old reading -- "backward accumulation order differs" --
is wrong. Two measurements kill it:

* Replaying the reference's own fused cross-entropy backward (torch, `shift_logits[:, :-1]`
  vs `ids[:, 1:]`, mean over 3 supervised tokens) on the **native's own logits** reproduces
  the native's `grad_logits` **bit-exactly (0 of 128)**. The backward is not merely close,
  it is the reference algorithm; the single odd element of `grad_logits`
  (`idx 89`, `2.794e-09` = 3 ulp of `1.03e-02`) is *inherited* from the forward logits.
* The remaining logits error is one ulp of `0.25` (`2.980e-08` over 53 of 128 entries),
  i.e. the forward. The RMSNorm backward then multiplies by `rstd` (`55.7` at `hidden=16`,
  `hidden_size=16`), which is why a `9.3e-10` input difference leaves `5.96e-08`, and two
  layers of weight multiplies leave `3.576e-07`.

Everything below was found with one tool: an exact fp32 simulator (Python `Fraction`,
round-half-even after *every* operation, `fma` exact) that replays candidate reduction
topologies against the HF boundary capture
(`%TEMP%/mm2_tensors_post.pt`, `capture` dict) and the native trace
(`.peft-oracle-fixtures/minimax_m2-strict/gradient-probe-2j9449vh/trace.json`). Scripts:
`mm2_oproj.py`, `mm2_survey.py`, `mm2_confirm.py`, `mm2_moe.py`, `mm2_sigmoid.py`,
`mm2_headbwd.py` in `%TEMP%`.

### Defect 1 -- attention LoRA-A is the reference's eight-lane muladd GEMM from K=32

Boundary map, fresh capture: the whole layer-0 attention block is **bit-exact** --
`layers.0.attention_output` == HF `self_attn.o_proj.input` at **0 of 128** -- while
`o_proj.value` differs on 3 of 64 entries by `2.328e-10` (1 ulp of a `~2e-3` element).

Survey of every LoRA'd projection in the fixture (K = input width, N = output width,
exact RN32 replay of HF's captured `base_layer.value`, `lora_A.value`, `lora_B.value`):

| module                    | K  | N  | topologies that reproduce HF exactly |
|---------------------------|----|----|--------------------------------------|
| `*.q/k/v/o_proj.base`     | 16 | 32 | serial FMA chain (unique class)      |
| `*.o_proj.base`           | 32 | 16 | serial FMA chain (unique class)      |
| `*.lora_B` (all)          | 2  | 16/32 | serial FMA chain (unique class)   |
| `*.lora_A` where K=16     | 16 | 2  | nine topologies coincide             |
| `*.lora_A` where K=32     | 32 | 2  | **`(8, stride, muladd, half-fold)` only** |

The 16-lane FMA kernel the attention path used is *not* in that last row's set. Both
readings were then confirmed end to end: replaying the full `o_proj` with the 16-lane FMA
LoRA-A reproduces the **native trace** bit-exactly (0 of 64) and leaves HF 3 of 64 apart;
replaying with the 8-lane muladd reproduces **HF** bit-exactly (0 of 64) and leaves the
native 3 of 64 apart. Base GEMM, LoRA-B and the `base + (delta * scale)` two-rounding
merge were already right.

Fix: `VulkanLinear::record_forward_mode` now selects `lora_a_forward_lane8_muladd` when
`self.input_dim >= 32` -- the same reduction-size keying the transformer-MLP path has used
since the gemma4 probe (`lora_a_forward_lane8_muladd.comp` already documents "a lane owns
two or more products ... the two topologies split"). Effect: `layers.0.attention_projection`
3 of 64 -> **0 of 64**.

### Defect 2 -- the MoE router evaluated HF's sigmoid with the driver's relaxed exp and divider

With the attention path bit-exact, the first forward divergence became the layer-0 MoE
output: `2.328e-10` over 55 of 64 entries (values `~7.9e-04`). An exact replay of HF's
`MiniMaxM2Experts.forward` (per-expert loop over the selected-token subsets, `sigmoid`
router, top-2, renormalize, `index_add_`) reproduces HF's capture bit-exactly (0 of 64),
and `mlp.input` is bit-exact (the post-attention norm output matched HF to the bit).

Then the cheapest possible test: solve the native's traced output as `w_a*E_a + w_b*E_b`
using **HF's own expert outputs** and integer-ulp shifts of the two routing weights. Every
token is explained exactly, with shifts `(0,-1)`, `(-1,-2)`, `(0,0)`, `(-1,+1)`. So the
expert path (gate_up, SiLU, product, down) and the weighted accumulation are bit-exact,
and the *only* wrong quantity was the routing weight -- i.e. `transformer_moe_router_topk.comp`,
which computed HF's `nn.functional.sigmoid` as the driver's `exp()` plus a reciprocal-multiply
division, and renormalized with the same relaxed divider.

The reference contract is the one already ported for `silu`: PyTorch CPU's vectorized
`1 / (1 + exp(-x))` with **SLEEF `expf8_u10`** and an **IEEE fp32 division**. Verified on the
fixture's own 12 router logits against `torch.sigmoid`: SLEEF + true division reproduces all
12; the same form with a correctly rounded exponential misses 1 of 12 (so it really is the
SLEEF polynomial, not a "more accurate" exp).

Fix in `transformer_moe_router_topk.comp`: include `fp32_fma_exact.glsl`,
`fp32_sqrt_recip.glsl`, `fp32_sleef_exp.glsl`;
`sigmoid_score(value) = fp32_div(1.0, 1.0 + sleef_expf_u10(-value))`; renormalize the
sigmoid route with `fp32_div` instead of `/=`. Effect: all four routing weight sets are now
exactly HF's (shifts `(0,0)` on every token), and `model.embed_tokens`'
`a_gradient_max_abs` went `3.5762786865234375e-07` -> **`1.7881393432617188e-07`**,
`pass: true`.

### Residual known item (documented; not gate-relevant)

The layer-0 MoE output still differs by `1.746e-10` over 54 of 64 entries. Cause is
isolated and is *not* a kernel-order guess: PyTorch's CPU GEMM is **M-dependent**, and HF
runs each expert on only its selected tokens (`M = 2,3,3` here) while the native grid runs
every expert over all `rows` (dense, with zero routing weights). Measured in torch:
`F.linear(x, W)` at `M=4` differs from the `M=2` subset result on the same row by up to
2 ulp of the `gate_up` output, and the same row at `M=1` differs again. Matching it needs
per-expert token compaction plus the reference's subset-GEMM topology (or reverse
engineering the M-dependent dispatch) -- a structurally larger change than this round's,
and one that does not move the gate: minimax_m2 now sits at `1.7881e-07 = 3*2^-24`, which
is precisely the suite's normal ceiling (six other green families report the identical
value; the suite median is `1.192e-07 = 2^-23`).

### Verification

* `minimax_m2` saved-modules, all four modules `pass: true`:
  `k_proj` `a 1.676e-08 / b 7.451e-09`, `input_layernorm` `4.657e-09 / 7.451e-09`,
  `lm_head` `2.980e-08 / 1.192e-07`, `model.embed_tokens` **`1.788e-07` / `1.192e-07`**
  (was `3.576e-07` / `1.192e-07`, `pass: false`); `frozen_base_max_abs = 0.0`,
  `resume_max_abs = 0.0`, `bank_isolation = true` on every module.
* Full 32-family `verify_peft_green_matrix.py --stage saved --jobs 2`:
  `{"families": 32, "failures": []}`, exit 0. The immediately preceding full matrix
  (`green-matrix-report.prior-1791341959202957600.json`) listed `['minimax_m2']`.
* gemma4 and gemma3 stay green inside that same run; the gemma4
  `pow(mean, -1.5)` repair and the rope tiny-argument repair are untouched by this round.
* `cargo test --lib`: 689 passed / 0 failed / 9 ignored.
* Regenerated/embedded artifacts: `shaders/transformer_moe_router_topk.spv` (glslang
  16.6.0, `-V --target-env vulkan1.1 -I.`), and the `transformer_parity`,
  `hierarchos-vulkan-transformer-logits`, `peft_bank_parity`, `peft_checkpoint` debug
  binaries rebuilt from the edited sources.

### State

Files changed this round: `hierarchos-vulkan/src/transformer.rs` (LoRA-A dispatch in
`record_forward_mode`) and `hierarchos-vulkan/shaders/transformer_moe_router_topk.comp`
plus its regenerated `.spv`. Nothing staged, nothing committed. With this round the full
32-family saved matrix is green for the first time; the tree is ready for the downstream
regression pass.

## Round 6: minimax_m2 three-stage handoff pass -- diagnostic repair, residual attribution, artifact provenance

Round 5 left minimax_m2's `lora` and `switch` stages holding pre-round evidence
(the saved-only matrix wrote `switch` as `status: not run`) and recorded the
`1.746e-10` layer-0 MoE difference as a known item whose cause was inferred rather
than measured. This round closes both, repairs a crashing diagnostic, and audits
every shader artifact this work stream changed.

### Fixed: `validation/diagnose_peft_backward_boundary.py` aborted on an unpaired boundary

The report loop indexed `row["candidates"][0]` on the assumption that the native
trace always holds a tensor with the same element count as every captured
boundary. When it does not -- the normal case when the trace came from a narrower
surface -- the run raised `IndexError: list index out of range` *after* the HF
capture had been computed, which is the failure observed against this tool in the
previous round: `--tensors` was the only salvageable artifact and the pairing
report was lost. The loop now prints `best=<no native tensor of matching size>` for
such a boundary and continues.

Verified through the tool's own interface, not by inspection:

* real trace (`gradient-probe-2j9449vh`, the post-Round-5 binaries): every
  captured boundary has a same-size native tensor, so the run reports the full
  pairing table and writes both `--output` and `--tensors`, exit 0. On this trace
  the pre-fix code would not have crashed either, so this run confirms the report
  is intact but is not itself evidence for the guard.
* synthetic trace (`{"zzz_no_match": [0, 0, 0]}`): no native tensor can pair with
  any boundary, which is exactly the condition that raised `IndexError`. The new
  branch executes for every captured boundary and the run still exits 0, so the
  guard is on the executed path rather than dead code.

### Residual MoE item: now measured, and deliberately not patched

New `validation/diagnose_minimax_m2_moe_row_count.py` (auto-discovers the newest
`gradient-probe-*/trace.json`, including rotations into `*-strict.prior-*`) replays
the fixture's own weights through three expert call shapes and compares each
against the native trace:

| expert call shape | expert GEMM `M` | vs native trace |
| --- | --- | --- |
| `subset` (HF's own) | `[2, 3, 3]` | `1.746e-10` over 54/64, not bit-exact |
| `dense` (native's) | `[4, 4, 4]` | **bit-exact, 0/64** |
| `mixed` (dense GEMMs + HF's own scatter) | `[4, 4, 4]` | **bit-exact, 0/64** |

Two results, both new this round:

* The native graph is **bit-identical** to the reference MoE block under the dense
  call shape. Dropping the expert dispatches from `M = rows` to `M = 2/3/3` is the
  *entire* remaining difference; the router, SiLU, elementwise product, the
  gate/up/down projection values and the routing-weight multiply are all already
  exact.
* `mixed` is bit-exact too, so HF's `index_add_` scatter and the native
  `moe_weighted_accumulate` already agree bit-for-bit on these values. The
  accumulation order is not part of the residual, which means the fix surface is
  exactly the two expert GEMMs' row count.

Why it is still not patched: the per-expert call size is data-dependent (`[2,3,3]`
here; in general any split of eight selections over three experts, including an
empty expert), and the dispatch grid is fixed at record time. Reproducing it needs
a GPU-side routed-token count plus selected-row gather/scatter feeding an
`M`-selected accumulation topology, mirrored in the backward expert GEMMs -- a
structural change to the *shared* MoE forward used by `mixtral`, `gpt_oss`,
`minimax_m3`, `minimax_m3_dense`, `qwen3_5_moe` and the `qwen4_exp_*` surfaces.
The measured residual is three orders of magnitude below the `2e-7` ceiling and no
stage's `pass` flag moves with it, so changing that path immediately before handoff
is all risk and no signal. The probe above exists so the next round starts from the
exact call-shape target instead of re-deriving it.

### Full 32-family, three-stage matrix on the current tree

Round 5 re-ran only `--stage saved`. Fresh run, `--jobs 2`, prebuilt binaries:

```
HIERARCHOS_PEFT_USE_PREBUILT=1 python validation/verify_peft_green_matrix.py --stage all --jobs 2
{"families": 32, "failures": []}
```

* `2026-10-07T03:58:18Z` -> `04:30:34Z`; all **96** stage results `pass: true`
  (`lora` 32/32, `switch` 32/32, `saved` 32/32); exit 0, so no family has a stale
  stage result any more.
* Provenance `inputs_unchanged: true`: one binary/shader/source state produced the
  whole run, so no result in it was inherited from an earlier build.
* minimax_m2, all three stages green for the first time:
  * `lora`: `peft_max_abs 3.351e-08`, `gradient_max_abs 4.657e-09`,
    `two_step_adamw 3.725e-09`, `frozen_base 0.0`, `resume_hf 3.725e-09`.
  * `switch`: `adapter_a 1.433e-08`, `adapter_b 1.984e-08`,
    `disabled_base 2.893e-08`; both adapters loaded and selected,
    `disabled_state true`.
  * `saved`: 4/4 modules -- `k_proj` `1.676e-08 / 7.451e-09`,
    `input_layernorm` `4.657e-09 / 7.451e-09`, `lm_head` `2.980e-08 / 1.192e-07`,
    `model.embed_tokens` `1.788e-07 / 1.192e-07`.
* `gemma3`, `gemma4`, `minimax_m3`, `minimax_m3_dense`, `mistral4` and the three
  `falcon_h1` variants are green in the same run, so Rounds 3-5 are untouched.

### Shader artifact provenance: every changed module reproduces exactly

All 338 `.comp`/`.spv` pairs were recompiled with the only compiler on this machine
(glslang 16.6.0, `.venv-vulkan/scratch/glslang/bin/glslang.exe`) to catch a source
edit whose `.spv` never got rebuilt -- the failure mode `build.rs` cannot see,
since it only embeds and hashes existing `.spv` files.

The decisive rule is that the embedded corpus carries whichever `--target-env` it
was built with, not that one env is canonical. With the matching env, **all 14
`.spv` files this work stream modified reproduce byte-for-byte from tracked
sources**:

* `--target-env vulkan1.0`: `linear_forward`, `silu_forward`,
  `transformer_attention_forward`, `transformer_attention_backward`,
  `transformer_gelu_tanh_forward`, `transformer_gelu_tanh_backward`,
  `transformer_cross_entropy`, `transformer_kimi_attn_res_softmax`,
  `transformer_rms_norm_input_grad`, `transformer_rms_norm_input_grad_residual`,
  and `falcon_h1_cross_entropy` (from `transformer_cross_entropy.comp` with
  `-DHIERARCHOS_LOG_SOFTMAX_GRAD=1`, matching the Round 4 note).
* `--target-env vulkan1.1`: `transformer_rope_forward`,
  `transformer_rope_backward`, and `transformer_moe_router_topk` -- so the Round 5
  sigmoid repair is definitely embedded in the dispatched module.

No stale or inert artifact was found. Two related clarifications:

* Comparing against a single env produces a wall of false mismatches; e.g. the
  `vulkan1.1` build of `silu_forward.spv` differs from the committed module in the
  SPIR-V header *and* mid-stream (SPIR-V 1.3 vs 1.0 code generation) at identical
  length and bound. Only the matching env is a valid byte test.
* Unmodified baseline modules assembled with shaderc/glslc (generator word
  `0x000d000b`) do not reproduce with glslang (generator `0x0008000b`); this is the
  documented situation in `VENDOR_TUNING.md` ("run it with a matching toolchain when
  auditing byte provenance", noting `glslc` produces equivalent modules with a
  different generator word) and is not a regression.

### Verification

* `cargo test --lib`: 689 passed / 0 failed / 9 ignored (27.62s).
* Full three-stage 32-family matrix above: exit 0, `inputs_unchanged: true`.
* `diagnose_peft_backward_boundary.py`: exit 0 on a real trace and on the synthetic
  trace that forces the new branch.
* `diagnose_minimax_m2_moe_row_count.py`: exit 0, no arguments needed.
* All 14 modified `.spv` files reproduce byte-for-byte.
* Nothing staged, nothing committed (`git diff --cached --name-only` empty).

### State

Changed this round: `hierarchos-vulkan/validation/diagnose_peft_backward_boundary.py`
(crash fix) and new `hierarchos-vulkan/validation/diagnose_minimax_m2_moe_row_count.py`.
No Rust, no shader and no binary was touched, so every Round 5 gate result stands and
the ally's regression sees the same tree plus two validation scripts. The one
outstanding item is the MoE expert-GEMM row count above, which is intentional,
measured, and below every gate.
## Round 7: minimax_m2's MoE expert GEMM is bit-exact at the reference's routed row count

### The residual Round 6 left behind

Round 6 closed with one known defect: the strict fixture's layer-0 MoE block sat
`1.746e-10` (max abs, 40/64 entries) away from the reference, and a cached-scheme
sweep over all `6^3` per-expert accumulation assignments reproduced the native
trace bit-exactly with exactly one of them -- `('l2m','l2m','l2m')`, i.e. every
expert accumulating as two interleaved muladd lanes. That is the `M == 2`
topology, but the fixture's router sends `[2, 3, 3]` tokens to the three experts,
so two of them should have been accumulating at a different row count.

The cause is structural. The reference `MiniMaxM2SparseMoeBlock` calls each
expert with only the tokens routed to it, and PyTorch's CPU GEMM picks its
accumulation topology from that call's row count `M`. The native graph keeps
routing on the device by dispatching every expert over all four rows and
zero-weighting the rows the expert did not receive, which is arithmetically
equivalent but not bit-identical. Round 6 measured the reference's rule
exhaustively (`gemm_rule.py`, exact FP32 round-to-nearest, re-confirmed on the
fixture's real K=16/N=32 and K=32/N=16 shapes):

| routed rows `M` | reference topology |
| --- | --- |
| `M == 1` | eight strided FMA lanes, reduced by cross-halves |
| `M == 2` | two strided lanes whose products are materialized in FP32 |
| `M == 3` | two strided FMA lanes, joined by one add |
| `M >= 4` | one serial FMA chain (`linear_forward.comp`) |

### What was built

Two new shaders plus a count kernel (all `--target-env vulkan1.0`, glslang
16.6.0, and all reproducing byte-for-byte from source):

* `transformer_moe_expert_count.comp` / `.spv` (2008 B, new): one entry per
  expert, the number of non-zero entries in that expert's routing column. The
  routers write exact zeros for unselected experts, so a column count *is* the
  routed token count. Push constants `{rows, experts}`; dispatched once per MoE
  layer in `record_forward` and `record_backward`.
* `transformer_moe_expert_forward.comp` / `.spv` (6776 B, rewritten): the fused
  topologies. `routed == 3` runs two interleaved FMA lanes joined by one add
  (the `linear_forward_lane2.comp` shape), `routed == 1` runs eight interleaved
  FMA lanes reduced by cross-halves, and everything else runs the serial FMA
  chain of `linear_forward.comp`. It declines `routed == 2`.
* `transformer_moe_expert_forward_muladd.comp` / `.spv` (4432 B, new): the
  `routed == 2` topology with its products materialized in FP32, and it declines
  every other count.

Rust (`src/transformer.rs`): two `TransformerKernels` slots
(`moe_expert_forward`, `moe_expert_forward_muladd`) sharing
`MoeExpertForwardPush`, and `record_transformer_moe_expert_projection` now
records both modules for each routed expert. The per-expert routed count lives in
`VulkanMoe::expert_counts` (allocated in `new` and in `replicate_shared_base`).
The pair's guards are complementary, so exactly one module writes each output
element; experts with a bias or an adapter still take the established dense
linear kernel, and the serial branch is arithmetically identical to the path that
was there before, so only the `M <= 3` experts changed behaviour. All four
`record_expert_forward` call sites carry the change (forward and backward, routed
experts only): the shared-expert calls still pass `None`.

### The decisive hazard: one NoContraction multiply de-fuses a whole module

The first implementation put all four topologies in one module with `precise`
accumulators, copying the `linear_forward_lane2.comp` idiom. The device had other
plans. Forcing the routed counts (a temporary count kernel writing fixed values,
then the same trace-and-sweep analysis) showed every branch silently accumulating
as muladd:

| forced counts | bit-exact assignment | what the branch was supposed to do |
| --- | --- | --- |
| `[2,3,3]` (one module) | `('l2m','l2m','l2m')` | `l2m`, `l2f`, `l2f` |
| `[1,4,5]` (one module) | `('l8m','l1m','l1m')` | `l8f`, `l1f`, `l1f` |
| `[1,4,5]`, every `precise` removed | `('l8f','l1f','l1f')` | **as intended** |

`linear_forward_lane2.comp` already recorded the family of effect ("on Intel Gen9
the compiler reads a NoContraction final add as a request to keep the whole
expression unfused and splits the upstream FMA ext-instructions into separate
multiply and add"). This round pinned the trigger down:

* `linear_forward.spv` and `linear_forward_lane2.spv` carry **zero**
  `NoContraction` decorations (`OpDecorate` 42, decoded from the committed
  modules), and both are verified fused on-device. lane2's `precise` accumulators
  emit none because nothing in them is contractible.
* The single-module build carried seven, all of them the three materialized
  products and four adds of the `routed == 2` branch -- yet the branches holding
  *no* decoration were de-fused as well.
* Stripping every `precise` from that same module (zero decorations) restored
  `('l8f','l1f','l1f')` from the same forced counts, with no other change.

So a module either materializes products or fuses FMAs, never both. The two
topology families are therefore separate modules, and the price is one extra
guarded dispatch per expert projection. A second, useful negative result: the
8-lane branch's plain (non-`precise`) cross-half fold reproduced the reference
tree bit-for-bit, so that fold does not need NoContraction either.

### Evidence

Per-topology probes, each a shader edit + rebuild + fresh native trace:

| forced counts | bit-exact assignment | topologies exercised |
| --- | --- | --- |
| `[1,2,3]` | `('l8f','l2m','l2f')` | `M=1` 8-lane fold, `M=2` muladd, `M=3` 2-lane FMA |
| `[9,9,9]` | `('l1f','l1f','l1f')` | `M>=4` serial FMA |
| real `[2,3,3]` | `('l2m','l2f','l2f')` | **the reference's own topology** |

`validation/diagnose_minimax_m2_moe_expert_topology.py` (new) is that sweep as a
permanent diagnostic: it caches each candidate topology's projection for every
expert's routed rows, sweeps all per-expert assignments, and reports the
bit-exact ones next to the rule's prediction. On the current tree, for both
layers of the strict fixture:

```
layers.0.mlp: rows=4 hidden=16 experts=3 top_k=2
  routed tokens per expert: [2, 3, 3]
  reference topology (experts [0, 1, 2]) ['l2m', 'l2f', 'l2f'] on M=[2, 3, 3]: differing=0/64 max_abs=0.000e+00 bit_exact=True
  bit-exact per-expert topology: ['l2m', 'l2f', 'l2f']
  conclusion: the trace agrees with the reference's per-expert topology

layers.1.mlp: rows=4 hidden=16 experts=3 top_k=2
  routed tokens per expert: [4, 0, 4]
  experts the router skipped (no observable topology): [1]
  reference topology (experts [0, 2]) ['l1f', 'l1f'] on M=[4, 4]: differing=0/64 max_abs=0.000e+00 bit_exact=True
  bit-exact per-expert topology: ['l1f', 'l1f']
  conclusion: the trace agrees with the reference's per-expert topology
```

Layer 0 exercises `M=2` and `M=3`, layer 1 exercises `M=4`; the probes cover
`M=1`. The block-level probe from Round 6 now reads:

```
layers.0.mlp: rows=4 hidden=16
  subset expert GEMM M=[2, 3, 3] (routed [2, 3, 3]) native_max_abs=0.000e+00 differing=0/64 bit_exact=True
  dense  expert GEMM M=[4, 4, 4] (routed [2, 3, 3]) native_max_abs=1.746e-10 differing=54/64 bit_exact=False
  conclusion: the native graph reproduces the reference's routed-row expert GEMM bit-for-bit
```

i.e. the native graph now reproduces the reference's routed-row call shape
exactly, and the dense shape it used before shows up as the residual.

Both diagnostics were also run against the probe traces to prove they detect a
disagreement rather than always agreeing: the `[1,2,3]` build reports
`bit-exact per-expert topology: ['l8f', 'l2m', 'l2f']` with `bit_exact=False`
against the rule's `['l2m', 'l2f', 'l2f']`, and the `[9,9,9]` build reports
`['l1f', 'l1f', 'l1f']`.

### Saved-module gate numbers (after the change)

From the 2026-10-07 matrix run, minimax_m2's saved stage (all four modules
`pass: true`, `bank_isolation: true`), which matches the direct
`verify_peft_saved_modules.py minimax_m2` run:

| module | a gradient | b gradient | a/b resume | a/b frozen base |
| --- | --- | --- | --- | --- |
| `model.layers.0.self_attn.k_proj` | 7.451e-09 | 7.451e-09 | 0.0 | 0.0 |
| `model.layers.0.input_layernorm` | 3.260e-09 | 4.657e-09 | 0.0 | 0.0 |
| `lm_head` | 1.490e-08 | 3.725e-09 | 0.0 | 0.0 |
| `model.embed_tokens` | 1.192e-07 | 1.192e-07 | 0.0 | 0.0 |

`k_proj` improved from Round 6's `1.676e-08 / 7.451e-09`; `model.embed_tokens`
stays at the suite's median `2^-23` (`1.192093e-07`), and the LoRA stage's
`peft_max_abs` is `3.911e-08`.

### Verification

* `cargo test --lib`: 689 passed / 0 failed / 9 ignored (25.81s) -- the same
  counts as the pre-change baseline.
* Full three-stage 32-family matrix (`--stage all --jobs 2`,
  `HIERARCHOS_PEFT_USE_PREBUILT=1`): exit 0, 32 families, `failures: []`,
  32/32 LoRA, 32/32 switch, 32/32 saved, and
  `green-matrix-report-provenance.json` reports `inputs_unchanged: true`
  (2026-10-07T05:40:52Z -> 06:08:40Z).
* Both MoE diagnostics exit 0 with the conclusions above, auto-discovering the
  newest trace (the matrix rotates the fixture directory, and both search the
  `.prior-*` siblings).
* Shader provenance: the three modules this round touched
  (`transformer_moe_expert_forward`, `transformer_moe_expert_forward_muladd`,
  `transformer_moe_expert_count`) each recompile byte-for-byte with
  `glslang -V --target-env vulkan1.0 -I.`; `build.rs` picks the new `.spv` up
  from its directory scan and found no signature collision.
* Nothing staged, nothing committed (`git diff --cached --name-only` empty).

### Documentation corrections made in this round

Two documents contradicted the current evidence and were corrected:

* `VENDOR_TUNING.md` gains a "Intel Gen9 module-level FMA de-fusion (measured)"
  subsection under shader provenance, recording the rule and the measurement that
  established it, since the same trap will apply to any future kernel that needs
  both materialized products and fused FMAs.
* `COMPATIBILITY.md`'s saved-module section still claimed 25/32 saved with seven
  red fixtures (nine module rows) and their 2026-10-05 values. All nine rows have
  been green since Rounds 3-5; the table now shows the former vs current values
  and the aggregate is 32/32 on all three stages, citing this round's run. No
  tolerance or fixture was changed.

### State

Changed this round: `hierarchos-vulkan/shaders/transformer_moe_expert_forward.comp`
and two new shaders (`..._forward_muladd.comp`, `..._expert_count.comp`) with
their `.spv`, `hierarchos-vulkan/src/transformer.rs` (two kernel slots, count
buffer plumbing, paired dispatch), `VENDOR_TUNING.md`, `COMPATIBILITY.md`, and
two validation scripts (`diagnose_minimax_m2_moe_row_count.py` refreshed for the
new baseline and its conclusion now derived from the measurement, new
`diagnose_minimax_m2_moe_expert_topology.py`). The minimax_m2 MoE residual is
closed: the layer-0 residual that Round 6 attributed to the expert GEMM row count
is now exactly zero, both layers agree with the reference's per-expert topology
bit-for-bit, every per-`M` topology was verified on its own probe, and the full
32-family three-stage matrix is green with no gate relaxed.
