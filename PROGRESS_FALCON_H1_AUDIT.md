# Falcon H1 / Falcon H1R Native Parity Audit

## 2026-09-21 final qualification and user frontends

### Final status: deterministic native fixture qualification PASS

All requested fixture gates now pass on AMD Radeon Graphics against the local
Transformers oracle. Parallel attention+Mamba2, real recurrent hybrid caching,
native backward, both two-step AdamW trajectories, trained save/reload and
CLI/GUI full-parameter training/fine-tuning are exercised. Production remains
Rust/Vulkan. This status is scoped to the tested fixtures; full H1R-7B numerical
qualification remains outside the verified scope. Falcon H1 LoRA/PEFT was later
qualified separately (see PROGRESS_PEFT_AUDIT.md).

### 2026-09-30 shipping recheck: outer RMSNorm reduction repair

A fresh frontend shipping pass exposed one further deterministic defect:
`cli-trained/model.layers.1.feed_forward.down_proj.weight` reached
`4.470348358154297e-7` against the local Transformers oracle under the
frontend's AdamW `eps=1e-8` schedule, while every dedicated standalone fixture
remained green. Operand-matched replay showed the down-projection GEMM already
diverged upstream: the oracle's outer RMSNorm (`input_layernorm`,
`pre_ff_layernorm`, `final_layernorm`) reduces the FP32 mean-square through an
eight-lane vector topology, whereas Falcon's graph selected the scalar
reduction. That one-ulp norm difference perturbed a near-cancelling gradient,
which the small-epsilon AdamW step amplified past the gate.

Repair: Falcon's three outer norms now select the existing vector CPU-rsqrt
reduction mode in `src/transformer/falcon_h1.rs`. The matched-operand
GEMM replay is exactly `0.0`, the repaired frontend suite
(`verify_falcon_h1_frontends.py --gui`) PASSes training, full fine-tuning,
reload and generation for the CLI and GUI, and all three Falcon PEFT surfaces
were requalified green afterward. No tolerance was changed and no shader
artifact was modified.

The final nonzero-decay frontend test exposed and fixed one additional runtime
bug: layer-1 Mamba `A_log` drift was `2.384185791015625e-07`, above the gate.
Added `HIERARCHOS_UNFUSED_ADAMW` staging in `shaders/adamw.comp`, compiled
`falcon_h1_adamw.spv`, and selected that kernel only during Falcon attachment
in `src/transformer/falcon_h1.rs`. Materializing FP32 decay/update intermediates
preserves the non-fused PyTorch optimizer rounding. Layer-1 `A_log` drift after
two CLI steps is now `3.637978807e-12`; tolerance was never changed.

Final optimizer measurements after that fix:

| Variant | Ordinary step 1 | Ordinary step 2 | Dual step 1 | Dual step 2 |
| --- | ---: | ---: | ---: | ---: |
| gated_after | 3.725290298e-09 | 1.192092896e-07 | 5.215406418e-08 | 5.215406418e-08 |
| gated_before | 9.371433407e-09 | 5.960464478e-08 | 6.343543646e-08 | 1.146945579e-07 |
| silu | 3.725290298e-09 | 5.960464478e-08 | 2.421438694e-08 | 2.607703209e-08 |

Forward and all 75 cached-token measurements remain as tabulated below.
Final ordinary-trained reload worst is `8.940696716308594e-08`; trained
re-exported generation worst is `4.470348358154297e-08`; re-exported parameters
remain exact. The earlier results below are retained as pre-AdamW-staging
history and are superseded by this final table where different.

`python hierarchos-vulkan/validation/verify_falcon_h1_frontends.py --gui` PASS:

- CLI two-epoch training: parameter drift `1.0244548320770264e-07`, worst layer-1 Mamba in-projection weight.
- CLI reload then two-epoch full fine-tuning with a fresh optimizer: cumulative drift `1.6205012798309326e-07`, worst layer-0 FFN up-projection weight. Nonzero weight decay `0.01`, epsilon `1e-8`, LR `0.0002` with native decay/no-decay groups.
- GUI native command-builder integration: 1 passed, 0 failed; training/fine-tuning exports bit-exact (`0.0`) against the corresponding CLI runs; generation succeeds from the saved package.
- CLI/GUI trained reload logits `[5.960464477539063e-08, 2.9802322387695312e-08]`; re-export cached generation per-token drifts are `[2.1420419216156006e-08, 3.725290298461914e-08, 4.470348358154297e-08, 2.9802322387695312e-08, 2.9802322387695312e-08]` after training and `[7.450580596923828e-08, 2.9802322387695312e-08, 2.9802322387695312e-08, 2.9802322387695312e-08, 4.470348358154297e-08]` after fine-tuning. Exact results saved to `.falcon-h1-fixtures/frontend-report.json`.

Final rebuild/check commands (all PASS):

- `glslc -DHIERARCHOS_UNFUSED_ADAMW=1 hierarchos-vulkan/shaders/adamw.comp -o hierarchos-vulkan/shaders/falcon_h1_adamw.spv`
- `cargo build --manifest-path hierarchos-native-cli/Cargo.toml -j 1`
- `cargo build --manifest-path hierarchos-vulkan/Cargo.toml --bin transformer_parity --bin transformer_generation_parity --bin hierarchos-vulkan-transformer-logits -j 1`
- `python hierarchos-vulkan/validation/verify_falcon_h1.py --generation --training --keep-going` including both ordinary per-step comparisons.
- `cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib -j 1`: 660 passed, 8 ignored, no failures.
- `cargo test --manifest-path hierarchos-native-cli/Cargo.toml --lib -j 1`: 28 passed.
- `cargo test --manifest-path hierarchos-gui/Cargo.toml -j 1`: 7 passed; the one opt-in test is separately PASS via the frontend validator.
- `python hierarchos-vulkan/validation/verify_hf_logits.py --headline-strict` and `python hierarchos-vulkan/validation/verify_hf_training.py --headline-strict`: both rerun after the optimizer fix, exit 0.
- `hierarchos-native-cli/target/debug/hierarchos-native-cli.exe transformer-finetune --help` documents full fine-tuning; `architectures --json` includes `falcon_h1`.
- `python hierarchos-vulkan/validation/generate_supported_architectures.py --check`: current, 146 canonical + 83 aliases.

No remaining blocker for the qualified full-parameter fixture path. Next optional
scope is production-size H1R-7B measurement on adequately provisioned hardware
or a separately implemented/qualified native PEFT path. The documented CLI/GUI
full-parameter workflow has no runtime Python dependency.

### Source and handoff verification

- Read this audit first; searched the memory registry (no Falcon hits) and preserved the dirty checkout. Read the applicable `C:/Users/User/Downloads/AGENTS.md` and current local Falcon H1 mixer/config source.
- Rechecked official config and index with `Invoke-RestMethod https://huggingface.co/tiiuae/Falcon-H1R-7B/resolve/main/config.json` and the analogous `model.safetensors.index.json` URL. Config matches `.falcon_h1r_config.json`; index contains 751 tensors and `total_size=15171297472`. Full 7B weights were not downloaded or run.
- The local oracle masks Mamba input and post-convolution states for multi-token sequences, then chooses chunk scan for prefill and selective state update for cached single tokens. Both branches run in every decoder layer. The native config/loader uses `falcon_h1`, not a Falcon/Mamba2 alias.
- Rebuilt probes and verified the prior Falcon-only SLEEF CE fix. Handoff reports isolated CE backward `0.0` and bit-exact isolated eight-lane RMS backward; those isolated diagnostics were not rerun here. Fresh end-to-end dual AdamW measurements below independently confirm the optimizer blocker is resolved. The handoff added `fp32_sleef_exp.glsl`, Falcon CE SPIR-V and CE/RMS diagnostic scripts; these were already present at session start.

### Fresh strict results (absolute-only `2e-7`)

- Three variants (`gated_after`, biased `gated_before`, `silu`) and 27 forward fixtures PASS; worst logit drift `8.940696716308594e-08`. Sequence lengths 1/3/4/7/9 include nonaligned chunks, padding, nonstandard GQA head width, RoPE, non-unit embedding/head/attention/SSM/MuP/MLP scales and optional gated norm/orderings.
- Native cache/full-prefix drift is **0.0 at every checked token**. Cached versus Transformers worst `7.450580596923828e-08` across five prefill lengths and five generated tokens each (75 checked token logits across variants). Actual recurrence is now required by the harness: eight `batch=1, seq=1` dispatches per cached run (two layers, four decode steps), zero in uncached runs. This supersedes the invalid older cache-only-by-label evidence.
- Ordinary two-step AdamW max parameter drifts: gated-after `1.1920928955078125e-07`; gated-before `1.1920928955078125e-07`; SiLU `5.960464477539063e-08`.
- Dual AdamW per-step drift: gated-after `[5.21540641784668e-08, 5.21540641784668e-08]`, worst `model.layers.0.mamba.out_proj.weight`; gated-before `[6.343907443806529e-08, 1.1987503967247903e-07]`, worst `model.layers.1.self_attn.k_proj.bias`; SiLU `[2.421438694000244e-08, 2.60770320892334e-08]`, worst `model.layers.1.self_attn.k_proj.weight`.
- Trained reload logits worst `4.470348358154297e-08`; re-exported tensors bit-exact; trained re-export recurrent generation worst `5.960464477539063e-08`.
- Full precision per-token data and optimizer reports are in `hierarchos-vulkan/.falcon-h1-fixtures/report.json`, with stdout in `qualification.log`. No failing logit tensor was observed in this continuation; first optimizer mismatch in the frontend harness is explained below.

Per-token cached/HF drift (values in units of `1e-8`, columns are five checked tokens):

| Variant | Prefill | Per-token drift |
| --- | ---: | --- |
| gated_after | 1 | 2.980232239, 3.725290298, 4.470348358, 4.470348358, 5.960464478 |
| gated_after | 3 | 2.980232239, 4.470348358, 2.048909664, 4.470348358, 2.980232239 |
| gated_after | 4 | 1.490116119, 2.607703209, 1.490116119, 7.450580597, 2.980232239 |
| gated_after | 7 | 1.490116119, 7.450580597, 2.235174179, 1.490116119, 3.725290298 |
| gated_after | 9 | 1.490116119, 4.470348358, 4.470348358, 4.470348358, 1.490116119 |
| gated_before | 1 | 2.980232239, 2.980232239, 2.235174179, 4.470348358, 4.470348358 |
| gated_before | 3 | 2.607703209, 2.980232239, 2.235174179, 2.980232239, 2.235174179 |
| gated_before | 4 | 1.490116119, 4.470348358, 4.470348358, 2.235174179, 2.980232239 |
| gated_before | 7 | 0.7450580597, 2.980232239, 4.470348358, 2.235174179, 2.235174179 |
| gated_before | 9 | 1.490116119, 2.980232239, 2.980232239, 2.980232239, 2.235174179 |
| silu | 1 | 4.470348358, 4.842877388, 1.769512892, 2.980232239, 2.235174179 |
| silu | 3 | 3.725290298, 2.980232239, 2.980232239, 3.725290298, 4.470348358 |
| silu | 4 | 1.676380634, 2.980232239, 5.960464478, 4.470348358, 2.980232239 |
| silu | 7 | 1.490116119, 2.980232239, 2.235174179, 2.980232239, 2.980232239 |
| silu | 9 | 2.980232239, 2.980232239, 1.490116119, 2.235174179, 2.980232239 |

### Frontend work and failures/root causes

- Added explicit `transformer-finetune --full-finetune`, rejecting combinations with adapter options. Added GUI `Transformer full-parameter fine-tuning`; existing LoRA default remains explicit and Falcon PEFT still fails closed. Full-parameter fine-tuning starts a fresh optimizer on loaded weights.
- Actual CLI training initially overflowed the Windows default main-thread stack (`3221225725`) before any tensor comparison. Fixed `main_entry` to run on a named 16 MiB worker stack, matching parity probes. This also fixes the GUI-launched CLI process.
- New frontend validator initially imported the global old SafeTensors before the validation dependency path was installed, causing unsupported `safe_open(backend=...)`; fixed import ordering to use the same validation dependencies as the existing harness.
- First frontend parameter mismatch: layer-0 `input_layernorm.weight`, `4.0531158447265625e-06`. Root cause was unequal optimizer groups, not forward/backward math: the existing native optimizer excludes outer RMSNorm/linear biases from decay; the initial frontend oracle decayed all parameters at 0.01. Matched oracle groups to the native `TrainableParameter.decay` flags, retaining nonzero decay and the strict `2e-7` gate. Falcon packed Mamba parameters retain their existing decay=true semantics.
- New `verify_falcon_h1_frontends.py` runs CLI training, full fine-tuning, export/reload and generation; the opt-in GUI test invokes the same argument builder/native subprocess as the Start action. GUI-rendering interaction is not claimed by that integration test.
- README now documents Hub/local H1R training, fine-tuning, saved-package generation, dataset formats, GUI steps, memory/qualification limits and the unsupported LoRA distinction. Architecture inventory regenerated: 146 canonical + 83 aliases = 229 spellings.

### Commands and regression results

- `cargo build --manifest-path hierarchos-vulkan/Cargo.toml --bin transformer_parity --bin transformer_generation_parity --bin hierarchos-vulkan-transformer-logits -j 1` PASS.
- `python hierarchos-vulkan/validation/verify_falcon_h1.py --generation --training --keep-going` PASS, first with base generation and again after adding dispatch assertions and expanded prefill lengths. Added an explicit ordinary step-1 check for the final rerun.
- `$env:HIERARCHOS_FALCON_H1_TRACE_CACHE='1'; & hierarchos-vulkan/target/debug/transformer_generation_parity.exe --model hierarchos-vulkan/.falcon-h1-fixtures/gated_after --fixture hierarchos-vulkan/.falcon-h1-fixtures/gated_after/generation.json --output hierarchos-vulkan/.falcon-h1-fixtures/cache-dispatch.json --use-cache true` PASS; eight recurrent dispatches.
- `cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib -j 1` PASS: 660 passed, 0 failed, 8 ignored.
- `python hierarchos-vulkan/validation/verify_hf_logits.py --headline-strict` PASS (existing family-specific gates, not a new universal `2e-7` forward claim).
- `python hierarchos-vulkan/validation/verify_hf_training.py --headline-strict` PASS.
- `cargo test --manifest-path hierarchos-native-cli/Cargo.toml --lib -j 1` PASS: 28 passed.
- `cargo test --manifest-path hierarchos-inference/Cargo.toml --lib -j 1` PASS: 12 passed.
- `cargo test --manifest-path hierarchos-gui/Cargo.toml -j 1` PASS: 7 passed, 1 explicit integration test ignored pending its fixture run.
- `python -m unittest discover -s hierarchos-vulkan/validation -p 'test_*.py'` PASS: 4 tests, 1 skipped.
- `cargo build --manifest-path hierarchos-native-cli/Cargo.toml -j 1`; `cargo build --manifest-path hierarchos-gui/Cargo.toml -j 1` PASS.
- `python hierarchos-vulkan/validation/generate_supported_architectures.py` regenerated registry docs; `git diff --check` PASS (CRLF conversion warnings only).
- `python hierarchos-vulkan/validation/verify_falcon_h1_frontends.py --gui` rerun after the two harness/stack corrections above; final result to be appended.

Files changed in this session: this audit; root README; CLI lib.rs/README; GUI main.rs/README; COMPATIBILITY.md; README_ARCHITECTURES.md and its generator; verify_falcon_h1.py; new verify_falcon_h1_frontends.py; AdamW shader source/new Falcon SPIR-V and Falcon kernel selection. Validators clear stale PASS reports at run start. Existing native Falcon/RMS/CE runtime work was preserved and exercised.

Next task before closing qualification: confirm final frontend integration and ordinary AdamW step-1 results; append their exact evidence. No full-7B or PEFT green claim.

## 2026-09-21 continuation: recovered prior-session measurements

- These measurements are supplied by the continuation handoff, pending rerun here: forward worst `5.960464478e-08`; cache worst `4.470348358e-08`, per-token `[2.98e-08, 2.98e-08, 1.676e-08, 4.470e-08, 4.470e-08]`; ordinary two-step training `1.192092896e-07`; strict dual AdamW step 1 **FAIL** `8.791685104e-07` at `model.layers.0.mamba.out_proj.weight`.
- Prior diagnostic: embeddings and all seven layer-0 RMS variance/mean rows exact; first divergent tensor `ln1_rstd` (rows 3/6 +1 FP32 ULP, row 5 -1 ULP), causing norm-output drift `2.384185791e-07`. Live unfused-products and Falcon rsqrt flags both 1. Installed PyTorch AVX2/AVX512 rsqrt uses rounded FP32 `1 / sqrt(x)`.
- Prior downstream diagnostic: in_proj `5.960e-08`, scan `4.657e-10`, gated input `1.118e-08`, out-proj grad-output `4.470e-08`, out-proj weight gradient `2.793967724e-09`. Cancellation leaves a gradient near `7.47e-09`, magnifying error through AdamW epsilon `1e-8`.
- Prior source-only adjacent-candidate rsqrt experiment was unverified: compiler invocations were rejected and SPIR-V remained stale. Exact prior commands were not provided in the handoff; no reconstructed command is claimed as run.
- Current continuation starts with audit read, `git status --short`, focused file discovery, and memory registry search (no Falcon hits). Existing dirty files preserved. Next: compile the pending RMS candidate, rebuild, refresh traces, then run strict dual AdamW before broader checks. Status remains **not green**.

### Live continuation findings and verification

- Compiler is working now. Compiling the prior floating residual candidate worsened ln1 drift to `4.768371582e-07`; rejected that implementation. Added `shaders/fp32_sqrt_recip.glsl`: native 32-bit integer extended products select correctly rounded FP32 sqrt and reciprocal separately, including ties-to-even. No Python/host math in production. Layer-0 embedding, variance, rstd, norm output and Mamba in-projection are now exact against the single-threaded oracle on the failing fixture.
- Important diagnostic correction: earlier `*_tmp.py` diagnostics used default PyTorch threads, unlike the strict harness. With `torch.set_num_threads(1)`, in_proj is exact; a supposed `5.960e-08` projection error was from mismatched validation settings. Updated the RMS/AdamW/cancellation diagnostic scripts and added ordered linear, convolution and backward diagnostics.
- Convolution isolated: PyTorch preactivation exactly matches sequential FMA; native previously used separate multiply/add. Fixed forward and backward recomputation to FMA; changed SiLU to division matching CPU SiLU staging; reused exact sqrt/div for Mamba gated norm. RMS backward now has a Falcon-only branch following the rsqrt autograd chain with materialized FP32 products.
- `cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib falcon_h1_rms_sqrt_div_rounding_matches_ieee -j 1 -- --nocapture` initially found two pre-existing integration omissions: test cache constructors lacked `falcon_h1_state`. Added `None` in both. Rerun PASS: 1 passed / 667 filtered; 2,176 normal-variance rows with varied mantissas/exponents and power-of-two boundaries matched host IEEE sqrt/div bitwise.
- `python hierarchos-vulkan/validation/verify_falcon_h1.py --generation --training --keep-going`: 27 forward fixtures PASS, worst `8.940696716e-08`; ordinary two-step AdamW all three variants PASS at `1.192092896e-07`; trained reload/reexport/generation all three PASS, worst `5.960464478e-08`, reexport tensors exact. Dual gated-before steps `[3.741297405e-08, 5.129186320e-08]` PASS; dual silu `[1.490116119e-08, 1.676380634e-08]` PASS; dual gated-after `[8.456408978e-07, 8.456408978e-07]` FAIL at layer-0 Mamba out_proj.weight. Report persisted in `.falcon-h1-fixtures/report.json`; harness exits nonzero and records both dual steps.
- **Cache evidence correction:** structured `generate_output` previously ran full-prefix math even with `use_cache=true`, so all earlier Vulkan cached/full-prefix zero drift results do NOT establish recurrent execution. Found and patched Falcon structured logits to use real hybrid-cache prefill and selective decode. Actual cache validation is pending the next rebuild/run. This invalidates the cached-generation PASS claim from the handoff as evidence of recurrence.
- Current backward trace before CE staging fix: logits `2.980232239e-08`, CE gradient `1.490116119e-08` even on identical logits, final-norm output gradient `1.862645149e-09`, layer-0 Mamba output gradient `4.470348358e-08`. Added a Falcon-only CE shader variant that follows PyTorch NLL(LogSoftmax) instead of algebraically collapsing its derivative. Validation pending.
- Commands run (repeated after shader changes): `glslc hierarchos-vulkan/shaders/transformer_rms_norm_forward.comp -o hierarchos-vulkan/shaders/transformer_rms_norm_forward.spv`; `glslc hierarchos-vulkan/shaders/falcon_h1.comp -o hierarchos-vulkan/shaders/falcon_h1.spv`; `glslc hierarchos-vulkan/shaders/transformer_rms_norm_input_grad.comp -o hierarchos-vulkan/shaders/transformer_rms_norm_input_grad.spv`; `glslc -DHIERARCHOS_LOG_SOFTMAX_GRAD=1 hierarchos-vulkan/shaders/transformer_cross_entropy.comp -o hierarchos-vulkan/shaders/falcon_h1_cross_entropy.spv`; `cargo build --manifest-path hierarchos-vulkan/Cargo.toml --bin falcon_h1_backward_trace_tmp --bin transformer_parity --bin transformer_generation_parity --bin hierarchos-vulkan-transformer-logits -j 1`; `python hierarchos-vulkan/validation/falcon_h1_refresh_trace_tmp.py`; `python hierarchos-vulkan/validation/falcon_h1_rms_diag_tmp.py`; `python hierarchos-vulkan/validation/falcon_h1_adamw_diag_tmp.py`; `python hierarchos-vulkan/validation/falcon_h1_cancel_diag_tmp.py`; `python hierarchos-vulkan/validation/falcon_h1_linear_diag.py`; `python hierarchos-vulkan/validation/falcon_h1_conv_diag.py`; `python hierarchos-vulkan/validation/falcon_h1_backward_diag.py`.
- Focused dual command: `python -c "import sys;sys.path.insert(0,'hierarchos-vulkan/validation');import verify_falcon_h1 as v;v.torch.set_num_threads(1);v.dual_adamw(v.tiny(),v.ROOT/'.falcon-h1-fixtures/dual/gated_after')"`.
- Files changed this continuation: audit; the RMS forward/backward GLSL/SPIR-V, new FP32 helper, Falcon GLSL/SPIR-V, CE GLSL/new Falcon SPIR-V, transformer.rs (Falcon backward/cache/CE routing and test cache constructors), falcon_h1.rs (rounding test and traces), strict harness and diagnostic scripts. Prior unrelated dirty work preserved. No green claim.

## 2026-09-21 (implementation in progress)

- Read this audit first. Preserved all pre-existing dirty files and generated caches.
- Baseline `cargo check --manifest-path hierarchos-vulkan/Cargo.toml --lib -j 1` failed with five non-exhaustive `FalconH1` match arms. These have been addressed; a subsequent library check passed.
- Official config fetched again with `curl.exe -L --fail --silent --show-error https://huggingface.co/tiiuae/Falcon-H1R-7B/resolve/main/config.json`. Official index fetched with `Invoke-RestMethod https://huggingface.co/tiiuae/Falcon-H1R-7B/resolve/main/model.safetensors.index.json`; total checkpoint size is 15,171,297,472 bytes. No full 7B weights downloaded.
- Source: current local `configuration_falcon_h1.py` and `modeling_falcon_h1.py`, including chunk SSD contraction, selective recurrence, gated normalization, parallel layer and canonical state-dict names. Attention head width is independent of hidden_size / num_heads. The HF MLP is `feed_forward`, final norm is `final_layernorm`, and the second block norm is `pre_ff_layernorm`.
- Added `hierarchos-vulkan/src/transformer/falcon_h1.rs` and `shaders/falcon_h1.comp`/`.spv`: config-driven Mamba geometry/MuP scales, native convolution, chunk SSD and recurrent state update, gated RMSNorm orderings, analytic backward and AdamW orchestration. Integrated parallel branch, attention/MLP multipliers, package load/export and combined KV/conv/SSM cache in `src/transformer.rs`.
- Added Falcon-specific finite additive attention-mask semantics to attention shaders (causal mode 2; other architectures keep prior behavior). Added diagnostic readback and the `HIERARCHOS_FALCON_H1_TRACE` option to the logit probe.
- Added `validation/verify_falcon_h1.py`: strict fixed `2e-7`, three tiny norm/gate variants, GQA with nonstandard head width, non-unit scales, chunk-aligned/nonaligned lengths, padding, cached/full-prefix generation, and two-step AdamW checks. This is an executable validation harness, not yet a passing claim.
- Commands run: `glslc hierarchos-vulkan/shaders/falcon_h1.comp -o hierarchos-vulkan/shaders/falcon_h1.spv`; analogous compilation of `transformer_attention_forward.comp` and `transformer_attention_backward.comp`; `cargo build --manifest-path hierarchos-vulkan/Cargo.toml --bin hierarchos-vulkan-transformer-logits --bin transformer_parity --bin transformer_generation_parity -j 1`; `python hierarchos-vulkan/validation/verify_falcon_h1.py --generation --training`.
- Early validation failures were loader/config errors, before any tensor comparison: independent head width rejected; missing `mlp.gate_proj.weight` (canonical name `feed_forward.gate_proj.weight`); missing `model.norm.weight` (canonical name `model.final_layernorm.weight`). Fixed those causes.
- Still **not green**. Numerical results, first divergent tensor, optimizer drift, save/reload and regression results will be recorded below after execution. PEFT is explicitly unsupported; ordinary full-parameter training is the requested path.

## 2026-09-20

### Status

- `falcon_h1` is **not green**. This audit was absent at session start and has been created for the native implementation work.
- Required parity gate remains `max_abs <= 2e-7` for deterministic logits, every checked cached token, gradients/optimizer state as applicable, and two-step AdamW parameter drift.

### Source findings

- Authoritative local oracle: `C:\Users\User\transformers\src\transformers\models\falcon_h1\`.
- `FalconH1DecoderLayer` is a true parallel block: one `input_layernorm` feeds both `FalconH1Mixer` (Mamba2) and GQA/RoPE attention, their independently scaled outputs are summed with the original residual, then `pre_ff_layernorm` + gated MLP is added residually.
- Every Falcon H1 layer is `hybrid`; this is not an alternating attention/Mamba schedule.
- Attention applies `key_multiplier` to K before RoPE, `attention_in_multiplier` to the normalized branch input, and `attention_out_multiplier` after `o_proj`.
- Mamba2 applies the 5-way MuP vector to packed `[z, x, B, C, dt]` projection slices, uses fp32 `A=-exp(A_log)`, softplus `dt+dt_bias`, grouped B/C state expansion, D skip, optional gated RMSNorm, causal depthwise convolution, and recurrent SSM state.
- Existing Qwen/OLMo Gated DeltaNet recurrence in this repository is mathematically different from Mamba2 and will not be used as a compatibility alias.
- Official `tiiuae/Falcon-H1R-7B` config currently reports hidden size 3072, 44 layers, 12 attention heads, 2 KV heads, head dim 128, Mamba intermediate 3072, 24 Mamba heads, state 256, conv kernel 4, chunk size 128, gated RMSNorm enabled, and non-unit embedding/LM-head/attention/SSM/MLP/MuP multipliers. All implementation parameters remain config-driven.

### Worktree safety

- The repository was already dirty at session start, including pre-existing edits to `hierarchos-vulkan/src/transformer.rs`, README/compatibility files, shaders, validation files, generated caches, and target directories. Those changes are treated as user-owned and are being preserved.

### Commands run

- `git status --short`
- `rg -n -i "falcon.?h1|h1r|mamba2|hybrid cache" C:\Users\User\.codex\memories\MEMORY.md`
- `rg --files C:\Users\User\transformers\src\transformers\models\falcon_h1`
- Targeted `rg`/`Get-Content` reads of `configuration_falcon_h1.py` and `modeling_falcon_h1.py` for attention, RMSNorm, Mamba2, decoder, MuP, recurrent update, and chunk-scan definitions.
- Targeted `rg`/`Get-Content` reads of `hierarchos-vulkan/src/transformer.rs` for architecture/config parsing, transformer layer execution, recurrent-cache plumbing, depthwise-convolution kernels, loader routing, backward, and AdamW hooks.
- `curl.exe -L --fail --silent --show-error https://huggingface.co/tiiuae/Falcon-H1R-7B/resolve/main/config.json`

### First divergent tensor / measured drift

- Not measured yet: there was no native `falcon_h1` execution path at session start, so the first blocker is implementation rather than a numerical comparison.
- Max-logit drift: not yet measured.
- Per-token cache drift: not yet measured.
- Two-step AdamW parameter drift: not yet measured.

### Blockers / next task

- Implement a dedicated native Falcon H1 Mamba2 module and `falcon_h1` architecture/config/loader integration without approximating it as Falcon or Gated DeltaNet.
- Wire the parallel branch, multipliers, hybrid cache, backward/AdamW, export/reload, then build tiny local Transformers fixtures and debug from the earliest divergent tensor.
