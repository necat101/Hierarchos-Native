# Kimi K3 Native Vulkan Progress Audit

This file is the rolling handoff/source-of-truth for native Kimi K3 text-backbone work in Hierarchos-Native. Update it as implementation and verification progress changes. The checklist and the `K3 text-backbone completion checkpoint — 2026-09-18` section describe the present state; later chronological progress snapshots are retained as historical evidence and may describe limitations that have since been removed.

## Goal

Implement real native `kimi_linear` support and make composite `kimi_k3` packages resolve to that text graph without aliasing K3 to Kimi K2/K2.5, DeepSeek, Qwen, GLM, or another existing architecture.

The production path must remain Rust/Vulkan only. Python, PyTorch, FLA, Triton, CUDA, libtorch, `pyo3`, remote-code execution, or subprocess fallbacks may be used only as optional validation/oracle tooling, never as runtime dependencies.

## Current repository baseline

The working tree already contains the prior eight-family compatibility/parity milestone. Those changes are intentionally preserved while K3 is added.

Existing strict two-step AdamW headline set from that milestone:

- DeepSeek V4
- Phi-4 Multimodal text
- Phi-3
- Kimi K2.5 text
- Mistral 4
- MiniMax M3 text
- Gemma 4
- MiniMax M2

The K3 work should extend this suite once Kimi-Linear parity is real; do not replace the existing eight-family evidence with registry-count claims.

## Authoritative K3 reference confirmed

Primary architectural source:

- `moonshotai/Kimi-K3` on Hugging Face
- `config.json`
- `configuration_kimi_k3.py`
- `modeling_kimi_linear.py`
- `modeling_kimi_k3.py`

The upstream package confirms:

- top-level `model_type = "kimi_k3"`
- top-level architecture `KimiK3ForConditionalGeneration`
- nested text `model_type = "kimi_linear"`
- nested text architecture `KimiLinearForCausalLM`
- text backbone is hybrid KDA + full MLA
- official checkpoint uses compressed-tensors MXFP4-style quantization metadata

Production K3 config details verified directly from the upstream package:

- `hidden_size = 7168`
- `num_hidden_layers = 93`
- `num_attention_heads = 96`
- `num_key_value_heads = 96`
- `q_lora_rank = 1536`
- `kv_lora_rank = 512`
- `qk_nope_head_dim = 128`
- `qk_rope_head_dim = 64`
- `v_head_dim = 128`
- `mla_use_nope = true`
- `mla_use_output_gate = true`
- `linear_attn_config.head_dim = 128`
- `linear_attn_config.num_heads = 96`
- `linear_attn_config.short_conv_kernel_size = 4`
- `linear_attn_config.use_full_rank_gate = true`
- `linear_attn_config.gate_lower_bound = -5.0`
- `attn_res_block_size = 12`
- `hidden_act = "situ"`
- `activation_situ_beta = 4.0`
- `activation_situ_linear_beta = 25.0`
- `first_k_dense_replace = 1`
- `num_experts = 896`
- `num_experts_per_token = 16`
- `num_shared_experts = 2`
- `moe_intermediate_size = 3072`
- `routed_expert_hidden_size = 3584`
- `latent_moe_use_norm = true`
- `moe_router_activation_func = "sigmoid"`
- `moe_renormalize = true`
- `routed_scaling_factor = 1.0`
- `max_position_embeddings = 1048576`

The upstream config provides explicit 1-based `kda_layers` and `full_attn_layers`. Production layout is 69 KDA layers and 24 full-attention layers. Native parsing must validate the explicit lists rather than infer a hard-coded periodic schedule.

## Reuse decisions already validated

### KDA recurrence and cache math

The backend already contains native GLM-5 KDA infrastructure with:

- vector-valued log decay
- Q/K L2 normalization
- decay-before-delta-update ordering
- beta-weighted delta correction
- recurrent state write/read
- recurrent forward kernel
- recurrent backward kernel
- parameter-gradient kernels
- short causal convolution
- cached one-token convolution update
- recurrent generation state
- gated RMSNorm output stage

The shared GLM-5 KDA shaders match the K3 delta-rule ordering closely enough to reuse/generalize the recurrence core instead of writing a second delta-attention engine.

Important limitation: K3 still needs its own checkpoint/projection ABI. Reusing the recurrence kernel does not mean treating K3 as `glm5_next`.

### MLA

K3 full-attention layers use DeepSeek-derived MLA concepts already represented in the native backend:

- Q LoRA
- compressed KV LoRA
- NOPE + RoPE split
- latent KV projection/norm structure

K3-specific MLA difference that must remain explicit:

- `mla_use_output_gate=true`
- attention output is multiplied elementwise by `sigmoid(g_proj(hidden_states))` before `o_proj`

### MoE routing substrate

The existing DeepSeek/Kimi sparse-MoE machinery is the right substrate for routing, shared experts, correction bias, sigmoid scoring, top-k selection, renormalization, and routed scaling.

K3's latent-MoE projection shell is genuinely new and must not be flattened into ordinary DeepSeek/K2.5 expert execution.

## K3-specific primitives confirmed as genuinely new

### Canonical architecture and wrapper

Required:

- new canonical native architecture `kimi_linear`
- `kimi_k3` composite package resolves nested `text_config` to `kimi_linear`
- Kimi K2/K2.5 continue through their existing DeepSeek-style graph
- K3 vision/projector execution must fail closed unless a separately proven native implementation exists

### KDA projection/checkpoint ABI

Upstream `KimiDeltaAttention` uses:

- `self_attn.q_proj.weight`
- `self_attn.k_proj.weight`
- `self_attn.v_proj.weight`
- separate `q_conv1d.weight`, `k_conv1d.weight`, `v_conv1d.weight`
- `self_attn.f_a_proj.weight`
- `self_attn.f_b_proj.weight`
- `self_attn.A_log`
- `self_attn.dt_bias`
- `self_attn.b_proj.weight`
- either full-rank `self_attn.g_proj.weight` or low-rank gate projections
- `self_attn.o_norm.weight`
- `self_attn.o_proj.weight`

The existing GLM-5 path uses a different tensor namespace and one packed convolution tensor, so K3 requires an explicit Kimi projection/checkpoint variant even if it reuses the same recurrence kernels.

### KDA short convolution

K3 has three independent depthwise short convolutions for Q, K, and V, each with SiLU activation. Production kernel size is 4, but implementation must remain generic.

Generation requires three convolution carry states plus recurrent KDA state per KDA layer.

### Full-rank KDA output gate

Production K3 uses `use_full_rank_gate = true`, so KDA output gating comes from one full-rank `g_proj(hidden_states)` rather than GLM-5's low-rank gate chain.

### SiTU

K3 `hidden_act = "situ"` semantics confirmed upstream:

`situ_gate = beta * tanh(gate / beta) * sigmoid(gate)`

If `linear_beta` is configured:

`situ_up = linear_beta * tanh(up / linear_beta)`

Final GLU output:

`situ_gate * situ_up`

Forward and backward must be native. Production values are beta 4.0 and linear_beta 25.0, but config parsing/execution must remain generic.

### Stable LatentMoE

K3 sparse MoE adds:

- `routed_expert_down_proj`: hidden -> latent routed-expert width
- optional `routed_expert_norm`
- routed experts operate in the latent width
- `routed_expert_up_proj`: latent -> hidden

Production latent width is 3584. First decoder layer is dense because `first_k_dense_replace = 1`; later configured layers are sparse.

### Attention Residuals / AttnRes

This is not an ordinary residual add.

Upstream behavior confirmed:

- candidate residual sources are stored block residuals plus the current prefix residual
- candidates are RMS-normalized for scoring
- learned score weight is `norm.weight * proj.weight`
- softmax is applied over residual sources
- output is the learned weighted mixture of candidate residual vectors
- block residuals are appended at layer boundaries governed by `attn_res_block_size`
- per-layer tensors:
  - `self_attention_res_norm.weight`
  - `self_attention_res_proj.weight`
  - `mlp_res_norm.weight`
  - `mlp_res_proj.weight`
- model-level final tensors:
  - `output_attn_res_norm.weight`
  - `output_attn_res_proj.weight`

Forward and backward must be implemented natively.

## Existing native code identified for extension

Primary implementation file:

- `hierarchos-vulkan/src/transformer.rs`

Existing hybrid linear-attention infrastructure:

- `VulkanTransformerConfig::linear_attention_layers`
- `HostQwenGatedDeltaNet`
- `HostQwenGatedDeltaProjection`
- `VulkanQwenGatedDeltaNet`
- native recurrent/short-convolution generation cache fields
- GLM-5 KDA shader set

Current `HostQwenGatedDeltaProjection` variants include:

- `Qwen3Next`
- `Qwen35`
- `OlmoHybrid`
- `Glm5`

K3 should add a separate Kimi projection variant rather than overloading `Glm5`.

The current GLM-5 implementation already allocates vector decay at `rows * key_dim`, uses the GLM-5-specific recurrence kernel, and has separate forward/backward gradient buffers. That is the best shared execution point for Kimi K3 recurrence math.

## Package/loading/export constraints

Canonical K3 package prefix is `language_model` for the text model.

Expected important text tensor families include:

- `language_model.model.embed_tokens.weight`
- `language_model.model.layers.{i}.*`
- `language_model.model.norm.weight`
- `language_model.lm_head.weight`

The loader/exporter must preserve unrelated composite package tensors where existing round-trip infrastructure supports passthrough.

Do not add remote-code execution to the Rust loader.

Do not require the local Transformers checkout at runtime.

## Quantization boundary

Official K3 config declares compressed-tensors MXFP4-style quantization (`mxfp4-pack-quantized`).

Architecture support and official quantized checkpoint support are separate claims.

Until the native compressed-tensor representation is implemented and proven, the official packed checkpoint must fail closed with an explicit diagnostic. Tiny FP32/BF16 Kimi-Linear packages must remain usable for native architecture validation.

## Vision boundary

Current target is K3 text-backbone support only.

Do not claim MoonViT/MoonViT-V2 image execution. The composite loader should preserve non-text tensors on round trip when the repository's passthrough contract supports it and reject native image execution explicitly.

## Historical implementation snapshot (superseded by the completion checkpoint)

The following records an earlier intermediate state, before full graph integration and parity closure. It is retained to preserve implementation history and must not be read as the current support boundary.

Confirmed current implementation in `hierarchos-vulkan/src/transformer.rs`:

- `VulkanTransformerArchitecture::KimiLinear` exists.
- Canonical architecture name serialization returns `kimi_linear`.
- `kimi_k3` is recognized as a text-wrapper model type alongside the existing Kimi K2/K2.5 wrappers.
- Existing hybrid linear-attention infrastructure remains centered on `linear_attention_layers`, `HostQwenGatedDeltaNet`, `HostQwenGatedDeltaProjection`, `VulkanQwenGatedDeltaNet`, and the shared GLM-5 KDA shaders/cache path.
- Distinct host/Vulkan `Kimi` KDA projection variants now exist with canonical Q/K/V, `f_a_proj`/`f_b_proj`, `b_proj`, and full-rank `g_proj` semantics.
- Kimi checkpoint loading consumes canonical `self_attn` names and merges three independent Q/K/V depthwise-convolution tensors into one Q-then-K-then-V channel-contiguous native buffer. Component export deterministically splits that buffer back into the three canonical tensors.
- Kimi KDA forward/backward now reuses the vector-decay GLM-5 recurrence and gated-RMSNorm kernels while routing the full-rank output gate directly through `g_proj`; it does not reuse GLM-5's low-rank gate chain.
- Kimi full-attention loading is now hybrid-schedule aware: only configured non-KDA layers consume MLA tensors, and `mla_use_output_gate=true` loads canonical `self_attn.g_proj.weight` only for those full-attention layers. The gate reuses the existing full-width sigmoid attention-gate Vulkan forward/backward path, which sits between the raw attention result and `o_proj`.
- `KimiLinear` config/schedule parsing and K3 wrapper/text-prefix rebasing are implemented and regression-tested. Top-level full-model Kimi load/export/PEFT enablement remains intentionally incomplete until the remaining K3 graph primitives are present.
- SiTU is now implemented natively in the dense/gated-MLP path with forward/backward Vulkan kernels and finite-difference coverage.
- Stable LatentMoE now has a native hidden -> routed-latent projection, optional latent RMSNorm, routed-expert execution in latent width, and routed-latent -> hidden projection. Focused loader plus forward/backward finite-difference tests pass on the available Vulkan device.
- A standalone native Kimi AttnRes operator now exists with RMS-normalized learned scoring, softmax over residual sources, weighted residual mixing, analytic backward, canonical tensor loading, and focused finite-difference coverage. It is not yet integrated into the full Transformer layer/block-residual lifecycle or final-output path, so full AttnRes support remains incomplete.
- Full mixed-model load/export, graph-integrated AttnRes, generation, and strict external parity closure are still outstanding.

At this historical point, registry recognition alone was not yet sufficient to describe K3 support. The later completion checkpoint records the subsequent execution-path and parity closure.

Known dirty files at the start of K3 work:

- `hierarchos-vulkan/COMPATIBILITY.md`
- `hierarchos-vulkan/shaders/transformer_rms_norm_forward.comp`
- `hierarchos-vulkan/shaders/transformer_rms_norm_forward.spv`
- `hierarchos-vulkan/shaders/transformer_rms_norm_input_grad.comp`
- `hierarchos-vulkan/shaders/transformer_rms_norm_input_grad.spv`
- `hierarchos-vulkan/src/bin/transformer_parity.rs`
- `hierarchos-vulkan/src/transformer.rs`
- `hierarchos-vulkan/validation/verify_hf_logits.py`
- `hierarchos-vulkan/validation/verify_hf_training.py`

Do not discard or overwrite those existing user changes.

## Disk-space state

At the beginning of this continuation, C: had roughly 45.3 GB free.

Previously completed cleanup already removed about 10.3 GiB of Cargo build artifacts. Current `hierarchos-vulkan/dist` copies are small enough that they are not the primary storage pressure.

Further cleanup should remain limited to regenerable build/checkouts/caches after verifying they are not the user's only copy of source or test artifacts.

## Implementation checklist

- [x] Add `KimiLinear` / `kimi_linear` as a canonical native architecture scaffold.
- [x] Add `kimi_k3` wrapper resolution to nested `text_config`. Verified with a tiny nested `kimi_linear` config through `VulkanTransformerConfig::from_hf_value`.
- [x] Keep `kimi_k2` and `kimi_k25` on their existing graph. `kimi_k2` still normalizes to the existing DeepSeek-V3 text graph, the K2.5 wrapper still resolves through it, and the K2.5 forward/two-step AdamW regressions pass.
- [x] Parse and validate 1-based `kda_layers` and `full_attn_layers` as a complete, non-overlapping partition of all layers.
- [x] Parse K3 linear-attention geometry/config generically and consume the validated geometry in the Kimi KDA loader/runtime ABI.
- [x] Add a distinct Kimi KDA projection/checkpoint ABI.
- [x] Support three independent Q/K/V short convolutions with SiLU. Canonical tensors are loaded/exported separately and execute through one mathematically equivalent packed depthwise kernel; cached token-by-token generation now matches the full-prefix mixed graph under the K3 `2e-7` contract.
- [x] Reuse/generalize existing vector-decay KDA recurrent forward/backward kernels for the Kimi projection.
- [x] Support K3 full-rank KDA `g_proj` output gate in native forward/backward.
- [x] Preserve KDA recurrent + convolution state in generation cache.
- [x] Add K3 MLA output gate before `o_proj`.
- [x] Implement SiTU forward/backward.
- [x] Implement latent-MoE down/norm/up shell around routed experts.
- [x] Implement AttnRes forward/backward and final output AttnRes, including the per-layer block-residual lifecycle and the model-level final mixer before terminal normalization.
- [x] Map canonical SafeTensors names for load/export across the full mixed K3 graph.
- [x] Preserve/pass through unrelated K3 composite tensors where supported.
- [x] Fail closed on K3 vision execution. Direct MoonViT config execution is rejected by the text runtime; the composite package remains text-backbone-only while untouched vision/projector tensors are preserved on export.
- [x] Fail closed on unsupported MXFP4 compressed-tensors packages.
- [x] Add tiny deterministic Kimi-Linear fixture generation through the Moonshot/Transformers oracle helper.
- [x] Add KDA-only parity coverage. Current external Moonshot/HF oracle max absolute logit error: `5.587935448e-8`.
- [x] Add MLA-only parity coverage. Current external Moonshot/HF oracle max absolute logit error: `6.146728992e-8`.
- [x] Add mixed KDA/MLA schedule parity coverage. Current mixed text max absolute logit error: `5.215406418e-8`.
- [x] Add AttnRes boundary coverage in the full mixed decoder graph, including final output AttnRes.
- [x] Add dense-first-layer + latent-MoE coverage. The mixed fixture uses dense layer 0 followed by sparse Stable LatentMoE layers and participates in full forward/backward/AdamW parity.
- [x] Add SiTU forward/backward/finite-difference coverage.
- [x] Add cached token-by-token vs full-prefix generation parity at `<= 2e-7` logits/hidden-state tolerance.
- [x] Add save/reload round-trip coverage for the full mixed Kimi graph.
- [x] Add at least one native optimizer-step training test; the focused test proves KDA and AttnRes parameters both update.
- [x] Add Kimi K3/Kimi-Linear to the deterministic forward parity suite.
- [x] Add Kimi K3/Kimi-Linear to the strict two-step AdamW suite under the same `< 2e-7` contract. Current max absolute parameter error: `5.963374861e-8` across 18,450 values.
- [x] Re-run K2/K2.5 regressions. K2/K2.5 graph-normalization tests pass, K2.5 forward parity remains green, and its two-step AdamW max absolute parameter error is `1.192092896e-7`.
- [x] Regenerate architecture inventory/matrix: `145 canonical + 83 aliases = 228 advertised model types`.
- [x] Update `COMPATIBILITY.md`/README to distinguish K3 text-backbone support from unsupported MoonViT/vision execution and unsupported official MXFP4 compressed-tensors packages.

## Required final verification

Before calling K3 complete, run and record actual results for:

- `cargo check --manifest-path hierarchos-vulkan/Cargo.toml --all-targets`
- `cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib`
- native CLI tests
- GUI tests
- architecture-generation check
- Python validation suite
- Kimi-Linear forward parity
- Kimi-Linear backward/optimizer parity
- generation cache parity
- save/reload parity
- `cargo fmt --check`
- `git diff --check`

## K3 text-backbone completion checkpoint — 2026-09-18

The native Kimi K3 / KimiLinear text-backbone implementation is now functionally closed against the checklist above. The production boundary remains intentional: MoonViT/vision/projector execution is not implemented, and the official `compressed-tensors` MXFP4 packed checkpoint representation is rejected until separately implemented and validated.

Final verification observed in this worktree:

- `cargo check --manifest-path hierarchos-vulkan/Cargo.toml --all-targets` -> passed.
- `cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib` -> `654 passed, 0 failed, 8 ignored`.
- `cargo test --manifest-path hierarchos-native-cli/Cargo.toml` -> `27 passed, 0 failed`.
- `cargo test --manifest-path hierarchos-gui/Cargo.toml` -> `6 passed, 0 failed`.
- `cargo test --manifest-path hierarchos-inference/Cargo.toml` -> `12 passed, 0 failed`.
- Python validation unit suite -> `4 tests`, `OK`, `1 skipped`.
- Architecture generation -> current at `145 canonical + 83 aliases`; coverage audit against `C:\Users\User\transformers` completed successfully.
- Nine-family headline forward suite -> passed. K3 mixed text max absolute logit error `5.215406418e-8` at `atol=2e-7, rtol=0`.
- K3 focused forward oracle -> KDA `5.587935448e-8`, MLA `6.146728992e-8`, mixed `5.215406418e-8`; all zero failing values at `2e-7` absolute tolerance.
- Nine-family strict two-step AdamW suite -> passed. K3 max absolute parameter error `5.963374861e-8`; K2.5 remains `1.192092896e-7`.
- Mixed cached generation, full checkpoint export/reload, K3 MXFP4 fail-closed, MoonViT fail-closed, and auxiliary-tensor passthrough regressions -> passed.
- `cargo fmt --check` remains unavailable as a reliable gate for the enormous `transformer.rs`: repeated earlier runs aborted after rustfmt attempted multi-gigabyte allocations. Do not record this gate as passed unless rustfmt itself completes in a future environment.

## Handoff protocol for this task

`PROGRESS_K3_AUDIT.md` is the canonical persistent handoff record for native Kimi K3 work.

Every future continuation of this task should:

1. Read this audit before relying on harness-generated context summaries or compaction output.
2. Treat the audit as the source of truth for completed work, open work, verification evidence, compatibility boundaries, and immediate next steps.
3. Update the audit after every meaningful implementation or verification batch, including failures that change the next-step plan.
4. Record only checks that actually ran; do not mark work complete from registry recognition, compile intent, or unverified assumptions.
5. Preserve the prior eight-family parity milestone and other dirty user changes unless the task explicitly requires modifying them.
6. Before ending a continuation, add a concise progress report containing the files changed, tests/checks run, numerical results where relevant, remaining blockers, and the next concrete implementation step.

This audit-based handoff protocol should be preferred over the harness's unreliable automatic context-compaction path for this repository task.

## Progress report — 2026-09-17 continuation

Current continuation findings:

- Recovered the task after harness context compaction and re-established the repository state from the live worktree.
- Confirmed the dirty tree still contains the prior eight-family compatibility/parity work and must not be discarded.
- Confirmed `KimiLinear` and `kimi_k3` recognition scaffolding are already present in `transformer.rs`.
- Confirmed the existing GLM-5 KDA implementation is still the most appropriate recurrence/backward/cache substrate to generalize, but K3 requires its own tensor ABI because upstream uses separate Q/K/V convolutions, `f_a_proj`/`f_b_proj`, full-rank `g_proj`, and K3-specific canonical names.
- Confirmed no Kimi-specific projection variant exists yet in `HostQwenGatedDeltaProjection` / `VulkanQwenGatedDeltaProjection`.
- Began a fresh `cargo check --manifest-path hierarchos-vulkan/Cargo.toml --lib` to establish the current compile boundary before further K3 edits; record the final result here once it completes.

Compile-boundary result:

- `cargo check --manifest-path hierarchos-vulkan/Cargo.toml --lib` currently fails with six Rust `E0004` non-exhaustive-match errors caused by the newly added `KimiLinear` enum variant.
- Missing match coverage is currently at `default_lora_target_module`, model host-weight loading, PEFT/module routing, PEFT export routing, and top-level parameter tensor enumeration paths in `transformer.rs`.
- This is a scaffolding integration failure, not a Vulkan/math failure yet. The next edit batch must restore exhaustive matching without routing K3 through an existing incompatible graph.

Next concrete step: finish the compile-boundary audit, then implement Kimi-Linear config parsing/schedule validation and the distinct Kimi KDA projection/checkpoint variant before touching the higher-level SiTU, latent-MoE, and AttnRes layers.

## Progress report — 2026-09-17 latest continuation

Newest verified findings before context compaction:

- Re-ran `cargo check --manifest-path hierarchos-vulkan/Cargo.toml --lib` against the live worktree. It still fails with exactly six Rust `E0004` non-exhaustive-match errors caused by `VulkanTransformerArchitecture::KimiLinear`; no new Vulkan/math compile errors are visible yet.
- The six current enum-integration gaps are at:
  - `default_lora_target_module`
  - top-level host-weight loading in `from_hf_package`
  - PEFT/module routing
  - PEFT export routing
  - parameter-module enumeration
  - top-level parameter tensor enumeration
- These must be wired fail-closed or through a genuine Kimi-Linear path. Do not silence the errors by aliasing `KimiLinear` to DeepSeek/Kimi K2/K2.5/GLM/Qwen.
- `from_hf_value` already has the generic nested-`text_config` wrapper fallback, and `kimi_k3` is present in the wrapper-recognition list. The remaining K3 config work is to add a real `kimi_linear` parser/normalizer and verify that `kimi_k3.text_config.model_type = "kimi_linear"` resolves through it while K2/K2.5 retain their current paths.
- Existing hybrid linear-attention scheduling is represented by `VulkanTransformerConfig::linear_attention_layers`; validation currently only allows Qwen3-Next/Qwen3.5/Qwen4-Exp/GLM-5/OLMo Hybrid. `KimiLinear` must be added only after parsing and validating the explicit K3 `kda_layers` + `full_attn_layers` partition.

### Upstream KDA math re-verified

Direct inspection of `moonshotai/Kimi-K3/modeling_kimi_linear.py` confirmed the exact KDA construction and forward ordering:

- `q_proj`: hidden -> `num_heads * head_dim`
- `k_proj`: hidden -> `num_heads * head_dim`
- `v_proj`: hidden -> `num_heads * head_dim`
- three independent `ShortConvolution` instances: `q_conv1d`, `k_conv1d`, `v_conv1d`, each with SiLU activation
- `A_log` shape is `[num_heads]`
- `f_a_proj`: hidden -> `head_dim`
- `f_b_proj`: `head_dim` -> `num_heads * head_dim`
- `dt_bias` shape is `[num_heads * head_dim]`
- `b_proj`: hidden -> `num_heads`
- production `use_full_rank_gate=true` uses `g_proj`: hidden -> `num_heads * head_dim`
- `o_norm` is gated RMSNorm with sigmoid gate activation
- `o_proj`: `num_heads * head_dim` -> hidden

The upstream execution order is:

1. project Q/K/V
2. run separate causal short convolutions with SiLU
3. compute vector forget gate as `f_b_proj(f_a_proj(hidden_states))`
4. compute scalar-per-head beta logits with `b_proj(hidden_states)`
5. execute KDA with Q/K L2 normalization, vector decay from `A_log` + `dt_bias`, sigmoid beta, and the delta-rule recurrent update
6. compute the output gate (`g_proj` in production)
7. apply sigmoid-gated RMSNorm to the recurrent output
8. apply `o_proj`

Cached one-token decoding stores three convolution states plus one recurrent KDA state per KDA layer. Training uses chunk KDA; cached single-token generation switches to recurrent KDA.

### Shared Vulkan recurrence decision

The existing GLM-5 KDA shaders were re-read and are mathematically compatible with the reusable K3 recurrence core at the important semantic points:

- vector-valued decay per head/channel
- `A_log` exponentiation and `dt_bias`
- optional lower-bound safe gate
- sigmoid beta
- FP32 Q/K L2 normalization
- decay is applied to state before delta correction
- memory read uses the decayed state and normalized key
- delta is `(value - memory) * beta`
- state write is key outer-product delta correction
- query read occurs from the updated state
- gated RMSNorm uses sigmoid(gate)

Therefore K3 should reuse/generalize the GLM-5 recurrence/backward/gated-RMSNorm kernels instead of introducing a second recurrence engine.

Important: this reuse is only for equivalent math. K3 still requires its own architecture, loader/export ABI, projection variant, schedule, and cache bookkeeping.

### Separate Q/K/V convolution representation

K3's checkpoint ABI must keep canonical tensors:

- `self_attn.q_conv1d.weight`
- `self_attn.k_conv1d.weight`
- `self_attn.v_conv1d.weight`

However, because the native depthwise causal-convolution kernel is channel-independent, the three convolution weight tensors can be concatenated into one contiguous internal buffer ordered Q then K then V, matching the already-packed Q/K/V activation layout. This preserves exact per-channel math while avoiding a second convolution kernel.

The loader/exporter must still preserve the three canonical HF tensor names and split/merge them deterministically on load/save. Generation likewise needs logically separate Q/K/V convolution carry regions even if they are stored contiguously in one GPU buffer.

### Existing KDA abstraction status

The current shared implementation still consists of:

- `HostQwenGatedDeltaProjection::{Qwen3Next,Qwen35,OlmoHybrid,Glm5}`
- `HostQwenGatedDeltaNet`
- `VulkanQwenGatedDeltaProjection::{Qwen3Next,Qwen35,OlmoHybrid,Glm5}`
- `VulkanQwenGatedDeltaNet`

There is still no Kimi-specific projection variant. The next projection variant should encode canonical K3 tensors explicitly, likely including:

- `q`, `k`, `v`
- `forget_a` / `f_a_proj`
- `forget_b` / `f_b_proj`
- `beta` / `b_proj`
- full-rank `gate` / `g_proj` for production K3
- canonical separate Q/K/V convolution weights packed internally only after load
- `A_log`, `dt_bias`, `o_norm.weight`

The generic Gated-DeltaNet wrapper can remain shared if Kimi-specific dimensions, decay shape, cache layout, parameter names, and backward routing are represented explicitly rather than inferred from GLM-5.

### SiTU confirmation

Upstream SiTU was re-verified directly:

`situ_gate = beta * tanh(gate / beta) * sigmoid(gate)`

When `activation_situ_linear_beta` is present:

`situ_up = linear_beta * tanh(up / linear_beta)`

Output is `situ_gate * situ_up`. Kimi's MLP concatenates gate/up projections conceptually before applying this activation. This remains unimplemented natively and still requires forward + backward coverage.

### Immediate next step after compaction

1. Add fail-closed/exact `KimiLinear` coverage to the six enum match sites so the crate compiles without pretending K3 is another architecture.
2. Implement `from_kimi_linear_value` (or equivalent) with strict explicit validation of 1-based `kda_layers` and `full_attn_layers` as a complete, duplicate-free, non-overlapping, in-range partition of all decoder layers.
3. Parse `linear_attn_config` generically (`head_dim`, `num_heads`, `short_conv_kernel_size`, `use_full_rank_gate`, `gate_lower_bound`) and preserve K3 MLA geometry/config.
4. Add the distinct Kimi KDA projection/checkpoint variant and canonical SafeTensors mapping, packing the three convolution tensors internally only after load.
5. Reuse the existing GLM-5 vector-decay recurrence/backward/gated-RMSNorm kernels for the Kimi variant after shape/dispatch validation.
6. Add tiny KDA-only config/loader/forward/backward tests before moving on to K3 MLA output gating, SiTU, latent-MoE, and AttnRes.

## Progress report — 2026-09-17 post-compaction continuation

Verified implementation progress:

- Restored exhaustive `KimiLinear` enum integration at the six compile-blocking match sites in `hierarchos-vulkan/src/transformer.rs`.
- `default_lora_target_module()` now uses canonical `q_proj` as the conservative Kimi-Linear default target name.
- Host checkpoint loading, PEFT module discovery/attachment, PEFT export, and HF parameter export now fail closed with explicit Kimi-Linear diagnostics until the real K3 projection/checkpoint ABI and canonical SafeTensors mapping are present. These arms intentionally do not alias K3 to DeepSeek/K2.5/GLM/Qwen.
- Re-ran `cargo check --manifest-path hierarchos-vulkan/Cargo.toml --lib` after the edit. It passed successfully with 17 pre-existing dead-code/unused warnings and no errors.

The compile boundary is therefore no longer the immediate blocker. Next: implement native `kimi_linear` config parsing plus strict `kda_layers`/`full_attn_layers` partition validation, then add the distinct Kimi KDA projection ABI.

### Restored handoff detail from the interrupted continuation

- The initial `KimiLinear` enum-integration blocker has already been cleared. Kimi-Linear now fails closed at unsupported load/export/PEFT paths instead of being aliased to an existing family.
- The generic `VulkanTransformerConfig::linear_attention_layers` bitmap is the intended representation for K3's explicit KDA/full-MLA partition; do not introduce a second scheduler unless a later semantic requirement proves the bitmap insufficient.
- The active implementation batch is intentionally narrowed to the config/schedule contract and the distinct Kimi KDA tensor ABI. This is the first point where real K3 execution semantics begin.
- Before encoding parser defaults, tensor shapes, or checkpoint mappings, refresh the exact Moonshot K3 `config.json` / `configuration_kimi_k3.py` / `modeling_kimi_linear.py` source and follow that source rather than prose approximations.
- A clean `cargo check --manifest-path hierarchos-vulkan/Cargo.toml --lib` was already obtained after the fail-closed enum integration, so config/ABI work should be tested against that stable compile baseline rather than re-solving the six old match errors.

### Critical upstream text-MLA positional-encoding finding — record before further K3 work

Direct re-inspection of the current Moonshot `modeling_kimi_linear.py` changes an important implementation assumption for K3 text MLA:

- `KimiMLAAttention` reads `qk_rope_head_dim` and splits Q/K into `q_pass` + `q_rot` and compressed-KV into `k_pass` + `k_rot`, but the checked text-attention implementation does **not** apply a rotary transform to those `q_rot` / `k_rot` slices.
- The module explicitly sets `self.rotary_emb = None`.
- In `KimiMLAAttention.forward`, `q_rot` and `k_rot` are concatenated back with the non-rotary portions and sent directly to the attention implementation. `position_ids` is accepted by the signature but is not used by this MLA path.
- A repository-wide search of the checked `modeling_kimi_linear.py` found no `apply_rotary_pos_emb`, `rotate_half`, or equivalent text-RoPE call. The only `q_rot` / `k_rot` handling in that file is the split/reshape/expand/concatenate sequence described above.
- `modeling_kimi_k3.py` **does** contain `apply_rope(...)`, but the verified calls are in the MoonViT vision tower (`xq, xk = apply_rope(xq, xk, rope_freqs_cis)`), not in the `KimiLinear` text decoder.
- The production K3 `text_config` inspected in the same source snapshot contains `qk_rope_head_dim=64` but does not serialize a `rope_theta` field. This reinforces that the field name alone must not be treated as evidence that the current text graph performs ordinary DeepSeek-style RoPE.

Implementation consequence:

- Do **not** silently inherit DeepSeek-V3/K2 rotary behavior for `KimiLinear` merely because the geometry fields are named `qk_rope_head_dim` / `q_rot` / `k_rot`.
- The native K3 MLA path should initially mirror the checked Moonshot text implementation exactly: preserve the split dimensions and shared `k_rot` expansion semantics, but leave text RoPE disabled unless a newer/alternate authoritative K3 source or checkpoint-specific code proves that a rotary transform is applied elsewhere.
- Keep the current `KimiLinear` architecture distinct from `DeepseekV3`; this finding is another concrete reason not to normalize K3 to the K2/K2.5 graph.
- When the K3 MLA forward/parity test is added, include a source-matched test that would detect accidental rotary modification of the `q_rot` / `k_rot` slices.

This finding is architecture-critical and must survive future handoffs/compactions.

## Progress report — 2026-09-17 Kimi config/schedule closure

Verified in the live worktree after the post-compaction parser patch:

- `from_kimi_linear_value` is now present and `VulkanTransformerConfig::from_hf_value` dispatches canonical `model_type="kimi_linear"` through it.
- The generic composite-wrapper fallback resolves top-level `model_type="kimi_k3"` through nested `text_config.model_type="kimi_linear"`; a dedicated regression test proves the resulting architecture remains `VulkanTransformerArchitecture::KimiLinear` rather than aliasing to DeepSeek/K2/K2.5/GLM/Qwen.
- Kimi attention scheduling is represented by the existing `linear_attention_layers` bitmap and now strictly validates the upstream 1-based `kda_layers` / `full_attn_layers` lists as a duplicate-free, non-overlapping, in-range, complete partition.
- Added a production-layout regression using the real 93-layer pattern: 69 KDA layers and 24 full-attention layers (multiples of 4 through layer 92 plus layer 93). The test verifies every one-indexed layer maps to the expected native selector.
- Added fail-closed schedule regressions for duplicate KDA entries, KDA/full-attention overlap, a missing layer, and an out-of-range layer.
- The tiny wrapper/config regression also verifies `rotary_dim=0` for the current Moonshot text implementation, `hidden_act="situ"`, dense-first-layer MoE scheduling, sigmoid routing, and shared-expert normalization through the existing common config container.

Checks actually run:

- `cargo test --manifest-path .\hierarchos-vulkan\Cargo.toml --lib kimi_linear_attention_partition` -> 1 malformed-schedule regression passed, 0 failed.
- `cargo test --manifest-path .\hierarchos-vulkan\Cargo.toml --lib kimi_linear_production_attention_partition_is_exact_and_one_indexed` -> 1 production 93-layer regression passed, 0 failed.
- `cargo test --manifest-path .\hierarchos-vulkan\Cargo.toml --lib kimi_k3_wrapper_resolves_nested_kimi_linear_config_without_aliasing` -> 1 wrapper/config regression passed, 0 failed.

New implementation constraint exposed by this batch:

- `VulkanTransformerConfig` currently has only the generic `linear_attention_layers` selector; K3's `linear_attn_config` execution geometry is validated but not yet persisted in a dedicated config object. The upcoming distinct Kimi KDA projection/runtime ABI must carry or derive `num_heads`, `head_dim`, `short_conv_kernel_size`, full-rank gate selection, and the gate lower bound explicitly rather than borrowing GLM-5 defaults.

Next concrete step: add a distinct Kimi projection/checkpoint variant to `HostQwenGatedDeltaProjection` / `VulkanQwenGatedDeltaProjection`, wire canonical Q/K/V + f_a/f_b + b + g tensor geometry, and map the three canonical short-convolution tensors into the shared packed internal convolution buffer without losing round-trip names.

### File-backed K3 package ownership fix

The config-only wrapper test exposed a separate real-package gap: before this batch, `transformer_text_config_from_package()` and the tensor-prefix rebasing helpers knew about `kimi_k25` but not `kimi_k3`. A real outer K3 package would therefore have kept the top-level `kimi_k3` config instead of extracting its nested `kimi_linear` text config, and canonical `language_model.model.*` / `language_model.lm_head.*` tensors would not have been rebased into the common native text store.

Fixed and verified:

- `transformer_text_config_from_package()` now treats `kimi_k3` as a nested-`text_config` wrapper.
- `transformer_text_wrapper_prefix("kimi_k3")` is `language_model.model`.
- `transformer_text_wrapper_lm_head_prefix("kimi_k3")` is `language_model.lm_head`.
- The multimodal wrapper merge test now includes K3 and proves trained text tensors are re-keyed to those canonical package locations while an unrelated vision tensor is preserved unchanged.

Checks actually run:

- `cargo test --manifest-path .\hierarchos-vulkan\Cargo.toml --lib kimi_k3_package_uses_nested_text_config_and_language_model_prefixes` -> 1 passed, 0 failed.
- `cargo test --manifest-path .\hierarchos-vulkan\Cargo.toml --lib multimodal_wrapper_tensor_merge_rekeys_text_and_preserves_auxiliary_weights` -> 1 passed, 0 failed with K3 included in the wrapper matrix.

This means the next Kimi checkpoint loader should consume the rebased common `model.layers.{i}.*` namespace internally while exporting back through the K3 wrapper to canonical `language_model.model.layers.{i}.*` and sibling `language_model.lm_head.*` names.

## Progress report — 2026-09-17 Kimi KDA projection/runtime closure

Files changed in this batch:

- `hierarchos-vulkan/src/transformer.rs`
- `PROGRESS_K3_AUDIT.md`

Implemented and verified:

- Added/finished the distinct `HostQwenGatedDeltaProjection::Kimi` and `VulkanQwenGatedDeltaProjection::Kimi` execution path instead of aliasing K3 to GLM-5.
- Kimi KDA forward now projects independent Q/K/V, packs them into the shared channel layout, computes `f_b_proj(f_a_proj(hidden))`, scalar-per-head beta logits, and the production full-rank `g_proj(hidden)` output gate, then reuses the native vector-decay KDA recurrence + gated-RMSNorm kernels.
- Kimi KDA backward now routes `grad_z` directly through the full-rank `g_proj` and backpropagates the forget-gate low-rank chain separately, while retaining shared recurrent/convolution backward kernels.
- Freeze, bias-training, and optimizer-step routing now include the Kimi projection linears.
- Canonical component export uses `model.layers.{i}.self_attn.*` for Kimi and emits `q_proj`, `k_proj`, `v_proj`, `f_a_proj`, `f_b_proj`, `b_proj`, `g_proj`, `dt_bias`, `A_log`, `o_norm.weight`, and separate `q_conv1d`/`k_conv1d`/`v_conv1d` tensors.
- The three canonical convolution tensors are kept independent at the checkpoint boundary but concatenated Q/K/V internally. This is exact for the current native depthwise causal convolution because each channel is independent; no cross-channel convolution occurs. Cached carry can therefore remain one contiguous Q/K/V buffer while preserving three logical regions.
- A new Vulkan component regression, `kimi_linear_kda_projection_runs_forward_backward_and_exports_canonical_abi`, executes Kimi KDA forward + backward and checks finite/nonzero output/input gradients plus the canonical export namespace and deterministic Q/K/V convolution splitting.

Experimental bug found and fixed:

- The first run of the new regression failed because Kimi component export still named the output norm `self_attn.norm.weight` even though the loader/upstream ABI requires `self_attn.o_norm.weight`. The architecture-specific export condition now includes `KimiLinear`; rerunning the exact test passed.

Checks actually run:

- `cargo check --manifest-path hierarchos-vulkan/Cargo.toml --lib` after the Kimi runtime/exhaustiveness edits -> passed, 0 errors, same 17 pre-existing warnings.
- First run of `cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib kimi_linear_kda_projection_runs_forward_backward_and_exports_canonical_abi -- --nocapture` -> failed exactly on missing canonical `model.layers.0.self_attn.o_norm.weight`; this exposed the export bug above.
- Re-run of the same focused test after the fix -> 1 passed, 0 failed, 648 filtered out. The test executed on the available Vulkan device rather than taking the no-device early return.

Remaining K3 blockers after this batch:

- K3 MLA full-attention output gate before `o_proj`.
- SiTU forward/backward.
- LatentMoE projection shell.
- AttnRes and final output AttnRes.
- Full mixed KDA/MLA Kimi-Linear model fixture and package load/export closure.
- Cached generation parity for the Kimi KDA path and then the mixed graph.
- External Moonshot/HF oracle forward/backward/optimizer parity and save/reload parity.
- Explicit official MXFP4 packed-checkpoint fail-closed/proper support boundary and K3 vision boundary verification.

Next concrete step: implement the K3 MLA output gate as a Kimi-specific extension of the native MLA path, preserving the already-recorded no-text-RoPE behavior from the checked Moonshot source, then add a tiny MLA-only forward/backward regression before moving to SiTU.

## Progress report — 2026-09-17 K3 MLA output-gate closure

Files changed in this batch:

- `hierarchos-vulkan/src/transformer.rs`
- `PROGRESS_K3_AUDIT.md`

Implemented and verified:

- `load_mla_attentions()` now accepts `KimiLinear` and consumes tensors only for configured full-attention layers. KDA layers no longer force dummy MLA entries or risk consuming the full-attention table out of phase.
- Kimi full-attention `self_attn.g_proj.weight` is loaded only when `mla_use_output_gate=true`, with output width `num_attention_heads * v_head_dim` rather than the larger MLA Q width.
- The existing native attention-gate execution path is reused without a new shader: for Kimi it selects the per-element sigmoid branch, multiplies the raw MLA attention output elementwise, and feeds the gated value into `o_proj`. The existing backward path propagates both the attention gradient and the `g_proj` gradient.
- The Transformer graph consumes compact Kimi MLA tables only on non-linear-attention layers while the existing per-layer optional gate table keeps KDA slots empty. This preserves the explicit KDA/full-MLA schedule parsed from the upstream config.
- Added `kimi_linear_full_attention_loader_is_hybrid_schedule_aware_and_loads_output_gate`, using the tiny 3-KDA/1-MLA Kimi config. It verifies exactly one MLA block is loaded, the first three gate slots are empty, the fourth layer owns the canonical full-width gate, and the gate has no bias.

Checks actually run:

- `cargo test --manifest-path .\\hierarchos-vulkan\\Cargo.toml --lib kimi_linear_full_attention_loader_is_hybrid_schedule_aware_and_loads_output_gate -- --nocapture` -> 1 passed, 0 failed.
- `cargo test --manifest-path .\\hierarchos-vulkan\\Cargo.toml --lib kimi_linear_ -- --nocapture` -> 5 passed, 0 failed, including the Vulkan KDA forward/backward component regression.
- `cargo test --manifest-path .\\hierarchos-vulkan\\Cargo.toml --lib mla -- --nocapture` -> 6 passed, 0 failed, covering the existing MLA-family regression slice.
- `cargo test --manifest-path .\\hierarchos-vulkan\\Cargo.toml --lib attention_gate -- --nocapture` -> 2 passed, 0 failed.
- `git diff --check -- .\\hierarchos-vulkan\\src\\transformer.rs` -> passed; only Git's existing LF-to-CRLF warning was emitted.
- `cargo fmt --manifest-path .\\hierarchos-vulkan\\Cargo.toml -- --check` did **not** complete: rustfmt attempted to allocate 16,945,790,492 bytes and aborted with an out-of-memory error while processing the very large `transformer.rs`. Treat this as a formatter/resource limitation, not a passing formatting check. Future handoffs must not report `cargo fmt --check` green until it actually runs successfully.

Critical implementation note from this batch:

- K3's MLA gate width is `num_attention_heads * v_head_dim`, not `query_hidden_size()`. Production K3 therefore gates the attention value width before `o_proj`; using the Q width would be dimensionally wrong because Q includes both NOPE and the nominal rotary slice.

Remaining K3 blockers after this batch:

- SiTU forward/backward.
- LatentMoE projection shell.
- AttnRes and final output AttnRes.
- Full mixed KDA/MLA Kimi-Linear model fixture and package load/export closure.
- Cached generation parity for Kimi KDA and the mixed graph.
- External Moonshot/HF oracle forward/backward/optimizer parity and save/reload parity.
- Explicit official MXFP4 packed-checkpoint and vision fail-closed verification.

Next concrete step: implement native generic SiTU-GLU forward/backward using the configured `activation_situ_beta` and `activation_situ_linear_beta`, add deterministic finite-difference coverage, and then attach that activation to the Kimi dense/MLP path before starting the LatentMoE shell.

## Progress report — 2026-09-17 SiTU closure

Files changed in this batch:

- `hierarchos-vulkan/src/transformer.rs`
- `hierarchos-vulkan/shaders/transformer_situ_forward.comp`
- `hierarchos-vulkan/shaders/transformer_situ_backward.comp`
- `PROGRESS_K3_AUDIT.md`

Implemented and verified:

- Added native Vulkan SiTU-GLU forward semantics: `beta * tanh(gate / beta) * sigmoid(gate)`, multiplied by either the raw up projection or the optional `linear_beta * tanh(up / linear_beta)` transform.
- Added the exact analytic backward derivatives for both gate and up branches.
- Wired SiTU into the existing dense gated-MLP forward/backward path for `activation_function == "situ"`; no Python/FLA/Triton/CUDA fallback is involved.
- Kimi-Linear config parsing now preserves non-default `activation_situ_beta` and nullable `activation_situ_linear_beta`; a null linear beta leaves the up branch unbounded.
- Added direct Vulkan reference-value checks plus central finite-difference gradient checks and a no-linear-beta forward case.

Checks actually run:

- `glslc shaders/transformer_situ_forward.comp -o shaders/transformer_situ_forward.spv` -> passed.
- `glslc shaders/transformer_situ_backward.comp -o shaders/transformer_situ_backward.spv` -> passed.
- `cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib kimi_linear_situ -- --nocapture` -> 2 passed, 0 failed; the Vulkan test executed on the available device.
- A fresh `cargo fmt --manifest-path hierarchos-vulkan/Cargo.toml -- --check` did not complete: rustfmt attempted to allocate 18,253,901,008 bytes and aborted with OOM. This independently reproduces the prior formatter/resource failure at a slightly larger requested allocation; it is not a green formatting check.

Next concrete step: implement K3 Stable LatentMoE. Preserve the existing router/top-k/shared-expert machinery, but add the Kimi-specific hidden -> routed latent projection, optional latent RMSNorm, routed experts in latent width, and latent -> hidden projection without flattening K3 into ordinary DeepSeek/K2.5 expert execution.

## Progress report — 2026-09-18 post-compaction recovery: LatentMoE + AttnRes component verification

Recovered live worktree state that was newer than the previous audit text and verified it before treating it as completed work.

Files/components present in the recovered state:

- `hierarchos-vulkan/src/transformer.rs` contains the Stable LatentMoE shell and Kimi AttnRes host/runtime/component-loader code.
- New AttnRes Vulkan shaders are present:
  - `hierarchos-vulkan/shaders/transformer_kimi_attn_res_softmax.comp/.spv`
  - `hierarchos-vulkan/shaders/transformer_kimi_attn_res_softmax_backward.comp/.spv`
  - `hierarchos-vulkan/shaders/transformer_kimi_attn_res_mix.comp/.spv`
  - `hierarchos-vulkan/shaders/transformer_kimi_attn_res_mix_backward_prob.comp/.spv`
  - `hierarchos-vulkan/shaders/transformer_kimi_attn_res_mix_backward_candidate.comp/.spv`
- `load_kimi_attn_res_stack()` reads canonical per-layer `self_attention_res_*` / `mlp_res_*` tensors and model-level `output_attn_res_*` tensors.
- `VulkanKimiAttnRes` implements RMSNorm-based candidate scoring, learned scalar projection, source softmax, weighted residual mixing, and analytic backward through both the softmax/scoring path and the direct weighted-mixture path.
- `VulkanMoe` contains the Kimi Stable LatentMoE routed shell: optional `latent_down_proj`, optional routed latent RMSNorm, expert execution at routed latent width, and `latent_up_proj` back to model hidden width. The learned router and shared expert remain at the ordinary hidden width.

Checks actually run after recovery:

- `cargo check --manifest-path hierarchos-vulkan/Cargo.toml --lib` -> passed, 0 errors. The build emitted existing dead-code/unused warnings; importantly, AttnRes symbols are currently reported unused, which is direct evidence that the standalone operator is not yet connected to the complete decoder graph.
- `cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib kimi_linear_latent_moe -- --nocapture` -> 2 passed, 0 failed, 654 filtered out:
  - `kimi_linear_latent_moe_loader_reads_canonical_block_sparse_abi`
  - `kimi_linear_latent_moe_runs_situ_forward_and_backward_with_finite_difference`
- `cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib kimi_linear_attn_res -- --nocapture` -> 2 passed, 0 failed, 654 filtered out:
  - `kimi_linear_attn_res_loader_reads_canonical_tensor_names`
  - `kimi_linear_attn_res_matches_reference_and_finite_difference`

Current boundary after these checks:

- Stable LatentMoE is implemented and component-verified, but the full tiny Kimi mixed graph still needs a dense-first-layer + sparse-LatentMoE integration test.
- AttnRes math and tensor loading are implemented and component-verified, but `HostKimiAttnRes`, `HostKimiAttnResStack`, `VulkanKimiAttnRes`, and the AttnRes kernels are still dead code in an ordinary library build. Do not mark AttnRes complete until the per-layer residual-source lifecycle and model-level final output AttnRes are wired into forward/backward/optimizer/export paths.
- Top-level `KimiLinear` checkpoint construction still intentionally fails closed in `from_hf_package_with_config_and_encoder_seq_len`; full-model loading must remain disabled until the graph is complete enough to execute the canonical text backbone without silently omitting K3 semantics.
- The previously recorded rustfmt OOM remains unresolved. No `cargo fmt --check` pass should be claimed.

Next concrete step: wire the verified AttnRes operator into the Kimi decoder graph. Preserve block residuals at `attn_res_block_size` boundaries, form the exact candidate-source set expected by Moonshot for each self-attention and MLP residual mixer, propagate gradients back into every selected residual source, and apply the final model-level `output_attn_res_*` mixer before the terminal normalization/head as required by the authoritative Kimi implementation. Then add a tiny graph-level AttnRes boundary regression before enabling the full Kimi checkpoint loader.

## Findings checkpoint — 2026-09-18 authoritative Kimi AttnRes graph semantics

Durable source finding recovered directly from Moonshot's current `modeling_kimi_linear.py` and rechecked against the live native graph before any integration edits:

- `_apply_attn_res(prefix_sum, block_residual, proj, norm)` forms the candidate tensor by concatenating all archived block residuals with the current `prefix_sum` as the final source. It computes RMS normalization in FP32, scores each source with the elementwise product of the RMSNorm weight and the scalar projection weight, softmaxes across sources, and returns the FP32 weighted mixture cast back to the candidate dtype. This matches the standalone native `VulkanKimiAttnRes` component already verified.
- At decoder entry, `block_residual` is an empty `[num_tokens, 0, hidden]` tensor. Each layer begins with `prefix_sum = hidden_states`.
- Before self-attention, the learned self-attention residual mixer runs only when at least one archived block residual exists. Its candidate sources are `block_residual + current prefix_sum`.
- Block boundaries are determined by the zero-based condition `layer_idx % attn_res_block_size == 0`. At such a boundary, the current pre-attention `prefix_sum` is appended to `block_residual`, then the local `prefix_sum` is set to `None` before the attention branch executes.
- After self-attention, if the local prefix was retained, attention output is added to it; at a boundary layer where it was cleared, the attention output becomes the new local prefix by itself.
- The MLP AttnRes mixer always runs in the AttnRes-enabled path. Its candidate sources are the current post-attention local prefix plus all archived block residuals. The mixed hidden state is then passed through the ordinary `post_attention_layernorm` and MLP/MoE branch.
- After MLP/MoE, its branch output is added into the local prefix when one exists, otherwise it becomes the local prefix. The layer returns both that prefix and the updated archived `block_residual` table.
- After the final decoder layer, Moonshot applies model-level `output_attn_res_*` with candidate sources `block_residual + final hidden_states`, then applies the ordinary terminal RMSNorm. Therefore the native model-level output AttnRes must execute before `final_norm` / LM projection, not after it.
- The live native graph currently still has the explicit top-level `KimiLinear` package-load fail-closed guard (`"Kimi-Linear checkpoint loading is not enabled until the native K3 KDA/MLA tensor ABI is fully implemented"`). This must remain in place until the complete K3 residual topology and full-model execution path are actually connected and parity-tested.
- The generic native decoder currently performs ordinary pre-norm residual adds inside `VulkanTransformerLayer::record_forward_with_cross_attention`. K3 AttnRes therefore cannot be implemented as a model-only pre/post wrapper; it requires Kimi-specific residual inputs/outputs at both the attention and MLP sublayer boundaries while preserving the existing KDA/MLA branch computations.

Implementation implication recorded for handoff safety:

- Prefer graph/layer integration that reuses existing attention and MLP kernels but overrides only the Kimi residual topology. Do not duplicate KDA/MLA/SiTU/LatentMoE computation paths.
- Any graph implementation must also mirror the reverse residual fanout in backward: gradients from each learned AttnRes mixture must accumulate into every selected archived source and the current prefix path, including block-boundary anchors and the final output mixer.
- Acceptance remains strict: component tests are necessary but not sufficient. K3 completion requires full-model forward/backward/optimizer/save-reload parity against Transformers + Moonshot custom code with maximum absolute logit drift `<= 2e-7`.

## Final verification closure — 2026-09-18

This supersedes the older blocker notes above for the current worktree. The missing full-graph Kimi-Linear integration was completed after those intermediate checkpoints, and the final pre-push verification below was run against the resulting tree.

Authoritative reference check:

- Moonshot's live `moonshotai/Kimi-K3` remote-code files still identify the released source as commit `c5d1dd4` (`configuration_kimi_k3.py`, `modeling_kimi_linear.py`, and the top-level K3 package). The validation oracle continues to instantiate Moonshot's `KimiLinearConfig` / `KimiLinearForCausalLM` module graph and substitutes only source-equivalent PyTorch implementations for the unavailable FLA primitives plus differentiable execution for Moonshot's inference-only MoE dispatch.
- The K3 forward contract remains a hard maximum absolute logit error of `2e-7` with `rtol=0`.

Final checks actually run on the current AMD Radeon Vulkan device:

- `.\\.venv-vulkan\\Scripts\\python.exe hierarchos-vulkan\\validation\\verify_hf_logits.py --headline-strict` -> passed for all nine headline families with zero failing values. K3 mixed-graph max absolute logit error: `5.21540641784668e-8`.
- `.\\.venv-vulkan\\Scripts\\python.exe hierarchos-vulkan\\validation\\verify_hf_logits.py --families kimi_k3_kda kimi_k3_mla kimi_k3_text` -> all three source-backed K3 probes passed under `atol=2e-7, rtol=0`: KDA `5.587935447692871e-8`, MLA `6.146728992462158e-8`, mixed `5.21540641784668e-8`.
- `.\\.venv-vulkan\\Scripts\\python.exe hierarchos-vulkan\\validation\\verify_hf_training.py --headline-strict` -> all nine families passed exactly two AdamW steps under the `2e-7` parameter-drift contract. Maximum absolute parameter errors: DeepSeek-V4 `1.1920928955078125e-7`; Phi-4 multimodal text `1.1920928955078125e-7`; Phi-3 `1.1920928955078125e-7`; Kimi K2.5 text `1.1920928955078125e-7`; Kimi K3 text `5.963374860584736e-8` across 18,450 parameter values; Mistral-4 `2.60770320892334e-8`; MiniMax-M3 `3.5390257835388184e-8`; Gemma-4 `1.2200325727462769e-7`; MiniMax-M2 `1.1920928955078125e-7`.
- `cargo test --manifest-path hierarchos-vulkan\\Cargo.toml --lib kimi_k3` -> 4 passed, 0 failed, covering nested `kimi_k3 -> kimi_linear` wrapper resolution, canonical `language_model.*` package prefixes, unsupported MoonViT vision fail-closed behavior, and MXFP4 compressed-tensor fail-closed behavior.
- `cargo test --manifest-path hierarchos-vulkan\\Cargo.toml --lib` -> 654 passed, 0 failed, 8 ignored. The ignored tests are explicitly opt-in GPU research/microprofile cases; the ordinary library regression suite is green. This run includes K3 SiTU math, KDA forward/backward/export, K3 AdamW parameter updates, mixed-graph checkpoint roundtrip, and mixed-generation cache consistency.
- `git diff --check` -> passed. Git emitted only the existing LF-to-CRLF working-copy warnings.
- `cargo fmt --manifest-path hierarchos-vulkan\\Cargo.toml -- --check` -> did not complete because `rustfmt` attempted to allocate `49,504,912,608` bytes and aborted OOM. This is a formatter/resource limitation, not a passing formatting check and not a test failure; do not report rustfmt green for this tree.

Final parity conclusion for this worktree:

- Kimi K3/Kimi-Linear is green against the Moonshot-derived reference for isolated KDA, isolated MLA, and the production-style mixed hybrid graph, including full forward/backward/two-step AdamW coverage at substantially less than the required `2e-7` maximum drift.
- The other eight previously supported strict two-step AdamW architectures remain green under the same current validation harness; no optimizer-parity regression was observed.
- The full Rust library regression suite is green. No source changes were required by this final verification pass other than recording this closure in the audit.
- Before staging/pushing, review the existing untracked/generated directories and Python bytecode shown by `git status` (`target/`, `.k3-oracle-*`, `.k3-native-export-*`, `__pycache__`, and temporary K3 trace/roundtrip helpers) so verification artifacts are not accidentally included unless intentionally desired.
