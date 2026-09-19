# Hierarchos + Transformers Vulkan training backend

This crate is the native Vulkan training backend shared by coherent-v9
Hierarchos and Hugging Face Transformer models. The original Hierarchos/RWKV
graph remains intact, while `VulkanTransformer` adds causal and bidirectional
self-attention, Transformer normalization/MLP variants, full backward, AdamW,
dropout, and native LoRA/PEFT training without routing model math through
PyTorch.

Transformer packages can be trained directly with
`hierarchos-vulkan-transformer-train`, or through the unified
`hierarchos-native-cli transformer-train` / `transformer-finetune` commands.
Host-side Rust handles Hugging Face download/tokenization and dataset I/O; the
model forward pass, attention, loss, backward pass, LoRA math, and optimizer
step execute through Vulkan. The authoritative architecture inventory is
generated from `VulkanTransformerArchitecture::ALL` and its explicit aliases;
see [README_ARCHITECTURES.md](README_ARCHITECTURES.md), or run
`hierarchos-native-cli architectures`, for the current list and counts.
Coverage spans GPT/OpenAI-GPT, BERT/BigBird/RoBERTa/XLM
encoder families, GPT-2/Neo/NeoX/J/Falcon/BLOOM/MPT/CodeGen/CTRL, BART/MBART/
Marian/Pegasus-style encoder-decoder families, OPT/BioGPT/XGLM, Llama/Mistral/
Ministral/Gemma/Granite/Cohere/OLMo-style decoders, T5/MT5/UMT5, and
Mixtral/Qwen2-MoE/Qwen3-MoE/Cohere2-MoE/Granite-MoE/OLMoE/FlexOlmo routed MoE,
plus Qwen2/Qwen3, Phi/Phi-3, GLM/GLM4, ERNIE-4.5 dense, Hunyuan v1
dense, BitNet, Emu3 text, Arcee/AFM, Jais2, VaultGemma, HyperCLOVAX, Apertus,
Helium, Cohere Compass text, SmolLM3, EXAONE-4, Nemotron, Seed-OSS, CWM,
NanoChat, StarCoder2, EuroBERT, ESM/ESMC, ModernBERT, and ModernBERT Decoder.
Text-backbone package compatibility also covers Fuyu/Persimmon, Mistral3,
Qwen2/Qwen2.5/Qwen3 vision wrappers, Qwen2.5-Omni text, GLM4V/GLM-4.5V, HunYuanVL,
Gemma3, Phi-4 Multimodal, EXAONE-4.5, Cohere Compass, Aria, Emu3, GOT-OCR2,
Aya Vision, LLaVA/LLaVA-NeXT/LLaVA-OneVision and video variants, Idefics2/3,
VipLLaVA, SmolVLM, Qwen2-Audio, Voxtral, FastVLM, InternVL, Janus, Ovis2,
AudioFlamingo3, Qwen3-ASR, Granite Speech/Granite Speech Plus, GLM-ASR,
MusicFlamingo, VibeVoice-ASR, and Kimi K3's native `KimiLinear` language model
while preserving untouched non-text package tensors on export.
These contracts include the relevant fused/split QKV layouts, MHA/MQA/GQA,
RoPE/ALiBi, sliding/local attention, RMSNorm/LayerNorm, tied/untied heads, and
bidirectional masked-language-model objectives where the family defines them.
Kimi K3 is a text-backbone contract: hybrid KDA/MLA scheduling, KDA recurrent
and convolution generation state, MLA output gating, SiTU, Stable LatentMoE,
AttnRes, backward/AdamW, cached generation, and SafeTensors round-trip are native.
MoonViT/vision-projector execution is not claimed. The official K3
`compressed-tensors` `mxfp4-pack-quantized` weights are deliberately rejected;
use an unquantized FP32/BF16 KimiLinear package for native execution. Kimi K2 and
K2.5 continue to use their existing DeepSeek-style text graph.
OLMoE follows the current Hugging Face packed-expert ABI and preserves its
full-projection Q/K RMSNorm -> `clip_qkv` -> RoPE ordering. ModernBERT masked-LM
execution also preserves
the Hugging Face prediction transform exactly (`dense -> activation -> LayerNorm
-> decoder`), including classifier/norm bias policy, tied decoder weights,
backward propagation, AdamW updates, LoRA base freezing, and SafeTensors export.

This is broad native Transformer coverage, not yet universal compatibility with
every architecture shipped by the Python `transformers` package. The upstream
registry also contains hundreds of vision, audio, multimodal, wrapper, hybrid,
and specialized model types. Architectures that require graph primitives not
implemented here yet—such as routed-MoE/MLA topologies with routing semantics
outside the native packed-expert implementations, hybrid SSM/convolution
blocks, or multimodal towers—remain fail-closed rather than silently running
different math. T5-family relative position bias and encoder-decoder
cross-attention are implemented natively.

Registry coverage can be audited against the exact adjacent Transformers
checkout without importing the Python package (and therefore without depending
on optional Python package versions):

```powershell
python hierarchos-vulkan/validation/audit_transformers_coverage.py
python hierarchos-vulkan/validation/audit_transformers_coverage.py --json
python hierarchos-vulkan/validation/audit_transformers_coverage.py --all-tasks
```

The audit reports the complete `configuration_*.py` model-type registry plus
the AutoModel causal-LM, masked-LM, and seq2seq-LM mappings, then diffs each
against the canonical Vulkan graph contracts and explicit package aliases. It
is a coverage inventory, not a numerical-parity claim: newly advertised model
types still require architecture-specific parity tests before they should be
added to the Vulkan registry.

The audit also inventories all other AutoModel task mappings (classification,
vision, audio, multimodal, and others). Their model-type overlap is explicitly
separate from task implementation: a supported text backbone does not establish
support for its multimodal tower or classification head.

Recent correctness work repairs MiniMax-M2 packed-expert down-projection loading
and the Switch Transformers encoder's dense/sparse checkpoint paths. Switch
encoders now load their own expert weights and independent sparse-layer schedule.
`VulkanSeq2SeqTransformer::from_hf_package_with_batch_size` supports batched
encoding and teacher-forced logits, including different encoder/decoder sequence
lengths. Existing batch-one construction and generation APIs remain available.

Native FP32 MoE input jitter now runs in Vulkan during training, with the same
Philox multipliers replayed in backward. It affects both router and expert
projections, preserves the normalization tape, and is disabled during evaluation.
Mixtral, MiniMax-M2, DBRX and Switch use this operation. The reference training
suite checks Mixtral, MiniMax-M2 and DBRX with identical jitter draws in HF;
this does not imply identical RNG streams to an arbitrary PyTorch run.

See [COMPATIBILITY.md](COMPATIBILITY.md) for reproducible validation commands,
the tested scope, and remaining work toward full Transformers parity.

## Standalone distribution

The native backend can be staged and built without the surrounding Python
Transformers checkout. From the repository root on Windows:

```powershell
.\hierarchos-vulkan\package-standalone.ps1 -OutDir C:\path\to\hierarchos-native-backend
```

The packager creates an isolated Cargo workspace containing only
`hierarchos-inference`, `hierarchos-vulkan`, `hierarchos-native-cli`, and
`hierarchos-gui`, then builds release binaries and places the user-facing
executables together under `bin/`. Pass `-SkipBuild` to create a source-only
package. The generated package has its own README and license and does not
depend on `src/transformers`, Python, PyTorch, or this repository layout at
runtime.

Generation now has an HF-style policy layer through
`VulkanTransformerGenerationConfig`. `generation_config.json` fields for
`max_new_tokens`/`max_length`, `min_length`, `min_new_tokens`, `do_sample`,
`temperature`, `top_h`, `top_k`, `top_p`, `min_p`, `typical_p`,
`epsilon_cutoff`, `eta_cutoff`, `repetition_penalty`, `no_repeat_ngram_size`,
`bad_words_ids`, `suppress_tokens`, `begin_suppress_tokens`,
`forced_bos_token_id`, scalar-or-list `forced_eos_token_id`, `num_beams`,
`length_penalty`, `early_stopping`, and scalar-or-list `eos_token_id` are
parsed natively. The corresponding sequence constraints run in both
single-beam and beam decoding, including KV-cached paths; sampling warpers use
the current Transformers order so the two decoding paths share one policy.
`VulkanTransformer::generate_ids` supports greedy decoding, deterministic
Philox-seeded multinomial sampling, beam search, and Philox-seeded beam-search
sampling; `generate_ids_streaming`
emits committed greedy/sampled tokens incrementally. Single-beam causal
generation now uses a native per-layer device-resident KV cache for GPT-2,
Llama/OpenLLaMA, Mistral/Mixtral, and Qwen2/Qwen3 dense/MoE decoder contracts,
including learned-position and RoPE offsets plus MQA/GQA key/value geometry.
The established contiguous cache remains the default. Setting
`cache_implementation="paged"` opts compatible causal attention layers into a
native Vulkan paged KV arena with logical-to-physical page tables, lazy physical
page allocation, prefix sharing, tail-page copy-on-write for beam forks, and
freed-page reuse. The attention shader consumes the page table directly, so
paged decoding does not repack K/V into a contiguous temporary before attention.
For end users, the native CLI spelling is `--cache-implementation paged` on
`transformer-generate`. The desktop GUI exposes the same policy under
Transformer generation -> `KV cache` -> `Paged Vulkan KV`. Both surfaces leave
paging disabled unless it is selected explicitly. The native CLI also strips
package-level `cache_implementation="paged"` and `continuous_batching_config`
metadata before applying user-supplied overrides, so a downloaded model package
cannot silently enable either serving optimization. An explicitly supplied
`--generation-config` remains an intentional opt-in surface.
Architectures with special cache state (for example recurrent or compressed
attention layers) retain their existing native cache topology, and an explicit
paged request fails closed when the generation graph cannot use the paged path.
For serving-style request batches, repeating `--prompt` plus
`--continuous-batching` opts into the native continuous scheduler (the GUI has
the equivalent `Continuous batching` gate and an additional-prompts field).
Continuous mode implies paged KV. Pageable layers allocate from one shared
physical arena per layer across all active sequences; when a request finishes,
its page-table references are dropped immediately and unreferenced pages return
to the arena free list for subsequent requests. Active single-token decode
steps are recorded together and submitted as one Vulkan command batch, while
each request keeps independent sampling state, stopping criteria, and page
table. This is continuous request scheduling with shared KV allocation; it is
not a claim that the full Transformer forward has been fused into one tensor
batch. The current continuous path supports decoder-only greedy or multinomial
sampling, `num_beams=1`, `num_return_sequences=1`, and no external encoder
context. Native pages are 16 tokens; an explicit
`continuous_batching_config.block_size` must be `16`. The native scheduler
currently accepts only `block_size` and `max_requests_per_batch` from that
config object. Other Hugging Face continuous-batching allocator, offload,
CUDA-graph, or scheduler controls are rejected explicitly instead of being
silently ignored.
Beam search on compatible causal families forks only the post-pruning live
hypotheses, so each surviving beam advances one token instead of rerunning its
full prefix. Sampling follows HF's
beam-mode order: log-softmax, logits processors/warpers, accumulated beam score,
then EOS-aware joint beam×vocabulary sampling without replacement. Unsupported
generation topologies and decoder paths with external encoder context retain
the correctness-first full-prefix executor.

Current milestone:

- A first trainer-facing native loop is available as `hierarchos-vulkan-train`.
  It reads pre-tokenized JSONL, deterministically shuffles with the same portable
  Fisher-Yates policy used by the replay/session layer, pads variable-length
  rows with masked `-100` labels, performs labeled coherent-v9 training through
  the full Vulkan graph, supports canonical gradient accumulation, and exports
  a normal training checkpoint package. The exported `model.safetensors`
  remains the same row-major FP32-master ABI consumed by PyTorch CPU/CUDA and
  `hierarchos-inference`; there is no Vulkan-only weight format. This MVP uses
  fresh zero recurrent/context state for each dataset row and intentionally
  drops a short final batch rather than silently changing the graph batch
  geometry.
- Multi-device training now carries persistent joint-scheduler knowledge forward
  in `vulkan_joint_runtime_profile.v1.json` beside the exported package. The
  profile is keyed by the Hierarchos architecture contract, batch/accumulation
  and token geometry, ordered Vulkan device + driver UUIDs, and the resolved
  transport-backend vector. On the next Vulkan run from that package, the
  previous winning transport width becomes a ceiling on the pre-construction
  physical-adapter vector planner, so the production graphs and persistent
  transport arenas are allocated at the learned width from the beginning. The
  live VRAM vector remains an unconditional safety gate and can still reduce
  that width further when any heterogeneous replica has less headroom.
  Runtime scoring now records per-lane queue-to-completion throughput plus
  timeline-retirement telemetry, applies an explicit heterogeneity-efficiency
  term so a slow replica lowers the arm score, and enables Vulkan timestamp
  queries only on bounded scoring windows. Timestamp GPU nanoseconds from
  independent logical devices are folded into the same profile; device-group
  lanes are marked as sharing one submission arena so those counters are not
  double-counted. A candidate must now earn two clean scored windows before the
  scheduler treats it as confidence-ready, and arm ranking uses a one-sided
  confidence bound over heterogeneity-adjusted throughput so one lucky window
  cannot become a sticky cross-run winner. The running confidence statistics are
  stored in the same profile. The scheduler now goes one layer finer than the
  joint arm as well: persisted per-lane throughput is converted into a
  confidence-ramped relative capacity vector and fed directly into the
  contiguous sequence-shard planner. A fast discrete GPU can therefore receive
  a larger fraction of an optimizer window than a slower integrated GPU instead
  of both being forced to equal token weight. Current-arm measurements are used
  first and missing lanes borrow same-device evidence from other explored arms,
  so switching a global topology does not erase what the run already learned
  about the hardware. The first four observations ramp gradually away from
  equal shares, and relative capacities are bounded to keep noisy telemetry from
  starving a lane. Sequence order and canonical replica-index gradient reduction
  order are unchanged, so this scheduler refinement does not alter checkpoint
  math or the PyTorch/CUDA-compatible FP32-master parameter ABI.

  Scored windows now also retain two phase-local service signals: cumulative
  synchronous replica-gradient reduction time and the AdamW / preceding
  broadcast-retirement boundary time. These are persisted per global arm under
  `phase_service` and logged alongside the learned lane weights. They are the
  measurement seam for factoring `gradient_stream_chunk_values` and
  `optimizer_broadcast_overlap` into independently learned phase arms. Once an
  arm has the same two clean windows required by the global confidence gate, the
  report publishes separate best-known gradient-reduction and optimizer-boundary
  advisors; those advisors are allowed to disagree, making phase conflicts
  visible. Those phase scores now drive the factorized selector described below
  instead of being flattened back into one Cartesian global-arm decision. During training,
  scored measurements are flushed
  to the output package every eight clean windows and immediately when the
  winning arm changes. The write is staged, synced, and atomically renamed;
  restarting the same `--model ... --output ...` command prefers the live
  output-side profile before falling back to the source model package, so an
  interrupted rental/job can reuse tuning work even before the final model
  checkpoint export completes. The sidecar is scheduler-only metadata:
  `model.safetensors`,
  optimizer state, and the canonical FP32-master checkpoint ABI remain
  unchanged for PyTorch CPU/CUDA and native Rust inference. Setting
  `HIERARCHOS_VULKAN_DISABLE_JOINT_RUNTIME_AUTOTUNE` disables both profile reuse
  and live joint-runtime exploration.
  The scheduler has now crossed the factorization boundary: gradient transport
  width, optimizer/broadcast overlap, and tape geometry are ranked from their
  own marginal evidence, only one factor bootstraps/explores on a selection
  step, and the resulting three coordinates are materialized into a composite
  arm only for the window that will actually execute. The global arm table is
  therefore an observation store rather than a Cartesian policy surface. A
  tape geometry is itself factorized into independently ranked sequence-
  microbatch and state-checkpoint-stride coordinates, so the scheduler does not
  recreate the old Cartesian surface one layer down. Runtime scoring now also
  carries per-adapter uncertainty into heterogeneous sharding. Each clean lane
  window updates persisted Welford throughput moments alongside the lane EWMA;
  shard capacity is derived from a one-sided confidence-adjusted throughput
  estimate rather than the EWMA alone. The existing early-window confidence
  ramp still pulls a new lane toward equal share, but a noisy or thermally
  unstable adapter can no longer win extra work from one lucky throughput
  spike. Older joint-runtime profiles remain readable because the new lane
  confidence fields default cleanly when absent. Device reports expose the raw
  confidence sample count, mean, relative uncertainty, and conservative
  throughput estimate so cross-vendor qualification can audit the learned
  split directly.
  The whole-arm scorer also uses an explicit steady-window quality layer: after any composite-coordinate
  switch, two complete optimizer windows are treated as settling/warmup and are
  excluded from selector evidence. Scored windows snapshot live device-local
  usage and the coarse `VK_EXT_memory_budget` pressure bucket per execution
  lane. Once an arm already has two confidence windows, a window is flagged as
  throttle-suspect only when effective throughput falls below 80% of its
  adaptive baseline while GPU timestamp nanoseconds/token simultaneously rise
  above 120% of baseline. High-memory-pressure and throttle-suspect counts are
  persisted and reported for qualification; they are observational for now,
  rather than silently deleting samples from the selector, so cross-vendor runs
  can audit the heuristic before it becomes a policy gate. A
  persisted winner may be imported from a separate hardware-lab artifact with
  `--joint-runtime-profile PATH`; the trainer accepts it only when architecture,
  batch/accumulation/token geometry, ordered device+driver UUIDs, and transport
  backends exactly match the live run. `--lock-joint-runtime-profile` is the
  qualification mode: it freezes all three measured coordinates to the imported
  winner, while the live phase-memory vector remains a safety gate, so A/B runs
  can prove an exact measured schedule without copying runtime metadata into the
  canonical model package or changing checkpoint tensor semantics.
- Cross-vendor hardware qualification is now one command:

  ```powershell
  python tools/qualify_vulkan_hardware.py --device-index 0
  ```

  `HIERARCHOS_VULKAN_MICROPROFILE_DEVICE_INDEX` also makes both retained LM
  microprofiles addressable by physical-device index, so a multi-adapter lab
  host can collect AMD, Intel, or NVIDIA Vulkan measurements one adapter at a
  time. The qualification runner now also runs a mandatory numerical-safety
  leg on that exact selected adapter before performance/parity evidence: the
  finite-preserving clamp boundary derivative, NaN/Inf gradient rejection,
  robust huge-finite global L2 norm, RWKV decay clamp saturation, channel-mix
  key/deep-embed saturation, packed recurrent-state clamp backward, token-front-end
  GELU saturation, ROSA/LTM gate and qproj saturation, and manager control clamps.
  The front-end stress legs deliberately force both saturated and in-range
  coordinates and compare the resulting clamp-aware backward masks against
  PyTorch. `--skip-numerical-safety` exists only for profiling/debugging;
  release qualification should leave it enabled. The runner records the Vulkan UUID/driver fingerprint,
  width-448 LM-backward arm matrix, rows16 projection->stats seam timestamps,
  the full labeled PyTorch/Vulkan training-parity certificate, native Rust
  inference reload, and CUDA inference when CUDA is present. Pass
  `--runtime-profile PATH` to replay a measured profile's tape coordinate through
  the parity oracle, `--require-cuda` on an NVIDIA qualification host, and
  `--trajectory-cycles N` to add repeated Vulkan <-> PyTorch training-state
  handoffs. Raw logs and a versioned JSON certificate are emitted under
  `benchmark_results/` by default.

  The labeled parity certificate now records the exact budgeted execution plan
  used by every optimizer window: sequence microbatch, recurrent checkpoint
  stride, H/L backward segment schedules, H/L kernel geometries, RWKV numerics,
  memory-pressure bucket, and the profile evidence that drove the choice. It
  also audits those coordinates against the persisted training-submission
  profile database. A plan with zero prior exact full-arm observations and
  nonzero runtime-matched factor observations is marked
  `synthesized_from_profiled_factor_evidence=true`. This turns factorization
  from an internal scheduler property into cross-vendor qualification evidence:
  a newly composed plan must still pass the same PyTorch parameter/recurrent
  trajectory and native-Rust/CUDA inference gates as a previously measured arm.

  A real run on the local AMD Radeon Graphics target now closes part of the old
  hardware-evidence gap. At width 448, vocabulary 50,257, and two rows, the
  release LM-backward sweep selected
  `fp16-ce-tape-rows16-dot4-fused-adjoints+dw-vocab8+fused-private-hidden-tile256-wg256`
  at `4.4250 ms`; the older packed/native families were about `65-68 ms`. The
  separate rows16 seam measured `19.037722 ms` for the serial projection/stats
  pair versus `6.965197 ms` for dot4, with `2.98e-7` maximum logit drift and
  `1.49e-7` maximum row-stat drift. A full local qualification certificate also
  passed PyTorch/Vulkan training interchange at `2.94e-6` worst parameter drift,
  recurrent state at `1.31e-6`, and native-Rust inference reload at `3.58e-7`.
  CUDA was not present on this AMD host, so CUDA execution remains an explicit
  NVIDIA-host gate rather than an inferred claim.

  The labeled PyTorch/Vulkan optimizer oracle is no longer hard-coded to three
  updates. `tools/verify_vulkan_labeled_sequence_parity.py --update-count N`
  deterministically extends the masked/TBPTT fixture for sustained optimizer
  trajectories, records per-window Vulkan wall times plus total training time,
  and still closes the final canonical SafeTensors parameter/recurrent-state
  drift check. The default remains three updates for quick CI; larger values are
  intended for the next hundreds/thousands-of-window stability qualification.

  A post-factorization quick qualification on the same AMD adapter also passed
  one complete Vulkan -> PyTorch CPU -> Vulkan state-handoff cycle. Parameters,
  AdamW `exp_avg`, and AdamW `exp_avg_sq` were bit-identical at the backend
  handoff (`max_abs=0` for all three), while the final native-Rust inference
  comparison remained within `1.79e-7`. The corresponding certificate is
  `benchmark_results/vulkan_hardware_qualification.v1.device0.json`; its CUDA
  leg is explicitly marked skipped rather than treated as evidence.
- Example native training invocation:

  ```powershell
  cargo run --release --manifest-path hierarchos-vulkan/Cargo.toml --bin hierarchos-vulkan-train -- `
    --model path/to/model_package `
    --dataset path/to/pretokenized.jsonl `
    --output path/to/vulkan_trained_package `
    --epochs 1 --batch-size 2 --gradient-accumulation-steps 4
  ```

  Each JSONL row requires `input_ids` and may provide same-length `labels`,
  `attention_mask`, and `loss_weights`. If `labels` is omitted it defaults to
  `input_ids`, allowing the canonical Hierarchos next-token shift to match the
  PyTorch labeled-sequence path.
- Raw Vulkan compute through `ash` (no CUDA, DirectML, PyTorch, or vendor compute
  runtime in the Rust trainer).
- Device-local tensor allocations are preferred. On discrete GPUs, minibatch
  uploads and diagnostics use host-visible staging buffers instead of forcing
  the trainable graph into host-mapped memory.
- Staged minibatch copies, dependent compute kernels, and loss readback are
  recorded into one command buffer/queue submission per steady-state step.
- PyTorch-compatible `lm_head.weight` layout `[vocab_size, context_dim]`.
- Vulkan linear forward, mean cross-entropy, LM-head backward, and decoupled
  AdamW update. The production `out_norm -> lm_head -> CE` path is now
  vocabulary-streaming: one 64-lane workgroup per row projects vocabulary
  entries directly from normalized hidden state and the tied LM weight while
  maintaining an online log-sum-exp reduction. It persists only four FP32
  values per row (`max_logit`, inverse exp-sum, target bits, row loss).
  Backward normally regenerates vocabulary tiles from the same FP32 or
  packed-FP16 execution weight and consumes their CE adjoints immediately. The
  autotuned packed-FP16 CE-tape arms retain one bounded `[rows, vocab_size]`
  FP32 logit scratch as a parity/performance anchor, but no longer rewrite that
  scratch into a second full CE-adjoint representation: `W^T` and dW derive the
  adjoint directly from each logit plus four-value row stats at consumption.
  This removes one complete `rows*vocab` read+write pass and makes the remaining
  logit tape the next explicit memory target. The small standalone
  `HierarchosHeadTrainer` intentionally retains the fully materialized path as a
  compatibility/reference implementation.
- Vulkan `out_norm` LayerNorm forward/backward, including gradients flowing back
  toward the recurrent body and AdamW updates for its affine parameters. Norm
  vectors correctly use the Python trainer's no-weight-decay parameter group.
- Tied token-embedding gradient accumulation into the same `lm_head.weight`
  gradient. Repeated token IDs use a portable FP32 atomic add built from Vulkan
  core integer compare-and-swap, avoiding vendor float-atomic extensions.
- Coherent-v9 `SharedTokenAdapter` forward/backward in Vulkan: affine-free
  LayerNorm -> down projection -> SiLU -> up projection -> learned bias. The
  primitive can load/export the real `h_deepembed_adapter`,
  `l_deepembed_adapter`, and `rosa_adapter` SafeTensors prefixes and returns the
  gradient that feeds the tied token embedding. DeepEmbed's explicit no-decay
  optimizer exception and ROSA's matrix-decay/vector-no-decay split are both
  supported and parity-tested.
- Checkpoint-bound Vulkan `nn.Linear` training primitive for the real Hierarchos
  manager/worker topology (`l_feedback_proj`, `h_to_context`, `h_halt_proj`,
  `l_input_proj`, `context_drift_proj`, `l_to_out`) and the same q/in/router
  projection family. Affine and biasless paths preserve PyTorch `[out, in]`
  layout; the project-specific `val_proj` no-decay exception is representable.
- RWKV-v8 FP32 matrix-state forward/backward is Vulkan-native and deterministic:
  `S' = S*decay + (S@-kk) outer (kk*a) + v outer k`, followed by `S'@r`.
  Backward combines gradients from both the recurrent state edge and current
  time-mix readout without float atomics. Parity is exercised at Hierarchos'
  preferred 64x64 head geometry.
- The r/k/v side of the RWKV time-mix cell is now composed into one Vulkan
  forward/backward command graph: `x_norm`/previous-state interpolation,
  receptance/key/value projections, per-head `kk` L2 normalization, in-context
  scaled-key construction, matrix-state recurrence, and the full reverse path
  back through all three projection matrices and their time-mix/key parameters.
  The composed op loads `h_rnn`/`l_rnn` tensors directly from standard
  `model.safetensors` without transposition or renaming, and package-loaded
  execution is bit-identical to the inline-weight Vulkan path in the parity
  harness.
- The RWKV-v8 low-rank `a/w/g` branches are Vulkan-native in both directions:
  `a = sigmoid(a0 + (xa @ a1) @ a2)`,
  `g = sigmoid(xg @ g1) @ g2`, and
  `w = -softplus(-(w0 + tanh(xw @ w1) @ w2)) - 0.5`. The dedicated parameter
  matmul kernels preserve Hierarchos/PyTorch raw `nn.Parameter` layouts
  `[width, rank]` and `[rank, width]`, so the same tensors load without
  transposition in Vulkan, PyTorch/CUDA, and native Rust inference. Backward
  returns gradients for all six low-rank matrices, `a0/w0`, `x_w/x_a/x_g`, and
  the shared normalized/current-previous activation edge. Standard
  `h_rnn`/`l_rnn` SafeTensors loading is parity-tested against direct upload.
- The LN1-adjacent low-rank producer now has a 14-storage-binding fast path
  that tiles `x_norm`/previous interpolation and the three first-stage
  `[width, rank]` matmuls into one 16x16 Vulkan dispatch. `xw/xa/xg` are still
  materialized for the unchanged backward tape, but the matmuls consume the
  just-produced values through workgroup memory instead of reading those three
  scratch tensors back from global memory. Devices with fewer storage bindings
  retain the older interpolation plus three-matmul chain.
- FP16-storage recurrent rematerialization now has a 26-storage-binding full
  producer for low-rank shapes up to 128, including Hierarchos' production
  64/64/96 geometry. One 128-lane workgroup owns a row and keeps the w/a/g rank
  activations resident through `tanh`/`sigmoid`, the second low-rank matmuls,
  decay, and final gate output. The backward tape (`xw/xa/xg`, hidden values,
  `w_tanh`, `w_pre`, `a_pre`, and `g_sigmoid`) is still materialized, so the
  PyTorch/CUDA SafeTensors and autograd contract is unchanged. Set
  `HIERARCHOS_RWKV_LOW_RANK_DISABLE_FULL_FORWARD_FUSION=1` to force the staged
  14-binding path for profiling or regression comparison.
- The production full FP16 low-rank producer now matches its second-stage
  `[rank, width]` inner-product geometry to the physical packed-half layout.
  Each 128-lane workgroup keeps one first-stage rank channel per lane, but each
  lane owns adjacent second-stage output columns and unpacks one `half2` weight
  word into two independent FP32 FMA accumulators. Pairing the first-stage rank
  channels as well was measured and rejected: on the wave64 Radeon target it
  reduced the active 64/64/96 producer lanes to 32/32/48 and regressed the
  kernel despite fewer packed loads. On the width-448, vocab-50257 production
  fixture (`tokens=1`, `sequences=1`, strict numerics, FP16 storage / FP32
  compute), two profiler warmups followed by five measured submissions produced
  9-dispatch low-rank samples of 5.948, 5.162, 5.067, 4.829, and 4.630 ms
  (median 5.067 ms). This keeps dispatch topology unchanged and leaves future
  first-stage work focused on weight sharing that preserves lane occupancy.
- A subgroup-shuffle first-stage packed-word-sharing arm is available for the
  same 128-lane full FP16 producer. For even ranks such as production 64/64/96,
  the even lane issues each W1/A1/G1 u32 load and shuffles the unchanged packed
  bits to its odd partner; both lanes remain active and retain independent FP32
  accumulators. A lane-mapping check falls back to direct loads if subgroup lane
  order is not adjacent in local-invocation order. The experiment is intentionally
  opt-in with `HIERARCHOS_RWKV_LOW_RANK_ENABLE_SUBGROUP_PACKED_SHARE=1`: on the
  wave64 Radeon production fixture (width 448, vocab 50257, `tokens=1`,
  `sequences=1`, FP16 storage / FP32 compute), five warmups followed by fifteen
  measured submissions gave a 9-dispatch low-rank median of 4.973 ms for
  subgroup sharing versus 4.443 ms for the topology-identical portable packed
  loads, an 11.9% regression. Keeping the shader arm allows a future wave32
  NVIDIA/Vulkan measurement without penalizing the current production target.
  A forced-subgroup-32 qualification on the same AMD driver now gives the same
  architectural signal: after five warmups, fifteen measured 9-dispatch samples
  produced a 4.546 ms median for subgroup sharing versus 4.202 ms for portable
  packed loads, an 8.2% kernel regression. Whole optimizer-step timing was noisy
  enough to change sign across repeated runs, so the kernel timestamps are the
  decision metric here. This is a wave32 scheduling proxy, not an NVIDIA result;
  the arm remains opt-in until it wins on a real NVIDIA Vulkan target.
- Persistent gradient bookkeeping batches four independent scratch-gradient
  additions into one compact 1D Vulkan dispatch when eight storage bindings are
  available. Each tensor keeps its own element/workgroup extent inside the
  dispatch, avoiding rectangular over-launch while preserving the per-tensor
  accumulation order used by PyTorch/AdamW parity. Lower-binding devices retain
  the single-gradient accumulation kernel.
- Hierarchos' native 32/32/64 low-rank geometry now has a deeper 19-binding
  fast path that absorbs LN1 into that producer. One 64-lane workgroup owns a
  row: lane 0 accumulates mean/variance in the same scalar order as the portable
  LayerNorm kernel, then the workgroup materializes one canonical `x_norm` plus
  `mean/rstd` while feeding w/a/g interpolation directly into the three
  first-stage projections. The exact `x_norm`, `mean/rstd`, and `xw/xa/xg`
  backward tape remains materialized, so PyTorch/CUDA checkpoint and autograd
  semantics are unchanged. Wider low-rank shapes or devices with fewer than 19
  storage bindings automatically retain the separate LN1 path.
- The low-rank `a/w/g` graph is now fused directly into the recurrent r/k/v +
  matrix-state command graph. `a` and `w` are generated and consumed entirely
  on-device, shared x/previous gradients are accumulated on Vulkan, and the
  fused path completes in one command buffer / queue submission.
- The recurrent output path is Vulkan-native through coherent-v9's real RWKV
  post-mix: per-head `GroupNorm(H, C, eps=64e-5)`, the receptance/scaled-key
  `r_k` bonus, `g` gating, and the standard PyTorch-layout `output.weight`
  projection. Backward joins bonus gradients into r/k/v before projection/key
  backward and routes the gate gradient into the low-rank g branch.
- Coherent-v9 channel-mix is now Vulkan-native as a composable training op:
  `ln2 -> x_k_cm interpolation -> key_cm -> ReLU^2 -> DeepEmbed multiply ->
  value_cm -> residual`. Backward covers LN2 affine parameters, both projection
  matrices, the recurrent channel-mix edge, and the gradient back into the
  DeepEmbed producer. Its record API accepts a GPU DeepEmbed buffer, creating
  the direct composition seam needed to join the already Vulkan-native
  SharedTokenAdapter once that adapter exposes a caller-owned command-buffer
  record path; no tensor-layout change is required.
- SharedTokenAdapter now exposes that caller-owned record path, and a fused
  adapter -> channel-mix training op binds the adapter's `4*C` output directly
  as DeepEmbed and binds channel-mix's DeepEmbed gradient directly into adapter
  backward in the same command buffer. Token features and the final tied-token
  feature gradient remain the only host-facing adapter edges in this slice.
- The raw coherent-v9 cell residual is now composable in one Vulkan submission:
  `x -> LN1 -> full time-mix -> residual -> LN2/channel-mix -> residual`, with
  SharedTokenAdapter DeepEmbed recorded in the same graph. Time-mix recording is
  split into forward/backward halves so channel backward supplies the true
  residual gradient before the recurrent reverse pass; LN1 parameter/input
  gradients and both previous-cache gradients are produced on-device.
- Hierarchos' public recurrent state layouts are Vulkan-native and
  differentiable. Both `legacy-input-cache` (`3 + head_size`) and
  `explicit-output` (`4 + head_size`) layouts are unpacked, finite-preserving
  clamped, repacked, and differentiated on-device, including the clamp mask and
  explicit-output gradient slot.
- Multi-step recurrent scheduling is Vulkan-resident. The TBPTT scheduler keeps
  packed state history on the device, uses reverse forward-recomputation instead
  of retaining the whole activation tape, carries one packed state-gradient
  buffer backward through time, and zeros only that gradient carry at the same
  detach boundaries used by the PyTorch trainer.
- `RwkvTbpttSequenceOp::train_step` now owns persistent AdamW first/second
  moments and sequence gradient accumulators for the full 33-tensor fused cell.
  It follows Hierarchos optimizer grouping v2: 24 RWKV tensors use weight decay;
  LN1/LN2/GroupNorm affine pairs and all three DeepEmbed adapter tensors are
  no-decay. Every reverse-timestep gradient is accumulated before scratch
  buffers are reused, and parameters are stepped once after the full TBPTT
  reverse sweep. `run` remains a weights-readonly parity/diagnostic path.
- TBPTT can now start its DeepEmbed edge from token IDs instead of
  Python-materialized token features. A Vulkan embedding gather reads the
  standard tied `lm_head.weight` `[vocab, context]` matrix directly into the
  per-step token-feature buffers; reverse TBPTT scatters the resulting
  SharedTokenAdapter gradients back into that same matrix with the existing
  portable FP32 CAS atomic add. The tied matrix has persistent AdamW moments,
  advances once per sequence beside the 33 cell tensors, and is exported back
  into the same SafeTensors package without a layout conversion. The legacy
  precomputed-feature entrypoint remains available for isolated parity work.
- Vulkan buffer ownership is now reference-counted rather than raw-handle
  duplicated. `GpuBuffer` clones share one allocation lifetime, and
  `SharedLmHeadParameter` owns the single physical `lm_head.weight` allocation,
  gradient accumulator, AdamW moments, and optimizer step used by H-DeepEmbed,
  L-DeepEmbed, and the output loss. Token-ID TBPTT supports phased tied-gradient
  updates (`BeginAccumulation` / `Accumulate`), while the output trainer can
  preserve those sparse gradients, add the dense cross-entropy gradient, and
  execute one final shared LM-head AdamW step. A regression test proves this
  phased path is numerically identical to one combined tied-gradient update.
- `HierarchosTrainingGraph` is the first model-package-level Vulkan ownership
  root above the individual recurrent schedulers. It constructs H-RWKV,
  L-RWKV, out_norm/loss, and the six learned manager/worker projection seams
  (`l_feedback_proj`, `h_to_context`, `h_halt_proj`, `l_input_proj`,
  `context_drift_proj`, `l_to_out`) against one checkpoint. The projection
  graph registers its six matrices plus four biases in one persistent AdamW
  state and exposes recordable forward/backward seams with immediate named
  gradient accumulation for repeated uses. A real coherent-v9 package
  construction harness verifies all shapes and the three-way tied LM identity.
- `RwkvTbpttSequenceOp` and `HierarchosOutNormHeadTrainer` now have
  caller-owned record/finalize phases underneath their compatibility
  entrypoints. `HierarchosTrainingGraph::train_recurrent_and_loss_one_submit`
  uses those phases to encode H-RWKV, L-RWKV, both sparse DeepEmbed gradients,
  out_norm, dense LM loss, and the current optimizer islands into one
  `ComputeBatch` and exactly one Vulkan queue submission. The old phased
  H -> L -> loss path remains available for parity work. A tiny coherent-v9
  regression starts both paths from identical model packages and currently
  matches every recurrent output/gradient, H/L parameter snapshot, shared
  `lm_head.weight`, out_norm affine parameter, and loss bit-for-bit on the test
  device.
- `RwkvTbpttSequenceOp` also exposes a token-level graph ticket used by
  `HierarchosTrainingGraph::train_projection_coupled_token_one_submit`. This
  path interleaves all six learned manager/worker projections with live H/L
  recurrent buffers: `l_feedback_proj` gathers `state_hidden(L)` directly from
  the packed Vulkan state, H/L residual inputs are projection-produced on the
  device, manager/worker projection gradients are summed into device-resident
  recurrent upstream gradients, and the feedback gradient is scattered back
  into L's packed state before TBPTT finalization. All ten projection tensors
  accumulate into one persistent AdamW registry and step once in the same queue
  submission as both recurrent optimizers and the shared tied LM update. A
  gradient-pure regression (weight decay disabled) proves that all ten tensors
  receive real backward gradients and that the packed L initial state receives
  the feedback gradient.
- Recurrent gradient accumulation is now a separate lifecycle from recurrent
  optimizer advancement. Graph tickets can reset or preserve the persistent
  cell-gradient accumulator, completed shadow branches can scatter their tied
  DeepEmbed gradients without stepping or scheduling host readback, and one
  explicit recurrent AdamW operation advances all accumulated branches. The
  public `train_step_with_token_ids_accumulated_branches` harness proves this
  contract against PyTorch: two recurrent branches perform separate backwards,
  then both the 33-tensor RWKV cell and tied `lm_head.weight` advance exactly
  once. On the current AMD Vulkan device the verifier's worst parameter error is
  about `1.1e-6`, while the final branch recurrent-state error is below `2.4e-7`.
  Existing projection-coupled, loss-coupled, and H/L/out_norm fusion regressions
  remain passing after the lifecycle split.
- Persistent AdamW registries have a portable named SafeTensors companion-state
  format (current `hierarchos-vulkan-adamw-v3`, with v1/v2 read compatibility) and
  the training-package manifest is now `hierarchos-vulkan-training-state-v6`
  (with v1-v5 read compatibility). v5 made the checkpoint's FP32-master versus
  derived execution-mirror semantics explicit. v6 also serializes the tied
  parameter topology: canonical `lm_head.weight` and PyTorch's
  `tok_emb.weight` alias must resolve to one trainable master. The PyTorch bridge
  validates that object identity before hydrating AdamW or pending gradients, so
  a backend cannot silently resume with two independently optimized copies. The
  v6 loaders/exporters also verify the physical SafeTensors payload for every
  optimizer-bound slot: the declared master must actually be stored as F32 on
  both the native Rust/Vulkan and PyTorch bridge boundaries. F16/BF16 remains
  valid only as source inference storage or as a derived runtime execution
  mirror, never as the authoritative portable training master. The alternating
  Vulkan/PyTorch trajectory verifier can also choose a different precision for
  the returning Vulkan leg, explicitly exercising FP32-producer ->
  FP16-storage-consumer handoff: the destination graph rebuilds its compact
  mirrors locally from the portable masters and writes its own runtime precision
  on the next checkpoint.
  AdamW v3 closes a second optimizer-boundary ambiguity by serializing the
  canonical per-slot `decay`/`no-decay` topology alongside independent slot
  steps. Native Vulkan validates that topology against its live optimizer
  registry, while the PyTorch CPU/CUDA bridge rejects a declared no-decay slot
  placed in a nonzero-weight-decay group (and rejects unsupported
  `maximize=True`) before optimizer state is hydrated. Legacy v1/v2 companions
  remain readable with unknown decay topology and are upgraded on the next
  backend-authored snapshot.
  Full raw-token graphs export `model.safetensors`, `optimizer.safetensors`, and
  `training_state.json`; an optimizer-boundary package stops there, while a
  package captured inside a multi-microbatch accumulation window also contains
  `gradients.safetensors`. That gradient companion is the canonical 95-slot FP32
  registry keyed by ordinary model tensor names. v4 closes the last
  backend-specific hole in that registry: an active PyTorch-TBPTT
  `val_proj.weight` gradient is serialized with the LTM alignment objective
  weight already applied, while Vulkan removes that factor when restoring its
  own deferred internal accumulator. A v4 open window can therefore move
  Vulkan -> PyTorch CPU/CUDA or Vulkan -> Vulkan without changing optimizer
  semantics; the bridge still upgrades legacy v1-v3 active `val_proj` gradients
  on read. The manifest records consumed
  and optional target normalization mass plus the physical tied-LM gradient
  topology, allowing a live window to restore into either the shared tied
  `lm_head.weight` allocation or the optimizer staging allocation without
  changing the portable file. `tools/vulkan_optimizer_bridge.py` consumes the
  same package into `torch.optim.AdamW` and `Parameter.grad` on CPU/CUDA,
  resolving canonical `lm_head.weight` onto the declared tied
  `tok_emb.weight` object. Exact weighted-token continuation maps Vulkan
  `mean-by-supervision-weight` to PyTorch `weighted-token`; unsupported
  normalization contracts fail closed. Optimizer/training state remains
  separate from ordinary model tensors, so the same model package continues to
  serve PyTorch CPU/CUDA and `hierarchos-inference`. `val_proj.weight`, formerly
  the last full-training registry gap, is the 95th canonical slot: the optional
  LTM value-alignment objective runs on Vulkan with PyTorch's detached target/
  readout semantics and the same explicit no-decay optimizer policy.
- `tools/verify_vulkan_mid_window_cross_backend.py` now exercises that adaptive
  LTM controller instead of disabling it for the backend handoff. On the AMD
  Vulkan host used for development, the canonical open gradient registry matches
  a fresh PyTorch weighted-token numerator within `7.87e-6`, the restored
  controller EMA/readiness state within `1.57e-7`, current-format Vulkan
  open-checkpoint self-resume is bit-identical to uninterrupted Vulkan, the
  Vulkan -> PyTorch optimizer-window closure differs by at most `4.77e-7`, and
  returned native inference differs from PyTorch CPU by `3.58e-7`. The same bridge selects CUDA
  automatically when an NVIDIA device is available; `--require-cuda` keeps a
  real CUDA execution as an explicit hardware-gated verification rather than a
  requirement for AMD-side development.
- The PyTorch-shaped labeled raw-token frontend can now keep that canonical
  full-model gradient window open across independent Vulkan queue submissions.
  `BeginAccumulation`, `Accumulate`, and `FinishAccumulation` preserve masked
  rows, weighted CE/z-loss, historical TBPTT recurrent/context detaches, and
  auxiliary objectives while delaying normalization and AdamW until the closing
  microbatch. `tools/verify_vulkan_labeled_sequence_parity.py
  --accumulation-steps 3` exercises three masked/TBPTT microbatches as three
  Vulkan submissions but one optimizer step. On the current host it changed 93
  PyTorch state tensors with `3.54e-6` worst trained-parameter drift and reloaded
  the Vulkan-trained package through native Rust inference with `5.96e-7`
  maximum logit drift. CUDA inference was unavailable on that host, so NVIDIA
  execution remains covered by the unchanged portable SafeTensors contract and
  the CUDA-required verifier on a CUDA machine.
- Cross-logical-device gradient reduction now has a persistent zero-host-copy
  backend for compatible Vulkan devices. The primary graph retains two imported
  opaque external-memory windows plus forward/return binary semaphore pairs
  across optimizer windows, so handle export/import and semaphore construction
  are paid on the cold reduction rather than every training step. Each slot
  transfers ownership for the full persistent allocation even when its final
  tensor chunk is shorter, while copies and reductions remain bounded to the
  live chunk. `HierarchosGradientStreamStats::persistent_transport_reused` and
  the trainer's `gradient_stream_persistent_reuses` counter make the amortized
  route observable. On the current AMD Radeon target,
  `tools/verify_vulkan_raw_token_tape_parity.py` selected
  `opaque-external-memory`, moved all 52,685 values across 95 canonical gradient
  tensors with zero host gradient bytes, and proved that the second independent
  logical-device reduction reused the same two transport slots. The full
  raw-token, sparse/dense, exact-TBPTT, open-checkpoint, optimizer, and control
  parity suite remained passing; the portable SafeTensors/PyTorch/CUDA model ABI
  is unchanged.
- Closed-step replica broadcast now reuses that Vulkan transport contract for
  canonical parameters plus both AdamW moment planes. Compatible device-group
  or opaque-external pairs stream bounded chunks directly into the replica and
  refresh compact parameter mirrors locally; only optimizer counters,
  curriculum position, and LTM-controller state remain on the tiny CPU control
  plane. Unsupported pairs still share one canonical portable host snapshot as
  a correctness fallback. The trainer reports `replica_state_transport` plus a
  `replica_state_stream` telemetry object so direct chunks, values, slot count,
  persistent reuse, bounded host/device bytes, and host fallbacks are visible.
  On the current AMD Radeon target the raw-token parity harness broadcast all
  three planes for 95 tensors as 285 opaque-external-memory chunks / 158,055
  FP32 values with two slots, zero host payload, bit-exact replica parameters
  and optimizer state, and persistent-slot reuse on the immediate second
  broadcast. The trainer now captures an immutable transport-only source view
  whose cloned Vulkan buffers carry an optimizer-generation read lease, so the
  non-`Sync` TBPTT graph itself never crosses worker threads. One long-lived
  execution worker now owns each replica graph for the trainer lifetime;
  optimizer windows enqueue lightweight broadcast and compute jobs into those
  resident lanes. All peers drain the same closed generation concurrently, and
  host-fallback peers lazily share one portable snapshot through a single
  `OnceLock`. Primary next-window forward/backward is launched before the
  preceding ticket is retired, while each replica's FIFO lane advances from its
  own broadcast directly into its next-window forward/backward without waiting
  for the slowest peer. Faster replicas may compute ahead, and completed detached
  gradient sources are reduced as soon as the next replica index is available,
  so ordered primary-side DMA/reduction can overlap later replica compute.
  Canonical AdamW parameter/moment mutation is now a range wavefront rather than
  a whole-generation join. Device-group replicas with timeline semaphores and an
  independent queue lane preallocate their two peer-copy slots once, retain one
  monotonic retirement timeline per replica, and reserve every upcoming range
  value at broadcast launch. The generation guard can therefore hand those
  future waits to AdamW before the worker reaches the range. Source copies keep
  the primary physical-device mask but execute on the replica's queue lane, so a
  prequeued AdamW wait on the primary lane cannot block the command that will
  signal it. The final `exp_avg_sq` source copy for each range signals both the
  ordinary transport handoff and the reserved retirement value. Ready runs with
  predeclared waits stay bounded to the optimizer wavefront coalescing floor
  instead of collapsing to the final model-wide timeline value. AdamW now keeps
  those bounded runs in flight: the host submits each primary-queue batch behind
  its future timeline dependencies without waiting its fence, queues the optional
  lower-dtype mirror refresh behind the same FIFO, then drains from the newest
  fence backward only after the whole wavefront has been issued. That removes the
  per-run host fence round trip as well as the early-arrival Condvar edge on
  qualified device-group lanes. Timeline-capable submissions also detach their
  command buffer, descriptor pools, local upload chunks, and timestamp query
  pools into a device-level retirement arena keyed by the queue-completion
  timeline. Persistent device-group broadcast slots consequently retain only
  peer memory plus handoff/return timeline state; they contain no submitted-batch
  owners and no unused destination scratch allocation. The terminal mirror
  refresh retains only its lightweight timeline dependency plus the captured
  source generation, so broadcast -> mirror-refresh -> next compute can remain
  queue-resident while the device reclaims transient resources from queue
  progress. Drivers without a second queue lane, timeline
  semaphores, or proven peer transport retain the previous dynamically
  published/host-safe retirement path. The portable SafeTensors/PyTorch/CUDA
  state ABI is unchanged.
  This also removes serialized peer fanout, the broadcast-vs-primary phase
  barrier, and the all-replicas broadcast-to-compute barrier without pretending
  the recurrent training graph is `Sync`.
  `replica_state_stream.broadcast_scheduler`, `transient_submission_retirement`,
  `persistent_device_group_slots`, `worker_threads`, `worker_jobs`, `worker_reuses`, `compute_handoffs`,
  `optimizer_predeclared_retirement_timeline_lanes`, the overlap flags, and
  `gradient_reduction_order` expose the persistent pipeline in trainer telemetry.
- Lower-dtype model storage is now qualified together with lower-memory Vulkan
  execution storage instead of treating those boundaries independently.
  `tools/verify_vulkan_low_dtype_training_qualification.py` runs both FP16 and
  BF16 PyTorch-readable SafeTensors checkpoints through
  `fp16-storage-parity`, three masked historical-TBPTT AdamW updates, package
  export, and native Rust inference. The optimizer epsilon is an explicit part
  of this portable contract: the qualified mixed-storage setting is `1e-6`.
  With the historical `1e-8` default, individually harmless backend gradient
  roundoff in near-zero Adam slots can be magnified by second-moment
  normalization after multiple updates; the stress fixture reached about
  `4.61e-5` H-state drift despite sub-`5e-6` parameter drift. At `1e-6`, the
  same FP16-checkpoint/FP16-storage parity arm drops to about `1.97e-6`
  recurrent-state drift and `1.43e-6` parameter drift, while native inference
  remains within `3.58e-7` of PyTorch CPU. The BF16-checkpoint case passes at
  about `7.03e-6` recurrent-state drift, `9.54e-7` parameter drift, and
  `4.77e-7` native-inference drift. `--adamw-eps` is now a first-class ordinary
  trainer option and exact-resume identity field so PyTorch CPU/CUDA and Vulkan
  continuations cannot silently disagree on this stabilizer. Existing training
  keeps the legacy `1e-8` default unless the portable mixed-precision setting is
  selected explicitly. Add `--require-cuda` to the qualification command on an
  NVIDIA host to make the CUDA inference leg mandatory.
- v3/v4 training packages can additionally carry a pickle-free host replay sidecar:
  `training_replay.json` contains typed replay topology and
  `training_replay.safetensors` contains tensor/RNG payloads. The sidecar covers
  the PyTorch DataLoader/sampler cursor, Python/NumPy/PyTorch RNG state,
  scheduler/scaler state, and recurrent/LTM/ROSA carrier state. This is the
  handoff boundary used by the ordinary trainer when
  `--resume-from-ckpt <vulkan-package-dir>` is supplied. Cross-backend resume
  remains strict about data/tokenizer identity, architecture, objective,
  optimizer grouping, accumulation geometry, gradient transforms, and LR
  schedules while permitting Vulkan-vs-PyTorch execution-only differences such
  as AMP, compilation, activation checkpointing, and device-specific loss
  chunking. A Vulkan package without the replay sidecar is intentionally treated
  as weights-only by `--model-path`, not as an exact training resume.
  `HierarchosTrainingGraph::export_training_checkpoint_package_with_replay`
  now emits that sidecar directly from Rust, including U8 generator-state and
  FP32 recurrent-state tensors, and updates the v3 manifest only after the
  companion files are present. The raw-token checkpoint smoke immediately reads
  the Rust-written sidecar through the Python bridge, so native Vulkan training
  no longer needs an out-of-band Python mutation step to publish portable replay
  state.
- Model-level stochasticity now has a backend-neutral counter ABI instead of a
  backend RNG blob. `philox4x32-10-word-v1` maps a 64-bit seed plus an absolute
  32-bit word cursor onto Random123-compatible Philox4x32-10 output. Each
  stochastic graph operation reserves an immutable word range once; activation
  rematerialization reuses that reservation rather than consuming the cursor a
  second time. Canonical dropout compares each word against an integer
  `floor(p * 2^32)` threshold, so the mask decision is identical in the Rust
  reference, Vulkan shader, and the PyTorch CPU/CUDA reference path in
  `hierarchos/training/stochastic.py`. The typed execution policy serializes the
  canonical `{algorithm, seed, next_word}` state and therefore needs no
  Python/NumPy/Torch RNG payload for these operations. The ordinary PEFT LoRA
  fine-tune path now replaces active adapter `nn.Dropout` modules with that
  canonical reservation source. Non-reentrant PyTorch activation checkpointing
  records each original forward's immutable reservation tape and replays the
  tape during rematerialization without advancing the model cursor or restoring
  a CPU/CUDA RNG state. Exact resumes from older backend-native stochastic
  checkpoints deliberately retain their historical RNG source rather than
  silently changing trajectory semantics mid-run.
- Native mixed-precision overflow detection now stays on Vulkan until a single
  four-byte result is read back. The persistent AdamW registry scans the exact
  accumulated gradients that the next update would consume, including the
  direct tied `lm_head.weight` override. `HierarchosLossScalingState` implements
  dynamic scale backoff/growth and unscale-factor step semantics, while
  `discard_full_model_accumulation_after_overflow` clears a failed accumulation
  window without advancing AdamW or per-slot step counters. Finite windows now
  close through the same canonical full-model optimizer recorder as ordinary
  training: outer token/supervision normalization and loss unscale are composed
  on-device, then AdamW is recorded into the same Vulkan command buffer without
  materializing gradients on the host. The dynamic close commits scaler state
  only after the matching device clear/step succeeds and synchronizes output/LTM
  bookkeeping with the externally closed optimizer step. The local AMD parity
  harness reports zero parameter and AdamW-moment drift between the ordinary
  close and the deferred dynamic finite-window close.
- The portable data-stream cursor is executable native state rather than only
  manifest metadata. Rust can materialize and advance both `epoch-shuffle` and
  `length-grouped-batch` orders with the same SplitMix64/Fisher-Yates recipe as
  Python, including stable length sorting, preserve-order bucket shuffling,
  mid-epoch batch cursors, and epoch rollover. Regression vectors are checked
  against the Python samplers.
- The remaining repeated-destination FP32 training reductions no longer depend
  on Vulkan workgroup scheduling order or vendor float atomics. LTM key-gradient
  backward is destination-segmented: each `(slot, key_column)` owns one
  monotonically source-ordered pass over selected pairs, replacing the previous
  quadratic elected-writer scan with work linear in sequence rows for fixed LTM
  geometry. Repeated tied-embedding gradients now take a deterministic
  reduce-by-key path at every supported batch size. Batches up to 1,024 tokens
  use one 256-lane fixed-network bitonic sort. Larger batches use a stable
  multi-workgroup 4-bit LSD radix pipeline: each 256-position block emits a
  16-bin histogram, one compact prefix pass converts block counts to absolute
  bucket offsets, and each block performs a source-stable scatter. Both sort
  paths preserve original source positions, then the same segmented gradient
  kernel folds each token row exactly once in FP32 source order. The historical
  quadratic elected-writer shader is retained only as a forensic parity anchor;
  production training no longer falls back to it. ROSA gate backward continues
  to write per-row contributions followed by its fixed row-order reduction.
- LTM top-k election is now a segmented Vulkan reduction rather than one serial
  invocation per token row. A 64-lane workgroup scans disjoint slot segments,
  keeps lane-local top-k lists, and performs a deterministic workgroup K-way
  merge with the same strict-score/lower-slot tie semantics as the previous
  shader. `tools/verify_vulkan_token_memory_frontend_parity.py` continues to
  report exact `torch.topk` indices on the local AMD Vulkan device.
- The output trainer now has a split device-resident backward/optimizer phase.
  `HierarchosTrainingGraph::train_loss_coupled_token_one_submit` uses it to
  execute `l_to_out -> enc residual -> out_norm -> tied lm_head -> cross
  entropy` around the live L ticket. Cross-entropy's Vulkan `grad_input` is the
  actual upstream gradient for `l_to_out`; there is no host-supplied
  `l_to_out_grad` in this entrypoint. The dense LM gradient is recorded before
  recurrent backward, while the tied LM AdamW step is delayed until both H and
  L DeepEmbed branches have scattered their sparse gradients into the same
  accumulator. A weight-decay-zero verifier proves all ten projection tensors,
  including `l_to_out`, move from real graph gradients in one queue submission.
- Coherent-v9 control math now has first-class Vulkan forward/backward kernels.
  Hard ACT computes clamped sigmoid hazards, cumulative-CDF hard selection with
  `min_h_steps`, selected-output gather/scatter, and the exact differentiable
  quantile-depth surrogate used by `hard_act_depth_straight_through`. Context
  control computes the clamped stride LERP and `[enc, sliding_context + drift]`
  worker input with reverse gradients to `enc`, previous/target context, and
  drift. Drift seed/update kernels implement `tanh`, finite clamp, optional L2
  norm clamp, and the full reverse derivative for both state-derived initial
  drift and recurrent `current + tanh(delta) * drift_delta_scale`. A direct
  PyTorch parity harness holds every tested forward/backward edge within
  `6e-8` max absolute error on the current AMD Vulkan device. Hard ACT and
  context/drift now also expose caller-owned recording interfaces, so these
  primitives can be embedded into the same command buffer as recurrent tickets
  without an intermediate host synchronization.
- Forked worker refinement is now live in the full graph rather than only in an
  isolated recurrence harness. `train_worker_refinement_loss_one_submit`
  derives the initial drift from the real L state, runs a device-resident shadow
  L chain with repeated `l_input_proj -> L -> context_drift_proj -> drift`
  transitions, then restarts the committed L transition from the original real
  state and connects it to `l_to_out -> out_norm -> tied lm_head -> CE`. Reverse
  mode replays the drift chain, accumulates all repeated projection uses, merges
  the shadow and committed real-state adjoints, and advances H, L, projection,
  and output optimizer islands once in one Vulkan queue submission. Row-local
  convergence is now part of that same static command graph: accepted candidate
  L states/drifts are frozen per row after `mean(abs(drift_delta)) < l_conv_atol`,
  while peer rows continue refining. The coherent-v9 mean-square commitment
  hinge is accumulated only over active steps and has a device-resident backward
  edge into every accepted candidate drift. An adversarial two-row verifier
  chooses `l_conv_atol` between the rows' first-step drift magnitudes, proving
  mixed convergence/freeze semantics while backpropagating a nonzero commitment
  gradient. It matches the same exported PyTorch checkpoint with `6.0e-8` loss
  error, `1.43e-6` worst graph-value error, and `6.26e-8` worst updated-parameter
  error on the current AMD Vulkan device.
- The first device-resident manager hard-ACT candidate graph is also live.
  Vulkan now runs the real H-RWKV transition, copies its packed state directly
  into an isolated shadow H workspace, executes later ponder candidates without
  a host state seam, stacks candidate outputs/states/halt logits with offset
  buffer copies, and feeds the existing hard-ACT selector. The selected H output
  and selected packed state are gathered from the same per-row index in one queue
  submission. A four-candidate PyTorch verifier matches halt probabilities to
  `2.98e-8`, selected output to `1.49e-8`, and selected packed state to `2.39e-7`
  max absolute error on the current AMD Vulkan device.
- Hard-ACT is now connected to the full one-submit training graph in reverse as
  well as forward. The selected H-output adjoint is scattered back to the winning
  candidate per row, the straight-through hard-depth surrogate produces an
  adjoint for every halt logit, and every shadow H-RWKV candidate is traversed in
  reverse before the real H step. The same selected index also scatters a packed
  H-state adjoint, so a future token can backpropagate through hard-selected
  manager state commitment. Shadow and real H residual-input gradients are summed
  before the shared `enc + l_feedback_proj(...)` edge.
- The hard-ACT depth reverse edge is now row-wise rather than logit-wise. One
  invocation reconstructs the final survival prefix once, walks the differentiable
  CDF terms backward with a suffix derivative, and emits every halt-logit adjoint
  for that row. This removes the old nested prefix rebuild (`O(h_steps^3)` work
  across a batch) in favor of `O(h_steps)` work per row, uses no additional
  scratch allocation, and preserves the same PyTorch straight-through depth and
  checkpoint contracts.
- The full hard-ACT/worker/loss path now places explicit Vulkan `_finite_clamp`
  forward/backward edges at the manager residual, every manager candidate output,
  the selected `h_to_context` result, every `l_input_proj` result, every shadow and
  committed L output, and the final `enc + l_to_out(...)` residual. The parity
  harness deliberately lowers `activation_clamp` to `0.12`, verifies that the
  manager-input clamp actually saturates, and still matches PyTorch with
  `1.42e-7` loss error, `1.70e-6` worst graph-value error, and `4.02e-7` worst
  updated-parameter error on the current AMD Vulkan device.
- The full hard-ACT/worker/loss path now also owns one named AdamW registry for
  the complete trainable slice. H-RWKV, H DeepEmbed, L-RWKV, L DeepEmbed, all
  ten manager/worker projection tensors, `out_norm.{weight,bias}`, and the one
  tied `lm_head.weight` are registered by their exact PyTorch/SafeTensors model
  names and advance at one shared global step. Reused recurrent and projection
  scratch gradients are accumulated into that registry immediately after the
  backward kernel that produced them. The tied LM path now keeps one shared
  gradient buffer live across `BeginAccumulation -> Accumulate ->
  FinishAccumulation`. LM dW kernels overwrite it for the first reverse in the
  window and add in-place on later reverses; the final AdamW step consumes that
  same named gradient directly. Multi-token/microbatch training therefore no
  longer needs either the LM-sized dW scratch-to-shared sweep or the
  shared-to-canonical optimizer sweep on the default path. Set
  `HIERARCHOS_VULKAN_LM_FORCE_DENSE_GRAD_STAGING=1` and/or
  `HIERARCHOS_VULKAN_LM_FORCE_CANONICAL_GRAD_STAGING=1` to restore those
  historical staging boundaries for A/B diagnostics.
  The current coherent-v9 tiny-model verifier covers all 79 registered tensors,
  rejects duplicate/missing names, runs the whole update in one Vulkan queue
  submission, and retains the same `4.02e-7` worst updated-parameter error.
- The canonical 79-tensor registry now has an explicit microbatch lifecycle:
  `BeginAccumulation`, `Accumulate`, and `FinishAccumulation` preserve summed
  gradients across complete worker/loss submissions and advance AdamW only on
  the final submission. A two-microbatch full-graph PyTorch reference now
  passes on the current AMD Vulkan device after the persistent tied-gradient
  and rows16-dot4 migrations with `5.21e-7` CE-loss error, `1.31e-6` worst
  graph-value error, and `6.55e-7` worst updated-parameter error while the
  optimizer global step advances exactly once. That run selected
  `fp16-ce-tape-rows16-dot4+dw-vocab8` automatically. The historical
  one-token API remains a `Step` compatibility wrapper over the same registry.
  For same-binary profiling, set
  `HIERARCHOS_VULKAN_LM_FORCE_CANONICAL_GRAD_STAGING=1` to restore the former
  shared-gradient -> canonical-gradient copy. On the current AMD development
  GPU, a production-width 448 / vocabulary 50,257 / one-token packed-FP16 run
  measured `59.74465 ms` with canonical staging versus `56.57435 ms` with the
  direct AdamW gradient source, reducing step time by 5.31% and increasing
  throughput by 5.60% with the same rows16 fused-adjoint LM arm and WG64/WG64
  recurrent geometry.
- Primary recurrent tickets can now start directly from a device-resident
  packed state, matching the shadow-workspace capability. The committed L
  restart in the full worker graph uses this path already, copying the original
  real L state device-to-device rather than re-uploading the host slice. This is
  the first concrete state-handoff seam needed by the outer multi-token owner.
- `HierarchosSequenceStateArena` now gives that outer owner a stable rolling
  boundary for committed H/L packed state plus independent H/L reverse
  adjoints. After a full token, the arena captures the hard-ACT-selected H
  state, committed L state, and both initial-state adjoints entirely with Vulkan
  device copies; host snapshots exist only for parity/debugging. The full-graph
  verifier checks all four captured tensors against PyTorch while also running
  the two-microbatch accumulated update.
- `HierarchosTokenTape` now owns a real multi-token training horizon. It keeps
  committed H/L state slots, hard-ACT choices, worker effective-step data, and
  row-active trajectories on-device. During reverse traversal token `t + 1`
  writes its initial packed-state adjoints into a Vulkan carry that token `t`
  consumes directly as its future-state adjoints. The two-token regression
  matches the former explicit host handoff exactly for states, adjoints, losses,
  control decisions, and all 79 AdamW moment tensors while stepping once.
- `HierarchosTokenTapeArena` lifts that ownership boundary across independent
  sequences. Every slot has private H/L state, hard-control checkpoints, loss
  readbacks, and reverse-adjoint carries, while all slots feed the same canonical
  79-tensor gradient registry. A two-sequence/four-token regression records all
  forward checkpoints, all reverse rematerializations, total-token gradient
  normalization, and one AdamW update in a single Vulkan queue submission. On
  the current AMD Vulkan device it is bit-identical to the explicit host-state
  accumulation reference for states, adjoints, losses, controls, and optimizer
  moments.
- `ComputeBatch` now allocates descriptor sets from chunked pools and caches
  immutable descriptor bindings for the lifetime of a command batch. The first
  recorded two-sequence/four-token regression used 9,155 compute dispatches and
  625 unique descriptor sets across 5 pools, instead of the historical one
  pool/set allocation per dispatch. The current regression is down to 8,007
  dispatches, 585 unique descriptor sets, and 3,977 shader dependency barriers
  after recurrent-adjoint/output-gradient fusion, affine linear+bias fusion,
  three-way value-gradient fan-in fusion, grouped r/k/v projection kernels, and
  residual-output projection fusion. Batch uploads are always recorded as ordered
  staging copies, including when device-local memory is host-visible; those
  copies now suballocate from packed host-visible arena chunks rather than
  allocating one staging buffer per upload. Stable pipeline and push-constant
  state is retained across adjacent compatible dispatches. On the current AMD
  regression, 120 uploads / 45,728 bytes fit in one staging arena buffer, while
  the 8,007 dispatches require 5,665 pipeline binds and 5,978 push-constant
  writes. Descriptor-set binds remain one per dispatch and are still an explicit
  state-setting target. The ordered-copy rule still prevents later token uploads
  from overwriting not-yet-executed replay inputs on integrated GPUs.
- SafeTensors import/export and full-package tensor replacement.
- Vulkan descriptor-set and pipeline layouts are now interned per device by
  structural binding ABI plus push-constant range, so compatible kernels share
  the exact layout handles. A regression verifies reuse across the linear
  forward/input-gradient pair.
- Channel-mix TBPTT now fuses the external normalized-gradient add with both
  LayerNorm backward kernels. The fused regression matches the unfused result
  within `1e-6` and removes two dispatches plus one shader dependency barrier
  from that historical three-dispatch chain.
- Packed-cell backward now fuses the public packed-state output adjoint with the
  caller's cell-output gradient. The dedicated Vulkan kernel preserves the same
  clamp derivative and packed-state layout while removing the following
  `vector_add` dispatch from every recurrent backward that carries that edge.
- Graph projections, standalone affine trainers, and SharedTokenAdapter up
  projections can execute PyTorch-layout `[out, in]` linear+bias in one Vulkan
  dispatch. Bias gradients remain separate in backward, so checkpoint tensor
  names/layouts and optimizer semantics are unchanged.
- Full time-mix backward uses a three-input vector-add kernel when the value
  branch has both post-mix and external recurrent-state gradients, collapsing a
  two-dispatch fan-in chain without changing accumulation order at any other
  branch.
- RWKV time-mix now evaluates the same-shape r/k/v projection trio with one
  Vulkan dispatch in forward, one in input-gradient backward, and one in
  weight-gradient backward. Each branch preserves the generic linear kernel's
  serial FP32 FMA order and exact PyTorch `[out, in]` tensor layout. On the
  two-sequence/four-token regression this removes 320 dispatches without changing
  any PyTorch-parity observable.
- The common `head_size <= 64` time-mix forward path now goes one level deeper:
  a head-owned Vulkan kernel tiles `x_norm`/previous interpolation through shared
  memory, evaluates the three unchanged PyTorch-layout r/k/v matrices, performs
  key normalization/in-context scaling, and advances the recurrent matrix state
  in one dispatch. Existing xr/xk/xv/r/k/v scratch tensors are still materialized
  for the unchanged backward kernels, so SafeTensors/CUDA interchange semantics
  do not move. The fast path is enabled only when the Vulkan device exposes the
  24 storage-buffer bindings it needs; otherwise the older portable chain remains
  active. On the two-sequence/four-token trace, the former 288 forward dispatch
  occurrences collapse to 96 and remove 192 shader dependency barriers as well.
- The post-mix output projection and channel-mix value projection can now fold
  their row-local residual into the final linear store. The channel-mix scratch
  branch buffer is gone, and the packed-cell path no longer materializes the
  post-mix output just to add it back to the residual. This removes another 192
  dispatches and 192 shader dependency barriers from the same regression while
  leaving standalone post-mix behavior and checkpoint semantics unchanged.
- The profiler-selected post-mix follow-on now fuses `ln_x` group normalization
  with the RWKV bonus/gate producer on devices exposing at least 13 storage
  bindings. It still materializes normalized activations, mean/rstd, the bonus
  scalar, and the gated vector for the unchanged backward graph. The common
  `width <= 512` path goes further when 16 storage bindings and a 256-lane work
  group are available: one row-owned kernel keeps the gated row in shared memory
  and folds the PyTorch-layout output projection plus residual into the same
  dispatch. Smaller descriptor limits and wider models retain the staged
  portable path. Both A/B regressions match the legacy Vulkan chain within
  `1e-6`, so SafeTensors/PyTorch/CUDA checkpoint semantics remain unchanged.
- Channel mix now has the matching width-specialized fusion ladder. With at
  least 13 storage bindings, one row-owned kernel preserves the serial `ln2`
  reduction, interpolation, and all backward tapes while folding the
  PyTorch-layout key projection and exact clamped `ReLU^2 * DeepEmbed`
  activation into one dispatch. With at least 15 bindings, that same workgroup
  also stages the `4*C` FFN row in shared memory and evaluates the value
  projection plus residual, collapsing the complete four-dispatch channel-mix
  forward into one. The `width <= 512`/256-lane gate keeps this specialization
  bounded, while the older producer and generic-linear paths remain automatic
  fallbacks. Direct four-stage A/B parity is within `1e-6`.
- Packed-cell scheduling now owns a deeper state-boundary fusion without moving
  public state layout logic into channel mix. On devices exposing at least 19
  storage-buffer bindings, the existing full channel-mix arithmetic is recorded
  by a packed-cell kernel that also writes the clamped public recurrent state in
  the same dispatch. LN1 cache, LN2 cache, `v_first`, optional explicit output,
  and matrix-state slots keep the exact existing packed ABI; the matrix state is
  still produced and owned by time mix and enters the fused scheduler kernel
  read-only. Lower-binding-count devices retain channel-mix followed by
  `rwkv_state_pack` automatically.
- The cell backward compiler now folds the final residual-gradient add into LN1
  input backward. Channel mix still materializes its historical
  `grad_output + grad_ln2_input` buffer because time-mix backward consumes that
  exact recurrent upstream edge; LN1 backward then adds that buffer in its final
  store instead of scheduling a second `vector_add`. This removes the targeted
  cell-level two-add fan-in without changing time-mix gradient ordering or
  standalone channel-mix behavior.
- Final `out_norm -> lm_head` and SharedTokenAdapter down-projection seams now
  have storage-binding-gated fusion paths. The output head folds LayerNorm into
  the tied PyTorch-layout LM projection while still materializing normalized
  hidden/mean/rstd for the unchanged backward pass. Adapters go one step deeper
  and fold affine-free LayerNorm, the down projection, and SiLU into one dispatch
  while preserving normalized, down-preactivation, and hidden tapes. Devices
  with at least 8/9 storage-buffer bindings take the corresponding fast paths;
  smaller descriptor limits retain the portable legacy kernels. On the
  two-sequence/four-token profiler regression the remaining 96-count
  `layer_norm_forward -> linear_forward` adapter edge disappears, then the new
  fused-kernel -> SiLU edge disappears as well: dispatches fall from 7,299 to
  7,107 and shader dependency barriers from 3,269 to 3,077 with token-tape
  state, adjoint, loss, optimizer, and control parity all exact.
- The common coherent-v9 `token_adapter_rank <= 64` path now fuses the adapter
  up projection into that same forward kernel. One workgroup owns a token row,
  keeps the rank-sized SiLU vector in shared memory, and fans the PyTorch-layout
  `[output_dim, rank]` up matrix across a 256- or 512-lane width specialization.
  The fast path requires 12 storage-buffer bindings and is capped at
  `output_dim <= 512`; wider adapters deliberately retain the separately tiled
  `linear_bias_forward` path so dispatch-count fusion cannot serialize a large
  hidden-width projection. Normalized, mean/rstd, down-preactivation, and hidden
  tapes are still materialized for the unchanged backward kernels, so the
  SafeTensors/PyTorch/CUDA tensor ABI is unchanged. On the AMD Radeon Graphics
  development device the `448 -> 64 -> 448` A/B benchmark is bit-identical to
  the legacy Vulkan path, removes one dispatch and one shader barrier per full
  adapter training step, and measured 1.077x faster over eight
  forward/backward/AdamW steps. A `448 -> 64 -> 768` control takes the legacy
  path exactly. In the two-sequence/four-token full graph this removes all 96
  former adapter up-projection dispatches: 7,107 -> 7,011 dispatches,
  3,077 -> 3,013 shader barriers, 4,669 -> 4,573 pipeline binds, and
  5,110 -> 5,014 push-constant writes, with full state/adjoint/loss/optimizer/
  control parity preserved.
- Sequence-state arenas and standalone token tapes now own persistent upload
  arenas, so staging chunks survive across successive batches for the same
  sequence owner. Multi-sequence token-tape training retains its existing
  arena-level persistent uploader for the combined submission.
- Atomic multi-tensor SafeTensors replacement for `lm_head.weight`,
  `out_norm.weight`, and `out_norm.bias` while preserving all untouched tensor
  names/layouts and metadata.
- Model config validation reused from `hierarchos-inference`, keeping native CPU
  inference and Vulkan training on the same coherent-v9 contract.

The coherent-v9 RWKV cell interior no longer needs Python-supplied activations,
recurrent-state splits, state gradients, or DeepEmbed token features for this
training slice. Raw residual/LN1, full time-mix, channel-mix,
SharedTokenAdapter DeepEmbed, packed state ownership, TBPTT scheduling,
33-tensor cell gradient accumulation, and the tied token-embedding edge are all
Vulkan-native with PyTorch parity. Shared graph ownership above an individual
cell is now established, the tied embedding can accumulate across H, L, and LM
loss before one optimizer update, and the recurrent/output slices can execute in
one queue submission. The first projection-coupled token graph now removes the
host-materialized H/L residual-input and recurrent-upstream-gradient seam: all
six learned manager/worker projections participate directly in the Vulkan
forward/backward graph. The final `l_to_out -> out_norm -> LM loss` edge is now
live as well. Worker drift refinement now schedules the recurrent tickets
directly with a second device-resident activation/state arena, so the shadow tape
survives while the committed L transition restarts from the original real state.
The four-step-shadow/one-step-commit recurrence harness stays within about
`1.6e-6` parameter error, and the full three-step worker-refinement/loss graph is
also PyTorch-parity proven as described above.

The next integration boundary has moved again. Worker row-local convergence,
freeze semantics, commitment forward/backward, hard-ACT selected-output/depth
backpropagation, shadow-H recurrence, selected-state commitment, and the graph-
owned activation clamps are no longer blockers on the full worker/loss path, and
the H-cell/L-cell/projection/out_norm/tied-LM optimizer islands have now been
collapsed into the named full-model registry. Its moments/step use the existing
resumable SafeTensors AdamW checkpoint format. Full-model microbatch gradient
accumulation, primary-arena device-state starts, and the outer multi-token
state/adjoint tape are now live. Token replay is split into a forward-only
checkpoint phase and reverse rematerialization: the checkpoint sweep stops at
the committed L state, records no output-loss or gradient kernels, and preserves
only token-owned recurrent/control boundaries. Reverse rematerializations now
record into that same caller-owned command buffer, using per-token loss readback
slots and compact recurrent finishes that avoid scheduler-global readback
aliasing. Final state, initial adjoint, and control readbacks are appended to the
same batch, so a tape with `N` tokens now uses one Vulkan queue submission rather
than `N + 2`. `HierarchosSequenceGradientNormalization::MeanByToken` also scales
the canonical 79-tensor gradient registry on-device immediately before the
sequence AdamW step; the legacy `train_token_tape` entrypoint retains summed
gradients for compatibility. Multi-sequence accumulation now uses the same
normalization after the total token count across the tape arena, so it matches a
PyTorch accumulation window without changing the SafeTensors parameter or AdamW
companion-state contracts. Descriptor-pool churn is no longer proportional to
dispatch count, repeated binding combinations are cached within a batch, and
upload staging is now packed into reusable-within-batch arena chunks. Stable
  pipeline/push state removes thousands of state-setting commands from the
  historical 9,155-dispatch regression without changing any PyTorch-parity
  observable. The low-rank pass established a 6,339-dispatch / 2,629-barrier
  baseline. The profiler-selected post-mix/channel-mix pass now records 5,859
  dispatches and 2,149 shader barriers: another 480 dispatches and 480 barriers
  removed without changing state, adjoints, loss, optimizer moments, or control
  decisions. Relative to the original 9,155-dispatch regression that is 3,296
  fewer dispatches. The 6,339 milestone was itself 2,816 fewer than the
  historical baseline, 2,180 fewer than the
  8,519-dispatch baseline immediately before the grouped-projection/residual
  fusion pass, 1,476 fewer than the preceding 7,815-dispatch milestone, 1,060
  fewer than the 7,399-dispatch milestone, and 964 fewer than the immediately
  preceding 7,303-dispatch milestone.

The token-side boundary has now crossed the memory front-end as well.
`HierarchosTokenFrontendOp` consumes raw token IDs plus recurrent context,
gathers the tied `lm_head.weight`, advances bounded coherent-v9 ROSA from a
persistent Vulkan history buffer, applies the shared-factorized ROSA adapter and
learned gate/router, runs `qproj`, scaled LTM similarity, deterministic hard
top-k, slow+fast value gather, LTM gating, assembles
`[token_x, persistent, gated_ltm_values]`, and executes `in_proj -> GELU ->
finite_clamp(30)` without a PyTorch-preprocessed `enc` seam. Reverse mode covers
the ROSA gate/adapter/router, qproj, the selected-score LTM addressing surrogate,
LTM keys/values/gate/router, persistent, in_proj, recurrent `prev_context`, and
both raw/predicted tied-embedding paths. A deliberately non-neutral parity
fixture currently holds forward error below `2.3e-8` and worst gradient error
below `1.5e-8` on the AMD Radeon Graphics device, with exact hard top-k indices.
The bounded ROSA predictor now uses an exact two-generation suffix-match state.
For each prior end position `j`, the current match length is one token equality
plus the preceding generation at `j-1`; one 64-thread workgroup evaluates those
positions in parallel, then selects the longest match and rightmost prior
occurrence exactly. Winner selection is a deterministic shared-memory reduction,
so the shader no longer rereads the freshly written generation or serializes its
winner through global atomics. For bounds up to 65,535, two match lengths are
packed into each ordinary 32-bit storage word without requiring Vulkan 16-bit
storage features; larger experimental bounds retain the full-u32 fallback. At
the coherent-v9 default `rosa_max_context=512`, private two-generation match
state therefore falls from 4,096 to 2,048 bytes per lane. This removes the old
triangular candidate lattice and its per-candidate token scans: work is `O(C)`
constant-time comparisons per appended token, or `O(C^2)` across a full bounded
segment, without hashes or collision-dependent behavior. A 529-token persistence
regression now crosses the real 512-token segment boundary through arbitrary
caller chunks with predictions exactly matching Python, then verifies explicit
state reset. History remains lane-major (`[batch, rosa_max_context]`); a two-lane
regression advances 12 interleaved time steps with staggered resets and matches
two independent Python ROSA states exactly. Legacy unbounded ROSA keeps the
native-Rust suffix automaton fallback until it has an explicit unbounded device
history allocation/checkpoint policy.

`HierarchosTrainingGraph::from_model_package_with_token_frontend` joins that
front-end to the existing hard-ACT/worker/loss graph and one canonical AdamW
registry. `train_raw_token_worker_refinement_loss_one_submit` is the first raw
token-to-loss training entrypoint: bounded ROSA, memory retrieval/gating,
manager/worker recurrence, LM loss, the GPU-resident `d(enc)` reverse edge, all
front-end gradients, and AdamW execute in exactly one Vulkan queue submission.
The registry grows from 79 to 95 tensors while retaining exactly one physical
and optimizer-owned `lm_head.weight`; front-end-enabled graphs reject the legacy
host-`enc` update API so inactive memory tensors cannot silently receive decay or
momentum. A batch-2 one-token regression now matches the old host-`enc` graph
bit-for-bit for loss, `d(enc)`, H output, and L output while retaining one queue
submission and the full 95-slot optimizer registry. The same regression now
checks the previously easy-to-miss qproj context edge explicitly: the full raw
graph's `d(previous_context)` is exactly the legacy worker/context adjoint plus
the standalone Vulkan memory-front-end adjoint (`0` max absolute drift on the
current AMD device), so raw-token training no longer truncates that gradient.

The raw-token path now crosses the outer multi-token boundary too.
`train_raw_token_tape_with_normalization` performs the true token sweep with one
persistent ROSA lane per batch row, but checkpoints only each token's discrete
ROSA prediction ID and valid bit. Reverse traversal replays the learned
ROSA/qproj/LTM/in-proj frontend from those two device-resident rows, so suffix
history is never advanced twice and a full activation tape is unnecessary. A
three-token, batch-2 regression runs forward checkpointing, recurrent replay,
frontend reverse, and the single 95-tensor AdamW update in exactly one Vulkan
queue submission. A separate host-`enc` tape comparison is bit-identical for
per-token loss, final H/L packed state, initial H/L packed-state adjoints,
hard-control checkpoints, and every legacy optimizer moment except the shared
`lm_head.weight` slot, whose additional raw/predicted frontend embedding
gradient is intentional. Its saved ROSA decisions are exactly
`[[-1,-1],[-1,-1],[5,3]]` for two independent repeated-pattern lanes after the
reverse sweep, proving rematerialization did not mutate or replace the forward
suffix decisions. The raw-token registry now contains 16 frontend slots; the
auxiliary-only `val_proj.weight` slot receives moments on sampled value-alignment
steps while the ordinary frontend slots continue to receive their normal reverse
gradients.
The token-tape memory planner accounts for the two 32-bit ROSA replay rows per
token, and the same discrete-checkpoint contract is retained by dense
microbatching and sparse state replay. ROSA's first state-bandwidth pass is now
in place: shared-memory winner reduction removes the second state scan/global
atomics and bounded-width packing halves match-state storage at the production
context bound without changing the prediction-ID/valid-bit replay ABI or model
checkpoint format. The occupancy pass is now hardware-aware as well. Vulkan 1.1
subgroup capabilities are queried from the selected physical device and the
predictor switches to subgroup arithmetic only when compute-stage BASIC and
ARITHMETIC operations are actually supported; the original 64-lane shared-tree
kernel remains the portability fallback. Subgroup SPIR-V is compiled at
32/64/128/256 workgroup widths. For packed bounded state the geometry scheduler
still derives a safe width from the actual `ceil(max_context / 2)` state-word
count, but the production `257..=512` context range no longer assumes that the
wave64 Radeon result transfers to another GPU. When both 128- and 256-lane
groups are supported, model construction warms and measures the 128-lane kernel
against the optimized 256-lane single-packed-pair kernel, caches the decision by
device name/subgroup width/context/batch geometry, and selects 256 only when its
median submit time wins by at least 2%. A failed microprofile falls back to the
geometry scheduler instead of preventing model load. This makes the same binary
profile-driven on wave32/wave64 NVIDIA, AMD, and other Vulkan subgroup
geometries without introducing a vendor table. `HIERARCHOS_ROSA_WORKGROUP_SIZE`
still provides an explicit `32|64|128|256` override;
`HIERARCHOS_ROSA_DISABLE_AUTOTUNE=1` restores geometry-only selection, and
`HIERARCHOS_ROSA_AUTOTUNE_LOG=1` reports the measured candidates and decision.
None of these choices changes model state, replay checkpoints, or the
SafeTensors ABI.

The same measured-dispatch policy now reaches into the RWKV training backward
recurrence. For the packed-state path that receives the external `v_first`
state gradient, Hierarchos benchmarks the complete compatible matrix-state
backward schedules instead of assuming the deepest available fusion is fastest:
`rkv-add3 + key-transform + key-param-reduce`, fused `rkv-add3-key + reduce`,
and (for batch 1) the single-dispatch `rkv-add3-key-reduce` specialization. The
profile is warmed, median sampled, cached by Vulkan device/subgroup/model
geometry, and only replaces the deepest compatible fusion when another schedule
wins by at least 2%. A profiling failure falls back to the deepest fusion rather
than blocking model construction. `HIERARCHOS_RWKV_STATE_BACKWARD_SCHEDULE`
accepts `auto|rkv|rkv-key|rkv-key-reduce` for controlled parity/profiling runs;
`HIERARCHOS_RWKV_STATE_BACKWARD_DISABLE_AUTOTUNE=1` keeps the structural
deepest-fusion policy, and `HIERARCHOS_RWKV_STATE_BACKWARD_AUTOTUNE_LOG=1`
reports timings and the selected schedule. This changes command topology only:
the recurrence equations, FP32 reduction order within each existing shader,
PyTorch row-major parameter tensors, SafeTensors names, optimizer state, and
CUDA/native-Rust checkpoint interchange remain unchanged.

On the current AMD Radeon Graphics test target (subgroup 64, width 128,
head-size 64), the first warmed batch-2 profile measured about `0.1332 ms` for
the three-dispatch `rkv-add3+key+reduce` chain versus `0.1605 ms` for the deeper
`rkv-add3-key+reduce` fusion, so the runtime correctly selected the nominally
less-fused schedule. At batch 1, `0.1407 ms` versus `0.1422 ms` was inside the
2% guard, so the single-dispatch deepest fusion remained selected. That is the
intended behavior: fusion depth is no longer treated as a vendor-independent
performance truth.

The scheduler now extends one layer farther outward and profiles recurrence plus
the neighboring r/k/v projection backward as one training-step segment. It
crosses each compatible recurrence schedule with four projection topologies:
fused or split projection-input/time-mix backward, with the independent
projection-weight gradient recorded before or after that input-gradient path.
The full-cell search also crosses the three compatible low-rank fan-in
topologies, so the production search space can reach 36 candidates. The timing
loop first drives an idle GPU to a steady clock, then uses one cheap elimination
sample across the complete search space. It keeps the faster half of each
schedule axis plus the one-shot winner and the deepest default, and only those
finalists enter the five-sample alternating forward/reverse median race. This
avoids turning integrated-GPU DVFS or host-submit jitter into a false schedule
win without paying five robust samples for every point in the 3-D Cartesian
product. The selected segment is used
directly by the external-`v_first` full cell backward; no tensor layout or
gradient equation changes. `HIERARCHOS_RWKV_PROJECTION_BACKWARD_SCHEDULE`
accepts `auto|weight-fused|fused-weight|weight-split|split-weight`, while
`HIERARCHOS_RWKV_BACKWARD_SEGMENT_DISABLE_AUTOTUNE=1` and
`HIERARCHOS_RWKV_BACKWARD_SEGMENT_AUTOTUNE_LOG=1` disable or report the combined
autotune respectively. The existing state-backward override composes with the
projection override, making every candidate reproducible for parity tests.

Combined winners are now persistent across launches. The cache key includes
Vulkan device name, subgroup width, model width/head size, batch, the exact
`w/a/g` low-rank tuple, and whether the profile is core-only or the complete
time-mix cell, so a winner is never borrowed across an incompatible rank or
training boundary. On Windows the default cache is
`%LOCALAPPDATA%/Hierarchos/vulkan-rwkv-backward-segment-v1.json`; XDG and
`$HOME/.cache` locations are used on other hosts. Set
`HIERARCHOS_RWKV_BACKWARD_SEGMENT_CACHE_PATH` to choose a file,
`HIERARCHOS_RWKV_BACKWARD_SEGMENT_DISABLE_PERSISTENT_CACHE=1` to make the cache
process-local again, or `HIERARCHOS_RWKV_BACKWARD_SEGMENT_REAUTOTUNE=1` to force
a fresh measurement and replace the saved winner. Cache I/O is best-effort and
never changes the tensor/checkpoint ABI.

On the same Radeon/subgroup-64/width-128/head-size-64 target, the stabilized
combined profile selected the shallow `rkv-add3+key+reduce` recurrence for both
tested batches. At batch 1 the old deepest topology
`rkv-add3-key-reduce + weight->fused-input-mix` measured about `0.1099 ms`, while
the selected shallow segment measured about `0.0847 ms` (roughly 23% lower).
At batch 2 the deepest compatible
`rkv-add3-key + weight->fused-input-mix` segment measured about `0.1078 ms`,
while the selected shallow split-projection segment measured `0.0763 ms`
(roughly 29% lower). A second hot profile produced the same recurrence family
ordering, while projection ordering remained close enough at batch 1 for the 2%
hysteresis to prefer stability over sub-percent churn. Packed-cell PyTorch
parity remains within FP32 tolerance, so the performance choice is purely Vulkan
command topology and remains checkpoint-compatible with PyTorch/CUDA inference.

That combined scheduler is now profiled again at the true time-mix cell
backward boundary once `a/w/g` and post-mix are attached. Each recurrence +
projection candidate is timed inside one command stream containing post-mix
backward, recurrence/key backward, r/k/v projection weight/input backward, the
complete low-rank `a/w/g` reverse graph, and the final shared `x_norm` / previous
token fan-in. The cache key distinguishes this full-cell profile from the older
core-only microprofile, so constructing a partial recurrent op cannot seed a
full training cell with a locally optimal decision. On the AMD Radeon Graphics
target (subgroup 64, width 128, head-size 64), this immediately exposed a real
cross-boundary interaction: at batch 1 the core-only segment selected
`rkv-add3+key+reduce + fused-input-mix->weight` at about `0.0850 ms`, while the
same candidate inside the complete cell measured about `0.1665 ms` and the
full-cell scheduler instead selected
`rkv-add3+key+reduce + split-input-mix->weight` at about `0.1653 ms`. At batch 2
the core and full-cell profiles both selected the shallow split-input topology,
measuring about `0.0766 ms` and `0.1696 ms` respectively. The deeper recurrence
families were materially slower in the full-cell profile (`~0.185-0.199 ms` at
batch 1 and `~0.191-0.202 ms` at batch 2). This is the intended next scheduling
boundary: local fusion wins are advisory until they survive the downstream
cache, descriptor, and RAW-dependency environment of the actual training cell.
Setting `HIERARCHOS_RWKV_BACKWARD_SEGMENT_AUTOTUNE_LOG=1` while running
`tools/verify_vulkan_rwkv_full_time_mix_parity.py` now forwards both the core and
full-cell timing traces for reproducible hardware comparisons.

The next measurement boundary is the real raw-token TBPTT training submission,
not the isolated cell. `hierarchos-vulkan-training-submission-bench` calls
`train_raw_token_tape_sequences_budgeted` directly, so every timed sample
includes worker refinement, recurrent tape/replay, gradient accumulation,
budget-selected dense or sparse scheduling, queue submission/readback, and the
single canonical full-model AdamW step. It reports p50 optimizer-step latency,
batch-scaled tokens/second, outer token positions/second, selected H/L backward
schedules, queue count, and the memory plan. It consumes the same multi-step JSON
case shape used by the raw-token tape parity harness:

```powershell
cargo run --release --manifest-path hierarchos-vulkan/Cargo.toml `
  --bin hierarchos-vulkan-training-submission-bench -- `
  --model path/to/rust_model `
  --case path/to/raw_token_tape_case.json `
  --tokens 8 --sequences 2 --warmup 2 --iterations 5
```

For a self-contained hardware smoke benchmark, the Python wrapper exports the
existing tiny coherent PyTorch fixture to the ordinary native SafeTensors
package and invokes that same Rust submission:

```powershell
python tools/benchmark_vulkan_training_submission.py `
  --tokens 8 --sequences 2 --warmup 2 --iterations 5 --autotune-log
```

Steady-state training can request `--readback loss-only` to keep parity-only ACT
snapshots plus final/initial recurrent-state diagnostics off the host transfer
path while still returning token losses and optimizer metadata. The default is
`full`, preserving existing verifier/debug behavior. On the AMD Radeon Graphics
development device, an explicit dense `8 tokens x batch 2` FP32 plan measured
`32.71 ms` / `489.4 batch-tokens/s` with full readback and `30.22 ms` / `529.6
batch-tokens/s` with loss-only readback: about 7.6% lower median step latency and
8.2% higher throughput, with the same one queue submission and matching loss
trajectory.

Cross-vendor subgroup experiments can be made explicit on that wrapper instead
of relying on ambient environment variables. For example, the low-rank
first-stage A/B above can be reproduced on a device supporting subgroup-size
control with `--required-subgroup-size 32 --low-rank-first-stage-arm portable`,
then repeated with `--low-rank-first-stage-arm subgroup-packed-share`. Omitting
either flag preserves the inherited environment, while an explicit `portable`
selection clears the subgroup-packed-share opt-in for the child benchmark.

For GPU-side kernel attribution, set `HIERARCHOS_VULKAN_PROFILE_KERNELS=1` on
the same benchmark. Every compute dispatch is bracketed with Vulkan timestamp
queries and each submitted batch reports aggregate GPU time by stable kernel
category plus the hottest individual shader names. This is intended for
bottleneck discovery: the reported `gpu_ms` is the sum of compute-dispatch
intervals and does not include host command recording, transfer-only work, or
readback latency, while the timestamp commands themselves add profiling
overhead. Graph construction can also emit small autotuning batches before the
real training submission; the full optimizer step is the large batch containing
the loss, recurrent, gradient, and AdamW categories together. Run ordinary
benchmarks without this environment variable when comparing end-to-end
throughput.

The packed-FP16 LN1/low-rank producer also has a compact two-row specialization
for `width <= 32` with all three low-rank dimensions `<= 32`. One 64-lane
workgroup owns two rows, preserving each row's scalar LayerNorm reduction/FMA
order while avoiding a mostly idle second half-wave on the common width-32
training geometry. Set
`HIERARCHOS_RWKV_LOW_RANK_DISABLE_TWO_ROW_LN_FORWARD_FUSION=1` to force the
one-row kernel for same-build A/Bs. On the wave64 AMD Radeon Graphics fixture
(`width=32`, ranks `8/8/8`, batch `2`, 8 token positions, FP16-storage parity),
the hot LN1/low-rank shader fell from about `3.12 ms` to `2.91 ms` per profiled
optimizer step. A seven-sample non-profiled same-build A/B moved median step
latency from `37.203 ms` to `34.787 ms` and throughput from `430.07` to `459.95`
batch-tokens/s while the full PyTorch-parity/token-tape verifier remained green.

The outer tape scheduler is now a bounded online autotuner instead of profile
playback. Matching persistent candidates retain their complete raw observation
history, but exploitation is ranked by a recency-weighted conservative
throughput bound rather than raw median throughput alone. New observations decay
older same-geometry evidence by observation order, so a thermal/DVFS regime
change can overtake a large historical benchmark set without deleting that
history. The bound combines the adaptive throughput estimate, cross-record
dispersion, and a sampling-noise floor, so a noisy one-off winner still has to
earn enough margin to displace a repeatedly measured plan. The raw median,
adaptive estimate, lower-confidence exploit score, upper-confidence exploration
score, relative uncertainty, decayed effective iteration weight, observation
age, record count, and raw measured-iteration count are exposed in the selected
memory plan and benchmark JSON.
The byte estimator and the driver-reported working-set limit remain hard gates;
confidence never makes an over-budget candidate executable.

Normal budgeted training also performs bounded exploration. The first eligible
step and, by default, every sixteenth eligible step run a small decayed-UCB
policy over the current safe plan and its nearby sequence-microbatch/checkpoint
neighbors. Safe unmeasured arms bootstrap first; once the local arms have data,
the upper-confidence score combines adaptive throughput, uncertainty, and
decayed effective sample weight. A candidate that has not been sampled recently
therefore becomes progressively eligible for remeasurement instead of being
permanently frozen after its first historical result. Exploration is the real
optimizer step, not a duplicate benchmark update: only execution geometry
changes, so the same full-model gradients, AdamW semantics, tensor names, and
SafeTensors checkpoint contract remain in force. After the step completes, its
batch-scaled tokens/second observation is appended and synced to
`vulkan_training_submission_profiles.v1.jsonl` as `plan_mode=online-explore` and
is immediately folded into the cumulative adaptive statistics. Ordinary
automatic replay decisions between probe steps are still excluded from scheduler
training data, so the selected winner cannot reinforce itself simply by being
selected.

`HIERARCHOS_VULKAN_TAPE_EXPLORE_EVERY=N` changes the exploration cadence,
`HIERARCHOS_VULKAN_DISABLE_TAPE_ONLINE_AUTOTUNE=1` disables online exploration
and persistence, and `HIERARCHOS_VULKAN_TAPE_PROFILE_LOG=1` reports confidence,
exploration, and persistence decisions. The existing
`HIERARCHOS_VULKAN_TAPE_PROFILE_DB` path override and
`HIERARCHOS_VULKAN_DISABLE_TAPE_PROFILES=1` switch continue to control the
persistent database as a whole. None of these scheduler choices changes the
checkpoint ABI, so a Vulkan-trained package remains directly loadable by the
PyTorch CPU/CUDA path and native Rust inference targets.

On the current AMD Radeon Graphics target, the persisted batch-2 full-cell
winner was
`rkv-add3+key+reduce + fused-input-mix->weight + low-rank-fused-base-fan-in`.
For two sequences of eight outer positions at batch two (32 batch tokens per
optimizer step), five measured end-to-end samples produced a 59.44 ms p50 and
538.36 batch-tokens/s. Forcing the older deep
`rkv-add3-key+reduce + weight->fused-input-mix + low-rank-fused-outer-fan-in`
topology on the identical fixture produced 66.99 ms and 477.68 batch-tokens/s.
That is about 11.3% lower optimizer-step latency and 12.7% higher throughput for
the autotuned topology, while the per-step loss trajectory remained identical.

Explicit shared history/source-state tiling has also been implemented as a
profiling specialization rather than assumed beneficial. The workgroup
cooperatively stages up to 512 history tokens plus the active 256 packed source
state words into shared memory before the recurrence. On the current AMD Radeon
Graphics (subgroup 64), a warmed five-sample 512-token/32-lane profile measured
4.837 ms for shared64, 5.240 ms for subgroup32, 3.229 ms for subgroup64,
2.607 ms for subgroup128, 2.907 ms for subgroup128 with cache tiling, 2.278 ms
for generic subgroup256, 2.194 ms for subgroup256-single-pair, and 2.292 ms for
subgroup256-single-pair with cache tiling. Tiling therefore regressed the
128-lane candidate by about 11.5% and the optimized 256-lane candidate by about
4.5% on this GPU, so it is deliberately profiler-only. The untiled 256
single-pair kernel remains about 15.8% faster than untiled 128 here, and the new
runtime autotuner selects it on this device while retaining exact Python/Vulkan
ROSA parity. A different GPU is free to select 128 instead.

Full-model Vulkan checkpoints now write current trainable tensors back into the
ordinary SafeTensors package while preserving untouched tensors and metadata,
and the explicit PyTorch-compatible `memory_gate_warmup_step` is persisted
separately from AdamW's update count. The merged 95-slot AdamW companion state
round-trips bit-exact. A trained checkpoint produced by the raw Vulkan graph has
been reloaded successfully by both `HierarchosCore` and the native Rust
inference binary; `ltm.fast_vals` remains bit-exact when it is not trained, while
the updated `in_proj.weight` and `val_proj.weight` are visible after reload. This keeps the model-file
contract suitable for PyTorch CPU/CUDA and native inference without a Vulkan-
specific weight format.
  Projection input-gradient now flows directly into time-mix interpolation
  backward on devices supporting 16 storage bindings, with the old two-dispatch
  path retained as a portability fallback. Channel mix likewise fuses `ln2`
  normalization with its interpolation while preserving the normalized/mean/rstd
  backward tape. The former `linear3_input_grad -> rwkv_time_mix3_backward` and
  `layer_norm_forward -> rwkv_channel_mix_forward` seams are gone from the
  dependency trace. LN1 is likewise fused into the low-rank producer on the
  native 32/32/64 path: the former
  `layer_norm_forward -> rwkv_low_rank_producer_forward_fused` edge (count 96)
  disappears. The profiler-selected follow-on pass now also absorbs the
  downstream w/a/g rank-output chains, so the former low-rank-producer -> tanh,
  parameter-matmul+bias, sigmoid, and decay seams are gone on the 31-binding
  fast path. The former hottest remaining LayerNorm
  producer edge, `layer_norm_forward -> linear_forward` (count 100), is now gone
  too: the four output-head occurrences fold into the tied LM projection and the
  96 adapter occurrences fold through the down projection and SiLU. Those same
  96 adapter occurrences now also absorb the up projection on the
  width-specialized fast path, eliminating the former
  `layer_norm_linear_silu_forward_fused -> linear_bias_forward` edge without
  changing the SafeTensors ABI.
  Dependency-chain profiling now resolves each stable SPIR-V FNV-1a signature
  back to its canonical checked-in kernel name at build time. Trace lines retain
  the raw producer/consumer hashes for machine compatibility and append
  `producer_name`/`consumer_name`, so the next fusion target can be selected
  directly from measured graph pressure without a separate log-resolution pass.
  The profiler-selected low-rank fusion now finishes all three rank-output
  branches inside `layer_norm_low_rank_producer_forward_fused` on devices
  exposing at least 31 storage-buffer bindings: w-branch tanh/w2+bias/decay,
  a-branch a2+bias/sigmoid, and g-branch sigmoid/g2. `w_hidden`, `w_tanh`,
  `w_pre`, `a_hidden`, `a_pre`, `g_hidden`, and `g_sigmoid` remain materialized,
  so the existing backward kernels and PyTorch/SafeTensors parameter layouts are
  unchanged; lower-binding-count devices retain the standalone forward chain.
  On the AMD Radeon Graphics two-sequence/four-token regression this
  profiler-guided low-rank pass moves 7,011 -> 6,339 dispatches, 3,013 -> 2,629
  shader barriers, 4,573 -> 3,997 pipeline binds, and 5,014 -> 4,342
  push-constant writes while state, adjoint, loss, optimizer, and control parity
  remain exact.
  The immediate profiler-guided successor then collapses the measured
  `rwkv_group_norm_forward -> rwkv_bonus_gate_forward -> linear_residual_forward`
  and `layer_norm_channel_mix_forward_fused -> linear_forward ->
  relu2_deepembed_forward -> linear_residual_forward` forward chains. On the
  same device/workload the resulting 5,859-dispatch graph uses 2,149 shader
  barriers, 538 unique descriptor sets, 3,517 pipeline binds, and 3,862
  push-constant writes. Full PyTorch comparison still reports loss absolute
  error `1.41693115e-7`, worst graph-value error about `1.69e-6`, worst parameter
  error about `4.02e-7`, and exact device-tape state/adjoint/loss/optimizer/control
  parity. The next packed-boundary/backward-fan-in pass moves the same
  two-sequence/four-token workload from 5,859 -> 5,731 dispatches and 2,149 ->
  2,085 shader barriers while preserving exact device-tape
  state/adjoint/loss/optimizer/control parity. The former
  `layer_norm_channel_mix_full_forward_fused -> rwkv_state_pack` edge is gone:
  the trace now records
  `rwkv_group_norm_bonus_gate_linear_residual_forward_fused ->
  packed_cell_channel_mix_state_forward_fused`. The cell-level second residual
  add is likewise absorbed into `layer_norm_input_grad_residual_fused`; other
  `vector_add -> vector_add` dependency chains remain elsewhere in the backward
  graph and are now separable profiler targets rather than part of this cell
  residual boundary.
  The next RAW-fan-in pass attacks those measured backward seams directly while
  retaining the PyTorch/SafeTensors tensor contract. Low-rank time-mix backward
  now folds the recurrent-core input/previous adjoints into
  `rwkv_time_mix3_backward_fused_add`; matrix-state backward folds post-mix r/k
  fan-in through `rwkv_matrix_state_backward_fused_rk_add` and, when the
  recurrent external-v branch is present, folds the full three-way value fan-in
  through `rwkv_matrix_state_backward_fused_rkv_add3`. Channel mix similarly
  absorbs its enclosing residual edge into
  `layer_norm_backward_fused_add_residual`. Descriptor-limited devices retain
  the existing standalone-add fallbacks. On the AMD Radeon Graphics
  two-sequence/four-token regression this moves 5,731 -> 5,527 dispatches and
  2,085 -> 1,901 shader barriers, with 521 descriptor sets, 3,281 pipeline binds,
  and 3,686 push-constant writes. Full PyTorch comparison remains at loss
  absolute error `1.41693115e-7`, worst graph-value error about `1.69e-6`, and
  worst parameter error about `4.02e-7`; device-tape
  state/adjoint/loss/optimizer/control parity remains exact. The former
  `rwkv_matrix_state_backward_fused -> vector_add`, matrix-state `-> vector_add3`,
  and `layer_norm_backward_fused_add -> vector_add` RAW seams are absent from
  the resulting dependency trace. The outer-cell LN1/state adjoint fan-in now
  has a 16-storage-binding fast path as well:
  `rwkv_time_mix3_backward_fused_add_outer` preserves the legacy
  `(base + low_rank) + state` FP32 accumulation order while writing the final
  normalized-input adjoint directly. Devices exposing only 15 storage bindings
  retain `rwkv_time_mix3_backward_fused_add` plus the standalone outer
  `vector_add`, and lower-limit devices retain the fully generic path. This is
  still an execution-graph optimization only; parameter names, layouts,
  SafeTensors interchange, and AdamW semantics are unchanged.
  Vulkan tensor storage is now suballocated from aligned reusable memory blocks
  instead of issuing one `vkAllocateMemory` call per `GpuBuffer`. Free slices are
  coalesced, oversized blocks are released once empty, and mapped uploads/
  readbacks account for each buffer's suballocation offset. The default backing
  block is 16 MiB and can be tuned with `HIERARCHOS_VULKAN_MEMORY_BLOCK_MIB`
  (1..=1024) for unusually constrained or large-memory devices. Allocation
  failures now report logical live-buffer bytes, reserved pooled bytes, backing
  allocation count, and the Vulkan device allocation limit. On the AMD Radeon
  Graphics full-graph ownership regression, 1,520 logical Vulkan buffers pack
  into 2 `VkDeviceMemory` allocations (2,559,468 live bytes in 33,554,432
  reserved bytes). `VulkanDevice::memory_stats` and
  `HierarchosTrainingGraph::memory_stats` expose the same runtime telemetry, and
  the construction parity harness asserts that pooling remains active.
  Budget-aware scheduling now sits above that allocator. `VulkanDevice::memory_budget`
  queries `VK_EXT_memory_budget` when the driver exposes it and aggregates the
  live budget/usage of device-local heaps; older Vulkan stacks fall back to
  physical heap size plus Hierarchos' own reserved-memory usage and explicitly
  report that the extension was unavailable. `HierarchosTokenTapeFootprint`
  predicts persistent recurrent boundaries, per-token control/loss storage, and
  peak control readbacks before a tape is allocated. `HierarchosTapeMemoryPolicy`
  defaults to 85% of the reported device-local budget with another 512 MiB held
  aside for upload staging, command/descriptor growth, display pressure on APUs,
  and allocations outside the persistent tape estimate.

  `HierarchosTrainingGraph::plan_token_tape_memory` preserves the requested full
  token span first and reduces simultaneous independent sequence count until the
  predicted tape fits. `train_token_tape_sequences_budgeted` executes that plan:
  it allocates only one selected sequence arena at a time, keeps the canonical
  79-tensor gradient registry live across Vulkan submissions, applies a single
  global `MeanByToken` denominator, and performs AdamW exactly once at the end of
  the logical accumulation window. The parity harness now forces a one-sequence
  memory window for a two-sequence workload and verifies that the resulting
  two-submit update matches the original one-submit path. If one dense sequence
  still cannot fit, the planner now executes sparse recurrent-state checkpoint
  replay: it retains only every selected H/L boundary, rematerializes one bounded
  segment at a time in reverse, and carries the same device-resident H/L adjoints
  across segment boundaries, preserving full BPTT rather than truncating it. An
  eight-token/stride-two regression forces the budget planner into four replay
  segments and compares final H/L state, initial H/L adjoints, losses,
  hard-control checkpoints, and every AdamW moment against the dense tape.
  `hierarchos-vulkan-full-training-graph-inspect`
  also reports the raw device-local budget and accepts `--plan-sequences` plus
  `--plan-tokens` (with optional `--tape-budget-fraction` and
  `--tape-reserve-mib`) to inspect a plan before training. These changes remain
  purely runtime scheduling/storage management: PyTorch tensor shapes,
  SafeTensors names, optimizer semantics, and CUDA/native-Rust interchange are
  unchanged.
  The full training graph now also compiles its first explicit phase-lifetime
  scratch schedule. Named FP32 working ranges are assigned inclusive
  forward/worker-backward/manager-backward/optimizer epochs and colored onto
  physical Vulkan scratch slots. The initial conversion covers the worker
  forward input clamp/state mask plus the manager state-gradient sum, manager
  vector sum, and worker-backward vector sum. The two worker-forward ranges
  remain separate because they are simultaneously live, as do the two manager
  ranges; forward, worker-backward, and manager-backward lifetimes then reuse
  the best-fit slots across phase boundaries. The
  slot backing comes from the timeline-safe transient scratch slab, so command
  submission keeps the aliased lease alive until every referencing Vulkan
  timeline epoch retires. `HierarchosTrainingGraphSummary` and
  `training_working_set_plan()` report logical bytes, physically planned bytes,
  reclaimed bytes, semantic lifetimes, and slot assignments. This schedule is
  runtime-only and does not enter model SafeTensors, optimizer state, portable
  replay, or PyTorch/CUDA interchange.

  `hierarchos-vulkan-full-training-graph-inspect` exposes the same compiled
  working-set numbers. When a token-tape plan is requested it also reports
  `estimated_vulkan_training_peak_bytes`, defined as Hierarchos-owned live
  logical graph bytes plus the live leased portion of the physical transient
  scratch slab plus the planned token-tape peak. Passing PyTorch's
  `peak_cuda_memory_bytes` from
  `benchmark_training_pipeline.py` as `--pytorch-peak-bytes N` adds a direct
  `vulkan_to_pytorch_peak_ratio`. This deliberately compares an allocator-style
  Hierarchos estimate with PyTorch's `torch.cuda.max_memory_allocated()` rather
  than Vulkan heap budget/usage, which may include display and unrelated driver
  allocations.
  The token-tape policy now also owns the first compiled kernel-geometry lattice.
  The hottest head-owned RWKV state/key backward kernels and the neighboring
  low-rank shared-input fan-in kernels are checked in at local sizes 32, 64, and
  128 under the stable policy labels `rwkv-state-bwd-wg32`,
  `rwkv-state-bwd-wg64`, and `rwkv-state-bwd-wg128`. H and L may choose different
  variants, and those labels are persisted beside microbatch, checkpoint stride,
  and H/L backward topology in the same end-to-end optimizer-step throughput
  profile. Online exploration still uses sparse coordinate arms for the broad
  policy lattice, but wave32/head64 recurrent work now gets a sparse residual
  search over recurrence-fusion depth and paired WG64/WG128 geometry. Projection
  ordering and low-rank fan-in stay fixed while this interaction is measured.
  The paired-geometry center is probed together with one-tower-at-a-time H and L
  fusion-depth spokes; H-neighbor x L-neighbor combinations are deliberately not
  materialized because the persistent marginal selectors can compose them. With
  two neighboring depths per tower this is at most 10 coupled arms across both
  geometries instead of 18, without losing the genuinely coupled H/L occupancy
  measurement. Candidate depths are filtered against the
  compiled state and low-rank workgroup specializations before they can become an
  online optimizer step. WG32 uses strided channel
  ownership when the RWKV head is 64-wide, while the low-rank fan-in variants
  repartition channel ownership by workgroup size. Their per-channel batch loops,
  the serial FP32 key-normalization, and recurrent reductions all retain the
  original accumulation order, so this policy dimension changes occupancy rather
  than the PyTorch/SafeTensors numerical ABI. Geometry-tagged throughput records
  now carry policy revision 2: pre-fan-in geometry measurements are ignored while
  older geometry-agnostic tape observations remain usable.
  A second, deliberately separate policy axis now controls reduction numerics.
  `strict-parity` remains the default and keeps the historical serial FP32
  accumulation order. The opt-in `fast-subgroup` arm changes the two head-local
  key-normalization dot products (`sum(z*z)` and `sum(grad_kk*kk)`) to
  cooperative subgroup arithmetic. `fast-recurrent-tree` and
  `fast-recurrent-tiled` keep key normalization strict while parallelizing the
  larger recurrent row/column reductions with interleaved or contiguous lane
  partials. `fast-recurrent-subgroup` now has three subgroup-native
  microarchitectures behind the same stable policy label. An exact one-wave
  workgroup gives one recurrent row to each lane. When one subgroup spans the
  head, a wider workgroup containing only complete hardware subgroups assigns
  disjoint rows/columns to those waves and uses subgroup reductions inside each
  wave. When the head is wider than one subgroup but no wider than two, adjacent
  subgroup pairs deliberately cooperate on the same head-local row/column dot:
  each wave reduces its native slice, elected lanes publish two shared partials,
  and one cross-wave add completes the reduction. This specifically opens the
  64-wide RWKV head on wave32 NVIDIA-style hardware; WG64 forms one pair and
  WG128 forms two independent pairs. The same paired-wave reduction is used for
  the fused key-normalization dots, so the native path never pretends one wave
  contains all 64 elements. All recurrent subgroup arms fail closed unless the
  workgroup is an integer number of complete subgroups and the head can be
  covered by at most two of them. The fast arms preserve tensor shapes, optimizer state,
  SafeTensors names, and PyTorch/native/CUDA checkpoint interchange while
  explicitly permitting FP32 accumulation-order drift.
  For cross-vendor qualification, `HIERARCHOS_VULKAN_REQUIRED_SUBGROUP_SIZE`
  can force one supported compute subgroup width through
  `VK_EXT_subgroup_size_control`. The override is opt-in and validated against
  the selected device's advertised min/max subgroup sizes and compute-stage
  support; without the environment variable every pipeline keeps the driver's
  native subgroup width. This is useful for exercising NVIDIA-style wave32
  scheduling on hardware that can expose multiple subgroup widths without
  changing the model, shader ABI, or checkpoint layout.
  The mixed-precision storage boundary has also started without weakening that
  ABI. `VulkanDevice::mixed_precision_capabilities` now reports and enables
  supported FP16 storage/compute features and separately reports whether the
  driver exposes `VK_KHR_shader_bfloat16`. The current Rust `ash` bindings predate
  the BF16 feature structure, so BF16 native arithmetic is not claimed or enabled
  yet. `VulkanFp32MasterParameterMirror` provides the first executable parameter
  path: it quantizes a canonical FP32 parameter into a packed two-elements-per-
  `uint` FP16 or BF16 device mirror, with explicit round-to-nearest-even, and can
  expand that mirror back to FP32 for kernels not yet specialized for half
  storage. FP16 packing deliberately does not rely on GLSL `packHalf2x16` for
  writes: the training ABI uses an explicit IEEE-754 FP32 -> FP16
  round-to-nearest-ties-to-even converter so the packed bits match
  `torch.float16` across Vulkan drivers and PyTorch CPU/CUDA. `unpackHalf2x16`
  remains safe for exact half -> FP32 decode. The source parameter is read-only
  during both operations. AdamW
  parameters, accumulated gradients, first/second moments, reductions, and
  SafeTensors checkpoint values therefore remain FP32 while half-storage/native-
  compute kernels are introduced incrementally. The end-to-end submission
  benchmark emits the enabled FP16 and exposed BF16 capability bits so cross-
  vendor profile collection can decide which precision arms are legal before
  timing them. The first full-graph precision arm is now opt-in as
  `HIERARCHOS_VULKAN_TRAINING_PRECISION=fp16-storage-fp32-compute`. The
  persistent AdamW registry owns coherent packed FP16 mirrors for the six
  bandwidth-heavy RWKV low-rank matrices in each H/L tower, all six learned
  manager/worker projection matrices (`l_feedback_proj`, `h_to_context`,
  `h_halt_proj`, `l_input_proj`, `context_drift_proj`, and `l_to_out`), and the
  single shared `lm_head.weight`. Mirrored slots use
  a fused FP32-AdamW-plus-FP16-pack shader so the execution copy is produced
  directly from the post-update FP32 master without a second parameter read or
  refresh dispatch. The low-rank, manager/worker projection, and dense
  output-head forward/input-adjoint shaders bind those packed mirrors directly;
  projection biases remain FP32, and weight gradients still target the canonical
  FP32 layout. The generic packed linear forward/input-adjoint and fused
  `out_norm -> lm_head` path now consume neighboring half values in pairs so two
  scalar FMAs can share a packed-word decode while retaining the original FP32
  accumulation order. On devices exposing both `StorageBuffer16BitAccess` and
  shader `Float16`, the vocabulary-dominant LM input-adjoint can additionally
  view the exact same packed mirror bytes as native `float16_t` storage and
  promote each load directly to FP32 arithmetic, avoiding repeated
  `unpackHalf2x16` in the `W` and `W^T` passes. Devices without both Vulkan
  features keep the portable packed-u32 shader; no tensor, optimizer, or
  SafeTensors layout changes between the two consumers. Set
  `HIERARCHOS_VULKAN_DISABLE_NATIVE_FP16_LM_INPUT_GRAD=1` to force the portable
  packed-u32 consumer for cross-vendor A/B qualification even when native
  16-bit storage is available. The FP16 output path now makes that choice with
  a one-time device/geometry autotuner instead of assuming native-half is
  faster. The persistent key includes device name, effective subgroup size,
  compute shared-memory capacity, context width, vocabulary size, loss-row
  capacity, native-FP16 capability, and a fingerprint of every SPIR-V kernel in
  the raced LM slice, so rebuilding either implementation invalidates stale
  measurements automatically. The
  portable packed arm is the noise-resistant baseline: native-half must improve
  the median by more than 2% before it replaces the portable choice. Set
  `HIERARCHOS_VULKAN_LM_EXECUTION_ARM=fp16-packed|fp16-ce-tape|fp16-ce-tape-rows8|fp16-ce-tape-rows16|fp16-ce-tape-rows16-dot4|fp16-ce-tape-rows16-fused-adjoints|fp16-ce-tape-rows16-dot4-fused-adjoints|fp16-ce-tape-rows16-cluster4-fused-adjoints|fp16-native|fp16-native-reuse64|fp16-native-reuse128|fp16-native-reuse224`
  for a forced A/B,
  `HIERARCHOS_VULKAN_LM_FUSED_ADJOINT_TOPOLOGY=shared-hidden|private-hidden`
  to isolate the fused-adjoint shared-memory/register coordinate,
  `HIERARCHOS_VULKAN_LM_REAUTOTUNE=1` to ignore a stored decision,
  `HIERARCHOS_VULKAN_LM_AUTOTUNE_LOG=1` to print the race, or
  `HIERARCHOS_VULKAN_LM_AUTOTUNE_CACHE_PATH=<path>` to isolate its JSON cache.
  `HIERARCHOS_VULKAN_LM_AUTOTUNE_DISABLE=1` keeps the portable baseline and
  `HIERARCHOS_VULKAN_LM_DISABLE_PERSISTENT_CACHE=1` keeps measurements
  process-local. The full worker/loss graph also removes the former dense-dW
  staging sweep: because H starts each per-graph tied-gradient phase by clearing
  the shared FP32 `lm_head.weight` accumulator, the LM dW kernel writes that
  buffer directly and later H/L DeepEmbed scatter-adds join it in place. Set
  `HIERARCHOS_VULKAN_LM_FORCE_DENSE_GRAD_STAGING=1` to restore the old
  full-matrix scratch -> shared-gradient accumulation for profiling. On the
  current AMD development GPU, a same-binary width-448 / vocabulary-50,257 /
  two-token packed-FP16 A/B measured `106.3535 ms` with staging versus
  `98.5887 ms` direct, a 7.30% step-time reduction / 7.88% throughput gain with
  `fp16-ce-tape-rows16-fused-adjoints` and identical WG64/WG64 recurrent
  geometry. FP32 deliberately remains the outer training-precision arm:
  racing it silently against FP16 here would change the execution weights and
  therefore the PyTorch numerical oracle. The graph summary and benchmark JSON
  expose the resulting explicit precision/topology state (`fp32`, `fp16-packed`,
  `fp16-ce-tape`, `fp16-ce-tape-rows8`, `fp16-ce-tape-rows16`,
  `fp16-ce-tape-rows16-dot4`,
  `fp16-ce-tape-rows16-fused-adjoints`,
  `fp16-ce-tape-rows16-dot4-fused-adjoints`,
  `fp16-ce-tape-rows16-cluster4-fused-adjoints`,
  `fp16-native`/reuse32, `fp16-native-reuse64`, `fp16-native-reuse128`, or
  `fp16-native-reuse224`) while the inner race is guaranteed to preserve
  identical FP16 execution values.

  The native input-adjoint also now attacks the remaining duplicate LM-weight
  traffic directly. While each vocabulary lane reconstructs its logit, the
  workgroup repacks a tuner-selected 32/64/128/224 hidden-pair prefix into shared
  memory; after the CE adjoint is known, the `W^T` phase consumes those exact
  cached half values instead of rereading them from device memory. The reuse32
  arm consumes 12,544 bytes total (8 KiB weight tile plus resident hidden row and
  CE scratch), staying below Vulkan's guaranteed 16 KiB compute shared-memory
  floor. Reuse64 consumes 20,736 bytes, reuse128 consumes 37,120 bytes, and
  reuse224 consumes 61,696 bytes. Candidate creation is gated by the physical
  device's `maxComputeSharedMemorySize`, and the persisted tuner only races arms
  that fit that device and are useful for the current context width. Thus a GPU
  exposing at least 64 KiB can retain all 224 weight pairs for a 448-wide LM
  head and erase the complete second `W^T` read without adding a device buffer
  or changing checkpoint representation. On the current AMD
  Radeon Graphics development GPU (subgroup 64), a fresh width-32 / vocabulary
  50,257 / two-row race measured `11.0852 ms` for portable packed and `8.8498
  ms` for native-half, so the persisted arm selected `fp16-native`. A separate
  forced end-to-end A/B on the same two-sequence x eight-token, microbatch-2,
  checkpoint-stride-1 strict-parity fixture measured `181.315 ms` p50 for
  portable packed versus `170.8685 ms` for native-half, a 6.11% optimizer-step
  speedup. The
  tied token-embedding lookup deliberately continues to read the canonical FP32
  master in this tranche, and LM-head weight gradients remain FP32. No complete
  mirrored matrix is expanded through transitional FP32 scratch on these native
  packed consumers. Under `fp16-storage-fp32-compute`, arithmetic and
  accumulation, weight gradients, AdamW moments, and model checkpoints remain
  FP32. The training-submission benchmark
  reports separate H/L `*_low_rank_fp16_parameter_storage_active` bits plus
  `projection_fp16_parameter_storage_active` and
  `lm_head_fp16_parameter_storage_active`, and rejects a requested FP16 sample
  unless all four packed consumer families are really installed. Those bits
  are retained in externally collected profile records, making a precision-arm
  timing auditable instead of allowing a silent FP32 fallback. The precision
  policy is part of persistent tape
  profile geometry (legacy records mean `fp32`), and the benchmark/collector
  accept `--precision` / `--precisions` so FP16 evidence cannot contaminate an
  FP32 ranking. Persisted composite arms are also revalidated as a geometry +
  numerics pair before application; an old profile cannot select a workgroup
  geometry that only exists under a different RWKV numerical policy and then
  fail later inside the fused state-backward chain. On the current AMD Radeon
  Graphics development GPU (subgroup 64),
  the first post-LM-head automatic-scheduler A/B used an isolated profile DB,
  width 32, batch 2, two sequences x four outer positions, one warmup, and three
  measured optimizer steps. Both precision arms independently selected the same
  safe microbatch-2/checkpoint-stride-1 WG64 strict-parity plan. FP32 measured
  `43.51 ms` p50 / `367.76` batch-tokens/s; packed FP16 measured `46.18 ms` /
  `346.49` batch-tokens/s, about a 5.8% throughput regression on that tiny
  fixture. After expanding the precision arm across the six manager/worker
  projection matrices and adding pairwise packed-linear loads, a fresh
  profile-disabled A/B on the same width-32, batch-2, two-sequence x four-token
  geometry again selected microbatch 2 / checkpoint stride 1 / WG64 strict
  parity. FP32 measured `39.84 ms` p50 / `401.64` batch-tokens/s and the expanded
  packed-FP16 arm measured `40.55 ms` / `394.61` batch-tokens/s, narrowing the
  tiny-fixture gap to about 1.7%. The expansion therefore remains opt-in: parity and storage reduction
  are necessary gates, but the scheduler will not promote a precision arm until
  representative large-vocabulary/model geometry demonstrates a stable win.
  The output head has since crossed the next memory boundary: the production
  training graph no longer allocates its `[rows, vocab_size]` logits scratch.
  Streaming LM projection maintains online log-sum-exp row state and both LM
  adjoints regenerate/consume vocabulary tiles directly. A post-change
  production-vocabulary (`50,257`) width-32, batch-2, two-sequence x four-token
  smoke profile on the same AMD Radeon Graphics target, with microbatch 2,
  checkpoint stride 1, strict numerics, one warmup and three measured steps,
  recorded `108.49 ms` / `147.48` batch-tokens/s FP32 and `89.49 ms` / `178.80`
  batch-tokens/s packed-FP16 p50. After pairing packed-half reads in regenerated
  dot products while retaining one-hidden-column-per-lane input-adjoint
  consumption, the packed arm remeasured `85.73 ms` / `186.63` batch-tokens/s
  p50 on the same three-sample fixture. Treat these as the memory-first streaming
  baseline, not a final scheduler preference: the streaming weight adjoint is
  deliberately simple and is the next candidate for cooperative hidden-column
  tiling. Direct materialized-vs-streaming Vulkan testing covers a vocabulary
  tail beyond one 64-entry tile, and the complete worker-refinement PyTorch
  oracle passes in both precision arms with loss absolute differences of
  `1.42e-7` (FP32) and `1.79e-7` (packed FP16). Raw-token checkpoint interchange
  also passes PyTorch/native-Rust reload with maximum inference-logit errors of
  `3.58e-7` and `2.38e-7`, respectively; CUDA execution remains exercised by
  the same harness when an NVIDIA runtime is present.

  Generic linear input adjoints now reuse both sides of their dot product at
  workgroup scope without relaxing PyTorch-order numerics. A 16x16 workgroup
  stages each 32-column `grad_output` tile once per row and the corresponding
  32x16 weight tile once per input-column block; all 16 rows then consume the
  shared weight tile while every destination still executes its FMAs in
  monotonically increasing output-column order. The FP32 shader uses 4 KiB of
  shared storage and the packed-FP16 pair shader uses 6 KiB, both below the
  Vulkan 16 KiB minimum compute shared-memory guarantee. On the AMD Radeon
  Graphics development target, a controlled same-fixture profile (width 32,
  vocabulary 64, batch 2, eight token positions, two warmups and five measured
  optimizer steps) reproduced the former `linear_input_grad` seam at
  `5.04684 ms` median across 233 dispatches. Recompiling only this shader with
  weight reuse reduced that median to `1.06724 ms`, a 78.85% reduction / 4.73x
  kernel speedup; host optimizer-step p50 moved from `68.0286 ms` to
  `65.3146 ms`. Projection PyTorch parity remains within `1.24e-8` maximum
  input-gradient error, and both packed-FP16 and native-FP16 full worker/loss
  parity suites pass after the change. Tensor layouts, FP32 masters, optimizer
  state, SafeTensors names, and CUDA/native-Rust checkpoint interchange are
  unchanged.

  Precision qualification now uses the same full training boundaries as the
  production graph rather than only storage round-trips. Run
  `python tools/verify_vulkan_worker_refinement_loss_parity.py --precision fp16-storage-fp32-compute`
  to compare the packed-FP16 H/L and manager/worker projection execution paths,
  gradients, one-submit token tapes, AdamW updates, and updated parameters
  directly against the FP32 PyTorch reference. The PyTorch oracle explicitly
  rounds the mirrored recurrent, projection, and dense LM execution reads to FP16
  while retaining/restoring their FP32 master
  parameters for AdamW; it also checks that the Vulkan post-step LM mirror is
  bit-exact FP16(RNE) of the Vulkan FP32 master. Run
  `python tools/verify_vulkan_raw_token_training_graph.py --precision fp16-storage-fp32-compute`
  to train through the raw-token frontend, write the canonical FP32 SafeTensors
  model plus optimizer checkpoint, reload the model in PyTorch and native Rust,
  and compare inference logits. That second harness also executes the reloaded
  checkpoint on CUDA when `torch.cuda.is_available()`; hosts without an NVIDIA
  CUDA device report an explicit skip. Both harnesses assert H/L low-rank,
  manager/worker projection, plus LM-head packed-consumer activity, so neither can accidentally qualify a
  partial or FP32 fallback as the expanded FP16 precision arm.
  `python tools/verify_vulkan_cuda_vulkan_trajectory.py --require-cuda` is the
  stronger NVIDIA handoff gate: it takes a real Vulkan-produced full-model
  training package and repeatedly alternates PyTorch CUDA AdamW steps with real
  Vulkan raw-token training steps. The default is three alternating pairs and
  `--cycles N` controls the stress length. Every boundary round-trips the
  canonical model masters plus named AdamW first/second moments and reports the
  maximum parameter/moment handoff drift before final CUDA inference. On
  non-CUDA development hosts, `--cpu-fallback` runs the identical package and
  optimizer orchestration through PyTorch CPU so all Vulkan boundaries and
  serialization drift checks can still be regression-tested locally. This CPU
  fallback validates interchange/state continuity, not CUDA arithmetic itself.

  The first intentionally numerically distinct training-compute arm is
  `HIERARCHOS_VULKAN_TRAINING_PRECISION=fp16-storage-fp16-lm-backward`.
  It keeps the same FP16 execution mirrors plus canonical FP32 masters,
  accumulated gradients, AdamW moments, and SafeTensors layout, but moves the
  CE -> LM linear backward boundary onto native Float16 arithmetic. Stable
  softmax/log-sum-exp and z-loss adjoint formation remain FP32; the final
  source-scaled CE adjoint is rounded to IEEE FP16 before `W^T`/`dW`, each
  linear-backward product executes as an FP16 multiply, and every product is
  widened into the existing FP32 accumulators. A GradScaler source scale can
  therefore overflow at the real half-compute boundary and remains visible to
  the existing non-finite scan/unscale/step-skip machinery. Construction fails
  when Vulkan 16-bit storage plus shader Float16 support is unavailable rather
  than silently falling back to the storage-only numerical policy. Because the
  arithmetic is intentionally different, this first compute arm is kept out of
  the equivalence-only LM autotuner and currently uses the native reuse32 input
  adjoint topology.

  The same policy now advances beyond the LM head without moving the numerical
  safety rails. `out_norm` backward uses native FP16 products for its incoming
  adjoint/affine products while mean, variance, `rstd`, all reductions, and the
  destination gradients stay FP32. The six manager/worker affine projections
  use native FP16 products for `dW` and `db`, but deliberately keep upstream
  `dX = dY W` on the packed-FP16-storage / FP32-compute kernel. In the recurrent
  H/L low-rank blocks, unscaled callers keep parameter-gradient products in FP32
  and promote only the first-stage `w1/a1/g1` input adjoints to native FP16.
  Source-scaled backward domains already extend the input-adjoint boundary to
  the `w2/a2/g2 -> w1/a1/g1` inter-stage products. A further experimental gate,
  `HIERARCHOS_VULKAN_NATIVE_FP16_LOW_RANK_PARAMETER_GRAD=1`, also moves all six
  low-rank matrix-gradient products into Float16, then widens every product into
  the existing FP32 reductions/destinations. Bias reductions remain FP32. The
  parameter-gradient gate defaults off: its first three-update masked-TBPTT
  qualification exposed optimizer-trajectory drift, so it is retained as a
  measurable research arm rather than silently changing the portable contract.
  The low-rank parity harness now has an isolated `--native-fp16-dw-diagnostic`
  mode that executes exactly one `g2` dW dispatch. On the AMD Radeon Graphics
  development device that dispatch differs from the portable FP32 dW by
  `5.3868629e-05`. Although the compiled SPIR-V contains Float16 operand
  conversions, a Float16 `OpFMul`, and a conversion back to FP32, the observed
  result is bit-identical to serial FP32 accumulation of
  `FP32(FP16(x)) * FP32(FP16(dy))`; it differs from an explicitly FP16-rounded
  product oracle by `3.07261944e-05`. The verifier classifies this behavior
  explicitly instead of hiding it behind a tolerance.

  That single-dispatch result also narrows the earlier `g2` trajectory failure.
  In the source-scaled three-update labeled-sequence diagnostic, update 0
  reproduces `l_rnn.g2` pending-gradient drift of `2.70557404e-03` with native
  low-rank dW enabled. Disabling only
  `HIERARCHOS_VULKAN_NATIVE_FP16_LOW_RANK_PARAMETER_GRAD` reduces the same
  `l_rnn.g2` discrepancy to `1.16825104e-05`, about 232x smaller. Switching the
  PyTorch dW oracle from FP16-rounded products to the AMD-observed widened-input
  product semantics changes the enabled result only to `2.70533562e-03`.
  Therefore the large first-update `g2` gap is not a one-dispatch multiplication
  rounding mismatch: it is created when the native-half dW arm repeatedly
  quantizes slightly different recurrent activation/adjoint pairs and those
  per-use scratch gradients are folded, in order, into the existing FP32
  optimizer accumulator. The experimental parameter-gradient gate remains off
  by default while that repeated-use boundary is investigated further. The
  labeled parity harness now exposes `--diagnostic-tokens`, `--max-h-steps`,
  and `--max-l-steps` so that reuse geometry can be swept without changing the
  production graph. Keeping the default recurrent depths and truncating the
  same update-0 prefix gives native-dW `l_rnn.g2` drift of `6.19888306e-04`
  at 2 tokens, `8.68380070e-04` at 3 tokens, and `2.70557404e-03` at 6 tokens.
  At 2 tokens, disabling only native low-rank dW reduces the same tensor to
  `2.38418579e-06`. Reducing only worker refinement depth from 2 to 1 leaves
  `2.48074532e-03` at six tokens, so historical sequence reuse is the stronger
  amplification axis than the extra inner worker refinement on this fixture.
  Two alternative dW arithmetic arms now make that recurrent boundary directly
  measurable without changing FP32 masters, gradients, optimizer state, or the
  PyTorch-layout checkpoint tensors. Setting
  `HIERARCHOS_VULKAN_NATIVE_FP16_LOW_RANK_PARAMETER_GRAD_WIDEN_PRODUCT=1`
  preserves FP16 operand quantization but widens both rounded operands before
  the multiply. Setting
  `HIERARCHOS_VULKAN_NATIVE_FP16_LOW_RANK_PARAMETER_GRAD_COMPENSATED=1` instead
  splits each FP32 operand into high/low FP16 terms and reconstructs the three
  dominant products (`hi*hi + lo*hi + hi*lo`) in native half before the existing
  FP32 accumulation. The widened and compensated flags are mutually exclusive.
  The compensated arm is the current parity candidate because it recovers the
  rounding information lost by the repeatedly reused native-half operands while
  preserving an actual Float16 multiply path.

  `HierarchosTrainingGraphSummary` reports this distinction explicitly as
  `h_low_rank_parameter_grad_arithmetic` and
  `l_low_rank_parameter_grad_arithmetic`, with stable labels `fp32`,
  `native-fp16`, `native-fp16-widened-product`, and
  `native-fp16-compensated-operands`. The older native-FP16 active booleans stay
  in the summary for compatibility. The standalone qualification harness sweeps
  several row/rank/width geometries by default and compares every available mode
  against an isolated FP32 dW reference while reporting submission throughput
  and Vulkan allocator deltas:

  ```powershell
  cargo run --release --bin hierarchos-vulkan-rwkv-low-rank-dw-bench
  cargo run --release --bin hierarchos-vulkan-rwkv-low-rank-dw-bench -- --geometry 32x96x448 --geometry 64x128x448 --modes fp32,compensated --warmup 16 --iterations 128
  $env:HIERARCHOS_VULKAN_PROFILE_KERNELS = "1"
  cargo run --release --bin hierarchos-vulkan-rwkv-low-rank-dw-bench -- --geometry 32x96x448 --modes compensated --warmup 16 --iterations 128
  ```

  The host-side timing intentionally records repeated resident-buffer dispatches
  into one submission and excludes allocation/readback from the measured region.
  `HIERARCHOS_VULKAN_PROFILE_KERNELS=1` adds Vulkan timestamp-query measurements
  on stderr for pure GPU-kernel qualification. `kernel_resident_bytes` is the
  logical input + adjoint + dW working set; allocator live/reserved deltas are
  reported separately so arena slack is not confused with tensor storage.
  This source-scaled restriction is intentional: without scaling, the
  zero-initialized first-stage matrices can produce roughly `1e-7` learning
  signals early in training, inside FP16's subnormal range; promoting the
  inter-stage adjoint caused device-vs-PyTorch sign changes large enough for
  AdamW to choose the opposite first-step direction. Recurrent state updates,
  nonlinear/state-sensitive backward math, FP32 gradient accumulators, AdamW
  masters/moments, and SafeTensors checkpoint tensors are unchanged.

  The next recurrent seam is now available as an explicit experimental arm:
  `HIERARCHOS_VULKAN_NATIVE_FP16_RECURRENT_PROJECTION_BACKWARD=1`. It moves the
  three `receptance/key/value` projection input adjoints onto a native Float16
  multiply without changing the canonical FP32 PyTorch-layout recurrent
  weights. Each incoming recurrent adjoint is lifted by up to 1024x before the
  Float16 multiply, the product is widened immediately, and accumulation plus
  unscale remain FP32. The lift is reduced when needed to keep the half operand
  and product in range, while exceptional values fall back to the canonical
  FP32 product instead of manufacturing an FP16 overflow. This specifically
  targets Vulkan implementations that flush FP16 subnormals: a smallest-normal
  lift of 1024 covers the useful IEEE-half subnormal interval without requiring
  device-specific denormal behavior at the checkpoint boundary. The fused
  FP32 projection/time-mix backward path is bypassed while this arm is active so
  the native-half projection `dX` cannot be silently skipped by the scheduler.
  The arm defaults off until broader model-scale parity gates are accumulated.

  `HIERARCHOS_VULKAN_NATIVE_FP16_OUT_NORM_BACKWARD=0`,
  `HIERARCHOS_VULKAN_NATIVE_FP16_PROJECTION_BACKWARD=0`, and
  `HIERARCHOS_VULKAN_NATIVE_FP16_LOW_RANK_BACKWARD=0` are independent diagnostic
  kill-switches for those promoted seams; all three default on under
  `fp16-storage-fp16-lm-backward`. `HierarchosTrainingGraphSummary` exposes the
  LM-head, `out_norm`, dense-projection, and H/L low-rank native-backward active
  bits so benchmark/profile records cannot silently conflate the storage-only,
  partially native, and promoted mixed-precision paths. It additionally exposes
  independent H/L recurrent-projection native-FP16 active bits for the opt-in
  loss-scaled `r/k/v dX` experiment plus H/L low-rank parameter-gradient active
  bits for `HIERARCHOS_VULKAN_NATIVE_FP16_LOW_RANK_PARAMETER_GRAD=1`, plus the
  exact H/L low-rank dW arithmetic labels described above.

  `cases/rwkv_time_mix_fp16_projection_parity.json` is a deterministic
  subnormal-danger regression fixture for that seam. On the AMD Radeon Graphics
  development device the opt-in arm remains finite and does not erase any
  nonzero `grad_x_norm` or `grad_previous` entries. Against the strict FP32
  projection path its maximum absolute differences are `6.113314e-11` for
  `grad_x_norm` and `6.843619e-11` for `grad_previous`; all three recurrent
  projection weight gradients are bit-identical because their FP32
  checkpoint-facing reduction path is unchanged. The compiled SPIR-V contains
  an actual 16-bit `OpFMul` followed by `OpFConvert` back to FP32, so this is a
  native-half arithmetic arm rather than storage-only emulation.

  The matching PyTorch oracle is
  `python tools/verify_vulkan_worker_refinement_loss_parity.py --precision fp16-storage-fp16-lm-backward`.
  On the AMD Radeon Graphics development device the complete worker-refinement
  fixture passes with `1.79412842e-07` loss absolute difference,
  `8.28027725e-04` worst graph-adjoint difference, and `3.34982760e-05` worst
  post-step parameter difference with the promoted LM + out-norm + projection-
  parameter-gradient + first-stage-low-rank-adjoint contract active. Its
  deferred dynamic loss-scale close remains
  exact at the optimizer boundary (`parameter_diff=0`, `moment_diff=0`). The
  raw-token checkpoint/reload gate accepts the same precision and writes the
  ordinary FP32 training package; PyTorch and native Rust reload it with
  `2.38418579e-07` maximum inference-logit error on the same host. The CUDA
  reload leg remains automatic when an NVIDIA runtime is present; this AMD
  development host reports that leg as an explicit skip.
  The current AMD development driver advertises subgroup sizes 32..64. With
  `HIERARCHOS_VULKAN_REQUIRED_SUBGROUP_SIZE=32`, the head-64 paired-wave branch
  is therefore executable directly: forced WG64 (one pair) and WG128 (two
  pairs) both pass the four-step PyTorch TBPTT fixture with
  `5.96046448e-08` maximum output error, `1.78813934e-07` maximum packed-state
  error, and `4.47034836e-08` maximum input-gradient error. The corresponding
  two-update persistent AdamW fixture also passes for both geometries across 33
  tensors with `5.70900738e-07` maximum parameter error, and the Vulkan-written
  trained-cell SafeTensors package remains PyTorch/native-interchangeable.
  It is compiled only for Vulkan devices exposing compute-stage subgroup
  arithmetic and uses Vulkan 1.1 / SPIR-V 1.3 binaries. Numerics is persisted as
  part of the same online tape-policy identity: legacy records mean
  `strict-parity`, while every fast arm must earn its own measured throughput.
  The TBPTT parity runner reports the selected numerics and kernel geometry in
  its machine-readable result. On the AMD development GPU, an explicitly active
  head-64/WG128 `fast-recurrent-tiled` run matches PyTorch through four TBPTT
  steps within `1.78813934e-07` maximum packed-state error, and two persistent
  AdamW updates remain within `5.70900738e-07` maximum parameter error while the
  trained SafeTensors package round-trips successfully. The parity fixture also
  accepts `--heads` and `--head-size`; at head-32/WG64, where two cooperative
  lanes are likewise active, strict/tree/tiled all complete two AdamW updates
  within `4.76837158e-07` maximum parameter error and preserve the same
  SafeTensors interchange contract.
  The new multi-wave recurrent-subgroup arm is also qualified directly: on the
  wave64 AMD target, forced head-64/WG128 `fast-recurrent-subgroup` matches
  PyTorch through four TBPTT steps with `5.96046448e-08` maximum output error,
  `1.78813934e-07` maximum packed-state error, and `4.47034836e-08` maximum
  input-gradient error. An isolated release microprofile shows why WG128 is now
  worth exposing to the policy on this device: relative to the exact-wave WG64
  recurrent-subgroup arm, WG128 improves the batch-1 shallow fused state kernel
  from `0.0683 ms` to `0.0598 ms` and the deepest batch-1 key-reduce fusion from
  `0.0733 ms` to `0.0574 ms`; at batch 2 the corresponding measurements improve
  from `0.0740 ms` to `0.0599 ms` and `0.0816 ms` to `0.0608 ms`. These are local
  microprofile results, not a universal preference: other numerics arms still
  win individual schedules, so geometry plus numerics remain independently
  measured dimensions of the end-to-end policy.
  On a five-sample rerun of the current AMD development fixture, strict WG64
  measured 22.9428 ms per optimizer step versus 23.4906 ms for the first
  subgroup strategy (about 2.4% slower), so the
  self-optimizing policy correctly has evidence to reject that arm on this GPU
  instead of treating subgroup arithmetic as universally faster.
  The recurrent tree/tiled experiment reaches the same conclusion about policy
  discipline: short isolated runs can make tiled look faster, but interleaved
  profile records are noisy enough that the confidence-ranked scheduler retains
  `strict-parity` on both the 32-wide/WG64 fixture and the widened
  head-64/WG128 fixture. The profiler accepts `--fixture-width` so this reduction
  geometry can be exercised without changing checkpoint compatibility, and
  `--numerics` is a comma-separated matrix axis whose records carry geometry
  policy revision 2 plus the exact reduction policy.
The older compatibility training entrypoints deliberately retain their legacy
optimizer islands while this full-sequence API replaces them.

## Build shaders

The checked-in `.spv` files make ordinary Cargo builds independent of the Vulkan
SDK. When editing shader sources, regenerate them with Khronos `glslc`:

```powershell
glslc shaders/linear_forward.comp -o shaders/linear_forward.spv
glslc shaders/linear_forward_fp16_packed.comp -o shaders/linear_forward_fp16_packed.spv
glslc shaders/linear_bias_forward.comp -o shaders/linear_bias_forward.spv
glslc shaders/linear_bias_forward_fp16_packed.comp -o shaders/linear_bias_forward_fp16_packed.spv
glslc shaders/linear_residual_forward.comp -o shaders/linear_residual_forward.spv
glslc shaders/linear3_forward.comp -o shaders/linear3_forward.spv
glslc shaders/linear3_input_grad.comp -o shaders/linear3_input_grad.spv
glslc shaders/linear3_time_mix_backward_fused.comp -o shaders/linear3_time_mix_backward_fused.spv
glslc shaders/linear3_weight_grad.comp -o shaders/linear3_weight_grad.spv
glslc shaders/cross_entropy_grad.comp -o shaders/cross_entropy_grad.spv
glslc shaders/cross_entropy_linear_row_stats_streaming.comp -o shaders/cross_entropy_linear_row_stats_streaming.spv
glslc shaders/cross_entropy_linear_row_stats_streaming_fp16_packed.comp -o shaders/cross_entropy_linear_row_stats_streaming_fp16_packed.spv
glslc -DLM_CE_WRITE_LOGIT_TAPE=1 shaders/cross_entropy_linear_row_stats_streaming_fp16_packed.comp -o shaders/cross_entropy_linear_row_stats_streaming_fp16_packed_tape.spv
glslc shaders/cross_entropy_linear_logit_tape_fp16_packed_rows8.comp -o shaders/cross_entropy_linear_logit_tape_fp16_packed_rows8.spv
glslc -DLM_CE_VOCAB_TILE=16 -DLM_CE_PACKED_SHARED=1 shaders/cross_entropy_linear_logit_tape_fp16_packed_rows8.comp -o shaders/cross_entropy_linear_logit_tape_fp16_packed_rows16.spv
glslc -DLM_CE_VOCAB_TILE=16 -DLM_CE_PACKED_SHARED=1 -DLM_CE_DOT_LANES_PER_VOCAB=4 shaders/cross_entropy_linear_logit_tape_fp16_packed_rows8.comp -o shaders/cross_entropy_linear_logit_tape_fp16_packed_rows16_dot4.spv
glslc --target-env=vulkan1.1 -DLM_CE_VOCAB_TILE=16 -DLM_CE_PACKED_SHARED=1 -DLM_CE_DOT_LANES_PER_VOCAB=4 -DLM_CE_USE_CLUSTERED_REDUCTION=1 shaders/cross_entropy_linear_logit_tape_fp16_packed_rows8.comp -o shaders/cross_entropy_linear_logit_tape_fp16_packed_rows16_cluster4.spv
glslc shaders/cross_entropy_row_stats_tile_partials.comp -o shaders/cross_entropy_row_stats_tile_partials.spv
glslc -DLM_CE_VOCAB_TILE=16 shaders/cross_entropy_row_stats_tile_partials.comp -o shaders/cross_entropy_row_stats_tile_partials_rows16.spv
glslc shaders/cross_entropy_logits_to_grad_inplace.comp -o shaders/cross_entropy_logits_to_grad_inplace.spv
glslc shaders/cross_entropy_linear_weight_grad_streaming.comp -o shaders/cross_entropy_linear_weight_grad_streaming.spv
glslc -DLM_DW_VOCAB_ROWS_PER_GROUP=4 shaders/cross_entropy_linear_weight_grad_streaming_fp16_packed.comp -o shaders/cross_entropy_linear_weight_grad_streaming_fp16_packed_rows4.spv
glslc shaders/cross_entropy_linear_weight_grad_streaming_fp16_packed.comp -o shaders/cross_entropy_linear_weight_grad_streaming_fp16_packed.spv
glslc -DLM_DW_VOCAB_ROWS_PER_GROUP=16 shaders/cross_entropy_linear_weight_grad_streaming_fp16_packed.comp -o shaders/cross_entropy_linear_weight_grad_streaming_fp16_packed_rows16.spv
glslc -DLM_DW_VOCAB_ROWS_PER_GROUP=4 shaders/cross_entropy_linear_weight_grad_tape_fp16_packed.comp -o shaders/cross_entropy_linear_weight_grad_tape_fp16_packed_rows4.spv
glslc shaders/cross_entropy_linear_weight_grad_tape_fp16_packed.comp -o shaders/cross_entropy_linear_weight_grad_tape_fp16_packed.spv
glslc -DLM_DW_VOCAB_ROWS_PER_GROUP=16 shaders/cross_entropy_linear_weight_grad_tape_fp16_packed.comp -o shaders/cross_entropy_linear_weight_grad_tape_fp16_packed_rows16.spv
glslc shaders/cross_entropy_linear_input_grad_streaming.comp -o shaders/cross_entropy_linear_input_grad_streaming.spv
glslc shaders/cross_entropy_linear_input_grad_streaming_fp16_packed.comp -o shaders/cross_entropy_linear_input_grad_streaming_fp16_packed.spv
glslc shaders/cross_entropy_linear_input_grad_tape_fp16_packed.comp -o shaders/cross_entropy_linear_input_grad_tape_fp16_packed.spv
glslc shaders/cross_entropy_linear_adjoints_tape_fp16_packed_fused.comp -o shaders/cross_entropy_linear_adjoints_tape_fp16_packed_fused.spv
glslc -DLM_CE_FUSED_PRIVATE_HIDDEN=1 shaders/cross_entropy_linear_adjoints_tape_fp16_packed_fused.comp -o shaders/cross_entropy_linear_adjoints_tape_fp16_packed_fused_private_hidden.spv
glslc -DLM_CE_FUSED_PRIVATE_HIDDEN=1 -DLM_CE_FUSED_VOCAB_TILE=128 shaders/cross_entropy_linear_adjoints_tape_fp16_packed_fused.comp -o shaders/cross_entropy_linear_adjoints_tape_fp16_packed_fused_private_hidden_tile128.spv
glslc -DLM_CE_FUSED_PRIVATE_HIDDEN=1 -DLM_CE_FUSED_VOCAB_TILE=128 -DLM_CE_FUSED_WORKGROUP_SIZE=128 shaders/cross_entropy_linear_adjoints_tape_fp16_packed_fused.comp -o shaders/cross_entropy_linear_adjoints_tape_fp16_packed_fused_private_hidden_tile128_wg128.spv
glslc -DLM_CE_FUSED_PRIVATE_HIDDEN=1 -DLM_CE_FUSED_VOCAB_TILE=256 shaders/cross_entropy_linear_adjoints_tape_fp16_packed_fused.comp -o shaders/cross_entropy_linear_adjoints_tape_fp16_packed_fused_private_hidden_tile256.spv
glslc -DLM_CE_FUSED_PRIVATE_HIDDEN=1 -DLM_CE_FUSED_VOCAB_TILE=256 -DLM_CE_FUSED_WORKGROUP_SIZE=128 shaders/cross_entropy_linear_adjoints_tape_fp16_packed_fused.comp -o shaders/cross_entropy_linear_adjoints_tape_fp16_packed_fused_private_hidden_tile256_wg128.spv
glslc -DLM_CE_FUSED_PRIVATE_HIDDEN=1 -DLM_CE_FUSED_VOCAB_TILE=256 -DLM_CE_FUSED_WORKGROUP_SIZE=256 shaders/cross_entropy_linear_adjoints_tape_fp16_packed_fused.comp -o shaders/cross_entropy_linear_adjoints_tape_fp16_packed_fused_private_hidden_tile256_wg256.spv
glslc shaders/cross_entropy_linear_input_grad_tile_reduce.comp -o shaders/cross_entropy_linear_input_grad_tile_reduce.spv
glslc -DLM_CE_FUSED_VOCAB_TILE=128 shaders/cross_entropy_linear_input_grad_tile_reduce.comp -o shaders/cross_entropy_linear_input_grad_tile_reduce_tile128.spv
glslc -DLM_CE_FUSED_VOCAB_TILE=256 shaders/cross_entropy_linear_input_grad_tile_reduce.comp -o shaders/cross_entropy_linear_input_grad_tile_reduce_tile256.spv
glslc --target-env=vulkan1.1 shaders/cross_entropy_linear_input_grad_streaming_fp16_native.comp -o shaders/cross_entropy_linear_input_grad_streaming_fp16_native.spv
glslc --target-env=vulkan1.1 -DLM_WEIGHT_REUSE_PAIRS=64 shaders/cross_entropy_linear_input_grad_streaming_fp16_native.comp -o shaders/cross_entropy_linear_input_grad_streaming_fp16_native_reuse64.spv
glslc --target-env=vulkan1.1 -DLM_WEIGHT_REUSE_PAIRS=128 shaders/cross_entropy_linear_input_grad_streaming_fp16_native.comp -o shaders/cross_entropy_linear_input_grad_streaming_fp16_native_reuse128.spv
glslc --target-env=vulkan1.1 -DLM_WEIGHT_REUSE_PAIRS=224 shaders/cross_entropy_linear_input_grad_streaming_fp16_native.comp -o shaders/cross_entropy_linear_input_grad_streaming_fp16_native_reuse224.spv
glslc shaders/cross_entropy_row_loss_extract.comp -o shaders/cross_entropy_row_loss_extract.spv
glslc shaders/linear_weight_grad.comp -o shaders/linear_weight_grad.spv
glslc shaders/linear_input_grad.comp -o shaders/linear_input_grad.spv
glslc shaders/linear_input_grad_fp16_packed.comp -o shaders/linear_input_grad_fp16_packed.spv
glslc --target-env=vulkan1.1 -DHIERARCHOS_NATIVE_FP16_BACKWARD_COMPUTE=1 shaders/linear_input_grad_fp16_packed.comp -o shaders/linear_input_grad_fp16_native_compute.spv
glslc shaders/layer_norm_forward.comp -o shaders/layer_norm_forward.spv
glslc shaders/layer_norm_linear_forward_fused.comp -o shaders/layer_norm_linear_forward_fused.spv
glslc shaders/layer_norm_linear_forward_fused_fp16_packed.comp -o shaders/layer_norm_linear_forward_fused_fp16_packed.spv
glslc shaders/layer_norm_linear_silu_forward_fused.comp -o shaders/layer_norm_linear_silu_forward_fused.spv
glslc -DADAPTER_WORKGROUP_SIZE=256 shaders/layer_norm_adapter_forward_fused.comp -o shaders/layer_norm_adapter_forward_fused_256.spv
glslc -DADAPTER_WORKGROUP_SIZE=512 shaders/layer_norm_adapter_forward_fused.comp -o shaders/layer_norm_adapter_forward_fused_512.spv
glslc --target-env=vulkan1.1 shaders/rosa_predict_bounded.comp -o shaders/rosa_predict_bounded.spv
glslc --target-env=vulkan1.1 shaders/rosa_predict_bounded_lanes.comp -o shaders/rosa_predict_bounded_lanes.spv
glslc --target-env=vulkan1.1 -DROSA_USE_SUBGROUP_REDUCTION=1 -DROSA_WORKGROUP_SIZE=128 shaders/rosa_predict_bounded.comp -o shaders/rosa_predict_bounded_subgroup_128.spv
glslc --target-env=vulkan1.1 -DROSA_USE_SUBGROUP_REDUCTION=1 -DROSA_WORKGROUP_SIZE=128 shaders/rosa_predict_bounded_lanes.comp -o shaders/rosa_predict_bounded_lanes_subgroup_128.spv
glslc --target-env=vulkan1.1 -DROSA_USE_SUBGROUP_REDUCTION=1 -DROSA_WORKGROUP_SIZE=128 -DROSA_CACHE_HISTORY_STATE=1 shaders/rosa_predict_bounded.comp -o shaders/rosa_predict_bounded_subgroup_128_cache_tiled.spv
glslc --target-env=vulkan1.1 -DROSA_USE_SUBGROUP_REDUCTION=1 -DROSA_WORKGROUP_SIZE=128 -DROSA_CACHE_HISTORY_STATE=1 shaders/rosa_predict_bounded_lanes.comp -o shaders/rosa_predict_bounded_lanes_subgroup_128_cache_tiled.spv
glslc --target-env=vulkan1.1 -DROSA_USE_SUBGROUP_REDUCTION=1 -DROSA_WORKGROUP_SIZE=256 -DROSA_SINGLE_PACKED_PAIR_PER_LANE=1 shaders/rosa_predict_bounded.comp -o shaders/rosa_predict_bounded_subgroup_256_single_pair.spv
glslc --target-env=vulkan1.1 -DROSA_USE_SUBGROUP_REDUCTION=1 -DROSA_WORKGROUP_SIZE=256 -DROSA_SINGLE_PACKED_PAIR_PER_LANE=1 shaders/rosa_predict_bounded_lanes.comp -o shaders/rosa_predict_bounded_lanes_subgroup_256_single_pair.spv
glslc --target-env=vulkan1.1 -DROSA_USE_SUBGROUP_REDUCTION=1 -DROSA_WORKGROUP_SIZE=256 -DROSA_SINGLE_PACKED_PAIR_PER_LANE=1 -DROSA_CACHE_HISTORY_STATE=1 shaders/rosa_predict_bounded.comp -o shaders/rosa_predict_bounded_subgroup_256_single_pair_cache_tiled.spv
glslc --target-env=vulkan1.1 -DROSA_USE_SUBGROUP_REDUCTION=1 -DROSA_WORKGROUP_SIZE=256 -DROSA_SINGLE_PACKED_PAIR_PER_LANE=1 -DROSA_CACHE_HISTORY_STATE=1 shaders/rosa_predict_bounded_lanes.comp -o shaders/rosa_predict_bounded_lanes_subgroup_256_single_pair_cache_tiled.spv
glslc shaders/layer_norm_input_grad.comp -o shaders/layer_norm_input_grad.spv
glslc shaders/layer_norm_affine_clamp_backward_inplace.comp -o shaders/layer_norm_affine_clamp_backward_inplace.spv
glslc shaders/layer_norm_input_grad_residual_fused.comp -o shaders/layer_norm_input_grad_residual_fused.spv
glslc shaders/layer_norm_param_grad.comp -o shaders/layer_norm_param_grad.spv
glslc shaders/layer_norm_backward_fused_add.comp -o shaders/layer_norm_backward_fused_add.spv
glslc shaders/layer_norm_backward_fused_add_residual.comp -o shaders/layer_norm_backward_fused_add_residual.spv
glslc shaders/layer_norm_channel_mix_forward_fused.comp -o shaders/layer_norm_channel_mix_forward_fused.spv
glslc shaders/embedding_forward.comp -o shaders/embedding_forward.spv
glslc shaders/embedding_grad_accumulate.comp -o shaders/embedding_grad_accumulate.spv
glslc shaders/embedding_token_sort.comp -o shaders/embedding_token_sort.spv
glslc shaders/embedding_radix_histogram.comp -o shaders/embedding_radix_histogram.spv
glslc shaders/embedding_radix_prefix.comp -o shaders/embedding_radix_prefix.spv
glslc shaders/embedding_radix_scatter.comp -o shaders/embedding_radix_scatter.spv
glslc shaders/embedding_grad_segmented.comp -o shaders/embedding_grad_segmented.spv
glslc --target-env=vulkan1.1 shaders/ltm_topk.comp -o shaders/ltm_topk.spv
glslc shaders/ltm_similarity_key_grad.comp -o shaders/ltm_similarity_key_grad.spv
glslc shaders/rosa_gate_backward.comp -o shaders/rosa_gate_backward.spv
glslc shaders/rosa_gate_grad_reduce.comp -o shaders/rosa_gate_grad_reduce.spv
glslc shaders/silu_forward.comp -o shaders/silu_forward.spv
glslc shaders/silu_backward.comp -o shaders/silu_backward.spv
glslc shaders/bias_add.comp -o shaders/bias_add.spv
glslc shaders/bias_grad.comp -o shaders/bias_grad.spv
glslc --target-env=vulkan1.1 -DHIERARCHOS_NATIVE_FP16_BACKWARD_COMPUTE=1 shaders/bias_grad.comp -o shaders/bias_grad_fp16_native_compute.spv
glslc shaders/rwkv_matrix_state_forward.comp -o shaders/rwkv_matrix_state_forward.spv
glslc shaders/rwkv_matrix_state_backward_rows.comp -o shaders/rwkv_matrix_state_backward_rows.spv
glslc shaders/rwkv_matrix_state_backward_cols.comp -o shaders/rwkv_matrix_state_backward_cols.spv
glslc shaders/rwkv_matrix_state_backward_fused_rk_add.comp -o shaders/rwkv_matrix_state_backward_fused_rk_add.spv
glslc shaders/rwkv_matrix_state_backward_fused_rkv_add3.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3.spv
glslc -DRWKV_STATE_BACKWARD_WORKGROUP_SIZE=32 shaders/rwkv_matrix_state_backward_fused_rkv_add3.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3_wg32.spv
glslc -DRWKV_STATE_BACKWARD_WORKGROUP_SIZE=128 shaders/rwkv_matrix_state_backward_fused_rkv_add3.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3_wg128.spv
glslc shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform.spv
glslc -DRWKV_STATE_BACKWARD_WORKGROUP_SIZE=32 shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_wg32.spv
glslc -DRWKV_STATE_BACKWARD_WORKGROUP_SIZE=128 shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_wg128.spv
glslc -DFUSE_KEY_PARAM_REDUCE=1 shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce.spv
glslc -DFUSE_KEY_PARAM_REDUCE=1 -DRWKV_STATE_BACKWARD_WORKGROUP_SIZE=32 shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce_wg32.spv
glslc -DFUSE_KEY_PARAM_REDUCE=1 -DRWKV_STATE_BACKWARD_WORKGROUP_SIZE=128 shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce_wg128.spv
glslc --target-env=vulkan1.1 -DRWKV_FAST_SUBGROUP_KEY_NORM=1 shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_subgroup.spv
glslc --target-env=vulkan1.1 -DRWKV_FAST_SUBGROUP_KEY_NORM=1 -DRWKV_STATE_BACKWARD_WORKGROUP_SIZE=32 shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_subgroup_wg32.spv
glslc --target-env=vulkan1.1 -DRWKV_FAST_SUBGROUP_KEY_NORM=1 -DRWKV_STATE_BACKWARD_WORKGROUP_SIZE=128 shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_subgroup_wg128.spv
glslc --target-env=vulkan1.1 -DRWKV_FAST_SUBGROUP_KEY_NORM=1 -DFUSE_KEY_PARAM_REDUCE=1 shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce_subgroup.spv
glslc --target-env=vulkan1.1 -DRWKV_FAST_SUBGROUP_KEY_NORM=1 -DFUSE_KEY_PARAM_REDUCE=1 -DRWKV_STATE_BACKWARD_WORKGROUP_SIZE=32 shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce_subgroup_wg32.spv
glslc --target-env=vulkan1.1 -DRWKV_FAST_SUBGROUP_KEY_NORM=1 -DFUSE_KEY_PARAM_REDUCE=1 -DRWKV_STATE_BACKWARD_WORKGROUP_SIZE=128 shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce_subgroup_wg128.spv
glslc --target-env=vulkan1.1 -DRWKV_FAST_SUBGROUP_RECURRENT_FUSION=1 shaders/rwkv_matrix_state_backward_fused_rkv_add3.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3_recurrent_subgroup.spv
glslc --target-env=vulkan1.1 -DRWKV_FAST_SUBGROUP_RECURRENT_FUSION=1 -DRWKV_STATE_BACKWARD_WORKGROUP_SIZE=32 shaders/rwkv_matrix_state_backward_fused_rkv_add3.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3_recurrent_subgroup_wg32.spv
glslc --target-env=vulkan1.1 -DRWKV_FAST_SUBGROUP_RECURRENT_FUSION=1 -DRWKV_STATE_BACKWARD_WORKGROUP_SIZE=128 shaders/rwkv_matrix_state_backward_fused_rkv_add3.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3_recurrent_subgroup_wg128.spv
glslc --target-env=vulkan1.1 -DRWKV_FAST_SUBGROUP_RECURRENT_FUSION=1 shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_recurrent_subgroup.spv
glslc --target-env=vulkan1.1 -DRWKV_FAST_SUBGROUP_RECURRENT_FUSION=1 -DRWKV_STATE_BACKWARD_WORKGROUP_SIZE=32 shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_recurrent_subgroup_wg32.spv
glslc --target-env=vulkan1.1 -DRWKV_FAST_SUBGROUP_RECURRENT_FUSION=1 -DRWKV_STATE_BACKWARD_WORKGROUP_SIZE=128 shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_recurrent_subgroup_wg128.spv
glslc --target-env=vulkan1.1 -DRWKV_FAST_SUBGROUP_RECURRENT_FUSION=1 -DFUSE_KEY_PARAM_REDUCE=1 shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce_recurrent_subgroup.spv
glslc --target-env=vulkan1.1 -DRWKV_FAST_SUBGROUP_RECURRENT_FUSION=1 -DFUSE_KEY_PARAM_REDUCE=1 -DRWKV_STATE_BACKWARD_WORKGROUP_SIZE=32 shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce_recurrent_subgroup_wg32.spv
glslc --target-env=vulkan1.1 -DRWKV_FAST_SUBGROUP_RECURRENT_FUSION=1 -DFUSE_KEY_PARAM_REDUCE=1 -DRWKV_STATE_BACKWARD_WORKGROUP_SIZE=128 shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform.comp -o shaders/rwkv_matrix_state_backward_fused_rkv_add3_key_transform_reduce_recurrent_subgroup_wg128.spv
glslc shaders/rwkv_time_mix3_forward.comp -o shaders/rwkv_time_mix3_forward.spv
glslc shaders/rwkv_low_rank_producer_forward_fused.comp -o shaders/rwkv_low_rank_producer_forward_fused.spv
glslc shaders/rwkv_low_rank_producer_forward_fused_fp16_packed.comp -o shaders/rwkv_low_rank_producer_forward_fused_fp16_packed.spv
glslc shaders/rwkv_low_rank_full_forward_fused_fp16_packed.comp -o shaders/rwkv_low_rank_full_forward_fused_fp16_packed.spv
glslc --target-env=vulkan1.1 -DRWKV_LOW_RANK_FIRST_STAGE_SUBGROUP_PACKED_SHARE=1 shaders/rwkv_low_rank_full_forward_fused_fp16_packed.comp -o shaders/rwkv_low_rank_full_forward_fused_fp16_packed_subgroup.spv
glslc shaders/layer_norm_low_rank_producer_forward_fused.comp -o shaders/layer_norm_low_rank_producer_forward_fused.spv
glslc shaders/layer_norm_low_rank_producer_forward_fused_fp16_packed.comp -o shaders/layer_norm_low_rank_producer_forward_fused_fp16_packed.spv
glslc shaders/layer_norm_low_rank_producer_forward_fused_fp16_packed_two_rows.comp -o shaders/layer_norm_low_rank_producer_forward_fused_fp16_packed_two_rows.spv
glslc shaders/rwkv_time_mix3_linear3_key_state_forward_fused.comp -o shaders/rwkv_time_mix3_linear3_key_state_forward_fused.spv
glslc -DRWKV_TIME_MIX_FORWARD_WORKGROUP_SIZE=32 shaders/rwkv_time_mix3_linear3_key_state_forward_fused.comp -o shaders/rwkv_time_mix3_linear3_key_state_forward_fused_wg32.spv
glslc shaders/rwkv_time_mix3_linear3_key_state_forward_fused_two_rows.comp -o shaders/rwkv_time_mix3_linear3_key_state_forward_fused_two_rows.spv
glslc shaders/rwkv_time_mix3_linear3_key_state_forward_weight_reuse2.comp -o shaders/rwkv_time_mix3_linear3_key_state_forward_weight_reuse2.spv
glslc -DRWKV_WEIGHT_REUSE_TILE=32 shaders/rwkv_time_mix3_linear3_key_state_forward_weight_reuse2.comp -o shaders/rwkv_time_mix3_linear3_key_state_forward_weight_reuse2_tile32.spv
glslc -DRWKV_WEIGHT_REUSE_TILE=64 shaders/rwkv_time_mix3_linear3_key_state_forward_weight_reuse2.comp -o shaders/rwkv_time_mix3_linear3_key_state_forward_weight_reuse2_tile64.spv
glslc shaders/rwkv_time_mix3_linear3_key_state_forward_packed_fast.comp -o shaders/rwkv_time_mix3_linear3_key_state_forward_packed_fast.spv
glslc -DRWKV_PACKED_FAST_WORKGROUP_SIZE=32 shaders/rwkv_time_mix3_linear3_key_state_forward_packed_fast.comp -o shaders/rwkv_time_mix3_linear3_key_state_forward_packed_fast_wg32.spv
glslc shaders/rwkv_time_mix3_linear3_key_state_forward_packed_fast_weight_reuse2.comp -o shaders/rwkv_time_mix3_linear3_key_state_forward_packed_fast_weight_reuse2.spv
glslc -DRWKV_WEIGHT_REUSE_TILE=32 shaders/rwkv_time_mix3_linear3_key_state_forward_packed_fast_weight_reuse2.comp -o shaders/rwkv_time_mix3_linear3_key_state_forward_packed_fast_weight_reuse2_tile32.spv
glslc -DRWKV_WEIGHT_REUSE_TILE=64 shaders/rwkv_time_mix3_linear3_key_state_forward_packed_fast_weight_reuse2.comp -o shaders/rwkv_time_mix3_linear3_key_state_forward_packed_fast_weight_reuse2_tile64.spv
glslc shaders/rwkv_time_mix3_backward.comp -o shaders/rwkv_time_mix3_backward.spv
glslc -DRWKV_TIME_MIX_BACKWARD_WORKGROUP_SIZE=32 shaders/rwkv_time_mix3_backward.comp -o shaders/rwkv_time_mix3_backward_wg32.spv
glslc -DRWKV_TIME_MIX_BACKWARD_WORKGROUP_SIZE=128 shaders/rwkv_time_mix3_backward.comp -o shaders/rwkv_time_mix3_backward_wg128.spv
glslc shaders/rwkv_time_mix3_backward_fused_add.comp -o shaders/rwkv_time_mix3_backward_fused_add.spv
glslc -DRWKV_TIME_MIX_BACKWARD_WORKGROUP_SIZE=32 shaders/rwkv_time_mix3_backward_fused_add.comp -o shaders/rwkv_time_mix3_backward_fused_add_wg32.spv
glslc -DRWKV_TIME_MIX_BACKWARD_WORKGROUP_SIZE=128 shaders/rwkv_time_mix3_backward_fused_add.comp -o shaders/rwkv_time_mix3_backward_fused_add_wg128.spv
glslc shaders/rwkv_time_mix3_backward_fused_add_outer.comp -o shaders/rwkv_time_mix3_backward_fused_add_outer.spv
glslc -DRWKV_TIME_MIX_BACKWARD_WORKGROUP_SIZE=32 shaders/rwkv_time_mix3_backward_fused_add_outer.comp -o shaders/rwkv_time_mix3_backward_fused_add_outer_wg32.spv
glslc -DRWKV_TIME_MIX_BACKWARD_WORKGROUP_SIZE=128 shaders/rwkv_time_mix3_backward_fused_add_outer.comp -o shaders/rwkv_time_mix3_backward_fused_add_outer_wg128.spv
glslc shaders/rwkv_key_transform_forward.comp -o shaders/rwkv_key_transform_forward.spv
glslc shaders/rwkv_key_transform_backward.comp -o shaders/rwkv_key_transform_backward.spv
glslc --target-env=vulkan1.1 -DRWKV_KEY_TRANSFORM_USE_SUBGROUP_REDUCTION=1 shaders/rwkv_key_transform_backward.comp -o shaders/rwkv_key_transform_backward_subgroup.spv
glslc shaders/rwkv_key_transform_param_reduce.comp -o shaders/rwkv_key_transform_param_reduce.spv
glslc shaders/parameter_matmul_forward.comp -o shaders/parameter_matmul_forward.spv
glslc shaders/parameter_matmul_input_grad.comp -o shaders/parameter_matmul_input_grad.spv
glslc shaders/parameter_matmul_weight_grad.comp -o shaders/parameter_matmul_weight_grad.spv
glslc --target-env=vulkan1.1 -DHIERARCHOS_NATIVE_FP16_BACKWARD_COMPUTE=1 shaders/parameter_matmul_weight_grad.comp -o shaders/parameter_matmul_weight_grad_fp16_native_compute.spv
glslc --target-env=vulkan1.1 -DHIERARCHOS_NATIVE_FP16_BACKWARD_COMPUTE=1 -DHIERARCHOS_FP16_DW_WIDEN_PRODUCT=1 shaders/parameter_matmul_weight_grad.comp -o shaders/parameter_matmul_weight_grad_fp16_widened_compute.spv
glslc --target-env=vulkan1.1 -DHIERARCHOS_NATIVE_FP16_BACKWARD_COMPUTE=1 -DHIERARCHOS_FP16_DW_COMPENSATED_OPERANDS=1 shaders/parameter_matmul_weight_grad.comp -o shaders/parameter_matmul_weight_grad_fp16_compensated_compute.spv
glslc shaders/sigmoid_forward.comp -o shaders/sigmoid_forward.spv
glslc shaders/sigmoid_backward.comp -o shaders/sigmoid_backward.spv
glslc shaders/tanh_forward.comp -o shaders/tanh_forward.spv
glslc shaders/tanh_backward.comp -o shaders/tanh_backward.spv
glslc shaders/rwkv_decay_forward.comp -o shaders/rwkv_decay_forward.spv
glslc shaders/rwkv_decay_backward.comp -o shaders/rwkv_decay_backward.spv
glslc shaders/vector_add.comp -o shaders/vector_add.spv
glslc shaders/vector_add3.comp -o shaders/vector_add3.spv
glslc shaders/rwkv_group_norm_forward.comp -o shaders/rwkv_group_norm_forward.spv
glslc shaders/rwkv_group_norm_bonus_gate_forward_fused.comp -o shaders/rwkv_group_norm_bonus_gate_forward_fused.spv
glslc shaders/rwkv_group_norm_bonus_gate_linear_residual_forward_fused.comp -o shaders/rwkv_group_norm_bonus_gate_linear_residual_forward_fused.spv
glslc shaders/rwkv_group_norm_input_grad.comp -o shaders/rwkv_group_norm_input_grad.spv
glslc shaders/rwkv_group_norm_param_grad.comp -o shaders/rwkv_group_norm_param_grad.spv
glslc shaders/rwkv_bonus_gate_forward.comp -o shaders/rwkv_bonus_gate_forward.spv
glslc shaders/rwkv_bonus_gate_backward.comp -o shaders/rwkv_bonus_gate_backward.spv
glslc shaders/channel_reduce.comp -o shaders/channel_reduce.spv
glslc shaders/rwkv_channel_mix_forward.comp -o shaders/rwkv_channel_mix_forward.spv
glslc shaders/layer_norm_channel_mix_key_relu2_deepembed_forward_fused.comp -o shaders/layer_norm_channel_mix_key_relu2_deepembed_forward_fused.spv
glslc shaders/layer_norm_channel_mix_full_forward_fused.comp -o shaders/layer_norm_channel_mix_full_forward_fused.spv
glslc shaders/rwkv_channel_mix_backward.comp -o shaders/rwkv_channel_mix_backward.spv
glslc shaders/relu2_deepembed_forward.comp -o shaders/relu2_deepembed_forward.spv
glslc shaders/relu2_deepembed_backward.comp -o shaders/relu2_deepembed_backward.spv
glslc shaders/rwkv_state_unpack.comp -o shaders/rwkv_state_unpack.spv
glslc shaders/rwkv_state_unpack_vectors.comp -o shaders/rwkv_state_unpack_vectors.spv
glslc shaders/rwkv_state_pack.comp -o shaders/rwkv_state_pack.spv
glslc shaders/rwkv_state_pack_backward.comp -o shaders/rwkv_state_pack_backward.spv
glslc shaders/rwkv_state_pack_backward_fused_add.comp -o shaders/rwkv_state_pack_backward_fused_add.spv
glslc shaders/rwkv_state_grad_pack.comp -o shaders/rwkv_state_grad_pack.spv
glslc shaders/packed_cell_channel_mix_state_forward_fused.comp -o shaders/packed_cell_channel_mix_state_forward_fused.spv
glslc -DPACKED_CELL_FORWARD_WORKGROUP_SIZE=32 shaders/packed_cell_channel_mix_state_forward_fused.comp -o shaders/packed_cell_channel_mix_state_forward_fused_wg32.spv
glslc -DPACKED_CELL_FORWARD_WORKGROUP_SIZE=64 shaders/packed_cell_channel_mix_state_forward_fused.comp -o shaders/packed_cell_channel_mix_state_forward_fused_wg64.spv
glslc -DPACKED_CELL_FORWARD_WORKGROUP_SIZE=128 shaders/packed_cell_channel_mix_state_forward_fused.comp -o shaders/packed_cell_channel_mix_state_forward_fused_wg128.spv
glslc shaders/packed_state_slot_forward.comp -o shaders/packed_state_slot_forward.spv
glslc shaders/packed_state_slot_backward.comp -o shaders/packed_state_slot_backward.spv
glslc shaders/gradient_accumulate.comp -o shaders/gradient_accumulate.spv
glslc shaders/gradient_accumulate_scaled.comp -o shaders/gradient_accumulate_scaled.spv
glslc shaders/gradient_accumulate4.comp -o shaders/gradient_accumulate4.spv
glslc shaders/gradient_scale.comp -o shaders/gradient_scale.spv
glslc shaders/gradient_scale_from_buffer.comp -o shaders/gradient_scale_from_buffer.spv
glslc shaders/gradient_scale_from_buffer_indexed.comp -o shaders/gradient_scale_from_buffer_indexed.spv
glslc shaders/gradient_scale_strided_from_buffer.comp -o shaders/gradient_scale_strided_from_buffer.spv
glslc shaders/gradient_nonfinite_flag.comp -o shaders/gradient_nonfinite_flag.spv
glslc shaders/gradient_lassq_partials.comp -o shaders/gradient_lassq_partials.spv
glslc shaders/gradient_lassq_reduce.comp -o shaders/gradient_lassq_reduce.spv
glslc shaders/gradient_clip_coefficient.comp -o shaders/gradient_clip_coefficient.spv
glslc shaders/ordered_f32_sum.comp -o shaders/ordered_f32_sum.spv
glslc shaders/adamw.comp -o shaders/adamw.spv
glslc shaders/adamw_range.comp -o shaders/adamw_range.spv
glslc shaders/adamw_fp16_mirror.comp -o shaders/adamw_fp16_mirror.spv
glslc shaders/fp32_to_fp16_packed.comp -o shaders/fp32_to_fp16_packed.spv
glslc shaders/fp16_packed_to_fp32.comp -o shaders/fp16_packed_to_fp32.spv
glslc shaders/fp32_to_bf16_packed.comp -o shaders/fp32_to_bf16_packed.spv
glslc shaders/bf16_packed_to_fp32.comp -o shaders/bf16_packed_to_fp32.spv
glslc shaders/hard_act_select.comp -o shaders/hard_act_select.spv
glslc shaders/hard_act_depth_backward.comp -o shaders/hard_act_depth_backward.spv
glslc shaders/indexed_step_gather.comp -o shaders/indexed_step_gather.spv
glslc shaders/indexed_step_scatter_backward.comp -o shaders/indexed_step_scatter_backward.spv
glslc shaders/context_lerp_concat_forward.comp -o shaders/context_lerp_concat_forward.spv
glslc shaders/context_lerp_concat_backward.comp -o shaders/context_lerp_concat_backward.spv
glslc shaders/drift_update_forward.comp -o shaders/drift_update_forward.spv
glslc shaders/drift_update_backward.comp -o shaders/drift_update_backward.spv
glslc shaders/row_keep_forward.comp -o shaders/row_keep_forward.spv
glslc shaders/row_keep_backward.comp -o shaders/row_keep_backward.spv
glslc shaders/worker_convergence.comp -o shaders/worker_convergence.spv
glslc shaders/commitment_accumulate.comp -o shaders/commitment_accumulate.spv
glslc shaders/commitment_backward.comp -o shaders/commitment_backward.spv
glslc shaders/commitment_finalize.comp -o shaders/commitment_finalize.spv
glslc shaders/rosa_predict_bounded.comp -o shaders/rosa_predict_bounded.spv
glslc shaders/rosa_predict_bounded_lanes.comp -o shaders/rosa_predict_bounded_lanes.spv
```

The LM-head backward tuner now selects a complete FP16 backward plan: an
input-adjoint/CE-traffic arm plus a cooperative dW vocabulary fanout. The
`fp16-ce-tape` arm spills the logits already computed by CE forward into the
existing FP32 CE scratch and reuses each packed W^T tile across all rows. W^T
and dW now derive the mean-reduced CE adjoint directly from each preserved
logit plus the compact row statistics at the point of consumption; the old
global `logits -> grad_logits` rewrite is no longer part of production
submission. That removes one complete `rows * vocab` FP32 read+write pass while
preserving the same FP32 master/checkpoint contract and exact FP16 execution
weights. Force that diagnostic arm with
`HIERARCHOS_VULKAN_LM_EXECUTION_ARM=fp16-ce-tape`.
The cross-row `fp16-ce-tape-rows8` and `fp16-ce-tape-rows16` arms reverse the
forward traversal so one resident vocabulary tile serves every loss row. They
also emit compact `(tile_max, scaled_exp_sum, target_logit)` partials during the
projection and reduce those partials per row, eliminating the separate
full-vocabulary FP32 logit-tape read previously needed for CE statistics. The
rows16 build retains its W tile as packed half2 values in shared memory and
unpacks only at FP32 dot-product time; at width 448 this keeps the shared
footprint to about 16.2 KiB instead of about 30.5 KiB for an unpacked rows16
tile, preserving occupancy while halving vocabulary workgroups relative to
rows8. Before the fused-adjoint arm below, the local 32 KiB/subgroup-64 AMD
Radeon profile at vocab 50,257 selected rows16. After the direct-adjoint change, the
two-row width-448 matrix measured about 22.78-22.86 ms for rows16, 24.39-24.62
ms for rows8, 41.05-41.28 ms for the row-major CE tape, and 66.40-66.77 ms for
portable packed streaming depending on dW fanout. These are target-specific
kernel measurements, so other devices still race the candidates rather than
inheriting that choice as a heuristic.
The rows16 projection-to-stats boundary now has its own ignored GPU timestamp
microprofile. On the same AMD target at width 448, vocab 50,257, and 16 rows,
the serial rows16 projection measured about `18.93 ms` per pass while the
compact stats reducer remained about `0.021 ms`; the reducer is only about
`0.11%` of that pair and the intermediate partial buffer is 603,264 bytes.
That made projection->stats fusion the wrong target. Instead,
`fp16-ce-tape-rows16-dot4` keeps the same packed 16-vocabulary W tile but assigns
four lanes to each vocabulary row, so all 64 lanes participate in the width-448
dot phase. Its four-way partial storage brings the complete shared footprint to
exactly 16 KiB. On the same 16-row geometry it measured about `6.90 ms` for the
projection, a 2.74x kernel speedup, with only `2.98e-7` max logit drift and
`1.49e-7` max CE-stat drift from the serial rows16 reduction. The complete
width-448 LM backward matrix improved from about `61.96 ms` for the best serial
rows16+dW plan to about `52.75 ms` for dot4, roughly a 15% end-to-end reduction;
the selector chose `fp16-ce-tape-rows16-dot4+dw-vocab8` in its own five-sample
race. At two rows the existing fused-adjoint arm still wins decisively, so dot4
is an additional autotuned geometry rather than a global replacement. The FP32
logit tape remains intentionally unchanged.
The experimental `fp16-ce-tape-rows16-fused-adjoints` arm keeps that rows16
forward unchanged, then inverts the backward ownership: one 64-lane workgroup
owns one 64-vocabulary tile and the complete width-448 hidden dimension. It
holds up to eight rows of normalized hidden pairs plus CE adjoints in exactly
16 KiB of shared memory, reads/exponentiates every taped logit once globally,
loads every packed W element once, writes dW directly, and emits one dX partial
per 64-vocabulary tile. A tiny follow-up reduction consumes those partials in
ascending tile order, preserving the established W^T cross-tile add order
without float atomics. The arm is therefore gated to at most eight loss rows;
its largest width-448 partial scratch is about 10.75 MiB at vocab 50,257 and is
allocated only when the arm is runnable. On the local 32 KiB/subgroup-64 AMD
Radeon, the release microprofile measured 7.13 ms at two loss rows versus
22.20 ms for the best unfused rows16 plan (about 3.1x faster), and 14.49 ms at
eight loss rows versus 39.00 ms (about 2.7x faster). The arm remains autotuned
instead of becoming a device heuristic, so other Vulkan targets must earn the
same choice from their own measurements.
The dot4 projection can now feed that same fused backward without changing its
master-weight, checkpoint, FP32-logit, dW, or dX contracts. On the same local
AMD target at width 448, vocabulary 50,257, and two loss rows, the production
backward microprofile measured `7.063 ms` for serial rows16 + fused adjoints,
`5.790 ms` for dot4 + fused adjoints, and `5.729 ms` for the clustered-dot4
variant. The clustered variant replaces the dot4 workgroup partial array and
barrier with four-lane `subgroupClusteredAdd` reductions, reducing shared memory
from exactly 16,384 bytes to 16,192 bytes while retaining the same contiguous
hidden-pair partitioning. It is exposed only when compute subgroup arithmetic
and clustered operations are both reported by Vulkan, and remains in the
device/geometry autotune race rather than becoming a vendor heuristic. Relative
to the previous serial fused-adjoint winner, that two-row profile is about 19%
faster; the clustered arm narrowly beat ordinary dot4 by about 1%, so devices
without clustered subgroup support lose no execution path.
The fused adjoint now has a second, independently autotuned hidden-value
topology. `fused-shared-hidden` is the original 16 KiB path. The new
`fused-private-hidden` path observes that each normalized hidden pair is written
and read by the same lane, so it keeps those values lane-private and leaves only
the 2 KiB CE-adjoint window in workgroup shared memory. Arithmetic order, FP16
execution mirror, FP32 master weights/logits/gradients, dX partial ordering, and
checkpoint representation are unchanged. At width 448, vocabulary 50,257, and
two rows on the local subgroup-64 AMD target, serial fused adjoints improved
from `7.007 ms` shared to `5.846 ms` private; dot4+fused improved from
`5.800 ms` to `4.630 ms`; clustered-dot4+fused improved from `5.764 ms` to
`4.636 ms`. The selector chose dot4 + private hidden in its five-sample race.
The fused-adjoint topology coordinate retains the same 2% persistence margin used by
the complete plan selector, so register-heavy devices do not inherit this AMD
choice from a timing tie. Force only this coordinate for diagnostics with
`HIERARCHOS_VULKAN_LM_FUSED_ADJOINT_TOPOLOGY=shared-hidden|private-hidden|private-hidden-tile256`.
The dX-partial seam now has a third fused topology that keeps private hidden
values but widens both vocabulary ownership and the workgroup to 256 lanes.
The first naive wide-tile experiment kept WG64: it cut scratch traffic but
serialized more vocabulary work per lane and lost badly. GPU timestamps also
showed why: at width 448, vocab 50,257, and eight rows the tile64 reducer was
only about `0.38-0.41 ms`, roughly 10% of the fused-emission + reduction pair.
Matching the workgroup to the wider tile restores that parallelism. On the
same local AMD target, the isolated eight-row fused+reduce seam improved from
`4.082 ms` for tile64/WG64 to `3.598 ms` for tile256/WG256 while deterministic
FP32 dX scratch fell from `10.746 MiB` to `2.693 MiB`; dX max drift was only
`6.98e-9`. At two rows the same seam improved from `3.160 ms` to `2.193 ms`
while scratch fell from `2.687 MiB` to `0.673 MiB`, with `2.33e-8` max dX drift.
The production width-448 LM matrix then measured clustered-dot4 + fused private
tile64 at `4.6245 ms` versus `4.4145 ms` for clustered-dot4 + private
tile256/WG256, and the normal autotuner selected the wide topology. The full
79-tensor PyTorch worker-refinement oracle still passed (`5.21e-7` loss
difference, `1.31e-6` worst graph-value difference), and the raw-token
checkpoint smoke passed PyTorch reload, native Rust inference reload
(`1.79e-7` max native-vs-PyTorch output difference), and optimizer round-trip.
CUDA execution is still not available on this host, so the NVIDIA runtime leg
remains an interchange contract rather than a locally executed validation.
A fixed-recurrent-plan strict-parity whole-step A/B/A was intentionally not
used to claim an end-to-end speedup: private-tile64 / tile256-WG256 /
private-tile64 medians were `83.786 / 86.421 / 87.704 ms`. The candidate is
inside the baseline's run-order/DVFS drift envelope, so the durable signal is
the LM-local win plus the smaller persistent scratch. Whole-step optimization
attention should therefore remain on recurrent execution and scheduling.
The first follow-up on that recurrent frontier removes avoidable lane-0
serialization from `packed_cell_channel_mix_state_forward_fused`: mean and
variance still reduce serially in their original FP32 order for strict parity,
but the subsequent per-column normalize/mix/state writes are independent and
now fan out across the 256-lane workgroup. On the same local AMD/subgroup-64
production-shape fixture (width 448, vocab 50,257, FP16 storage with FP32
compute, WG64 H/L recurrence, strict numerics, recurrent and LM plans pinned),
GPU timestamp attribution moved the packed channel-mix forward from `9.317 ms`
to `7.923 ms` across 18 dispatches and the complete recurrent category from
`21.493 ms` to `19.021 ms` in the measured optimizer submission. An ordinary
seven-sample candidate/baseline/candidate sandwich then produced whole-step
medians of `87.592 / 90.896 / 88.956 ms`: both candidate legs beat the enclosed
baseline by more than the scheduler's 2% hysteresis, while the bracketed
candidate average is about 2.9% lower latency. This is a target-specific
whole-step signal rather than a vendor-wide Vulkan claim. The packed-cell
PyTorch parity gate passed (`2.68e-7` worst packed-state drift), and the broader
79-tensor worker-refinement oracle remained within the existing mixed-precision
tolerance (`5.21e-7` loss difference, `1.31e-6` worst graph-value difference);
dense, microbatched, and sparse-replay token-tape checks remained exact. The
change does not alter tensor names, state layout, optimizer state, or the
SafeTensors interchange contract, so the existing PyTorch/native-Rust/CUDA
checkpoint boundary is unchanged.
The next width-32 recurrent-forward sweep deliberately kept those production
defaults unless a candidate survived both kernel-local and complete-step
measurement. `rwkv_time_mix3_linear3_key_state_forward_fused` now has an
explicit WG32 build selectable with
`HIERARCHOS_VULKAN_TIME_MIX_FORWARD_WORKGROUP_SIZE=32` (WG64 is the default),
and a width/head <=32 paired-row research arm selectable with
`HIERARCHOS_RWKV_TIME_MIX_ENABLE_TWO_ROW_FORWARD_FUSION=1`. On the local
AMD/subgroup-64 width/head-32, FP16-storage-parity `8 x 1` fixture, WG32 measured
about `2.16-2.18 ms` across 96 dispatches versus `2.11-2.12 ms` for WG64, while
the paired-row wave64 arm measured about `2.29-2.34 ms`; neither is promoted by
default. The paired-row arm nevertheless passed the complete 79-tensor
worker-refinement PyTorch oracle (`1.42e-7` loss difference, `1.69e-6` worst
graph-value difference), with dense, microbatched, and sparse-replay token-tape
checks exact. A one-row/two-row/one-row seven-sample whole-step sandwich gave
`34.960 / 35.013 / 35.355 ms`, confirming no durable end-to-end win on this
target.
`packed_cell_channel_mix_state_forward_fused` likewise now exposes explicit
32/64/128/256-lane builds through
`HIERARCHOS_VULKAN_PACKED_CELL_FORWARD_WORKGROUP_SIZE`, while keeping WG256 as
the default. On the same fixture, the kernel sweep measured roughly `4.17 / 2.79
/ 2.24 / 1.96 ms` respectively. The apparently faster WG128 profiled whole-step
sample did not survive an unprofiled WG256/WG128/WG256 sandwich (`34.011 / 34.698
/ 35.325 ms`), so no target-specific heuristic was baked into the runtime. All
four geometries passed the packed-cell PyTorch oracle with the same `2.68e-7`
worst packed-state drift. These retained arms are intended for cross-vendor
profiling rather than assumed portable wins.
The next recurrent optimization therefore attacks traffic rather than lane
count. First-pass TBPTT now has a packed forward-only transition: it unpacks
only the O(width) time/channel/vector cache slots, reads the O(width*head_size)
matrix state directly from packed history, and writes the next matrix state
directly into the next TBPTT history buffer. Because reverse TBPTT already
rematerializes each cell forward, this first sweep also suppresses global
writes for backward-only r/k/v and channel-mix tapes. The fused channel/state
tail writes its residual output directly into the per-timestep output history,
so neither the new packed state nor the cell output takes an intermediate
device-to-device copy. Backward remains on the original full-tape path.
`rwkv_tbptt_packed_forward_parity` is the A/B gate and exposes a per-sequence
runtime switch rather than relying on build-time assumptions. On the local AMD
Radeon width/head-32 coherent-v9 fixture, a 32-step strict-parity run reported
zero max absolute difference for outputs, final packed state, input gradients,
token-feature gradients, and the initial packed-state gradient. Seven-sample
end-to-end medians were `12.7668 ms` for the legacy first-forward path and
`12.4133 ms` for the packed/direct-history path, a `1.0285x` speedup. This is a
small-model target-local result, not a portable throughput claim, but it
validates the architectural direction: recurrent history can cross fused
stages without changing tensor names, public state layout, optimizer state, or
the PyTorch/native-Rust/CUDA SafeTensors interchange contract.

Projection-weight traffic now has an opt-in research arm selected with
`HIERARCHOS_RWKV_TIME_MIX_ENABLE_WEIGHT_REUSE2=1`. One 128-lane workgroup owns
one recurrent head and two batch rows. The rows execute concurrently while a
single shared-memory tile stages the canonical PyTorch `[out, in]` r/k/v weight
rows once for both rows, so no transpose, prepack, Vulkan-only master weight, or
checkpoint conversion is introduced. All compatible 16/32/64-column variants
are now registered as separate measured topology arms: tile16 consumes 15,232
bytes of shared memory and is below the Vulkan 16 KiB minimum guarantee, while
tile32 and tile64 require 27,904 and 53,248 bytes respectively and are admitted
only when the selected device advertises enough compute shared memory. The
autotune key includes device/subgroup, model width/head size, batch-pair count,
odd-row tail, and full-versus-packed recurrence. Candidate measurements are
interleaved and require a 2% win before displacing the portable baseline. The
same topology search is wired into the packed forward-only TBPTT transition so
the traffic experiment reaches the actual training first-forward path rather
than only reverse-pass rematerialization. Set
`HIERARCHOS_RWKV_TIME_MIX_WEIGHT_REUSE2_AUTOTUNE_LOG=1` to print the measured
arms and selected topology.

On the local AMD Radeon/subgroup-64 width-128, head-size-64, batch-2 fixture, the
new projection-local autotuner measured the full recurrence at approximately
`0.1090 / 0.1170 / 0.1586 ms` for baseline/tile16/tile32, and the packed
recurrence at `0.0906 / 0.1199 / 0.1609 ms`. Both searches therefore retained
the baseline. This deliberately supersedes the old "largest tile that fits"
choice: a topology can be correct and still be the wrong answer for a particular
device/shape.

Reverse/rematerialization projection traffic now has a separate reuse experiment
instead of inheriting the first-forward decision. The new
`linear3_input_grad_weight_reuse2` kernel uses one 128-lane workgroup for two
batch rows and stages canonical `[out, in]` r/k/v tiles once for both dX rows.
Its tile16/32/64 shared-memory footprints are 12,288 / 24,576 / 49,152 bytes,
and a dedicated process-local autotuner races every compatible tile against the
original `linear3_input_grad` dispatch. On the same Radeon, direct GPU readback
was bit-for-bit identical for every supported tile at batch 2 and odd-tail batch
3. The measured batch-2 times were about `0.0775 / 0.0980 / 0.0960 ms` for
baseline/tile16/tile32; batch 3 measured `0.0803 / 0.0937 / 0.0979 ms`, so the
runtime again retained baseline. This backward arm is therefore infrastructure,
not a claimed win on this target: it lets future devices and larger pair counts
prove the reuse hypothesis without weakening PyTorch parity. Enable selection
logging with `HIERARCHOS_RWKV_PROJECTION_INPUT_GRAD_REUSE2_AUTOTUNE_LOG=1`; the
direct parity/profile gate is:

```powershell
cargo test -p hierarchos-vulkan profile_projection_input_grad_weight_reuse2 --lib -- --ignored --nocapture
```

The complete PyTorch core oracle was also run with the largest supported reverse
reuse tile forced by
`HIERARCHOS_RWKV_PROJECTION_INPUT_GRAD_REUSE2_DISABLE_AUTOTUNE=1`. SafeTensors
row-major interchange passed, with worst forward drift `1.34e-7` and the tested
projection/mix gradients remaining at or below `4.47e-8`. Forcing the full
first-forward reuse topology with
`HIERARCHOS_RWKV_TIME_MIX_ENABLE_WEIGHT_REUSE2=1` plus
`HIERARCHOS_RWKV_TIME_MIX_WEIGHT_REUSE2_DISABLE_AUTOTUNE=1` passed the same
oracle and numerical envelope. These forced runs are correctness gates; normal
execution still uses the measured selectors described above.

On the local AMD Radeon/subgroup-64 width-128, head-size-64, batch-2, eight-step
fixture, a seven-sample packed-TBPTT baseline/reuse/baseline sandwich measured
`10.7445 / 10.4117 / 10.6857 ms`. Relative to the `10.7151 ms` bracketed
baseline average, the retained arm was about `1.0291x` faster (`2.83%` lower
latency). This remains opt-in rather than a default because that result is one
target/shape and the full-tape leg did not show the same improvement. Correctness
gates were stronger: the batch-2 PyTorch r/k/v core oracle kept its existing
`1.34e-7` worst forward absolute difference and passed direct row-major
SafeTensors loading; the FP16-storage/FP32-compute 79-tensor worker-refinement
oracle passed with `1.79e-7` loss difference, `1.31e-6` worst graph-value
difference, and `6.56e-7` worst parameter difference, while dense,
microbatched, and sparse-replay token-tape comparisons remained exact. A
batch-3 packed A/B also reported zero output/state/input/token/initial-state
adjoint difference, covering the odd tail of the two-row dispatch mapping.
For the LM-topology measurements above, recurrent/optimizer work still dominates
a complete Hierarchos step, so those remain LM-backward measurements rather
than a claimed whole-step speedup; their end-to-end effect is small enough to
require longer target-specific runs.
In one explicit strict-parity A-B-A run with identical microbatch/checkpoint and
WG64 recurrent geometry, private/shared/private medians were `53.884 ms`,
`55.132 ms`, and `51.772 ms`; an earlier shorter pair reversed by about 1.3%,
so the kernel-local result is the durable claim while complete-step throughput
should continue to be judged by longer, interleaved target-device sampling.
The forced dot4 + private-hidden path also passed the full 79-tensor PyTorch
worker-refinement oracle (`5.21e-7` loss difference, `1.31e-6` worst graph-value
difference), then passed the raw-token one-submit checkpoint smoke: PyTorch
reload, native Rust inference reload (`2.38e-7` max native-vs-PyTorch output
difference), and optimizer checkpoint round-trip all passed. CUDA execution was
not available on the local host, so that runtime leg remains unverified here.
Override the dW half of the plan with `HIERARCHOS_VULKAN_LM_BACKWARD_TOPOLOGY=dw-vocab4`,
`dw-vocab8`, or `dw-vocab16` for arms with a separate dW kernel. The fused-adjoint
arm produces dW internally, so that topology coordinate is intentionally inert.
To expose the full width-448 plan matrix on one
target GPU, run the ignored microprofile explicitly:

```powershell
cargo test -p hierarchos-vulkan lm_width448_backward_topology_microprofile --lib --release -- --ignored --nocapture
```

It defaults to vocabulary size 50,257 and two rows. For a shorter diagnostic
pass, set `HIERARCHOS_VULKAN_LM_MICROPROFILE_VOCAB_SIZE` and/or
`HIERARCHOS_VULKAN_LM_MICROPROFILE_ROWS` before invoking the test.
To isolate only the rows16 forward->stats seam with exact Vulkan dispatch
timestamps, run:

```powershell
$env:HIERARCHOS_VULKAN_PROFILE_KERNELS='1'
cargo test -p hierarchos-vulkan lm_rows16_forward_stats_seam_microprofile --lib --release -- --ignored --nocapture
```

`HIERARCHOS_VULKAN_LM_SEAM_PROFILE_ROWS`,
`HIERARCHOS_VULKAN_LM_SEAM_PROFILE_VOCAB_SIZE`, and
`HIERARCHOS_VULKAN_LM_SEAM_PROFILE_REPETITIONS` override its default
16-row/50,257-vocabulary/32-repeat geometry.

## Cross-backend training economics gate

`tools/benchmark_vulkan_pytorch_parity.py` now races the complete native Vulkan
labeled-TBPTT/AdamW trajectory against the equivalent PyTorch CPU and, when
available, CUDA trajectory. Both backends begin from the same canonical
SafeTensors package and execute the same deterministic updates. PyTorch remains
an external oracle/competitor only: the Vulkan training process itself has no
PyTorch dependency. The benchmark fails if the final trainable parameters drift
beyond its parity threshold, then reloads the Vulkan-trained package through
PyTorch CPU, native Rust inference, and CUDA inference when an NVIDIA device is
present.

FP32 remains the reference economics baseline, but the same gate now qualifies
all four native training precision contracts: `fp32`,
`fp16-storage-fp32-compute`, `fp16-storage-parity`, and
`fp16-storage-fp16-lm-backward`. Mixed-precision PyTorch timing legs use the
same rounded execution values as Vulkan while restoring FP32 masters before
AdamW, so the final checkpoint boundary remains canonical FP32 SafeTensors and
can be reloaded unchanged by native Rust inference or PyTorch CPU/CUDA.
Optional hourly-price inputs are converted to cost per billion model token
positions, so local hardware, cloud Vulkan instances, and PyTorch/CUDA
instances can be compared with the same token accounting:

```powershell
python tools/benchmark_vulkan_pytorch_parity.py `
  --warmup 1 --iterations 5 `
  --vulkan-usd-per-hour 0.25 `
  --pytorch-cuda-usd-per-hour 0.80 `
  --output benchmark_results/cross_backend_training.json
```

The currently qualified aggressive mixed-precision shape is source-scaled:

```powershell
python tools/benchmark_vulkan_pytorch_parity.py `
  --warmup 1 --iterations 5 `
  --precision fp16-storage-fp16-lm-backward `
  --dynamic-loss-scale 1024 `
  --output benchmark_results/cross_backend_training_fp16_dls.json
```

That loss scale is not merely a numerical afterthought. In the aggressive FP16
policy, unscaled native-FP16 projection dW/db is cancellation-sensitive. On the
local AMD device a one-update strict test reached about `5.88e-4` parameter
drift and therefore did not clear the existing `5e-4` qualification threshold.
With source scaling enabled, the graph intentionally keeps those projection
parameter reductions in FP32 while retaining the aggressive FP16 execution
path elsewhere. The same strict trajectory then fell to `2.24e-6` maximum
parameter difference. The dynamic GradScaler-style optimizer boundary also
fuses normalization, unscale, finite detection, global norm, clipping, scaler
transition, and predicated AdamW into one Vulkan submission; the one-update
path therefore uses two total queue submissions instead of the ordinary FP32
path's three.

The Rust labeled-sequence executable now reports allocator/budget telemetry in
the same JSON result: logical live-buffer bytes, suballocator reserved bytes,
driver allocation count, and `VK_EXT_memory_budget` device-local
budget/usage/availability. The comparison labels Vulkan reserved bytes and
`torch.cuda.max_memory_allocated()` separately because they are useful but not
identical memory metrics.

On the local AMD Radeon width-32/vocab-64 smoke, one warmup plus three measured
updates produced a `49.2822 ms` Vulkan median versus `209.1077 ms` for PyTorch
CPU, or about `4.24x` the model-token throughput. The four-update final package
remained within `5.67e-6` maximum parameter difference from PyTorch, and native
Rust inference reloaded the Vulkan-trained package within `3.58e-7` maximum
logit difference from PyTorch CPU. The Vulkan graph held 24,229,124 logical live
buffer bytes in 50,331,648 reserved bytes across three driver allocations. This
is a tiny-fixture/local-device qualification result, not a general CUDA or
cross-vendor speedup claim; this host has no CUDA device, so the CUDA timing and
reload gates remain explicitly unverified here.

A warmed mixed-precision economics smoke using
`fp16-storage-fp16-lm-backward --dynamic-loss-scale 1024` measured three updates
at `51.6837`, `51.7821`, and `53.0647 ms` on Vulkan, for a `51.7821 ms` median,
versus a `217.7384 ms` PyTorch CPU median on the same deterministic trajectory.
The four-update final package stayed within `4.17e-6` maximum parameter
difference, and native Rust inference reloaded it within `3.58e-7` maximum
logit difference from PyTorch CPU. A separate three-update strict trajectory
stayed within `8.66e-6` maximum parameter difference and used exactly six queue
submissions, confirming the two-submission-per-step dynamic path across
multiple optimizer steps. Treat these timings as local tiny-fixture
qualification data rather than a general speedup claim; a real NVIDIA CUDA host
is still required before drawing cross-device economics conclusions.

The fused FP16 LN1/low-rank producer now also has a portable 64-lane topology
for low-rank branches up to rank 128. Each lane owns at most two rank outputs,
while every first- and second-stage dot product keeps canonical increasing-index
FP32 FMA order and reads the same optimizer-coherent packed-FP16 mirrors as the
rank-64 path. This removes the former production-geometry cliff at width 448:
the coherent fixture uses H/L `g_rank=96`, which previously made the fused
producer unavailable before the first optimizer update. On the local AMD host,
width 448 / vocabulary 50,257 / batch 2 / 64-token sequences now completes a
source-scaled mixed-precision optimizer step with `4.3698e-4` maximum parameter
difference from the PyTorch CPU oracle, below the existing `5e-4` qualification
gate. The same run reserved 1,783,627,776 Vulkan bytes and measured 7,054.87 ms
per Vulkan step versus 21,041.70 ms for PyTorch CPU (about 2.98x model-token
throughput on this host). A separate width-448 interchange gate reloaded the
Vulkan-trained package through native Rust inference within `1.43e-6` maximum
output difference from PyTorch CPU. These are local AMD qualification numbers,
not an NVIDIA/CUDA performance claim.

Budgeted tape windows can now be qualified together with dynamic loss scaling.
The graph already exposed planner-selected and explicit-plan dynamic AMP entry
points plus the clipped dynamic finisher; the labeled runner and economics
harness no longer reject that supported combination. Safety restrictions on
resuming/opening partial budgeted windows and pending-gradient capture remain in
place. The production-shaped qualification above used this combined path and
reported one planner-selected budgeted plan, two total queue submissions, no
loss-scale overflow, and an unchanged scale of 1024. A direct qualification is:

```powershell
python tools/benchmark_vulkan_pytorch_parity.py `
  --fixture-width 448 --fixture-vocab 50257 `
  --fixture-batch 2 --fixture-tokens 64 `
  --precision fp16-storage-fp16-lm-backward `
  --dynamic-loss-scale 1024 --budgeted-windows `
  --skip-cuda `
  --output benchmark_results/cross_backend_training_w448_v50257_t64_budgeted_amp.json
```

## PyTorch parity smoke test

From the repository root:

```powershell
python tools/verify_vulkan_training_parity.py
python tools/verify_vulkan_out_norm_parity.py
python tools/verify_vulkan_tied_embedding_parity.py
python tools/verify_vulkan_shared_adapter_parity.py
python tools/verify_vulkan_projection_parity.py
python tools/verify_vulkan_rwkv_matrix_state_parity.py
python tools/verify_vulkan_rwkv_time_mix_core_parity.py
python tools/verify_vulkan_rwkv_low_rank_parity.py
python tools/verify_vulkan_rwkv_fused_time_mix_parity.py
python tools/verify_vulkan_rwkv_full_time_mix_parity.py
python tools/verify_vulkan_rwkv_channel_mix_parity.py
python tools/verify_vulkan_rwkv_adapter_channel_mix_parity.py
python tools/verify_vulkan_rwkv_cell_slice_parity.py
python tools/verify_vulkan_rwkv_packed_state_parity.py
python tools/verify_vulkan_rwkv_packed_cell_parity.py
python tools/verify_vulkan_rwkv_tbptt_parity.py
python tools/verify_vulkan_rwkv_tbptt_train_parity.py
python tools/verify_vulkan_rwkv_tbptt_tied_embedding_parity.py
python tools/verify_vulkan_rwkv_tbptt_branch_accumulation.py
python tools/verify_vulkan_rwkv_tbptt_fork_parity.py
python tools/verify_vulkan_full_training_graph.py
python tools/verify_vulkan_full_training_graph_fusion.py
python tools/verify_vulkan_full_training_graph_projection_coupling.py
python tools/verify_vulkan_full_training_graph_loss_coupling.py
python tools/verify_vulkan_worker_refinement_loss_parity.py
python tools/verify_vulkan_control_parity.py
python tools/verify_vulkan_manager_hard_act_parity.py
python tools/verify_vulkan_token_frontend_parity.py
python tools/verify_vulkan_token_memory_frontend_parity.py
python tools/verify_vulkan_rosa_persistence_parity.py
python tools/verify_vulkan_raw_token_tape_parity.py
python tools/verify_vulkan_labeled_sequence_parity.py
python tools/verify_vulkan_labeled_sequence_parity.py --accumulation-steps 3
python tools/verify_vulkan_labeled_sequence_parity.py --checkpoint-dtype fp16
python tools/verify_vulkan_labeled_sequence_parity.py --checkpoint-dtype bf16
python tools/verify_vulkan_model_interchange.py
python tools/verify_vulkan_mixed_model_interchange.py
python tools/verify_vulkan_cuda_vulkan_trajectory.py --require-cuda
python tools/verify_vulkan_cuda_vulkan_trajectory.py --cpu-fallback --cycles 3
```

These tests cover repeated cross-entropy + AdamW updates, `out_norm`
forward/backward and parameter updates, repeated-ID tied embedding gradient
accumulation, both shared-adapter optimizer regimes, affine/biasless projection
training, production-size RWKV-v8 matrix-state forward/backward, the composed
r/k/v time-mix recurrent core (including projection/key-parameter gradients),
the complete low-rank `a/w/g` graph, the single-submit fused recurrent graph,
the full GroupNorm/bonus/gate/output-projection post-mix, channel-mix
ReLU2/DeepEmbed, differentiable packed/clamped recurrent state, multi-step
detach-scheduled TBPTT, all-33-tensor gradient accumulation, two consecutive
persistent AdamW updates, token-ID -> tied-embedding -> DeepEmbed TBPTT with a
34th trainable tied matrix and repeated-token gradient accumulation, masked
PyTorch-shaped labeled training with cross-submission weighted-token
accumulation, and Vulkan -> SafeTensors -> PyTorch/native-Rust interchange. The full-training fusion
harness also compares the historical three-submit H/L/loss sequence against the
new one-submit command graph from identical weights and fails on any recurrent,
optimizer, out_norm, LM-head, or loss drift above its parity tolerance.
The token-front-end harness additionally proves token-ID gather, repeated-token
embedding scatter-add, persistent/in-projection gradients, pre-gated LTM input
gradients, and `in_proj -> GELU` output parity against PyTorch. It also executes
the same seam through `hierarchos-inference`; when CUDA is present it repeats the
reference computation on CUDA from the unchanged SafeTensors package.
The projection-coupling harness separately proves one-submit H/L projection
interleaving, all-ten-tensor gradient flow with weight decay disabled, packed
state feedback-gradient routing, and bit-exact projection AdamW checkpoint
save/load. The loss-coupling harness removes the synthetic `l_to_out` upstream
gradient and proves the real cross-entropy gradient moves `l_to_out`, out_norm,
and the tied LM parameter while preserving the single-submit shared-gradient
ordering. The control-parity harness compares hard ACT selection/depth backward,
context interpolation/concat backward, and drift seed/recurrent update backward
directly against PyTorch autograd.
The RWKV package-load tests additionally prove that standard `h_rnn` tensor
names/layouts are consumed directly without a backend-specific checkpoint
transform. The interchange harness also executes its CUDA inference check
automatically when run on an NVIDIA/CUDA host.

## Interchange contract

`HierarchosHeadTrainer::from_model_package` loads `lm_head.weight` from the same
`model.safetensors` consumed by `hierarchos-inference`. Model tensors at this
boundary may now be FP32, FP16, or BF16 SafeTensors in canonical PyTorch
row-major shape. Vulkan training and native Rust inference promote FP16/BF16
values to FP32 master/compute values on load, so a PyTorch mixed-precision model
package needs no backend-specific repack or transpose. After training,
`export_model_package` copies the package and replaces the trained tensor with
canonical FP32 while preserving the dtype and bytes of untouched tensors. The
result may therefore be a standards-compliant mixed-dtype SafeTensors package;
it remains directly consumable by PyTorch CPU/CUDA and by native Rust inference.

The full labeled-sequence/TBPTT parity harness exposes the same boundary through
`--checkpoint-dtype fp16|bf16`. Both lower-precision source formats are qualified
against the FP32-compute training graph before optimizer/checkpoint/native
inference handoff. Checkpoint storage dtype is deliberately independent from the
Vulkan execution-precision policy. The lower-precision source-package boundary
is now also qualified with both aggressive internal FP16 execution policies,
`fp16-storage-parity` and `fp16-storage-fp16-lm-backward`, using a production
dynamic loss scale of 1024 and AdamW epsilon `1e-6`. On the local AMD fixture all
four source-dtype/execution-policy pairs clear the `8e-6` parameter and recurrent
state ceilings. The worst observed recurrent-state difference was about
`5.55e-6` for FP16 source weights; BF16 source weights stayed at or below about
`1.61e-6`. The same qualification also covers open accumulation-window
checkpoint/resume, loss-scale overflow/backoff, repeated Vulkan -> PyTorch ->
Vulkan optimizer continuation, and native Rust inference reload. CUDA execution
remains a hardware-gated leg on this host rather than an inferred pass.

Run the consolidated lower-precision qualification with:

```powershell
python tools/verify_vulkan_low_dtype_training_qualification.py
```

The `hierarchos-vulkan-package-step` binary exercises that path directly:

```powershell
cargo run --release --manifest-path hierarchos-vulkan/Cargo.toml `
  --bin hierarchos-vulkan-package-step -- `
  --model path/to/rust_model `
  --case head_batch.json `
  --output-model path/to/trained_rust_model
```
