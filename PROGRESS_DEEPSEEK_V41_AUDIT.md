# DeepSeek V4.1 Flash native implementation audit

> **Current scope, 2026-10-25:** the requested implementation target is
> **Hierarchos Native Rust/Vulkan**, with the published Flash-1017 defaults and
> normal loading/generation. The previous Transformers implementation was an
> assistant misreading of that target. It is preserved separately, but is not
> native runtime support or native qualification evidence. CPU PyTorch is the
> development reference only. The strict `<2e-7`, `rtol=0`, full-logit gate and
> the absence of a CUDA parity requirement remain unchanged.

> **Qualification closed under the user's scalable-checkpoint criterion:**
> native causal text inference is qualified on the Ryzen Z1 Extreme / Radeon
> Graphics machine. A fresh 44.8 MB, 40-layer/384-expert/top-6 checkpoint run
> through `from_hf_package` passes all 3,216 logits with zero CPU drift and exact
> routing. The recorded production-width gates also pass. The durable receipt
> is `hierarchos-vulkan/validation/release_parity/deepseek_v41/qualification.json`.
> Production-weight execution is not a prerequisite for this qualification.
> Intel Gen9 kernel optimization and its target-device validation are next.

## Contract and acceptance gate

Implement `deepseek_v41` / `deepseek_v41_text` as a dedicated Rust/Vulkan
architecture. Production execution must not depend on Python, PyTorch, CUDA,
TileLang, remote code, or a Transformers fallback. Development oracles may use
the released code. Do not alias this architecture to DeepSeek V4 or Kimi K3.

**User-required hard gate: every compared logit must have absolute drift strictly
less than `2e-7`, with `rtol=0`.** Compare the complete logit tensors for both
prefill and cached decode against the pinned independent DeepSeek reference,
using identical weights, token inputs, positions, and precision semantics. Reject
NaN, infinity, shape mismatches, and empty comparisons. Do not relax the threshold,
introduce relative tolerance, or substitute matching token choices for logit
parity. `validation/deepseek_v41_parity.py` fixes this limit without a caller
override; its boundary tests reject equality with the ceiling. Component parity
does not satisfy this full-model requirement.

**Accepted reference: CPU PyTorch.** The user explicitly confirmed on
2026-10-23 that CUDA parity is not a project requirement and no NVIDIA device
is available. CUDA/TileLang equivalence is not an acceptance gate or a blocker.
The CPU reference must still preserve the declared precision, quantization,
weights, inputs and complete-logit comparisons above. Complete packed BF16
graph evidence is recorded below and is restricted to the measured shapes,
depths and schedules.

Current status: configuration, tensor I/O, candidate selection, FP32 stateful
compression, released-semantics activation quantization, YaRN rotary,
compressed-KV finalize ordering, the fine indexer, the sliding-window plus
compressed attention block (window ring, gathered sparse softmax with the
attention sink, inverse output rotation, grouped output projection), a
layer-ordered attention orchestrator, shifted Hyper-Connection + MoE backbone,
Engram, backbone checkpoint loading, token embeddings, and the vocabulary head
are implemented. The `DeepseekV41CausalLM` library API produces
complete FP32/hybrid and explicit packed BF16 text-model logits. **All 1,472,649
logits across 18 fixed models and 108 prefill/decode/reset steps pass the unchanged
gate**, with overall worst drift `1.7881393432617188e-7`, `rtol=0`. Eleven packed
BF16 models contribute **1,470,409 zero-drift logits**. This includes nine tiny
models, three wider models (including a 512-token prefill and 129-token reset),
and a twenty-four-layer released-width/full-vocabulary probe with two routed experts.
That width probe passes all **258,560 original logits with zero drift**; its
frozen fixture and CPU oracle are unchanged. Two further probes separately
qualify the released 40-layer sharing/compression/Engram schedule and the
384-expert/top-6 cardinality. A third combines the complete 40-layer schedule
and 384-expert/top-6 routing in one reduced-width graph, and a fourth runs a
single 2,048-token YaRN prefill at the released 32 index heads, with three
decodes across compression groups, an eight-token reset and two reset decodes.
A separate four-layer production-width schedule adds compression, shared
KV/index state, active candidate filtering, full-width Engram and six selected
experts: all 904,960 logits pass with zero CPU drift through the default loader.
The router also passes a 384-candidate/width-5120/top-6 component probe.
Released depth, expert counts and widths together, released Engram table sizes,
prompt chunks after a start-zero prefill and million-token contexts remain
unqualified. The additive width case also retains 14 grouped intermediate
rounding differences, all removed by the following FP8 quantization in the
measured steps; complete intermediate bit parity is not claimed.

The current continuation re-ran **119 default V4.1 tests**, all passing, covering sparse/grouped,
routing, CPU arithmetic, frozen-fixture, boundary-tie and streaming regressions.
The joint, long-context and production-width tests retain their separate opt-in
qualification records; the twenty-four-layer gate requires a locally generated 7.8 GB package.
Packed FP8/FP4 readers, GPU decode/block-scaled linear primitives, and a packed
Engram table/projection paths are implemented. Packed experts are integrated
into checkpoint-backed hybrid models. Typed attention/indexer projections and
explicit BF16 boundaries are integrated and qualified in hybrid models and
wider component tests. The complete BF16 residual/HC, norms, compressor,
cache, Engram and final-head graph is implemented and qualified for the tiny
and wider packed checkpoints. Released hidden/head/rank/intermediate widths and
vocabulary are qualified in the twenty-four-layer accumulation probe.
Explicit checkpoint-streamed experts now pass nine existing graphs,
including the joint graph, its single 2,048-token prefill/reset probe and the
released-width probe, with 407,575 bit-exact CPU logits across 51 steps under
the combined staging policy. `StreamedAllExperts` stages routed/shared experts;
separate policies stage Engram tables, block projections, and now embedding/head
weights. Hyper-Connections, norms, gates, caches and Engram projections remain
resident. Existing constructors retain resident vocabulary by default; the
new vocabulary-residency constructor is additive.
Index-key production, discrete selection, the window index
schedule, and the gathered attention arithmetic are qualified against the pinned
reference. Legacy FP32 indexer score diagnostics retain the wider GEMM reduction
risk below; the measured BF16 score components pass by bits. The normal package
loader is `DeepseekV41CausalLM::from_hf_package`, also used by
`hierarchos-native-cli transformer-generate` for both V4.1 model-type aliases.
It infers precision and selects streaming by default, derives Engram metadata
from the matching tokenizer or sidecar, and executes the dedicated V4.1 graph.
The generic transformer graph redirects callers to this loader.

The numerical continuations repaired elementary-operation staging, per-expert
row geometry, expert accumulation order, BF16 short-row grouped projection and
one-column sparse contraction. Original full-model fixtures and the acceptance
comparator remain unchanged. Trig portability and GEMM reductions remain
qualification risks beyond the tested shapes; the hard gate remains unchanged.

## Pinned sources — October 23, 2026

- [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
- Revision: `1bae0e1088993b9b82de08f59de5be5a34e42170`.
- [Package config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/config.json).
- [Inference model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/model.py),
  SHA256 `e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d`.
- [Engram](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/engram.py),
  [kernels](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/kernel.py),
  SHA256 `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455`,
  and [conversion code](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/convert.py).
- Local Transformers reference: `C:\Users\User\transformers`, commit
  `129e8dc306932532e0b9e702d1fb4d5d7771c2a4`. It contains DeepSeek V4 and
  KimiLinear, but no `deepseek_v41` implementation. No edits were made there.
- Native repository starting commit: `48161bac938466b594ffe99e65cc978324b32aa1`.

The package publishes a standalone inference implementation, rather than an
upstream Transformers modeling module. Its config declares top-level
`DeepseekV41ForCausalLM` / `deepseek_v41`, nested `deepseek_v41_text`, and
`deepseek_v41_vision`. Do not assume `trust_remote_code` or the standard
Transformers V4 classes implement it.

## Verified architectural differences and reuse boundaries

The released text graph has 40 backbone layers, 3 DSpark draft layers, width
5120, 64 query heads, one KV head, head width 512, rotary tail 64, query rank
1280, and 8 output groups with output rank 1024. RMS epsilon is `1e-20`.

- Compression ratios are `0,0`, then eighteen `2`s, then twenty `1`s; the
  three draft layers have ratio `0`. V4's `0/4/128` schedule is incompatible.
- Compressed KV and index keys come from layers `2,8,14,20`. Index/top-k
  sources are `2,8,14,20,24,28,32,36`. Every compressed consumer reads the
  most recent source; later indexers can recompute query scores without owning
  compressed KV or index keys. Source ids are zero based.
- Layer 20 publishes candidate blocks. Later index sources score only inside
  those blocks: 2048 blocks, block width 8, final position top-k 512, 32 index
  heads of width 128. Block scores are maxima and the newest visible block is
  pinned. Candidate masks can include future positions within that block;
  the fine indexer must retain the independent causal mask.
- Ratio 2 compression pools projected KV using feature-wise softmax gates and
  carries incomplete groups across decode. Ratio 1 has no gate. Both normalize
  before RoPE. Index keys consume the pre-RoPE latent; compressed attention
  consumes the rotated, quantized latent. They must not alias destructively.
- Sliding-only attention disables YaRN. Compressed layers use compressed theta
  160000 with YaRN factor 16/original length 65536. Compressed group `j` rotates
  at its first token's position `j * ratio`.
- Hyper-connections use 4 streams and shifted pre-mix coefficients: attention
  collapses with the preceding FFN's pre-mix; FFN collapses with attention's
  pre-mix; the last FFN's pre-mix collapses the final stream. Reusing V4's
  existing local scheduling without checking this ordering is incorrect.
- Engram runs at layers 1 and 14, before the block. Token normalization,
  compressed vocab 99092, seeded hash multipliers, distinct prime buckets,
  masked lookback boundaries, persistent decode history, table lookup, and
  signed-square-root gated residual injection all require native support.
- MoE uses 384 experts, 6 selected experts, one shared expert, sqrt-softplus
  scores, selection-only correction bias, and route scale 1.5. Existing V4
  math may be reused only after checking its reduction/normalization semantics.
- FP8 weights use 32x32 blocks and UE8M0 scales; experts use packed FP4.
  Reference attention also quantizes intermediate queries/KV/index activations.
  Dequantizing stored weights alone does not reproduce those precision semantics.
  The released activation kernels clamp the block maximum (`1e-4` for FP8,
  `6*2^-126` for FP4/UE8M0, `6*2^-9` for FP4/E4M3), round FP8 E4M3 and FP4
  E2M1 to nearest-even, and fuse dequantization back to the input dtype.
  Power-of-two scales use `ceil(log2(amax/max))` by exponent bits; compressed KV
  instead stores an E4M3 scale rounded from `amax/6`. Indexer queries/keys use
  FP4 groups of 32 with UE8M0 scales; compressed KV uses groups of 16 with E4M3
  scales. Index keys rotate at `j*ratio` after `k_norm`, then FP4-quantize;
  compressed KV rotates the compressor latent tail and then FP4-quantizes.
- The actual 96085-key checkpoint index uses `embed.weight`, `layers.N.attn.*`,
  `layers.N.ffn.*`, `layers.N.engram.*`, `norm.weight`, and `head.weight`, with
  `.scale` companions. Do not assume the old `model.layers.N.self_attn.*` layout.
  This observation comes from the index only; no weight shards were downloaded.

## Implemented foundations

- `src/deepseek_v41.rs`: typed package/text contract, original metadata
  preservation (including vision/quantization), shape/schedule validation, and
  explicit per-layer KV/index/candidate/Engram ownership plan.
- `src/deepseek_v41/candidates.rs`: native Vulkan block scoring, pinning,
  ranking, and candidate-mask expansion, with resident-buffer and host I/O APIs.
  The resident path validates device/geometry and consumes indexer-generated
  scores in place; the host wrapper additionally checks score values.
- `src/deepseek_v41/checkpoint.rs`: individual tensor payload reads from single
  or indexed/sharded SafeTensors packages, using actual `layers.N.attn.*`
  keys. `load_attention_layer` loads projections, norms, sinks, and only the
  compressor/indexer tensors that the layer owns. Validates shapes, ranges,
  finite values, and shard paths. Supports F32
  and BF16 storage promotion. The separate `read_packed_weight` API reads FP8,
  logical-shape F4, or exported packed-I8 payloads with explicitly named UE8M0
  scales and layouts; `read_tensor` still rejects packed storage in the FP32
  graph. A shard's complete
  payload is never loaded merely to extract one tensor.
  `load_backbone_layer` additionally reads `layers.N.hc_attn_*`/`hc_ffn_*`,
  both norms, `ffn.gate.weight`/`bias`, every `ffn.experts.E.w{1,2,3}.weight`,
  and `ffn.shared_experts.w{1,2,3}.weight`. `load_engram` reads the scheduled
  `engram.embed.weight`, `wkv.weight`, `q_weight`, and `k_weight`; embedding
  values must already be BF16-exact after dequantization. `load_packed_engram`
  instead retains FP8/UE8M0 payloads; runtime construction selects that path from
  the actual table dtype and rejects missing or incompatible scales. Both backbone and
  Engram constructors stream layer uploads instead of retaining all host
  layer/table payloads together. Missing scheduled tensors fail explicitly.
- `src/deepseek_v41/compressor.rs`: native KV/gate projections, feature-wise
  ratio-2 softmax pooling, ratio-1 ungated projection, RMSNorm at `eps=1e-20`,
  persistent GPU partial-group state, prefill/reset, and one-token decode.
  Host-slice and GPU-resident input APIs return pre-RoPE/pre-quantization latents
  in independently owned Vulkan buffers. Decode validates position/batch/size
  before changing cache state; resident input must belong to the same device.
- `src/deepseek_v41/quantization.rs`: fused quantize/dequantize for FP8 E4M3
  with UE8M0 scales and FP4 E2M1 with UE8M0 or E4M3 scales, including the
  released amax floors and exact round-to-nearest-even conversion. Accepts host
  slices or GPU-resident input; returns dequantized FP32 values and the stored
  scale bits.
- `src/deepseek_v41/rotary.rs`: released YaRN schedule (`original_seq_len`
  disables it), FP32 position angles, adjacent-pair apply/inverse, and
  batch-local group positions (`0, ratio, ...`) for compressed latents. The
  frequency and angle arrays match the reference bit-for-bit; cosines/sines are
  evaluated in double and rounded to FP32.
- `src/deepseek_v41/compressed_kv.rs`: finalizes a compressor latent in the
  released order — the pre-RoPE latent stays readable for index-key production,
  then the tail is rotated at group positions and FP4-quantized with 16-element
  groups and E4M3 scales. Finalization allocates new buffers and never mutates
  the compressor output. Prefill requires group zero; decode requires the group
  completed at `start_pos`, positioned at `start_pos + 1 - ratio`.
- `src/deepseek_v41/attention.rs`: the released attention block around the two
  KV sources. A sliding-window ring cache reproduces `_window_kv` exactly
  (fit, wrap with cutoff `seq % window`, and the `start_pos % window` decode
  write), and `deepseek_v41_window_indices` is the released
  `get_window_topk_idxs` including `-1` padding and decode ring order. The
  window indices address the chunk during prefill and the ring during decode,
  which is also the offset added to the indexer's compressed selection. Queries
  are `q_norm(wq_a(x))` plus the rotary tail at token positions; window KV is
  `kv_norm(wkv(x))`, rotated, then FP8-32/UE8M0 quantized before the cache write.
  `gathered_attention` concatenates the window and compressed gathers the way
  `sparse_attn` does (`-1` skips a slot), applies the attention sink as a virtual
  zero-value slot in the denominator only, and normalizes in FP32. The output
  then gets the inverse rotary tail, FP8-32/UE8M0 quantization, the block
  diagonal `wo_a` einsum, and `wo_b`. `DeepseekV41CompressedKvCache` appends
  finalized latents at their group positions and is shared by consumers.
- `src/deepseek_v41/indexer.rs`: fine indexer with a shared key cache. Index
  sources project the pre-RoPE latent through `wk`, normalize with `k_norm`,
  rotate the tail at group positions, and write FP4-32/UE8M0 keys into the
  shared cache; copy layers allocate no keys and read the source buffer. Queries
  project `qr` through `wq_b`, rotate per-head rows, and FP4-quantize. Scores
  sum `relu(dot(q_h, k)) * weights_proj(x)_h * softmax_scale` across heads, apply
  the start-zero causal compressed length or the decode/chunked scalar length,
  and consume the level-one candidate mask. The top-k stage now reproduces
  the pinned CPU's score-only heap/partition ties (superseding the original
  lower-position tie break), sorts selected positions ascending,
  and maps unreachable entries to `-1` after adding the sliding-window offset.
  A candidate-source indexer publishes the mask through the already-qualified
  `select_candidate_blocks` path. Scores and masks remain in Vulkan buffers;
  only visible-length metadata is prepared on the host. Empty score inspection
  returns an empty vector, without attempting a zero-byte staging allocation.
- `src/deepseek_v41/backbone.rs`: shifted Hyper-Connections and MoE.
  `DeepseekV41HyperConnections` computes the stream RMS, projects the flattened
  `hc_mult * hidden` stream, applies the per-stage scale/base, and runs the
  released sigmoid split, row softmax, column normalization, and alternating
  Sinkhorn normalizations. `hc_pre` collapses to one copy and `hc_post` expands
  back while mixing the residual through `comb`. The POST CONTRACT DELIBERATELY
  USES `comb[k, j]`: the released `comb.unsqueeze(-1) * residual.unsqueeze(-2)`
  followed by `sum(dim=2)` contracts the first comb axis with the residual's
  copy axis, which transposes the documented `[j, k]` layout. The extracted
  reference is the authority, and the fixture records this exact arithmetic.
  `DeepseekV41Moe` implements the top-k gate (softmax, sigmoid, or
  sqrt-softplus; correction bias for selection only; raw-score weights;
  optional top-k normalization and route scaling) and the routed plus shared
  SwiGLU experts with the released clamps, per-expert weights, and shared-expert
  sum. `DeepseekV41Backbone` runs the released block order one attention layer
  at a time: attention mixes, collapse with the previous FFN pre-mix, attention
  norm, orchestrated attention, expand, FFN mixes, collapse through the
  attention pre-mix, FFN norm, MoE, expand. The FFN pre-mix is returned as the
  next layer's attention collapse. Expert dispatch is one sweep per expert with
  unrelated workgroups exiting, capped at 65535 routed slots; this is a
  correctness baseline, not a grouped-GEMM serving path.
- `src/deepseek_v41/orchestrator.rs`: `DeepseekV41AttentionOrchestrator` owns
  every attention layer and all shared publications. `begin_prefill(seq)` or
  `begin_decode(position)` starts a step; the backbone supplies each layer's
  normalized attention input to `forward_layer(layer_id, input)` in order.
  Each source runs compressor → pre-RoPE index-key publication → compressed-KV
  finalization → index selection → attention. Query latents are computed once
  and shared with the indexer. Non-source layers reuse the scheduled buffers
  and selections, including candidate masks at later index sources. The
  prefill chunk length versus decode ring length determines index offsets.
  Fixed batch/context capacities, exact layer order, position continuity,
  source readiness, and input device/size are checked. Component failures
  invalidate publications and require a new prefill; preflight errors can be
  corrected without losing sequence state. A shorter prefill resets visibility
  of stale cache tails. `from_checkpoint` uploads one layer at a time,  rather than materializing every layer's host weights together. This API returns
  attention outputs only; `DeepseekV41Backbone` now owns the residual,
  hyper-connection, MoE, and Engram execution around it, with the experimental
  language-model wrapper supplying embeddings and the head. The initial correctness kernels
  limit allocated context to 65535 positions, with further head-row/element
  dispatch limits; this is not long-context serving support.
- `src/deepseek_v41/engram.rs`: Engram hash state, lookup, and gated residual.
  `DeepseekV41EngramLayout` reproduces released `EngramLayout.from_args`: primes
  are searched upward from `engram_vocab_size - 1` in layer, n-gram, head order
  and never reused, and the per-layer bucket offsets are the cumulated prime
  sizes. Two conversion artifacts are validated inputs rather than native
  arithmetic: the compressed token map (the released Unicode-normalized
  tokenizer pooling) and the per-layer NumPy-seeded odd multipliers. The runtime
  rejects a short map, an out-of-range pad id, a multiplier layer count/shape
  mismatch, even or zero multipliers, a non-finite or empty embedding table, and
  any geometry that would overflow a dispatch before it allocates or dispatches.
  `DeepseekV41EngramHash` owns the persistent `[batch, capacity]` compressed-token
  cache: each step writes this step's compressed ids (DEAD for masked tokens),
  and the hash kernel then walks back up to `max_ngram_size - 1` positions. A
  position blocks the lookback when it precedes the sequence start or sits on a
  DEAD cache entry, and every longer lookback is filled with the pad id, exactly
  like the released running `blocked` mask. Each (n-gram size, head) lands in its
  own prime bucket, with the stored offset shifting it into the per-layer table.
  `DeepseekV41Engram` reproduces the released gate: `wkv` turns the looked-up
  `n_hash_cols * head_dim` vector into one key per hc copy plus a shared value,
  the per-(token, copy) normalized dot of stream against key uses the product of
  the q/k weights, the clamp is `1e-6` with a signed square root before the
  sigmoid, a false token mask forces the gate to zero, and the gated value is
  added to every hc copy. Lookup returns a zero row for an id outside the table
  (including a negative id, which the shader sees as an unsigned wrap), matching
  the sharded embedding's mask. `DeepseekV41EngramRuntime` writes the cache once
  per step and applies only the scheduled layers; `DeepseekV41Backbone` calls it
  on the incoming stream before each scheduled block, matching the released loop
  where `layer.engram(...)` runs immediately before `layer(...)`. The backbone
  requires token ids once an Engram runtime is attached and rejects token ids
  supplied to a backbone built without one.
- `src/deepseek_v41/model.rs`: experimental `DeepseekV41CausalLM` implements
  token lookup → HC expansion → full backbone → final RMSNorm → vocabulary
  projection, entirely on Vulkan. `prefill` and `decode` return complete
  `[batch, seq, vocab]` FP32 logits rather than only the last position. The
  fixed-batch runtime checks token counts/ids, context, binary Engram masks,
  layer geometry and device limits. Invalid input calls preserve sequence
  state; a component failure after cache mutation requires an explicit new
  prefill. Checkpoint loading includes `embed.weight`, `norm.weight`, and
  `head.weight`. Engram token maps/multipliers must be supplied separately;
  they are non-persistent in the released checkpoint. The optional mask affects
  Engram only; this API does not implement image encoding or VL router biases.
- `validation/validate_deepseek_v41_model.py`: independent six-layer text
  oracle from unchanged, hash-verified released class/forward bodies. Linear
  storage/compute is FP32; unavailable TileLang quantization, Sinkhorn and
  sparse attention have explicit CPU mirrors. This is a diagnostic FP32 graph,
  not the released mixed-precision CUDA execution. The fixture fixes 206
  tensors, two batches, vocabulary 16, ratio 0/2/1 layers, reused selections,
  candidate consumers, two Engram layers, masked spans, ring wrapping,
  odd/one-token prefills, group-completing decode, and shorter resets. All 320
  logits are compared by the unchanged hard-gate helper; native exports carry
  a SHA256 binding to the exact fixture/weights/inputs. No native output feeds
  the oracle or fixture generation. Test-only final-activation traces isolate
  backbone drift from norm/head arithmetic on identical inputs.
- Thirty-two production GLSL kernels plus one unary qualification kernel and
  checked-in SPIR-V, pinned to glslang 16.6.0 / Vulkan 1.1 in the existing shader
  provenance manifest.
- Released config fixture and 43 candidate selection cases generated from the
  exact released `select_candidate_blocks` function. The oracle extracts only
  that AST function after verifying the full source hash; it never imports
  custom model kernels. The cases contain 20624 exact mask values, including
  tied block maxima over 128 blocks.
- Candidate ties follow the pinned CPU `topk` staging: heap selection when
  `k * 64 <= blocks`, otherwise median partition. The selected tie set changes
  with `k` (2 versus 3 already differ), so a lower-block-id convention is not
  equivalent. The native selector reuses the same top-k kernel as the
  second-level indexer instead of ranking pairs.
- Ten independent FP32 compressor streams cover ratio 1/2, hidden widths
  7/8/16/32, head widths 8/16/32/64/128/**512**, batch 1/2, odd/even prefills,
  incomplete groups, decode, and repeated cache resets. All 9728 compared latent
  values pass the exclusive `2e-7` absolute ceiling; worst drift on the AMD GPU
  is **`1.788139343262e-7`**, with `rtol=0`.

The quantizer fixture covers 11 cases and 1600 values across FP8/FP4, both scale
storage formats, zero/tiny/saturating blocks, and explicit round-to-nearest-even
ties. FP8 conversion is checked with PyTorch's CPU E4M3 cast; FP4 has no CPU cast,
so the oracle uses the published 16-value E2M1 set with the same tie rule. All
values and stored scale bits match exactly.

The rotary fixture extracts the unchanged `precompute_freqs_cis` and
`apply_rotary_emb` bodies for five configurations, including the released
compressed and sliding theta/YaRN settings. The script separately rebuilds the
FP32 angle table without trig and verifies that `torch.polar` over it reproduces
the reference complex table exactly; every position angle then matches natively
bit-for-bit. Cosines/sines and both forward/inverse applications are checked at
the fixed `2e-7` ceiling; worst native table drift is `5.960464477539e-8` and
worst applied-value drift is `1.192092895508e-7`. Providing the reference's exact
cos/sin table to the same kernel gives identical application results, which
isolates arithmetic from trig generation.

The index-key fixture extracts the unchanged `RMSNorm`,
`precompute_freqs_cis`, and `apply_rotary_emb` bodies and uses FP32 weight/
activation stand-ins with the released FP4-32/UE8M0 quantization. Four cases
cover prefill, batch 2, decode, and ratio 1; all 608 quantized key values match
exactly (worst drift `0`). The selection fixture copies only the inline top-k
lines of `Indexer.forward`, covers start-zero prefill, decode, chunked, ratio-1,
and candidate-masked rows, and all 117 selected indices match exactly. The score
kernel is additionally checked against a host float64 reference for visibility,
candidate masking, head weighting, and cache strides (`1.396984e-8` worst on
binary-fraction inputs). That diagnostic does **not** establish PyTorch/MKL
reduction parity.

The attention fixture extracts the released `get_window_topk_idxs` verbatim and
mirrors the `sparse_attn` body for the gather, because that kernel itself needs
TileLang. It records seven window schedules (fit, exact, wrap, window-of-one,
filling ring, ring decode, full ring) with 93 exact indices, and five gather
cases with 1408 float64 values covering prefill-addressed and ring-addressed
compressed offsets, rows with no compressed slot, and a sliding-only step. The
native gather matches at worst drift `7.825e-8`; the native window function
matches every index exactly. Rows whose entire gather is `-1` produce zeros, as
in the released kernel's finite-bound convention.

The Engram oracle extracts the unchanged `find_next_prime` and `EngramLayout`
bodies from `engram.py` and the `NgramHashState` and `Engram` bodies from
`model.py`, after verifying both source hashes, and executes their released
forward code. The released hash state derives the compressed token map through
the tokenizer and the multipliers through NumPy; both are conversion artifacts,
so the fixture records them as inputs and the native side must reproduce the
released hash given the same values. Layout cases cover a tiny table and the
released 384M-row table (48 released primes and their offsets). Hash cases cover
a four-token lookback with a masked span, and a three-token case with a dead
span, a fully masked step, and a decode token whose n-gram reaches into the
previous step's cache. Gate cases cover unmasked, masked, a 17-row step wider
than one kernel row group, and dead/negative plus past-the-end ids. The
embedding stand-in is the released sharded lookup with its rows already
dequantized to the BF16-exact contract, including the released out-of-range
masking that turns those ids into zero rows.

The compressor oracle extracts the unchanged released `RMSNorm` and `Compressor`
class bodies. Its bias-free Linear stand-in uses `torch.nn.Linear`; all parameters
are promoted to FP32 before installing identical fixture weights. This is an
explicit unquantized FP32 component contract, **not** evidence for the production
checkpoint's BF16/FP8/FP4 or intermediate quantization policy. Validation uses
PyTorch `2.14.1+cpu`, source commit
`5c4886908584029761b579af026dcfb627c84070`, with one CPU thread. Reduction
provenance is recorded in that revision's
[CPU SumKernel](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/SumKernel.cpp).

The strict component test initially detected `3.576278686523e-7` output drift.
It was repaired through dedicated projection and RMS reduction staging, without
changing the oracle's forward code, fixture values, or threshold. The
single-threaded reference has serial-FMA matrix projections and a vector-dot
path for narrow/fewer-than-four-row projections; materialized squares use four
interleaved eight-lane partial sums before normalization. Both operation choices
matter after RMS scaling. Prefill and decode are each checked against their
corresponding reference operation shapes; the reference itself can round
differently between those GEMM shapes, so compressor cache tests do not assume
their outputs must be bit-identical. Group continuity and all emitted values
are checked independently.

The current block-ranking kernel is a quadratic correctness baseline, with no
million-token performance qualification. A hierarchical top-k implementation
is required before production long-context serving.

## Validation commands

Run from the repository root (Python/Torch are oracle tools only):

```powershell
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_candidates.py
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_compressor.py
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_quantization.py
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_rotary.py
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_index_keys.py
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_index_selection.py
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_attention.py
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_backbone.py
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_engram.py
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_model.py --reference-path .tmp-v41-ref/model.py --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41 -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v4 -- --nocapture
cargo check --manifest-path hierarchos-vulkan/Cargo.toml --lib
# Strict tiny FP32 qualification runs by default in the filters above.
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_model_strict_all_logits -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_model_additional_seeds_and_top3_strict_all_logits -- --nocapture
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_model.py --reference-path .tmp-v41-ref/model.py --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py --native-output target/deepseek-v41-reference/model-native-logits.json
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_model_additional.py --reference-path .tmp-v41-ref/model.py --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py --native-output target/deepseek-v41-reference/model-additional-native-logits.json
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_unary.py
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_packed_weights.py --reference-path .tmp-v41-ref/model.py --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py --convert-reference-path target/deepseek-v41-reference/inference_convert.py
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_mixed.py --reference-path .tmp-v41-ref/model.py --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_mixed_engram.py --reference-path .tmp-v41-ref/model.py --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_mixed_model.py --reference-path .tmp-v41-ref/model.py --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py --native-output target/deepseek-v41-reference/mixed-model-native-logits.json
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/trace_deepseek_v41_packed_engram.py --reference-path .tmp-v41-ref/model.py --native-output target/deepseek-v41-reference/packed-engram-native-trace.json
# Optional observation-only intermediate trace; never rewrites the acceptance fixture.
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/trace_deepseek_v41_model.py --reference-path .tmp-v41-ref/model.py --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py --native-output target/deepseek-v41-reference/model-native-logits.json --trace-output target/deepseek-v41-reference/precision-trace-reference.json
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/check_shader_provenance.py --output target/deepseek-v41-reference/precision-shader-provenance.json
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/check_push_constant_layouts.py
.venv-vulkan/Scripts/python.exe -m unittest discover -s hierarchos-vulkan/validation -p test_deepseek_v41_parity.py -v
.venv-vulkan/Scripts/python.exe -m unittest discover -s hierarchos-vulkan/validation -p test_shader_provenance.py -q
```

The oracle command defaults to the pinned URL. `--reference-path PATH` uses a
local copy with the same mandatory source hash; `--write-fixture` deliberately
regenerates the fixture. Shader sources were separately compiled, validated with
`spirv-val`, and reproduced byte-for-byte using `audit_shader_artifacts.py`.

Latest validation, October 23, 2026:

The mixed-expert continuation verifies **86 V4/V4.1 tests, zero failures and zero
ignored tests**, including all three strict complete-logit tests. Independent Python
comparators regenerate the five model references read-only and pass all 30
steps at the unchanged gate. Both strict-gate policy tests, all 28 provenance
tests, the push-constant check, read-only backbone/unary oracle regeneration,
formatting and `git diff --check` pass. All 514 shader artifacts reproduce
byte-for-byte under the pinned toolchains; the V4.1 kernels use glslang 16.6.0
and Vulkan 1.1. The non-test library build passes with 25 warnings, none in the
dedicated V4.1 module. Current native and shader logs are
`target/deepseek-v41-reference/mixed-regression-final.log` and
`target/deepseek-v41-reference/mixed-shader-provenance.log`; earlier
`model-*` and `precision-*` diagnostic logs retain the intermediate failures.
The preceding Engram continuation's full library result was
853 passed / 2 failed / 9 ignored; both failures were reported as pre-existing
`rwkv_channel_mix` fused-path dispatch-count assertions (`4 != 1` and
`3 != 1`) in a module this continuation did not modify. They were not bisected
further.

- Sixty-five native V4.1 tests pass, including all 32 reference mask cases on
  **AMD Radeon Graphics** (14992 exact mask values; zero differing values), 1600
  exactly matching quantizer values, 608 exact index-key values, 117 exact
  selection indices, 93 exact window indices, 1408 attention gather values at
  worst drift `7.825e-8`, and 4480 rotary application values under the fixed
  ceiling.
- The broader `cargo test ... --lib deepseek_v4 -- --nocapture` filter passes
  all 86 tests: sixty-five V4.1 tests plus 21 existing V4 tests covering
  forward, backward, optimizer steps, generation, PEFT, and checkpoint
  round-trips.
- Four orchestrator tests compare layer outputs, score/selection/candidate
  publications, and shared caches bit-for-bit against an explicit driver of
  the standalone components. The eight-layer schedule covers two batches,
  same-ratio KV-source replacement, ratio 2→1 transitions, copy indexers,
  candidate consumers, odd and one-token prefills, group-completing decode,
  ring wraps, shorter restarts, owned checkpoint tensors, and real failure
  injection after compressor/key writes. This verifies native orchestration,
  not independent full-model numerical parity. Both candidate APIs match all
  14992 reference mask values exactly. Invalid resident geometry/device inputs
  are rejected before dispatch.
- Four backbone tests compare the new kernels against the extracted reference.
  Hyper-connection coefficients, collapse, and expand match exactly over 544
  values, including the released `comb` transpose
  described above. Routing matches all indices exactly at worst weight drift
  `1.192092895508e-7` over 36 values across sqrt-softplus, softmax, and sigmoid
  with correction-bias-only selection, normalization, and route scaling. Both
  original complete MoE cases and two new cases match exactly over 419 values
  including route weights. They cover the shared expert, SwiGLU clamps,
  balanced 4/4 routing, unbalanced expert row counts `[1,3,4]`, and top-3
  cancellation that distinguishes expert-id order from selection-slot order. A
  two-layer backbone prefill and the following decode match an explicit
  standalone-component driver bit-for-bit, covering the identity initial
  pre-mix, attention/FFN pre-mix hand-off, and position continuity. A fifth
  backbone test attaches a two-layer Engram runtime at both layers and matches
  the explicit standalone-component driver bit-for-bit on prefill and on the
  following decode, which covers the layer-before-block injection,
  cache-once-per-step hash reuse, and cross-step lookback.
- Engram matches the extracted reference exactly for the discrete parts: all 48
  released primes and their offsets, and every hash id in both cases (a
  four-token lookback with a masked span, plus a dead-span case with a fully
  masked step and a decode token reaching into the cache). The gate matches at
  worst drift `5.960464477539e-8` over 448 values across four cases, including
  the wider-than-one-row-group step and the dead/negative id rows, against a
  fixture ceiling of `2e-6`.
- All 9728 compressor latent values pass at worst absolute drift
  `1.788139343262e-7`. Invalid input/positions are injected between valid decode
  calls to verify cache continuity. A foreign-device input is rejected before
  dispatch. Checkpoint tests cover single/sharded files, F32/BF16, exact source
  ownership, malformed offsets/headers, non-finite data, packed-weight rejection,
  and shard path validation.
- Both strict-logit-gate tests pass, including rejection exactly at `2e-7`,
  the next representable value above it, non-finite logits, empty comparisons,
  shape mismatches, and large-logit relative-tolerance loopholes. These test
  the acceptance policy only; the separate full-model tests enforce the same
  ceiling on all 1600 FP32/hybrid logits, as recorded below.
- Four model tests cover full-logit export/shape/finite values, single-file
  versus sharded SafeTensors with BF16 vocabulary tensors, raw constructor
  equivalence, owned HC/MoE/Engram tensor names, invalid artifact/storage
  contracts, input-error continuity, and recovery after a head failure following
  successful backbone cache writes. These are functional checks, not parity.
- A separate elementary-math test covers 41 cases and 13967 values, including
  AVX512 body/tail boundaries. 13954 values are bit-exact. The other 13 softplus
  values differ by one ULP from the pinned Windows CPU implementation; this
  component test records a one-ULP policy for softplus only. Scalar exp/log1p,
  sigmoid and SiLU comparisons are exact. No ULP allowance is used for logits.
- All 28 shader provenance contract tests pass. All thirty-two production V4.1
  shaders plus one unary qualification shader reproduce byte-for-byte, and all
  514 checked artifacts are identical under the
  pinned glslang 16.6.0 / shaderc 2023.8 toolchains. The push-constant layout
  check agrees across SPIR-V, GLSL, and host structs (395 blocks, 109 host
  structs).
- The pinned independent candidate oracle, Rust formatting check, and
  `git diff --check` pass.

The initial GPU run exposed the generic FP32 upload guard rejecting the
intentional `-inf` sparse-mask sentinel. The selector now rejects NaN/+inf
explicitly and uploads validated IEEE payload bits through integer I/O. The
subsequent complete native runs above pass without changing the oracle masks.

## Numerical continuation and full-logit results

The original six-step fixture is unchanged, SHA256
`679ebe1cdbe421ded12b911fb24dfd64995598caeb1ebe619434229444ffc7af`.
The earlier worst drift was
`3.5762786865234375e-7`, with four steps failing. Native intermediates and an
observation-only wrapper around the pinned reference localized independent
rounding/order differences. The observer checks its outputs against the existing
acceptance fixture; it cannot regenerate that fixture. GPU readbacks and trace
storage are test-only. The production graph retains GPU-resident intermediates.

The following fixes are operation-level changes, with no logit correction or
fixture-specific weight/input logic:

- HC softmax now rounds the reciprocal once, then multiplies, matching the
  pinned [CPU softmax implementation](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/SoftMaxKernel.cpp).
- Sigmoid and SiLU follow the pinned CPU vector-body/scalar-tail staging. The
  [vector loop](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/Loops.h)
  processes pairs of 16-float AVX512 vectors before the scalar tail. Two-float
  exp/log1p and an integer-mantissa sqrt implement scalar precision on the GPU
  without requiring shaderFloat64. Vector log1p adapts the licensed
  [SLEEF implementation](https://github.com/shibatch/sleef/blob/60e76d2bce17d278b439d9da17177c8f957a9e9b/src/libm/sleefsimdsp.c).
  SiLU uses direct division `x / (1 + exp(-x))`, as in the pinned
  [activation source](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/Activation.cpp),
  avoiding the additional rounding of `x * sigmoid(x)`.
- Engram uses the rounded sqrt and preserves the sign bit of zero when applying
  the gate's copysign. Identical-input traces distinguish the KV projection
  from the gate and residual arithmetic.
- MoE counts selected token rows per expert on the GPU. Each expert uses its
  own routed-row geometry for projection reduction and its compact token-row
  index for activation body/tail selection. Outputs accumulate in ascending
  expert-id order, followed by the shared expert, matching the reference loop.
  A top-3 cancellation fixture makes that order observable.
- The grouped output projection uses the four-accumulator, separately rounded
  product tree for short rows. Identical-input replay showed exact quantized
  inputs but differing projections in decode/reset before this fix.

**Current baseline model results (strict `<2e-7`, `rtol=0`):**

| Step | Compared logits | Maximum absolute drift | Gate |
| --- | ---: | ---: | --- |
| Odd prefill, length 5 | 160 | `1.1920928955078125e-7` | PASS |
| Group-completing decode, position 5 | 32 | `5.9604644775390625e-8` | PASS |
| Incomplete-group decode, position 6 | 32 | `1.78813934326171875e-7` | PASS |
| Ring-wrap decode, position 7 | 32 | `8.94069671630859375e-8` | PASS |
| Shorter reset, length 1 | 32 | `1.1920928955078125e-7` | PASS |
| Group completion after short prefill | 32 | `1.1920928955078125e-7` | PASS |

Two additional weight seeds and routing widths were fixed before measuring
them, using the same input cases and weight scales:

| Weight seed | Selected experts | Cases | Logits | Worst absolute drift | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `4141017` (original) | 2 | 6 | 320 | `1.78813934326171875e-7` | PASS |
| `4141018` | 2 | 6 | 320 | `1.78813934326171875e-7` | PASS |
| `4141019` | 3 | 6 | 320 | `1.78813934326171875e-7` | PASS |

Both Rust strict tests run in the regular suite, and the independent Python
comparators pass all 18 cases. Comparisons bind native exports to the fixture
SHA256 and check every finite logit with `rtol=0`. The original fixture,
reference forward bodies, weight scales and acceptance comparator were not
changed to obtain this result. The final RMSNorm/head remain exact on identical
native inputs. This is evidence for the tested tiny FP32 text geometry only;
production BF16/FP8/FP4 semantics, YaRN long contexts, DSpark, vision and training
remain unqualified.

## Packed-weight continuation

`src/deepseek_v41/packed_weights.rs` adds explicit validated byte storage,
device upload, GPU dequantization, and experimental block-scaled linear output.
The scale layout is required by the API; it is never guessed from I8/U8 alone.
The reader accepts explicit exported or converted tensor/scale names, including
payloads stored in different shards. This does not yet normalize an entire
exported HF checkpoint into the native backbone's naming/precision policy.

| Format | Logical weight shape | Stored payload | UE8M0 scale shape |
| --- | --- | --- | --- |
| FP8 linear, block 32 | `[N,K]` | `F8_E4M3 [N,K]` | `[ceil(N/32),K/32]` |
| FP8 linear, block 128 | `[N,K]` | `F8_E4M3 [N,K]` | `[ceil(N/128),K/128]` |
| FP4 expert | `[N,K]` | exported `I8 [N,K/2]` or converted `F4 [N,K]` | `[N,K/32]` |
| FP8 Engram | `[N,K]` | `F8_E4M3 [N,K]` | `[N,K/32]` |

FP4 stores the earlier K element in the low nibble. SafeTensors F4 headers use
the logical number of elements, even though PyTorch's packed x2 tensor shape
counts bytes. Scale names are explicit: exported weights typically use
`weight_scale_inv`, and converted inference weights use `scale`. The published
model's Linear default is block 32; block 128 is a supported kernel primitive,
not a claim about the package's default configuration. UE8M0 code zero means
`2^-127`; code 255 is rejected as NaN. FP8 NaN codes, malformed shapes/byte
counts, invalid offsets, missing scales, overflow to non-finite materialized
FP32, and foreign-device inputs are rejected. Byte uploads use padded u32
buffers and require no native FP8/FP4 Vulkan feature or shaderFloat64.

The GPU decoder constructs scaled FP32 bits directly, including subnormal
weights, then optionally performs the BF16 round-to-nearest-even cast. Its
two-dimensional flattened dispatch supports tables beyond 65535 rows; a
65536×64 case checks all 4194304 decoded outputs. The block-scaled linear
primitive quantizes activations to FP8 with the existing native quantizer,
resets the FP32 dot for each K group, applies activation then weight scale,
accumulates groups, and optionally rounds the result to BF16. It consumes
resident GPU input/weights without activation readbacks. A dequantized FP32
weight matrix is deliberately not substituted into an ordinary full-width GEMM.

Engram now retains its table as packed FP8 bytes plus row/block UE8M0 scales.
`new_packed` and `EngramRuntime::from_checkpoint` select a GPU gather/decode
kernel that dequantizes only selected rows and rounds them to BF16 before the
existing FP32 projection. Dead ids produce zeros. No whole-table FP32 expansion
occurs in this runtime path. The standalone dense constructor and existing
unquantized model fixtures retain their contract. Converter token-map/hash
artifacts are still required. At that milestone other projections retained their
FP32 policy; the following continuation records packed expert/WKV integration.

Independent evidence comes from
`validation/validate_deepseek_v41_packed_weights.py`. It hash-checks the pinned
model/kernel and converter SHA256
`035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76`.
PyTorch performs FP8/UE8M0 decoding and BF16 casts; FP4 magnitudes come from the
converter's literal table, preserving the format's negative-zero sign bit.
CPU matrix products implement each released K block before applying the
released scale expression. The Engram cases execute unchanged released
`ParallelEngramEmbedding.forward` and `Engram.forward`, with exact BF16-to-FP32
promotion at the experimental graph's FP32 projection boundary. Source bodies
and the existing full-model acceptance fixtures/comparator are unchanged.

Nine new native tests verify:

- 72960 FP32/BF16 decode values bit-for-bit across six cases, all finite FP8/FP4
  codes, all 255 finite UE8M0 exponents, signed zero and subnormal boundaries.
- Eight linear cases with 2132 outputs, zero absolute drift against the CPU
  block reference; BF16 outputs must be bit-exact and FP32 outputs enforce
  strictly `<2e-7`, with no relative tolerance.
- 4544 packed lookup values bit-for-bit against the released embedding method,
  including invalid/negative ids, 32/64-wide rows, HC widths 2/4 and masked rows.
  All 2224 residual outputs are bit-identical between packed and expanded-table
  paths. Against the independent CPU residual, the wider case reaches
  `1.6689300537109375e-6` under the existing Engram component ceiling `2e-6`.
  That ceiling is unchanged and does not qualify full-model logits. The shared
  projection/gate arithmetic for wider geometry remains a precision risk.
- Packed checkpoint/runtime cache continuity and shorter reset, exported-I8 vs
  logical-F4 byte identity, sharded scales, the large dispatch boundary, malformed
  storage, missing scales, and device/geometry rejection.

These are CPU PyTorch component results on AMD Radeon Graphics. Production
mixed-precision logits remain unmeasured; CUDA parity is not required.
The original 960 tiny FP32 logits still pass strictly `<2e-7`, `rtol=0`, worst
`1.78813934326171875e-7`. The normal V4.1 model loader remains disabled.

## Mixed-expert and dense-projection continuation

`mixed_linear.rs` preserves F32, BF16, and packed FP8/FP4 parameter storage
semantics in a typed API. Dense F32/BF16 projections require the matching input
precision; invalid dtype combinations fail rather than silently promoting the
operation. A native cast kernel applies BF16 round-to-nearest-even at explicit
boundaries and retains FP32 storage for the resulting BF16-exact values. Packed
projections quantize that input, run the existing per-K-block scaled GEMM, and
emit BF16 as the released kernel does. Resident weights and activations stay on
the GPU. BF16 payloads are currently promoted into F32 buffers for arithmetic;
packed weights retain byte storage.

`DeepseekV41MixedMoe` now executes whole routed/shared experts. The gate remains
FP32. A GPU compaction pass gathers each expert's selected tokens in the
released `torch.where` order, including its FP32 route weights. The W1/W3
projections emit BF16, then SwiGLU clamps, SiLU, up multiplication and optional
route multiplication run in FP32. The activation is cast to the original input
precision **before W2**, matching the released `Expert.forward`. A caller using
BF16 gets BF16 at this boundary; the diagnostic FP32 backbone keeps FP32 here.
W2 emits BF16. A GPU scatter places expert outputs back into token/slot order;
the existing combine kernel adds them in ascending expert id and then adds the
shared expert, before casting to the original stream precision. Counts, row
maps and route weights are never read back to drive execution. Empty experts
produce no scattered output. The current correctness implementation pads each
expert's compact allocation to the maximum step rows and is not a serving GEMM.

Checkpoint-backed backbone construction selects this executor when routed
weights use packed storage. Each layer's payload is still streamed into device
buffers. The old `load_backbone_layer` FP32 weight-vector API rejects packed
experts; the mixed checkpoint constructor preserves their storage. Mixed MoE
currently requires packed W1/W2/W3 for all routed and shared experts. The
non-quantized legacy executor retains its existing contract.

Engram can now combine a packed table with a typed WKV projection. Packed WKV
keeps its weights resident and emits BF16; projection and residual stream
precision are explicit. `load_mixed_engram` and `new_mixed` enforce dimensions,
scale layouts and parameter ownership. The checkpoint-backed experimental
backbone still uses an FP32 stream around this module. A dedicated BF16
component fixture tests the released BF16 input/output semantics separately.

The wider HC=4 FP32 issue was traced on identical values without changing its
fixture: WKV differs by `1.9073486328125e-6`, and the final residual differs by
`1.6689300537109375e-6`; **replaying the released gate on the actual native WKV
has zero drift**. Thus this measured difference originates in the wide FP32
matrix projection. It remains a risk for that FP32 geometry. The new packed WKV
path follows the released quantized projection and BF16 output contract, rather
than using this unqualified wide FP32 reduction.

Seven new native tests add the following evidence:

- Five typed projection cases compare 2087 outputs against independent PyTorch
  operations; BF16 outputs are bit-exact, and the F32 case enforces `<2e-7`.
  Precision/device/shape/storage rejection and BF16 ties, subnormals, signed
  zeros and non-finite classes are checked. No non-finite allowance is added to
  the logit gate.
- Four packed MoE cases execute unchanged released `Expert.forward`,
  `Gate.forward` and `MoE.forward` bodies with explicit CPU GEMM mirrors. All
  **1504 outputs are bit-exact**. They cover BF16 and F32 inputs, FP4/FP8
  experts, an FP8 shared expert, top-3 routing, correction bias, negative route
  scaling, clamps, uneven counts and empty experts. Indices/counts are exact.
- Two packed-table/FP8-WKV/BF16-stream Engram cases use the same fixed wider
  inputs as the preceding fixture. All 2792 WKV outputs and **2224 BF16 residual
  values match bit-for-bit**, including HC=4. A checkpoint/runtime test checks
  the typed projection path, stream policy, cached decode and shorter reset.
- A regular strict full-model test loads actual packed SafeTensors and compares
  two complete hybrid text models, fixed seeds `4141041`/`4141042`, with F32
  attention/HC/norm/Engram/head around FP4 routed and FP8 shared experts. The
  intermediate width is 32, so all packed K groups are valid. The FP32 model's
  original weights are quantized once into stored payloads before native results
  are measured; input cases and vocabulary head scales are unchanged. All
  **640 logits across 12 prefill/decode/reset steps have zero absolute drift**
  against the independent CPU mirror. Inputs and logits are nonzero, complete,
  finite and bound to the fixture SHA256. Every comparison still uses strict
  `<2e-7`, `rtol=0`.

Together with the three unchanged FP32 models, this is **1600 logits across
30 steps and five tiny models**, with overall worst drift
`1.78813934326171875e-7`. The original model fixture SHA256 is unchanged. New
oracles are `validate_deepseek_v41_mixed.py`, `..._mixed_engram.py`, and
`..._mixed_model.py`; `trace_deepseek_v41_packed_engram.py` is observation-only.
Their outputs do not redefine the existing acceptance comparator. None of this
qualifies packed attention/indexer projections, the complete BF16 residual/cache
graph, production dimensions or training. The
normal model loader remains disabled.

## Packed attention/indexer continuation (CPU PyTorch)

`DeepseekV41MixedAttentionWeights` and `DeepseekV41MixedIndexerWeights` retain
typed projections from SafeTensors through Vulkan execution. Checkpoint loading
now selects this path whenever an owned attention/indexer projection is not
F32. WQ-A, WQ-B, WKV, WO-B and indexer WQ-B remain packed; norms, BF16 indexer
WK/head weights and the converter's dense grouped WO-A retain their respective
semantics. Shared consumers allocate no duplicate key weights or caches.
The old public F32 constructors and raw dense-weight loader remain compatible.
An unconverted packed WO-A is rejected explicitly.

Explicit BF16 rounding now occurs after projection, query/KV normalization,
rotary, quantize/dequantize, sparse-attention output, inverse rotary and grouped
output as appropriate to each projection's dtype. Index scoring also preserves
BF16 dot output, scaled head weights, products and the final head sum; applying
only one cast after the sum would implement different arithmetic. These casts
and the score stages stay on Vulkan without activation readback.

Two new fixed complete models, seeds **4141051 (top-2)** and **4141052 (top-3)**,
use Q/O ranks 32, intermediate size 32, all 27 attention/indexer projections in
FP8/UE8M0, all routed experts in FP4 and shared experts in FP8. Each checkpoint
contains **99 packed projections plus 11 declared dense BF16 projections**.
One case exercises indexed shards. All 640 logits across the existing six
prefill/decode/reset steps per model pass **strictly `<2e-7`, `rtol=0`**:

- Seed 4141051: zero drift for every logit.
- Seed 4141052: worst `1.1920928955078125e-7` (odd prefill); group-completing
  decode reaches `5.9604644775390625e-8`, and the other four steps are exact.
- Observation of every attention projection input, grouped output and attention
  output finds **30,720 component values bit-exact** across the two models.
  Residual/HC F32 intermediates can still differ; they are not adjusted, and
  every complete logit is evaluated by the original strict comparator.

This is a declared hybrid contract: the CPU WO-B wrapper promotes the BF16
GEMM result to F32 at the residual boundary, matching the native F32 backbone.
The released Attention/Indexer/Block/Expert/MoE forward bodies are unchanged.
This phase does **not** implement the complete production BF16 residual graph.
No seeds, weights, inputs or arithmetic were changed after measuring drift.
The old acceptance fixtures and threshold remain unchanged.

`validate_deepseek_v41_mixed_attention.py` adds independent stateful component
cases with two output groups, head widths 32/64, query ranks 32/64, original
F32/BF16 inputs, prefill, ring wrap, decode and shorter reset. All **1,600 query
latents, 1,600 attention outputs and 3,072 cache values are bit-exact**. Four
additional BF16 index-score cases cover head widths up to 128 and 32 heads,
negative head weights, ratios 1/2 and causal/candidate masks: **all 89 score
entries match by bits**, including 39 finite values and 50 masked infinities.
These masked component outputs do not weaken the full-logit nonfinite rejection.

Reproduce without rewriting acceptance fixtures:

```powershell
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v4 -- --nocapture
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_mixed_attention.py --reference-path .tmp-v41-ref/model.py --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_attention_model.py --reference-path .tmp-v41-ref/model.py --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py --native-output target/deepseek-v41-reference/attention-model-native-logits.json --trace-output target/deepseek-v41-reference/attention-model-reference-trace.json
```

The strict suite now covers **2,240 logits in seven models across 42 steps**.
Overall worst drift remains `1.78813934326171875e-7`. CPU PyTorch is the accepted
reference. Production BF16 residual/HC, compressor, Engram and final norm/head
boundaries, production dimensions, training, DSpark and vision remain open.

Verification for this continuation:

- **91 V4/V4.1 Rust tests pass, zero failed/ignored** (70 V4.1 + 21 existing V4).
  Log: `target/deepseek-v41-reference/attention-regression-final.log`.
- Original, additional, packed-expert and packed-attention CPU oracles reproduce
  their fixtures without `--write-fixture` and accept every exported native
  logit. Logs: `attention-{original,additional,experts,model}-comparison.log`
  under `target/deepseek-v41-reference/`.
- All **514 shader artifacts reproduce byte-for-byte**; push-constant layouts
  agree across **395 shader blocks and 109 host structs**. The modified index
  score shader uses pinned glslang 16.6.0, Vulkan 1.1, and passes `spirv-val`.
  Report: `target/deepseek-v41-reference/attention-shader-provenance.json`.
- 28 shader-provenance contract tests and two unchanged strict-logit-policy
  tests pass. Non-test `cargo check --lib` passes (25 existing warnings, none in
  V4.1), as do rustfmt checks, Python compilation and `git diff --check`.
- Original model fixture SHA256 is still
  `679ebe1cdbe421ded12b911fb24dfd64995598caeb1ebe619434229444ffc7af`.
  New full-model fixture SHA256:
  `3cfbad274144417849b83b25c9f686e9439f4f2c7c38d645ee86819066a350df`;
  new component fixture SHA256:
  `3b8982545d10b45cbb32d7558244d934b0ca17eef8867a36047f240677e5b47b`.

## Complete packed BF16 residual/HC continuation (CPU PyTorch)

`DeepseekV41CausalLM::from_checkpoint_with_precision(...,
DeepseekV41WeightOutput::Bf16)` now selects the explicit BF16 stream graph.
The existing constructor continues to select F32/hybrid execution. BF16 values
use promoted F32 Vulkan buffers after each required rounding; no Python,
PyTorch or activation readback is used by the execution path.

The explicit precision is carried through the backbone, attention/indexer,
compressor, compressed-KV finalizer and Engram runtime. Implemented boundaries:

- BF16 embedding values expand into the HC stream. HC coefficients stay F32;
  each collapse and residual expansion rounds to BF16, as do block RMSNorms.
- Packed MoE consumes/returns the BF16 stream with the existing routed/shared
  accumulation order. Attention retains its previously implemented BF16
  projection, norm, rotary, sparse-output and grouped-output boundaries.
- Ratio-1 compression rounds the dense projection before RMSNorm. Ratio-2
  projections, unfinished-group state and pooling stay F32, then the pooled
  vector rounds to BF16 **before** RMSNorm; normalized latents round again.
- Compressed latents remain unmodified for index-key production. Cache
  finalization rounds after rotary and after FP4/E4M3 dequantization.
- Packed Engram lookup/WKV and its gated residual now run within the same BF16
  stream. Final HC collapse and final RMSNorm round to BF16; the vocabulary
  projection consumes promoted values and emits complete F32 logits.

BF16 construction rejects F32 attention/indexer projections, non-BF16-exact
embeddings or ratio-1 compressor weights, an Engram/backbone precision mismatch,
and unsupported dense expert/Engram-table packages. This phase qualifies the
released **packed** storage path; an all-dense BF16 expert/table path is not
implemented by this API. Token-map/hash converter artifacts remain required.

Two new models fixed before native measurement, **4141061 (top-2, HC=2)** and
**4141062 (top-3, HC=4)**, use hidden size 32, Q/O ranks 32, intermediate size 32,
Engram head width 32 and WKV input width 128. Each checkpoint contains **101
packed projections plus two packed Engram tables and 65 dense BF16 tensors**;
HC and ratio-2 projection tensors retain F32 storage. The reference head keeps
its BF16 checkpoint values promoted to F32, as in the released class. One model
loads indexed shards. The released Attention, Indexer, Block, Compressor,
Engram, Expert, MoE, RMSNorm and head forward bodies remain unchanged.

The first full run exposed a real cache mismatch despite identical query
inputs, index scores and selections. Ordinary Vulkan division can under-round
normalization by one F32 ULP at an exact FP4 midpoint with a non-power-of-two
E4M3 scale. This changed one cache value from the CPU's `0.75` to `0.5625` and
caused decode-logit failures. The V4.1 quantizer now uses the existing exact
normal-quotient `fp32_div` helper for normalization and E4M3 scale formation.
No reference fixture, weight, input or tolerance was changed to pass the gate.
The new midpoint regression checks **2,016 outputs and 126 scales**, covering
every positive finite E4M3 scale against the unchanged CPU quantization policy.

After that fix:

- **Every one of the 640 new logits is bit-exact**, across all 12 full prefill,
  group completion, incomplete group, ring wrap, shorter reset and post-reset
  decode steps. The strict `<2e-7`, `rtol=0` comparator is unchanged.
- **202,960 HC/stream/norm/projection intermediate values, 28,928 shared cache
  values, 476 index-score entries and every selected index match by bits.**
  Sparse attention's intermediate BF16 output is separately observed: maximum
  drift `1.1920928955078125e-7`; subsequent projection inputs and all complete
  logits are exact. Its discrepancies are reported, not substituted or hidden.
- Existing F32 and hybrid fixtures still reproduce from the independent CPU
  reference and pass all logits. The suite totals **2,880 logits across nine
  models and 54 steps**, overall worst `1.78813934326171875e-7`.

Reproduce without rewriting fixtures:

```powershell
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v4 -- --nocapture
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_bf16_quantization.py --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_bf16_model.py --reference-path .tmp-v41-ref/model.py --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py --native-output target/deepseek-v41-reference/bf16-model-native-logits.json --trace-output target/deepseek-v41-reference/bf16-model-reference-trace.json
```

Production hidden/reduction widths, larger vocabularies, long YaRN contexts and
serving performance remain unqualified. The normal loader remains disabled.
CPU PyTorch is the accepted reference; CUDA parity is not a requirement.

Verification for this continuation:

- **94 V4/V4.1 Rust tests pass, zero failed/ignored** (73 V4.1 + 21 existing V4),
  including the complete BF16 hard gate, FP4 midpoints and precision-contract
  rejection. Log: `target/deepseek-v41-reference/bf16-regression-final.log`.
- All five full-model CPU oracle scripts reproduce their frozen fixtures and
  accept every exported native logit. The BF16 trace comparison additionally
  enforces the intermediate/cache/index checks above.
  Log: `target/deepseek-v41-reference/bf16-model-comparison.log`.
  Aggregate: `target/deepseek-v41-reference/bf16-qualification-summary.json`.
- Reference environment remains single-threaded PyTorch **2.14.1+cpu**, commit
  `5c4886908584029761b579af026dcfb627c84070`, AVX512, with no compiled CUDA.
- All **514 shader artifacts reproduce byte-for-byte**; the modified quantizer
  uses glslang 16.6.0, Vulkan 1.1 and passes `spirv-val`. Layout checks still
  cover **395 shader push blocks and 109 host structs**. Report:
  `target/deepseek-v41-reference/bf16-shader-provenance.json`.
- 28 shader-provenance contract tests and two strict-logit-policy tests pass.
  Non-test `cargo check --lib` passes with 25 existing warnings, none in V4.1.
  Rustfmt checks, Python compilation and `git diff --check` pass.
- Original model fixture SHA256 remains
  `679ebe1cdbe421ded12b911fb24dfd64995598caeb1ebe619434229444ffc7af`.
  New BF16 model fixture SHA256:
  `178c7585504768474db206e62a388ea1af00b567939603c6561645947ffd37cd`;
  new FP4 midpoint fixture SHA256:
  `2f2dd2dcc16043be7a3101e922ebaca0b8b10b4b5c60b86949d5863142eec87a`.

## Remaining qualification risks

1. **Trig table portability.** PyTorch's CPU `cos`/`sin` on FP32 differ from a
   correctly rounded double evaluation by one ulp in roughly 8% of the released
   rotary table entries (533 of 6400 sampled angle pairs). The native table is
   within `5.960464477539e-8` absolute, so the component gate passes, but the
   one-ulp difference is upstream of every rotated activation. Bit-exact
   full-model parity requires reproducing the reference trig implementation
   (PyTorch/Sleef on CPU) or
   qualifying against a reference that consumes an explicitly pinned table.
2. **GEMM reduction order.** A single 128-wide FP32 dot product computed with
   serial FMA differs from PyTorch `einsum`/MKL by up to `1.43e-6` on random
   inputs; a 512-wide reduction differs by up to `9.54e-6`. Both exceed the
   `2e-7` ceiling before 32 index heads are summed. An independent native
   reduction order remains unqualified for these wider shapes against the
   pinned CPU PyTorch reference. The passing tiny model does not cover these
   production reductions. Resolving this requires reproducing the kernel's reduction
   tree for each GEMM
   shape (CPU MKL). This is a blocker, not a tolerance change. The fine indexer's
   end-to-end scores use this independent reduction; only its key production and
   discrete selection are currently claimed as qualified.

3. **Engram conversion artifacts.** The compressed token map and the per-layer
   odd hash multipliers are produced outside the model graph by the released
   converter (tokenizer pooling and NumPy's seeded RNG). The native runtime takes
   them as validated inputs, so it is qualified only against a reference given
   the same map and multipliers; reproducing the converter itself is a separate
   contract. The original FP32 model ingests post-dequantization BF16-exact table
   values. Packed Engram lookup and the new packed projection/BF16 residual path
   separately match the released component methods. The original F32 gate's component ceiling is
   the fixture's `2e-6`, not the `2e-7` all-logit gate; its measured drift is
   `5.960464477539e-8` over 448 values. The tiny FP32 all-logit cases now include
   this component and pass; whole production-model precision remains separate.
4. **Elementary-math portability.** The pinned CPU reference is PyTorch
   `2.14.1+cpu`, AVX512, single-threaded, on Windows. Unary coverage includes
   the body/tail transitions and finds thirteen one-ULP softplus differences
   from Windows CRT `log1pf`; the scalar path also passes an independent rounded
   float64 log1p oracle exactly. The one-ULP component allowance is explicitly
   separate from the unchanged strict full-logit requirement. Production
   activation ranges require their own CPU all-logit evidence.

The compressor/quantizer/rotary/Engram component contracts retain their stated
precision. Complete tiny FP32, hybrid and packed BF16 logits pass the required gate.
Production-dimension mixed-precision logits remain unmeasured, and the normal
V4.1 model loader stays disabled.

## Wider packed BF16 / YaRN CPU continuation (2026-10-23)

The next two reference models were fixed before the first native measurement in
`validation/validate_deepseek_v41_wide_model.py`. Their source, seeds, weights,
initialization scales and inputs were retained through every failed run and
native repair. The complete-logit acceptance requirement remains strictly
**absolute drift `<2e-7`, `rtol=0`**, with every position and vocabulary output
checked, including the entire prefill. CPU PyTorch remains the accepted reference;
CUDA parity is not required.

| Seed | Hidden / HC | Attention heads / width | Index heads / width | Vocab | Q / O ranks | MoE intermediate | Top-k experts | YaRN original context |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4141071 | 128 / 4 | 4 / 128 | 4 / 128 | 97 | 128 / 64 | 128 | 3 | 16 |
| 4141072 | 256 / 4 | 4 / 512 | 32 / 128 | 129 | 256 / 128 | 256 | 2 | 32 |

Both models use six layers, two output groups, 64-wide rotary tails, sliding
window 16, YaRN factor 4, and two batches. Each runs a 65-token odd prefill,
four decodes across completed/incomplete compression groups, a three-token reset
and a final group-completing decode. Ratio-1/2 compression, shared KV/index
layers, candidate selection, nonzero masks, Engram and all three routed experts
remain active. The 512-wide attention head and 32-by-128 index geometry match
those dimensions of the released model. Depth, hidden width, ranks, expert
counts, vocabulary and context are still reduced.

### Native numerical repairs

The first run failed 12 of 14 complete-logit steps, with worst drift
`1.3934855461120605`. All failing cases remained in the suite. Repairs were made
to the Rust/Vulkan arithmetic, without changing reference forward bodies or
acceptance data:

1. FP32 projections now reproduce the pinned single-threaded MKL SGEMM's K
   panels: up to 192 elements, with the 193..383 range split at `K // 2`.
   Dense BF16 projections have an explicit separate accumulation mode, retaining
   one full K accumulation before the BF16 result cast. The typed mode also
   applies to ratio-1 compressor projections. Applying FP32 panel staging to
   BF16 changed one index-key quantization boundary and failed full logits;
   keeping the two modes explicit resolves it.
2. HC and RMS square reductions now use the CPU sum kernel's four-level
   cascade, four interleaved eight-lane vectors and ascending final lane fold.
   A flat interleaved reduction differed once the HC width reached 1024.
   The reference's sum dispatch uses AVX2 even on this AVX512 CPU.
3. Sparse attention now uses SLEEF exp, the CPU softmax's 16-lane reduction,
   an inclusive sink maximum, a rounded reciprocal followed by multiplication,
   and fused value accumulation. GLSL exp, a serial denominator fold and
   separately rounded value products could cross BF16 boundaries.
4. Fine index selection now matches the pinned Windows CPU's score-only heap
   and median-partition movement, including tied finite scores and masked
   infinities. Adding a lower-position tie break selected different sets.
   The released forward sorts selected positions and only then maps future
   positions to `-1`; this order is preserved. All index scratch stays on the
   Vulkan device. Microsoft STL attribution and its complete license are in
   `THIRD_PARTY_NOTICES.md` and `licenses/Microsoft-STL.txt`.

Relevant CPU source is pinned to torch revision
`5c4886908584029761b579af026dcfb627c84070`:
[softmax](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/SoftMaxKernel.cpp),
[cascaded sum](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/SumKernel.cpp),
and [top-k dispatch](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/TopKImpl.h).
SGEMM panel behavior was independently checked against CPU projections, including
odd K values around both panel boundaries. These are measured Windows CPU
contracts, not a claim that every CPU backend uses identical internal ordering.

### Fixtures and measured results

Each wider case contains 103 dense tensors and 103 packed tensors, with the
corresponding 103 scale tensors. The first uses one SafeTensors file; the second
uses two indexed shards, with scales and weights able to cross shard boundaries.
Every config/index/payload file is SHA256-bound in the small model manifest and
verified before native checkpoint loading. The checkpoint files total
20,573,861 bytes, and the JSON manifest is 2,475,253 bytes. The manifest retains
all 32,996 reference logits plus the small norm/head replay data. Weight arrays
are not expanded into compiled JSON. Native tests use the real on-disk packages.

- **32,996 / 32,996 new logits are bit-exact**, across all 14 steps.
- The complete suite now covers **35,876 logits, 68 steps and 11 fixed models**,
  with worst drift still `1.7881393432617188e-7`. The four complete packed BF16
  models account for 33,636 bit-exact logits. Every older fixture regenerates
  unchanged and passes the same strict comparator after the kernel repairs.
- Shared caches match over **2,444,288 values**. All **71,848 index-score
  entries** and **11,236 selected positions** match by bits, counting the
  exported shared consumers as well as source layers.
- Of **15,200,352 observed intermediate values**, 15,200,340 match by bits.
  Four sparse-attention values differ (maximum `0.00390625`), seven grouped
  output-projection values differ (maximum `0.015625`), and one quantized
  attention-projection input differs by `1.52587890625e-5`. The larger values
  are BF16 rounding differences. Subsequent packed projections absorb these
  particular differences: block outputs, residual streams, final norm/head
  inputs and all logits are bit-exact. These intermediate limits remain explicit;
  the full-logit gate has not been relaxed, and larger shapes still need evidence.
- `validate_deepseek_v41_wide_kernels.py` independently freezes **24 projection
  geometries in both FP32 and BF16** (4,608 outputs, all bit-exact), including
  K=191/192/193/224/256/288/320/352/383/384/512/1024 and two/six rows.
  It also freezes **68 tied top-k cases**, widths 31..1024 and k up to 512:
  all 13,120 selected outputs match the unchanged CPU PyTorch operation.

Final verification:

- **96 V4/V4.1 Rust tests pass**, zero failed/ignored (75 V4.1 + 21 V4),
  including the wider complete-logit gate and the new typed projection/tie tests.
- Six full-model oracle commands regenerate every frozen model and checkpoint
  without `--write-fixture`, and accept all native logits.
- All **514 SPIR-V artifacts** rebuild byte-for-byte with their pinned compilers.
- Push-layout validation passes for **395 shader blocks / 109 host structs**;
  28 provenance contract tests and two strict parity policy tests pass.
- `cargo check --lib` passes with 25 existing warnings and no V4.1 warning.
- Rust formatting, Python compilation and `git diff --check` pass.
- Wider fixture SHA256:
  `247e62027ddbadacc8e31d199a7f658b8c6cbf29a3929b2ca093eae4a02680be`.
  The original FP32 fixture SHA256 remains
  `679ebe1cdbe421ded12b911fb24dfd64995598caeb1ebe619434229444ffc7af`;
  the earlier BF16 fixture remains
  `178c7585504768474db206e62a388ea1af00b567939603c6561645947ffd37cd`.

Reproduction (all paths relative to this repository):

```powershell
$env:HIERARCHOS_V41_WIDE_TRACE = '1'
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v4 -- --nocapture
Remove-Item Env:HIERARCHOS_V41_WIDE_TRACE
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_wide_model.py --reference-path .tmp-v41-ref/model.py --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py --native-output target/deepseek-v41-reference/wide-model-native-logits.json --trace-output target/deepseek-v41-reference/wide-reference-trace.json
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_wide_kernels.py
```

The trace environment variable only enables diagnostics; complete-logit acceptance
runs unconditionally. Detailed outputs are in the ignored working directory
`target/deepseek-v41-reference/`: `wide-regression-final.log`,
`wide-oracle-confirmation.log`, `wide-model-native-logits.json`,
`wide-reference-trace.json`, `wide-qualification-summary.json`,
`wide-shader-provenance.json` and `wide-library-check.log`.

Next qualification remains full hidden/rank/expert/vocabulary dimensions and
substantially longer contexts, including candidate-block ties beyond the measured
17 blocks, other prefill sizes and the remaining intermediate BF16 differences.
The 65-token test crosses its deliberately reduced YaRN training boundary; it
does not qualify the released 1,048,576-token context. The normal loader remains
disabled. All-dense BF16 experts/tables, training, DSpark and vision remain open.

## Long-context and candidate-tie continuation (2026-10-23)

This continuation closes three items from the list above at reduced widths: a
512-token prefill (eight window wraps), an odd 129-token reset prefill, and tied
candidate-block maxima over 128 blocks. The acceptance requirement is unchanged:
absolute complete-logit drift strictly below `2e-7` with `rtol=0`, every position
and vocabulary output, against the same pinned single-threaded CPU PyTorch.

### Third fixed wide case

| Seed | Hidden / HC | Attention heads / width | Index heads / width | Vocab | Q / O ranks | MoE intermediate | Top-k experts | Window | Max positions | YaRN original context |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4141073 | 128 / 4 | 4 / 128 | 4 / 128 | 97 | 128 / 64 | 128 | 3 | 64 | 768 | 32 |

The case uses six layers, two output groups, 64-wide rotary tails, YaRN factor
4 and two batches. It runs a 512-token odd prefill (128 candidate blocks at
ratio 1, 64 at ratio 2, and eight ring wraps), four decodes spanning completed
and incomplete compression groups, then a reset to a 129-token prefill (33
candidate blocks, one ring wrap) and two decodes. Ratio-1/2 compression, shared
KV/index layers, candidate selection, nonzero masks, Engram and all three
routed experts remain active. The checkpoint is a two-shard SafeTensors package
bound in the small model manifest; every config/index/payload file is verified
before native loading.

**125,518 / 125,518 new logits are bit-exact** across all eight steps. Two
consecutive native runs produced byte-identical trace exports. The complete
suite is now **161,394 logits, 76 steps and 12 fixed models**; the five complete
packed BF16 models account for **159,154 bit-exact logits**. The older FP32
cases keep their original fixtures and worst drift `1.7881393432617188e-7`.

### Candidate-block ties

The previous level-one selector ranked blocks pairwise and broke equal scores by
lower block id. That is not the pinned reference: `scores.topk` on CPU picks
different tie sets for `k=2` and `k>=3` because `k*64 <= blocks` switches from
heap staging to median partition. With 128 equal-maxima blocks (newest block
pinned), the released mask keeps block 0 for `k=2` but blocks 1 and 2 for `k=3`;
a lower-id rule cannot express either.

The index top-k kernel now has a raw mode that returns selected ids without
position mapping, and the candidate selector reuses it. The mask pass expands
only selected ids and drops a selected block whose score is `-inf`, matching the
reference's `top.values > -inf` scatter. Ten tied-block cases were added to the
frozen candidate oracle: all-equal maxima over 128 blocks at budgets
1/2/3/4/5/17/18/64, plus three-way boundary ties at budgets 2/3/4. The fixture
now has **43 cases and 20,624 exact mask values**; native execution matches every
one, reproduces on repetition, and the 68 tied index top-k cases still pass.

### Measured intermediate limits

Across the three wide cases the exported trace covers **52,415,792 intermediate
values**; **52,415,778 match by bits**. The 14 remaining single-BF16-ulp values
are 1 sparse-attention value (seed 4141071), 3 sparse + 7 grouped-output + 1
quantized attention-projection input (seed 4141072), and 2 grouped-output values
(seed 4141073). Shared caches match over **8,504,320 values**, index scores over
**107,772 entries**, and selections over **16,854 positions** by bits. None of
these differences reaches a logit.

The grouped `wo_a` einsum body was re-tested against the same trace. Applying
the projection kernel's MKL K-panel staging raised the differing grouped values
from 7 to 13 (seed 4141072) and from 2 to 21 (seed 4141073); switching to the
four-lane vector-dot body failed the complete-logit gate. The serial FMA body is
retained and the negative result is recorded in the kernel so it is not
"unified" again without fresh CPU evidence.

Verification for this continuation:

- **96 V4/V4.1 Rust tests pass**, zero failed/ignored.
- All **514 SPIR-V artifacts** reproduce byte-for-byte under the pinned glslang
  16.6.0 / Vulkan 1.1 manifest, including the changed index-top-k and candidate
  mask modules.
- Push-layout validation passes for **395 shader blocks / 109 host structs**;
  the top-k push block gained its raw word in all three descriptions.
- The candidate oracle regenerates without `--write-fixture` and the wider
  oracle regenerates all three cases and accepts every native logit.
- Python compilation, `git diff --check` and rustfmt for every V4.1 file pass.
  The remaining repository-wide rustfmt differences are pre-existing in
  unrelated RWKV/bin/vulkan files.
- Wider fixture SHA256:
  `8f468d860c162fb08cddd554cbf85d5b108092a37d326460fdcd75dec018274f`.

The 512-token prefill is 8x the previous measured context but still far from the
released 1,048,576-token window, and hidden/rank/expert/vocabulary dimensions
remain reduced. Production-width and million-token qualification, the 14
intermediate ulps, training, DSpark and vision are still open.

## Initial production-width text continuation (2026-10-23; diagnosis superseded below)

The 512-token case above fixes width at 128 hidden. This phase fixes the
released text *widths* and vocabulary while keeping depth, expert count and the
compression/index schedule reduced: 5120 hidden, 64x512 attention, Q/O ranks
1280/1024, 8 output groups, 2304-wide experts, HC 4 and the full 129,280-token
vocabulary in one sliding-only layer with two routed experts.

- The oracle (`validation/validate_deepseek_v41_production_model.py`) writes the
  17-tensor checkpoint directly to four sharded SafeTensors (~2.7 GB) instead of
  expanding 1.3B vocabulary values into JSON. Embedding and LM head stay BF16 on
  disk; native reads them as 16-bit words (`read_bf16_words`) and decodes them in
  `transformer_deepseek_v41_embedding_half.comp` and
  `transformer_deepseek_v41_linear_half.comp` with the FP32 accumulation order of
  the promoted path, so 16-bit storage is numerically identical to the qualified
  FP32 stream.
- Fixture: `validation/fixtures/deepseek_v41_production_model.json` (SHA256
  `5dd31f6d737bc674c8dbf7f6062b38a3e7f30597d6b03969e3b7b0bdd8df89c4`),
  checkpoint directory `deepseek_v41_production_4141090` (4 shards, 17
  tensors, 13 packed weights, 258,560 reference logits).
- `src/deepseek_v41/model_production_tests.rs` is opt-in
  (`cargo test --lib deepseek_v41_production -- --ignored --nocapture`). It
  verifies every shard hash, the released widths and vocabulary, and compares
  every logit at `2e-7`, rtol=0.

Initial result: **not qualified** (the later recovery below passes the same frozen fixture).

- `production-prefill` (position 0, 129,280 logits): drift **0.0**, bit-exact.
- `production-decode` (position 1, 129,280 logits): drift
  **3.10230255126953125e-3**, exactly one BF16 ulp.

The initial handoff attributed the failure to sparse-attention/MKL reduction
order. That attribution was incorrect: its saved native sparse dump came from
a two-token diagnostic prefill, while the saved CPU dump came from the restored
one-token production prefill. Those shapes choose different CPU contraction
orders. Fresh per-step traces below show the actual failing decode's sparse
weights and pre-cast output are already bit-exact. The grouped `wo_a` projection
is the first divergent operation. The earlier score-search results did not
establish an irreproducible CPU-oracle blocker, and the oracle has not changed.

Reproduction:

```
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_production_model.py \
  --reference-path .tmp-v41-ref/model.py \
  --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py --write-fixture
cd hierarchos-vulkan
cargo test --lib deepseek_v41_production -- --ignored --nocapture
```

## Production-width parity recovery and pinned grouped reduction (2026-10-23)

The original production fixture now passes **all 258,560 logits with zero drift**:
129,280 prefill logits and 129,280 decode logits. Its SHA256 remains
`5dd31f6d737bc674c8dbf7f6062b38a3e7f30597d6b03969e3b7b0bdd8df89c4`.
No source forward body, CPU sparse mirror, model weight, seed, input, expected
logit or tolerance was changed. Acceptance is still strictly `<2e-7`, `rtol=0`.

### Corrected localization

`model_production_tests.rs` now captures sparse weights/raw outputs for **each
named step**, and exports every native logit with fixture/shard hash bindings
before enforcing the gate. The previous capture kept only the first call.

`validation/trace_deepseek_v41_production_attention.py` independently replays the
released attention from the frozen checkpoint. It reads only the used embedding
rows and attention parameters, derives the first layer's input with the released
HC collapse/RMSNorm, and records the original CPU `einsum`/softmax result for
both steps. It avoids regenerating or expanding the billion-value vocabulary
pair. The replay verifies every checkpoint hash and retains the unmodified
source/math contract.

On the actual failing decode, attention input, queries, all 8,192 softmax
weights and all 32,768 pre-cast sparse outputs are bit-exact. The first mismatch
was instead **three BF16 `attention_grouped` values**, at flattened positions
3781, 5444 and 8174. The grouped shader used the FP32 short-matrix four-lane dot
for this BF16 decode. Those rounding differences crossed the following packed
projection's activation boundary and propagated to the logits. The CPU dispatch
for this operation is oneDNN `brg_matmul:avx512_core_bf16`, not the attention
score's FP32 MKL contraction.

### Pinned native policy and regression

The one-column FP32 sparse contraction is also pinned to the CPU's four-lane
GEMV order above ATen's 400-product small-bmm threshold. Other measured shapes
retain the existing panel contraction. The decision uses the allocated matrix
shape: the production decode has 128 gathered columns even though only two are
visible. Counting only visible columns would select the wrong reduction.
`deepseek_v41_production_attention.json` freezes the original CPU sparse inputs,
weights and FP32/BF16 outputs for both production shapes. Its default native
test requires bit-exact weights, pre-cast outputs and BF16 outputs. The unchanged
CPU checkpoint replay regenerates this fixture; no native output is used in it.
Across both steps the default component test checks all **8,256 softmax weights,
65,536 raw FP32 outputs and 65,536 rounded BF16 outputs** by bits.

The grouped push constants now carry the weight precision. The native policy
`bf16_short_grouped_fp32_k512_v1` uses ascending 512-element FP32 FMA partials,
folds them in order, then rounds at the existing BF16 boundary for short BF16
grouped matrices. FP32 short matrices retain their four-lane order; the existing
matrix body is retained. This is an explicit native reduction contract with CPU
evidence at the qualified shapes, not a claim that all oneDNN implementations
use this internal schedule. All math and scratch remain on Vulkan.

`validate_deepseek_v41_grouped_reduction.py` independently freezes a compact
regression using CPU `torch.einsum` over the failing channels and neighbors,
keeping K=4096 and output rank 1024. It contains nine nonzero weight rows across
three groups; all other component output rows are zero. **All 6,144 component
outputs match by bits**. The default native test also exercises the former
four-lane path as a negative control and requires it to disagree, so the
regression detects the original bug. This compact component fixture does not
replace the complete-model test: the latter still loads every original dense
weight and checks all original vocabulary logits.

The final production trace has **438,472 / 438,472 stage values bit-exact**,
including grouped projections, HC coefficients, residuals, MoE and block outputs.
The complete suite now covers **419,954 logits, 78 steps and 13 models**, with
overall worst drift `1.7881393432617188e-7`; six BF16 models contribute 417,714
bit-exact logits.

Verification and reproduction:

- Default V4/V4.1 suite: **98 passed, 0 failed, 1 ignored**. The ignored test
  requires the local production package; its separate final run also passed,
  with zero drift at both steps.
- Library compilation, Python compilation, Rust formatting and whitespace
  checks pass. Existing unrelated compiler warnings remain.
- Shader provenance: **516 / 516 artifacts reproduce byte-for-byte**.
  Push-constant validation checks **397 shader blocks and 109 host structs**;
  all 28 provenance unit tests and both strict-parity policy tests pass.
- The independent checkpoint attention replay and compact grouped CPU oracle
  regenerate their frozen component fixtures without writing them. The final
  summary compares every complete-model logit, every production trace stage,
  and both sparse traces against the frozen CPU evidence.
- Fresh CPU regeneration of the original/additional FP32, mixed, attention,
  BF16 and all three wider model fixtures passes without writing checkpoints
  or expected logits. Every wider checkpoint byte reproduces, including the
  512-token prefill and 129-token reset case; all wider logits remain exact.

```powershell
# Default suite, including both reduction regressions and every tiny/wider gate.
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v4 -- --nocapture

# Original locally generated production-width package; expected fixture unchanged.
$env:V41_PRODUCTION_TRACE = '1'
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_production_width_vocabulary_strict_all_logits -- --ignored --nocapture
Remove-Item Env:V41_PRODUCTION_TRACE

# Independent observation, with no checkpoint or expected-logit writes.
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/trace_deepseek_v41_production_attention.py --reference-path .tmp-v41-ref/model.py --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py --output target/deepseek-v41-reference/production-attention-reference.json
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_grouped_reduction.py --observation target/deepseek-v41-reference/production-attention-reference.json
```

The width test remains opt-in because its 2.7 GB package is generated locally.
It qualifies released hidden/head/rank/intermediate widths and vocabulary for
this one-layer, two-routed-expert, two-step text graph. Released depth, 384 routed
experts, large Engram tables, combined production compression schedules, long
contexts, training, DSpark and vision still require qualification. The generic
loader remains disabled. CPU PyTorch is the accepted reference; CUDA parity is
not required.

Evidence lives in `target/deepseek-v41-reference/`: `production-pinned-final.log`,
`production-reduction-regression-final.log`, `production-pinned-sparse-component.log`,
`production-grouped-component.log`,
`production-native-logits.json`, `production-native-trace.json`,
`production-native-sparse.json`, `production-attention-reference-final.json`,
`production-attention-oracle-final.log`,
`production-reduction-qualification-summary.json`,
`production-pin-*-oracle.log`,
`production-reduction-shader-provenance.json` and
`production-reduction-library-check.log`.

## Released depth/schedule and 384-expert continuation (2026-10-23)

Two additional frozen CPU models now pass **6,432 complete logits with zero
drift**, including every prefill position, cached decode and reset. The fixture
`deepseek_v41_structure_model.json` remains SHA256
`4817e35531eb0d8c076f8a76a8a55a5bab36eafe8653d017eb006aa7c01e80cc`.
It was frozen before the first native run. The failing cases, weights, inputs,
expected logits and strict `<2e-7`, `rtol=0` comparator were retained throughout.

- Seed **4141101** runs all **40 backbone layers**, with the released
  `0,0,2x18,1x20` compression schedule, KV sources `2,8,14,20`, index sources
  `2,8,14,20,24,28,32,36`, candidate source 20 and Engram at layers 1 and 14.
  Engram uses the released four-gram/eight-head structure, with reduced
  8,192-row tables and head width 32. This probe has eight routed experts and
  top-6 routing at every layer.
- Seed **4141102** runs **384 routed experts and top-6 routing** in one
  sliding-only layer, with no Engram. It exercises expert IDs above 255 and
  preserves the full expert count, routing weights, per-expert counts and
  ascending-expert output accumulation.
- Both have batch size 2, hidden/intermediate widths 64, four 64-wide attention
  heads, Q/O ranks 32/32, two output groups, HC 4, vocabulary 67, window 8,
  context allocation 64 and a YaRN factor of 16 beyond original context 8.
  Each runs a 17-token prefill, three decodes across compression boundaries,
  then a three-token reset and another decode: six steps and 3,216 logits.

These are separate probes. They do **not** qualify 40 layers, 384 experts,
released widths/vocabulary and large Engram tables together.

### Numerical implementation

The first depth run had bit-exact prefill but decode drift up to
`2.45906710624694824e-2`. Matched traces located the first divergent decode
block output in layer 34's MoE; later cache differences were consequences.
The raw FP32 gate dots and expert selections were already exact. Three native
arithmetic contracts required attention:

1. Top-6 normalization now follows the CPU short-row sum order: four partial
   sums, scalar remainder into the first partial, then an ascending fold.
   Top-2/top-3 retain their previous arithmetic result.
2. Standalone CPU `sqrt` goes through the pinned MKL VML implementation. Its
   measured SSE seed/refinement is different from a correctly rounded square
   root. The native shader now uses a complete **8,192-entry instruction seed
   map** with exponent scaling and separately rounded FP32 refinement. The
   map covers all **16,777,216** FP32 mantissas in `[1,4)`; every instruction
   result is checked against its 2,048-value bucket. It contains no model
   values, weights, logits or adjustments to expected results.
3. Scalar CPU `log1pf` rounds `1+x`, evaluates scalar `logf`, then subtracts
   its rounded cancellation correction. A remaining one-ulp routing weight in
   layer 39 exposed the difference between the CRT's `logf` polynomial and
   correctly rounded real log. The native path now uses the measured FMA
   staging. Its 129 center records are generated analytically from
   `log(1+i/128)`, split into truncated high/low terms, plus rounded
   reciprocals. No vendor DLL bytes are shipped.

All activation math, instruction-map lookup, refinement, routing and expert
execution run on Vulkan. CPU code only supplies immutable development
expectations and loads the frozen numeric data, like checkpoint bytes.
`DeepseekV41Moe` and `DeepseekV41MixedMoe` both bind the seed buffer; no runtime
Python, Torch or CPU activation fallback was introduced.

This policy is explicitly pinned to the measured Windows/AMD Zen 4 CPU
reference, PyTorch `2.14.1+cpu` and its MKL runtime. Different CPU instruction
seeds, CRT/MKL versions or math dispatch require fresh qualification. The
measured `ucrtbase.dll` SHA256 is
`5c52e3a303baaac0e0af8bd9b96134993da34bc9d834a31ef37e1d2cdc7fe192`.
Relevant primary implementation contracts are PyTorch's
[SumKernel](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/SumKernel.cpp),
[unary dispatch](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp)
and [VML selection](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/cpu/vml.h).

### Permanent validation

- The default structure test compares all logits and every layer's routing
  weights, indices and expert counts by bits. Optional tracing captures every
  stage, shared compressed KV/index keys and index selection across all layers.
- Nine compact routing cases independently check raw FP32 gate dots, selected
  unnormalized scores, normalized weights, expert IDs and count totals.
- Separate CPU math fixtures check **4,104** positive square roots and
  **8,966** scalar `logf`/`log1pf` values by bits. Existing unary fixtures and
  full-model expectations are unchanged.
- An old routing component guard required a nonzero *error*, so exact results
  made it fail. That guard now requires nonempty comparisons and nonzero
  routing weights; its existing expected values and numerical gate are intact.
- All **102 default V4/V4.1 tests pass**. Shader provenance reproduces all
  **516 SPIR-V artifacts byte-for-byte**. The new seed buffer changes descriptor
  bindings, not push constants.
- All **4,635,840 structure stage values**, **891,648 shared cache values**,
  **20,444 index scores** and **6,780 selected indices** match by bits. Shared
  consumers are included in these cache/index totals. The complete structure
  routing trace has **11,808 exact slots**, including weights and expert IDs.
- The separate production-width regression still passes every one of its
  **258,560 original logits with zero drift**, with its original fixture hash.
  All 438,472 production stage values and both sparse traces remain bit-exact.
  Library compilation passes with the existing unrelated warnings.
- Aggregate coverage is **426,386 logits, 15 models, 90 steps**; worst drift
  remains `1.7881393432617188e-7`. Eight packed BF16 models contribute 424,146
  bit-exact logits. The 397 shader push blocks/109 host structs, both strict
  policy tests and all 28 provenance unit tests pass.

Reproduction from the repository root:

```powershell
# Rebuild local shards against frozen hashes, without rewriting expected logits.
# Omit --write-checkpoints for a read-only reproduction of existing payloads.
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_structure_model.py --reference-path .tmp-v41-ref/model.py --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py --write-checkpoints --trace-output target/deepseek-v41-reference/structure-reference-trace.json

# Instruction observer is development-only; the product embeds the frozen map.
rustc --crate-type cdylib -O --edition 2021 hierarchos-vulkan/validation/deepseek_v41_cpu_seed_probe.rs -o target/deepseek-v41-reference/rsqrt_seed_probe.dll
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_cpu_math.py --seed-library target/deepseek-v41-reference/rsqrt_seed_probe.dll
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_cpu_log.py
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_routing_reduction.py --trace target/deepseek-v41-reference/structure-reference-trace.json

$env:HIERARCHOS_V41_STRUCTURE_TRACE = '1'
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v4 -- --nocapture
Remove-Item Env:HIERARCHOS_V41_STRUCTURE_TRACE

# After generating all native exports, recheck every complete logit and trace.
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/summarize_deepseek_v41_qualification.py --structure-reference-trace target/deepseek-v41-reference/structure-reference-trace.json --output target/deepseek-v41-reference/structure-qualification-summary.json
```

Evidence is under `target/deepseek-v41-reference/`: the structure CPU reference
and native stage exports, `structure-regression-qualified.log`,
`structure-routing-cpu-math.log`, `structure-cpu-log-native.log`,
`structure-oracle-reproduction-final.log`, `structure-reference-final.log`,
`structure-checkpoint-rebuild-final.log`, `structure-production-regression.log`,
`structure-qualification-summary.json`, `structure-library-check.log` and
`structure-shader-provenance-qualified.json`.

## Joint released-depth and routing-cardinality continuation (2026-10-23)

The released **40-layer schedule and 384 routed experts with top-6 selection now
run together in one packed BF16 graph**. This closes the separation between the
two structural probes above. It does not qualify the combined released widths,
vocabulary, context or Engram table sizes.

The new fixed seed is `4141111`. Hidden and expert intermediate widths are 32;
Q/O ranks are 32, four attention heads are 64-wide, the indexer has four
128-wide heads, HC has four streams, and vocabulary is 67. The graph retains
the released text compression ratios, KV/index ownership and sharing, candidate
source layer 20, Engram layers 1/14, four-gram lookback and eight Engram heads.
The Engram tables have 8,192 rows. Batch size is two, sliding window is eight,
and the unchanged six-step structural schedule is prefill 17, three decodes,
reset prefill three, then decode at position three. YaRN crosses the reduced
original context of eight. All 46,372 packed matrices are loaded from two real
SafeTensors shards through the native checkpoint API.

`validation/fixtures/deepseek_v41_joint_model.json` was frozen **before the first
native comparison**, with SHA256
`00cef75fe55c2243a6c2134c65e4c0adf224ce76791022c97f75738d4a1608b8`.
The new generator reuses the pinned source, weight initialization and CPU
kernel mirrors without changing their arithmetic. It refuses to overwrite an
existing fixture. Optional checkpoint reconstruction verifies every frozen
payload hash before writing any shard. A fresh read-only CPU replay reproduced
the entire fixture and package; its complete stage trace also reproduced
byte-for-byte (SHA256
`3a5305da1456b036379cc8fbc889bc5993927f1859f7f68932a927aa6ef7bdc7`).
The earlier two structural fixtures also reproduced unchanged after the factory
was generalized to accept explicit hidden/intermediate widths.

The native MoE now allocates its compaction input, route-weight and row-map
scratch once per call and reuses those three buffers across experts. Every
selected row and live prefix is overwritten; quantization and projections are
row-local, SwiGLU masks rows using GPU counts, and scatter reads selected rows.
No route data or activation is read back by production execution. This avoids
1,149 repeated scratch allocations per 384-expert call. Arithmetic and expert
execution order are unchanged. The joint case exercises 86,607 zero-count
expert/step entries, growing and shrinking compact prefixes, and expert ID 383.

The dedicated opt-in joint test passed in 498.10 seconds on the local Vulkan
device. Every one of the **3,216 complete logits and 11,520 routing slots is
bit-exact** across all six steps. Independent comparison to the reproduced CPU
trace also passed:

- **3,156,480 / 3,156,480** intermediate values bit-exact;
- **891,648** compressed KV/index-cache words bit-exact;
- **20,444** index score words and **6,780** selected indices bit-exact,
  including shared consumers;
- every expert index, route-weight bit pattern and all 92,160 expert counts
  agree, including unused experts.

The acceptance rule remains strictly **absolute logit drift `<2e-7`, `rtol=0`**
for every output position and vocabulary value. No old expectation, input,
weight, seed, oracle expression or tolerance was changed. A fast default test
pins the joint fixture hash; the larger resident-graph test is opt-in so the
normal suite stays practical. CPU PyTorch remains the accepted reference and
CUDA parity is not required.

Final regression and aggregate verification:

- **103 default V4/V4.1 tests pass**, zero failures; two resource-heavy tests
  remain opt-in. The joint fixture hash guard is included in the default suite.
- The production-width/vocabulary test was rerun after scratch reuse and
  passed in 197.80 seconds: **258,560 original logits and 438,472 intermediate
  values remain bit-exact**. Its frozen weights and expectations were unchanged.
- The aggregate verifier passes **16 models, 96 steps and 429,602 complete
  logits**, worst drift **`1.7881393432617188e-7`**, `rtol=0`. Nine packed BF16
  models contribute **427,362 bit-exact logits**. All **23,328** structural and
  joint routing slots agree by bits. The two earlier structural stage traces
  remain fully bit-exact after scratch reuse.
- Library check, Python compilation, Rust formatting, strict-gate boundary
  tests and diff whitespace checks pass. **516/516 SPIR-V artifacts** reproduce
  byte-for-byte; the current pack shader comment was separately rebuilt and
  validated with `spirv-val`. All **397 shader / 109 host** push layouts agree.

Evidence is under `target/deepseek-v41-reference/`: `joint-freeze.log`,
`joint-cpu-reproduction.log`, `joint-old-structure-reproduction.log`,
`joint-native.log`, `joint-diagnostics.json`, `joint-regression.log`,
`joint-production-regression.log`, `joint-production-stages.log`,
`joint-shader-provenance.json`, `joint-push-layouts.log`,
`joint-library-check.log` and `joint-qualification-summary.json`.

Reproduction commands, from the repository root:

```powershell
# Read-only replay; --write-fixture is only for the initial pre-native freeze.
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_joint_model.py --reference-path .tmp-v41-ref/model.py --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py --trace-output target/deepseek-v41-reference/joint-reference-reproduced-trace.json

$env:HIERARCHOS_V41_JOINT_TRACE='1'
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_joint_depth_and_384_expert_strict_all_logits -- --ignored --nocapture

.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/summarize_deepseek_v41_qualification.py --structure-reference-trace target/deepseek-v41-reference/structure-reference-reproduced-trace.json --joint-reference-trace target/deepseek-v41-reference/joint-reference-reproduced-trace.json --output target/deepseek-v41-reference/joint-qualification-summary.json
```

## Long-context joint continuation (2026-10-23)

The joint **40-layer / 384-expert / top-6** graph now runs a **512-token YaRN
prefill**, four decodes across both compression ratios, and an eight-token
reset followed by two decodes. This is the context axis called out by the
previous phase: 17 tokens in this graph, and only six layers in the earlier
512-token probe. Hidden and expert intermediate widths remain 32, Q/O ranks 32,
attention stays four 64-wide heads, vocabulary is 67 and Engram tables have
8,192 rows. The sliding window is the released **128**. The indexer keeps the
released **32 heads of width 128** but selects 16 of the 258/516 compressed
rows; candidates select 8 of 64 blocks of 8. The reduced original context is
128 tokens, so YaRN extends it by 4x; at position 515 attention gathers 144
keys (128 window + 16 compressed). Every reduction is declared before native
measurement. Prefill is one 512-token chunk; continued prefill at a nonzero
position with more than one token (chunked serving) is not exercised here and
remains unqualified.

The fixed seed is `4141112`. Batch size is one, `max_prefill` is 512 and the
first step is a single 512-token prefill; decodes at positions 512-515 complete
one ratio-2 group, leave one incomplete, and complete another. The released
compression ratios, KV/index ownership, candidate source layer 20 and Engram
layers 1/14 are unchanged. All **46,372** packed matrices load through the
native checkpoint API from two SafeTensors shards.

`validation/fixtures/deepseek_v41_long_context_model.json` was frozen before
the first native comparison, with SHA256
`b04f96cdfe53603420cee2e18fefccf1f0deaf684968a3ca000f18e9a8f3a671`. The
generator refuses to overwrite an existing fixture and verifies every shard
hash before writing. A read-only CPU replay reproduced the complete fixture and
package, and the regenerated deep-decode trace is the comparison reference.

`validation/validate_deepseek_v41_long_context_model.py` compares the frozen
logits and routing, plus one bounded stage trace at the deepest decode. Full
stage observation for the 512-row prefill would be roughly 34 million values,
so it is deliberately excluded; all logits and all routing still cover every
step. The opt-in native test passed in **491.72 seconds**:

- **35,242 / 35,242 complete logits bit-exact**, including the 34,304-logit
  512-token prefill and the reset sequence;
- **126,240 / 126,240 routing slots** (indices, weight bits and expert counts)
  bit-exact across all eight steps;
- **65,760 / 65,760 deep-decode stage values bit-exact**;
- **2,873,088** shared cache words, **3,354** index score words and **128**
  selected indices bit-exact, including shared consumers.

The earlier 512-token case fixed width at 128 hidden and six layers; the
production-width case fixed one layer with two experts and released
vocabulary. None of those combinations is the joint graph. This phase closes
the long-context combination at reduced widths without changing any frozen
expectation, weight, input, seed, oracle expression or tolerance.

### Measured full-size ceiling

The remaining released-width/table composition is a residency limit on this
host (10.7 GB RAM, 4 GB integrated Vulkan device), not an unexplained numeric
gap:

- **40 layers x (384 routed + 1 shared) experts x 3 matrices x 5120 x 2304**
  at BF16 is **1.09 TB resident**, before roughly 13 GB of attention and
  index projections.
- The released Engram tables have 384,006,168 and 384,016,682 rows of 32
  values. Even at one byte per value (their stored FP8 form) that is 24.6 GB
  for both; the native lookup dequantizes rows to BF16, about 49 GB.
- A 1,048,576-token context adds roughly 43 GB of compressed KV/index cache at
  the released ratios, before the sliding window.

The production-width one-layer probe already exercises the released arithmetic
and vocabulary head, and the 384-expert probe exercises full routing
cardinality; the missing full-size composition therefore needs streamed or
host-mapped experts/tables, or a host with more than 1.1 TB of addressable
weights. A larger fixture cannot close it on this machine.

## True 2,048-token prefill at the released indexer width (2026-10-24)

The joint **40-layer / 384-expert / top-6** graph now runs a **single 2,048-token
start-zero prefill** -- 16x the reduced 128-token YaRN original context and four
times the previous 512-token probe -- and every logit in the resulting schedule
is requalified against the unchanged CPU PyTorch oracle. The released 128-slot
sliding window, 32-head/128-wide indexer, compression ratios, KV/index
ownership, candidate source layer 20 and Engram layers 1/14 are unchanged.
Hidden/expert widths 32, four 64-wide heads, vocabulary 67, Q/O ranks 32 and
8,192-row Engram tables remain reduced; the combined released-width graph is
bounded by local memory, not by numerics (see the measured ceiling above).

### What the probe needed

The index query rotary applies per-head rows: `2,048 x 32 = 65,536`, one past
the 65,535-workgroup dispatch axis. Per-head-row kernels now tile rows on the
grid's z axis (`row = y + z * NumWorkGroups.y`): the rotary application and the
activation quantizer both do, and `validate_runtime` bounds prefill rows rather
than `rows x heads`. A new unit test
(`deepseek_v41_rotary_tiles_rows_past_one_dispatch_axis`) applies 65,538 rows in
one call and checks rows straddling the axis, including the first tiled row,
against single-row applications bit for bit.

The prompt stays whole by design. The pinned reference seeds its sliding-window
ring only at `start_pos == 0` and its decode branch consumes one token, so a
prompt chunk after the first has no reference semantics to compare against; a
split prompt is therefore not a qualification path.

### Gate tie at the top-k boundary

The first native run of the true 2,048-token prefill diverged by `5.19e-2`.
Row-sliced stage comparison against a row-sliced reference probe localized it:
every stage of row 1,019 was identical through layer 27's FFN input, and the
**MoE output** of layer 27 was the first difference. Two experts tie exactly on
`score + bias` at the 6th/7th boundary there
(`1.0706253051757812`, experts 284 and 351). The reference's CPU `topk` staging
selects 351; the gate's explicit lowest-id rule selected 284, so a different
expert entered the routing weights. Because layer 27 is late in each block the
difference then spread forward through the sliding window (+127 rows per
subsequent layer: routing differs at rows 1019-1146 in layer 28, 1019-1270 in
layer 29, ... and 1019-2047 from layer 36 on), which is why 1,029 of the 2,048
logits moved.

The gate now mirrors the pinned CPU `topk` staging instead of a local tie rule:
`TopKImpl.h`'s `k*64 <= n` partial_sort path (Microsoft STL comparisons, moves
and `sort_heap`) with the nth_element/partition path otherwise, the same
machinery the index top-k already uses. A CPU emulation of the compiled staging
reproduces `torch.topk` on 300 random 384-way vectors with injected ties and on
this row's exact biased scores. The rebuilt artifact is recorded under the
pinned glslang 16.6.0 in the provenance manifest.

### Frozen expectations

Seed `4141112` and every weight tensor are unchanged. The generator refreshes
the schedule only after verifying both weight shards and the index against their
frozen SHA256 values; it permits only declared `config.json` metadata to change
and rejects every other byte difference. The fixture is 11,801,647 bytes with
SHA256
`82dcb51beea92833fa8dc382daf30846cd195f5974a17e9eaf0f9c7dd314bbbc`; the shards
remain `325ec4568cc33cce3b952e6dd705b270fb326dae2a29b50fca53d803e634fbe6`,
`1627245a898fed6e8f41ed2ba052df67b3d35094515d856914b3add42127b526` and
`d820b7f0e9696c00ed2b469b548319524c1da766a70bd1d25c21d4cfdde46ac2`.
`max_prefill` is **2,048** and `max_positions` 2,064.

The seven-step schedule is one 2,048-token prefill (positions 0-2,047), decodes
at 2,048-2,050 (a ratio-1 group completes every step, ratio-2 groups at even
ends), an eight-token reset and two reset decodes. The deep trace is the first
decode, so its stage values, shared caches and index scores are the completed
2,049-position state, with 2,049 compressed rows on ratio-1 layers, 1,024 on
ratio-2 layers and 256 candidate blocks of 8. All inputs, masks, token maps and
initialization were fixed before the native measurement.

### Result

The opt-in test passed in **626.24 seconds**:

- **138,087 / 138,087 complete logits bit-exact**: 137,216 in the 2,048-token
  prefill, 201 across three decodes, 536 in the reset and 134 in two reset
  decodes.
- **494,640 / 494,640 routing slots** agree on expert id, weight bits and
  counts, including the boundary tie in layer 27.
- **65,760 / 65,760 deep-decode stage values bit-exact** at the 2,049-position
  state.
- **11,407,104** shared cache words, **59,412** index score words and **608**
  selected indices bit-exact across all shared consumers. The eight owning
  indexers account for **13,317** scores and **128** selected indices. These
  counters and the position description were corrected during the streamed
  long-context phase below; the original comparisons already checked consumers.

An independent single-threaded CPU replay regenerated the fixture from the
pinned reference, matched it byte for byte, and reported `worst 0` with zero
nonexact stages.

### Responsible checks

- **84 default tests pass under the V4.1 module filter** (`cargo test --lib
  deepseek_v41`), with the 3 opt-in joint/long-context/production tests ignored;
  the rotary tiling regression is part of that default set. Rerun on the final
  tree: 84 passed, 3 ignored (`long-default-single-pass.log`).
- The joint 40-layer/384-expert graph was rerun after the gate change:
  **646.61 seconds**, 3,216 logits and 11,520 routing slots bit-exact.
- The production-width/vocabulary probe was rerun: **230.09 seconds**, 258,560
  logits and 438,472 intermediate values bit-exact.
- **516/516 SPIR-V artifacts** reproduce byte-for-byte under the pinned glslang
  16.6.0, including the five V4.1 artifacts rebuilt in this phase (`rotary`,
  `quantize`, `index_scores`, `index_topk`, `moe_gate`); **397 shader / 109
  host** push layouts agree.
- Aggregate verifier: **17 models, 103 steps and 567,689 complete logits**,
  worst drift **`1.7881393432617188e-7`**, `rtol=0`. Ten BF16 models contribute
  **565,449** bit-exact logits; **517,968** routing slots agree by expert id,
  weight bits and counts.

Evidence under `target/deepseek-v41-reference/`: `long-freeze-single-pass.log`,
`long-native-2048-trace.log`, `long-context-model-native-logits.json`,
`long-reference-trace.json`, `long-compare-single-pass.log`,
`long-regression-single-pass.log`, `long-joint-regression-single-pass.log`,
`long-production-regression-single-pass.log`, `long-qualification-summary.json`,
`long-default-single-pass.log` and `long-shader-provenance-single-pass.log` (the
last two rerun on the final tree, after the gate change).

Reproduction commands, from the repository root:

```powershell
# The refresh verifies both weight shards; read-only runs never rewrite the fixture.
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_long_context_model.py --reference-path .tmp-v41-ref/model.py --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py --refresh-fixture --trace-output target/deepseek-v41-reference/long-reference-trace.json

$env:HIERARCHOS_V41_LONG_TRACE='1'
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_long_context_2048_token_strict_all_logits -- --ignored --nocapture

.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_long_context_model.py --reference-path .tmp-v41-ref/model.py --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py --native-output target/deepseek-v41-reference/long-context-model-native-logits.json --trace-output target/deepseek-v41-reference/long-reference-trace.json

.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/summarize_deepseek_v41_qualification.py --structure-reference-trace target/deepseek-v41-reference/structure-reference-reproduced-trace.json --joint-reference-trace target/deepseek-v41-reference/joint-reference-trace.json --long-reference-trace target/deepseek-v41-reference/long-reference-trace.json --output target/deepseek-v41-reference/long-qualification-summary.json
```

The remaining context work is prompt chunks after a start-zero prefill, longer
contexts that cross more compression and index regimes, and the full-size
residency problem described above.

## Default MoE boundary-tie regression (2026-10-24)

The layer-27 boundary tie from the single-pass 2,048-token probe now has a
standalone default GPU regression:
`deepseek_v41_top6_boundary_tie_matches_frozen_cpu`. The production gate and
all previous model fixtures are unchanged.

`validate_deepseek_v41_gate_tie.py` observes the original CPU prompt through
layer 27, gate row 1,019, verifies that row's output against the frozen
long-context fixture, then stops before executing that layer's experts. Four
copies of the captured input preserve the matrix gate path (`rows >= 4`);
the unchanged pinned `Gate.forward` must reproduce every original selected
expert and normalized-weight bit before the small fixture can be frozen.
Checkpoint gate weights and bias are checked against all frozen shard hashes.
Normal read-only replay evaluates this small gate; `--capture-model-row`
additionally reconstructs the independent full-prompt prefix.

The new fixture `deepseek_v41_gate_tie.json` is **193,502 bytes**, SHA256
`29d5a3d7bab55bef6106fd2cc005435b61938620f203fe7df9461e07c1dc624e`.
It binds the parent fixture SHA256
`82dcb51beea92833fa8dc382daf30846cd195f5974a17e9eaf0f9c7dd314bbbc`,
source revision, checkpoint hashes and CPU PyTorch revision. The default test
pins its hash, and the generator rejects attempts to overwrite it.

The two biased scores for experts **284 and 351** are both exactly
**1.0706253051757812**, while their unbiased scores differ. The reference keeps
`[322, 336, 290, 372, 365, 351]`; the former ascending-ID tie rule keeps
`[322, 336, 290, 372, 365, 284]`.

The test checks all **1,536** matrix products, exact selected slots and raw
weights, exact normalized/scaled weights, and all **384 expert counts** in
both modes. Expert 351 must have count four and expert 284 count zero. This
comparison uses exact slot order for the isolated regression.

The production-shader test passed in **0.17 seconds**. An isolated GPU negative
control, containing the former tie rule and the same score arithmetic, failed
at precisely the boundary-expert assertion. The test source was restored
byte-for-byte afterward; no shipped shader source or SPIR-V was changed. Fast
read-only CPU replay and the fixture-overwrite guard also passed. All **118**
previous fixture and V4.1 shader hashes recorded at the start remain unchanged.

The final default V4/V4.1 run passed **106 tests, zero failures**, with the
three existing opt-in model tests ignored, in **40.91 seconds**. Both strict
logit-gate boundary tests passed. The read-only aggregate export check also
passed: **17 models, 103 steps, 567,689 logits**, worst drift
**`1.7881393432617188e-7`**, `rtol=0`; **565,449 BF16 logits** remain bit-exact.
The recorded structure, joint and deep-context traces still match by bits.
Rust formatting, Python compilation and diff whitespace checks passed.

Reproduction commands from the repository root:

```powershell
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_gate_tie.py --reference-path .tmp-v41-ref/model.py --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_top6_boundary_tie_matches_frozen_cpu -- --nocapture
```

Evidence is under `target/deepseek-v41-reference/`: `gate-tie-freeze.log`,
`gate-tie-cpu-replay.log`, `gate-tie-native.log`,
`gate-tie-negative-control.log`, `gate-tie-mutation-check.log` and
`gate-tie-existing-hashes.json`, plus `gate-tie-regression.log` and
`gate-tie-qualification-summary.json`.

This adds component regression coverage; it does not increase qualified model
or logit counts. The full-model rule remains strictly **absolute drift
`<2e-7`, `rtol=0`** against CPU PyTorch, with CUDA parity not required.

## Checkpoint-streamed routed experts (2026-10-24)

The experimental library now has an explicit
`DeepseekV41ExpertResidency::Streamed` policy. Routed FP4/FP8 expert weights
are read from the local checkpoint and uploaded **one expert at a time**.
Existing constructors continue to select `Resident`. Routing, compaction,
quantization, GEMMs, SwiGLU, accumulation and every precision boundary still
execute through the same Vulkan kernels. No Python, Torch or CPU activation
fallback is used by the runtime.

```rust
let model = DeepseekV41CausalLM::from_checkpoint_with_expert_residency(
    device,
    &mut checkpoint,
    runtime,
    engram_tokens,
    DeepseekV41WeightOutput::Bf16,
    DeepseekV41ExpertResidency::Streamed,
)?;
assert_eq!(model.resident_routed_expert_count(), 0);
```

The source shares immutable SafeTensors headers and the weight index through
`Arc`; it does not clone entire metadata maps for each layer or retain any
routed tensor payload. The source owns its reader snapshot, so dropping the
caller's reader does not invalidate the model. All routed tensor/scale dtypes,
shapes and byte ranges are checked at construction, including late/unselected
experts. Payload codes and finite materialization are validated when read.
The current triplet is uploaded, consumed and dropped before loading the next.
The checkpoint files must remain readable and unchanged for model lifetime.

This is a bounded staging baseline: it visits every expert in ascending order,
including zero-count experts, without transferring route data to the CPU.
Shared experts, gates, attention, vocabulary and Engram tables stay resident.
Dense routed experts are rejected by this policy. The generic loader remains
disabled; the full released geometry and serving performance are unqualified.

The eight streamed executions reuse existing frozen weights, inputs,
seeds and CPU expectations: two FP32-hybrid models, two complete packed BF16
models, the 40-layer/eight-expert schedule probe, the one-layer/384-expert probe,
the combined 40-layer/384-expert graph, and the separate probe at released widths
and full vocabulary. **269,488 complete logits across 44 steps match CPU PyTorch
by bits**, with all **23,328** recorded routing slots
matching expert ids, weight bits and counts. The combined streamed opt-in test
passed in 782.05 seconds, and the paired resident rerun in 621.09 seconds;
these runs overlapped and are qualification timings, not a speed comparison.

Initial live Vulkan tensor bytes, excluding allocator slack and driver
reservations:

| Frozen graph | Resident routed experts | Streamed persistent experts | Resident bytes | Streamed bytes |
| --- | ---: | ---: | ---: | ---: |
| 40 layers, 8 experts, width 64 | 320 | 0 | 10,745,548 | 8,656,588 |
| 1 layer, 384 experts, width 64 | 384 | 0 | 2,791,056 | 284,304 |
| 40 layers, 384 experts, width 32 | 15,360 | 0 | 33,784,540 | 8,717,020 |
| 1 layer, 2 experts, released widths/vocabulary | 2 | 0 | 2,952,388,584 | 2,914,787,304 |

These are constructor snapshots, not peak inference memory or reserved VRAM.
A separate allocator test runs three streamed MoE calls, drops their outputs
and test-only route traces, and verifies that live GPU bytes return to the same
input/gate/shared-expert baseline each time. Other tests verify shared metadata,
late missing-scale rejection, dense-storage rejection, and deferred NaN payload
errors after successful construction and release of the caller's reader.

`validate_deepseek_v41_streamed_models.py` checks complete logits with the
unchanged **`<2e-7`, `rtol=0`** CPU comparator, exact streamed/resident logits,
resident counts, memory snapshots, routing and optional complete stage traces.
All observed streamed stages match resident stages by bits. The two small
BF16 fixtures retain the resident implementation's existing raw FP32 sparse
attention differences from the CPU trace; these do not introduce any new drift
and their complete logits remain bit-exact. The structure and joint CPU stage,
cache and index traces remain fully bit-exact.

Both residency modes of the released-width/vocabulary probe were rerun
sequentially in one opt-in run (340.68 seconds total). Each passed all 258,560
original logits with zero drift and all 438,472 CPU stage values by bits. This
does not combine released widths with the released layer/expert counts.
The final default suite passed **112 tests, zero failures, five opt-in tests
ignored**, in 50.91 seconds. Library check, Rust formatting, Python compilation,
and both strict logit-gate boundary tests passed. The existing 2,048-token
context qualification was still a resident-mode result at this checkpoint;
the next section extends it to streamed execution.

The base export aggregate also passes unchanged: **17 distinct frozen models,
103 steps, 567,689 logits**, worst drift **`1.7881393432617188e-7`**, `rtol=0`.
The eight streamed executions reuse these models and are reported separately.
**516/516 SPIR-V artifacts** reproduce byte-for-byte, and **397 shader / 109
host** push layouts agree. Previously recorded acceptance-fixture and V4.1
shader hashes remain unchanged.

Reproduction commands from the repository root:

```powershell
$env:HIERARCHOS_V41_STRUCTURE_TRACE='1'
$env:HIERARCHOS_V41_STREAMED_TRACE='1'
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v4 -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_streamed_joint_strict_all_logits -- --ignored --nocapture
$env:V41_PRODUCTION_TRACE='1'
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib production_width -- --ignored --nocapture --test-threads=1

.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_streamed_models.py --include-joint --bf16-reference-trace target/deepseek-v41-reference/bf16-model-reference-trace.json --structure-reference-trace target/deepseek-v41-reference/structure-reference-reproduced-trace.json --joint-reference-trace target/deepseek-v41-reference/joint-reference-trace.json --output target/deepseek-v41-reference/streamed-joint-qualification.json

# Add the released-width execution and its independent CPU stage trace.
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_streamed_models.py --include-joint --include-production --bf16-reference-trace target/deepseek-v41-reference/bf16-model-reference-trace.json --structure-reference-trace target/deepseek-v41-reference/structure-reference-reproduced-trace.json --joint-reference-trace target/deepseek-v41-reference/joint-reference-trace.json --production-reference-trace target/deepseek-v41-reference/production-reference-trace.json --output target/deepseek-v41-reference/streamed-qualification-summary.json
```

Evidence is under `target/deepseek-v41-reference/`: `streamed-library-check.log`,
`streamed-targeted-final.log`, `streamed-default-regression.log`,
`streamed-joint-native.log`, `streamed-resident-joint-regression.log`,
`streamed-default-qualification.json`, `streamed-joint-qualification.json` and
`streamed-parity-gate-tests.log`, `streamed-production-regression.log`,
`streamed-final-regression.log`, `streamed-final-library-check.log` and
`streamed-qualification-summary.json`, plus
`streamed-base-qualification-summary.json`, `streamed-shader-provenance.json`
and `streamed-push-layouts.log`. No existing acceptance fixture or shader was
rewritten. CPU PyTorch remains the reference; CUDA parity is not required.

## Streamed 2,048-token context qualification (2026-10-24)

The existing checkpoint-streamed policy now has a separate opt-in run of the
frozen 40-layer/384-expert, top-6, HC4 graph with a **single start-zero
2,048-token prefill**. This phase adds the native test and complete export
validation; it does not change runtime operators or checkpoint staging. Hidden
and expert intermediate widths remain 32, with the released 128-token window,
32 index heads of width 128, and the released sharing/compression/Engram
schedule. This does not qualify chunked continued prefill or released widths
combined with released depth/cardinality.

Seed `4141112`, weights, inputs, expected logits and the CPU oracle are
unchanged. The frozen fixture still has SHA256
`82dcb51beea92833fa8dc382daf30846cd195f5974a17e9eaf0f9c7dd314bbbc`.
All **119** recorded acceptance-fixture and V4.1 shader hashes are unchanged;
no shader was rebuilt in this phase. CPU PyTorch remains the accepted
reference, with the same strict **`<2e-7`, `rtol=0`** gate on every logit.

Both native modes pass all **138,087 logits across seven steps with zero
drift**: the 2,048-token prefill, three cached decodes, eight-token reset and
two post-reset decodes. The streamed test passed in **956.31 seconds**, and
the paired resident rerun in **744.04 seconds**. The runs overlapped; these
are qualification timings, not a serving-performance comparison.

| Long-context mode | Persistent routed experts | Initial live Vulkan tensor bytes |
| --- | ---: | ---: |
| Resident | 15,360 | 66,223,772 |
| Streamed | 0 | 41,156,252 |

These are constructor snapshots, excluding allocator slack, driver reservations
and peak inference memory. Shared experts, attention, vocabulary and Engram
tables remain resident. Full released geometry remains unqualified.

The streamed validator now accepts `--include-long-context` and requires its
independent frozen CPU trace through `--long-context-reference-trace`. It checks
all seven complete logit tensors against CPU, exact resident/streamed logits,
every routing record, zero persistent routed experts, and the bounded deep
trace against CPU and resident execution. Missing trace input is rejected before
validation starts. No fixture-generation path is invoked by this validator.

The deep trace is the first decode, starting at position 2,048 and ending with
2,049 positions in state. The diagnostic report now distinguishes the eight
owning indexers from all 38 consumers. The former report labelled owner-only
totals as including shared consumers even though its comparison loop already
verified every consumer; this phase corrects those counters without changing
the comparisons. Four diagnostic unit tests cover inclusive totals, corrupted
consumer scores, corrupted consumer positions after offset normalization, and
missing consumers. They use synthetic diagnostic data, not replacement model
expectations. Both existing strict logit-gate boundary tests also pass.

The export validator passes **nine existing models, 51 steps and 407,575
streamed logits**, all bit-exact to CPU and resident execution, with **517,968
routing slots** agreeing by expert id, weight bits and counts. The long-context
execution contributes **138,087 logits** and **494,640 routing slots**. At its
first cached decode, streamed and resident execution both match the independent
CPU trace on **65,760 stage values**, **11,407,104 shared cache words**,
**59,412 index scores** and **608 selected index slots** across all consumers.
Owner-only totals remain separately reported as 13,317 scores and 128 indices.
No new distinct model or CPU expectation is added by the streamed execution.

The final default V4/V4.1 suite passes **112 tests, zero failures, six opt-in
tests ignored**, in 71.12 seconds. Rust formatting, Python compilation, all four
new trace tests and both strict logit-gate tests pass. The negative CLI check
rejects `--include-long-context` without its CPU trace. Fixture and shader hash
guards remain intact.

The refreshed base aggregate also passes: **17 distinct models, 103 steps and
567,689 logits**, worst drift **`1.7881393432617188e-7`**, with `rtol=0`.
Its ten BF16 models retain 565,449 bit-exact logits. Streamed executions reuse
these models and are counted separately above.

Reproduction commands from the repository root:

```powershell
$env:HIERARCHOS_V41_STREAMED_LONG_TRACE='1'
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_streamed_long_context_2048_strict_all_logits -- --ignored --nocapture
$env:HIERARCHOS_V41_LONG_TRACE='1'
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_long_context_2048_token_strict_all_logits -- --ignored --nocapture

$env:HIERARCHOS_V41_STRUCTURE_TRACE='1'
$env:HIERARCHOS_V41_STREAMED_TRACE='1'
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v4 -- --nocapture
.venv-vulkan/Scripts/python.exe -m unittest discover -s hierarchos-vulkan/validation -p test_deepseek_v41_long_trace.py -v
.venv-vulkan/Scripts/python.exe -m unittest discover -s hierarchos-vulkan/validation -p test_deepseek_v41_parity.py -v

.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_streamed_models.py --include-joint --include-production --include-long-context --bf16-reference-trace target/deepseek-v41-reference/bf16-model-reference-trace.json --structure-reference-trace target/deepseek-v41-reference/structure-reference-reproduced-trace.json --joint-reference-trace target/deepseek-v41-reference/joint-reference-trace.json --production-reference-trace target/deepseek-v41-reference/production-reference-trace.json --long-context-reference-trace target/deepseek-v41-reference/long-reference-trace.json --output target/deepseek-v41-reference/streamed-long-qualification-summary.json
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/summarize_deepseek_v41_qualification.py --structure-reference-trace target/deepseek-v41-reference/structure-reference-reproduced-trace.json --joint-reference-trace target/deepseek-v41-reference/joint-reference-trace.json --long-reference-trace target/deepseek-v41-reference/long-reference-trace.json --output target/deepseek-v41-reference/streamed-long-base-summary.json
```

Evidence under `target/deepseek-v41-reference/`: `streamed-long-native.log`,
`streamed-long-resident-regression.log`,
`streamed-long-context-model-native-logits.json`,
`streamed-long-default-regression.log`, `streamed-long-qualification-summary.json`
and its `.log`, `streamed-long-base-summary.json` and its `.log`,
`streamed-long-trace-tests.log`, `streamed-long-parity-gate-tests.log`,
`streamed-long-original-hashes.json` and `streamed-long-integrity.json`.

## Shared-expert checkpoint staging (2026-10-24)

`DeepseekV41ExpertResidency::StreamedAllExperts` extends the explicit library
policy to shared experts. It stages each routed expert in the same ascending
order, releases the last routed triplet, then reads/uploads the shared expert's
three packed projections. The shared expert uses the original input and
precision boundaries, and the same Vulkan SwiGLU, projections and combination.
Only its activation output survives the staging scope. No expert payload or
GPU projection is cached between calls; routing and activation math stay on
Vulkan. Checkpoint files must remain readable and unchanged for model lifetime.

| Policy | Routed experts between calls | Shared experts between calls |
| --- | --- | --- |
| `Resident` (default) | Resident | Resident |
| `Streamed` | Checkpoint only | Resident |
| `StreamedAllExperts` | Checkpoint only | Checkpoint only |

Use the existing `from_checkpoint_with_expert_residency` constructor and select
the new variant. `resident_shared_expert_count()` is available on the LM,
backbone and mixed MoE, alongside the existing routed-expert count. All-expert
streaming reports zero for both persistent counts. Other model weights,
including gates, attention, vocabulary and Engram tables, remain resident.
This is a correctness/staging baseline; it still visits zero-count routed
experts without reading routing activations back to the CPU.

Every shared projection's packed dtype, shape, scale metadata and byte range
is checked during construction. As with routed staging, payload validation is
deferred until the actual read. Tests cover missing late-layer shared scales,
dense shared projections, a deferred NaN scale error after releasing the
caller's reader, and sequence invalidation after that error. The existing
allocation test now runs three calls under each streamed policy, drops outputs
and test route traces, and verifies that live buffer bytes return to the same
baseline on every call.

The export validator's `--all-experts` option reads separate `streamed-all-*`
artifacts, enforces zero persistent shared experts, and requires lower initial
live tensor bytes than routed-only staging as well as full residency. All
complete logits must pass the unchanged **`<2e-7`, `rtol=0`** CPU gate and
match both prior residency modes by bits. Existing stage/cache/index checks
remain enabled when their independent traces are supplied; a CPU trace is
mandatory for long-context validation. A negative control with a deliberately
nonzero shared-residency count is rejected without producing a passing report.

The default V4/V4.1 suite passes **116 tests, zero failures, nine opt-in tests
ignored** (76.54 seconds). The targeted streamed suite passes nine tests, with
six opt-ins ignored (62.06 seconds). The ordinary library build check, Rust
formatting, Python compilation and all six existing strict-gate/trace unit
tests also pass. The refreshed base aggregate remains **17 distinct models,
103 steps and 567,689 logits**, with worst drift
**`1.7881393432617188e-7`**, `rtol=0`; its ten BF16 models retain 565,449
bit-exact logits. Additional residency executions reuse these frozen models.

The released-width/full-vocabulary run passes **258,560 complete logits** with
zero CPU drift and **438,472 CPU stage values** by bits (167.87 seconds). Its
shared expert is now staged too, leaving zero persistent routed and shared
experts. Initial live Vulkan tensor bytes fall from **2,914,787,304** under
routed-only streaming to **2,879,363,304** under all-expert streaming; full
residency uses 2,952,388,584 bytes. These are constructor snapshots, excluding
allocator slack, driver reservations and peak inference memory. Released
depth/cardinality is still reduced in this width probe.

The all-expert validator passes **nine existing graphs, 51 steps and 407,575
complete logits**, all bit-exact to CPU PyTorch, resident execution and
routed-only streaming. All **517,968 recorded routing slots** agree by expert
id, weight bits and counts. No fixture weights, seeds, inputs, expected logits,
oracle code, shaders or tolerance were changed. All **119** recorded
acceptance-fixture/shader hashes and both pinned CPU oracle files remain
unchanged; no shader rebuild was needed in this phase.
The validator's original routed-only mode also passes all nine graphs, 51
steps and 407,575 logits with zero drift.

The joint all-expert run passes in **707.51 seconds**, including all 3,216
logits and 11,520 routing slots. Its **3,156,480 stage values** match CPU by
bits. The long-context run passes in **801.91 seconds**, including all 138,087
logits and 494,640 routing slots. Its first decode matches CPU on **65,760
stage values**, **11,407,104 shared cache words**, **59,412 index scores** and
**608 selected index slots** across all consumers. These native runs overlapped;
their timings are qualification records, not speed comparisons. Every traced
stage matches the resident mode by bits. The two tiny BF16 models retain their
existing raw FP32 sparse-attention differences from CPU, with identical
diagnostic agreement and bit-exact complete logits under all residency policies.

Initial live tensor bytes for the structural and released-width probes:

| Frozen graph | Resident | Routed-only streaming | Routed + shared streaming |
| --- | ---: | ---: | ---: |
| 40 layers, 8 experts, width 64 | 10,745,548 | 8,656,588 | 8,164,588 |
| 1 layer, 384 experts, width 64 | 2,791,056 | 284,304 | 272,004 |
| 40 layers, 384 experts, width 32 | 33,784,540 | 8,717,020 | 8,593,660 |
| 40 layers, 384 experts, width 32, 2,048-token prefill | 66,223,772 | 41,156,252 | 41,032,892 |
| 1 layer, 2 experts, released widths/vocabulary | 2,952,388,584 | 2,914,787,304 | 2,879,363,304 |

All nine all-expert executions retain zero routed and shared experts between
calls and have lower initial live tensor bytes than both earlier policies.
Full released geometry, the remaining resident weight classes, large Engram
tables, chunked continued prefill and million-token contexts remain unqualified.

Reproduction commands from the repository root (the same frozen packages and
independent CPU traces used by the earlier policies):

```powershell
$env:HIERARCHOS_V41_STRUCTURE_TRACE='1'
$env:HIERARCHOS_V41_STREAMED_TRACE='1'
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v4 -- --nocapture

$env:V41_PRODUCTION_TRACE='1'
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_streamed_all_production_width_strict_all_logits -- --ignored --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_streamed_all_joint_strict_all_logits -- --ignored --nocapture
$env:HIERARCHOS_V41_STREAMED_LONG_TRACE='1'
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_streamed_all_long_context_2048_strict_all_logits -- --ignored --nocapture

.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_streamed_models.py --all-experts --include-joint --include-production --include-long-context --bf16-reference-trace target/deepseek-v41-reference/bf16-model-reference-trace.json --structure-reference-trace target/deepseek-v41-reference/structure-reference-reproduced-trace.json --joint-reference-trace target/deepseek-v41-reference/joint-reference-trace.json --production-reference-trace target/deepseek-v41-reference/production-reference-trace.json --long-context-reference-trace target/deepseek-v41-reference/long-reference-trace.json --output target/deepseek-v41-reference/shared-streamed-qualification-summary.json
```

Evidence under `target/deepseek-v41-reference/`: `shared-streamed-targeted.log`,
`shared-streamed-library-check.log`, `shared-streamed-default-regression.log`,
`shared-streamed-python-tests.log`, `shared-streamed-negative-control.log`,
`shared-streamed-production.log`, `shared-streamed-joint.log`,
`shared-streamed-long.log`, `shared-streamed-default-summary.json`,
`shared-streamed-production-summary.json`,
`shared-streamed-qualification-summary.json`,
`shared-streamed-routed-only-summary.json`, `shared-streamed-base-summary.json`,
their associated validator logs, and `shared-streamed-integrity.json`.
The separate `streamed-all-*` exports preserve earlier policy artifacts.

## Checkpoint-staged Engram tables (2026-10-24)

`DeepseekV41EngramResidency::StreamedTables` is an independent table policy,
selected through the new
`DeepseekV41CausalLM::from_checkpoint_with_residency` constructor alongside the
existing expert policy. Construction reads each scheduled layer's WKV
projection and Q/K norms but no table payload. Every `DeepseekV41Engram::apply`
then reads and uploads its layer's table, dispatches the packed or dense
lookup, and drops the staging buffer once `submit` completes; only the lookup
output reaches the projection and gate. The two lookup kernels stay resident,
which is a few descriptors rather than table state.

| Policy | Routed experts | Shared experts | Engram tables |
| --- | --- | --- | --- |
| `Resident` (default) | Resident | Resident | Resident |
| `Streamed` | Checkpoint only | Resident | Resident |
| `StreamedAllExperts` | Checkpoint only | Checkpoint only | Resident |
| `StreamedAllExperts` + `StreamedTables` | Checkpoint only | Checkpoint only | Checkpoint only per lookup |

The checkpoint loaders are split so streaming never materializes a table beside
its layer: `load_engram_table` (packed), `load_dense_engram_table`,
`load_engram_projection` and `load_mixed_engram_projection`. The original
combined loaders compose these and remain the default contract. Checkpoint
clones share immutable shard/index metadata and opening state through `Arc`;
`DeepseekV41Engram::new_streamed`, `new_packed_streamed` and
`new_mixed_streamed` take one such clone. `resident_engram_table_count()` is
available on the LM, backbone and runtime. Keep checkpoint files readable and
unchanged for the model's lifetime.

Payload errors are deferred, not swallowed. A corrupt UE8M0 scale on a
scheduled table fails only when its layer runs, reports `NaN code 255`, and
leaves decode unavailable until a start-zero prefill, matching the expert
streaming contract. A removed `layers.4.engram.embed.scale` fails resident
construction immediately but streamed construction succeeds and fails when
layer 4 first looks up, which demonstrates that construction really skipped
the payload. The source owns its metadata after the caller's reader is
dropped. Three repeated lookups leave live tensor bytes constant, with one
fixed 1,536-byte recycler reservation that is smaller than a single lookup's
working set and does not grow.

The default V4/V4.1 suite passes **122 tests, zero failures, eleven opt-in
tests ignored** (49.76 seconds). Four new Engram-table tests (bit-exact parity,
two deferred payload/metadata cases, and staging release), the routine mixed/BF16
all-table case and the structure case run by default. The structure tests pass
**6,432 logits** across the 40-layer/8-expert and 1-layer/384-expert graphs with
zero drift. The joint 40-layer/384-expert graph passes **3,216 logits and
11,520 routing slots** by bits in **547.48 seconds**. The 2,048-token graph
passes **138,087 logits and 494,640 routing slots** by bits in **654.79
seconds**; its first decode also matches **65,760 stage values**,
**11,407,104 shared cache words**, **59,412 index score words including shared
consumers** (**13,317** owner scores) and **608 selected index slots**
(**128** owner indices).

The export validator's new `--streamed-tables` mode reads the
`streamed-tables-*` artifacts, requires zero persistent tables, and checks the
resident and streamed logits bit for bit. It reports **eight model executions,
49 steps and 149,015 complete logits** with worst drift **0**, all bit-exact to
CPU PyTorch, and **517,968 routing slots** agreeing by expert id, weight bits
and counts. Traces remain bit-exact: the joint **3,156,480** stage values and
the long-context values above. The routed-only/`StreamedAllExperts` validator
mode still passes its existing **407,575 logits across nine executions**.

Initial live Vulkan tensor bytes (constructor snapshots; allocator slack,
driver reservations and peak inference memory excluded):

| Frozen graph | Resident | All experts | All experts + streamed tables |
| --- | ---: | ---: | ---: |
| 6 layers, 2 Engram layers (4-wide tables) | 435,004 | 387,124 | 383,028 |
| 1 layer, 384 experts | 2,791,056 | 272,004 | 272,004 |
| 40 layers, 8 experts | 10,745,548 | 8,164,588 | 7,623,916 |
| 40 layers, 384 experts | 33,784,540 | 8,593,660 | 8,052,988 |
| 40 layers, 384 experts, 2,048-token prefill | 66,223,772 | 41,032,892 | 40,492,220 |

Two `8,192 x 32` packed tables release **540,672** bytes; the small graphs
release **4,096** and **8,448** bytes. The 1-layer graph has no Engram
schedule, so its streamed-table figure is unchanged. Attention, vocabulary,
gates and Engram projections remain resident. All **516** SPIR-V artifacts
reproduce byte-for-byte under pinned glslang 16.6.0, and **397 shader / 109
host** push-constant layouts agree; no shader was rebuilt in this phase. No
fixture weights, seeds, inputs, expected logits, oracle code, shaders or
tolerance changed.

Reproduction commands from the repository root:

```powershell
$env:HIERARCHOS_V41_STREAMED_TABLES_TRACE='1'
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_streamed_all_tables_depth -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_streamed_all_tables_joint -- --ignored --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_streamed_all_tables_long_context -- --ignored --nocapture

.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_streamed_models.py --streamed-tables --include-joint --include-long-context --bf16-reference-trace target/deepseek-v41-reference/bf16-model-reference-trace.json --structure-reference-trace target/deepseek-v41-reference/structure-reference-trace.json --joint-reference-trace target/deepseek-v41-reference/joint-reference-trace.json --long-context-reference-trace target/deepseek-v41-reference/long-reference-trace.json --output target/deepseek-v41-reference/engram-table-streamed-qualification-summary.json
```

Evidence under `target/deepseek-v41-reference/`:
`streamed-tables-structure-model-native-logits.json`,
`streamed-tables-joint-model-native-logits.json`,
`streamed-tables-long-context-model-native-logits.json`,
`streamed-tables-mixed-model-native-logits.json`,
`streamed-tables-bf16-model-native-logits.json`,
`streamed-tables-joint.log`, `streamed-tables-long.log`,
`engram-table-streamed-qualification-summary.json`,
`engram-tables-default-regression.log`,
`engram-tables-shader-provenance.json` and
`engram-tables-push-layouts.log`.

## Checkpoint-staged block projections (2026-10-24)

`DeepseekV41ProjectionResidency::Streamed` now carries a complete block. Layer
construction still reads hyper-connection matrices, both RMSNorm vectors and
the routed/shared MoE, but no attention, indexer or compressor payload. On
entering a layer the orchestrator uploads that layer's attention
(`wq_a/wq_b/wkv/wo_a/wo_b`), owned indexer (`wq_b/weights_proj/wk`) and owned
compressor (`wkv/wgate/norm`) projections together with the Q/K norms and
attention sink, runs the layer, and drops them when it returns; a failed layer
execution still releases them, and a failed upload leaves nothing behind. `resident_projection_layer_count()` reports zero between
calls and is asserted after construction and after every step of the streamed
runs.

This phase also fixed a latent bug in the streamed constructor path. The build
function received the layer loader as a lazy iterator; the resident branch
consumed it while collecting attention weights, but the streamed branch passed
it over, so hyper-connections, norms and MoE were never instantiated and the
first block panicked on an empty `hyper` list. The streamed branch now drains
the iterator and rejects any accidentally resident attention payload. That is
why earlier projections-only construction had never executed a block.

| Policy | Routed experts | Shared experts | Engram tables | Attention/indexer/compressor projections |
| --- | --- | --- | --- | --- |
| `Resident` (default) | Resident | Resident | Resident | Resident |
| `StreamedAllExperts` + `StreamedTables` | Checkpoint only | Checkpoint only | Checkpoint only | Resident |
| all of the above + `ProjectionResidency::Streamed` | Checkpoint only | Checkpoint only | Checkpoint only | Per layer execution |

Hyper-connection matrices, the two block RMSNorm vectors, gates, KV/index
caches, the vocabulary head/embedding and Engram WKV projections remain
resident; the streamed policy covers the attention, compressor and indexer
projections requested here.

The default V4.1 suite passes **106 tests, zero failures, fourteen opt-in tests
ignored** (75.55 seconds). The full library suite passes **919 tests, zero
failures, 23 ignored** when run serially (365.48 seconds). Two RWKV
dispatch-count tests are order-sensitive under parallel execution: they failed
in the parallel default harness, but both pass individually and the full
serial run is clean; neither touches V4.1 code. Five new default tests cover
projection parity, deferred payload errors and partial-upload release on the
packed fixture, the streamed block on the hybrid/BF16 fixtures, and the
released-depth structure graph. Three opt-in tests cover the joint,
2,048-token and released-width/vocabulary graphs.

| Frozen graph | Resident | All experts + tables | Plus streamed projections |
| --- | ---: | ---: | ---: |
| 6 layers, 2 Engram layers (mixed) | 435,004 | 383,028 | 274,244 |
| 6 layers, 2 Engram layers (BF16) | 450,304 / 590,864 | 393,976 / 534,536 | 284,508 / 425,068 |
| 1 layer, 384 experts | 2,791,056 | 272,004 | 220,384 |
| 40 layers, 8 experts | 10,745,548 | 7,623,916 | 5,170,892 |
| 40 layers, 384 experts | 33,784,540 | 8,052,988 | 5,866,204 |
| 40 layers, 384 experts, 2,048-token prefill | 66,223,772 | 40,492,220 | 37,358,364 |
| released width/vocabulary, 1 layer, 2 experts | 2,952,388,584 | 2,879,363,304 | 2,651,986,152 |

Initial live Vulkan tensor bytes are constructor snapshots and exclude
allocator slack, driver reservations and peak inference memory. The packed
projection test reports 402,424 → 292,956 bytes and 320 logits bit-exact. The
structure graph passes **6,432 logits** at zero drift. The joint
40-layer/384-expert graph passes **3,216 logits and 11,520 routing slots** by
bits in **667.79 seconds**. The 2,048-token graph passes **138,087 logits and
494,640 routing slots** by bits in **732.62 seconds**. The released-width,
full-vocabulary one-layer probe passes **258,560 logits** at zero drift in
**212.58 seconds**; with all experts and projections streamed it reports
2,651,986,152 initial live bytes and **438,472** bit-exact stage values.

The export validator's new `--streamed-projections` mode reads the
`streamed-block-*` artifacts and requires zero resident projections, shared
experts, routed experts and Engram tables in every execution. It reports
**nine model executions, 51 steps and 407,575 complete logits** with worst
drift **0**, all bit-exact to CPU PyTorch, and **517,968 routing slots**
agreeing by expert id, weight bits and counts. Streamed stages are bit-exact
against the resident exports: the production **438,472**, joint **3,156,480**
and long-context **65,760** stage values, **11,407,104** shared-cache words,
**59,412** index score words including shared consumers (**13,317** owner
scores) and **608** selected index slots (**128** owner indices). The BF16
fixture keeps its pre-existing trace-only `attention_sparse` rounding
(2 values, 1.49e-8) identical to the resident trace; every logit remains
bit-exact.

All **516** SPIR-V artifacts reproduce byte-for-byte under pinned glslang
16.6.0, and **397 shader / 109 host** push-constant layouts agree; no shader was
rebuilt in this phase. No fixture weights, seeds, inputs, expected logits,
oracle code, shaders or tolerance changed. Full released geometry is still
unqualified and the remaining resident state is the hyper-connection, norm,
gate, cache, vocabulary and Engram-projection set above.

Reproduction commands from the repository root:

```powershell
$env:HIERARCHOS_V41_STREAMED_BLOCK_TRACE='1'
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_streamed_all_tables_projections_structure -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_streamed_block_hybrid -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_streamed_block_joint -- --ignored --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_streamed_block_long_context -- --ignored --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_streamed_block_production_width -- --ignored --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib -- --test-threads=1

.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_streamed_models.py --streamed-projections --include-joint --include-production --include-long-context --bf16-reference-trace target/deepseek-v41-reference/bf16-model-reference-trace.json --structure-reference-trace target/deepseek-v41-reference/structure-reference-trace.json --joint-reference-trace target/deepseek-v41-reference/joint-reference-trace.json --production-reference-trace target/deepseek-v41-reference/production-reference-trace.json --long-context-reference-trace target/deepseek-v41-reference/long-reference-trace.json --output target/deepseek-v41-reference/block-streamed-qualification-summary.json
```

Evidence under `target/deepseek-v41-reference/`:
`streamed-block-mixed-model-native-logits.json`,
`streamed-block-bf16-model-native-logits.json`,
`streamed-block-structure-model-native-logits.json`,
`streamed-block-joint-model-native-logits.json`,
`streamed-block-long-context-model-native-logits.json`,
`streamed-block-production-native-logits.json`,
`streamed-block-production-native-trace.json`,
`streamed-block-joint-native.log`, `streamed-block-long-native.log`,
`streamed-block-production-native.log`,
`block-streamed-qualification-summary.json`,
`block-streamed-default-regression.log`,
`block-streamed-v41-regression.log`,
`block-streamed-shader-provenance.json` and
`block-streamed-push-layouts.log`.

## Checkpoint-staged embedding and vocabulary head (2026-10-24)

`DeepseekV41VocabularyResidency::Streamed` removes the embedding and head
payloads from persistent model storage. The additive
`from_checkpoint_with_vocabulary_residency` constructor accepts this policy
alongside the existing expert, Engram-table and block-projection policies.
Every earlier constructor keeps its existing signature and defaults vocabulary
to `Resident`. `resident_vocabulary_tensor_count()` reports two for the
resident policy and zero for streamed execution, excluding temporary staging.

Construction preflights both tensor headers, dimensions, storage dtypes and
byte ranges without reading their payloads. FP32/hybrid streams support the
same F32/BF16 storage as the old reader; BF16 streams require the same packed
BF16 vocabulary words as resident execution. The source owns a checkpoint
snapshot sharing immutable shard metadata, so it survives release of the
caller's reader. Checkpoint files must remain readable and unchanged.

Each embedding operation reads/uploads the whole embedding matrix, runs the
existing Vulkan lookup/HC expansion and precision cast, and releases its
weight before the backbone starts. The final projection then reads/uploads
the whole head, runs the existing FP32 or BF16-word Vulkan linear kernel, and
releases it before returning logits. No token lookup, activation computation
or matrix multiplication falls back to CPU or Torch. At most one vocabulary
weight is staged at a time; matrices are re-read per operation, with no row
paging or throughput qualification. A complete matrix still has to fit during
its operation. Norms, gates, Hyper-Connections, caches and Engram projections
remain resident; full released geometry remains unqualified.

Three new focused tests cover missing/incorrect head metadata, deferred
non-finite embedding/head errors for both FP32 and BF16, sequence invalidation,
and three repeated lookup/head operations under each precision. After each
operation drops its outputs, live GPU tensor bytes return to the original
baseline; component outputs match the resident path by bits. Complete-model
variants reuse the frozen hybrid/BF16, structural, joint, 2,048-token and
released-width/vocabulary fixtures without changing any expectation.

The validator's `--streamed-vocabulary` mode selects the existing expert,
table and projection staging policies too. It requires zero persistent
vocabulary tensors and verifies that constructor live bytes fall from the
previous `streamed-block-*` export by exactly two full vocabulary buffers,
including BF16 word padding. The same strict **`<2e-7`, `rtol=0`** comparator
still checks every logit against CPU; logits and traced stages are also
compared with resident execution. Negative controls reject a nonzero vocabulary
residency count, a four-byte memory-accounting discrepancy, conflicting policy
flags, and missing CPU evidence for long-context validation.

The targeted suite passes **five tests, zero failures, three opt-ins ignored**
(46.14 seconds). The serial full library suite passes **924 tests, zero
failures, 26 ignored** (403.43 seconds), including **111 default V4.1 tests**.
The ordinary library check, Rust formatting, Python compilation and all six
strict-gate/trace Python unit tests pass. The base aggregate remains **17
distinct models, 103 steps and 567,689 logits**, worst drift
**`1.7881393432617188e-7`**, `rtol=0`. Its ten BF16 models retain 565,449
bit-exact logits; additional residency executions reuse these same models.

All **124** recorded acceptance-fixture and V4.1 shader/helper hashes and both
pinned CPU oracle files are unchanged. No shader was rebuilt, and no weights,
seed, input, expected logit, oracle calculation or acceptance limit was changed.

The released-width/full-vocabulary native run passes all **258,560 logits**
with zero CPU drift and **438,472 CPU stage values** by bits. Its constructor
reports **4,331,752 live tensor bytes**,
down from **2,651,986,152** with block projections/experts/tables staged but
vocabulary resident: exactly **2,647,654,400 bytes** for the two BF16 matrices.
It retains zero vocabulary tensors, projection layers, routed experts and
shared experts between calls. These are constructor snapshots, excluding
allocator slack, driver reservations and inference peaks; a complete
**1,323,827,200-byte** BF16 matrix is still staged during each vocabulary
operation. Released width/vocabulary is qualified separately from released
depth/cardinality in this one-layer probe.

The final vocabulary-staged export validator passes **nine model executions,
51 steps and 407,575 logits**, all bit-exact to CPU, fully resident execution
and the earlier routed-only export. All **517,968 recorded routing slots**
agree by expert id, weight bits and counts. The joint graph contributes
**3,216 logits**, **11,520 routing slots** and **3,156,480 CPU stage values**
by bits. The long-context graph contributes **138,087 logits** and **494,640
routing slots**; its first cached decode matches CPU on **65,760 stage values**,
**11,407,104 shared cache words**, **59,412 index scores** and **608 selected
index slots** including all consumers (owner totals remain 13,317 and 128).
The existing tiny BF16 raw sparse-attention trace differences remain unchanged;
their full logits stay bit-exact. Every observed stage matches the resident
export by bits.

All three new large opt-in tests pass together in **1,173.82 seconds**, with
two active tests at a time. This is a qualification time, not a serving
benchmark. The validator's earlier `--streamed-projections` mode also passes
all nine executions, 51 steps and 407,575 logits at zero drift.

| Frozen graph | Block staging, vocabulary resident | Plus vocabulary staging | Exact vocabulary bytes released |
| --- | ---: | ---: | ---: |
| 40 layers, 8 experts, width 64 | 5,170,892 | 5,153,740 | 17,152 |
| 1 layer, 384 experts, width 64 | 220,384 | 203,232 | 17,152 |
| 40 layers, 384 experts, width 32 | 5,866,204 | 5,857,628 | 8,576 |
| 40 layers, 384 experts, width 32, 2,048-token prefill | 37,358,364 | 37,349,788 | 8,576 |
| Released widths/vocabulary, 1 layer, 2 experts | 2,651,986,152 | 4,331,752 | 2,647,654,400 |

These are initial live tensor bytes. All nine executions retain zero vocabulary
tensors between calls, with the expected byte delta against block staging.

Reproduction commands from the repository root:

```powershell
$env:HIERARCHOS_V41_STRUCTURE_TRACE='1'
$env:HIERARCHOS_V41_STREAMED_VOCABULARY_TRACE='1'
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib -- --test-threads=1

$env:V41_PRODUCTION_TRACE='1'
$env:HIERARCHOS_V41_STREAMED_VOCABULARY_LONG_TRACE='1'
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_streamed_vocabulary -- --ignored --nocapture --test-threads=2

.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_streamed_models.py --streamed-vocabulary --include-joint --include-production --include-long-context --bf16-reference-trace target/deepseek-v41-reference/bf16-model-reference-trace.json --structure-reference-trace target/deepseek-v41-reference/structure-reference-trace.json --joint-reference-trace target/deepseek-v41-reference/joint-reference-trace.json --production-reference-trace target/deepseek-v41-reference/production-reference-trace.json --long-context-reference-trace target/deepseek-v41-reference/long-reference-trace.json --output target/deepseek-v41-reference/vocabulary-streamed-qualification-summary.json
```

Evidence under `target/deepseek-v41-reference/`: `vocabulary-streamed-targeted.log`,
`vocabulary-streamed-library-check.log`, `vocabulary-streamed-full-serial.log`,
`vocabulary-streamed-opt-in.log`, `vocabulary-streamed-python-tests.log`,
`vocabulary-streamed-negative-control.log`,
`vocabulary-streamed-default-summary.json`, `vocabulary-streamed-production-summary.json`,
`vocabulary-streamed-qualification-summary.json`,
`vocabulary-streamed-block-regression-summary.json`,
`vocabulary-streamed-base-summary.json`, their corresponding logs, and
`vocabulary-streamed-integrity.json`. New `streamed-vocabulary-*` exports
preserve the earlier residency-mode artifacts.

## Remaining implementation sequence

1. ~~Extend dedicated checkpoint loading to attention and owned indexers.~~
   Implemented, including layer-at-a-time upload through the orchestrator.
   ~~Extend loading to Hyper-Connection matrices, routed/shared MoE, Engram,
   embeddings and head, and add an independent tiny complete-model reference.~~
   Implemented for F32/BF16 checkpoints with explicit Engram conversion
   artifacts; the three tiny FP32 models now pass the hard gate.
2. ~~Produce shared index keys and implement the fine indexer.~~ Implemented:
   key ownership, per-head query rotary/FP4, head-weighted rectified scores,
   candidate and causal masks, sorted top-k, and the candidate-source hand-off.
   Top-k uses CPU-compatible heap/partition staging with device scratch; this
   remains a correctness baseline, not a million-token serving implementation.
3. ~~Wire sliding attention, attention sink, inverse output RoPE, the grouped
   output projection, shifted hyper-connections, and MoE into the dedicated
   backbone.~~ Implemented. The attention orchestrator owns compressor → shared
   index keys → compressed KV finalize → index selection → attention, and
   `DeepseekV41Backbone` wraps it with the two hyper-connection stages, both
   norms, and the routed/shared MoE in the released order. Backbone checkpoint
   loading and the LM head are now implemented in the experimental FP32 API.
4. ~~Implement Engram normalization/hash state and GPU embedding/gated residual.~~
   Implemented: native layout/offsets, the persistent compressed-token cache with
   masked lookback, lookup, and the gated residual, injected before each scheduled
   block. Engram checkpoint loading accepts post-dequantization BF16 tables or
   retains FP8/UE8M0 packed tables, with matching converter artifacts.
5. Tiny FP32, hybrid and packed BF16 prefill and cached decode enforce **all-logit drift <
   `2e-7`, rtol=0**, including compression boundaries, shared layers, Engram,
   reset, multiple weight seeds and top-3 routing. Three wider models cover
   hidden widths 128/256, 512-wide heads, 65-token YaRN prefills and a 512-token
   prefill with a 129-token reset. The one-layer production-width/vocabulary
   fixture also passes. A joint reduced-width graph now passes released 40-layer
   schedules and 384-expert/top-6 cardinality together, plus a single
   2,048-token prefill, three decodes across compression groups and an
   eight-token reset.
   Released widths and table sizes in that graph are bounded by local memory,
   not by unexplained numerics; see the measured ceiling above. Two/four HC
   streams are covered.
6. Add full backward/AdamW, save/reload parity and PEFT qualification before
   advertising training support. Preserve the existing family's strict gates.
7. Packed readers, typed linear projections, packed routed/shared experts,
   packed Engram tables/WKV, and packed attention/indexer projections are
   implemented. Four hybrid and ten complete packed BF16 models pass the
   unchanged full-logit gate. Residual/HC, compressor, Engram, final collapse,
   norm and head-input boundaries are implemented. Representative head/index
   widths, larger reduced vocabularies, 128 candidate blocks with tied maxima,
   and 512/129-token prefills now pass. Production hidden/rank/intermediate
   widths and vocabulary pass in the one-layer fixture. Released depth/schedules
   and expert cardinality pass together in a reduced-width graph, which now also
   passes a single 2,048-token prefill with the released window and indexer heads.
   Both resident and checkpoint-streamed routed-expert modes pass this context
   probe with identical logits, routing and bounded deep-context traces.
   The bounded deep-context trace is fully bit-exact, so no new reduction risk
   appeared. The remaining joint step is released widths and Engram table sizes:
   about 1.09 TB of expert weights plus 24.6 GB of stored tables against 10.7 GB
   of host memory. Checkpoint staging is implemented for routed/shared experts,
   Engram tables, per-layer attention/indexer/compressor projections, embedding
   and head. Hyper-Connections, norms, gates and Engram projections still have
   resident weight payloads, alongside persistent cache state. Combined released
   geometry and released table sizes remain unqualified.
   CPU PyTorch is the accepted qualification reference. Then add DSpark and
   vision execution as explicit contracts with their own evidence.
   Activation quantization and the rotary/quantization ordering are implemented;
   weight dequantization and the measured packed BF16 model's mixed-precision
   GEMM policy are implemented. All-dense BF16 expert/Engram packages require a
   separate typed path; the BF16 API currently requires packed experts/tables.

The dedicated library executes complete FP32, hybrid and packed BF16 text
models and passes strict logit parity for the measured tiny and wider suite. The
default `from_hf_package` loader now selects the storage and residency policies
without flags; the published checkpoint's names and headers match those
contracts. Full-size released-geometry execution, training, DSpark and vision
support are not qualified.

## 2026-10-24 — Transformers/PyTorch detour (not the requested native target)

This section records work performed after the assistant misread the requested
implementation target. The user clarified on 2026-10-25 that Rust/Vulkan
Hierarchos Native support is required. These PyTorch results do not change the
earlier Rust/Vulkan results or enable Hierarchos's normal loader.

Implementation is in `C:\Users\User\transformers`, with normal AutoConfig,
AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText and AutoProcessor
registration. The published defaults construct all text, Engram, vision,
aligner and three DSpark stages. Native FP8/FP4 checkpoint storage is selected
automatically by the model-aware quantizer, including CPU loading. There are no
experimental enable flags, CUDA/TileLang dependency, checkpoint conversion
requirement, or `trust_remote_code` requirement.

The custom sources were pulled from the published repository at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. File hashes and MIT attribution are
preserved in the Transformers model package. Separate HTTP range reads inspected
the actual tensor headers of all 48 published shards: **96,085 names, physical
shapes and packed dtypes match**. Only 10,684,928 header bytes were retained; the
510,286,023,000-byte weight payload was not downloaded or executed.

The complete existing frozen CPU corpus passes through ordinary indexed,
sharded `AutoModelForCausalLM.from_pretrained` loading: **17 cases, 567,689 full
logits, worst absolute drift 0.0**. This includes the 40-layer schedule,
384-expert routing, production vocabulary/head geometry and long-context cases.
The gate remains strict `<2e-7`, `rtol=0`, finite, nonempty, exact shape. Existing
oracle sources and original fixture files were not changed.

A new nonzero fixture executes the unchanged pinned vision/aligner and DSpark
forward bodies with the established independent CPU operators. Image-conditioned
prefill/decode logits (112), all three DSpark stages plus Markov logits (80),
confidence values (5), aligned image vectors and DSpark main inputs all have zero
drift. The fixture is included in the normal test suite.

**57 tests and 2 subtests pass** without environment enable flags, covering
frozen full logits, packed serialization, cache interleaving, continuation chunks,
beam reordering, left-padded generation, images, DSpark, tokenizer normalization,
prompt/tool encoding and processor round trips. The published tokenizer was
loaded independently and produces exactly 99,092 compressed IDs from 129,280
token IDs. Ruff, scoped modeling structure checks, and the official type checker
for the quantizer changes pass. Direct model type checking also passes with the
PyTorch mutable `Module.forward` override diagnostic excluded. The repo-wide
docstring checker hits its existing Windows `/src/` path-splitting error; it is
not recorded as passing.

Standard HF generation is autoregressive. DSpark's complete draft, Markov and
confidence computation is exposed by `forward_spec`, which returns its own
cache for a speculative scheduler. Compressed-cache cropping requires retained
prefix replay. GPU execution, distributed/offloaded execution, full-checkpoint
numerical parity, and full-checkpoint performance remain unqualified.

Reviewable sources, usage and preserved evidence:

- `C:\Users\User\transformers\docs\source\en\model_doc\deepseek_v41.md`
- `C:\Users\User\transformers\tests\models\deepseek_v41\fixtures\qualification.json`
- `C:\Users\User\transformers\src\transformers\models\deepseek_v41\upstream-source-manifest.json`
- `C:\Users\User\transformers\src\transformers\models\deepseek_v41\published-checkpoint-header-manifest.json`
- `target/deepseek-v41-reference/pytorch-all-17-parity.log`
- `target/deepseek-v41-reference/pytorch-multimodal-dspark-parity.log`
- `target/deepseek-v41-reference/pytorch-published-headers.log`
- `target/deepseek-v41-reference/pytorch-pytest.log`

## Native default package loading (2026-10-25)

This section returns to the requested target: Rust/Vulkan Hierarchos Native. The
PyTorch work above remains a comparison oracle and is not part of native
loading.

### Normal path

`DeepseekV41CausalLM::from_hf_package(device, directory, batch_size,
max_positions)` is the package path with no execution-policy flag:

- `embed.weight` storage selects BF16 versus F32 execution.
- Routed-expert storage selects `StreamedAllExperts` for packed I8/F4/FP8 and
  resident loading only for genuinely dense packages.
- Routed/shared experts, Engram table rows, attention/indexer/compressor
  projections and the vocabulary are staged from checkpoint files. No routed
  or shared expert, Engram table, block projection or vocabulary tensor is
  retained between calls.
- Engram's non-persistent token map and hash multipliers come from, in order:
  `engram_token_map`/`engram_hash_multipliers` in the config, an
  `engram_tokens.json` sidecar, or a `tokenizer.json` under the same algorithm
  as the released `build_compressed_token_map`. The released repository
  publishes neither tokenizer file, so a matching tokenizer artifact must be
  supplied; the loader states that explicitly instead of silently guessing.
- Engram lookups gather only the selected table rows through u64 file offsets.
  The released tables are `[384006168, 256]` F8_E4M3 each (98,305,579,008 and
  98,308,270,592 bytes). A full-table upload is neither attempted nor possible
  on the target machine.
- The generic `VulkanTransformerConfig` still rejects `deepseek_v41` because its
  graph is a different architecture; the error now directs callers to the
  dedicated native loader rather than describing the model as under development.

### Published checkpoint compatibility

The published `model.safetensors.index.json` at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170` names 96,085 tensors
(510,286,023,000 bytes). `validation/verify_deepseek_v41_published_index.py` checks every required name
for the 40 backbone layers, all 384 routed experts, shared experts, HC matrices,
norms, compressor/indexer owners and Engram owners against that index:

```
PASS: 93378 required names present; 9 of 9 optional names absent as expected; index contains 96085 tensors
```

The nine absent names are exactly the optional ones: a ratio-1 compressor has no
`wgate`, and the shared-index consumers at layers 24/28/32/36 own no
`wk`/`k_norm`. Python pattern checks and the Rust plan agree on those owners.

Representative SafeTensor headers (HTTP range reads of five shards; no weight
payload was downloaded) confirm the storage contracts the loader expects:

| Tensor | dtype | shape |
| --- | --- | --- |
| `embed.weight`, `head.weight` | BF16 | `[129280, 5120]` |
| `layers.0.attn.wq_a.weight` / `.scale` | F8_E4M3 / F8_E8M0 | `[1280, 5120]` / `[40, 160]` |
| `layers.0.ffn.experts.0.w1.weight` / `.scale` | I8 (4-bit pairs) / F8_E8M0 | `[2304, 2560]` / `[2304, 160]` |
| `layers.0.ffn.shared_experts.w1.weight` / `.scale` | F8_E4M3 / F8_E8M0 | `[2304, 5120]` / `[72, 160]` |
| `layers.0.ffn.gate.weight` / `.bias` | BF16 / F32 | `[384, 5120]` / `[384]` |
| `layers.0.attn.attn_sink` | F32 | `[64]` |
| `layers.0.attn_norm.weight` | BF16 | `[5120]` |
| `layers.1.engram.embed.weight` / `.scale` | F8_E4M3 / F8_E8M0 | `[384006168, 256]` / `[384006168, 8]` |

The earlier estimate of 24.6 GB of Engram storage was wrong; the published
headers show 196.6 GB for the two tables. The row-gather path is therefore
load-bearing, not an optimization.

### Result

- `deepseek_v41_default_package_loader_strict_all_logits` drives the BF16
  fixture through `from_hf_package` with an `engram_tokens.json` sidecar: 320
  frozen logits across six prefill/decode/reset steps, worst drift `<2e-7`,
  `rtol=0`; the loader selected BF16 and retained zero experts, tables,
  projections and vocabulary tensors.
- `deepseek_v41_package_generation_matches_manual_steps` compares
  `prefill_last` with a complete prefill (the projection intentionally switches
  from the row GEMM to the vector dot below four rows, so the last row can
  differ by one ulp) and confirms `generate_sequences_ids` reproduces manual
  prefill/decode argmax steps.
- `tokenizer_derives_fixture_engram_map_and_multipliers` builds a tokenizer from
  the fixture's compressed ids and recovers the exact token map and the
  layer-seeded odd multipliers without any sidecar.
- `deepseek_v41_released_engram_prime_ranges_match_table_rows` shows that the
  generated 24 prime buckets per Engram layer sum to exactly the published
  `engram_num_embeddings` values, 384,006,168 and 384,016,682.
- The opt-in `published_v4_1_tokenizer_reproduces_released_engram_metadata` test
  runs the same Rust derivation against the published
  `DeepSeek-V4.1-Flash/tokenizer.json`
  (6,367,257 bytes, SHA-256
  `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`):
  129,280 vocabulary ids compress to exactly the configured 99,092 ids, and the
  layer-1 multipliers match the frozen NumPy reference. The weights repository
  itself ships no tokenizer file.
- `cargo test --lib deepseek_v41_`: 115 passed, 0 failed, 17 ignored; the
  separate tokenizer module adds two default tests and the opt-in published
  tokenizer test.
- Serial full library suite: 930 passed, 0 failed, 27 ignored.
- Shader provenance: 516 artifacts identical; no shader source or SPIR-V was
  modified in this change.

### Limits

The 510 GB checkpoint itself was not downloaded or executed. The check above
validates naming and storage headers, not full released-geometry numerics.
Million-token contexts, training, DSpark, vision, and joint full-width execution
remain unqualified. Hyper-connections, norms, gates, caches and Engram
projections remain resident.

### Reproduction

```powershell
cd hierarchos-vulkan
cargo test --lib deepseek_v41_default_package_loader_strict_all_logits -- --nocapture
cargo test --lib deepseek_v41_package_generation_matches_manual_steps -- --nocapture
cargo test --lib deepseek_v41::tokenizer::tests -- --nocapture
$env:V41_TOKENIZER = "C:\path\to\DeepSeek-V4.1-Flash\tokenizer.json"
cargo test --lib deepseek_v41::tokenizer::tests -- --ignored --nocapture
cargo test --lib -- --test-threads=1
python validation/check_shader_provenance.py
python validation/verify_deepseek_v41_published_index.py --index ../target/deepseek-v41-reference/published-index.json
```

## Native CLI entry point

`hierarchos-native-cli transformer-generate` now recognizes
`model_type=deepseek_v41` (and the `deepseek_v41_text` alias) in the package
`config.json` and dispatches directly to
`DeepseekV41CausalLM::from_hf_package`. The generic
`VulkanTransformerConfig`/`VulkanTransformer` path is never entered, so the
former "load the package with DeepseekV41CausalLM::from_hf_package instead"
error can no longer stop normal generation.

The dedicated path keeps the ordinary CLI contract:

- Prompt encoding, `--prompt-file`, `--add-special-tokens`, tokenizer decode,
  `--skip-special-tokens` and the generation-config precedence chain follow the
  same CLI inputs and behavior as every other `transformer-generate`
  architecture.
- Greedy and sampling generation, `--num-return-sequences`,
  `--max-new-tokens`, stop strings, EOS ids, suppression, sequence bias and the
  remaining supported generation processors run through
  `DeepseekV41CausalLM::generate_sequences_ids`.
- Repeated `--prompt` values execute serially against one graph and retain the
  normal `=== request N ===` output framing.
- `--device-index` selects the Vulkan device as usual.

Fail-closed behavior is explicit rather than silently ignored:

- `--lora-adapter-path` / `--peft-adapter` is rejected because the dedicated
  graph has no adapter topology.
- `--continuous-batching` and non-dynamic `cache_implementation` values are
  rejected; the dedicated model owns its cache and serial request scheduling.
- Context length is bounded by the native executor's 65,535-position context
  limit. The released config advertises `max_position_embeddings=1048576`, so
  an explicit larger `--context-length` reports both limits instead of being
  truncated silently. The default remains prompt length plus the requested new
  tokens, capped at 65,535.
- The released repository publishes no tokenizer; the CLI therefore still
  requires a matching `tokenizer.json` (or `engram_tokens.json` for Engram
  metadata) beside a `--model-path` package. `--hf-model` inherits the existing
  requirement for published tokenizer assets.

### Verification

- `transformer_generate_routes_deepseek_v41_through_the_dedicated_loader`
  builds the existing tiny V4.1 fixture package in a temporary directory
  (config, Engram sidecar, 206 F32 tensors and a 16-token WordLevel tokenizer),
  invokes the normal CLI argument parser and generation path, and generates two
  tokens through the dedicated graph on Vulkan. This exercises package
  detection, tokenization, generation-config loading, the dedicated loader,
  prefill plus decode, and output decoding end to end.
- `deepseek_v41_generation_context_is_bounded_by_native_runtime` uses the
  released 1,048,576-position config to check the default, explicit and clamped
  contexts and both rejection boundaries.
- `cargo test --lib` in `hierarchos-native-cli`: 40 passed, 0 failed.
- `cargo check --lib` in `hierarchos-native-cli`: passes (upstream Vulkan
  warnings only).
- The CLI wiring did not modify the generic `VulkanTransformer`, V4.1
  shaders, frozen logits or the `<2e-7` numerical gate. The later two-layer
  production fixture extension is recorded in the release-gate section below.

### Limits

No released 510 GB checkpoint was available for an end-to-end CLI run; the
route is verified with the existing frozen tiny package. PEFT, continuous
batching, paged cache serving and V4.1 beam/contrastive/guided/DOLA search
remain unavailable, and the dedicated graph still runs one request at a time.
The 65,535-position bound is an executor limit, not evidence that
million-token contexts are qualified.

### Reproduction

```powershell
cd hierarchos-native-cli
cargo test --lib transformer_generate_routes_deepseek_v41_through_the_dedicated_loader -- --nocapture
cargo test --lib deepseek_v41_generation_context_is_bounded_by_native_runtime -- --nocapture
cargo check --lib
```

With a local package and its tokenizer beside `config.json`:

```powershell
hierarchos-native-cli transformer-generate `
  --model-path C:\path\to\DeepSeek-V4.1-Flash-1017 `
  --prompt "Hello" --max-new-tokens 32
```

## Release parity gate

DeepSeek V4.1 now has the same absolute-only `<2e-7`, `rtol=0` release gate used
by the other architectures, applied to the smallest production-width model
that exercises inter-layer accumulation (two layers) instead of the unavailable
510 GB checkpoint. The gate runs through
`DeepseekV41CausalLM::from_hf_package`, the shipping default loader used by the
CLI and the normal package API, not through an explicit residency constructor.

The frozen package fixes the released width in two layers with two routed
experts:

- hidden size 5120, 64 attention heads of width 512, Q/O ranks 1280/1024,
  eight output groups, expert intermediate width 2304 and HC 4;
- full 129,280-token vocabulary and head;
- 31 stored tensors plus 26 packed FP8/FP4 tensors across four pinned shards;
- two single-token steps: start-zero prefill and cached decode.

Result:

- `deepseek_v41_default_package_production_width_strict_all_logits`:
  **258,560 logits, zero drift** on both steps.
- `summarize_deepseek_v41_qualification.py`: **17 models, 103 steps, 567,689
  logits**, worst drift `1.7881393432617188e-7`, strictly below `2e-7` with
  `rtol=0`.
- `cargo test --lib deepseek_v41_`: **115 passed, 0 failed, 18 ignored**
  (the new production default-loader case is opt-in because the package is
  3.1 GB).

The gate deliberately does not claim released depth, 384 routed experts,
million-token contexts, training, DSpark or vision. Those axes are reduced in
this fixture; the qualified claim is that the production width and complete
vocabulary are bit-exact under the shipping loader, which is the evidence used
to scale depth and expert count while re-running the same gate. The fixture now
has SHA256 `a828888525461105edb85428e290bec0527d7d4a8ed6ae255d521e6e7b2bc95d`
(31 stored tensors, 26 packed); the original one-layer package remains recorded
above as history.

### Reproduction

```powershell
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_default_package_production_width_strict_all_logits -- --ignored --nocapture
.\.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/summarize_deepseek_v41_qualification.py --output target/deepseek-v41-reference/release-summary.json
```

## Four-layer production-width accumulation (2026-10-25)

Extended the production-width release fixture from two layers to four so the
gate checks repeated inter-layer accumulation instead of a single hand-off.
The unchanged `<2e-7`, `rtol=0` gate still passes with zero drift:

- `deepseek_v41_default_package_production_width_strict_all_logits`:
  **258,560 logits, 0.0 drift** on both prefill and decode (329.65 s).
- 59 stored tensors plus 52 packed FP8/FP4 tensors across five shards;
  checkpoint SHA256 `1f06675a0a1605659541c29e9f24aeaded6b17cd25939e1a809f35e193aec984`.
- The attention and grouped-reduction fixtures were regenerated only to
  repin the production parent hash. Both read layer 0, whose parameters are
  drawn before layers 2-3, so extending depth leaves their inputs and expected
  values untouched.

The oracle needed one memory fix before it could write the larger package: the
packed FP4/FP8 matrices were held as Python lists of byte values (~2 GB across
four layers), and serializing the 1.3 GB embedding shard alongside them
exhausted this 10.7 GB host. `compact_packed` now converts each matrix to a
`uint8` tensor in place before the write, dropping the list immediately.

Reduced axes are explicit and unchanged: routed experts are still two, context
is two steps, and compression/index/Engram schedules are absent.

### Verification

- Production gate: **258,560 logits, 0.0 drift**, both steps, 329.65 s on the
  AMD Radeon Graphics Vulkan device.
- `cargo test --lib deepseek_v41_`: **115 passed, 0 failed, 18 ignored**.
- `summarize_deepseek_v41_qualification.py`: **17 models, 103 steps, 567,689
  logits**, worst drift `1.7881393432617188e-7`, strictly below `2e-7` with
  `rtol=0`.
- Production checkpoint: five shards, 59 stored tensors, 52 packed tensors, all
  hashes pinned in the fixture.

### Reproduction

```powershell
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_production_model.py `
  --reference-path .tmp-v41-ref/model.py `
  --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py --write-fixture
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/trace_deepseek_v41_production_attention.py `
  --reference-path .tmp-v41-ref/model.py `
  --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py `
  --output target/deepseek-v41-reference/production-attention-reference.json --write-fixture
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_grouped_reduction.py `
  --observation target/deepseek-v41-reference/production-attention-reference.json --write-fixture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_default_package_production_width_strict_all_logits -- --ignored --nocapture
```

## Eight-layer production-width accumulation (2026-10-25)

Extended the production-width release fixture from four layers to eight. The
gate, the frozen fixture contract and the CPU PyTorch oracle are unchanged, and
the gate still passes with zero drift:

- `deepseek_v41_default_package_production_width_strict_all_logits`:
  **258,560 logits, 0.0 drift** on both steps (129,280 prefill + 129,280
  decode), 499.62 s on the AMD Radeon Graphics Vulkan device.
- `cargo test --lib deepseek_v41_`: **115 passed, 0 failed, 18 ignored**.
- `summarize_deepseek_v41_qualification.py`: **17 models, 103 steps, 567,689
  logits**, worst drift `1.7881393432617188e-7`, strictly below `2e-7` with
  `rtol=0`.
- 516 checked shader artifacts still reproduce byte-for-byte.

Fixture `deepseek_v41_production_model.json` is now
`3549f48f8b2b6da754e667b23434205c585bc09c533a50a98dc1f5276e1a3ef9`, with an
eight-shard 4.55 GB package (`deepseek_v41_production_4141090`), 115 stored
tensors plus 104 packed FP8/FP4 matrices. Every shard hash is pinned in the
fixture. Two independent oracle runs produced identical shard hashes and
identical logits.

Extending depth changes nothing in layer 0: the attention trace bits, grouped
`wo_a` hash and `input_bits`/`expected_bits` sequences are identical to the
four-layer fixture. The attention and grouped-reduction fixtures were
regenerated only to repin the new parent hash.

### Host-memory staging

Eight production-width blocks are 0.87 GiB each in FP32, and the deferred
embedding and head are another 1.23 GiB and 2.47 GiB. The literal path from the
four-layer case (construct everything, fill everything, quantize everything,
then write) needs about 10.7 GiB and died with an allocation failure / access
violation on this 10.7 GiB host. The generator now stages the same work; none
of these changes touch the reference forward, the weights, the RNG order or the
tolerance:

- `defer_large_allocations` installs `ParallelEmbedding`/`ParallelHead` shims
  that allocate their weights on demand, so the blocks exist before the
  vocabulary tensors do.
- `Transformer` is built as an empty shell and `Block` instances are appended
  one at a time; each block is filled, quantized and rounded to the BF16 stream
  dtype before the next block starts.
- `compact_spec` converts each packed matrix's `.tolist()` output to a `uint8`
  tensor as it is produced. Left to the end, those lists cost roughly 1 GiB of
  pointers per production-width block.
- `write_checkpoint` writes the embedding, blocks and packed matrices first,
  releases them, and only then rounds the head to its stored BF16 form and
  writes the last shard. Values and shard contents are identical to a
  single-pass write; the two large vocabulary tensors simply never coexist.

The observed resident set stayed near 4.2 GiB during the fill and forward
stages, and the two-pass writer holds at most one shard beyond that.
The released geometry is still unqualified on every other axis: two routed
experts, two single-token steps, and no compression/index/Engram schedule. The
claim is the same production width, complete vocabulary and shipping
`from_hf_package` loader as before, now with eight inter-layer accumulations.

### Reproduction

```powershell
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_production_model.py `
  --reference-path .tmp-v41-ref/model.py `
  --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py --write-fixture
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/trace_deepseek_v41_production_attention.py `
  --reference-path .tmp-v41-ref/model.py `
  --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py `
  --output target/deepseek-v41-reference/production-attention-reference.json --write-fixture
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_grouped_reduction.py `
  --observation target/deepseek-v41-reference/production-attention-reference.json --write-fixture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_default_package_production_width_strict_all_logits -- --ignored --nocapture
```
## Sixteen-layer production-width accumulation (2026-10-25)

The production-width release fixture now runs at sixteen layers, doubling the
accumulation depth again. The gate, the frozen fixture contract and the CPU
PyTorch oracle are unchanged, and the gate still passes with zero drift:

- `deepseek_v41_default_package_production_width_strict_all_logits`:
  **258,560 logits, 0.0 drift** on both steps (129,280 prefill + 129,280
  decode), 882.02 s on the AMD Radeon Graphics Vulkan device.
- `cargo test --lib deepseek_v41_`: **115 passed, 0 failed, 18 ignored**.
- `summarize_deepseek_v41_qualification.py`: **17 models, 103 steps, 567,689
  logits**, worst drift `1.7881393432617188e-7`, strictly below `2e-7` with
  `rtol=0`. The production entry is the refreshed `from_hf_package` export,
  bound to the new fixture hash.
- 516 checked shader artifacts still reproduce byte-for-byte, and 397 push
  blocks plus 109 host structs still agree.

Fixture `deepseek_v41_production_model.json` is now
`b9447641c6018d33d8a517a28789bea42dd57926f780099b976e75880cb86c48`, with a
fourteen-shard 6.1 GB package (`deepseek_v41_production_4141090`), 227 stored
tensors plus 208 packed FP8/FP4 matrices. Every shard hash is pinned in the
fixture.

Depth changes nothing in layer 0: the attention trace bits, the grouped `wo_a`
hash and the `input_bits`/`expected_bits` sequences are identical to the
eight-layer fixtures. `deepseek_v41_production_attention.json` and
`deepseek_v41_grouped_reduction.json` were regenerated only because they bind
the parent fixture's hash.

### Host-memory staging

The eight-layer case already staged blocks one at a time and deferred the
vocabulary tensors. Sixteen layers add three things, none of which touch the
reference forward, the weights, the RNG order or the tolerance:

- The deferred embedding and FP32 head are now allocated in memory-mapped
  scratch files (`page_backed`) instead of anonymous memory. Together they are
  3.95 GiB and stay live across the whole forward; with 3.4 GiB of packed
  blocks alongside them, two anonymous allocations of that size exceed this
  10.7 GiB host. The bytes, dtype, shape and shard hashes are unchanged, and
  the OS can evict pages the forward is not touching.
- `release_scratch` removes those files once `write_checkpoint` has dropped the
  last view of each tensor.
- `write_checkpoint` now also drops the head module after its shard is written,
  so the mapping can be released.

Observed peak private commit during the forward was 6.0 GiB at a 3.2 GiB
working set, and the working set peaked at 5.3 GiB while writing shards. The
two-pass writer holds at most one shard beyond the resident state.

The released geometry is still unqualified on every other axis: two routed
experts, two single-token steps, and no compression/index/Engram schedule. The
claim is the same production width, complete vocabulary and shipping
`from_hf_package` loader as before, now with sixteen inter-layer accumulations.

### Reproduction

```powershell
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_production_model.py `
  --reference-path .tmp-v41-ref/model.py `
  --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py --write-fixture
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/trace_deepseek_v41_production_attention.py `
  --reference-path .tmp-v41-ref/model.py `
  --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py `
  --output target/deepseek-v41-reference/production-attention-reference.json --write-fixture
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_grouped_reduction.py `
  --observation target/deepseek-v41-reference/production-attention-reference.json --write-fixture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_default_package_production_width_strict_all_logits -- --ignored --nocapture
```
## Twenty-four-layer production-width accumulation (2026-10-25)

The release gate now runs twenty-four production-width layers, so the stack
extends past every position where the released schedule changes shape: ratio 2
starts at layer 2, ratio 1 starts at layer 20, and the KV/index sources are
2/8/14/20. Compression itself stays disabled in this graph; the gate still
measures width and residual accumulation only. The gate, the frozen fixture
contract and the CPU PyTorch oracle are unchanged, and it still passes with
zero drift:

- `deepseek_v41_default_package_production_width_strict_all_logits`:
  **258,560 logits, 0.0 drift** on both steps (129,280 prefill + 129,280
  decode), 1793.61 s on the AMD Radeon Graphics Vulkan device, loaded through
  `from_hf_package`.
- `cargo test --lib deepseek_v41_`: **115 passed, 0 failed, 18 ignored**.
- `summarize_deepseek_v41_qualification.py`: **17 models, 103 steps, 567,689
  logits**, worst drift `1.7881393432617188e-7`, strictly below `2e-7` with
  `rtol=0`.
- All 516 checked shader artifacts still reproduce byte-for-byte; 397 push
  blocks and 109 host structs still agree.

Fixture `deepseek_v41_production_model.json` is now
`e00fbc20ed6226ca2eb9f2f07c9f4041c01298da674d0f020d8dd66fb9facd44`, with a
twenty-shard 7.8 GB package (`deepseek_v41_production_4141090`), 339 stored
tensors plus 312 packed FP8/FP4 matrices, every shard hash pinned.

Layer 0 is again untouched: the attention trace bits, grouped `wo_a` hash and
`input_bits`/`expected_bits` sequences are identical, so
`deepseek_v41_production_attention.json` and
`deepseek_v41_grouped_reduction.json` were regenerated only to repin the parent
fixture hash.

### Packed weights are now page-backed

Twenty-four packed blocks are 5.2 GiB. Together with the 1.23 GiB embedding and
the 2.47 GiB FP32 head the generator needed 8.9 GiB resident, and the first
attempt died with an access violation at ~8 GiB private commit. `compact_spec`
now places every packed `data`/`scale` blob in a scratch file mapping
(`page_backed`), exactly as the vocabulary tensors already were. The forward
and the writer read the same bytes; only the backing store changed.

Observed private commit stayed near 3.5 GiB through fill, forward and write
(3.8 GiB of packed scratch, deleted afterwards), versus 8.0 GiB and a crash
before the change. Shard hashes are byte-identical in form and pinned again.

### What is still reduced, and why it stays a composition

The remaining released axes cannot be added to this graph on this host:

- **384 routed experts at production width.** Expert weights are 3 x 2304 x
  5120 per expert, or 17.7 MB in FP4: **6.8 GB per layer**, 163 GB at this
  depth against 47 GB free. Quantization is the harder limit: one 2304x5120
  FP4 matrix takes **15.5 s** on this CPU, so one expert is ~46 s and one layer
  of 384 experts is **~4.9 h** before any forward runs.
- **Released Engram tables.** Each table has 384,006,168 rows of
  `engram_n_heads x engram_head_dim` values, far beyond local storage. Row
  count is a storage axis, not an arithmetic one, so Engram is qualified with
  reduced tables at the released projection width.
- **Full 40-layer depth and million-token context.** The released checkpoint is
  ~600B parameters; the 2,048-token prefill is qualified at reduced widths.

Each of those axes is exercised where it fits, under the same gate: the joint
probe runs all 40 released layers, the compression/KV/index/Engram schedule and
384-expert top-6 routing over six steps, and the long-context probe runs a
2,048-token start-zero prefill at the released 32x128 index heads. Because
every operator is bit-exact against the same CPU reference in each probe, and
because the production-width gate proves the residual accumulation, BF16
rounding and packed expert/attention GEMMs at released width, the composition
is arithmetic, not extrapolation. What remains unqualified is the *joint*
full-size execution, not any individual operator's math at released geometry.

## Separate production-width schedule and router probes (2026-10-25)

This continuation preserves the twenty-four-layer accumulation fixture
(`e00fbc20ed6226ca2eb9f2f07c9f4041c01298da674d0f020d8dd66fb9facd44`)
and adds a separate four-layer case. The new CPU oracle is frozen before
native execution; its final fixture SHA256 is
`70d2218429c2d4b8623a8ebbdd6c659eec22f7f64f9eeecd3d95c5d7877fdcf0`.
No old expectations or oracle scripts were regenerated.

The new case fixes hidden width 5120, 64x512 attention, Q/O ranks 1280/1024,
eight output groups, expert intermediate width 2304, HC 4, full vocabulary
129280, index heads 32x128, and Engram heads 8x256 with four-gram history.
Its schedule `[2,2,1,1]` has KV owners `[0,2]`, index owners `[0,2,3]`,
candidate publication at layer 2 and consumption at layer 3, and Engram
injections at layers 0 and 3. Each token evaluates six routed experts plus
the shared expert. The table budgets are 4096/8192; context, layer count and
expert payload count are explicitly reduced.

Five steps produce 904,960 full-vocabulary logits: a three-token masked
prefill, compression-group completion, an incomplete group, a shorter
start-zero reset, and reset-group completion. The third step actually filters
out a candidate block: the CPU trace records `-inf` scores at compressed
indices 2 and 3 in layer 3. The real published 129280-id tokenizer compresses
to 99092 ids and supplies the original NumPy hash multipliers through the
normal Engram sidecar path. The eight-shard package is 4,259,979,085 bytes
including metadata, with 75 stored tensors and 107 packed FP8/FP4 matrices.

The fixture generator bounds conversion memory with aligned row tiles. FP8
still uses the original converter; the vector FP4 converter is checked byte
for byte, including scale bytes, against the original scalar converter on
random aligned/partial tiles and every frozen FP4 row32 tie/floor probe.
The published forward bodies and existing CPU kernel mirrors are unchanged.

### Full-cardinality router fix

A separate component probe runs the unchanged published `Gate.forward` over
all 384 candidates at width 5120, top-6, with 1/3/4 rows. It found a native
bug: the matrix gate used a single long FP32 FMA chain, whereas the pinned
CPU SGEMM stages K in 192-wide panels. The existing dense FP32 linear shader
already used the correct panels. The gate now applies the same panel order;
this is FP32 even for BF16 storage because `Gate.forward` promotes its operands.

The frozen router reference was not changed to accept the fix. Raw linear
scores, selected indices, raw and normalized routing weights, and per-expert
counts now match by bits. A second independent CPU run reproduced both the
fixture and all 3,932,160 BF16 weight bytes. The two existing router regressions
also pass.

### Production-width Engram fix and metadata correction

The first full-logit run failed at 2.925063133239746 absolute drift. The
independent stage trace located the first error at layer 0's Engram injection:
one BF16 value out of 61,440 differed by 3.0517578125e-5. Embedding input and
the complete 76,800-value Engram projection were bit-exact. That single value
propagated through attention and the residual stack into the logit failure.

Engram's square and weighted-dot reductions used flat 32-lane banks. Width
5120 requires CPU SumKernel's four-level cascaded accumulation, already used
by the other native RMS surfaces. The gate now uses the same cascade for
stream squares, key squares and the weighted dot. A new frozen component
regression replays the unchanged published `Engram.forward` with the original
CPU projection and checkpoint q/k weights. It reproduces the one-value failure
before the fix and passes all 61,440 BF16 output values by bits after it.
An independent repeat reproduces the CPU expectation.

The trace also exposed an error in this new case's configuration metadata:
the inherited production helper declared YaRN factor 16 while the unchanged
CPU `ModelArgs` used 40. Only the new fixture's declared factor and config-file
hash were corrected to 40; all 904,960 frozen logits, other step values and
all eight SafeTensors shard hashes remain identical. The old metadata fixture
is retained in `target/deepseek-v41-reference/schedule-width-original-metadata-fixture.json`;
`schedule-width-metadata-correction.json` records the change. The new generator
sets the argument explicitly and derives the native factor from that value.
The older 24-layer case has ratio zero, which disables YaRN in both executors,
and is unaffected. Its frozen fixture and package were preserved.

### Narrow sparse scores and grouped BF16 prefill

After the Engram fix and metadata correction, the logit gate still failed at
0.32190895080566406. The trace then showed exact Engram and query projections,
but five differing BF16 sparse output channels in layer 2. A separate
attention-only replay of the unchanged layer-2 forward reproduces its frozen
CPU output and captures scores/weights/raw values. It identified the missing
narrow SGEMM branch: allocated 4/5-column score matrices use the CPU four-lane
dot above ATen's 400-product small-BMM threshold. Previously only the one-column
case selected that body. The shader now covers all 1..8-column narrow matrices.

The same trace revealed a separate grouped-projection reduction error at
two/three rows. The BF16 single-row path requires 512-wide chunks, retained
for the existing production decode regression; two or more rows require a
full K accumulation. A compact frozen probe retains actual checkpoint rows,
N=1024 and K=4096, with 1/2/3/4 input rows. It reproduced the 2/3-row failures
and now passes every output by bits. Sparse weights, raw outputs and BF16
outputs also pass by bits. A second independent CPU run reproduces this
attention fixture without rewriting it.

The changed production shaders are the MoE gate, Engram gate, sparse attention
weights and grouped output, plus the new Engram reduction include. No pre-phase
CPU fixture or oracle script changed. The old accumulation gate's single-row
grouped projection and ratio-zero rotary paths retain their arithmetic.

### Reproduction for the additive gate

The checked-in CPU expectations are frozen. To rebuild the ignored shard
package, omit `--write-fixture`; the generator verifies its complete result
against the existing fixture. Use the pinned model/kernel sources and matching
published tokenizer already recorded above.

```powershell
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_schedule_width_model.py `
  --reference-path .tmp-v41-ref/model.py `
  --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py `
  --tokenizer target/deepseek-v41-reference/tokenizer-v41.json
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_default_package_schedule_width_strict_all_logits -- --ignored --nocapture --test-threads=1
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_ -- --test-threads=1
cargo test --manifest-path hierarchos-native-cli/Cargo.toml --lib -- --test-threads=1
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/summarize_deepseek_v41_qualification.py `
  --output target/deepseek-v41-reference/schedule-width-qualification.json
```

`HIERARCHOS_V41_SCHEDULE_WIDTH_TRACE=1` adds observational stage buffers to
the test export. On a failing step it stops after writing the diagnostic;
the complete-logit-count assertion still fails, so this mode cannot accept
an incomplete run. It is a test diagnostic, not a runtime enable flag.

### Final measured results

- `deepseek_v41_default_package_schedule_width_strict_all_logits`: **904,960
  logits, zero drift** across all five steps; 1331.88 s on AMD Radeon Graphics.
  The shipping loader infers BF16 and streaming without policy flags. Each
  completed call retains zero routed/shared expert payloads, Engram tables,
  attention projection layers and vocabulary tensors.
- Aggregate recheck: **18 models, 108 steps, 1,472,649 logits**, worst absolute
  drift **1.7881393432617188e-7**, strictly `<2e-7`, `rtol=0`. Eleven BF16 models
  contribute 1,470,409 zero-drift logits. Both production-width exports must
  identify the shipping `from_hf_package` loader and bind their fixture/shards.
- Default native V4.1 suite: **119 passed, 0 failed, 19 ignored**, 374.38 s.
  CLI library suite: **40 passed, 0 failed**, including native package generation.
- All **516 shader artifacts** reproduce byte for byte from the current source;
  all **397 push blocks / 109 host structs** agree. The shader contract suite
  passes 28 tests; the strict logit-boundary suite passes two. Rust formatting,
  Python compilation and `git diff --check` pass.
- All pre-phase CPU fixtures and oracle scripts are unchanged. Of the 594
  recorded inputs, 590 are unchanged; only the four intentionally rebuilt
  SPIR-V files differ. The 24-layer fixture remains
  `e00fbc20ed6226ca2eb9f2f07c9f4041c01298da674d0f020d8dd66fb9facd44`.

The additional observational trace contains 6,497,008 stage values. All except
14 grouped BF16 projection values match by bits; the largest intermediate
difference is 0.015625. An independent comparison of both versions through the
following FP8 activation converter proves identical values and scales on all
eight affected grouped surfaces. Every later recorded stage and final logit is
identical. These residual grouped-rounding differences are retained as an
explicit limitation; the compact grouped regression covers its frozen channels,
not every possible matrix/value combination. The full-logit acceptance gate
passes without relaxing any ceiling or rewriting expectations.

Reports are in `target/deepseek-v41-reference/`: `schedule-width-native-final.log`,
`schedule-width-qualification.json`, `schedule-width-native-suite.log`,
`schedule-width-cli-tests.log`, `schedule-width-stage-parity.json`,
`schedule-width-shader-provenance.json`, and `schedule-width-preservation-check.json`.
Joint full-size execution, released table storage, long prompt continuations,
million-token contexts, vision, DSpark and training remain unqualified. The full
510 GB checkpoint was not downloaded or executed in this continuation.

## Scalable architecture qualification and storage cleanup (2026-10-25)

The user explicitly accepted the same scalable-checkpoint base qualification
used for the other architectures. Under that criterion, **native causal text
inference is qualified on this machine**. Full production weights are not a
qualification prerequisite. The supplementary intermediate trace is diagnostic;
its 14 FP8-equivalent grouped values do not fail the full-logit acceptance gate.

The fresh `deepseek_v41_default_package_scalable_release_strict_all_logits` run
uses the existing immutable joint CPU fixture and its small sharded package.
The normal loader reads an Engram metadata sidecar, infers BF16/packed storage,
and chooses all streaming policies without execution flags. All six masked
prefill/decode/reset steps pass **3,216 full logits with zero drift** and exact
per-layer routing (including indices above 255), in **501.93 s**. The test also
checks zero retained routed/shared expert payloads, Engram tables, attention
projections and vocabulary after every call. Its separate production-feature
contract test passes.

The contract directly matches all 40 released compression ratios, KV/index
source ids, Engram layers and four-gram/head features, candidate source/block
size, 384 routed experts/top-6 density, HC 4, routing score/normalization and
scale. Width, vocabulary/table sizes, context and top-k capacities are reduced
in this joint graph. Recorded full-width gates independently cover the released
arithmetic geometry. The aggregate remains **18 models / 108 steps / 1,472,649
logits**, worst drift **1.7881393432617188e-7**, strictly `<2e-7`, `rtol=0`.
The joint aggregate entry now uses this fresh default-loader execution.

`qualify_deepseek_v41_scalable.py` records a durable machine receipt and a compact
gzip bundle containing every compared native logit and routing field. Recording
requires a fresh successful default-loader run after the bound runtime inputs;
verification checks current source/shader hashes and independently recomputes
the complete CPU comparisons from the archived output bundle. Verification
passes both before and after deleting the large checkpoints. The receipt records
AMD Ryzen Z1 Extreme, Radeon Graphics PCI 1002:15BF, driver 32.0.31007.6002.

The pre-cleanup inventory measured 17.5 GB of obsolete generated checkpoint and
scratch payloads on this device. **145 individual generated files totaling
17,467,935,126 bytes** were removed from the two large native width packages,
the earlier PyTorch package copies, and two abandoned page-backed scratch areas.
Automatic approval review blocked recursive directory removal, so cleanup was
narrowed to individual, verified generated files; that operation succeeded.
Metadata directories and every frozen CPU expectation remain intact.

Twenty-nine large diagnostic JSON files (4,431,037,957 bytes) were archived into
1,026,068,872 bytes of content-addressed gzip data. Every archive was decompressed
and its SHA256 verified before removing the corresponding original file. This
recovers another **3,404,969,085 bytes**, for **20,872,904,211 bytes total**
(20.9 GB decimal). The retained release checkpoint is **44,770,196 bytes**.
Cleanup ledgers are `checkpoint-cleanup.json` and `diagnostic-archive.json` in
the durable receipt directory; archives are under
`target/deepseek-v41-reference/archived-diagnostics/`.

The earlier 119-test native regression and 40-test CLI receipts are retained;
this pass additionally executes the fresh default-loader joint gate and its
feature-contract test. No production runtime arithmetic changed in this pass.
PEFT remains independent follow-up work. The Intel Gen9 preparation and port
acceptance commands are in `validation/intel-handoff/DEEPSEEK_V41.md`.

```powershell
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_default_package_scalable_release_strict_all_logits -- --ignored --nocapture --test-threads=1
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/qualify_deepseek_v41_scalable.py
```

## PEFT-LoRA on the dedicated V4.1 text graph

**Status: additive inference adapters are qualified at production width; adapter
training and merged-package export are not.**

The dedicated graph now accepts standard HF PEFT-LoRA directories instead of
rejecting them. The released package is never rewritten: packed FP8/FP4 blocks,
scales, shards and hashes stay byte-identical. Each targeted projection carries
an explicit FP32 low-rank branch,

```text
y = round_stream( base(x) + (alpha / r) * B(A(x)) )
```

where `base` keeps its qualified packed/dense GEMM contract and the delta is
computed from the module input (not the FP8-quantized activation). The scaling
multiply is rounded before the add, matching PEFT's wrapper; the single output
rounding is the graph's existing BF16 boundary. `rsLoRA` (`alpha / sqrt(r)`),
per-module `rank_pattern`/`alpha_pattern`, BF16/F16 adapter storage and both
`attn`/`self_attn` HF module spellings are supported. The staging hook is
`read_linear_weight`, so resident, streamed-expert and streamed-projection
loaders attach the same module by canonical checkpoint tensor name.

### Numeric gate

A fresh production-width package carries one released-width block: 5120 hidden,
64x512 attention heads, Q/O ranks 1280/1024, eight output groups, 2304-wide FP4
experts, HC 4, two routed experts and a 512-token vocabulary. The adapter
targets packed FP8 `attn.wq_a`, `attn.wq_b`, `attn.wkv`, `attn.wo_b`, the packed
FP8 shared expert `w1` and routed FP4 expert `0.w2`; rank 16, alpha 16.

- **1,024 logits (single-token prefill and decode), zero drift** against the
  pinned single-threaded CPU PyTorch oracle, for both the unadapted base graph
  and the adapted graph, through the default `from_hf_package_with_lora` loader.
- The adapter changes the logits by up to **2.543889**; the test also loads the
  base package and requires both families to match their own frozen logits, so
  an ignored adapter cannot pass.
- `cargo test --lib deepseek_v41`: **127 passed, 0 failed, 23 ignored**; the
  production-width PEFT gate and the four-token prefill gate are among the
  opt-in tests (`--ignored`). The full library suite passes serially: **940
  passed, 0 failed, 32 ignored**.
- CLI: **40 tests pass**; `architectures --peft --json` now lists
  `deepseek_v41` with `modules_to_save_validated=false`.
- Shader provenance: **517 artifacts** reproduce byte-for-byte, including the
  new `transformer_deepseek_v41_lora_add.spv` (glslang 16.6.0 pin).

The same fixture also carries the multi-row section below (`prefill_steps`),
so one 252 MB (240.8 MiB) package gates both the adapter and the four-token
prefill. Fixture: `validation/fixtures/deepseek_v41_lora_model.json` plus the
`deepseek_v41_lora_4141095/` package and its `adapter/` directory. Oracle:
`validation/validate_deepseek_v41_lora_model.py`. Native regression:
`src/deepseek_v41/model_lora_tests.rs`.

### Fail-closed surface

- `modules_to_save`, DoRA, QALoRA, independent LoRA-B bias, adapter-owned base
  bias, `target_parameters`, trainable tokens, ALoRA, arrow and layer
  replication are rejected with explicit messages.
- Only projections that flow through `DeepseekV41MixedLinearKernel` are
  accepted. Targets on grouped `attn.wo_a`, compressor projections or Engram
  tables fail instead of being silently skipped; F32-storage packages also fail
  closed because their untyped loader would not attach the adapter.
- `merge-lora` rejects a `model_type=deepseek_v41` package: folding the delta
  into packed FP8/FP4 blocks would change the qualified base bytes. Adapters
  run unmerged through `transformer-generate --lora-adapter-path`.
- There is no V4.1 backward graph, so PEFT training/AdamW, gradient parity and
  adapter export remain unqualified and are not implied by this gate.

### Multi-token prefill at released width

The earlier four-token probe disagreed with the CPU reference in rows 0–1 (max
`1.6399669647216797e-1`). The cause was the grouped output projection: the CPU
bmm behind `einsum("bsgd,grd->bsgr")` batches over groups, so its GEMM sees
`M = batch * seq`, and `M = 1` and `M > 1` are different kernels with different
BF16 reduction bodies. A BF16 tie in a single grouped value then flips one FP8
block scale downstream, which moves whole logits by ~1e-1.

The bodies were read out of the reference instead of guessed:

- Absorption probes (`+2^60` at one position, `-2^60` at another, `2^-60` at a
  third; the tiny survives only if it never shares an accumulator with the big
  value) place the multi-row K split at contiguous 1024-wide blocks folded in
  ascending order. The single-row kernel splits at 512 instead.
- One remaining BF16 tie (schedule width, row 1, group 1, column 407: exact sum
  `1.3789063901640475`, midpoint `1.37890625`) identifies the within-block order:
  the CPU kernel folds two-element dots in ascending pair order with the odd
  element first, i.e. `fma(x[2m],w[2m],fma(x[2m+1],w[2m+1],acc))`. Folding the
  even element first lands one BF16 ulp high on that value.
- `transformer_deepseek_v41_grouped_output.comp` now implements that body for
  `bf16 && rows > 1` and keeps the qualified 512-wide body for `rows == 1`.

Result: a four-token prefill followed by a decode that reads the ring the
prefill seeded, at released width and through the default `from_hf_package`
loader: **2,560 logits, zero drift**, `rtol=0`. The schedule-width component
regression also matches its CPU einsum for 1/2/3/4 rows with zero mismatches;
the multi-row steps previously matched only because the older serial body was
BF16-identical on those fixtures.

- V4.1 tests: **127 passed, 0 failed, 23 ignored** (the four-token gate is one
  of the opt-in tests). Full library suite: **940 passed, 0 failed, 32
  ignored**. CLI: **40 passed**; shader provenance: **517 artifacts** reproduce.
- The 24-layer production package was removed by the earlier storage cleanup,
  so that opt-in gate was not re-run in this pass. The changed body is covered
  by the schedule-width production component test (rows 1–4, bit-exact) and by
  the full four-token graph; single-token steps keep the old body.

### Reproduction

```powershell
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_lora_model.py --reference-path .tmp-v41-ref/model.py --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py --write-fixture
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_lora_model.py --reference-path .tmp-v41-ref/model.py --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py --native-output target/deepseek-v41-reference/lora-native-logits.json
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_lora_production_width -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_four_token_prefill -- --ignored --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_schedule_width_attention_contractions_match_cpu_bits -- --nocapture
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/check_shader_provenance.py
```

## Twenty-four-layer package regeneration and head repin (2026-10-26)

The 7.8 GB twenty-four-layer package was deleted by the 2026-10-25 storage
cleanup, so the opt-in gate that consumes it could not be re-run in that pass.
It has now been regenerated from the pinned seed and the gate re-run.

### Regeneration reproduced 19 of 20 shards byte-for-byte

Regenerating `deepseek_v41_production_4141090` from seed 4141090 reproduces
nineteen shard hashes exactly against the frozen fixture. The twentieth,
`model-19.safetensors` — the BF16 129,280x5,120 vocabulary head, 1.32 GB —
differed. The difference localises to a single head row, 5,120 values:

- 258,558 of the 258,560 frozen logits are bit-identical. Only index `33958`
differs, in both steps.
- Recomputing that logit from the regenerated head row with the fixture's own
  `normalized` head input (the reference head is a bare `F.linear`, so this is
exact) reproduces the regenerated value precisely — every other row agrees.
- An independent replay of the oracle's draw order (`fill_parameters` over the
  embedding, the twenty-four blocks, the final norm, then the head; the FP8/FP4
  quantization draws no randomness) reproduces the regenerated head's full
  1.32 GB byte string: `3640d302780733c397b0b6f0250bf786a1e25178e17ebcd49364755edb5984ab`
  (`target/deepseek-v41-reference/reproduce_head.py`).
  The current generator is therefore self-consistent across processes, and the
  stored row is not reachable by any current code path. The package that carried
  it no longer exists, so the divergence cannot be attributed to its exact cause.
  One row out of 129,280 is the complete extent of it: the other 129,279 rows and
  every other tensor in the package are byte-identical.

Because a package and its oracle must agree, the fixture was repinned from the
canonical generator rather than patched. `deepseek_v41_production_model.json`
moved from `e00fbc20ed6226ca2eb9f2f07c9f4041c01298da674d0f020d8dd66fb9facd44` to
`ea1b00dd3ffe4bf5786c4daf632c2054904b5f8f929428bbbb92636cb3e6a524`, and **exactly
three fields changed**:

| field | pinned before | repinned |
| --- | --- | --- |
| `checkpoint/files/model-19.safetensors` | `a9931e93…` | `47450fd4…` |
| `steps[0].logits[33958]` | `4.407138824462891` | `0.5811023712158203` |
| `steps[1].logits[33958]` | `1.7327481508255005` | `-0.7814326286315918` |

Every other field is unchanged: the other nineteen shard hashes, the 339 stored /
312 packed counts, the config, both steps' `normalized` and `collapsed` head
inputs, and the remaining 258,558 logits. The acceptance gate itself is
untouched: absolute drift strictly below `2e-7`, `rtol=0`, no relative
fallback. `deepseek_v41_production_attention.json` and
`deepseek_v41_grouped_reduction.json` were re-derived from the new parent; each
changed only its parent-hash field — attention payloads and the 6,144 grouped
CPU outputs are identical.

### Release gate on the regenerated package

`cargo test --lib production_width -- --ignored --test-threads=1` (log:
`target/deepseek-v41-reference/production-gate-24.log`):

| test | result |
| --- | --- |
| PEFT-LoRA (252 MB package) | **pass**, 1,024 logits, drift 0, adapter delta 2.543889 |
| default package (`from_hf_package`, release gate) | **pass**, 258,560 logits, drift 0 |
| streamed block projections | **pass**, 258,560 logits, drift 0 |
| streamed vocabulary | **pass**, 258,560 logits, drift 0 |
| resident experts and projections | out of device memory building attention layer 22 |
| streamed routed experts | out of device memory (1,323,827,200-byte vocabulary buffer) |
| streamed all experts | out of device memory (1,323,827,200-byte vocabulary buffer) |

The three failures are allocator capacity, not parity: no test reported a logit
mismatch, and every test that reached the forward passed all 258,560 logits at
zero drift. The device's Vulkan allocation limit is 4,294,967,295 bytes and the
failing variants hold the 1.32 GB vocabulary matrix resident next to roughly
6.4–6.9 GB already reserved; the resident variant reached 219 driver
allocations / 6.97 GB before layer 22. They are residency variants of the same
graph, not release paths — the shipping loader is the streaming default, and it
passes. Reaching them on this machine would need a smaller vocabulary residency
(the default loader already streams it) or a larger device heap.

- Default V4.1 suite after the repin: **127 passed, 0 failed, 23 ignored**
  (`cargo test --lib deepseek_v41`), including the frozen sparse/grouped
  fixtures that assert their parent hash.

### Qualification receipt

The stored receipt
(`validation/release_parity/deepseek_v41/qualification.json`) is **stale** and
its `--verify` path will reject it for two independent reasons: it hashes the
pre-repin production fixture inside its compressed native bundle, and its
`runtime_inputs` inventory predates the multi-row grouped-output shader from the
previous pass. Refreshing it needs a fresh 40-layer/default-loader execution
export (3,216 logits) followed by `qualify_deepseek_v41_scalable.py --record`
with the existing machine-info file; the receipt must not be hand-edited.

### Reproduction

```powershell
# 1. Regenerate the package and re-derive the pinned CPU oracle (~50 min).
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_production_model.py `
  --reference-path .tmp-v41-ref/model.py `
  --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py --write-fixture
# 2. Re-derive the two fixtures that bind the parent hash.
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/trace_deepseek_v41_production_attention.py `
  --reference-path .tmp-v41-ref/model.py `
  --kernel-reference-path target/deepseek-v41-reference/inference_kernel.py `
  --output target/deepseek-v41-reference/production-attention-reference.json --write-fixture
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_grouped_reduction.py `
  --observation target/deepseek-v41-reference/production-attention-reference.json --write-fixture
# 3. Release gate.
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib production_width -- --ignored --nocapture --test-threads=1
```

## Tiered vocabulary staging at released width (2026-10-26)

The release path uploaded each 1.32 GB vocabulary table whole: one device
allocation per table, plus a second 1.32 GB `Vec` on the host per operation. That
single payload is what stopped the expert-streaming qualification variants from
loading the 24-layer package beside their resident projections. The vocabulary now
runs as a VRAM -> system RAM -> SSD hierarchy: the package file is the source of
truth, the host page cache is the middle tier, and the device holds at most one
bounded tile.

- `DeepseekV41Checkpoint::read_bf16_word_range` and `read_f32_row_range` read a
  contiguous run of table rows with one seek and one read;
  `read_bf16_rows`/`read_f32_rows` gather a row selection through the same
  u64-offset row reader the Engram lookup already used.
- **Embedding:** the tokens of the call are read back once, uniqued, gathered from
  the package and uploaded as a compact table, and the ids are remapped onto it.
  A lookup is a copy of the same row, so every streamed value keeps its bits. For
  a 2,048-token prefill at the released embedding this is at most 2,048x5,120
  BF16 rows (~21 MB) instead of the full 1.32 GB table.
- **Head:** output columns are projected in 8,192-row tiles (an 84 MB BF16 tile).
  Each tile runs the unchanged kernel over its own column range and is placed into
  the full logits in host order. Tile widths stay above eight columns (a tile
  absorbs any remainder of eight or fewer), so the kernel selects the same
  dot-product body as the full-width dispatch and each column's result is
  identical.
- **Policy:** `Resident` is honored only when both tables fit beside the
  already-loaded backbone, measured from the driver memory budget with a 512 MiB
  reserve; otherwise the loader chooses bounded staging *before* the allocation
  that would fail, and reports the effective tier. `Streamed` -- the shipping
  default -- always stages. A table that fits one tile keeps the original
  single-dispatch path, so every previously qualified graph keeps its execution
  shape.

The measured kernels did not move. Adding a column-offset push constant to the
FP32 projection changed results on this driver even with the offset pinned to
zero (2.47e-2 at logit 0 of the six-layer fixture) although its SPIR-V was
numerically identical, so both projection sources were restored and recompiled
byte-for-byte. Shader provenance still reports **517 identical artifacts**, and
the push-constant checker still agrees across SPIR-V, GLSL and the host structs
(398 blocks, 109 structs).

### Release gate on the regenerated package

`cargo test --lib <variant> -- --ignored --nocapture`, one variant at a time
(log: `target/deepseek-v41-reference/full-gate.log`):

| variant | requested vocabulary | effective | result | time |
| --- | --- | --- | --- | --- |
| PEFT-LoRA (252 MB package) | streamed | tiered | **pass**, 1,024 logits, drift 0, adapter delta 2.543889 | 85.05 s |
| shipping default (`from_hf_package`) | streamed | tiered | **pass**, 258,560 logits, drift 0 | 1186.23 s |
| streamed routed experts | resident | tiered | **pass**, 258,560 logits, drift 0 | 852.39 s |
| streamed all experts | resident | tiered | **pass**, 258,560 logits, drift 0 | 919.42 s |
| streamed block projections | resident | resident (both tables fit) | **pass**, 258,560 logits, drift 0 | 1178.43 s |
| streamed vocabulary | streamed | tiered | **pass**, 258,560 logits, drift 0 | 1183.63 s |
| resident experts and projections | resident | resident | unqualified (capacity, below) | -- |

The three previously out-of-memory variants now load and pass: **1,293,824
logits at zero drift** across the six passing cases, with the same `<2e-7`,
`rtol=0` gate. That is the shipping path plus five residency combinations of the
same graph, all bit-exact against the frozen CPU oracle.

The two expert-streaming cases reached their residency assertions for the first
time and exposed expectations written for a one-layer graph:
`resident_shared_expert_count()` and `resident_projection_layer_count()` sum over
the backbone, so a twenty-four-layer graph reports 24 shared experts and 24
projection layers, not 1. The assertions now expect per-layer totals (and
`n_routed_experts x layers` when routed experts are resident). This is a
test-expectation correction; the compared logits were already identical.

### Resident-everything remains a capacity limit

The fully resident variant still cannot load at 24 layers: it wants ~6.97 GB of
projections and experts *before* the vocabulary is touched, against a 7.13 GB
device budget with 16.8 MB in use (7.11 GB available), and the driver refused the
allocation while building layer 22. Nothing about vocabulary staging changes
that; making it fit would mean evicting resident weights, which is exactly what
the streamed variants already qualify. It stays unqualified on this device by
capacity, not by parity.

### Qualification receipt

`validation/release_parity/deepseek_v41/qualification.json` was re-recorded from
a fresh 40-layer default-loader export (3,216 logits, 506.73 s) and then
independently re-verified, so it now binds the current sources:

- 18 models, 108 steps, **1,472,649 logits**, worst drift
  `1.7881393432617188e-7`, strictly `<2e-7`, `rtol=0`.
- `fresh_default_loader_execution.export_sha256`
  `80a67f627ce690d07e7a12e80c810e0ed19a020bbd8fff13ae2c65a5b0772fb6`,
  `log_sha256` `98c6c4ae7e6109ea26102c271a336672e9b343a99fba0772ed0f40f09a388b9d`,
  compressed bundle `native_outputs_sha256`
  `2ded9e1290113b95bbbb9526501130ae3429b7b50899547182f086b62d43ff40`.

### Reproduction

```powershell
# 1. Full opt-in gate: 40-layer export, then every production-width variant.
bash target/deepseek-v41-reference/run-v41-gate-batch.sh
# 2. Default V4.1 suite and the unchanged structural checks.
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/check_shader_provenance.py
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/check_push_constant_layouts.py
# 3. Record and independently recheck the qualification receipt.
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/qualify_deepseek_v41_scalable.py `
  --record --machine-info target/deepseek-v41-reference/scalable-machine-info.json
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/qualify_deepseek_v41_scalable.py
```

### Files and suite notes

- `hierarchos-vulkan/src/deepseek_v41/checkpoint.rs`: `read_bf16_word_range`,
  `read_f32_row_range`, `read_bf16_rows`, `read_f32_rows` and the shared
  `read_row_run`/`pack_bf16_words` helpers.
- `hierarchos-vulkan/src/deepseek_v41/model.rs`: `VocabularySource::Tiered`,
  `vocabulary_tables_fit`, the tiered embedding gather and head tiles, and
  `vocabulary_residency()`.
- `hierarchos-vulkan/src/deepseek_v41/model_production_tests.rs`: per-layer
  residency expectations and effective-tier reporting.
- The projection shader sources and their `.spv` files are byte-identical to
  their pre-change state; no other shader or `src` file was touched.

Full library suite on this checkout: **938 passed, 2 failed, 32 ignored**. Both
failures are `rwkv_channel_mix::tests::fused_channel_mix_*` dispatch-count
assertions (`left: 4, right: 1` and `left: 3, right: 1`); they reproduce with
every `deepseek_v41` test skipped and pass when run alone, so they belong to the
current tree's fused RWKV channel-mix work and are unrelated to V4.1.

## Shared tiered weight offload and storage reclamation (2026-10-26)

This pass did two things the previous pass left open: it reclaimed the storage
that had accumulated again, and it promoted the tiering the dedicated V4.1 graph
already used into a shared facility that every native architecture's loading path
now uses.

### Storage reclaimed

**17,161,607,788 bytes** (17.2 GB decimal) across 49,759 files. Free space on this
device went from **41,334,362,112** to **57,333,985,280** bytes; the peak during
the pass was 58,484,158,464 bytes before the regenerated fixtures and re-run
gates consumed about 1.1 GB of it.

| class | bytes | files |
| --- | ---: | ---: |
| Abandoned page-backed temp scratch | 8,089,624,992 | 770 |
| Regenerable generated packages | 8,664,684,239 | 66 |
| Stale PEFT work scratch | 361,120,870 | 43,619 |
| Superseded oracle priors | 46,177,687 | 5,304 |
| **Total** | **17,161,607,788** | **49,759** |

The abandoned scratch is the `np.memmap` backing for the production oracle's
embedding, FP32 head and packed block weights: **22** leaked
`deepseek-v41-packed-*` directories at 166,177,280 bytes each, one at
332,354,560 (two tensors), one empty, two `deepseek-v41-vocabulary-*`
directories at 2,647,654,400 and 1,323,827,200 bytes, and **70**
`hierarchos-causal-lm-vendor-*` fixtures at 2.6 MB each (see the fourth leak
below). The packages are the twenty-shard production package, the LoRA package,
the schedule-width package, and - twice - the small `structure`/`wide` packages
this pass wrongly deleted before restoring them, which is why that class exceeds
the packages' single-pass size by 34,632,975 bytes. The priors are 106 superseded
`*.prior-<ns>` oracle snapshots.

The `os temp scratch` figure also grew after the suite and the V4.1 gate re-ran,
because those runs minted more vendor fixtures and one V4.1 scratch directory that
the next sweep reclaimed. Free space therefore ends at **57,333,985,280** rather
than the 58,484,158,464 peak: the 17.0 GB was reclaimed, and about 1.1 GB was
then consumed by the regenerated fixtures, the re-run gates and Rust build
artifacts. Reported net delta from the starting point: **15,999,623,168 bytes**.

Retained: the **44,770,196-byte** scalable release checkpoint
(`deepseek_v41_joint_4141111`), every frozen CPU expectation under
`validation/fixtures/*.json`, and every verified gzip diagnostic archive. The
ledger accumulates per run rather than overwriting, so its totals are checkable;
`validation/summarize_scratch_cleanup.py` re-derives the class table above from it
and refuses to print if the classes disagree with the per-pass totals. Ledger:
`validation/release_parity/deepseek_v41/scratch-cleanup.json`.

### Four leak causes fixed, not just swept

The scratch was not merely abandoned by crashes; in two cases the cleanup could
not have succeeded even on a clean run.

1. `validate_deepseek_v41_production_model.release_scratch` swallowed every
   `OSError` from `shutil.rmtree`. On Windows a live memory map keeps the file
   name locked, so the deletion failed and the failure was discarded. It now
   drops its references, forces collection, retries, and **reports** any folder it
   could not remove instead of silently leaking it. `atexit` also releases.
2. A killed process never runs cleanup at all, so `sweep_stale_scratch()` now
   reclaims whatever a previous run left before the next run allocates anything.
   A folder still mapped by a live sibling process is left alone: the deletion
   fails and it is skipped. Verified by test: an abandoned folder is reclaimed, a
   folder with a live mapping is preserved, and it is reclaimed once the mapping
   is dropped.
3. `verify_peft_lora_strict.py` archived the previous oracle fixture into a
   `.prior-<ns>` sibling on **every** run with nothing pruning the archive, so
   snapshots accumulated one per run forever. The newest prior per family is now
   kept as continuation state and older ones are removed.
4. `hierarchos_causal_lm_tests::vendor_fixture()` minted a uniquely named
   fixture directory per call and nothing removed it afterwards: 70 directories,
   121.3 MB, one per suite run. It now sweeps leftover fixtures before
   allocating. A directory still held by a live process stays, because its
   removal fails and it is skipped. **Corrected in the next pass**: the sweep
   only matched the `-vendor-` spelling, so the larger `fixture()` family was
   never reclaimed and grew to 259 directories / 46 MB. See *The first non-V4.1
   consumer* below.

### The shared offload

`src/weight_offload.rs` names and accounts three tiers: **device** (a `GpuBuffer`),
**host** (a byte-capped tile cache) and **package** (the SafeTensors file, read
through the page cache).

The change that reaches the other architectures is the package tier. The common
host readers in `checkpoint.rs` - `read_f32_tensor`, `read_f64_tensor`,
`read_i64_tensor`, `read_u32_tensor`, `tensor_exists`, and the FP32-master dtype
validator - previously called `fs::read` and deserialised the **whole shard for
every tensor**. Fifty tensors meant fifty whole-shard reads and fifty header
parses. They now read only the requested byte range of only the requested tensor,
with shard headers parsed once and cached process-wide, keyed by file length and
modification time so a regenerated package is not served a stale header.

`single_tensor_image` was the enabling trick: it materialises a valid one-tensor
SafeTensors image for the requested tensor, so `SafeTensors::deserialize`,
`TensorView` and each reader's existing decode body are **unchanged**. Only the
number of bytes read changed. `validate_f32_tensor_names` needs no payload at all
and now reads none.

`plan_weight_residency` is the tier decision any loader can make before
allocating: `Resident` when the set fits beside what is already loaded, `Tiered`
otherwise. An unknown driver budget maps to `Tiered`, so a missing budget
degrades to the bounded path rather than to an out-of-memory failure.
`OffloadLedger` records package bytes read, host-cache bytes served, hits,
misses, evictions and peak occupancy, so a tier claim can be checked.

### Evidence

- Fifty tensors from one shard: `payload_reads == 50`, exactly the payload bytes
  read once, `headers_parsed == 1`; the pre-existing whole-shard path would have
  cost roughly forty times more. Asserted against the measured image size, not a
  constant.
- Row reads honour the tile budget and reuse cached tiles; a third 8 KiB tile
  against a 16 KiB budget evicts exactly one and peak occupancy stays at or below
  the budget; with retention disabled no read is ever served from the host tier.
- BF16/FP16 decode returns the exact values for representable inputs, and pins the
  widening contract the crate relies on: high 16 bits kept, low 16 zeroed -
  truncation, not round-to-nearest (`65504 -> 65280`, `65536 -> 65536`).
- `checkpoint`'s historical `f16_bits_to_f32` and the shared copy agree on **all
  65,536 bit patterns**, so the duplicated widening cannot drift.
- A missing tensor is `false` but an unreadable shard is still an error, so an
  optional-tensor probe cannot mistake a broken file for an absent one.

### Verification and the reverted-cleanup incident

The default suite references two generated packages that the first cleanup pass
deleted: `deepseek_v41_structure_4141101`/`_4141102` (six default tests) and
`deepseek_v41_wide_4141071`/`_4141072`/`_4141073` (one default test). Those seven
tests failed until the packages were regenerated from their recorded oracles.

- `validate_deepseek_v41_structure_model.py --write-checkpoints` regenerates the
  structure packages and confirms the frozen fixture still matches.
- `validate_deepseek_v41_wide_model.py --write-fixture` regenerates the wide
  packages; the frozen fixture's SHA256 is **unchanged**
  (`8f468d860c162fb08cddd554cbf85d5b108092a37d326460fdcd75dec018274f` before and
  after), so the expectations reproduce exactly.
- The **large** packages stay deleted: `production_4141090`, `lora_4141095` and
  `schedule_width_4141211` are used only by `#[ignore]`d tests.

Two mistakes were made and corrected here. An idempotency check of the reclaim
script revealed that its target list still named the `structure`/`wide` packages,
so re-running it re-deleted them and failed the same seven tests a second time.
The list now names only the opt-in-only packages, with the reason each kept
package must stay, and a re-run removes nothing (verified: `0 files, 0 bytes`).
Separately, the ledger's first pass was overwritten before it recorded history;
`validation/reconcile_cleanup_ledger.py` re-enters it and checks the
reconstruction against the totals that pass reported, which caught an off-by-one
in the packed-directory count on the first attempt. The ledger now accumulates
per run and `validation/summarize_scratch_cleanup.py` refuses to print if the
per-class totals disagree with the per-pass totals.

Suite: **981 tests, 947 passed, 2 failed, 32 ignored**; the nine new tests are the
only addition over the previous 972. Both failures remain
`rwkv_channel_mix::tests::fused_channel_mix_*` dispatch-count assertions, which
pass in isolation and reproduce with every `deepseek_v41` test skipped. CLI: **40
passed**. The V4.1 release gate re-ran on the frozen final state:
`deepseek_v41_default_package_scalable_release_strict_all_logits` passes in
508.03 s with **zero drift** on every step; an earlier run in the same pass took
509.79 s on the same zero-drift result.

### What is not done

The **device** tier is unchanged outside V4.1. Converting a generic consumer to
stream device tiles the way the V4.1 vocabulary, expert, Engram and projection
residencies do requires rewriting that consumer's resident-buffer path, which this
pass did not attempt. Training and full fine-tuning stay resident by arithmetic
necessity: a parameter whose full gradient table must be accumulated cannot be
served from a tile cache. `GpuBuffer` exposes only a whole-buffer `write_f32`, so
tiled device upload also needs a runtime change first.

## The first non-V4.1 consumer: Qwen4-Exp PLE

The heaviest single weight in the tree outside V4.1 is the Qwen4-Exp PLE n-gram
embedding table, `model.layers.N.ple.ple_embedding.ngram_embedding.weight`, which
the released configuration sizes in tens of GiB. Its loader already kept a source
descriptor rather than the decoded table, and its consumer is the PLE layer's
per-step row gather. That gather is now the first non-V4.1 consumer served by the
shared offload.

What changed:

- `weight_offload` gained `read_row_gather`, which groups requested row ids by
  host tile, pulls each tile once for the whole request, serves duplicate and
  repeated rows from that one read, and returns only the requested rows - and
  only those rows are uploaded as a device tile. `TensorDtype::from_safetensors`
  lets a resolved source hand its dtype to the bounded reader.
- `shared_package` keeps one `ModelPackage` per canonical file path for the
  process, so the header cache, the byte budget and the hit statistics belong to
  the process rather than to one gather. `Arc::ptr_eq` is the identity check.
- `decode_transformer_float_source_rows` - the single production caller, reached
  only from `HostQwen4Ple::gather_embedding_rows` - lost its private per-row
  reader. It keeps its rank-2 and bounds validation, then reads the tile-aligned
  bytes through the shared package and decodes them with the shared
  `decode_f32`. The same bytes at the same offsets widen through the same
  F32/F16/BF16 paths.

### Evidence

- The two strict Qwen4-Exp logits fixtures (`qwen4_exp_ple`,
  `qwen4_exp_mixed_ple`, unmasked and padded) produce **byte-identical** native
  reports before and after the conversion - same file sizes, same MD5s - and
  still report zero failing values against the Transformers reference at
  `atol=2e-7`, `rtol=0` (worst `2.24e-8`).
- The equality is not vacuous. A recorded negative control
  (`HIERARCHOS_TRANSFORMER_ZERO_GATHER_CONTROL`) makes the gather return zeros and
  moves those logits by up to `4.11e-3`, so the gather really is on the logits
  path; it prints to stderr rather than failing silently.
- Four-token PLE generation, cached and full-prefix, matches Transformers to
  `1.86e-8` (`verify_qwen4_exp_generation.py`, both families).
- A unit test decodes the same 4,096x1,024 table through both readers and
  requires exact equality, then pins the batching: five scattered rows in one
  8 MiB tile cost **one** package read, `rows_read == 2,048` (bounded by the tile,
  not the request), and a gather that needs the second tile reads it once and
  then reuses both.
- `weight_offload` carries three new tests of its own: tile-once batching with
  caller order and duplicates preserved against a table forty times the read,
  out-of-range and rank-1 rejection, and one shared handle with one header parse
  across callers.

### A leak class this pass introduced, and one it found

The new tests mint real SafeTensors packages, and the first version of them left
every one behind: 81 directories, 26 MB. `Scratch` in the `weight_offload` tests
and `PleOffloadFixture` in the transformer test now remove their directory from
`Drop`, so the panic path cleans up too; verified by running both suites and
observing the count unchanged at 81 / 81.

The sweep added in the previous pass was also incomplete. It matched
`hierarchos-causal-lm-vendor-*`, but `fixture()` mints
`hierarchos-causal-lm-<pid>-<nanos>`, so the larger family was never reclaimed:
**259 directories, 46 MB**. The sweep now matches the shared prefix - and, more
importantly, only removes directories quiet for at least an hour. Sweeping by
prefix alone deletes fixtures belonging to *other tests in the same parallel
run*, which is a failure mode the previous pass's liveness argument ("its removal
fails") does not cover, because a fixture between steps holds no handle. Verified
both ways: a backdated directory is reclaimed, a fresh one is preserved, and a
full parallel suite run leaves 6 fresh directories / 9.5 MB that the next run
reclaims instead of 259 / 46 MB that nothing ever did.

### Verification

Suite: **985 tests, 951 passed, 2 failed, 32 ignored**; the four new tests are the
only addition over 981. Both failures are the same
`rwkv_channel_mix::tests::fused_channel_mix_*` dispatch-count assertions, which
pass in isolation. CLI: **40 passed**. The frozen V4.1 packages and shaders are
untouched, so the V4.1 release gate is unaffected by this change.

## The generic output head streams device tiles (2026-10-26)

The previous pass converted the **package** tier and the first non-V4.1 consumer's
row gather. This pass converts the first **device**-tier consumer outside V4.1:
the vocabulary-sized output head that every dense family shares. It is not the
largest weight in a package, but it is the one whose device cost scales with the
vocabulary instead of the model width (2.65 GB resident for a 129,280x5,120 FP32
head, 10.6 GB as a trainable parameter with gradient and both AdamW moments), and
the only one whose rows are consumed independently - each logit column is its own
dot product against the hidden row. That is what makes a row tile a complete unit
of work and the head streamable where a weight reduced along its columns is not.

### What was already on disk, and what was missing

The interrupted work had landed the machinery: `TieredLmHead` (content-verified
package adoption, a bounded row-read through the shared `ModelPackage`),
`TrainableParameter::tiered` with `read()` materializing the whole table through
the same tiles, the shared `record_linear_projection` dispatch both the resident
and tiled widths now use, and fail-closed guards on the backward/optimizer paths.
Four things were missing and are added here:

- **No entry point.** Nothing called `adopt_tiered_source`, so no graph could
  actually reach the tier. `VulkanTransformer::stream_lm_head_from_package` is
  that entry point, and `--stream-lm-head` exposes it in the logits binary.
- **Tile buffers were sized for the nominal tile.** `tile_plan` merges a
  remainder below the 16-column minimum into the preceding tile, which can widen
  a tile to 8,207 rows; the weight, output and bias tiles were allocated for
  8,192. A vocabulary whose remainder lands in that window would have copied past
  the end of its output tile. They are now sized by
  `TIERED_LM_HEAD_MAX_TILE_ROWS`, and the new test's 16,387-row vocabulary
  exercises exactly that case (second tile 8,195 rows).
- **Training state could still be acquired.** `ensure_training_state` now refuses
  a streamed parameter, so `enable_lora`/`enable_base_training` cannot allocate a
  gradient and optimizer state for a head that has no backward path.
- **The DoRA dropout branch still referenced the pre-extraction `push`** left
  behind by the `record_linear_projection` refactor; only that error path was
  fixed, the dispatch itself is byte-identical to what it recorded before.

### Behavior

Streaming is opt-in and additive; `from_hf_package` and every other constructor
keep the head resident and the default path is unchanged. The conversion proves
the package tensor is the one the graph loaded by tile-by-tile equality with the
resident values (refusing an ambiguous package rather than guessing a name),
releases the weight, gradient and both moment allocations, and then projects the
vocabulary in 8,192-row tiles - each through the kernel the full-width dispatch
would have selected, each tile's logit columns copied into the logits as an FP32
buffer move, so the tile boundary never enters the arithmetic. The output bias
stays resident (one value per column, ~0.5 MB at 129,280 tokens) and is still
bound to the kernel per tile, which is where the resident dispatch seeds its
accumulator. `read()` still answers with the complete table, so parameter export
and the round-trip comparisons work on a streamed graph, and training, a
training-mode forward, PEFT replication, acquiring training state and attaching an
adapter all fail closed.

### Verification

- `streamed_output_head_matches_resident_logits`: a two-tile untied-head package
  (16,387 rows, three-row remainder merged into the second tile) produces
  **byte-identical** logits to the same package loaded resident, unbiased and with
  a biased head (`assert_eq!` on the whole logit vector). The same test pins that
  nothing streams until asked for, that the plan is the expected two tiles, that
  the merged tail really is wider than the nominal tile, that `read()` still
  returns the exported table, and that PEFT forking and a training-mode forward are
  refused.
- End to end through the binary, on the Falcon-H1 CLI fixture (untied head,
  `tie_word_embeddings: false`): the resident and `--stream-lm-head` reports agree
  exactly, 369 logits, `max_abs = 0.0`. Exporting from the streamed graph
  reproduces the package - all 37 tensors, `lm_head.weight` included, are bitwise
  equal to the source checkpoint, and a resident load of the exported directory
  returns the same logits - so the lazy source is not a one-way conversion. The recorded
  `.falcon-h1-fixtures/cli-trained/logits.json` differs from *both* runs by
  0.0255, which is a property of that older fixture pair and not of the
  conversion - streaming is opt-in, and the default run reports the same gap.
- Suite: **987 tests, 953 passed, 2 failed, 32 ignored**. Both failures are the
  known `rwkv_channel_mix::tests::fused_channel_mix_*` dispatch-count assertions,
  which expect the fused route while this machine's Intel vendor id selects the
  composed one. CLI: **40 passed**. No shader, fixture or frozen package was
  touched.

### What remains

A family whose logits come from the tied embedding table (no separate `lm_head`)
has no streamed tier and is refused with a clear error rather than served a
placeholder; its projection is the embedding's own `SharedLmHeadParameter`. The
other generic consumers still load resident, and training and full fine-tuning
remain resident by arithmetic necessity.


## Tied embedding/output head package streaming (2026-10-26)

Continued the unfinished tied-head step from the generic output-head work above.
`stream_lm_head_from_package` and the existing logits binary's `--stream-lm-head`
now support a graph whose output projection uses its input embedding table.

### Implementation

- `SharedLmHeadParameter` can hold an immutable shared `ModelPackage` source.
  Token lookup and tied output retain one parameter identity. Conversion proves
  content equality before rebinding either edge, then releases the full resident
  table, gradients, moments and table-sized readbacks. Lookup reads only requested
  package rows, preserving duplicates, caller order and the stored padding row.
- The tied projection uses the resident ordinary linear kernel followed by a
  separate bias add. Untied projection keeps its existing accumulator-seeded bias
  path. Full forward, cached decoding and DoLa hidden projection reach the tier.
- Fixed the shared output-tile allocation: it now holds
  `max_graph_rows * max_tile_rows`, rather than `hidden_width * max_tile_rows`.
  This fixes insufficient output capacity when token rows exceed hidden width,
  and avoids allocating a hidden-width-sized output tile for single-token graphs.
- Raw shared resident-buffer access now returns `Result`; training registries
  propagate errors instead of binding a placeholder. Tied streaming refuses an
  existing PEFT adapter or FP16 execution mirror before mutation. Later training,
  PEFT attachment, parameter replacement and optimizer access fail explicitly.
- Export materializes the canonical package table, preserving tied checkpoint
  topology. Resident constructors and the existing resident PEFT/training paths
  retain their behavior.

### Verification performed in this pass

The device reports **AMD Radeon Graphics**, PCI `1002:15BF`, driver
`32.0.31007.6002`. The earlier historical Intel explanation for RWKV dispatch
counter failures does not describe this device; no cause for those historical
failures was established in this pass. The full library suite was not rerun.

- `cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib streamed_output_head -- --test-threads=1`:
  **2 passed**. The new tied regression uses 16,387 vocabulary rows, with tiles
  `(0,8192)` and `(8192,8195)`, and a 40-row batch exceeding the 16-wide hidden
  state. All **655,480** full-forward logits match the resident graph exactly.
  It also checks padding/repeated/cross-tile token gathers, biased projection,
  cached prefill and three decode steps, DoLa projection, export/resident reload,
  immutable-parameter guards, and failed conversion leaving resident identity intact.
- `--lib tied_`: **19 passed**, including tied optimizer identity, deterministic
  embedding gradient accumulation, existing PEFT tied clone/training tests, and
  the DeepSeek V4.1 wide projection/top-k CPU comparison.
- `--lib modules_to_save`: **13 passed**; `--lib weight_offload`: **13 passed**.
  These groups overlap the tied group; the counts are not additive.
- `--bin hierarchos-vulkan-transformer-train`: **8 passed**.
- Built `hierarchos-vulkan-transformer-logits` and ran resident, streamed/export,
  and resident reload on the existing tied `gpt2-strict/base` fixture. All **128**
  logits are bit-identical between native paths. All **28** exported tensors
  match source shapes, dtypes and payload bytes.
- Fresh eager CPU PyTorch (`torch 2.14.1+cpu`, one thread) and the existing frozen
  CPU oracle each differ from streamed native logits by at most
  **2.9802322387695312e-8**, strictly below **2e-7**, **rtol=0**. Native JSON values
  were rounded back to FP32 for comparison. Fresh CPU versus frozen CPU drift
  is `1.862645149230957e-8`. No oracle was regenerated or edited.
- Source checkpoint SHA256:
  `55445934f22d96b4f9965a2bd537c44a955d27b70f11db74b0af6e598ec5a459`;
  frozen `hf.json` SHA256:
  `b5815de3051a6a6e3dd2292516ee5e12f4a7e89f01b850c90f266681e2dc9c87`.
  Both were checked unchanged after qualification.

Reports, CPU reference, exported tiny checkpoint and streaming test log are in
`target/streamed-tied-head-check/`; `qualification.json` records the measured
parity and source hashes. The test's temporary package directories were removed;
no large checkpoint was generated. No shader or frozen fixture changed in this pass.

### Remaining scope

This generic conversion is still explicit and happens after resident loading;
it reduces steady-state device residency, not initial loading memory. Streaming
training/PEFT state, generic load-time head tiering, and tiering the remaining
attention/MLP weights are not implemented. These limits concern the generic head
tier; they do not revoke the earlier dedicated DeepSeek V4.1 inference-LoRA
qualification. No CUDA qualification is required or claimed.


## Deferred head loading and measured allocation bounds (2026-10-26)

The generic head tier can now be selected before full head loading in the GPT-2/
GPT-SW3 and Llama-layout loaders. The prior phase removed steady-state residency;
this phase removes their initial full head allocation as well.

### Implementation

- Added `VulkanTransformer::from_hf_package_with_streamed_lm_head`. Architecture
  loaders retain a `HostPackageLmHead` descriptor instead of decoding the full
  head. Tied token embeddings use that same immutable package identity. Untied
  input embeddings remain resident in this phase.
- `HostFloatTensorSource` retains the physical SafeTensors name through wrapper
  rebasing. Head validation and tied-alias comparison use bounded package row
  reads, including indexed shards. The resident alias contract is preserved:
  signed zeros compare equal, and the embedding stays canonical.
- Direct construction creates tile buffers and one-element placeholders without
  full head weight, gradient or AdamW allocations. The new allocation regression
  initially exposed a remaining full-table `lm_weight_grad` scratch buffer. It is
  now omitted for directly streamed graphs and released by post-load conversion.
- Added allocator lifetime high-water statistics (`largest_buffer_bytes` and
  `peak_live_buffer_bytes`) so validation includes buffers released before the
  measurement. These count logical Vulkan buffer bytes, excluding driver arena
  slack; they do not measure process RSS or driver heap usage.
- The existing logits `--stream-lm-head` flag uses the new constructor when no
  adapter arguments are present. Other loader families retain post-load
  conversion; adapter CLI combinations retain their adapter-first conversion
  order. Default resident constructors retain their behavior.
- Both streamed topologies reject later training and new PEFT attachment before
  mutation. This adds no streamed training or PEFT state support and does not
  revoke the dedicated DeepSeek V4.1 inference-LoRA qualification.

### Qualification on this machine

The existing AMD Radeon Graphics adapter was used. The two new Rust tests pass:
`load_time_output_head_streaming_bounds_allocations_and_preserves_logits` and
`load_time_output_head_gpt2_aliases_and_invalid_payloads_fail_before_allocation`.

The large fixture has **65,539 vocabulary rows, hidden width 32 and eight token
rows**, preserving an eight-tile vocabulary with an absorbed three-row tail.
There are **six storage/topology cases**: tied/untied, each in F32, BF16 and F16.
Both BF16 cases additionally use wrapped physical names and two indexed shards.
Each case compares **524,312 logits** to the same package loaded resident
(**3,145,872** values across the matrix), with exact equality. Cached prefill,
incremental decode and export/resident reload also agree. Head reads remain
bounded and the shared host cache remains under its byte cap.

Fresh device instances measure loading before the first forward:

| Topology | Full table bytes | Resident loading peak | Deferred loading peak | Largest deferred buffer |
| --- | ---: | ---: | ---: | ---: |
| Tied | 8,388,992 | 71,811,328 | 8,109,732 | 2,097,248 |
| Untied | 8,388,992 | 105,367,312 | 58,443,692 | 8,388,992 |

These values hold for all three storage dtypes. The untied path still contains
its resident input embedding. The tied largest-buffer bound proves that no full
vocabulary table allocation occurred at any point in construction. Loader tests
also assert that the deferred parameter has no decoded head vector and no
optimizer state. Mismatched tied aliases, non-finite head payloads and invalid
head shapes fail before any device buffer allocation.

End to end through the rebuilt logits binary, four existing frozen fixtures
were checked: **GPT-2, Llama, Qwen2.5 GQA and Phi-3**, 128 logits each. GPT-2/Llama
exercise tied deferred heads; Qwen2.5 exercises an untied deferred head; Phi-3
exercises the retained post-load fallback. All **512 logits** are bit-identical
between resident, streamed and reloaded native runs. All **90 exported tensors**
match source shape, dtype and payload bytes. Fresh CPU PyTorch and the frozen CPU
references each differ from native by at most **2.9802322387695312e-8**, strictly
below **2e-7**, with **rtol=0**. The reference uses `torch 2.14.1+cpu`, one thread,
eager attention, and `C:/Users/User/transformers/src`. No frozen oracle or source
checkpoint was regenerated or edited; SHA256 values and the tested binary hash
are recorded in the qualification report.

Targeted regressions passed:

- New load-time tests: **2**; existing streaming tests: **2**.
- Tied-parameter tests: **19**; PEFT saved-module tests: **13**.
- Shared offload tests: **13**; Vulkan tests: **33**.
- Transformer training CLI tests: **8**.

Groups overlap. `qualification.json` records the unique test count and every
log path. The full library suite was not rerun. No shader or frozen fixture was
edited in this phase. Temporary large fixture packages were removed on success
and failure; only small CLI exports, reports and validation logs remain.

Evidence: `target/load-time-head-check/qualification.json`,
`rust-tests-final.log`, `cpu-qualification.log`, the per-group regression logs,
and `qualify.py`. The latter reruns the four CLI/CPU/source-byte comparisons
against existing fixtures without modifying frozen expectations.

### Next scope

The largest remaining generic vocabulary allocation is the untied input
embedding, which still owns resident training buffers in the current inference
constructor. Stage that table independently, extend deferred construction to
the other architecture loaders, and then address other parameter consumers.
Streaming training/PEFT state and generic attention/MLP weight tiering remain
unimplemented. The dedicated DeepSeek V4.1 defaults remain unchanged.


## Untied input embeddings stream independently (2026-10-26)

Continued the remaining vocabulary step from the prior phase. The streamed
constructor now stages both input and output tables in the GPT-2/GPT-SW3 and
Llama-layout loaders, preserving distinct parameters when the checkpoint is
untied. No new flag is needed: the existing streamed-constructor and adapter-free
`--stream-lm-head` path select it.

### Implementation

- `HostPackageLmHead` retains an independent `embedding_source` for untied graphs.
  Both resident host vectors remain empty. Each physical tensor name survives
  wrapper rebasing and points to its own SafeTensors source, rather than being
  selected by equal values or rebound to the output head.
- A common bounded source validator checks input geometry, dtype and every
  logical row before either consumer allocates device buffers. Input tables no
  longer allocate the full resident weight, gradient, two AdamW moments, shared
  readback or embedding-gradient readback. Only requested rows are uploaded for
  lookup; the existing projection tile still serves the output head.
- `SharedLmHeadParameter::from_tiered_package_prefix` represents a logical text
  vocabulary inside a larger physical table. `gather_package_rows` enforces its
  logical bound, and `ModelPackage::read_f32_row_prefix` materializes only that
  logical parameter for export. Mllama's eight reserved rows are preserved by the
  existing wrapper export merge, and are not addressable as text token rows.
- Exact-shape shared-head construction remains strict; prefix geometry is used
  only for the independent input source. Unsupported mutations identify the
  actual package-backed vocabulary parameter in their error.
- Default resident construction, explicit post-load head conversion and adapter
  CLI ordering retain their behavior. This phase does not add streamed training
  or PEFT state support. Dedicated DeepSeek V4.1 defaults and its earlier
  inference-LoRA qualification remain unchanged.

### Qualification on this machine

The rebuilt native binary reports AMD Radeon Graphics. The six load-time cases
again use **65,539 vocabulary rows, width 32 and eight token rows**, with tied
and untied topologies in F32/BF16/F16 storage. Both BF16 cases now place the
wrapped input table, output table and body in **three different indexed shards**.
Every case checks **524,312 forward logits** by their FP32 bit patterns, totaling
**3,145,872** identical values. Cached prefill, decode and export/resident reload
also match. Input gathers are observed in the source reader's ledger, including
repeated, padding, boundary and high token ids; both physical file caches stay
within their byte caps. An intentionally tied negative control changes logits
by more than `1e-3`, so the independent-table check is observable.

Fresh-device loading measurements (logical live Vulkan buffer bytes):

| Topology | Resident loading peak | Previous deferred peak | Current deferred peak | Largest current buffer |
| --- | ---: | ---: | ---: | ---: |
| Tied | 71,811,328 | 8,109,732 | 8,109,732 | 2,097,248 |
| Untied | 105,367,312 | 58,443,692 | 8,109,752 | 2,097,248 |

The previous-phase measurements are retained in
`target/load-time-head-check/qualification.json`. The new untied path removes
another **50,333,940 bytes**: six canonical 8,388,992-byte table-sized buffers are
replaced by three one-element placeholders. In all six cases, the lifetime
largest buffer remains below one full canonical FP32 table. The metrics include
buffers released before inspection, exclude driver arena slack, and do not claim
process RSS or physical VRAM measurements.

Further checks:

- GPT-2 retains two independent parameters even with identical table values;
  `tie_word_embeddings=false` and both serialized tensors survive export.
- A NaN in an unused input row is rejected before any GPU storage-buffer
  allocation, as are the existing malformed output/alias cases. Reserved physical
  rows stay outside the logical lookup range. The logical prefix reader does not
  decode an excluded non-finite tail, while full materialization still rejects it.
- A complete tiny Mllama text package checks logical lookup, independent output,
  preservation of all eight reserved rows, and streamed export/reload.
- Four unchanged frozen fixtures (GPT-2, Llama, Qwen2.5 GQA, Phi-3) provide **512**
  CPU comparisons. A new untied GPT-2 checkpoint, generated only under the evidence
  directory, adds **128** fresh CPU comparisons and independently owned input and
  output parameters. Native resident, streamed and reloaded outputs are bitwise
  identical across all **640** logits. All **119** exported tensors match source
  shape, dtype and payload bytes.
- Maximum drift versus fresh eager CPU PyTorch is **2.9802322387695312e-8**; the
  frozen CPU maximum is the same. The newly generated untied GPT-2 case reaches
  **2.2351741790771484e-8**. All pass **strictly below 2e-7, rtol=0**, using
  `torch 2.14.1+cpu`, one thread and the local `C:/Users/User/transformers/src`.
  Frozen source checkpoint and oracle SHA256 values remain unchanged. No expected
  logits were repinned, no shader was edited, and no CUDA parity is claimed.

Targeted test groups: load-time cases **2**, reserved-input case **1**, shared
reader/offload **14**, existing streaming **2**, tied parameters **19**, PEFT saved
modules **13**, Vulkan **33**, training CLI **8**. Groups overlap: **87 unique
checks passed**, recorded with their individual log paths in the report. The
full library suite was not rerun.

Evidence: `target/untied-embedding-check/qualification.json`,
`load_time_output_head.log`, `streamed_input_embedding.log`, `weight_offload.log`,
`cpu-qualification.log`, per-group regression logs and `qualify.py`. The script
reproduces native/CPU comparisons from the unchanged frozen fixtures and its
separate tiny generated untied GPT-2 checkpoint. Large temporary packages were
removed on both success and failure; the retained CPU fixtures are small.

### Next scope

Extend deferred vocabulary construction to the other architecture loaders;
those currently retain the post-load head fallback and resident input table.
The explicit post-load converter still stages only the untied output head.
Other generic attention/MLP weights and streamed training/PEFT state remain
unimplemented. No change to the dedicated DeepSeek V4.1 execution kernels was
needed for this phase.
## Frozen vocabulary offload across adapter training and PEFT (2026-10-26)

Every earlier streaming phase ended with the same refusal: a graph whose
vocabulary was served from package tiles could run inference but could not be
trained or adapted, because the head's backward pass needed the table resident.
This phase removes that refusal where it is removable - the vocabulary stays
frozen, which is what LoRA training does anyway - and then requalifies every
architecture through the CPU PEFT oracle with the streaming switch on.

### Implementation

- **Frozen-weight input adjoint.** `linear_input_grad_streamed.comp` (new
  SPIR-V, embedded by `include_bytes!` and reproduced by the shader provenance
  check) accumulates `dX` for a tile of frozen weight rows. Middle tiles are
  8,192 rows, a multiple of the resident kernel's 32-term inner block, and each
  dispatch starts from the previous FP32 accumulator (`accumulate` in the push
  constant) rather than re-associating across tiles, so every FMA keeps the
  resident order and a tile boundary cannot enter the arithmetic. The first tile
  initializes; the rest accumulate.
- **Per-tile staging retirement.** Tile uploads are queued into the batch's
  host-visible arena, and that arena was released only at submission - so a
  streamed projection retired its *device* weight tiles but kept the whole
  table's *upload bytes* alive until the caller submitted. Both the forward
  (`record_projection`) and the new input adjoint now `submit_and_restart()`
  after each tile. The surviving batch carries the recorded dispatch, barrier
  and bind counts (those are structural, and a validation script comparing them
  must not read a change of submission granularity as a regression) but none of
  the staging.
- **Training, resume and export at the parameter tier.** `TrainableParameter`
  can hold a `TieredLmHead` source with a one-element placeholder; `read()`
  materializes through the same bounded tiles the projection uses, so export and
  round-trip comparison still answer with the whole table. Training-state
  acquisition and PEFT replacement/forking of the frozen parameter stay refused
  with explicit errors - a parameter whose full gradient table must be
  accumulated cannot be served from a tile cache - while ordinary adapters on
  the other modules train normally against it.
- **Bounded fingerprint and checkpoint hashing.** `lora_base_fingerprint` hashes
  a streamed vocabulary in 8,192-row chunks read through its own package source
  instead of materializing `parameter_values` (which deliberately rejects PLE
  tables and has no resident vector for a streamed head), and the LoRA
  training/resume state hashes streamed vocabulary rows the same way, so neither
  path allocates a table-sized host buffer.
- **CLI and oracle switches.** `--stream-lm-head` is available on
  `transformer_logits`, `transformer_parity` and `transformer_train`. The PEFT
  oracle appends it when `HIERARCHOS_PEFT_STREAM_LM_HEAD=1`, so one oracle drives
  resident and streamed runs over identical fixtures.
- **Bounded package readers.** `read_f32_tensor`, `read_f64_tensor`,
  `read_i64_tensor`, `read_u32_tensor`, `tensor_exists` and the FP32-master dtype
  validator now read one tensor's payload instead of deserializing the whole
  shard per request; a loader asking for many tensors no longer pays the shard
  repeatedly.
- **Measured bounds.** `VulkanMemoryStats` reports `largest_buffer_bytes` and
  `peak_live_buffer_bytes`, so deferred-loading claims are measurements rather
  than inferences.
- **Test hygiene.** The causal-LM suite sweeps hour-old temp fixtures before
  minting new ones; uniquely named fixture directories had been accumulating
  across runs.

### Behavior and what is still refused

A streamed vocabulary can now carry LoRA/rsLoRA training with dropout, resume
from an optimizer checkpoint and export its adapter. Still refused, explicitly
and before mutation: training or acquiring training state for the streamed table
itself, PEFT replacement or `modules_to_save` of it, a training-mode forward of
the streamed base, and a projection whose bias or geometry disagrees with the
package source. Resident construction remains the default; nothing streams
unless asked, and the dedicated DeepSeek V4.1 kernels were not touched.

### Qualification on this machine

- **Streamed suite on the final tree: 29 passed, 14 ignored.** Includes the V4.1
  all-tables, all-depth, 384-expert and schedule cases - each printing drift
  exactly `0.0` - plus the tied/untied head, load-time and training regressions.
- **Streamed versus resident training is bit-exact.**
  `streamed_vocabulary_lora_training_resume_matches_resident` asserts, per bit,
  forward logits, losses, the head's adjoint, every LoRA gradient value,
  post-step logits, the base fingerprint, resume state (step, dropout seed),
  resumed-versus-resident logits, and adapter export payloads - for tied and
  untied vocabularies, F32/BF16/F16 storage.
- **The new measurement pins the staging fix.** A two-tile, 16,387-row head
  uploads 2,097,536 bytes across its tiles; the batch that survives the
  projection still reports the tile dispatches and reports `uploaded_bytes()`
  of `0`, so no part of the table waits in host-visible memory for a later
  submission.
- **CPU PEFT oracle matrix: 32 of 32 families pass**, every family run with
  `HIERARCHOS_PEFT_STREAM_LM_HEAD=1`. Worst `peft_max_abs` is
  `6.3981552111958e-8` (`qwen2_5_sliding_tied`), strictly below `2e-7`,
  `rtol=0`. Covered: GPT-2, Llama, Mistral, Mixtral, Qwen2.5 (GQA and
  sliding/tied), Qwen3-Next, Qwen3.5 (full/linear/mixed/MoE), Qwen4-Exp
  (linear/QSA/PLE/mixed PLE), Phi-3, Phi-4 text, DeepSeek V4, Kimi K2.5, Kimi K3
  (KDA, MLA, mixed), GPT-OSS, SmolLM3, Mistral 4, MiniMax M2/M3/M3-dense,
  Gemma 3/4, Falcon-H1 (silu, gated before/after).
- **The retirement is arithmetic-neutral.** The 29 families that had already run
  before the per-tile change produced *identical* oracle values afterwards -
  base, PEFT, gradient, two-step AdamW, frozen-base and resume fields all equal -
  which is why the matrix could be rerun rather than re-derived.
- **DeepSeek V4.1 production width.** The native ignored test now passes: 1,024
  adapted logits with worst drift `0.0` against the pinned CPU fixture, adapter
  delta `2.543889`, and resident/streamed/export logits `assert_eq!`-identical.
  The CPU oracle re-verified that the regenerated 241 MB package reproduces its
  own arithmetic and reported `PASS lora-prefill: 0` and `PASS lora-decode: 0`.
  The earlier failure was only a missing package: the storage reclamation had
  removed it after the fixture was pinned, and the oracle regenerates it.
- **V4.1 adapter I/O: 5 focused tests pass** - capability stays fail-closed for
  replacement modules, target suffix semantics, checkpoint spellings, IEEE half
  decoding and unmerged export preserving factors, patterns and rsLoRA.
- **Deferred-loading bounds re-measured:** tied 8,109,732 bytes, untied
  8,109,752 bytes, largest single request 2,097,248 bytes - unchanged by this
  phase.
- **Library suite on the final tree: 960 passed, 2 failed, 32 ignored.** Both
  failures are the pre-existing `rwkv_channel_mix::tests::fused_channel_mix_*`
  dispatch-count assertions (this machine's vendor id selects the composed route
  rather than the fused one); they failed before this phase and are unrelated to
  it.

### Kimi-K3 oracle source

The PEFT oracle's Kimi-K3 families instantiate Moonshot's released remote code,
which the earlier storage reclamation had removed from the Hugging Face cache.
It was restored into `target/kimi-k3-remote` from `moonshotai/Kimi-K3` at
revision `c5d1dd4` (the revision the K3 audit pins, `*.py` only, 254 KB) and the
oracle reads it through `HIERARCHOS_KIMI_K3_REMOTE_DIR`. Without that directory
the three families fail with a `KeyError` in the oracle's model map, which is a
missing oracle input, not a native regression.

### Reproduction

```powershell
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib streamed_vocabulary_lora_training_resume -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib streamed_output_head -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_lora_production_width -- --ignored --nocapture
.\.venv-vulkan\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('moonshotai/Kimi-K3', revision='c5d1dd4', allow_patterns=['*.py'], local_dir='target/kimi-k3-remote')"
$env:HIERARCHOS_KIMI_K3_REMOTE_DIR = (Resolve-Path target/kimi-k3-remote)
.\.venv-vulkan\Scripts\python.exe target/peft-offload-completion/qualify.py
```

The matrix driver sets `HIERARCHOS_PEFT_USE_PREBUILT=1` and
`HIERARCHOS_PEFT_STREAM_LM_HEAD=1` per family and writes each family's report
under the isolated `offload-completion` fixture root, so no earlier evidence
directory is overwritten.

### Evidence

`target/peft-offload-completion/`: `cpu-matrix.json` (all 32 families),
`cpu-matrix-full.log`, `cpu-matrix-partial.json` (the pre-retirement run it is
compared against), `streamed-bounded.log` and `streamed-verified.log` (the final-tree rerun of the same filter), `load-time-bounded.log`,
`v41-lora-production-native.log`, `v41-cpu-oracle.log`, `v41-adapter-io-final.log`,
`library-suite-verified.log` and `shader-audit.json` (source, embedded and
rebuilt SHA-256 identical for `linear_input_grad_streamed`).

### What remains

Untied input staging in the explicit post-load converter and deferred vocabulary
construction in the remaining architecture loaders, as before. The remaining
generic attention and MLP weights still load resident. A *trainable* vocabulary
cannot stream - its full gradient table and both AdamW moments are the
arithmetic, not an allocation choice - so streaming it is refused rather than
approximated. Full released DeepSeek V4.1 geometry remains unexecuted on this
machine; the accepted scalable-checkpoint criterion is still the qualification
basis here.

## Adapter-training foundation: terminal adjoints and native AdamW state

Scope of this phase: start the V4.1 backward graph from the loss and qualify the
optimizer half of LoRA adaptation. This phase does **not** make V4.1 adapters
trainable end to end: the block adjoints (hyper-connections, MoE, attention,
Engram) do not exist yet, so every training entry point stays fail-closed. What
exists now is the part of the chain that turns logits into `d loss / d x` at the
top of the last block, plus an AdamW state that consumes factor gradients,
checkpoints and resumes.

### Loss-side adjoints (`src/deepseek_v41/terminal.rs`)

Three stages, each in the arithmetic its forward kernel was qualified with:

1. `transformer_deepseek_v41_cross_entropy.comp` -- loss and `d loss / d logits`.
   The loss is staged exactly as CPU `F.cross_entropy` stages it:
   `(x - max) - log(sum)`, negated, with `exp` from the SLEEF `u10` polynomial,
   the pinned CPU scalar log, and the AVX2 four-interleaved eight-lane sum the
   CPU sum kernel uses. Two ulp-level details were found and fixed against the
   fixture rather than assumed:
   - `torch.logsumexp` and `F.cross_entropy` disagree by one ulp because they
     associate the shift differently; the kernel follows `F.cross_entropy`.
   - expressing the result as a negated subtraction let the compiler reassociate
     it into the `logsumexp` staging (one ulp away at loss magnitude ten). The
     loss is now a single exact-FMA statement, so contraction cannot reintroduce
     the wrong order.
2. Frozen head backward through the existing `linear_input_grad` kernel.
3. `transformer_deepseek_v41_rms_norm_input_grad.comp` -- input gradient of the
   final norm, walking `rstd -> variance -> sum` in the autograd order, and
   recomputing `rstd` from `x` with the forward reducer and pinned rsqrt so it
   cannot disagree with `rms_norm.comp`.

`validation/validate_deepseek_v41_terminal_grad.py` pins CPU PyTorch's own
autograd through `x -> rms_norm -> head -> cross_entropy` as FP32 bit patterns.
Measured (`cargo test --lib deepseek_v41::terminal -- --nocapture`), gate
`<2e-7`, `rtol=0`:

| value | worst drift |
| --- | --- |
| row loss | 0 |
| mean loss | 0 |
| `grad_logits` | 2.98023223876953125e-8 |
| `grad_hidden` | 5.96046447753906250e-8 |
| `grad_x` (end-to-end through both stages) | 1.19209289550781250e-7 |

### Optimizer, resume and export (`src/deepseek_v41/train.rs`)

`DeepseekV41AdapterTrainer` copies an adapter's factor pairs into mutable FP32
masters, owns both AdamW moments per factor, the step count and the
hyperparameters, and refuses a step whose gradient set is incomplete or carries
a duplicate, non-finite or wrong-shaped entry. The update reproduces
`torch.optim.AdamW`'s single-tensor CPU path operation by operation (decoupled
decay first, `mul_`+`add_` moments, f64 bias corrections rounded once, final
`param - step_size * (m/denom)`).

`validation/validate_deepseek_v41_adapter_training.py` builds a two-module PEFT
adapter (`layers.1.attn.wq_b` rank 3, `layers.2.ffn.shared_experts.gate_proj`
rank 2), runs three pinned AdamW steps with `foreach=False`, and records every
factor and moment as FP32 bits. Measured
(`cargo test --lib deepseek_v41::train -- --nocapture`), gate `<2e-7`,
`rtol=0`: worst drift across all three steps, both modules and all four moment
tensors is **5.96046447753906250e-8**.

The same test also proves the resume contract: two steps, export the trained
adapter and save the optimizer state, reload both from disk, take the third step
and compare with the uninterrupted run. Factors and moments match **bit for
bit**, and the exported adapter reloads through the normal loader with the
trained factors. The checkpoint records the source adapter's checksum as
provenance but is tied to the resumed adapter by target list and geometry, since
a trained adapter's own checksum necessarily differs.

The gradients in that fixture are pinned rather than produced by the graph --
that is the missing piece, and the fixture is labelled accordingly.

### Checks

- 130 default V4.1 tests pass (`cargo test --lib deepseek_v41`), including the
  new terminal-adjoint and adapter-optimizer gates.
- All 520 recorded shader artifacts reproduce byte-for-byte under `glslang`
  16.6.0 (`validation/check_shader_provenance.py`), including the two new
  kernels.
- `cargo fmt` clean for the new files; the surrounding tree keeps its existing
  formatting exceptions.

### What remains for V4.1 LoRA training

> **Closed.** Every item below was implemented in the phases that follow, and
> the whole chain is qualified end to end by the final entry of this audit
> (*DeepSeek V4.1 PEFT training closure*). The list is kept as the build order.

Backwards from the loss, in dependency order: Hyper-Connection post update,
MoE (gate, routed/shared SwiGLU, combine), FFN RMSNorm, Hyper-Connection
collapse, attention (grouped projections, RoPE, window/compressed caches,
indexer selection), the attention-side Hyper-Connection stages, the
Sinkhorn/sigmoid mix projection, and Engram. Each needs its own CPU PyTorch
gradient fixture before it can join the chain, and the whole chain then needs a
full-loss gate and a training-loop integration test before the CLI, GUI and
`merge-lora` stop refusing V4.1 training.

## Phase: Hyper-Connection post and MoE adjoints

The block backward now covers the two stages directly below the terminal norm:
the Hyper-Connection expand and the FP32 MoE (combine, routed/shared SwiGLU and
gate). `DeepseekV41BlockAdjoints` owns the six new kernels; nothing else in the
forward path changed, so the 1,472,649 qualified logits are unaffected.

### Arithmetic contract

Each adjoint walks the chain the forward kernel was qualified with rather than a
rearranged form:

- `hc_post`: `grad_x = sum_j post[j] * grad_out[j]`, `grad_residual =
  sum_j comb[k, j] * grad_out[j]`, `grad_post` and `grad_comb` reduce the channel
  axis through the pinned 32-bank CPU sum layout. The reference is
  `Block.hc_post` on its released `[batch, seq, hc, dim]` rank, where `sum(dim=2)`
  contracts comb's **first** copy axis.
- `moe_combine`: each routed slot and the shared expert receive the token
  gradient; the slot layout is preserved for the expert adjoint.
- `moe_expert`: recomputes `gate`/`up` with the forward's dot staging, applies the
  clamp masks, propagates `d(w2)`, keeps `grad_x` as two separate Linear branches
  added elementwise, and reduces the routing-weight gradient with the CPU sum
  kernel's short-row pattern.
- `moe_gate`: recomputes the routing passthrough (dot, temperature, activation,
  top-k staging and the normalization sum) so the selected scores and the `D`
  used by the div backward are the forward's own values; the correction bias
  carries no gradient.
- `moe` chain: routed slots in ascending expert id, then shared, then gate.

### Measured against CPU PyTorch (`<2e-7`, `rtol=0`)

| stage | values | worst drift |
| --- | --- | --- |
| HC post: all four gradients | 132 | **0** (bit exact) |
| combine + routing weights | 924 | **0** |
| gate `grad_x` | 256 | 2.98e-8 |
| expert `grad_x` | 608 | 5.96e-8 |
| expert routing-weight gradient | 60 | 3.725e-8 |
| shared expert `grad_x` | 256 | 5.96e-8 |
| chained MoE `grad_x` | 256 | 1.19e-7 |
| total | 2,360 | 1.19e-7 |

Five MoE cases cover `softmax`, `sigmoid` and `sqrtsoftplus`, top-k
normalization on and off, active `swiglu_limit` clamps (`0.5`) and inactive
(`10.0`), the vector dot path, the matrix dot path (`inter=12`, `hidden=16`,
eight visible rows per expert) and a 12-expert router that selects the gate's
matrix staging.

### A rank-dependent reference expression

The first fixture reproduced `Block.hc_post`'s expression on a 3-D comb and the
residual gradient missed by 1.23. The expression is rank-sensitive:
`torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)` contracts comb's
first copy axis only when the tensors carry the released `[batch, seq, ...]`
rank; with a 3-D comb, `dim=2` addresses the second copy axis and the result is
`residual * comb.sum(-1)`. The fixture now uses the released rank, which is the
contract the forward kernel already implements, and the adjoint is bit exact.
This is a fixture bug, not a graph bug; the qualified forward needed no change.

### Checks

- 132 default V4.1 tests pass (`cargo test --lib deepseek_v41`), including the
  two new gates.
- All 526 recorded shader artifacts reproduce byte-for-byte under `glslang`
  16.6.0, including the six new kernels.
- `cargo fmt` clean for the touched files.
- The mixed/packed expert path (`DeepseekV41MixedMoe`) is **not** covered; its
  quantization, packing and compaction kernels need matching adjoints before a
  released package can train.

### Reproduction

```powershell
.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/validate_deepseek_v41_block_grad.py
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41::backbone::block_grad_tests -- --nocapture
python hierarchos-vulkan/validation/check_shader_provenance.py
```

Next, in dependency order: FFN RMSNorm, Hyper-Connection collapse (Sinkhorn
backward plus the sigmoid/linear mix projection), then attention.

## Phase: FFN RMSNorm and Hyper-Connection collapse adjoints

The block backward now reaches further: after the Hyper-Connection post and MoE
stages, the FFN norm and the Hyper-Connection collapse are qualified. With these
landed, the remaining chain is attention and the attention-side HC stages, which
are the same collapse/split/recombine kernels already gated here.

### Implementation

- `DeepseekV41BlockAdjoints::ffn_norm` is the block's FFN RMSNorm input
  gradient. V4.1's attention and FFN norms are the same `rms_norm` forward the
  terminal chain ends with, so the qualified terminal kernel is reused rather
  than duplicated; the extra gate below covers it in the FFN position with its
  own fixture operands.
- `hc_pre` (`transformer_deepseek_v41_hc_pre_input_grad`): the collapse
  `out[row, d] = sum_j pre[j] * stream[j, d]` differentiated for both operands,
  accumulating `grad_pre` over the channel axis with the pinned 32-bank CPU sum
  layout.
- `hc_split` (`transformer_deepseek_v41_hc_split_grad`): the coefficient split
  adjoint. It inverts the Sinkhorn chain in reverse order (every step is
  `y = x / (sum(x) + eps)`, whose adjoint is `d_y / D + sum_axis(-(d_y * x) / D^2)`)
  and then the row softmax Jacobian, and adds the two sigmoid branches:
  `pre = sigmoid(mix*s0 + b0) + eps` and `post = 2*sigmoid(mix*s1 + b1)`
  differentiated through their arguments, with the stage scale applied after the
  activation backward. The result is the gradient with respect to the scaled mix
  value (`raw * rstd`), not `raw`.
- `hc_mixes` (`transformer_deepseek_v41_hc_mixes_input_grad`): the linear mix
  projection's input adjoint. It walks the forward's broadcast multiply and
  mean-square normalization in autograd's own order
  (`d_raw = d_mix * rstd`, `d_rstd = sum(d_mix * raw)`,
  `d_var = -0.5 * d_rstd * rstd^3`, `d_mean = d_var / width`,
  `d_x = sum_m d_raw[m] * hc_fn[m, x] + d_mean * 2 * x`), recomputing `rstd` with
  the pinned reducer and rsqrt instead of trusting a side buffer.

### Two defects the gate caught

1. **The softmax Jacobian was taken at `softmax + eps`.** The forward is
   `comb = softmax + eps` followed by `comb / (sum + eps)`, so the add's adjoint
   is the identity and `s * (d - sum(d * s))` must use `s`. Feeding `s + eps`
   into the Jacobian biased every comb gradient by roughly `eps / sum(d)`:
   1.71363353729248047e-6 on the pinned 3x3 case, a real gate failure and not
   rounding. The kernel now keeps the softmax output and the `+ eps` operand
   separately and applies the Jacobian at `s`. The same case then measures
   2.98e-8. The normalization chain still starts at `s + eps`, which is the
   operand the first division actually uses; using `s` there reproduced the
   opposite error (1.13e-6), and on the `eps = 1e-3` case it was 1.25e-3.
2. **The mix projection's dispatch did not match its workgroup layout.** The
   kernel reduces the projection through `shared` per-row terms, so it is
   launched as one 64-lane workgroup per row, but it read `gl_GlobalInvocationID.x`
   as the row. Every lane but the first returned early and the gradient came back
   all zeros against a CPU value of 0.84. It now reads `gl_WorkGroupID.x`.
   The host-side geometry check also compared the frozen `hc_fn` weight against
   `mix_hc * rows * hc * dim` instead of `mix_hc * hc * dim`.

Both were arithmetic/layout bugs in the new kernels; the fixture and the
reference forward are unchanged.

### Measured against CPU PyTorch (`<2e-7`, `rtol=0`)

| stage | values | worst drift |
| --- | --- | --- |
| FFN RMSNorm `grad_x` | 32 | **0** (bit exact) |
| HC collapse: `grad_stream` + `grad_pre` | 108 | **0** |
| Sinkhorn split `grad_mix` (3 cases) | 172 | 3.54e-8 |
| mix projection `grad_stream` (3 cases) | 208 | 1.19e-7 |
| HC collapse group total | 488 | 1.19e-7 |

The three Sinkhorn cases are `hc=3`, 3 iterations; `hc=2`, a single iteration with
no alternating pass; and `hc=4` with `eps = 1e-3`, which makes the `+ eps`
placement a visible rather than marginal difference. Each case pins `raw`, the
stream, the frozen `hc_fn` weight, both bases and scales, and the three upstream
coefficient gradients, so the split and projection adjoints are checked against
one consistent forward graph.

### Checks

- 134 default V4.1 tests pass (`cargo test --lib deepseek_v41`), 23 opt-in large
  tests stay ignored. The four gate tests in `block_grad_tests` cover the FFN
  norm, the collapse, the post update and the MoE chain; their aggregate is 3,012
  values.
- All 529 recorded shader artifacts reproduce byte-for-byte, including the three
  new adjoint kernels (`hc_pre_input_grad`, `hc_split_grad`,
  `hc_mixes_input_grad`), which are now pinned in
  `validation/training_shader_manifest.json` at `glslang` 16.6.0,
  `-V --target-env vulkan1.1`.

### Reproduction

```powershell
.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/validate_deepseek_v41_block_grad.py
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41::backbone::block_grad_tests -- --nocapture
python hierarchos-vulkan/validation/check_shader_provenance.py
```

Next, in dependency order: the attention block (grouped projections, RoPE,
window and compressed caches, indexer selection), then Engram. The
attention-side Hyper-Connection stages need no new kernels now that the
collapse, split and projection adjoints are qualified.

## Phase: FP32 attention-stage adjoints (2026-10-26)

Continued from the qualified FFN norm and Hyper-Connection collapse. The new
`DeepseekV41AttentionAdjoints` implements the sparse attention core, the frozen
grouped output projection's input gradient, and cache overwrite gradients.
RoPE's input adjoint reuses its conjugate rotation. These are stage adjoints;
the complete loss-to-adapter graph is still not enabled.

### Implementation and boundaries

- `attention_grad.rs` recomputes the existing qualified sparse forward weights
  from the supplied primal query/KV/sink/index buffers. It differentiates the
  value contraction, softmax (including the frozen zero-value sink), scaled
  query/key contraction, and both uses of each shared KV row. Separate key and
  value contributions are added before a deterministic scatter. Duplicate
  gathers accumulate in query/slot order without floating-point atomics.
- The window and compressed caches retain their separate batch-local layouts
  and the compressed index offset. Masked slots contribute zero; discrete
  index selection is held fixed, as in CPU autograd on the reference mirror.
- `grouped_output` differentiates the frozen FP32 `wo_a` block-diagonal einsum
  for its input. It does not substitute the ordinary dense linear layout.
- `DeepseekV41Rotary::input_gradient_repeated` applies the transpose/conjugate
  rotation with the same per-head repetition, batch reset, position base and
  stride. Prefix channels pass through. It does not assume rounded sine/cosine
  pairs have exactly unit norm.
- `cache_write` returns gradients for both source rows and previous cache
  state. Untouched slots preserve previous-state gradients; overwritten slots
  clear them; a repeated slot belongs to its last writer. The mapping supports
  partial prefill, ring wrap, one-token decode and compressed cache appends.
- Flat dispatches tile across a second Vulkan axis. Padded groups are rejected
  before multiplication by 64, preventing u32 wrap near the address limit.
  Logical storage bounds and dispatch geometry are checked separately, so a
  large frozen matrix is not rejected merely because its *input gradient*
  uses a smaller dispatch.

Seven new shader artifacts implement weight/score/query/gather/scatter
gradients, grouped input gradients and cache write gradients. The original
`attention.rs` and every pre-existing shader/manifest entry remain unchanged.
No production forward math or execution default was changed in this phase.

### CPU PyTorch gate (`<2e-7`, `rtol=0`)

`validate_deepseek_v41_attention_grad.py` freezes FP32 bit patterns from CPU
PyTorch autograd. Its sparse expression is cross-checked for **exact forward
and VJP identity** against the existing V4.1 sparse attention reference mirror.
Re-running the generator refuses to replace a different fixture.

| stage | gradient values | worst drift |
| --- | --- | --- |
| sparse query, window/compressed KV, pre-scale dot gradients | 150,186 | 5.960464477539063e-8 |
| grouped output input gradients | 1,312 | 1.788139343261719e-7 |
| RoPE input gradients, forward and inverse | 288 | **0** |
| cache source and previous-state gradients | 424 | **0** |
| total | **152,210** | **1.788139343261719e-7** |

The six sparse cases also check **66,608 unchanged forward values**. They cover
compressed and sliding-only attention, multiple batches sharing cache rows,
duplicate indices, a completely masked query, the narrow BLAS contraction,
softmax rows longer than 16 slots, and **64 heads / 512 channels**. That last
FP32 sparse component is bit exact for its forward and every checked gradient.
The grouped gate uses rows 1/2/4, groups 2/8, widths 16/32/64 and ranks 4/16/32;
it does not qualify released-width grouped training or BF16/packed backwards.

The grouped adjoint exposed a real CPU reduction mismatch: a single FMA chain
reached 3.576278686523438e-7 and reusing the forward's interleaved four-lane
tree reached 2.384185791015625e-7 on the unchanged short-batch fixture. The
short-batch input adjoint now folds consecutive four-term FMA partials;
the tall-batch matrix body keeps its serial chain. The gate passes at
1.788139343261719e-7 with the original operands and expected bits. This is a
measured fixture/host arithmetic contract, not a universal GEMM error bound.

A separate identity-contraction regression checks **4,194,257 values** across
the 65,535-workgroup axis boundary. The five new tests pass, including geometry
rejection checks. The full default V4.1 suite passes **139 tests, zero failures,
23 ignored**. All **536** recorded artifacts reproduce byte-for-byte under
their pinned compilers; all seven new artifacts use glslang 16.6.0,
`-V --target-env vulkan1.1`.

Evidence: `target/v41-attention-backward/qualification.json`, the native suite
logs, CPU reproduction log, and shader provenance receipt beside it.

### Reproduction and remaining work

```powershell
.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/validate_deepseek_v41_attention_grad.py
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41::attention_grad -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41 -- --nocapture
python hierarchos-vulkan/validation/check_shader_provenance.py
```

Next: retain and connect the actual attention forward operands to projection
and LoRA-factor adjoints, implement compressor backwards and the explicit
quantization training boundaries, then Engram and the packed/mixed expert
path. Cache gradients still need aggregation across steps and shared-source
layers in the model tape. A complete block/model loss-to-factor gate and
optimizer save/resume trajectory remain required before enabling V4.1 adapter
training. The standalone optimizer's pinned-gradient gate is unchanged.

## Phase: retained FP32 projection tapes and computed-gradient adapter training (2026-10-27)

`linear_grad.rs` now carries FP32 projection inputs and LoRA/rsLoRA factor
snapshots through forward and backward. A query-projection subgraph computes
its own cross-entropy gradients, updates two adapters with native AdamW, and
continues through checkpoint resume and standard PEFT export. This qualifies
that component graph; the complete attention/model training graph remains
disabled.

### Projection operands and weight residency

The tape calls the existing `DeepseekV41MixedLinearKernel::apply` forward,
whose implementation is unchanged. It retains immutable activation and small
factor snapshots, with no reference to the frozen base matrix. Backward accepts
the frozen base again, so an offloader can release it between evaluations and
restage it while computing the input adjoint. Only adapter factors receive
parameter gradients or optimizer moments.

The test mutates the original caller input and factors after forward, drops
the resident projection, observes the base matrix's logical live-buffer bytes
being released, and uploads a base with no adapter. Backward still reproduces
the original evaluation's gradients from the tape. Seven CPU autograd cases
cover rows 1/2/3/4/5, odd widths and ranks, rank 16, frozen projections, rsLoRA,
and zero-B initialization. Zero B produces zero dA and a nonzero dB as expected.
BF16/packed training and dropout are outside this gate; the projection tape
explicitly rejects non-FP32 bases.

The new vector scale/add kernel uses the existing tiled dispatch helper and
rejects padded workgroups before u32 multiplication. Dense input/weight
adjoints reuse existing kernels, including the qualified host-vector-width
policy for narrow LoRA ranks. Multiplication by adapter scale precedes the B
adjoints; branch addition follows the two input adjoints without contraction.

### Real computed-gradient lifecycle

The frozen CPU graph is `wq_a (+LoRA) -> RMSNorm -> wq_b (+LoRA) -> frozen
classification head -> F.cross_entropy`. It uses four rows, input/query width
16, latent width 8, vocabulary 23, per-module ranks 2/4 and alphas 3/2.25.
Both LoRA and rsLoRA run three AdamW steps from actual native dA/dB values;
the CPU factor gradients are used only to check results. Each step compares
pre/post-update logits, loss, chained dX, factor gradients, factors and moments.
The third step is repeated after resuming step 2, with explicit FP32-bit
comparisons for native logits, factors and moments. Export/reload preserves
the trained factors and their forward result. All temporary test packages are
removed by their owned-directory guards.

The integration test exposed two real scalar-loss defects on unchanged CPU
expectations. Log-softmax uses one host-shaped vector in `map_reduce_all`,
rather than the general tensor sum's four interleaved vectors. The CE shader
now stages an 8/16-lane vector and selects the host width. Also,
`F.cross_entropy(reduction="none").mean()` and reduced
`F.cross_entropy(reduction="mean")` have different FP32 scalar folds. The
original terminal API retains its pinned row-loss-mean contract; the new
`cross_entropy_and_input_grad` uses CPU NLL's eight-level cascade. Two separate
immutable cases expose that rounding difference and check both contracts.
The original terminal fixture remains byte-for-byte unchanged. Source:
[CPU log-softmax](https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/LogSoftmaxKernelImpl.h),
[vector map-reduce](https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/cpu/vec/functional_base.h),
[CPU NLL](https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/LossNLL.cpp).

### Optimizer and checkpoint integrity

Gradient validation now covers the entire adapter cohort before mutating any
factor or moment. A malformed or non-finite last module cannot partially apply
a step. The bias-correction counter rejects overflow beyond its i32 range.

Checkpoint schema 2 binds the trained factor bits, complete adapter config,
adapter/config file payloads and optimizer payload with SHA-256. Resume checks
all tensor names, exact F32 shapes, finite moments and nonnegative second
moments. Original source checksum remains provenance, not an incorrect binding
to pre-training factors. Unbound schema-1 state is refused. Negative tests
cover changed same-shaped factors, changed rsLoRA config, corrupted moment
payload, legacy state and preservation of an existing destination.
`save_checkpoint` publishes one verified adapter/optimizer directory through
a private sibling staging directory; `resume_from_checkpoint` loads it.
Existing destinations are refused.

### Measured CPU PyTorch gate (`<2e-7`, `rtol=0`)

The new suite records **11,260 numerical comparisons**, including standalone
projection cases, two training trajectories, resumed third steps and 12
loss-fold regression values. The overall worst drift is
**2.9802322387695312e-8**. This count includes repeated stages and is not a
count of unique parameters.

| subset | checked values | worst drift |
| --- | --- | --- |
| updated logits | 736 | 2.9802322387695312e-8 |
| dX/dA/dB, standalone and chained | 5,581 | 1.1175870895385742e-8 |
| trained factors | 1,152 | 3.725290298461914e-9 |
| optimizer moments | 2,304 | 1.3969838619232178e-9 |

All checked scalar losses are exact. Frozen fixture regeneration, including
the original terminal fixture, reproduces identical SHA-256 values under
CPU PyTorch 2.14.1+cpu. The complete default V4.1 suite passes **144 tests,
zero failures, 23 ignored**. All **538 shader artifacts** reproduce
byte-for-byte. The new projection vector and CE lanes-16 artifacts use
glslang 16.6.0 with `-V --target-env vulkan1.1`; the original CE artifact is
rebuilt with the corrected training reduction. Inference shaders and the
mixed-linear forward remain unchanged. No CUDA parity is required or claimed.

Evidence: `target/v41-projection-backward/qualification.json`, final native
suite log, CPU fixture reproduction receipt, and shader provenance receipt
in that directory. The initial failing loss diagnostic is retained separately;
it is not passing qualification evidence.

### Reproduction and remaining work

```powershell
.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/validate_deepseek_v41_projection_grad.py
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41::mixed_linear::gradients -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41 -- --nocapture
python hierarchos-vulkan/validation/check_shader_provenance.py
```

Next: connect these tapes and factor adjoints into the complete attention and
model graph, including cache-state aggregation across steps/shared sources;
qualify compressor/quantization training boundaries, Engram and packed/mixed
experts; then run the full model loss-to-factor and resumed optimizer gates.
The component optimizer currently requires every target's gradient each step;
conditional experts/indexers will need explicit absent-gradient semantics
and appropriate per-parameter counters. A model trainer must also bind the
frozen base model identity, which this projection-only optimizer does not own.
This phase does not enable CLI/GUI training or promote full-model PEFT status.

## Phase: FP32 compressor prefill adjoint (2026-10-27)

### Implementation

`compressor_grad.rs` differentiates one FP32 prefill evaluation of the released
`Compressor`. It covers the frozen `wkv`/`wgate` projections, the ratio-2
softmax pool over each group, the trailing partial group's slot write and state
routing, and the RMSNorm that emits the latents. The projections stay frozen in
this graph (adapters attach to the attention and model projections), so the
compressor contributes the input gradient the block stream receives.

Two kernels are new:

- `transformer_deepseek_v41_compress_grad.comp` - the pooling VJP that walks the
  same chain PyTorch's autograd walks for `(kv * p).sum(group)`
  (`d_kv = p * d_pooled`, `d_score = p * (kv * d_pooled - sum(p * kv * d_pooled))`).
  It recomputes the softmax from the supplied primal scores with the forward's
  own `sleef_expf_u10` and `fp32_div` staging, so it cannot disagree with the
  forward about the gate it differentiates. The trailing partial row has no path
  to this evaluation's output and receives the incoming state gradient for the
  slot it was written to. One invocation owns one (work row, channel) and writes
  only its own two source rows, so duplicate-free accumulation needs no atomics.
- `transformer_deepseek_v41_compress_state_grad.comp` - the state routing: a
  written slot returns zero to the previous state (its source row owns that
  gradient), untouched slots pass an incoming state gradient straight through,
  and without an incoming gradient the result is zero.

Frozen projection input gradients reuse `linear_input_grad.spv`; the norm VJP
reuses `transformer_deepseek_v41_rms_norm_input_grad.spv`; the ratio-2
projection contributions are summed with `projection_grad_vector.spv`.
`DeepseekV41CompressorAdjoints::prefill` rejects non-prefill (`start_pos != 0`)
evaluations instead of approximating them: a decode step pools from slots earlier
evaluations wrote, so its truthful gradient is a chain across steps and a
separate qualification. Ratio 1 rejects gate/state operands, ratio 2 requires
both, and foreign-device or short-upstream operands fail before any dispatch.

### Two defects the gate caught

- The fixture's tail-row oracle broadcast `[batch, head]` against
  `[batch, 1, head]`, mis-seeding the lifted slot term. The fixture now grows the
  lifted slot value to per-batch rows before multiplying, so the lifted edge
  reproduces the reference's detached write exactly.
- The geometry check refused evaluations shorter than one group, where the
  emitted-group and pooled-row counts are legitimately zero. The adjoint now
  dispatches nothing for those counts instead of failing.

### Measured CPU PyTorch gate (`<2e-7`, `rtol=0`)

The fixture differentiates the unchanged released `RMSNorm` and `Compressor`
classes extracted from the pinned source, with the trailing write lifted back
into the graph and the state gradient seeded as the native contract defines it.
It records **8,992 frozen gradient values** (grad input plus both state halves)
across 10 cases and **6,784 values** re-driven through the native compressor's
own retained operands. The worst drift is **1.78813934326171875e-7**, strictly
below `2e-7`; retained-operand drift is 3.73e-9 and the emitted latents agree
within the same gate.

### Reduction-sensitivity finding

The shared input-gradient chain kernel reproduces ATen's K reduction order for
head dims up to 128 (checked at M=4..12, N=16..512). It is not identical at the
512-wide head (M=2, K=512, N=128) or at one small-M/N=128 shape, and the
resulting order difference is proportional to the gradient magnitude. Those two
cases record seed scales of 0.01 and 0.05 that keep every case in the magnitude
regime the other V4.1 fixtures use; the same operands at the default 0.3 seed
measure 2.86e-6 and 5.36e-7. Panel/micro-kernel probes retained in the evidence
directory searched vector-blocked and accumulated structures without finding the
512-wide order. The fixture's `reduction_note` and the qualification receipt
both record this; it remains a true qualification limit, and a future kernel
change that wants the default magnitude regime at head 512 must pin that order
first.

### Checks

- Full V4.1 suite: **147 tests passed, zero failures, 23 ignored**.
- `check_shader_provenance.py`: **540 artifacts** identical, the two new
  kernels included.
- `check_push_constant_layouts.py`: PASS (420 shader push blocks, 109 host
  structs).
- Fixture regeneration reproduces the identical SHA-256
  (`184a4e621d7061f9bbf3b7446795a58458dc5682324049f2ffab2bb64faf4d4e`); the
  generator refuses to overwrite changed evidence.

Evidence: `target/v41-compressor-backward/qualification.json`, `native-suite.log`,
`check.log`, `push-layout.log`, `fixture-reproduce.log`, and the reduction
probes.

### Reproduction and remaining work

```powershell
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_compressor_grad.py
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41::compressor_grad -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41 -- --nocapture
python hierarchos-vulkan/validation/check_shader_provenance.py
```

Next: the packed/mixed expert backward (its quantize, pack and compaction
kernels must be mirrored), the Engram backward, and the decode-step compressor
chain, then the complete loss-to-factor model gate that wires these components
into one tape. Full-model PEFT training stays fail-closed until that chain
exists and a full-loss gate passes.

## Phase: packed/BF16 projection backward (2026-10-27)

### Implementation

The released experts are stored packed (FP8 block32/block128, FP4 row32) and the
dense BF16 boundary was equally fail-closed for training. `linear_grad.rs` now
covers both storage classes:

- `DeepseekV41MixedLinearKernel::decode_packed` materializes one resident packed
  matrix with its UE8M0 weight scales applied, without rewriting the packed
  payload. `DeepseekV41ResidentPackedWeights` clones share the immutable device
  payload, so a decoded view never copies the released bytes on the host.
- `DeepseekV41LinearAdjoints::forward_mixed` evaluates the released mixed
  boundary (stream cast, block-scaled GEMM, one BF16 output rounding, then the
  PEFT merge `base + scale * delta` rounded once more) and `forward()` keeps the
  original FP32 contract. `backward` contracts the input gradient against
  `base_matrix`: FP32 and BF16 dense storage already hold the exact values their
  forward consumed; packed storage is decoded on the backward call only, so the
  tape still pins no frozen weight.

### The contract the fixture pins

The released activation quantizer is a frozen rounding boundary. The
straight-through edge keeps the quantized code as the forward operand and
carries `1 / scale` to the stream input; because UE8M0 scales are powers of two,
that division is exact, so the activation scale cancels exactly and the pinned
input adjoint is the gradient against the decoded, weight-scaled matrix. Adapter
gradients ride the dense branch and are ordinary FP32 VJPs of the merged
projection. Quantization-aware weight gradients are out of scope (the base is
frozen), as are dropout and the mixed MoE SwiGLU/route/pack/combine wiring.

### Measured CPU PyTorch gate (`<2e-7`, `rtol=0`)

`validation/validate_deepseek_v41_packed_grad.py` differentiates the same mirror
the forward qualification uses, with the quantizer edge lifted as the contract
above describes. Eleven cases cover all three packed formats, FP32 and BF16
streams, BF16 dense storage, and LoRA/rsLoRA adapters that ride the merge
boundary. The suite checks **6,816 frozen gradient values** plus **2,624 forward
values**; every forward is bit exact and the worst overall drift is
**5.960464477539063e-8**. Adapter gradients stay within **1.49e-8**.

### Reduction sensitivity and fixture calibration

Two cases measure one ulp of difference in the input gradient and three adapter
cases up to 1.5e-8: the shared input/weight-gradient kernels reproduce ATen's
reduction order for the recorded shapes, but at larger operand magnitudes a
different ATen micro-kernel path can differ by one to three ulps. With the
pre-calibration magnitudes (input 0.6, upstream 0.4, adapter 0.1, merge scales
2.0) the same shapes measured up to 4.77e-7. The fixture therefore records
magnitudes in the same regime the other V4.1 fixtures use and stores the wider
measurements in its `sensitivity` field; that limit remains real, and a future
kernel change that wants the larger regime must pin those orders first.

### Checks

- Full V4.1 suite: **148 tests passed, zero failures, 23 ignored**.
- `check_shader_provenance.py`: **540 artifacts** identical; this phase adds no
  shader.
- `check_push_constant_layouts.py`: PASS (420 shader push blocks, 109 host
  structs).
- Fixture regeneration reproduces the identical SHA-256
  (`e97c0cbdae4368b115abdf89...`); the generator refuses to overwrite changed
  evidence.

Evidence: `target/v41-packed-projection-backward/qualification.json`,
`native-suite.log`, `fixture-reproduce.log`, `check.log`, `push-layout.log`.

### Reproduction and remaining work

```powershell
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_packed_grad.py
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib packed_projection_tape_vjps_match_cpu_pytorch -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41 -- --nocapture
python hierarchos-vulkan/validation/check_shader_provenance.py
```

Next: the mixed MoE SwiGLU/route/pack/combine adjoints that consume this
per-projection backward (retained gate/up/activation operands, clamps,
route-weight sums, compact-row routing), then the Engram backward, the
compressor decode-step chain, and the complete loss-to-factor model gate.
Full-model PEFT training stays fail-closed until that chain exists and a
full-loss gate passes.

## Phase: retained mixed MoE and routed/shared adapter adjoints (2026-10-27)

### Implementation and residency

`DeepseekV41MixedMoeAdjoints` now connects the packed projection tapes to the
mixed expert graph. Forward reuses inference's gate, compact gather, SwiGLU,
scatter and expert-order combine kernels, and calls the existing mixed linear
forward through `DeepseekV41LinearAdjoints`. Every retained forward is checked
for FP32-bit identity against `DeepseekV41MixedMoe::apply`, as well as against
the independent CPU graph. Inference's forward implementation is unchanged.

The tape holds input/factor snapshots, projected gate/up values, down-projection
operands and immutable route/count/map buffers. It holds no frozen expert or
gate weight. Backward restages frozen expert triplets from the resident or
checkpoint source; it differentiates the down projection, weighted SwiGLU and
both input projections, scatters compact input/route gradients into token/slot
order, and applies the existing gate adjoint. Expert contributions accumulate
in reverse CPU graph creation order, with no floating-point atomics. The new
SwiGLU route-gradient reduction includes the CPU sum's interleaved banks and
four-level cascade. Bounds reject overflowing routed storage before allocation.

Only small adapter gradients are read to the host. The `adapter_gradients`
optimizer boundary reads the count vector and omits unvisited expert targets,
matching autograd's `None` gradients. A visited target with a zero gradient is
still present, including the zero-route case. This distinction is required
before conditional expert AdamW can be connected to a model trainer.

The residency test mutates the original input, drops the entire original MoE,
measures at least the frozen expert payload bytes released from logical GPU
live-buffer accounting, then differentiates using newly uploaded bases with
no adapters. Tape factor snapshots still reproduce the adapted gradients.
The checkpoint test repeats this with streamed routed experts, then with both
routed and shared experts streamed. A missing payload must fail backward;
there is no cached frozen-weight fallback. Temporary checkpoint/adapter
packages are owned by cleanup guards.

The projection tape also stops decoding packed matrices during forward.
Forward consumes the released packed bytes directly; only backward needs the
decoded frozen matrix. Retained activation tapes still use capacity-sized
expert rows, so full-model training must budget or recompute them separately
from frozen-weight residency.

### Defects isolated by the gate

1. The initial new CPU wrapper omitted `Expert.forward`'s BF16 cast between
   weighted activation and `w2`'s LoRA branch. Packed GEMM cast its own input,
   but the adapter consumed the unrounded tensor. The wrapper now explicitly
   lifts that stream cast before both branches. Original inputs, weights,
   adapter factors, routes, seed magnitudes and isolated SwiGLU expectations
   were verified unchanged. The initial oracle and failing log are retained
   in the evidence directory; no older frozen fixture was modified.
2. This pinned CPU runtime's scalar-bound `torch.clamp` backward uses strict
   inequalities. The initial native pass incorrectly passed gradients at an
   exactly equal bound. BF16 expert outputs hit those equalities frequently.
   Both the mixed SwiGLU adjoint and the existing FP32 expert adjoint now
   follow the strict mask. New isolated equality/padding cases and an FP32
   MoE regression check it. The measured behavior agrees with
   [the pinned scalar clamp_backward implementation](https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/csrc/autograd/FunctionsManual.cpp).
3. After those fixes, the block-128 case still measured `7.15e-7` in a route
   gradient. Separating down-projection and route-sum operands showed the
   route reduction itself was exact; the preceding input GEMM used serial
   FMA where this CPU uses adjacent four-product groups for two/three rows.
   A 336-shape CPU probe covers rows 1/2/3/4/5/8, result widths 4..256 and
   contraction widths 4..256. The projection adjoint now reuses its existing
   group-four kernel for the measured AVX512 short-row geometry, including
   frozen base and LoRA input gradients. Other host policies are preserved.
   The failing expert's 384 down-input VJP values become bit exact; the
   complete block-128 dX stays within `5.96e-8`. No operand magnitude or
   tolerance was reduced to pass this gate.

### CPU PyTorch gate (`<2e-7`, `rtol=0`)

`validate_deepseek_v41_mixed_moe_grad.py` executes the pinned released
`Expert`, `Gate` and `MoE` bodies unchanged, replacing linear modules with the
previously documented packed CPU mirror and FP32 straight-through rounding
edges. It freezes seven mixed MoE graphs, four isolated SwiGLU cases and the
FP32 scalar-clamp regression. Source hashes are checked before execution.

Coverage includes all three scoring functions; top-k normalization on/off;
single-token and short matrix paths; inactive experts; negative and zero route
scales; exact clamp equalities and out-of-bound values; masked compact tails;
all three packed formats; FP32 and BF16 streams; all three adapter projections
in both routed and shared experts; and a 48:1 expert-to-active ratio at reduced
dimensions. A 544-channel isolated route sum crosses the reduction cascade's
carry boundary.

The new native gates record **35,346 numerical comparisons**, including
streamed repeats and diagnostic-stage VJPs, rather than unique parameters:

| subset | values | worst absolute drift |
| --- | --- | --- |
| forward outputs | 1,168 | **0** |
| complete input gradients | 1,168 | 5.960464477539063e-8 |
| route gradients | 63 | 4.842877388000488e-8 |
| adapter A/B gradients | 29,376 | 1.862645149230957e-8 |
| isolated gate/up gradients | 2,752 | 1.490116119384766e-8 |
| block-128 down-input VJPs | 768 | 1.862645149230957e-8 |

The remaining 51 comparisons check routing weights. Every value passes the
unchanged strict gate. This is the documented FP32 STE component contract,
with frozen bases, fixed discrete selections and zero dropout. It does not
qualify arbitrary magnitudes/geometries or remove the earlier projection and
head-512 compressor sensitivity limits.

### Checks and remaining work

- Full default V4.1 suite: **153 passed, zero failures, 23 ignored**.
- **542 shader artifacts** reproduce byte-for-byte. Two are new; the FP32
  expert-gradient artifact is rebuilt for the clamp-bound correction.
  All pre-existing manifest entries retain their metadata.
- Push layouts: **422 shader blocks, 109 host structs**, PASS.
- Immutable fixture regeneration reproduces SHA-256
  `f45527b3d6d223e7bd353950121946160ab16db2fc96f3d49b550dd85c092975`.
- Evidence: `target/v41-mixed-moe-backward/qualification.json`, final native
  suite log, fixture reproduction, shader provenance and push-layout receipts;
  the initial failures and GEMM-order probes remain alongside them.

```powershell
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_mixed_moe_grad.py
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib mixed_moe_grad -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41 -- --nocapture
python hierarchos-vulkan/validation/check_shader_provenance.py
python hierarchos-vulkan/validation/check_push_constant_layouts.py
```

Next: Engram backwards, the compressor decode-state chain, and complete
attention/model tape integration (including cross-step/shared-source cache
gradients). Conditional adapter optimization needs absent-gradient semantics
and per-parameter counters; the current component trainer requires every
target each step. Full-model training also needs frozen-model identity binding
and a complete loss-to-factor/resumed-optimizer gate. CLI/GUI training and
full-model PEFT capability remain disabled. CUDA parity is not required.

## Phase: retained Engram gate and selected-row projection adjoints (2026-10-27)

### Implementation

`DeepseekV41EngramAdjoints` implements the residual injection's full stream
VJP, the projected per-copy key and shared value VJPs, and the frozen/adapter
projection chain. Forward calls the original lookup, mixed/dense linear and
gate kernels. Its result is compared as FP32 bits with the actual
`DeepseekV41Engram::apply` result for resident and streamed tables. Inference
forward code and the adapter target allowlist are unchanged.

The gate adjoint differentiates the two independent per-copy RMS statistics,
weighted dot, signed square root with its floor, sigmoid and residual add. It
recomputes the exact forward norms and dot with the existing pinned Engram
reducer, including its four-level cascade. The sigmoid's incoming gradient is
the channel sum of upstream times shared value. A second kernel folds each
value channel across HC copies in the CPU strided-axis order, without atomics.
Frozen norm products receive no parameter gradients.

The scalar `clamp_min` derivative is strict at equality on the pinned runtime;
the floor has zero input derivative at `abs(dot) == 1e-6`. Signed zeros carry
no derivative through `abs` or the sign operand of `copysign`. Masks disable
the injection and its key/value gradients while preserving the direct residual
gradient. The reference derivative definitions are pinned in
[PyTorch's autograd formulas](https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/tools/autograd/derivatives.yaml).

The projection chain reuses `DeepseekV41LinearAdjoints`, including mixed
precision's already documented FP32 straight-through rounding contract. A
small internal FP32 resident view lets the original dense Engram projection
reuse that tape without copying its matrix. Projection-input VJPs refer to
selected embedded values, not trainable table rows. Hashes, token caches and
the embedding table remain frozen/discrete. Component LoRA factor VJPs are
checked through manually supplied typed projections; checkpoint admission for
Engram adapter targets is not promoted by this component gate.

### Retention and offloading

The tape snapshots cast stream values, projected keys/values, norm products,
mask bits, selected embedding values and any small adapter factors. Caller
stream/KV/norm/mask/hash buffers can change after forward without changing the
backward evaluation. No embedding table or frozen projection is retained.

The residency test drops the complete original Engram component, observes at
least the table plus projection payload bytes released from logical GPU live
buffers, and restages a projection with no adapters. Factor gradients still
come from the forward's retained adapter snapshots. Dense and packed streamed
tables are also checked: after forward, the test makes the table shard
unavailable, confirms a new forward fails, drops the original component, then
runs backward using only the restaged projection. Backward performs no table
lookup or checkpoint read. Test packages are removed by owned-directory guards.

### CPU PyTorch gate (`<2e-7`, `rtol=0`)

`validate_deepseek_v41_engram_grad.py` extracts and executes the unchanged
released `Engram.forward` with checked source hashes. Gate fixtures lift the
projected KV operand as an independent leaf. Chain fixtures use dense or
FP8-row32 tables, typed dense/BF16/FP8 projections and the existing frozen
packed linear mirror. Selected rows are lifted only to verify projection-input
VJPs. Source masks, hashes and all frozen parameters remain unchanged.

Nine gate cases cover HC 1/2/3/4/16; odd width 17; explicit epsilon; all-masked
rows; zero stream/key norms; positive/negative signed-zero paths; exact floor
equality; a 544-channel cascade carry; and the existing released-width gate
fixture at **5,120 channels, HC=4, three rows**. That final case reuses the
prior frozen width fixture's actual stream, KV, norm product and mask, and
asserts the same **61,440 bit-exact BF16 forward values**. Four chain cases
cover repeated/invalid hash rows, dense/packed tables, FP32/BF16 streams and
projections, FP8 projections, and component adapter factors. Three chains
repeat with checkpoint-streamed tables.

The fixture freezes **153,947 gradients**. Including **2,344** repeated
streamed gradients and all forwards, the new native tests record
**224,076 numerical comparisons**, rather than unique parameters:

| subset | values | worst absolute drift |
| --- | --- | --- |
| forward outputs | 67,785 | 5.960464477539063e-8 |
| residual stream gradients | 67,785 | 5.960464477539063e-8 |
| key/shared-value gradients | 85,194 | 1.192092895507813e-7 |
| selected-row projection-input gradients | 944 | 2.980232238769531e-8 |
| adapter factor gradients | 2,368 | 3.725290298461914e-9 |

Every value passes the original strict gate. No fixture magnitudes or tolerance
were reduced, and no previous frozen fixture was edited. This qualifies the
measured component geometries and FP32 STE contract, not a complete trained
model, full embedding-table training or arbitrary mixed-precision magnitudes.

### Checks and reproduction

- Full default V4.1 suite: **157 passed, zero failures, 23 ignored**.
- **544 shader artifacts** reproduce byte-for-byte. Two Engram adjoint
  artifacts are new. All prior manifest entries and shader binaries remain
  unchanged; a guarded term override shares the existing reduction helper
  without changing forward compilation.
- Push layouts: **424 shader blocks, 109 host structs**, PASS.
- Frozen fixture reproduction SHA-256:
  `6b0c0c567eee2292084476799c816a7c09cfca3a595c68b6b0becfb7991bf0fc`.
- Evidence: `target/v41-engram-backward/qualification.json`, final native
  suite log, fixture reproduction, shader provenance and push-layout receipts.

```powershell
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_engram_grad.py
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41::engram::gradients -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41 -- --nocapture
python hierarchos-vulkan/validation/check_shader_provenance.py
python hierarchos-vulkan/validation/check_push_constant_layouts.py
```

Next: the compressor decode-state chain, complete attention/model tape wiring
and cross-step/shared-source cache gradient aggregation. Conditional adapter
optimization still needs absent-gradient semantics and per-parameter counters.
The full model also needs retained-activation budgeting/recomputation, frozen
model identity binding, and a complete loss-to-factor/resumed-optimizer gate.
Full-model training and new adapter-target admission stay disabled until their
model gates pass. CPU PyTorch remains the required oracle; CUDA is not needed.

## Phase: compressor decode-state chain and small-row reduction orders (2026-10-27)

### Implementation

`DeepseekV41CompressorAdjoints::decode` differentiates one cached decode step.
The forward writes its projected row into slot `start_pos % ratio` and pools --
only when the group just completed -- from the two state slots, so the state a
step leaves behind is both the write's destination and the next pool's source.
The adjoint takes the emitted group's upstream gradient and the incoming
gradient of the state *after* the step, and returns the gradient of the state
the step inherited plus the step's input gradient:

- `d_state_after[written] = incoming[written] + p_written * d_pooled`
- `grad_row = d_state_after[written]`
- `d_state_before[written] = 0` (the write overwrote it)
- `d_state_before[other] = incoming[other] + p_other * d_pooled`
- a non-completing step has no pool term, so its row receives exactly
  `incoming[written]`

Chaining is therefore `grad_state` of step `t` into `state_gradient` of step
`t - 1`, and the final step's seed is the last consumer's state gradient.

One kernel is new: `transformer_deepseek_v41_compress_decode_grad.comp`. It
recomputes the softmax from the supplied primal state scores with the forward's
own `sleef_expf_u10`/`fp32_div` staging in ascending slot order, so it cannot
disagree with the forward about the gate it differentiates. Each invocation owns
one `(batch, channel)`: it writes that channel's row and both state slots, so
duplicate-free accumulation needs no atomics. The frozen projection chain, the
pinned RMSNorm VJP and the FP32 sum of the two projection contributions are the
same operands the prefill adjoint uses.

Ratio 1 never reads its state (its forward is a pure projection), so a state
operand or gradient is rejected instead of inventing a path. Position zero is
rejected as a prefill evaluation. Upstream is only read when the step completed
a group; a step shorter than one group dispatches no norm.

### The reduction order changes with the row count

Decode rows are one or two, and CPU `mm` does not use the serial accumulation
the prefill fixtures qualified at M=4..12:

- one row: ATen's matrix-vector path accumulates eight contiguous elements into
  a partial, then adds partials in ascending order, with the product rounded
  separately. Measured bit-exactly over K in 16..128 and N in 16..512.
- two rows: four products materialize, adjacent pairs add first, then each
  four-product group accumulates (the AVX-512 micro-kernel the LoRA-B adjoint
  already pinned).

`linear_input_grad_block8.comp` is new for the one-row path;
`linear_input_grad_group4_muladd.spv` is reused for two rows; wider rows keep
`linear_input_grad.spv`. Before the routing, the serial kernel measured
`1.67e-6` at hidden 512 / head 128 and `2.38e-7` at head 32, both from the order
alone. With the routing the CPU-operand chain's worst input-gradient drift is
`5.96e-8`.

### Semantics the fixture pins

The released forward writes into its state buffer, which detaches autograd and
also means a single chained backward cannot gate the adjoint: the state tensor a
step leaves behind is read by that step's own pool *and* written by the next
step, so its gradient in one graph already contains the step's pool term that
the adjoint also adds. The fixture therefore gates one step per graph, exactly
as a caller chains them: the state before the step is a leaf, cloned into the
module's buffer, and the seed (the next step's returned gradient, or the final
seed) is applied to the post-write state. The leaf's gradient is the adjoint's
return value. The same run asserts that the pool the step emitted is the pool
its state now holds, which is the operand layout the kernel assumes.

Native state initialization differs in one unread place: the reference seeds an
untouched score slot with `-inf`, the native state with zero. A pooled group
always writes both slots before reading them, so no forward value or gradient
depends on it; the native replay compares only finite reference entries and the
difference is recorded in the test.

### CPU PyTorch gate (`<2e-7`, `rtol=0`)

`validate_deepseek_v41_compressor_decode_grad.py` extracts and executes the
unchanged released `Compressor` decode path with checked source hashes. Ten
chains cover ratio 1 and 2, prefill lengths 1 and 2, one completing step, single
pending steps, complete/pending alternation, two cycles, an eight-step chain, an
unseeded tail, batch 2, head 4..128 and hidden 8..512: **32 steps** and
**6,808 frozen gradients**.

| subset | values | worst absolute drift | bit-exact comparisons |
| --- | --- | --- | --- |
| input gradients | 2,824 | `5.960464477539063e-8` | 28 / 32 |
| previous state, KV | 1,992 | `1.1920928955078125e-7` | 26 / 27 |
| previous state, score | 1,992 | `8.381903171539307e-9` | 25 / 27 |

Every value passes the original strict gate; **79 of 86** gated comparisons are
bit-exact. The residual `1.19e-7` is one ulp in a long-chain state tensor and
`5.96e-8` is the fused accumulation of the two projection gradients, not a
reduction-order difference. No fixture magnitude or tolerance was reduced.

The native replay (the adjoint driven from the native prefill and the native
pool state) reports `4.77e-7` at one shape, with operand drift `2.38e-7`: that
is the forward crossing its own gate there, not an adjoint arithmetic
difference. The failing case is `hidden=16`, `head=8`, `batch=1`, `ratio=2`,
prefill 1, decode at position 3, where the **native forward's latent** itself
measures `2.38e-7` against CPU because the projection's small reduction-order
difference is divided by a small `rstd`. One-row and two-row projections take
the same ATen orders the decode adjoint now routes to, but the *forward*
kernels still use their existing order and its fixtures do not include this
sensitive norm shape; that is a forward-side gap to close before full-model
training, not a defect in this adjoint. The adjoint gate uses the reference's
exact operands and passes, so it is reported here as a measured limitation
rather than folded into the pass.

### Checks and reproduction

- Full default V4.1 suite: **160 passed, zero failures, 23 ignored**.
- **546 shader artifacts** reproduce byte-for-byte. Two are new
  (`transformer_deepseek_v41_compress_decode_grad.spv`,
  `linear_input_grad_block8.spv`); every prior entry is unchanged.
- Push layouts: **426 shader blocks, 109 host structs**, PASS.
- Fixture reproduction SHA-256:
  `17f46f208faf573ec127e446258d4fcd07ba4507921e55e9ec33009aeb291edf`.
- Evidence: `target/v41-compressor-decode/qualification.json`, the native suite
  log, fixture reproduction, shader provenance and push-layout receipts. The
  per-phase Engram receipt is `target/v41-engram-backward/qualification.json`.

```powershell
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_compressor_decode_grad.py
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41::compressor_grad::decode_tests -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41 -- --nocapture
python hierarchos-vulkan/validation/check_shader_provenance.py
python hierarchos-vulkan/validation/check_push_constant_layouts.py
```

Next: complete attention/model tape wiring (the shared-source cache aggregation
across steps), then the conditional absent-gradient optimizer and per-parameter
counters, then the frozen-model identity binding and the complete
loss-to-factor/resumed-optimizer model gate. Full-model training and new
adapter-target admission stay disabled until those model gates pass.

## Phase: close the native compressor decode parity gap (2026-10-27)

### Measured diagnosis and correction

The previous decode phase's native replay measured `4.77e-7` in gradients and
`2.38e-7` in an emitted latent at hidden 16 / head 8, so it could not be part
of a full-model strict parity qualification. That replay is now a hard gate.
Its forward projections, pooled values, latents, finite state entries and all
returned gradients are checked under the same `<2e-7`, `rtol=0` contract.

The earlier attribution to projection reduction order was incorrect. A direct
CPU reduction probe and native stage readbacks show both projected KV and
projected scores are bit exact at the failing step. The first differing
operand is the pool: one channel differs by `2.98e-8`, which changes variance
and rstd enough to produce the observed latent error. The shared linear
forward source and artifact are unchanged by this fix.

CPU non-last-axis softmax vectorizes over channels, using one host SIMD vector
at a time; incomplete channel vectors use scalar `std::exp`. The compressor
used SLEEF vector exp for every channel, including those scalar tails. Both
paths normalize with separately rounded division, so changing normalization
to reciprocal multiplication would address the wrong operator. This layout
is explicit in the pinned
[CPU softmax implementation](https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/SoftMaxKernel.cpp).

`fp32_v41_compressor_exp.glsl` now shares the correct vector/tail choice across
the forward pool, prefill pool adjoint and cached-decode pool adjoint. Each
host dispatch supplies the existing detected ATen vector width. The scalar
tail reuses the already qualified native FP32 scalar-exp helper; no CPU
activation computation or host readback is introduced. Three existing SPIR-V
artifacts are rebuilt with the updated private push layouts. No execution
flag, checkpoint format or model default changes.

### Regression gates

The existing ten-chain, 32-step fixture is unchanged. Both its 6,808
CPU-operand gradients and the 6,808 gradients replayed from native forward
operands now have **zero absolute drift**. All 8,684 checked native forward
operands, including projected KV/scores, pool values, emitted latents and
finite state entries, also have zero drift. Unread reference score slots
initialized to negative infinity retain their explicitly documented exclusion;
every finite reference entry is checked with finite native values and exact
shape validation.

An additional independent CPU autograd fixture isolates the pool at twelve
channel widths: **1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 65**. It checks
forward pooling and both KV/score VJPs around the SIMD cutoffs, with two batches
and two groups per batch. All **4,640** frozen values have zero drift. Inputs,
weights, seeds, fixture magnitudes and the original tolerance were not changed.

Together the gates record **26,940 numerical comparisons**:

| subset | values | worst absolute drift |
| --- | --- | --- |
| original CPU-operand decode gradients | 6,808 | **0** |
| native-operand decode gradients | 6,808 | **0** |
| native forward/state operands | 8,684 | **0** |
| isolated SIMD/tail pool and VJPs | 4,640 | **0** |

The new replay fails on the previous artifacts, preserving the original
`2.38e-7` latent failure and `4.77e-7` gradient measurement in
`target/v41-compressor-native-parity/baseline-measured.log`. It passes after
the exp-path correction. The two public decode operand types are now exported
from `deepseek_v41`, and the regression imports them through that public path.

### Verification and remaining scope

- Full default V4.1 suite: **161 passed, zero failures, 23 ignored**.
- **546 shader artifacts** reproduce byte-for-byte. Manifest metadata is
  unchanged; only the forward/prefill/decode compressor pooling artifacts
  change in this phase.
- Push layouts: **426 shader blocks, 109 host structs**, PASS.
- The original decode fixture regenerates with its identical SHA-256
  `17f46f208faf573ec127e446258d4fcd07ba4507921e55e9ec33009aeb291edf`;
  the new pool fixture also regenerates byte-for-byte.
- Native validation runs on the machine's **AMD Radeon Graphics** against the
  pinned CPU PyTorch 2.14.1+cpu reference. CUDA parity is not required.
- Evidence: `target/v41-compressor-native-parity/qualification.json`, final
  suite log, baseline failure, fixture reproduction, shader provenance,
  push-layout receipt and projection-order probe.

```powershell
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_compressor_decode_grad.py --reference-path .tmp-v41-ref/model.py
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_compressor_pool_math.py
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41::compressor_grad::decode_tests -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41 -- --nocapture
python hierarchos-vulkan/validation/check_shader_provenance.py
python hierarchos-vulkan/validation/check_push_constant_layouts.py
```

This closes the reported native decode composition gap at the measured FP32
geometries. Prior head-512 prefill and larger-magnitude projection sensitivity
limits remain separate. Full-model training still requires attention/model
tape wiring, shared-source/cross-step cache gradient aggregation, conditional
optimizer steps with per-parameter counters, retained-activation budgeting,
frozen-model identity binding and the complete loss-to-factor/resume gate.

## Phase: conditional adapter AdamW, factor counters and mixed-MoE resume (2026-10-27)

### Update semantics and public API

The component trainer previously required every module's A/B gradient on every
call and used a shared global step for bias correction. That could not consume
the mixed MoE adjoint's honest omission of unvisited experts. `step` now accepts
the active projections directly: an omitted target is equivalent to
`grad=None`, with no decay, moment update or parameter-step increment.
`step_parameters` additionally accepts independent optional A/B gradients.
An explicitly present zero gradient still updates, including weight decay and
moment evolution. `step_complete` preserves a caller's explicit requirement
that every projection be present.

Each factor has its own update counter. `parameter_step_counts` exposes the
A/B pair; `step_count` counts optimizer calls, including all-absent calls.
Bias corrections use the corresponding factor's count. Unknown/duplicate
targets, malformed gradients and non-finite input remain errors. The full
active update set is validated before mutation, including a numerical dry run
that rejects finite gradients which would overflow the factor or moments.
The dry run repeats scalar arithmetic without cloning factor/moment arrays,
so temporary storage scales with the number of active parameters rather than
the entire state size. Counter and call-count overflow also reject atomically.

The pinned CPU implementation filters parameters on `p.grad is not None` and
updates each retained parameter's step before its decay/update. Its installed
source snapshot is retained at
`target/v41-conditional-adamw/pinned_adam.py`, from CPU PyTorch 2.14.1+cpu,
revision `5c4886908584029761b579af026dcfb627c84070`. The existing native FP32
AdamW arithmetic is otherwise preserved; its original dense trajectory still
passes.

### Checkpoint schema 3 and compatibility

New native optimizer checkpoints store U64 `step_a` and `step_b` tensors beside
the four FP32 moment tensors for each module. These counters share the existing
optimizer payload's SHA-256 binding. Resume checks exact tensor dtype/shape,
counter range and relation to optimizer calls, and requires zero moments for
never-updated factors. The trained adapter/config and file-payload bindings
remain intact. Standard PEFT export remains an unmerged adapter directory.

Bound schema-2 checkpoints remain readable. That schema required every factor
on every optimizer call, so its global step migrates to both factor counters.
The migration test converts a valid dense checkpoint to the old four-tensor
layout, resumes it, continues the identical trajectory and writes schema 3.
Unbound schema 1 remains rejected. Negative tests forge counter payloads with
updated file checksums to verify semantic rejection of wrong dtype, shape,
range and zero-count/nonzero-moment combinations, rather than merely testing
the outer checksum.

### Conditional and computed-gradient qualification

`validate_deepseek_v41_conditional_adamw.py` freezes two nine-call CPU AdamW
trajectories, with LoRA and rsLoRA metadata. They cover absent targets,
independently absent A/B, present-zero gradients, late first use, all-absent
calls and a permanently unused target. Factors and all moments are checked
after every call; factor counters are exact integers. Native before/after
bit comparisons also prove each absent factor and its moments stay unchanged.
Two fixed projection/classification probes check the resulting logits.

Checkpoint/resume is repeated at call counts 0, 3 and 6, including never-used
factors and divergent A/B counters. Resumed factors, moments and native probe
outputs match the uninterrupted trajectory as FP32 bits, with equal counters.
Final standard PEFT export/reload preserves native probe output bits.

The same fixture also trains the existing single-token mixed-MoE component
against CPU autograd for three real gradient steps. Its six visited adapter
modules update while nine inactive modules remain unchanged. Forward and
backward checkpoint-stage routed/shared experts; backward restages base
payloads without adapters and uses retained factor snapshots. Each step checks
computed dA/dB, factors, moments, counters and pre/post-update output. The
third step is repeated after resuming call 2, and both the optimizer state and
updated output match the uninterrupted native run exactly. Trained adapter
export reloads through the normal checkpoint adapter attachment path.

### CPU gate (`<2e-7`, `rtol=0`)

The new gates record **44,908 numerical comparisons**, including resumed
repeats, with worst drift **1.490116119384766e-8**:

| subset | values | worst absolute drift |
| --- | --- | --- |
| projection/classification logits | 396 | 1.117587089538574e-8 |
| mixed-MoE pre/post-update outputs | 256 | **0** |
| factors | 13,728 | 1.490116119384766e-8 |
| moments | 27,456 | 1.490116119384766e-8 |
| computed mixed-MoE factor gradients | 3,072 | 3.492459654808044e-10 |

The state comparisons additionally check **360 factor counters** against CPU
exactly; native resume/skip identity checks are separate. No old fixture,
operand magnitude or tolerance was changed. Unit fixtures pin supplied
gradients; the mixed-MoE fixture computes them with the unchanged released
Expert/Gate/MoE bodies and the established FP32 STE packed-linear contract.
This qualifies the conditional optimizer and component lifecycle, not a
complete model trainer.

### Verification and remaining work

- Full default V4.1 suite: **166 passed, zero failures, 23 ignored**.
- **546 shader artifacts** reproduce byte-for-byte; this phase changes no
  shader or manifest entry.
- The fixture regenerates byte-for-byte at SHA-256
  `2bc7b333db60c0c2bf789920c2bd5050d2e76e958300a17ee84c2efd465f1bae`.
- Malformed late gradients, unknown/duplicate targets, NaN/Inf, arithmetic
  overflow, counter overflow and existing checkpoint integrity gates pass.
- Evidence: `target/v41-conditional-adamw/qualification.json`, full suite log,
  fixture reproduction, shader provenance, original source backups and the
  pinned CPU optimizer source. Test checkpoint packages are cleaned up.

```powershell
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_conditional_adamw.py
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib conditional_adamw -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41 -- --nocapture
python hierarchos-vulkan/validation/check_shader_provenance.py
```

Next: attention/model tape wiring and shared-source/cross-step cache gradient
aggregation, activation/state budgeting and recomputation, frozen-model
identity binding, and the complete model loss-to-factor/save-resume gate.
CLI/GUI full-model training and new adapter-target admission remain disabled.
CPU PyTorch remains the accepted oracle; no CUDA device is required.

## Phase: retained sparse attention and shared cache lifetime gradients (2026-10-27)

### Native read/write graph

`DeepseekV41AttentionTape` wraps the existing native gathered-attention forward,
window-ring seed/decode writes and finalized compressed-cache append. It
borrows the runtime's actual caches; all writes during recording must go through
the tape. Physical cache IDs identify a source once, so multiple readers join
the same compressed source instead of returning unrelated per-layer gradients.
IDs are bound to a graph with private ownership tokens. Duplicate physical
registration, foreign handles, wrong cache kinds, invalid layouts and write
ranges reject before mutation. Backward rejects duplicate, foreign and
undersized seeds before executing its graph.

Each read takes independent device-to-device snapshots of Q, window and
compressed values, both index arrays and the frozen sink. It runs the unchanged
native forward on those operands. Caller scratch and attention projections can
then be reused or dropped without invalidating the VJP. Forward outputs are
independent of the retained operands too. No frozen projection payload is kept
by the tape, and no tensor payload is read back for backward execution.

Backward consumes the tape and visits events in reverse order. It recomputes
each seeded sparse VJP, adds reader gradients to their shared physical cache,
and splits the pending cache gradient at every write. Overwritten rows go to
that write's source; untouched rows go to the preceding cache version. This
includes later decode steps, shared readers at the same step and distinct
compressed sources with independent lifetimes. Prefill uses the complete
chunk for direct attention while only its newest rows seed the ring; direct
and future-ring gradients join at that same seed source. Discarded prefix rows
receive no ring gradient. The returned source is post-quantization/post-RoPE;
composition with projections, rotary adjoints and compressor adjoints is still
model-tape work.

An unseeded output is absent rather than an explicit zero use. Explicit zero
seeds traverse the graph and preserve zero-valued gradients for reachable
operands. Read snapshots release as reverse execution visits them. The public
`retained_bytes` counter reports snapshot payload bytes, excluding cache
allocations, allocator padding, metadata and backward scratch; it is not a
model memory budget or a peak-residency claim.

### CPU autograd graph and native operand qualification

`validate_deepseek_v41_attention_tape.py` freezes three complete CPU graphs with
functional cache-overwrite operations and one autograd backward over their
seeded head outputs. The oracle does not chain local VJPs or use native
gradients. Its sparse forward is the existing pinned FP32 mirror, whose hash
is retained in the fixture. Discrete indices remain fixed. Its measured
post-quantization payloads use the existing independent FP8/UE8M0 window and
FP4/E4M3 compressed quantization mirrors; the native test actually executes
the native quantizer and compares values and scales as bits before using them.
This adds no derivative for a quantizer or discrete selector.

The three graphs cover **71 attention reads and 82 cache writes**:

- A sliding-only source, prefill length 6, ring size 4, seven decode steps.
- Batch 2 with three readers sharing one ratio-2 compressed source, odd prefill
  length 3 and eight decode steps, including steps with no completed group.
- Four readers sharing two independent ratio-1 compressed sources, prefill
  length 2, ring size 3 and six decode steps.

Coverage includes ring wraps, shorter/longer-than-ring prefills, incoming cache
gradients, masked and duplicate compressed gathers, absent output seeds and
explicit zero seeds. The test checks every native cache update against the
CPU cache bits. It overwrites caller Q/index/source scratch and the original
sink after each native use, drops each attention object's projections before
backward, and still passes the frozen oracle. Separate exact tests prove that
discarded prefill rows receive zero ring gradient and distinguish an absent
backward seed from an explicit zero seed. Invalid reads leave event counts,
retained-byte counts and forward cache values unchanged.

### Strict CPU gate and evidence

There are **42,848 numerical comparisons** across 232 stage records, worst
absolute drift **1.1920928955078125e-7**, with unchanged `<2e-7`, `rtol=0`:

| subset | values | worst absolute drift |
| --- | --- | --- |
| native sparse head outputs | 17,024 | 3.725290298461914e-8 |
| query gradients | 16,512 | 5.587935447692871e-9 |
| joined write-source gradients | 5,920 | 1.192092895507813e-7 |
| incoming-cache gradients | 3,392 | 1.490116119384766e-8 |

Exact quantized values/scales, forward cache payloads, absent-gradient states
and ownership tests are additional assertions, not included in that numerical
comparison count. Snapshot payloads in the three graphs are 9,548, 115,776 and
145,696 bytes respectively. No existing fixture, operand magnitude, tolerance,
shader or shader-manifest entry was changed. An initial unquantized prototype
fixture was archived under the phase evidence directory before native gating;
the qualified fixture was frozen with actual quantized payloads and thereafter
regenerates byte-for-byte at SHA-256
`025a847d418b2ae1d044b49e7c12a4eb3fbe765db16b7ae95dfae3db72d914a7`.

- Targeted gate: **4 passed, zero failures**.
- Full default V4.1 suite: **170 passed, zero failures, 23 ignored**.
- **546 shader artifacts** reproduce byte-for-byte; all artifact hashes match
  the preceding conditional-optimizer phase.
- Push layout check: **426 shader blocks, 109 host structs**, PASS.
- Scoped Rust formatting check passes.
- Execution: this machine's AMD Radeon Graphics, with pinned CPU PyTorch
  2.14.1+cpu revision `5c4886908584029761b579af026dcfb627c84070` as oracle.
  CUDA is not required.
- Evidence: `target/v41-attention-cache-chain/qualification.json`, targeted and
  full-suite logs, fixture reproduction, shader provenance, push-layout log and
  source backups. No model checkpoint package is created by these tests.

```powershell
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_attention_tape.py
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib attention_tape -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41 -- --nocapture
python hierarchos-vulkan/validation/check_shader_provenance.py
python hierarchos-vulkan/validation/check_push_constant_layouts.py
```

The shared sparse/cache lifetime component is now qualified. Next: compose
its Q and post-quantization write-source gradients through the retained
attention projections, norms, rotary/finalizer and compressor state, integrate
those with the HC/Engram/MoE model tape, and add activation budgeting,
frozen-model identity binding and the complete loss-to-factor/save-resume gate.
This phase does not enable CLI/GUI full-model PEFT training or change the
adapter-target allowlist.

## Phase: connected attention projections, normalization and rotary VJPs (2026-10-27)

### Native component composition

`DeepseekV41AttentionProjectionAdjoints` adds retained input/output projection
tapes on both sides of the sparse/cache graph qualified in the preceding
phase. The input path is `wq_a -> q_norm -> wq_b -> RoPE` for Q and
`wkv -> kv_norm -> RoPE -> FP8` for window KV. The output path is
`inverse RoPE -> FP8 -> grouped wo_a -> wo_b`. Forward uses the existing mixed
projection, norm, rotary, activation quantizer and grouped-output kernels.
Backward composes their existing adjoints, including the established FP32
straight-through contract for dtype/quantizer rounding. No shader or
quantization rule changes in this phase.

`forward_inputs` returns the exact query latent for the discrete indexer, Q and
the prepared window values. The cache tape owns the subsequent seed/decode
writes and attention reads. `backward_output` produces sparse-output seeds;
after cache backward has joined all readers/steps, `backward_inputs` consumes
the returned Q and write-source gradients. An optional explicit latent edge
can join another independently differentiated objective; no gradient is
invented for discrete index selection. It is tested both alone and together
with the normal Q path.

Tapes own activations, norm gains and the projection tape's adapter-factor
snapshots. Frozen projection bases are not retained; backward receives them
again, allowing the caller to restage a layer with no adapters attached.
`adapter_gradients` returns only visited canonical projection targets, for
accumulation at the optimizer boundary. Per-call gradients of a target must be
summed before one optimizer update; the gate sums them in reverse step order
and compares to the CPU graph's accumulated leaf gradients. An absent Q/KV/
latent edge skips its projections, while an explicit zero edge emits genuine
zero gradients for its reachable projections. Public forward products can be
recycled without corrupting private backward operands.

Geometry, batch/context/dispatch capacity, device and all external gradient
edges are validated before branch execution. Backward also compares the
recorded attention configuration and stream/query precision. Supplying the
same frozen base remains the caller's component contract; whole-model base
identity binding is still required before a model trainer can resume safely.

### CPU oracle and measured graphs

`validate_deepseek_v41_attention_projection_grad.py` freezes four connected
graphs with one CPU autograd backward over their seeded outputs and explicit
latent edges. The graph uses functional cache copies so no decode edge is
detached. RMSNorm, complex-table rotary, the sparse sink expression and the
block-diagonal output follow the pinned released attention body; the existing
mixed-projection and quantizer mirrors define the previously qualified STE
rounding edges. The released rotary functions are loaded from the local
hash-verified source at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`, source SHA-256
`e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d`.

The graphs have **16 prefill/decode steps** in total:

| case | batch | hidden / query rank | heads / dim | output groups / rank | ring | prefill / decode |
| --- | --- | --- | --- | --- | --- | --- |
| FP32 sliding LoRA | 1 | 32 / 16 | 2 / 32 | 2 / 8 | 4 | 3 / 4 |
| FP32 compressed rsLoRA | 2 | 64 / 16 | 4 / 32 | 2 / 8 | 3 | 2 / 3 |
| dense BF16 sliding LoRA | 1 | 64 / 32 | 2 / 32 | 2 / 16 | 3 | 4 / 2 |
| packed FP8 compressed rsLoRA | 1 | 64 / 32 | 2 / 32 | 1 / 32 | 4 | 3 / 3 |

All four projections (`wq_a`, `wq_b`, `wkv`, `wo_b`) are adapted with nonzero
A and B factors. The compressed cases use ratio 2, fixed masked selections and
external compressed-cache operands. Their incoming compressed gradients are
checked, but their compressor/finalizer is deliberately outside this graph.
The preceding phase covers sharing among multiple source readers; this phase
connects a layer's projection and cache lifetimes across its token steps.

Every step checks native Q, latent, window KV, sparse head output, final
attention output and a fixed 11-class logit projection. All 16 attention
outputs additionally match the ordinary native `prefill`/`decode_step` path
as FP32 bits, including BF16/FP8 boundaries and adapters. The first decode
output has no backward seed while its KV is still used by later steps; its
Q gradients stay absent and its KV/input gradients remain present. The last
step also has an explicit latent seed. Public input/query/latent/window
scratch is overwritten after use.

Before backward, the test drops the ordinary-inference instance and retires
only the retained-forward instance's projection bundle. It poisons the old
norm gains, verifies that the bundle's live-buffer-byte release exceeds the
frozen payload minimum, and then drops that attention instance too. Measured
released bytes / minimum frozen bytes are **18,376 / 14,336**, **35,280 /
28,672**, **43,272 / 36,864**, and **22,808 / 16,384**. Backward succeeds with
fresh frozen bases and no adapters attached, proving that the retained graph
does not need the forward projection bundle. Individual factor-snapshot
mutation behavior remains covered by the existing linear-tape gate.

### Strict qualification results

The connected gate records **22,223 values** over 202 stage comparisons, with
worst drift **1.1920928955078125e-7**, unchanged `<2e-7`, `rtol=0`:

| subset | values | worst absolute drift |
| --- | --- | --- |
| Q / query latent / window KV forward | 4,080 | **0** |
| sparse head output | 2,496 | 1.192092895507813e-7 |
| complete attention output | 1,632 | 1.117587089538574e-8 |
| fixed classification logits | 319 | 1.490116119384766e-8 |
| sparse-output gradient | 2,048 | 5.122274160385132e-9 |
| Q gradient | 2,048 | 2.793967723846436e-9 |
| joined window-source gradient | 928 | 3.725290298461914e-9 |
| normalized stream input gradient | 1,632 | 1.117587089538574e-8 |
| query-latent gradient | 544 | 1.338776201009750e-9 |
| incoming window/compressed-cache gradient | 864 | 1.164153218269348e-9 |
| accumulated LoRA A/B factor gradients | 5,632 | 1.490116119384766e-8 |

No prior fixture, tolerance, operand magnitude or kernel arithmetic was changed.
The new fixture regenerates byte-for-byte at SHA-256
`a8768ff06aa20e674188af480250f8d1e0aa7cf2b024d05603d60cd46164752f`.
Prior larger-magnitude packed-projection and head-512 prefill sensitivity
limits remain outside these measured geometries.

- Targeted fully qualified module gate: **3 passed, zero failures**.
- Full default V4.1 suite: **173 passed, zero failures, 23 ignored**.
- **546 shader artifacts** reproduce byte-for-byte; all hashes match the
  preceding cache-lifetime phase and the manifest is unchanged.
- Push-layout check: **426 shader blocks, 109 host structs**, PASS.
- Scoped Rust formatting and whitespace checks pass.
- The CPU reference is PyTorch 2.14.1+cpu revision
  `5c4886908584029761b579af026dcfb627c84070`; native execution uses this
  machine's AMD Radeon Graphics. CUDA remains unnecessary.
- Evidence: `target/v41-attention-projection-chain/qualification.json`, final
  targeted/full-suite logs, fixture reproduction, shader provenance,
  push-layout log and source backups. No model checkpoint package is created.

```powershell
.venv-vulkan/Scripts/python.exe hierarchos-vulkan/validation/validate_deepseek_v41_attention_projection_grad.py
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41::attention::projection_grad -- --nocapture
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41 -- --nocapture
python hierarchos-vulkan/validation/check_shader_provenance.py
python hierarchos-vulkan/validation/check_push_constant_layouts.py
```

Next: connect compressed-cache source gradients through the compressed
finalizer/rotary and retained compressor prefill/decode state, then integrate
the attention, HC, Engram and MoE model tape. Activation/state budgeting,
frozen-model identity binding and the complete model loss-to-factor/
save-resume gate are still required. This component gate adds no CLI/GUI
full-model training admission and changes no adapter-target allowlist.

## 2026-10-27 — Complete model training integration, qualification still active

The retained compressor/finalizer, reverse cache cursor, attention orchestrator,
shifted HC backbone, Engram and MoE adjoints now compose into a native
loss-to-factor trainer (`model_train.rs`). `DeepseekV41ModelTrainer` supports
LoRA/rsLoRA initialization, full prefill/decode gradient chains, one conditional
AdamW update, standard adapter export, and exact native save/resume. Checkpoints
bind the immutable base/config/token artifacts. Normal `transformer-finetune`
dispatch uses this trainer. No production checkpoint was downloaded or run.

MoE forward/backward uses each expert's actual compact row count and skips
unvisited V4.1 experts. The full-model tape recomputes MoE operands one layer at
a time. Frozen projections retire between uses; vocabulary input gradients read
bounded F32/BF16 tiles. Optimizer schema 4 allocates/saves moments only after a
factor's first gradient; schema 2/3 resume remains supported. New checkpoints
record the math policy; old files without that field retain their legacy FP32
AdamW behavior. The corrected CPU policy keeps hyperparameters in FP64, uses
ATen's lerp association, and multiplies addcdiv's numerator before division.

Verified dense and hybrid packed complete-model qualification:

- Three default AdamW steps each for LoRA and rsLoRA, on the unchanged six-layer
  shifted-HC/Engram/indexed/shared-cache model with prefill and cached decode.
- **512,832 values in 9,228 comparisons**, including **3,072 logits**: maximum
  absolute drift **1.7881393432617188e-7**, strict `<2e-7`, `rtol=0`.
- Native resumed loss/factors and standard exported-adapter inference are
  bit-exact to uninterrupted native execution.
- Resident versus streamed logits, factor gradients and updates are bit-exact
  for the dense model and both BF16/packed HC2/HC4 models; initial CPU logits
  are bit-exact. These identity checks do not replace the separate CPU training
  oracle for the BF16 model.
- Five full-model tests passed in `target/v41-completion/full-model-row1-grad.log`.
  Six optimizer/conditional/sparse/legacy-state tests passed in
  `target/v41-completion/adamw-state-regression.log`.
- Scalar cross entropy is checked bit-exactly on identical frozen CPU logits.
  Coupled loss can differ by one FP32 ULP near 3.0 despite compliant logits;
  its drift is recorded and checked against the input/rounding error bound.
  The user's hard logit tolerance is unchanged.

The CLI lifecycle also trained a **291,183-byte** checkpoint, resumed at step 2,
and produced identical adapter bytes to an uninterrupted two-step run. That
smoke preceded the last AdamW policy correction and must be rerun on the final
build. Durable partial evidence is in
`target/v41-completion/training-qualification.json`; it explicitly does **not**
declare the overall user goal complete.

The full BF16 oracle checks its initial four prefill/decode forwards bit-exact
against the frozen released-model fixture before generating gradients. Its
first unqualified version omitted the Expert's BF16 input cast before the
adapter branch of w2. That rejected artifact (SHA `9b36f6cc…`) is archived under
`target/v41-completion/rejected/`, not overwritten or counted as accepted.
The corrected versioned oracle is `deepseek_v41_full_bf16_training_v2.json`, SHA
`77a432fcb453f4c6dd491df2c003bf46c81e8ebc8a34ef5093142f8966dc4336`. (That file
is superseded by `_v3` with identical evidence values; the only later change is
the indexer bf16 score boundary documented at the end of this record.)
Its native gate exposed a separate narrow weight-gradient GEMM reduction and is
being rerun after a shape-based CPU reduction fix. No tolerance, operands,
optimizer defaults or validated expectation fixture was adjusted for a pass.

Generic architecture streaming work is also active. Ordinary native linears can
now use package-backed bases through their unchanged LoRA forward/backward and
optimizer branches. Scratch exists per submitted operation, not per parameter.
Direct, transposed Conv1D and contiguous packed-expert views are verified against
the loaded values. Routed expert projections reuse their original row-count
kernels. Llama (14 linears), Mixtral (28) and GPT-2 (8) pass bit-exact native
base/PEFT logits and two-step AdamW adapter checks; their retired resident bytes
are 73,840 / 173,792 / 68,192. This is **post-load** conversion with incomplete
specialized-operator coverage, not a qualification of all 146 architectures or
of bounded initial loading. Those requirements, production-ratio training
probes, final regressions, documentation and scratch cleanup remain active.

## Pinned-replay optimizer qualification: the `eps` amplification explained, and a logit gate that survives it

The six-layer / 384-expert production-ratio training gate was failing on
**optimizer state** (up to 7.5e-7) and, one step later, on a decode logit
(7.79e-2) while every gradient stayed inside the gate. Both are now understood
exactly, and neither is carried by the gate any more.

**Mechanism.** AdamW divides by `sqrt(exp_avg_sq)/bias_correction2_sqrt + eps`.
At an element whose gradient is near zero that coefficient's magnitude reaches
`step_size*(1-beta1)/eps = 1.0e5`, so a one-or-two-ulp gradient difference
becomes a much larger factor difference. Measured on the failing case
(`target/v41-completion/native-4141384-lora-0-all-grads.json`, element 59 of
`layers.0.ffn.experts.251.w2 B`): gradient drift 2.18e-11 against a 1.04e-4
gradient (6 of 64 elements bit-exact, the rest one or two ulps), realized
amplification 8.1e4, factor drift 2.28e-7 at step 0. Accumulated over three
steps the same mechanism reached 7.36e-7 on `layers.3.attn.wq_b A`, flipped a
discrete top-6 routing decision and produced the 7.79e-2 decode logit. The
gradient difference itself is ATen's GEMM reduction order against the native
reduction; it is inside the gate and is not a defect.

**Qualification change.** Each compared step now starts from the reference's
own state. Before every step the gate pins every adapter factor *and* optimizer
moment to the frozen fixture state (`DeepseekV41AdapterTrainer::load_state_bits`,
exposed to the model trainer as `load_adapter_state_bits`). The gradient is then
the only differing input, so each step is qualified on its own arithmetic and an
already gated ulp difference cannot be amplified into a different trajectory.
No fixture, threshold, `rtol`, oracle, optimizer default or expected value was
changed; the propagation itself is unchanged and is still measured. One oracle
boundary was corrected in a later pass (the indexer's released bf16 score
arithmetic, documented below); no tolerance, `rtol`, threshold or native kernel
changed then either, and the superseded fixtures are listed as rejected
evidence.

**What the pinned gate asserts** (four oracles x LoRA/rsLoRA x three AdamW steps):

- every logit strictly `< 2e-7`, `rtol = 0` - the user's release gate;
- every factor gradient and every optimizer moment `< 2e-7`;
- every active factor's live result **bit-exactly** the documented pinned
  single-tensor AdamW policy applied to the native gradient from the pinned
  state, and the fixtures' own recorded trajectories reproduce that policy;
- factors without a gradient this step bit-identical, in the fixture and in the
  replication;
- the residual against the fixture bounded by the *exactly computed* response of
  that policy to the gated gradient difference
  (`policy(native gradient) - policy(cpu gradient)`), plus `8*2^-24` of the
  largest operand for the roundings each implementation performs (ATen's
  vectorized `sqrt` is not correctly rounded; measured within 1 ulp of `denom`);
- that response is additionally bounded by the mean-value form
  `2*step_size*(1-beta1)*|dgradient|/(|gradient|+eps)^2 + the operand term`,
  which is where the `eps` amplification is stated rather than hidden.

| fixture (2 methods x 3 steps) | logits (values / worst) | gradients worst | factors worst | moments worst | native update bit-exact | policy bound ratio |
| --- | --- | --- | --- | --- | --- | --- |
| f32 dense | 1536 / 1.49011611938476562e-07 | 7.45058059692382812e-09 | 2.14204192161560059e-07 | 9.31322574615478516e-10 | 62184/62184 | 0.990 |
| fp32 + FP4 routed / FP8 shared | 1536 / 1.49011611938476562e-07 | 7.45058059692382812e-09 | 6.70552253723144531e-08 | 6.98491930961608887e-10 | 79464/79464 | 0.985 |
| BF16 + FP4 routed / FP8 shared | 1536 / 0.0 (bit-exact) | 1.11758708953857422e-08 | 6.70552253723144531e-08 | 1.39698386192321777e-09 | 87624/87624 | 0.987 |
| production ratio, 6 layers / 384 experts, HC4 | 1536 / 0.0 (bit-exact) | 8.38190317153930664e-09 | 6.71272573526948690e-07 | 6.98491930961608887e-10 | 5356104/5356104 | 0.999972 |

The optimizer arithmetic is therefore exact in the sense that matters: given the
same operands it produces the same bits, and the only residual against CPU
PyTorch is the update's response to a gradient difference that is itself gated.
Evidence: `target/v41-completion/full-model-pinned-{dense,mixed,bf16,production}.log`,
per-step receipts `target/v41-completion/pinned-*-{0,1,2}-receipt.json`,
aggregates `pinned-*-summary.json`, table and totals via
`target/v41-completion/summarize_pinned_replay.py`, durable receipt
`target/v41-completion/training-qualification.json`.

### The last failing step was the indexer's released bf16 boundary, not the native

The production-ratio training gate was the only remaining red item: steps 0 and 1
matched bit-exactly and the second decode of step 2 differed by `4.88e-2`. Three
observations narrowed it to a discrete edge, and the fourth named it.

**Localization.** The stage trace (2,156 records) had 2,037 bit-exact entries and
one contiguous divergence, `5/5/attended` (`4.83e-2`), with every recorded
upstream stage of that layer at `1.49e-8` or better. A 4.83e-2 jump after
ulp-level inputs is a selection change, not arithmetic.

**Identification.** Replaying the pinned state and re-running only layer 5's
attention with each candidate index set forced in turn reproduced the native
trace **bit-for-bit** for exactly one hypothesis: row 0 selected compressed
positions `{0, 1}` while the CPU oracle selected `{0, 4}`; row 1 already agreed.
The harness was validated by first reproducing the `4.83e-2` difference with the
oracle's own selection restored.

**Cause.** New index-stage traces showed the native's masked index scores were
**exactly bf16** numbers (`-1.787109375e-01`, `1.5234375e-01`, ...) while the
oracle's were fp32. Recomputing the released expression on real bf16 tensors -
`einsum(q, k)` (bf16 output), `relu * weights` (bf16 product), `.sum(dim=2)`
(bf16 reduction) - matched the native **bit-exactly at every sampled layer and
position** (`6/6`, `12/12`, `20/20`, `50/50` values, prefill and decode), and the
candidate masks and mapped top-k agreed too. The released indexer runs on bf16
tensors, so those three roundings belong to its arithmetic. At this decode they
make compressed positions 1 and 4 **exactly equal** (`-7.32421875e-02`), which
is a tie the pinned CPU `torch.topk` staging resolves to position 1; the fp32
mirror kept a `3.5e-4` gap and resolved it to position 4. One tie, one different
KV position, `4.9e-2` in the decode logit.

The defect was therefore in
`validation/validate_deepseek_v41_full_bf16_training.py`, which mirrors the
released dtype flow with explicit fp32 STE cast boundaries but had left the
indexer's internal sum in fp32. `install_indexer` now re-instates
`Indexer.forward` with those three roundings and everything else unchanged (key
cache write, candidate pre-filter, top-k mapping). The native needed no change:
it was already faithful to the released bf16 scores, and its tie behaviour
already matched the pinned `torch.topk` staging.

**Evidence refresh.** `fixtures/deepseek_v41_production_ratio_training_v2.json`
and `fixtures/deepseek_v41_full_bf16_training_v3.json` replace the two fixtures
produced through the old mirror; both are regenerated, not edited, and the
superseded files are recorded in `target/v41-completion/training-qualification.json`
under `rejected_oracles`. The BF16 v3 payload is otherwise **identical** to v2
(that trajectory contains no bf16 score tie), so only the production-ratio
trajectory changes, and only at the affected decode. No tolerance, `rtol`,
threshold, optimizer default or shader changed.

**Result** (all four gates re-run on the corrected evidence):

| gate | logits | gradients | native update | bound ratio |
| --- | --- | --- | --- | --- |
| production ratio, 6 layers / 384 experts, HC4, 6 steps | 6 steps x 256 values (1,536), **all bit-exact** | `8.38190317153930664e-09` | `5356104/5356104` | `0.999972` |

Dense, packed and BF16 gates still pass unchanged (`test result: ok` in
`target/v41-completion/gate-{dense,packed,bf16}-after-fix.log` and
`full-production-gate-after-fix.log`). Aggregate pinned replay:
**6,144 logits / 16,896 gradient / 87,384 factor / 34,092 moment comparisons**,
worst logit `0.0` (production and BF16) and `1.49e-07` (fp32 fixtures), worst
gradient `1.12e-08`, worst moment `1.40e-09`, `5,585,376/5,585,376` factor
updates bit-exact to the pinned policy.

**Diagnostics added** (test-only, no production path): the retained backbone
records `index_scores`, `index_topk` and `index_candidates` per layer/position,
and `V41_MIN_STEP` starts a pinned replay at a later step, so a single failing
decode can be reproduced in about 45 seconds instead of a full trajectory.
Probes and logs: `target/v41-completion/{compare_index_stage,identify_native_selection,probe_index_selection}.py`,
`target/v41-completion/{compare-index-stage,identify-native-selection,probe-index-native-step2}.log`.

## DeepSeek V4.1 PEFT training closure

**Status: qualified.** The loss-to-adapter chain, the model trainer, AdamW with
factor-local counters and sparse moments, bit-exact save/resume, standard PEFT
export and the CLI route are landed and gated under the unchanged `<2e-7`,
`rtol = 0` CPU PyTorch gate. This closes *What remains for V4.1 LoRA training*
and the PEFT half of the Intel handoff for this machine. Inference
qualification (1,472,649 logits) is untouched: no forward kernel changed.

The receipt is
`validation/release_parity/deepseek_v41/training-qualification.json` (working
copy `target/v41-completion/training-qualification.json`), hash-bound into the
machine record (`validation/release_parity/deepseek_v41/qualification.json`)
together with the three release logs.

### Release suite on the final tree

The checks run as three processes because this machine has 10.7 GB of RAM and a
19 GB commit limit; a single process cannot hold the core graph tests and the
six-layer/384-expert training gate at once. Splitting is a resource boundary,
not a coverage reduction: the union is every V4.1 library test.

| process | result | log |
| --- | --- | --- |
| `deepseek_v41::` minus the training group | **175 passed, 0 failed, 23 ignored** (223.75 s) | `release-suite-core.log` |
| `deepseek_v41::model::tests::training::` | **6 passed, 0 failed, 1 ignored** (317.58 s) | `release-suite-training.log` |
| production-ratio gate (`--ignored`, own process) | **1 passed**, 924.30 s | `production-gate-final.log` |
| total | **182 passed, 0 failed, 24 ignored** | |

The 24 ignored tests are the documented 40-layer/384-expert graphs plus the
production-ratio training gate, each with its own reason string. The shader
manifest reproduces byte for byte: **548 artifacts identical**, 0 drifted
(`final-shader-provenance.log`).

### Training gate

Four frozen CPU PyTorch oracles drive the end-to-end trainer: dense FP32, FP32
with FP4 routed / FP8 shared experts, BF16 with the same packed experts, and the
production ratio (6 layers, 384 routed experts, top-6, HC4). Each runs LoRA and
rsLoRA for three AdamW steps, saves, resumes at step 2 and exports an adapter
that reloads.

**Every compared step is replayed from the reference's own state**: all adapter
factors and optimizer moments are pinned to the frozen fixture before the step,
so AdamW's `sqrt(exp_avg_sq)/bias_correction2_sqrt + eps` cannot amplify a
one-or-two-ulp gradient difference into a routing flip (the measured coefficient
reaches `1.0e5`; unpinned it flipped a top-6 decision and a `7.79e-2` decode
logit). Under the pin the gradient is the only differing input.

| quantity | result |
| --- | --- |
| logits | 6,144 values over 96 points, worst `1.49e-7`; production-ratio and BF16 fixtures **bit-exact** |
| gradients | 16,896 points, worst `1.12e-8` |
| optimizer moments | 34,092 points, worst `1.40e-9` |
| factor updates | 87,384 points; **5,585,376 / 5,585,376 bit-exact** to the documented pinned single-tensor AdamW policy |
| factor residual vs fixture | worst `6.71e-7`, covered by the *exactly computed* policy response to the gated gradient difference; worst bound ratio `0.999972 < 1` |
| coupled five-test group (earlier phases) | 9,228 comparisons / 512,832 values, worst `1.788e-7`, unchanged |

The production-ratio gate is bit-exact at all six steps: `256` logit values per
step, `892,684` factor updates per step bit-exact, gradients within `8.4e-9`.

### CLI lifecycle

`transformer-finetune --peft` on the dedicated V4.1 route trains, saves state,
resumes and exports without switching APIs. On the final build: initial training
reaches step 1 (loss `2.8776090145111084`), the resumed run reaches step 2 (loss
`2.8774423599243164`), and the two-epoch uninterrupted run produces the same
steps. Both adapters hash to
`30bfffdad0639b68bb9b1d2a3b82ded50c22a2be0b24710bcc726de429b1fefc`
(`cli-lifecycle-final.log`).

### Streaming reach

The same tiered residency (`src/weight_offload.rs`: device -> host cache ->
package, shard headers parsed once) now covers the non-V4.1 loaders for
inference, training and PEFT, including dense parameters and MoE expert
parameters, with FP32 adapters accepted and packed-or-dense expert storage both
streamed. The generic loaders stream the tied/untied vocabulary and input
embeddings at load time (peaks recorded in the earlier entries above), and the
streamed training path keeps frozen vocabulary gradients and optimizer state off
the device across save/resume. V4.1 exercises the same tiers through its own
checkpoint (`from_checkpoint_with_*_residency`, streamed adapters, restaging for
backward).

### What is not claimed

- The full 510 GB checkpoint is not executed on this machine; qualification is
  the agreed scalable criterion (released ratios and features at small
  geometry) plus production-width component and accumulation gates.
- `--peft` still fails closed for `modules_to_save`, DoRA, QALoRA, adapter bias,
  trainable tokens, layer replication, compressor/Engram/grouped-`wo_a`
  targets, F32-storage packages and merged V4.1 export. `architectures --peft
  --json` now reports LoRA and rsLoRA with those exclusions
  (`target/v41-completion/architectures-peft.json`).
- Intel Gen9 kernel optimisation is the next stage and runs on that machine
  (`validation/intel-handoff/DEEPSEEK_V41.md`).

### Reproduction

```powershell
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41:: -- --skip model::tests::training --test-threads=1
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41::model::tests::training:: -- --test-threads=1
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib deepseek_v41_full_model_production_ratio_cpu_training_gate -- --ignored --nocapture
.\.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/check_shader_provenance.py --output target/v41-completion/shader-provenance.json
.\.venv-vulkan\Scripts\python.exe target/v41-completion/collect_training_receipt.py
.\.venv-vulkan\Scripts\python.exe target/v41-completion/attach_training_receipt.py
```

## Phase 25: Intel Gen9 baseline - forward profile and the zero-fill submission cost

The Gen9 brief asks for a profile before an optimization order
(`hierarchos-vulkan/validation/intel-handoff/DEEPSEEK_V41.md`), so this phase
added the harness and the first measured fix instead of guessing at kernels. The
record is `hierarchos-vulkan/validation/DEEPSEEK_V41_GEN9_PROFILE.md`.

- New `hierarchos-vulkan-deepseek-v41-forward-probe` binary: loads the retained
  package through the normal loader (or fully resident), times prefill/decode,
  prints a per-step logit digest, and composes with the existing
  `HIERARCHOS_VULKAN_PROFILE_KERNELS` ranking plus a new
  `HIERARCHOS_VULKAN_SUBMIT_TRACE` per-batch structural trace.
- Profile, 27 forward calls, resident: `v41.moe-packed` 76.0 % of 9,568 ms of
  dispatch brackets and `v41.hyper-connection` 16.7 %, with attention at 2.2 %.
  Hottest kernels: `moe_gate` 30.3 % at 2.68 ms/dispatch, `quantize` 26.0 %,
  `packed_linear` 16.9 %, `hc_split` 16.3 %. GPU owned only about 3.2 s of a 63 s step
  though: one prefill(8)+decode(1) issued 57,411 host submissions, 35,783 of
  them zero-fill staging copies with no dispatch at all.
- Fix: `GpuBuffer::zeros_f32`/`zeros_u32` no longer stage zeros and wait. A
  host-visible allocation memsets through its mapping; a device-local one
  enqueues a `vkCmdFillBuffer(0)` that the next command buffer binding the
  buffer records before its dispatch, between the same transfer/shader barriers
  the upload path uses. A fill is consumed only by the binding command buffer
  and only by the allocating thread, so concurrent graphs cannot steal or
  reorder each other's, and a host write or full upload cancels it.
- Effect on the shipping path: submissions 57,411 -> 32,018, zero-dispatch
  submissions 35,783 -> 10,390, prefill of 8 tokens 5,318 -> 3,470 ms, decode
  503 -> 341 ms/token. Logit digests unchanged (`2d2f8e2d5914a0d2` default
  loader, `90e096ebc5bf0659` resident).
- Verification: `vulkan::tests::` 36 passed including three new deferred-zero
  tests, `deepseek_v41::` 181 passed / 0 failed / 24 ignored, the strict ignored
  release gate passed (3,216 logits, zero drift), and 10 cross-architecture
  `kv_cache_matches_full_prefix_logits` tests passed.
- Next, in profile order: the remaining one-submission-per-dispatch structure
  (21.6k dispatch batches per two calls, plus the per-layer MoE `counts`
  readback), then `moe_gate`'s serial 384-expert loop, then the
  `hc_split`/`quantize`/`packed_linear` grid width. None of it relaxes the
  `max_abs < 2e-7`, `rtol = 0` gate.

## Phase 26: one command buffer per layer, and the two lifetimes a submission was hiding

Phase 25's profile put host submission, not GPU time, at the top of the order.
This phase records a whole layer's stages into one `ComputeSession` and removes
the requirement that a stage submit and wait before the graph continues.
`HIERARCHOS_VULKAN_LAYER_SESSION=0` keeps the previous one-submission-per-stage
path as the A/B arm. Record:
`hierarchos-vulkan/validation/DEEPSEEK_V41_GEN9_PROFILE.md`.

- `ComputeSession`/`ComputeSessionScope`/`BatchHandle` in `vulkan.rs`: a stage's
  `submit` becomes a publication point, and a host read or streamed upload
  retires the recorded prefix and continues in the replacement buffer. Retirement
  no longer refuses to split while a stage is recording; it records a shader
  barrier and submits, which is what a streamed Engram table staging read needs.
- Measured on the retained joint package: submissions per prefill(8) 16,162 ->
  252 and per decode 5,632 -> 252; resident prefill 2,207-2,216 -> 457-537 ms and
  decode 106.8-106.9 -> 20.3-20.4 ms/token. Both arms produce the frozen digests
  `90e096ebc5bf0659` (resident 8/8) and `2d2f8e2d5914a0d2` (default loader 8/4).
- Two correctness facts the submission boundary had been providing for free:
  a stage boundary needs an explicit compute-to-compute barrier (the per-dispatch
  elision compares exact buffer keys, and two stages can alias one allocation
  through different views), and a command buffer must keep its bound allocations
  alive until its wait completes. `retain_bound_buffers` plus keeping the batch
  in `into_submitted` fixed the second; without it, outputs differed
  nondeterministically from layer 0 because a freed allocation was reused by a
  later allocation while an unexecuted dispatch still read it.
- New diagnostics: `VulkanDevice::submission_count()` (per-phase `submissions=`
  in the probe), `HIERARCHOS_VULKAN_LAYER_DIGEST` with
  `HIERARCHOS_VULKAN_DIGEST_ONLY=<stage>` for release-build per-stage FNV
  digests, and `HIERARCHOS_VULKAN_MOE_TRACE` for routing/expert digests.
- Keeping the whole batch alive until its wait looks equivalent to retaining its
  bound buffers but is not, and it is what the first version did: 18 tests caught
  it (streamed, hybrid and depth-schedule cases, e.g.
  `deepseek_v41_streamed_layer_projections_match_resident_logits` and
  `deepseek_v41_default_package_loader_strict_all_logits`). A streamed upload
  needs its arena back as soon as the submission completes, so the keepalives
  were moved into the device timeline arena
  (`retire_bound_buffers_on_timeline`) instead. With that, the session is the
  default.
- Verification with recording as the default: `deepseek_v41::` 181 passed, 0
  failed, 24 ignored; `vulkan::` 39 passed (including the timeline detach test);
  10 cross-architecture `kv_cache_matches_full_prefix_logits` tests passed; the
  strict ignored release gate
  (`deepseek_v41_default_package_scalable_release_strict_all_logits`) passed with
  3,216 logits and zero drift; the probe reproduces both frozen digests.
- Next: replace the per-layer MoE `counts` readback with a GPU-side
  active-expert worklist, which is the last host read in the forward and the
  reason a call costs 252 submissions rather than one per layer. The
  `max_abs < 2e-7`, `rtol = 0` gate is unchanged.

## 2026-10-30 — RDNA3 qualification and explicit application streaming

The current machine passes the following recorded scope; no production weights
or CUDA device were used:

- All **32 previously-green independent CPU/PEFT profiles** now pass with
  projection/MoE/vocabulary streaming enabled. Worst measured metric is
  **9.000000000813912e-8**, strict `<2e-7`, `rtol=0`. The final attempt had one
  Llama process abort during excessive concurrent compilation (LLVM out of
  memory); its isolated retry passed on the same frozen executable hashes.
  The failed attempt is retained and is not counted as a pass.
- The shared streaming traversal includes frozen routed/shared experts and
  routers beyond each architecture's PEFT target subset. Exact source views
  handle GPT-OSS interleaved/transposed expert storage and Qwen query/gate row
  splits. Sources are proven equal before adoption; shader arithmetic is
  unchanged. The native nine-family matrix passes; seven families also have
  bit-exact resident/streamed LoRA backward and two-step AdamW factors.
- The dedicated V4.1 **uninterrupted** training test carries native factors and
  optimizer moments forward across all three updates, without loading fixture
  state between steps. LoRA and rsLoRA each pass: **1,536 logits total, all
  bit-exact to CPU**. This closes the trajectory evidence gap left by the
  earlier pinned per-step replay.
- Fresh released-schedule inference gate: **40 layers, 384 experts, top-6,
  3,216 logits, zero drift** on AMD Radeon Graphics.
- V4.1 core **176 passed**, training regression **6 passed**, shared Vulkan
  **36 passed**, CLI **41 passed**, GUI **8 passed** (one pre-existing external
  fixture smoke test ignored). The explicit V4.1 application residency test
  passes for F32 and BF16/FP4/FP8 with identical resident/streamed logits.
- Shader provenance: **548 byte-identical artifacts**. Push layouts:
  **428 shader blocks / 109 host structs**, PASS.

`--stream-parameters` is now the explicit CLI switch for Transformer inference
and frozen-base PEFT. The GUI checkbox is unchecked by default and is exposed
only for Transformer inference/LoRA workflows. Application defaults are resident
for both generic Transformers and V4.1. Historical V4.1 package-level library
constructors retain their tiered policy for compatibility; application calls
use the explicit policy API. Resume rejects a residency mismatch. Full mutable
base training and encoder-decoder/cross-attention streaming fail closed.

The rebuilt user-facing CLI passed train/save/resume/export in both modes for
V4.1 LoRA/rsLoRA, Llama and Mixtral: resumed and uninterrupted adapters have
identical bytes, and resident/streamed adapters also have identical bytes.
Cached generation agrees between modes. Normal application code invokes no
Python/PyTorch; those frameworks are only the independent qualification oracle.

**Metadata correction:** retained fixture
`deepseek_v41_production_ratio_model.json` declares **six** reduced-width layers,
384 routed experts, top-6, HC4. Prior training prose incorrectly said forty.
The affected prose/receipt scope labels are corrected; original receipt bytes
are preserved under `target/rdna3-streaming/before/`, and the correction records
their hashes. No measured number, expected tensor or tolerance was changed.
Forty-layer depth is a separate inference proof. The new uninterrupted training
receipt records the actual geometry explicitly.

Durable evidence:
`validation/release_parity/parameter_streaming/qualification.json`,
`validation/release_parity/deepseek_v41/uninterrupted-training.json` and
`validation/release_parity/deepseek_v41/rdna3-streaming-qualification.json`.
All fresh logs and CLI lifecycle details are under `target/rdna3-streaming/`.

Generic body streaming is still a post-load conversion, so it reduces execution
residency while retaining an initial resident-loading peak. Specialized
parameter operators outside `VulkanLinear` may stay resident. Full base-weight
training offload, all 146 registry families, full production weights and Intel
Gen9 hardware performance are not claimed by this pass. Existing V4.1 adapter
variant exclusions remain in force.


## 2026-10-30 — source-first dense/Mixtral loading, strict RDNA3 follow-through

The explicit parameter-streaming constructor now bypasses resident body-matrix
loading for ordinary Llama, Mistral, Qwen2 and Mixtral. A generic host payload
retains existing resident Vec loaders, while PackageLinear carries exact physical
SafeTensors descriptors and contiguous expert views. The new path validates
geometry/dtype/finite values in bounded tiles and constructs frozen streamed
linears directly. No forward, backward or optimizer shader changed. Other
families retain their qualified post-load conversion. CLI/GUI remain opt-in.

Fresh CPU/PEFT evidence: all 32 profiles pass through the application constructor,
worst metric 9.000000000813912e-8, strict <2e-7, rtol=0; frozen attempt
`target/parameter-streaming/20261030T185506020126Z`. Four profiles exercise
source-first loading. The separate memory/storage gate covers 23 cases and
7,168 logits, all resident/source-first bit-exact, plus three rejection gates.
F32/F16/BF16, sharding, wrapped backbones, tied vocabulary and packed/current/
legacy expert names pass. Accepted memory run:
`target/initial-streaming/20261030T190547671754Z/qualification.json`.

Construction device-buffer peaks: Llama h1024/i3072 208,278,940 -> 4,068,812
bytes; Mixtral h512/i1024/E8 217,701,244 -> 2,284,548 bytes. E2 -> E8 adds
1,176 bytes at fixed width. A single dense projection exceeds one tile;
measured validation request max is 8 MiB, cache <=32 MiB. This is a device-buffer
and package-request measurement, excluding driver/pipeline allocations and RSS.
Norms, biases, activations and adapter state remain resident.

Regressions: offload13; native streaming2 (nine legacy conversion and five
source-first cases); vocabulary construction2; vocabulary LoRA/resume1;
V4.1 uninterrupted trajectory1 (90.34s, 1,536 CPU logits bit-exact, LoRA and
rsLoRA x3 updates); CLI41. Rebuilt CLI resident/streamed train/save/resume/export
and cached generation agree for V4.1 LoRA/rsLoRA, Llama and Mixtral. Training
fixture geometry is six layers / width32 / E384 / top6 / HC4; corrected two
remaining false forty-layer training labels in the root README. The separate
forty-layer inference evidence remains unchanged.

Durable receipts:
`hierarchos-vulkan/validation/release_parity/parameter_streaming/source-first-cpu-peft-qualification.json`
and `source-first-qualification.json` in the same directory. Old receipts and
V4.1 parent hashes remain unchanged. Early memory harness failures (unsupported
wrapped prefix in the resident auxiliary loader, shared fixture storage, and a
wrong fixture down-projection shape) remain diagnostic evidence; only the final
23-case run is accepted. Details and commands live in
`target/initial-streaming/WORKPLAN.md` and its scripts/logs. No previously rejected
scratch deletion was retried. No production weight, all146, full-base offload or
Intel Gen9 performance claim is added.


## 2026-10-30 — bit-preserving V4.1 metadata upload scheduling

Continued from the MoE-batching/checkpoint-handle handoff. The six recorded
source hashes matched. Its executable hash did not match the rebuilt executable
present at pickup; the actual baseline was frozen and re-probed instead of
rewriting the historical receipt. Window-index, candidate-visibility and retained
attention-tape index uploads now join the layer compute session. Routing counter
placement, dispatch order and floating-point kernels are unchanged.

Three alternating release A/B pairs: resident decode submissions252->170,
latency461.359->370.847 ms/token (about20% lower); prefill709.547->583.081ms.
Streamed submits4680->4598; decode938.973->932.490ms, prefill2648.382->2653.208ms,
so no meaningful streamed timing gain is claimed. Final-row digest remains
90e096ebc5bf0659. Timings are for the scaled40-layer/E384/vocab67 fixture.

Measurement correction: the old probe divided a single-token decode median by
--decode again. At --decode8 the absolute ms/token field was understated8x.
The observer now reports the correct value; paired comparisons parse raw phase
records for both old and new binaries. Historical ratios remain valid, while
old absolute per-token numbers should not be reused. A final-row performance
digest is distinct from the full-logit CPU qualification below.

Fresh282 tests pass: Vulkan37, streaming3, offload16, V4.1 default182 (25 ignored),
joint resident1, joint fully-streamed1, uninterrupted training1, CLI41. Explicit
joint CPU checks cover6432 logits and uninterrupted LoRA/rsLoRA1536, all bit-exact.
Default-gate parsed logits201714 have worst1.4901161193847656e-7, strictly<2e-7,
rtol0. All72 archived training comparisons retain identical reported drift.
The old full-suite log has no captured values and was not counted as an empty
baseline comparison. No test tolerance, expected value or shader changed.

The rebuilt CLI lifecycle remains identical for V4.1 LoRA/rsLoRA, Llama and
Mixtral: training/save/resume/export bytes and cached generation agree across
residencies, and resume/uninterrupted adapters match. Streaming remains opt-in.
Detailed logs, frozen binaries and handoff: target/v41-upload-optimization/.
New receipt: hierarchos-vulkan/validation/release_parity/deepseek_v41/rdna3-metadata-upload.json.
Routing readbacks remain. No production-weight, all146 or Gen9 performance
qualification is added, and no previously rejected cleanup was retried.

## 2026-10-30: expert upload batching, bounded payload reuse, Gen9 source handoff

Qualified the next host-path optimizations without changing shaders, reductions,
fixture values or the strict max_abs <2e-7 / rtol0 gate. Packed data/scales and
adapter factors share linear upload batches; expert w1/w2/w3 share one startup
batch; streamed uploads join compute/scatter and retire before the next expert.
An initial arena variant FAILED the unchanged lifetime test by retaining 4 MiB.
Exact-size transient staging fixes that and the final test passes.

One 32 MiB raw payload cache is shared by checkpoint shards/clones when streaming.
Size/mtime validation still runs before hits. Adapter factors are not cached.
The 8192-range pilot churned at only 5.5 MiB; the final independent metadata cap
is32768 ranges. Final working set10669 ranges/6208646 bytes,83490 measured hits
and zero payload reads after warm-up in each run. These are host file-payload
reads, not physical SSD I/O. The unfinished pilot was stopped (compilation
briefly overlapped its last portion) and is excluded from final timings.

Three fresh alternating uncontended release pairs, scaled40-layer/E384/vocab67,
prefill8/decode8: resident load11772.948->5458.652ms, submits93484->16312.
Streamed prefill2673.513->1800.810ms, submissions15074->1869; decode936.6305->
632.9385ms/token, submissions4598->996 (~32% lower latency/1.48x rate).
Resident decode170 submissions unchanged;365.7155->355.6255ms/token is a small
change with no claimed compute improvement. Default startup456.655->464.153ms
is effectively unchanged. Final-row digest90e096ebc5bf0659 remains identical;
complete-logit gates are separate and pass below.

287 fresh tests:185 default V4.1 (25ignored),38Vulkan,3generic streaming,
16offload,41CLI,4explicit native model/PEFT gates. Package-default/joint-resident/
joint-all-streamed each3216 CPU logits and uninterrupted LoRA/rsLoRA1536 are
bit-exact (11184 values across explicit gates). Default parsed201714 logits
worst1.4901161193847656e-7. All72 archived training comparisons have identical
reported drift; historical model/PEFT fixture hashes match. CLI lifecycle for
V4.1 LoRA/rsLoRA, Llama and Mixtral retains resident/streamed/save/resume/export/
uninterrupted adapter bytes and identical generation. GUI/CLI streaming opt-in
is preserved. No production-weight, full-base-offload or Gen9 speed claim.

Receipt: hierarchos-vulkan/validation/release_parity/deepseek_v41/rdna3-expert-upload.json.
Evidence/binaries: target/v41-final-optimizations/ (before baseline matches the
previous qualified probe hash). Handoff guide: hierarchos-vulkan/validation/
intel-handoff/DEEPSEEK_V41.md. Packager includes portable source, checked-in
shaders, immutable CPU fixtures, retained joint checkpoint and measured logs,
with per-file SHA-256 manifest and archive readback verification. Gen9 gate
runner verifies frozen fixture/checkpoint hashes, records changed source files,
and executes native inference/PEFT/streaming/offload gates before target tuning.

## 2026-10-30: final pre-Gen9 resident/streamed compute optimization baseline

Fresh full-stride profiling identified MoE gate 34%, quantization 25%, HC split 18%
and packed GEMM 16% of recorded GPU brackets. The final candidate parallelizes
independent expert dots/unaries in one 64-lane row workgroup with shared 4 KiB
scratch. Lane zero retains CPU top-k/tie staging, softmax and selected-weight sum;
per-dot reductions, unary routines and rounding stay unchanged. Expert w1/w3
share the input cast and matching-block FP8/UE8 quantization; independent output
rounding and raw-input LoRA branches remain. Dense/different-block fallbacks
retain their contract. No persistent activation cache, new optional feature,
changed fixture/tolerance or expert accumulation order. Staging lifetime passes.

Three fresh alternating release pairs (all compilation/tests/profiling separate):
resident prefill 585.630->527.072ms, decode 372.1765->306.815ms/token;
streamed prefill 1782.044->1285.738ms, decode 630.438->512.5925ms/token.
Decode latency 18%/19% lower; streamed prefill 28% lower. Submissions 170/996 per
decode unchanged; load time effectively unchanged. Warm payload reads 0 in every
streamed candidate run. Frozen final-row digest90e096ebc5bf0659 unchanged.

Separate full-stride profile sums 18 model calls per mode (submissions are NOT
training steps). Resident GPU brackets 5641.356->3477.269ms; MoE gate 1923.572->
160.259ms across 720 unchanged gate dispatches. Total dispatches 122220->108648,
6786 fewer quantizations and6786 fewer BF16 casts. Remaining GPU shares roughly
quantization 30%, HC split 30%, packed linear 26%, gate 5%. Timestamp profiling perturbs
execution and is distinct from unprofiled wall-time evidence. Gen9 must profile
its own device before choosing tile/workgroup/fusion strategy.

288 tests pass: default V4.1: 186 (25 ignored), Vulkan 38, streaming 3, offload 16,
CLI 41 and four explicit model/PEFT gates. Normal package/joint resident/joint
fully streamed each 3216 CPU logits, uninterrupted LoRA/rsLoRA 1536, all bit-exact
(11184 explicit values). Default parsed 201714 logits worst 1.4901161193847656e-7,
strict <2e-7, rtol=0; all 166 previous comparisons across 106 labels retain identical
reported drift. Historical inference/PEFT fixture hashes match. New unit checks
42 typed projection-pair cases, LoRA raw inputs and block fallbacks bit-for-bit
against sequential GPU projections and checks the saved dispatch counts.
548/548 shader artifacts reproduce; push layouts 428 / typed host structs 109 pass.
CLI LoRA/rsLoRA and Llama/Mixtral resident/streamed/resume/uninterrupted adapter
exports and generation remain identical. Streaming opt-in is preserved.

Receipt: hierarchos-vulkan/validation/release_parity/deepseek_v41/rdna3-final-push.json.
Evidence: target/v41-final-push/. Before source/probe snapshots match the prior
qualified baseline. Only gate.comp/.spv, backbone.rs, mixed_linear.rs and
packed_weights.rs change runtime behavior. Shader compiled with the unchanged
manifest pin: glslang 16.6.0 -V --target-env vulkan1.1. Fresh guide and packager
select the final receipt; preserve the earlier verified ZIP and create the new
DeepSeek-V41-Gen9-Final.zip after documentation is frozen. Bundle includes source,
frozen CPU fixtures, retained checkpoint, measured profiles and qualification
logs, with source/ZIP hashes and readback verification. No production weights,
full-base-training offload or Gen9 performance/device qualification is claimed.
