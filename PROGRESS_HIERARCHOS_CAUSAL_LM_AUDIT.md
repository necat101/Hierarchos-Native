# Hierarchos causal-LM structural audit — 2026-10-01

Scope: Hierarchos coherent-v9 only, starting at `06497e1929e70fd418d926f8ae65382924ad7bb4`.
Transformer implementations, transformer shaders, inference implementations, and
transformer qualification flags are unchanged. The pre-existing untracked
`.peft-final-release-capabilities-20260930.json` is preserved.

## Architecture and objective

The Vulkan implementation already implements a causal language model. Its
recurrent architecture does not need to become a transformer to use the same
next-token training discipline:

1. The tied `lm_head.weight` supplies token embeddings and output vocabulary
   projection. DeepEmbed, bounded ROSA, persistent features, and LTM retrieval
   feed the token frontend.
2. The manager H recurrence consumes the current token representation and prior
   worker feedback. Hard ACT selects a recurrent state at manager positions;
   projected manager contexts are carried between positions.
3. The worker L recurrence refines the current output using those contexts and
   state-derived drift. Its projection, output normalization, and tied head
   predict the next token.
4. The graph uses `labels[t + 1]` exactly once. Prompt labels can be `-100`
   without removing their input tokens or cutting their causal influence.
   Optional lookahead labels supervise the final input of a TBPTT segment.
5. Weighted CE and z-loss use the shifted supervision mass. With logits `z`,
   the language term is the weighted mean of
   `-log softmax(z)[target] + z_loss_weight * logsumexp(z)^2`.
   Gradient accumulation normalizes the complete window once, rather than
   averaging independently normalized microbatch means.
6. Ponder and commitment remain Hierarchos objectives, with their own real-row
   and manager-position denominators. Historical TBPTT cuts backward state and
   context edges while retaining forward carriers. Optional backward caps keep
   the existing coherent-v9 derivative policy.
7. LTM alignment is a separate sampled/controller objective. This audit does not
   replace its schedule or enable unsupported fast-memory writes.

The transformer stack was a read-only reference for causal supervision, masked
targets, tied parameter ownership, gradient accumulation, and validation of the
actual native learned function. No transformer implementation was transplanted.

## Repairs

- **ROSA padding contamination:** reproduced a padded input `[1,2,1,9,8]` with
  attention mask `[1,1,1,0,0]` saving ROSA history `[1,2,1,9,8]`. H/L/context
  already froze correctly, but the suffix memory did not. The tape now passes
  its active-row mask into ROSA in dense and planned/sparse forward sweeps.
  Masked lanes leave history and match state untouched; resetting an inactive
  lane still clears its logical history. Reverse replay uses cached predictions
  and does not advance history. The shader control flags are private; public
  reset masks remain binary and checkpoint layouts are unchanged.
- **Pre-padded trainer rows:** `pack_batch` now canonicalizes masked labels to
  `-100` and their loss weights to zero, including defaulted labels/weights.
  It rejects mismatched shapes, fractional masks, left/interior padding, and
  invalid weights before graph execution. Completion weights and prompt masks
  on real tokens are preserved.
- **One objective preparation path:** dense execution now calls the same
  preparation routine as budgeted execution, removing 273 duplicated lines of
  label, mask, target, and auxiliary-adjoint construction.

## Reproducible validation

```powershell
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib hierarchos_causal_lm -j 1 -- --nocapture --test-threads=1
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --bin hierarchos-vulkan-train -j 1 -- --test-threads=1
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib token_frontend::tests -j 1 -- --test-threads=1
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib pytorch_tbptt_weighting_tests -j 1 -- --test-threads=1
cargo test --manifest-path hierarchos-inference/Cargo.toml --lib -j 1
python hierarchos-vulkan/validation/verify_hierarchos_rosa_shaders.py --output hierarchos-vulkan/validation/hierarchos_rosa_shader_report.json
```

The structural fixture is initialized entirely in Rust: vocabulary 16, hidden
width 32, head size 32, H stride 2, up to 3 H steps and 2 L steps, and bounded
ROSA context 8. It exercises 95 registered parameter tensors on the Vulkan
device reported as **AMD Radeon Graphics**. LTM alignment is disabled in this
fixture to isolate the causal and explicitly selected auxiliary objectives.

The maintained checks cover:

- Independent Rust CPU next-token CE/z-loss and final H/L state versus Vulkan
  (absolute tolerance `2e-5`; this is not a Transformer/PEFT parity certificate).
- Finite gradients, nonzero CE gradients into the embedding/head, frontend,
  H/L recurrence and output/context projections; one tied head registration.
- Future-suffix isolation, ignored targets, exact H/L/context padding identity,
  and unchanged parameter gradients (`2e-7`).
- Unequal weighted microbatch versus joint-batch AdamW updates (`2e-6`).
- Empty shifted supervision rejected without changing parameters.
- Dense/planned execution and sparse replay strides 1/2, both whole-tape and
  TBPTT chunk size 2, with CE/z-loss/ponder/commitment enabled (`2e-7`).
- All eight ROSA lane variants against the independent Rust suffix automaton,
  including mixed active lanes, pause/resume, masked resets, and context rollover.
- A 12-update causal-learning trajectory and SafeTensors export into Rust inference.

Shader provenance is recorded in
`hierarchos-vulkan/validation/hierarchos_rosa_shader_report.json`. The verifier
recompiles all eight variants into temporary storage and compares hashes; it
only changes embedded artifacts when explicitly invoked with `--write`.

## Qualification boundary

These checks establish the tested FP32 coherent-v9 structural contract. They do
not establish production-scale convergence, multi-GPU or mixed-precision parity,
adaptive ponder behavior, or a new full Python coherent-v9 parity certificate.
The independent forward oracle here is the existing Rust coherent-v9 inference
implementation. Existing per-token `losses` readbacks remain diagnostic row
means; they should not be interpreted as a globally weighted perplexity metric.

## Final run results

All commands above exited 0 against the final implementation:

| Check | Result |
| --- | --- |
| Native causal-LM structural suite | 3 passed; all 8 ROSA variants exercised |
| Hierarchos trainer unit suite | 40 passed |
| Token frontend regressions | 4 passed; 1 existing occupancy benchmark intentionally ignored |
| Objective/TBPTT weighting and cap regressions | 10 passed |
| Pure-Rust inference unit suite | 12 passed |
| Shader source/artifact reproduction | 8/8 exact hashes |
| `spirv-val --target-env vulkan1.1` on ROSA lane modules | 8/8 valid |
| `git diff --check` | Passed |
| Diff of transformer implementations/shaders and other frontend crates | Empty |

The 12-update native training fixture reduced mean shifted CE from **2.5682123**
to **0.9756531**, with the final value measured by the independent Rust inference
runtime after loading exported weights. This is a deterministic learning smoke
test, not a language-quality benchmark.

Local execution logs (ignored build evidence): `.hierarchos-causal-lm-test.log`,
`.hierarchos-trainer-tests.log`, `.hierarchos-frontend-tests.log`,
`.hierarchos-objective-tests.log`, `.hierarchos-inference-tests.log`, and
`.hierarchos-rosa-shader-verify.log`. The pre-fix padding failure is retained in
`.hierarchos-rosa-padding-repro.log`.
