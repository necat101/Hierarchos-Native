# AMD regression audit of the Intel Gen9 tuning commit (`985192e`)

Scope: my laptop agent's Intel-specific kernel tuning landed on `main` and regressed
the AMD Radeon target (Ryzen Z1 Extreme, an AVX-512 host) on six PEFT fixtures.
This report records the root cause, the fix, the AMD re-verification, and the
evidence that the Intel target is unchanged.

## Verdict

- **AMD is green again**: headline forward and two-step-AdamW training pass
  strict (`2e-7`), the full 32-family × 3-stage PEFT matrix is green with a
  provenance receipt that still matches the pushed tree input-for-input
  (`inputs_unchanged=true`, 3881 hashes re-derived), and `693 passed / 0 failed`
  in both debug and release lib suites.
- **Intel is provably unchanged**: the dispatched modules and parameters on an
  AVX2-only host are the Gen9-tuned ones, and the shader edit emits
  *byte-identical* SPIR-V at the AVX2 shape.
- The reverted/bisected commit did **not** contain a logic bug. It encoded an
  assumption about the *host CPU*, not about the GPU.

## 1. What the Intel commit changed

`985192e` ("Gen9 tuning") attributed two reductions to the oracle box:

| Reduction | Before | After `985192e` |
| --- | --- | --- |
| Narrow-output GEMM (`output_dim == 8`) | `linear_forward_lane4` (four interleaved lanes, materialized products) | `linear_forward_lane2` (two fused FMA lanes) |
| Materialized log-softmax CE fold | 16 lanes | 8 lanes |

`lane2` and `lane4` are both *emulations of PyTorch's ATen micro-kernels*; the
commit picked the AVX2 emulation because the reference box is AVX2-only.

## 2. Root cause

ATen dispatches its vectorized CPU kernels by **CPU ISA at run time**, and these
two reductions change *association* with that dispatch:

- AVX2-only host → 8-wide vectors: two fused FMA lanes, 8-lane sequential fold.
- AVX-512 host → 16-wide vectors: four interleaved lanes with materialized
  products, 16-lane fold.

So "which association is correct" is a property of the *host*, not the GPU.
Baking in the AVX2 shape made the same binary bit-exact on the Intel laptop and
one ulp off on this AMD/AVX-512 box — and the saved-module gradient path
amplifies that ulp through every lower norm.

Measured damage on this machine with the AVX2 shape applied everywhere:

| Fixture | `model.embed_tokens` adjoint, AVX2 shape | AVX-512 shape |
| --- | --- | --- |
| gemma4 | `3.219e-6` | `7.451e-9` |
| qwen2_5_sliding_tied | `2.980e-7` | `1.788e-7` |

Six previously green fixtures crossed the `2e-7` gate: `gemma3`, `gemma4`,
`minimax_m2`, `minimax_m3`, `smollm3`, `qwen2_5_sliding_tied`.

Bisection confirmed the source of the regression: reverting `transformer.rs` and
all modified `.spv` files to the pre-tuning revision `06497e1` restored gemma4
exactly (`7.451e-09`), and the lane selectors were the only topology-relevant
change in the commit.

## 3. The fix

Host-ISA-aware selection instead of a hard-coded shape, in
[hierarchos-vulkan/src/transformer.rs](hierarchos-vulkan/src/transformer.rs):

- `aten_vector_width()` — cached (`OnceLock`) capability probe
  (`is_x86_feature_detected!("avx512f")`); 16 on AVX-512 hosts, 8 elsewhere.
- `aten_vector_width_for(requested, host_has_avx512)` — pure policy function so
  the rule is testable without a device.
- `narrow_gemm_plan(width)` — `Lane2` for width 8, `Lane4` for width 16.
- `HIERARCHOS_ATEN_VECTOR_WIDTH=8|16` pins the shape for qualification runs.
- [hierarchos-vulkan/shaders/transformer_cross_entropy.comp](hierarchos-vulkan/shaders/transformer_cross_entropy.comp)
  gained `HIERARCHOS_LOG_SOFTMAX_LANES` (default 8) and a second build
  ([falcon_h1_cross_entropy_lanes16.spv](hierarchos-vulkan/shaders/falcon_h1_cross_entropy_lanes16.spv)).

Cost: one extra compute pipeline created at init on every host (the probe itself
is free after the first call, and both modules were already constructed before
this change). No per-token cost on either vendor.

Behaviour is documented in
[COMPATIBILITY.md](hierarchos-vulkan/COMPATIBILITY.md) ("Host ATen vector shape")
and [VENDOR_TUNING.md](hierarchos-vulkan/VENDOR_TUNING.md).

## 4. AMD re-verification (post-fix)

| Ladder rung | Result |
| --- | --- |
| Rust lib suites, debug | `693 passed; 0 failed; 9 ignored` |
| Rust lib suites, release | `693 passed; 0 failed; 9 ignored` |
| `verify_hf_logits.py --headline-strict` | pass — 21 comparisons, **0 failing values**, worst `1.192e-07` (smollm3 `padded`) vs `2e-7` ceiling |
| `verify_hf_training.py` (full, two-step AdamW) | pass — 34 rows, worst `1.192e-07` (`transformer.h.1.ln_2.weight`); the `--headline-strict` subset is also `pass` (11 rows, worst `1.192e-07`) |
| Vendor parity (`FORCE_VENDOR=amd` vs tuned) | 21/21 comparisons identical on both paths, worst `1.192e-07` |
| Full PEFT matrix (32 families × LoRA/saved/switching) | **32 families, 0 failures**, provenance `inputs_unchanged=true` (3881 hashed inputs), 32/32 on *every* stage |

Because the corrupted artifacts above had to be cleared, the qualification
binaries were rebuilt from the same frozen source (`cargo build -j 4`) and the
whole ladder was re-qualified on them, reproducing the recorded results
exactly: `693 passed / 0 failed` in both debug and release, headline forward
`pass` (21 comparisons, 0 failing values, worst `1.192e-07`), and the full
two-step-AdamW run `pass` (34 rows, worst `1.192e-07`).

The PEFT matrix was then re-run end to end on those rebuilt binaries, so the
receipt describes the tree as pushed rather than the pre-repair copies: **32
families, 0 failures**, 32/32 on each of LoRA, switching and saved, provenance
`inputs_unchanged=true`, worst `two_step_adamw_max_abs` `2.28e-8`
(`phi4_multimodal_text`). Re-deriving the run's 3881-input fingerprint from the
pushed tree reproduces it exactly — zero changed, zero missing, zero added —
across every `.spv` (the new 16-lane module included), every `.comp`/`.glsl`,
every `.rs`, `Cargo.toml`/`Cargo.lock`, `COMPATIBILITY.md`, the three debug
qualification binaries and the local Transformers oracle checkout.

The six previously failing fixtures now pass with every metric at or below
`5e-8`:

| Fixture | `peft_max_abs` | `gradient_max_abs` | `two_step_adamw_max_abs` |
| --- | --- | --- | --- |
| gemma4 | `4.92e-8` | `2.61e-8` | `2.24e-8` |
| gemma3 | `4.65e-8` | `1.49e-8` | `1.86e-8` |
| minimax_m2 | `3.14e-8` | `3.73e-9` | `1.96e-8` |
| minimax_m3 | `2.34e-8` | `2.45e-9` | `1.77e-8` |
| smollm3 | `2.27e-8` | `1.86e-9` | `1.73e-8` |
| qwen2_5_sliding_tied | `3.45e-8` | `3.73e-9` | `1.91e-8` |

## 5. Why the Intel target is unchanged

Four independent layers, strongest first:

1. **Shader edit is a no-op at the AVX2 shape.** Rebuilding the pre-fix
   revision `985192e`'s `transformer_cross_entropy.comp` and the committed one
   with the *same* local toolchain at 8 lanes (the `-DHIERARCHOS_LOG_SOFTMAX_GRAD=1`
   build) produces **byte-identical** SPIR-V — same md5
   (`998b807b…`), identical disassembly. The new lane parameter is inert
   whenever the lane count is left at its default.
2. **The committed Intel binaries are untouched and reproducible.** The commit
   adds only the new `falcon_h1_cross_entropy_lanes16.spv`; the two 8-lane
   modules are unmodified tracked blobs, so the Gen9 path executes exactly the
   bytes it executed before. Both are also reproducible from the committed
   source with the recipe in `README.md`: `transformer_cross_entropy.spv`
   rebuilds byte-for-byte (md5 `b6bda655…`), and the new 16-lane module does
   too (md5 `ab6b9d45…`, identical disassembly). `falcon_h1_cross_entropy.spv`
   alone differs from a local rebuild, and only because it was built by glslang
   939 while the local SDK ships 940: with ids normalized, the two instruction
   streams differ on 7 lines, all of them the same `+inf` materialized two ways
   (hoisted `OpConstant %float 0x1p+128` vs an inline
   `OpBitcast %float %uint 2139095040`). Same value, same control flow, and an
   artifact of the pre-existing tracked file rather than of this change.
3. **Dispatch is a pure function of host ISA.** On an AVX2-only host the probe
   resolves to 8, selecting `linear_forward_lane2` and the 8-lane CE module —
   exactly the Gen9-tuned pair — asserted by
   `aten_vector_width_follows_host_avx512_support_not_the_gpu_vendor`,
   `narrow_gemm_plan_follows_host_vector_width` and
   `narrow_gemm_and_log_softmax_kernels_track_the_host_vector_width`.
4. **Empirical A/B, re-run in this session.** Forcing
   `HIERARCHOS_ATEN_VECTOR_WIDTH=8` on this AMD box reproduces the AVX2 numbers
   to the last bit on gemma4 — `3.2187e-6`, `8.3447e-7`, `2.0862e-7`,
   `1.6391e-7` (`saved`-stage gradient max-abs), and the `saved` stage *fails*
   (`rc=1`) exactly as `985192e` did, while the default AVX-512 shape passes.
   The knob is live and the shape is load-bearing. No Intel-specific tuning was
   removed: `src/vendor.rs` and every Gen9 module are untouched by this change.

## 6. Lead: the Gen9 laptop's seven red saved rows

[VENDOR_TUNING.md](hierarchos-vulkan/VENDOR_TUNING.md) records the Gen9 laptop at
**25/32 saved** (32/32 LoRA and switching) with seven red rows: `gemma3`,
`gemma4`, `minimax_m2`, `minimax_m3`, `mistral4`, `qwen3_5_full`, `smollm3`.
That run used the same forced-AVX2 kernels this audit pins down, and its red set
overlaps five of the six rows that were red on this AMD box under the same
kernels. The laptop is a Skylake-class part (HD Graphics 520, `0x1916`), so it has
no AVX-512 and its probe correctly resolves to the 8-lane shape — my change
leaves it bit-for-bit as it was.

But the overlap is worth one cheap experiment the next time that laptop is
reachable, because it decides whether those seven rows are an oracle-boundary
artifact (the current explanation) or the same shape mismatch fixed here:

```powershell
HIERARCHOS_ATEN_VECTOR_WIDTH=16 python validation/verify_peft_saved_modules.py gemma4
```

If those rows go green under the 16-lane modules, then the laptop's *oracle* is
not AVX2-shaped (a wheel built for a different dispatch, or reductions outside
the lane2/lane4 pair), and the probe's answer there should be pinned to 16 for
qualification rather than inferred from the CPU. If they stay red, the
documented oracle-boundary explanation stands unchanged. Either outcome is a
one-line answer, and the env override exists precisely so it needs no rebuild.

## 7. Residual notes

- The AVX-512 twin is built by glslang 940 while the committed
  `falcon_h1_cross_entropy.spv` came from 939 (same front end, Vulkan 1.0
  target). Measured: the only normalized difference is the `+inf`
  materialization above. Both are valid SPIR-V 1.0 and qualified green;
  rebuilding the 8-lane grad module with the local 940 to match is a
  housekeeping item, not a correctness one — it would touch a tracked Gen9
  binary, so it is deliberately left out of this change.
- The forward headline moved slightly *between shapes* on this box (worst row
  `8.94e-8` under the AVX2 shape → `1.192e-7` under AVX-512, both far inside
  `2e-7`), while gemma4's unmasked row *improved* (`8.94e-8` → `4.66e-8`) and the
  saved-module surface — where a mismatched shape is amplified — is decisively
  better. Per-family calibration of the *narrow* shape is the remaining
  refinement; the uniform host-ISA rule is what current evidence supports.
- **Build hygiene on this machine (pre-existing, unrelated to the fix).** A
  full `cargo build` (16-way parallel) fails here with cascading `E0786 invalid
  metadata` / `E0463 can't find crate` errors. The signature is `rustc.exe ...
  STATUS_STACK_BUFFER_OVERRUN`: the compiler dies on several heavy
  `rwkv_*` / training-graph bins under full parallelism, and each death leaves
  partial artifacts behind, which is what the downstream "invalid metadata"
  errors actually report. Any single one of those bins builds clean alone, and
  `cargo build -j 4` completes — so it is host resource pressure (stack/memory),
  not a source error, and not caused by this change. It also explains the
  confusing state this session inherited.

## 8. Reproduce

```bash
cd hierarchos-vulkan
cargo build -j 4                                    # full parallel builds crash rustc here (see §7)
cargo test --lib                                    # 693 passed, 0 failed
../.venv-vulkan/Scripts/python.exe validation/verify_hf_logits.py --headline-strict
../.venv-vulkan/Scripts/python.exe validation/verify_hf_training.py
../.venv-vulkan/Scripts/python.exe validation/verify_peft_green_matrix.py --stage all --jobs 2

# pin the other host's ATen shape to A/B on one machine
HIERARCHOS_ATEN_VECTOR_WIDTH=8  ../.venv-vulkan/Scripts/python.exe validation/verify_peft_lora_strict.py gemma4
HIERARCHOS_ATEN_VECTOR_WIDTH=16 ../.venv-vulkan/Scripts/python.exe validation/verify_peft_lora_strict.py gemma4
```

## 9. Handoff: the measured decisions this box has no entries for

The ladder's kernel-topology and projection decisions are **per device**, and every
recorded entry so far is for the Intel Gen9 laptop. That is deliberate -- a decision is
evidence about one adapter -- but it means a green `cargo test` on this box does not
tell you which graph the *shipped default* runs here. It runs the structural/gate
default everywhere until this box measures its own entry. Seven decision records and two
policy/chain receipts are involved:

| record | consumer | what it decides | re-record on this box with |
| --- | --- | --- | --- |
| `validation/topology_decisions.json` | `src/rwkv_low_rank.rs` (LN1/low-rank autotuner) | `one-row` vs `two-rows` vs `column-slices` per (device, width, ranks, batch pairs, tail) | `verify_topology_decisions.py --record` |
| `validation/time_mix_projection_decisions.json` | `src/decision_records.rs` | fused time-mix projection vs its `linear3-forward` composition per (device, width, batch) | `bench_rwkv_packed_cell_time_mix.py --record-decisions` |
| `validation/channel_mix_forward_decisions.json` | `src/decision_records.rs` | fused channel-mix producer vs the staged composition per (device, width, batch) | `bench_rwkv_packed_cell_channel_mix.py --record-decisions` |
| `validation/low_rank_full_forward_decisions.json` | `src/decision_records.rs` | fused LN1/low-rank producer vs the staged chain it replaces per (device, width, batch, and optionally parameter-storage policy: the `fp16-storage` entries are the deep FP16 route's own verdicts) | `bench_rwkv_packed_cell_low_rank_forward.py --record-decisions` (dense) / `bench_rwkv_low_rank_fp16_forward.py --record-decisions` (FP16 policy) |
| `validation/key_state_forward_decisions.json` | `src/rwkv_time_mix_core.rs` | one route per (device, width, batch, storage policy) from `wide-fused`, `narrow-fused` and `staged`, with every candidate's median in `candidates` | `bench_rwkv_key_state_forward.py --record-decisions` |
| `validation/tbptt_packed_forward_decisions.json` | `src/rwkv_tbptt.rs` | the TBPTT first-forward transition, `packed` vs `legacy`, per (device, width, batch) and optionally parameter-storage policy (the `fp16-storage` entries are that policy's own A/B) | `bench_rwkv_tbptt_packed_chains.py --chain forward --record-decisions` (FP32) / the same with `--parameter-storage fp16-storage` |
| `validation/tbptt_packed_rematerialization_decisions.json` | `src/rwkv_tbptt.rs` | TBPTT reverse rematerialization, `packed` vs `legacy`, per (device, width, batch) and optionally parameter-storage policy | `bench_rwkv_tbptt_packed_chains.py --chain remat --record-decisions` (FP32) / the same with `--parameter-storage fp16-storage` |
| `validation/low_rank_fp16_forward_bench.json` (receipt, not a decision table) | operator evidence | deep FP16 producer vs the staged chain under FP16 parameter storage, and the policy-scoped verdicts it records for that route | `bench_rwkv_low_rank_fp16_forward.py --record` |

Both TBPTT chain tables are read under the active parameter-storage policy. The
`fp16-storage` policy has its own receipts (`tbptt_packed_forward_fp16_bench.json`,
`tbptt_packed_remat_fp16_bench.json`), recorded with the cell's packed execution
mirrors installed: the forward A/B stays bit-identical there and repeats the FP32
crossover, so that policy adopts its own measured `legacy` verdicts at batches 16
and 32 and carries no `withheld` entry. Rematerialization's measured verdicts are
all the structural default under both policies.

```bash
cd hierarchos-vulkan
cargo build --release --bin training_step_probe

# replay the topology decisions this tree ships; --record adds THIS box's entries and
# leaves the Intel ones untouched. Pass --probe explicitly: the verifier prefers a
# debug binary if one exists.
../.venv-vulkan/Scripts/python.exe validation/verify_topology_decisions.py \
    --probe target/release/training_step_probe.exe --record
../.venv-vulkan/Scripts/python.exe validation/verify_topology_decisions.py \
    --probe target/release/training_step_probe.exe

# the two projection A/Bs: each rebuilds its own release binary, measures every
# geometry, checks both graphs are bit-identical, and merges this adapter's winners
python validation/bench_rwkv_packed_cell_time_mix.py --record-decisions validation/time_mix_projection_decisions.json
python validation/bench_rwkv_packed_cell_channel_mix.py --record-decisions validation/channel_mix_forward_decisions.json
# the LN1/low-rank route: one producer dispatch against the staged chain.
# NOTE: this bench measures the packed cell, whose low-rank parameters are FP32, so
# the fused arm is the portable producer here too. Do not set
# HIERARCHOS_VULKAN_TRAINING_PRECISION expecting this command to change graph -- the
# packed cell does not read that policy. The deep FP16 arm has its own driver below.
python validation/bench_rwkv_packed_cell_low_rank_forward.py \
    --record-decisions validation/low_rank_full_forward_decisions.json

# the deep FP16 arm under the training graph's parameter-storage policy: the op
# quantizes its own FP32 masters into packed mirrors, so this is the arm the
# `fp16-storage-fp32-compute` policy installs. It reads the record above (its own
# `fp16-packed-mirror` entries first, then the policy-less ones) and records under
# that policy only, so the dense route's verdicts are untouched.
python validation/bench_rwkv_low_rank_fp16_forward.py \
    --record validation/low_rank_fp16_forward_bench.json \
    --record-decisions validation/low_rank_full_forward_decisions.json

# the key/state forward route: one fused dispatch against two, with both fused arms
# (wider-group and one-workgroup-per-head) timed in the same rotation. Measured in
# isolation (repeated dispatches per timed sample), and the record takes the arm and
# the route this box measured per geometry.
python validation/bench_rwkv_key_state_forward.py \
    --record validation/key_state_forward_bench.json \
    --record-decisions validation/key_state_forward_decisions.json

# the two packed TBPTT chains, once per parameter-storage policy: the fp16-storage
# pass installs the cell's packed FP16 execution mirrors on all three legs, which is
# the op-level analogue of the training graph's fp16-storage-fp32-compute policy.
python validation/bench_rwkv_tbptt_packed_chains.py --chain forward \
    --record validation/tbptt_packed_forward_bench.json \
    --record-decisions validation/tbptt_packed_forward_decisions.json
python validation/bench_rwkv_tbptt_packed_chains.py --chain remat \
    --record validation/tbptt_packed_remat_bench.json \
    --record-decisions validation/tbptt_packed_rematerialization_decisions.json
python validation/bench_rwkv_tbptt_packed_chains.py --chain forward \
    --parameter-storage fp16-storage \
    --record validation/tbptt_packed_forward_fp16_bench.json \
    --record-decisions validation/tbptt_packed_forward_decisions.json
python validation/bench_rwkv_tbptt_packed_chains.py --chain remat \
    --parameter-storage fp16-storage \
    --record validation/tbptt_packed_remat_fp16_bench.json \
    --record-decisions validation/tbptt_packed_rematerialization_decisions.json

# then the contracts on what was recorded (HIERARCHOS_RUN_GPU_VALIDATION=1 also re-runs
# each A/B live, which is how the Intel receipts were taken)
HIERARCHOS_RUN_GPU_VALIDATION=1 ../.venv-vulkan/Scripts/python.exe -m unittest discover \
    -s validation -p "test_rwkv_*receipt.py" -t validation
```

Two properties of those runs are worth knowing before reading their output:

* **Run them one at a time.** A second process on the adapter is now detected rather
  than absorbed: each autotune round times each candidate three times and keeps the
  fastest, and if the structural default's own per-round samples spread by more than
  1.15x, the sweep keeps the default and records `"contended": true`. A contended
  replay is reported as a note, never as a moved decision, and `--record` refuses to
  write one -- so a busy box says "this run did not measure" instead of turning another
  process's timing into this device's decision.
* **A shortest-path check exists.** `cargo test --release --lib -- a_contended_sweep`
  and `validation/test_topology_decisions.py` pin both directions of that rule without a
  GPU, so the contention logic can be checked on a machine that is mid-qualification.

Nothing above changes the arithmetic: every decision selects between complete Vulkan
shader graphs whose outputs are compared bit-for-bit before a record entry is written,
and the host only chooses which graph to dispatch.

## 10. Cross-validation on AMD: status and the exact commands

The four A/Bs, the phase profile and the records themselves are adapter-portable;
the *entries* are not, which is the design. Every checked-in entry in this tree is
the Intel Gen9 part's, so the AMD box runs the structural defaults until it measures
its own rows. A cross-adapter run is therefore two things, and only the first is a
CI job:

1. **Evidence** -- `.github/workflows/time-mix-ab.yml` job `cross-adapter-ab` runs
   the key/state A/B, the phase profile, the projection, channel-mix and low-rank
   A/Bs on a runner labelled `[self-hosted, windows, vulkan, amd]`, then runs the
   receipt contracts against *that adapter's* measurements and uploads the receipts.
   It records no decisions. **This job has not run in this session**: the only
   adapter attached to this machine is an Intel HD Graphics 520, and no AMD device
   was reachable, so no AMD number appears anywhere in this tree and the AMD rows
   remain absent rather than guessed.
2. **Entries** -- on a quiet AMD machine, for each record:

```bash
cd hierarchos-vulkan
# one forward, three candidates: wide-fused, narrow-fused, staged
python validation/bench_rwkv_key_state_forward.py \
    --batch 1 --batch 4 --batch 8 --batch 16 --batch 32 \
    --record validation/key_state_forward_bench.json \
    --record-decisions validation/key_state_forward_decisions.json
# the phase attribution behind whichever candidate wins above
python validation/profile_rwkv_key_state_forward.py \
    --batch 1 --batch 8 --batch 16 --batch 32 \
    --record validation/key_state_forward_profile.json
# the four producer routes, each keyed by (adapter, width, batch, parameter storage)
python validation/bench_rwkv_packed_cell_time_mix.py \
    --record-decisions validation/time_mix_projection_decisions.json
python validation/bench_rwkv_packed_cell_channel_mix.py \
    --record-decisions validation/channel_mix_forward_decisions.json
python validation/bench_rwkv_packed_cell_low_rank_forward.py \
    --record-decisions validation/low_rank_full_forward_decisions.json
python validation/bench_rwkv_low_rank_fp16_forward.py \
    --decision-margin 0.01 \
    --record validation/low_rank_fp16_forward_bench.json \
    --record-decisions validation/low_rank_full_forward_decisions.json
```

Each `--record-decisions` merges only `(device, width, batch, policy)` rows, so the
Intel entries stay exactly as measured. Rebuild after recording: the records are
`include_str!`-ed, and a route that is not its own fastest candidate fails at load,
so a bad merge is a build error rather than a silently slower default.

What to expect on a different adapter, stated as a check rather than a prediction:
the key/state record's vocabulary is the same three routes, but which one wins at a
given batch is unknown until measured -- the wider-group kernel's advantage came
from 64-lane phases, and its batch-32 collapse on this part is tied to that part's
EU count and workgroup placement. The AMD run is how that gets answered for AMD; it
is not something this tree should claim in advance.
