# AMD regression audit of the Intel Gen9 tuning commit (`985192e`)

Scope: my laptop agent's Intel-specific kernel tuning landed on `main` and regressed
the AMD Radeon target (Ryzen Z1 Extreme, an AVX-512 host) on six PEFT fixtures.
This report records the root cause, the fix, the AMD re-verification, and the
evidence that the Intel target is unchanged.

## Verdict

- **AMD is green again**: headline forward and two-step-AdamW training pass
  strict (`2e-7`), the full 32-family × 3-stage PEFT matrix is green with a valid
  provenance receipt (`inputs_unchanged=true`), and `693 passed / 0 failed` in
  both debug and release lib suites.
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
binaries were rebuilt from the same frozen source and the whole ladder was
re-qualified on them, reproducing the recorded results exactly: `693 passed / 0
failed` in both debug and release, headline forward `pass` (21 comparisons, 0
failing values, worst `1.192e-07`), and the full two-step-AdamW run `pass` (34
rows, worst `1.192e-07`). The matrix receipt's file hashes describe the
run-time copies of those binaries, not these later rebuilds.

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

1. **Shader edit is a no-op at the AVX2 shape.** Rebuilding HEAD's
   `transformer_cross_entropy.comp` and the modified one with the same toolchain
   at 8 lanes produces **byte-identical** SPIR-V
   (`58f6e92f…`), with identical disassembly.
2. **The committed Intel binaries are untouched.** `transformer_cross_entropy.spv`
   and `falcon_h1_cross_entropy.spv` are unmodified tracked artifacts, so the
   Gen9 path executes exactly the bytes it executed before. (A local rebuild
   differs from the committed files only because the committed files were built
   by glslang 939 and the local SDK ships 940; id-normalized instruction streams
   differ solely in how `+inf` is materialized — `OpConstant 0x1p+128` vs
   `OpBitcast %uint 0x7F800000` — i.e. the same value.)
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

- The AVX-512 twin is built by glslang 940 while the committed 8-lane modules
  came from 939 (same front end, Vulkan 1.0 target). Both are valid SPIR-V 1.0
  and qualified green; rebuilding all of them with one toolchain is a housekeeping
  item, not a correctness one.
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
