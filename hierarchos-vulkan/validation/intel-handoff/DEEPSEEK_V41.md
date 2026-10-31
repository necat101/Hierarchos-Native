# DeepSeek V4.1 Flash: Intel Gen9 kernel port baseline

**Current RDNA3 validation is limited to DeepSeek V4.1 Flash and the recorded
parameter/MoE streaming checks.** Gen9 validation of this handoff is pending;
earlier Intel/Ally reports are historical and do not certify the current tree.
See [validation scope and Git pull commands](../../../VALIDATION_STATUS.md).

Native causal text inference is qualified on the Windows AMD Ryzen Z1 Extreme /
Radeon Graphics machine under the accepted scalable-checkpoint criterion. The
next stage is kernel optimization and target-device validation on Intel Gen9.
The Gen9 target is not yet covered by this machine's device qualification.

The retained checkpoint is `validation/fixtures/deepseek_v41_joint_4141111/`
(44,770,196 bytes including Engram metadata). It uses all 40 released backbone
layers, 384 routed experts/top-6, HC 4, released compression ratios and KV/index
ownership, Engram placement/four-gram features, and candidate block size 8.
Widths, table/vocabulary sizes, context and top-k capacities are reduced. The
recorded production-width probes separately cover the released arithmetic widths.
The normal `DeepseekV41CausalLM::from_hf_package` loader is exercised without
execution-policy flags. Six masked prefill/decode/reset steps match all 3,216 CPU
logits with zero drift and exact routing. Experts above id 255 are exercised.

Run from the repository root on the selected target device:

```powershell
New-Item -ItemType Directory -Force target/deepseek-v41-reference | Out-Null
cargo test --release --manifest-path hierarchos-vulkan/Cargo.toml --lib -j 1 deepseek_v41_default_package_scalable_release_strict_all_logits -- --ignored --nocapture --test-threads=1
```

For the complete small-checkpoint regression pass, run:

```powershell
powershell -ExecutionPolicy Bypass -File hierarchos-vulkan/validation/run_deepseek_v41_gen9_gate.ps1
```

This uses one compiler job and debug tests to limit build memory. It runs the
default V4.1 tests, Vulkan infrastructure, generic parameter streaming, offload,
the normal package loader, resident and fully streamed 40-layer joint logits,
and the native PEFT uninterrupted/save/resume trajectory. It refuses empty test
filters and stops on failure. Logs and a target receipt go under
`target/gen9-v41-gate/`; choose a fresh `-OutputDirectory` when repeating it.
Inspect the recorded device names: the tests use the default Vulkan adapter.
The PEFT trajectory fixture is six layers, width 32, 384 experts/top-6 and HC 4;
the separate inference fixture has all 40 layers. Neither is a full-weight run.

Keep the immutable CPU expectations. The acceptance ceiling is strictly
`max_abs < 2e-7`, `rtol=0`, for every finite, nonempty, shape-matched logit;
matching generated ids alone does not qualify a kernel. CUDA is not required.
The existing machine receipt and complete compressed native outputs are in
`validation/release_parity/deepseek_v41/`. Verify that receipt with
`validation/qualify_deepseek_v41_scalable.py`; runtime changes invalidate it.
Record a separate target-device receipt after the optimized kernels pass.

Profile before choosing the optimization order. Candidate work includes fused
packed FP4/FP8 decode and block-scaled GEMM, grouped BF16 output projection,
sparse score/value contractions, and dispatch/submission overhead. The first
host-machine profile and the zero-fill submission fix it produced are recorded in
[`../DEEPSEEK_V41_GEN9_PROFILE.md`](../DEEPSEEK_V41_GEN9_PROFILE.md); re-run the
harness here before changing that order, since the ranking is per device. Use the
existing Gen9 PCI/vendor detection and watchdog-safe submission infrastructure;
query device capabilities rather than assuming subgroup width or optional types.

The host path now batches expert weights/scales/adapter factors, joins streamed
expert uploads with that expert's compute, and retires each expert before staging
the next. Exact-size upload staging is freed at retirement. Streaming also uses
a checkpoint-wide 32 MiB raw payload cache with at most 32,768 ranges; it still
checks shard size and modification time before a cache hit. Adapter factors are
not cached. These paths preserve storage bytes and shader arithmetic. Their
latest qualified baseline is
`../release_parity/deepseek_v41/rdna3-final-push.json`. All 288 fresh tests
pass: worst parsed logit drift `1.4901161193847656e-7`; 11,184 explicit normal
package/joint resident/joint streamed/PEFT trajectory logits are bit-exact.
All 166 previous default-gate logit comparisons (106 labels) retain identical
reported drift. Frozen CPU expectations and the strict ceiling are unchanged.

The final pass parallelizes independent expert dots/unaries in a 64-lane row
workgroup with 4 KiB shared scratch. Lane zero keeps the CPU top-k/tie staging,
softmax and selected-weight reductions in their original order. Expert gate/up
projections share the input cast and, when their activation blocks match, the
FP8/UE8 quantization. Different blocks/dense storage retain their paths; output
rounding remains independent and LoRA consumes the original module input.
There is no persistent activation cache or new optional device capability.

Three fresh alternating release pairs against the preceding qualified binary:

| RDNA3 scaled measurement | Before | Final |
| --- | ---: | ---: |
| Resident prefill8, ms | 585.630 | 527.072 |
| Resident decode, ms/token | 372.1765 | 306.8150 |
| Streamed prefill8, ms | 1782.044 | 1285.738 |
| Streamed decode, ms/token | 630.4380 | 512.5925 |

Decode latency is about 18% lower in resident mode and 19% lower streamed;
streamed prefill is 28% lower. Submission counts remain 170/996 per decode;
load time is effectively unchanged. Warm payload reads remain zero. These are
RDNA3 scaled-checkpoint measurements, not Gen9 or full-weight estimates. The
earlier upload/cache optimizations and their loading gains remain recorded in
`rdna3-expert-upload.json`; do not combine absolute timings from different runs.

The final full-stride profile sums 18 model calls in each mode. Resident GPU
dispatch brackets total 5641.356 -> 3477.269 ms; MoE gate 1923.572 -> 160.259 ms
(same 720 gate dispatches). Quantization and BF16 casts each lose 6,786
dispatches; total dispatches 122,220 -> 108,648. The remaining GPU ranking is
quantization 30%, HC split 30%, packed GEMM 26%, gate 5%. Re-profile on Gen9;
timestamps perturb execution, so unprofiled A/B supplies latency evidence.
`target/v41-final-push/profile-aggregate.json` and the raw logs are bundled.

All 548 shader artifacts reproduce and 428 push blocks / 109 host structs
agree. Rebuild a kernel with the compiler release and flags in
`validation/training_shader_manifest.json`; the gate uses glslangValidator
16.6.0, `-V --target-env vulkan1.1`. Existing SPIR-V is sufficient for initial
Rust qualification; tuning requires the pinned shader toolchains. Re-run
`validation/check_shader_provenance.py` and the push-layout checker after edits.

The application's parameter streaming remains opt-in (`--stream-parameters`,
or the unchecked GUI toggle). The probe's `--residency default` means the native
V4.1 package loader's streaming policy, while `resident` isolates compute. Its
`decode_ms_per_token` now reports the median single-token call directly. Older
probes divided that value by `--decode` twice; use raw `v41_probe_phase` records
when comparing historical results. A final-row digest is only a performance
guard: the Rust qualification gates compare complete logits to frozen CPU data.

To take an unprofiled target measurement after the gates pass:

```powershell
cargo build --locked --release --manifest-path hierarchos-vulkan/Cargo.toml --bin hierarchos-vulkan-deepseek-v41-forward-probe -j 1
.\hierarchos-vulkan\target\release\hierarchos-vulkan-deepseek-v41-forward-probe.exe --prefill 8 --decode 8 --steps 3 --warmup 1 --residency resident
.\hierarchos-vulkan\target\release\hierarchos-vulkan-deepseek-v41-forward-probe.exe --prefill 8 --decode 8 --steps 3 --warmup 1 --residency default
```

Record load time/submissions, raw prefill/decode phases, device/vendor, final-row
digest and the `v41_probe_io` cache counters. Disable profiling and submission
traces for timings; do not compile or run another GPU workload concurrently.
Routing counters still require host readbacks. Profile those costs on Gen9
before changing the worklist or packed GEMM strategy; RDNA3 timing improvements
do not predict Gen9 gains. Do not assume optional native FP16 or a subgroup size.

Preserve the measured numerical contracts: FP32 GEMM K panels, narrow score
dot ordering, BF16 single-row versus multi-row grouped accumulation, all BF16
rounding boundaries, HC/Engram cascaded reductions, and ascending expert-id
accumulation. FP8 scales/rounding and each block's multiply/add order are part
of the contract. The recorded width case has 14 intermediate grouped-rounding
differences that become identical after FP8 quantization; this is not permission
to weaken the full-logit ceiling during optimization.

PEFT additionally depends on the CPU input-adjoint reduction order: one-row
GEMMs use eight-element partials with separately rounded products, and two-row
GEMMs use four-product groups. Keep those paths, the CPU-matching normalization
and transcendental helpers, and their rounding boundaries. Replacing them with
native approximate math or changing contraction can break training parity even
when one inference prompt still matches.

Large generated width packages, obsolete PyTorch copies and abandoned scratch
were cleaned after qualification. CPU fixtures, generators, shard hashes and
compressed evidence remain. Rebuilding multi-GB packages is not required to
start the small-checkpoint Gen9 gate.

The portable handoff ZIP contains source for Vulkan, inference, CLI and GUI, locked
Cargo dependencies, checked-in SPIR-V and shader sources, frozen CPU fixtures,
the 44.8 MB joint checkpoint, historical receipts and this guide. Cargo must
fetch its registry dependencies unless already cached; Vulkan drivers and the
Rust toolchain are prerequisites. Python and remote model code are not needed
to execute the Rust gates. `HANDOFF-MANIFEST.json` binds every payload's size and
SHA-256; the packaging script verifies the archive by reading every entry back.
Historical receipts keep original-machine evidence paths and are provenance,
not a Gen9 qualification. Regenerate the bundle after source changes with
`validation/prepare_deepseek_v41_gen9_handoff.py --output OUTPUT.zip` from the
Vulkan directory (or use the full script path from the repository root).
