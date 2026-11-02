# Current validation scope

The [November 2 RDNA3 closure](hierarchos-vulkan/validation/RDNA3_GPT2_CLOSURE.md)
passes **166/166 forward runs** across 43 profiles / 26 architecture identifiers
and **72/72 generation requests**. The padded GPT-2-wide gap is closed at the
unchanged strict `<2e-7`, `rtol=0` gate with the original CPU words. V4.1 native
PEFT, resident/streamed GPT-2 CPU PEFT, offload and streaming regressions pass.
The new math is selected only for RDNA3 GPT-2; all 585 existing shader binaries
are unchanged and Gen9 routing is guard-tested. This host has no physical Gen9
adapter, so no new physical Intel run is claimed. This remains a scaled-checkpoint
certificate, not complete architecture-registry or all-profile CPU training coverage.

The [final Gen9 receipt](hierarchos-vulkan/validation/GEN9_FINAL_RECEIPT.md)
closes the Llama-wide streaming regression: 11.016 to 4.597 ms against the
preceding pass, with 19 submissions reduced to four. Bounded Gen9 upload
batching preserves every measured logit word. All 269 selected Rust tests,
166 forward runs and 72 generation runs pass at `max_abs < 2e-7`, `rtol=0`.
The broader eight-control run shows lower resident and streaming medians.
This is the final Gen9 optimization pass. The frozen-reference RDNA3 runner is
available; the physical closure above records the complete measured matrix. The
[earlier optimization receipt](hierarchos-vulkan/validation/GEN9_OPTIMIZATION_RECEIPT.md)
retains the original measurements and the exception this final pass closes.

The Gen9 changes are published for device testing on
`codex/gen9-final-rdna3-qualification`. Follow the
[RDNA3 regression runbook](RDNA3_REGRESSION.md) for a fresh AMD run; the scoped
physical result is linked above. Qualification archives are immutable
snapshots; only the new closure receipt claims the new physical RDNA3 pass.

The next Gen9 architecture-port phase is recorded in the
[architecture receipt](hierarchos-vulkan/validation/GEN9_ARCHITECTURES_RECEIPT.md):
166/166 resident/streamed forward runs across 43 profiles / 26 native architectures
pass at `max_abs < 2e-7`, `rtol=0`, worst `1.7881393432617188e-7`.
All 32 earlier RDNA profiles are covered for inference. Another 72 cached/full-prefix
generation requests pass, worst `8.940696716308594e-8` (SmolLM3 retains its
full-prefix fallback when caching is requested). Storage modes match every logit
word. The fresh V4.1 model gate passes 53 tests; six native regressions and 14
qualification negative controls pass. This does not qualify the remaining 120
architecture identifiers or extend every profile's CPU PEFT training certificate.

The Intel Gen9 optimization follow-up is recorded in the
[Gen9 receipt](hierarchos-vulkan/validation/DEEPSEEK_V41_GEN9_RECEIPT.md).
The 2026-11-01 Gen9 continuation passes all 189 default V4.1 tests with zero
failures (25 ignored by default), at the unchanged strict CPU logit limit.
The four explicit release/training gates compare 11,184 logits bit-exactly.
Vulkan, parameter streaming and offload bring the selected Gen9 gate to 250
passing tests. All five performance digests and 68 original frozen files remain
unchanged. Historical toolchain provenance still has 54 unverified entries;
see the [fixture audit](hierarchos-vulkan/validation/V41_FIXTURE_PROVENANCE.md).

**The published RDNA3 validation below binds the original handoff, before the
Gen9 shader changes.** It covers V4.1 Flash and recorded parameter/MoE streaming
checks, not the complete architecture registry or all precision policies.
Those historical receipts do not add current-revision physical qualification;
the November 2 closure above supplies the fresh RDNA3 result.

| Device / revision | Validated scope |
| --- | --- |
| AMD Radeon Graphics / Ryzen Z1 Extreme, 2026-11-02 GPT-2 closure | 166/166 forward runs, 72/72 generation requests; V4.1 native PEFT and resident/streamed GPT-2 CPU PEFT; 269 selected native tests, 50 GPT-2 tests and unchanged Gen9 shader artifacts |
| AMD Radeon Graphics / Ryzen Z1 Extreme, published `2695c93` handoff | V4.1 Flash native resident/streamed inference, LoRA/rsLoRA training, save/resume/export, and the recorded parameter/MoE streaming matrix |
| Intel HD Graphics 520, 2026-11-01 Gen9 follow-up | Retained scaled-checkpoint inference, resident/streamed execution, wide YaRN and native PEFT gates; 250 passing tests, plus all-target workspace and shader checks |
| Intel HD Graphics 520, 2026-11-01 architecture continuation | 166 forward runs, 72 generation runs, 26 native architectures; resident/streamed word identity; 53 fresh V4.1 model/training tests and six native regression tests |
| Earlier Intel and ROG Ally revisions | Historical evidence in their existing reports; not current-revision certification |

The final RDNA3 pass has 288 passing tests. Parsed default-gate logits cover
201,714 values, worst absolute drift `1.4901161193847656e-7`; the strict gate is
`max_abs < 2e-7`, `rtol=0`, against immutable CPU PyTorch expectations. The four
explicit package/resident/streamed/PEFT gates compare 11,184 logits bit-exactly.
Streaming stays opt-in through `--stream-parameters` or the unchecked GUI toggle.

Qualification uses the accepted scalable-checkpoint criterion. The retained
44.8 MB inference package exercises the 40-layer schedule, 384 experts/top-6,
HC4, compression, shared KV/index state and Engram with reduced widths/tables.
PEFT training uses six reduced-width layers with the production expert ratio.
Full production weights and CUDA are not required or claimed.

See the [final receipt](hierarchos-vulkan/validation/release_parity/deepseek_v41/rdna3-final-push.json)
and [streaming evidence](hierarchos-vulkan/validation/release_parity/parameter_streaming/).
The [published final-pass logs and profiles](hierarchos-vulkan/validation/release_parity/deepseek_v41/receipts/final-push/)
preserve the measured runs in compressed form. `HANDOFF-MANIFEST.json` binds the
published payloads; Git attributes preserve their qualified bytes on Windows.
Machine-specific paths in historical receipts are provenance; the Git checkout
contains the Rust sources, shaders, frozen CPU fixtures and retained joint
checkpoint needed for a fresh Gen9 run.

Pull the handoff branch on the Gen9 machine:

```powershell
git fetch origin
git switch --track origin/codex/deepseek-v41-gen9-handoff
powershell -ExecutionPolicy Bypass -File hierarchos-vulkan/validation/run_deepseek_v41_gen9_gate.ps1
```

For a new clone:

```powershell
git clone --branch codex/deepseek-v41-gen9-handoff https://github.com/necat101/Hierarchos-Native.git
```

Keep frozen expectations unchanged. Record a separate Gen9 receipt and profile
that device before choosing kernel tiles, workgroup sizes or fusion changes.
The [Gen9 guide](hierarchos-vulkan/validation/intel-handoff/DEEPSEEK_V41.md)
contains the gate, profiling commands and arithmetic contracts.
