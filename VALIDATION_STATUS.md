# Current validation scope

This is the DeepSeek V4.1 Flash / streaming handoff for Intel Gen9 optimization.
**Current-revision RDNA3 validation covers DeepSeek V4.1 Flash and the recorded
parameter/MoE streaming checks only.** It does not certify the complete
architecture registry or all training and precision policies on RDNA3.

| Device / revision | Validated scope |
| --- | --- |
| AMD Radeon Graphics / Ryzen Z1 Extreme, current handoff | V4.1 Flash native resident/streamed inference, LoRA/rsLoRA training, save/resume/export, and the recorded parameter/MoE streaming matrix |
| Intel Gen9, current handoff | Pending a fresh device qualification before optimization |
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
