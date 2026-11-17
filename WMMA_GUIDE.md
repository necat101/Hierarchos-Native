# RDNA3 WMMA guide (opt-in cooperative-matrix path)

This is the user-facing guide for the RDNA3 WMMA release. It describes how to
turn the path on, what it does to the numbers, and what it deliberately does
**not** claim. Deep engineering history lives in
[`RDNA3_WMMA.md`](RDNA3_WMMA.md); the machine-checked numerical scope lives in
[`RDNA3_RELEASE_CONTRACT.md`](RDNA3_RELEASE_CONTRACT.md); the sealed
qualification receipt is
[`20261117-phase42-current-source`](hierarchos-vulkan/validation/release_parity/multimodal_training/20261117-phase42-current-source/release-qualification.json).

> [!IMPORTANT]
> WMMA is **opt-in and experimental**. `off` is the default. On `off` the FP32
> Vulkan path is byte-for-byte the same code path it was before this release,
> and its logits are bit-identical to the previous `main` (see
> [FP32 non-regression](hierarchos-vulkan/validation/PERFORMANCE_REGRESSION.md)).

---

## 1. The three execution arms

| Arm | CLI value | Operands | Accumulation | Master weights | AdamW moments | Norm / softmax / loss |
| --- | --- | --- | --- | --- | --- | --- |
| FP32 (default) | `off` | FP32 | FP32 | FP32 | FP32 | FP32 |
| Mixed WMMA | `mixed` | FP16 | **FP32** | FP32 | FP32 | FP32 |
| Full FP16 | `full-fp16` | FP16 | **FP16** where advertised | FP32 | FP32 | FP32 |

- **`off`** — the compatibility path. No cooperative-matrix feature is queried
  and no WMMA pipeline is created. This is what an unset
  `HIERARCHOS_WMMA_MODE` produces, and what every earlier release did.
- **`mixed`** — the accelerated path. Eligible projections run as
  FP16 × FP16 with FP32 accumulation; parameters, gradients, optimizer moments
  and every normalisation/softmax/loss operation stay FP32. This is the arm the
  FP32-versus-Mixed numerical gates target.
- **`full-fp16`** — genuine FP16 cooperative accumulation, widened back to FP32
  on the GPU, restricted to reductions up to
  `FULL_FP16_MAX_REDUCTION = 64`. Longer reductions run the original FP32
  kernels and increment a fallback counter. This arm is admitted only for the
  numerically qualified Llama-64 profile on the qualified adapter/driver
  (see §2), and it is refused everywhere else.

Turning on a mode **never** silently degrades to FP32 while reporting WMMA: a
device that cannot satisfy the request fails with the probed reason.

---

## 2. Hardware and runtime requirements

| Requirement | Detail |
| --- | --- |
| OS | Windows x64 is the qualified and packaged platform. The Rust/Vulkan sources build on other desktop Vulkan platforms, but no physical qualification exists there. |
| Driver | A Vulkan 1.x compute driver. The qualified record is the AMD driver reporting `32.0.31007.6002` (internal key `00800184`) on `AMD Radeon Graphics`. |
| GPU | **RDNA3** with `VK_KHR_cooperative_matrix`, a SUBGROUP-scope `16x16x16` f16/f16→f32 tile (Mixed), FP16 storage and `VK_EXT_subgroup_size_control`. |
| Qualified PCI ID | **`1002:15bf`** only — the Ryzen Z1 Extreme / ROG Ally class device used for every measurement here. |
| Full FP16 | Additionally requires the exact qualified adapter/driver key and the 64-wide one-layer Llama profile (64 intermediate, 64 vocab, 8 heads, no dropout, 33–64 batch×seq rows). |

Not qualified, and not implied by this release: other RDNA3 parts, RDNA1/RDNA2,
Intel Gen9, multi-GPU device groups, and any non-Windows target. RDNA1/2 keep
their existing AMD FP32 route and Gen9 keeps its pre-WMMA route; both are
unchanged by this release.

Check what your machine actually negotiates before trusting any claim:

```powershell
hierarchos-native-cli.exe precision-info
hierarchos-native-cli.exe precision-info --device-index 0
```

It prints one JSON object — `mixed_supported`, `full_fp16_supported`,
`full_fp16_qualified`, `full_fp16_max_reduction`, the device name and the
decoded `device_key`. `full_fp16_supported` describes the driver;
`full_fp16_qualified` describes whether *this* project vouches for the numbers.

---

## 3. Enabling and using it

WMMA applies to the Transformer trainer commands
(`transformer-train`, `transformer-finetune`), which forward the policy to the
Vulkan trainer they launch. It is orthogonal to `--amp` / `--precision`: those
control storage/precision of the existing path, `--wmma-mode` controls the
cooperative arm.

```powershell
# 1. Default FP32 path — no flags, no cooperative dispatch.
hierarchos-native-cli.exe transformer-train `
  --model-path .\model --train .\dataset.jsonl --out-dir .\trained `
  --epochs 1 --batch-size 1 --seq-len 128 --lr 5e-5 --device-index 0

# 2. Same run with Mixed WMMA.
hierarchos-native-cli.exe transformer-train `
  --model-path .\model --train .\dataset.jsonl --out-dir .\trained-mixed `
  --epochs 1 --batch-size 1 --seq-len 128 --lr 5e-5 --device-index 0 `
  --wmma-mode mixed

# 3. LoRA fine-tuning with Mixed WMMA (adapter path).
hierarchos-native-cli.exe transformer-finetune `
  --model-path .\model --train .\dataset.jsonl --out-dir .\adapter-mixed `
  --lora-rank 8 --lora-alpha 16 --epochs 1 --batch-size 1 --seq-len 128 `
  --lr 1e-4 --device-index 0 --wmma-mode mixed

# 4. Full FP16, only where precision-info reports full_fp16_qualified=true.
hierarchos-native-cli.exe transformer-train `
  --model-path .\llama64 --train .\dataset.jsonl --out-dir .\trained-full `
  --epochs 1 --batch-size 2 --seq-len 24 --lr 5e-5 --device-index 0 `
  --wmma-mode full-fp16

# 5. Inference is unaffected by WMMA: generation always runs the FP32 route.
hierarchos-native-cli.exe infer --model-path .\trained-mixed `
  --prompt "Explain Vulkan compute." --max-new-tokens 128 --device-index 0

# 6. Inspect the route and the counters.
hierarchos-native-cli.exe precision-info --device-index 0
```

Accepted spellings, as implemented by `WmmaMode::parse` and the CLI:

| Spelling | Result |
| --- | --- |
| `off`, `none`, `fp32`, `--no-amp` | FP32 |
| `mixed`, `amp`, `mock-amp`, `--amp` | Mixed WMMA |
| `full-fp16`, `full_fp16`, `fullfp16`, `fp16` | Full FP16 |

Anything else is rejected: `unsupported --wmma-mode "turbo"; expected off, mixed, or full-fp16`.

The same policy can be set for a whole process tree through the environment,
which is how the CLI hands it to the Vulkan trainer it spawns:

```powershell
$env:HIERARCHOS_WMMA_MODE = "mixed"
```

`--help` on `transformer-train` or `transformer-finetune` prints the authoritative
flag list and the precision contract text.

---

## 4. Dispatch policy and fallback behaviour

WMMA is admitted per projection, not per process. A projection takes the
cooperative route only when **all** of these hold:

1. The mode is `mixed` (or `full-fp16` where qualified).
2. The device negotiated every required feature (otherwise the request fails).
3. The GEMM clears the profitability floor:
   `WMMA_MIN_PROFITABLE_MACS = 16_000_000`
   (`rows × input_dim × output_dim`), at least 16 rows/columns/K, at least 1024
   outputs, and at least four 32×32 output tiles.

Everything else keeps the tuned FP32 kernel. That is a **policy fallback**, and
it is counted separately from a dispatch.

The counters are deliberately distinct, so "requested but not dispatched" is
visible rather than implied:

| Counter | Meaning |
| --- | --- |
| cooperative dispatch count | a WMMA pipeline actually ran |
| phase count (`[forward, dW, dX]`) | which cooperative phases ran |
| `plain_f16` count | FP16-storage control kernel — **not** WMMA |
| shape / policy fallbacks | the FP32 kernel ran instead |
| Full FP16 reduction fallbacks | reduction exceeded 64 and stayed FP32 |

Consequences worth knowing before reading any receipt:

- Wide multimodal and native V4.1 fixtures are CPU-scale (8×8 images,
  `hidden_size` 128), so at their released geometry **no** projection reaches
  16M MACs. Their receipts record `accumulation_dispatches [0, 0]` and phases
  `[0, 0, 0]`: they are numerical qualifications of the FP32 route they actually
  executed, not evidence of cooperative dispatch.
- The precision-training evidence family — the components that do cross the
  floor — records real cooperative dispatches (a `192×24 = 4,608`-row text
  trajectory reports `[420, 0]` dispatches with phases `[140, 140, 140]`).
- Full FP16 on an oversized reduction increments the fallback counter and claims
  no accumulation; a 77-fallback run is a correct fallback run, not a WMMA run.

---

## 5. Numerical correctness and safety

- **FP32 masters survive every mode.** Mixed and Full FP16 hold FP16 execution
  mirrors that are refreshed from the FP32 masters on each dispatch, so a
  checkpoint, AdamW update or weight swap can never leave a stale FP16 copy
  feeding the graph. AdamW updates the FP32 masters.
- **Loss scaling and overflow.** Standalone GPT-2/Llama full training, ordinary
  LoRA, and shared CLIP/LLaVA joint transactions implement GPU loss scaling with
  global optimizer skipping. The executed overflow test skips both updates at
  `2^78` without touching either master set and preserves four skipped steps at
  `2^76` when scaler state is restored. Other architecture trainers keep their
  existing unscaled FP32 gradient policy, and automatic scaling is **not**
  claimed for them — FP32 gradient *storage* does not prevent quantisation once
  a gradient becomes an FP16 GEMM operand.
- **Checkpoint / export / reload.** Exported packages keep canonical FP32
  tensors; Mixed adapters resume bit-identically when the same base, data, batch,
  sequence and optimizer settings are used. Exact continuation still requires
  the native `training-state/` sidecar, not just the exported weights.
- **Tolerances are not a promise about your model.** The gates are
  `mixed` ≤ `2e-2` max / `3e-3` RMS against FP32, Full FP16 ≤ `5e-2` / `1e-2`,
  and the unrelated FP32 CPU gate stays at `max_abs < 2e-7`, `rtol=0`. They were
  measured on the frozen fixtures named in the receipt. They are not a warranty
  that an arbitrary checkpoint stays inside those numbers.

---

## 6. Performance, measured

Three different things, which must not be conflated:

**FP32 non-regression against the previous `main`** (`ead6cd3` → `5d40f68`,
seven workloads, 12 alternating ABBA rounds, 5 warm-up + 30 timed forwards,
distinct binaries). The statistic is the median of the twelve per-round
candidate/stable ratios:

| Workload | Paired ratio | Verdict |
| --- | ---: | --- |
| gpt2-unmasked | 0.9996 | equivalent |
| gpt2_wide-unmasked | 1.0007 | equivalent |
| gpt2_wide-padded | 0.9996 | equivalent |
| llama_wide-unmasked | 0.9617 | faster |
| llama_wide-padded | 0.9903 | equivalent |
| gpt2-wide-decode | 0.9962 | equivalent |
| gpt2-wide-batch32 | 0.9987 | equivalent |

Every workload is inside the ±2 % band, and FP32 logits were **bit-identical**
between the two arms in all 12 rounds of all seven workloads. The worst ratio is
`1.0007`, i.e. a 0.07 % difference — reported as equivalent, never as faster.

**Kernel-level WMMA** (single-GEMM latency, release build, one submit+fence per
measurement): `2.170×` at 512×768×768, `2.120×` at 512×2048×768, `2.309×` at
2048×768×2048, `1.832×` at 128×8192×768. Below the crossover the same kernel is
*slower* — `0.738×` at 64×64×64 and `0.758×` at 16×128×32 — and plain FP16
storage without matrix cores is slower than the FP32 kernel at every size
(`0.334×`–`0.975×`). The gain comes from matrix cores, not from FP16.

**End-to-end training is not faster.** In the measured eligible-geometry fixture
(512 rows/step, release, ABBA) the Mixed step was **slower** than FP32:

| Arm | Step time |
| --- | ---: |
| FP32 | 12.2911 ms |
| Mixed | 14.0444 ms |
| Paired ratio | 0.8708 (CI 0.8517–0.8898) |

That is why Mixed stays **off by default**, and why no end-to-end training
speedup is claimed anywhere in this release. Kernel ratios must not be
extrapolated to a training pipeline.

---

## 7. Known limitations

- Cooperative dispatch is qualified on **one** PCI ID (`1002:15bf`) and one
  driver record. Other RDNA3 devices may negotiate the features and still be
  unqualified numerically.
- Full FP16 is an experimental, narrow arm: the qualified Llama-64 profile plus
  a reduction cap of 64. It is refused for Transformer graphs that cannot widen
  FP16 accumulation back into FP32 masters.
- At released wide-multimodal and native V4.1 geometry the profitability floor
  declines every projection, so those receipts are FP32-route qualifications
  with zero cooperative dispatches.
- 194 of the 1,116 strict intermediate diagnostics remain failures of that
  additional internal-tensor check (186 vision-feature, 8 projected-gradient).
  They are disclosed, not relabelled, and the maintained logit gate is unchanged.
- Mixed end-to-end training is slower than FP32 on the measured fixture, as
  tabulated above.
- No physical RDNA3 CI runner is registered, so the RDNA3 receipts are
  source-bound local qualifications rather than a hardware-CI pass.

---

## 8. Where the evidence lives

| Artifact | Path |
| --- | --- |
| Sealed release receipt (2,004 sources, 32 evidence files, 33 binaries) | `hierarchos-vulkan/validation/release_parity/multimodal_training/20261117-phase42-current-source/` |
| FP32 non-regression, 12 rounds × 7 workloads, raw JSON | `hierarchos-vulkan/validation/performance/fp32-nonregression/20261117-phase42/` |
| Human-readable FP32 table and protocol | `hierarchos-vulkan/validation/PERFORMANCE_REGRESSION.md` |
| Numerical scope, machine-checked | `RDNA3_RELEASE_CONTRACT.md` |
| Engineering history and rejected routes | `RDNA3_WMMA.md` |
| Precision completion summary | `RDNA3_PRECISION_COMPLETION.md` |

Reproduce the release gates (no GPU required):

```powershell
python hierarchos-vulkan/validation/check_evidence_integrity.py --latest-release --strict
python hierarchos-vulkan/validation/check_unified_release.py
python -m unittest hierarchos-vulkan.validation.test_unified_release
```
