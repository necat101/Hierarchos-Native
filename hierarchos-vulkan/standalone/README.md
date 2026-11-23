# Hierarchos Native — Windows x64 Portable v0.2.6

This is the runtime package for the Hierarchos native Transformer backend. It is
extract-and-run on a supported Windows x64 computer: no installer, no Rust or
Cargo, no Python or PyTorch, no Visual Studio, no administrator setup, and no
separate Microsoft Visual C++ Redistributable installation.

Release version **0.2.5**. The package version is independent of the internal
crate versions in `Cargo.toml`, which keep their own established numbering.

The seven release executables are linked with the static MSVC C runtime
(`-C target-feature=+crt-static`). Their PE import tables are checked during
packaging; the package must not contain `VCRUNTIME140.dll`, `MSVCP140.dll`, or
another Visual C++ runtime DLL.

The archive is intentionally runtime-only. Source, Cargo manifests, shader
sources, build intermediates, fixtures, model weights, caches, and local test
logs are not included. Source remains available in the
[Hierarchos-Native repository](https://github.com/necat101/Hierarchos-Native).

## Package layout

```text
bin/
  hierarchos-native.exe                 # desktop GUI
  hierarchos-native-cli.exe             # CLI and linked Transformer training
  hierarchos-infer.exe                  # pure-Rust coherent-v9 inference
  hierarchos-vulkan-train.exe           # Hierarchos native training
  hierarchos-vulkan-devices.exe         # Vulkan device inventory
  hierarchos-vulkan-transformer-train.exe
  hierarchos-vulkan-transformer-logits.exe
LICENSE
THIRD_PARTY_NOTICES.md
licenses/
README.md
README_ARCHITECTURES.md                 # native architecture inventory
RELEASE_NOTES.md
CLEAN_MACHINE_TEST.md                   # v0.2.6 local acceptance report
BUILD_INFO.txt                          # source, lockfile, flags, binary hashes
DEPENDENCIES.txt                        # PE import closure for all seven EXEs
```

Keep all seven executables together in `bin/`. The CLI and GUI resolve their
companions relative to the executable directory. Do not move individual
executables into a different directory.

## Requirements

- Windows x64. Windows 11 x64 is the measured acceptance target; Windows 10 x64
  is expected to work but was not measured.
- A Vulkan 1.x loader and a compute-capable Vulkan GPU driver. The Vulkan loader
  (`vulkan-1.dll`) is supplied by the graphics software environment; **it is not
  bundled**, and it is not a substitute for the GPU driver. Windows and the GPU
  driver remain external runtime requirements.
- Enough RAM and disk for the selected model and output. **No model weights are
  bundled.**
- Models and outputs may be stored anywhere the user can read/write; they are
  not tied to the installation directory.

No Microsoft Visual C++ Redistributable is required by these executables. No
administrator rights, registry installation, machine-wide environment variable,
or installer is required for ordinary execution. Python, PyTorch, CUDA, a
Transformers checkout, the Vulkan SDK, and a shader compiler are not runtime
dependencies.

## WMMA is opt-in, and Full FP16 runs the P3 contract on C1 staging

This is the precision policy of the v0.2.6 binaries, stated precisely:

- **Global WMMA is off by default.** The process default is `WmmaMode::Off`; the
  FP32 Vulkan path is unchanged. No cooperative-matrix dispatch happens unless a
  mode is requested.
- **`--wmma-mode full-fp16` explicitly enables the Full FP16 path** (equivalently
  `HIERARCHOS_WMMA_MODE=full-fp16`).
- **`full-fp16` resolves `FullFp16Variant::P3` with no environment variable and
  no qualification flag**, exactly as in v0.2.5. P3 names the numerical contract
  — bank assignment, the float32 sum of the eight bank partials and the single
  binary16 rounding — and that contract is what both staging modules implement.
  `split8`, `two-bank` and `two-wave` remain selectable through
  `HIERARCHOS_WMMA_FULL_FP16_VARIANT` as rollback and comparison paths.
- **Two staging modules are embedded, and the binary chooses between them per
  dispatch.** Both are compiled into the executable: the v0.2.5 P3 module
  (23,280 B) and the C1 batched-staging module (36,912 B). C1 is used only when
  the host can prove, from the dispatch geometry and the bound operand
  capacities, that every packed 32-bit load is aligned and in bounds — full
  32-row/16-K tile coverage, even packed-pair strides, complete buffer coverage
  and indices well inside the 32-bit ceiling. Every other geometry — partial
  tiles, K not a multiple of 16, odd strides, short buffers — dispatches the
  original P3 module. There is **no** flag, environment variable or variant name
  that selects one staging schedule directly; the fallback is automatic and the
  counters in a receipt report which route ran.
- **Mixed mode remains separate and unchanged.** `--wmma-mode mixed` dispatches
  the FP32-accumulating cooperative module and does not consult the Full FP16
  variant.
- Admission is gated by the existing capability contract (AMD RDNA3-or-newer PCI
  ID, `VK_KHR_cooperative_matrix`, Vulkan memory model, 16-bit storage,
  `shaderFloat16`, subgroup-size control forcing 32, and a SUBGROUP-scope
  16x16x16 f16/f16->f16 tile). Unsupported hardware falls back rather than
  claiming the cooperative path.
- The **CLI frontend** additionally restricts Full FP16 text training to the
  64-wide, one-layer Llama profile. The packaged standalone trainer and logits
  executables use the same production selector without that frontend profile
  gate; they still enforce the capability contract, the 16M MAC floor and the
  reduction cap. A Full FP16 request does not imply that every operation will
  dispatch cooperatively: ineligible shapes and reductions above 64 keep their
  FP32 kernels and increment fallback counters.
- **The profitability floor is unchanged:** WMMA is admitted only at or above
  `WMMA_MIN_PROFITABLE_MACS = 16_000_000` multiply-accumulates, checked at every
  production admission point. Below that, the plain FP16-storage and FP32
  kernels are used.
- **The hard numerical gate is unchanged:** `0.03125` on the matched-oracle
  comparison. The historical 25 % tile16 headroom target remains informational
  and is not a release gate.

### Measured hardware scope

The v0.2.6 qualification applies to **one measured adapter**: AMD RDNA3 with PCI
vendor `0x1002`, device `0x15bf` (Phoenix-class integrated RDNA3), Vulkan SDK
1.4.321, driver `32.0.32015.14001`. It is **not** a claim for every RDNA3 adapter, for RDNA3.5, for
RDNA4, or for any non-AMD device. On other hardware the run is gated by the
capability contract and the results are unmeasured.

### What the performance numbers mean

The released figures are **per-dispatch GPU GEMM kernel timings** taken with
Vulkan timestamp queries around the compute dispatch. They are **not end-to-end
training-step speedups**, and no such claim is made.

## Quick start

Extract the archive anywhere writable, then from the extracted directory:

```powershell
# 1. First-run diagnostics
.\bin\hierarchos-native-cli.exe --help
.\bin\hierarchos-native-cli.exe doctor
.\bin\hierarchos-native-cli.exe devices
.\bin\hierarchos-native-cli.exe architectures
.\bin\hierarchos-native-cli.exe precision-info

# 2. Local inference against a model directory you already have
.\bin\hierarchos-native-cli.exe infer `
  --model-path C:\models\tiny-random-gpt2 `
  --prompt "Hello" --max-new-tokens 16 --no-do-sample

# 3. One-step native Vulkan training and checkpoint reload
.\bin\hierarchos-native-cli.exe transformer-train `
  --model-path C:\models\tiny-random-gpt2 `
  --train C:\data\one-row.jsonl `
  --out-dir C:\outputs\trained `
  --epochs 1 --batch-size 1 --seq-len 16 --lr 5e-5 --device-index 0

.\bin\hierarchos-native-cli.exe infer `
  --model-path C:\outputs\trained `
  --prompt "Reload checkpoint" --max-new-tokens 8 --no-do-sample

# 4. Desktop frontend
.\bin\hierarchos-native.exe
```

### First-run diagnostics you should see

- `doctor` reports Vulkan visibility and whether the expected companion
  executables resolve from `bin/`. A missing companion is printed as
  `not found`; `doctor` currently exits 0 for that diagnostic, so read its
  output rather than relying only on the exit code.
- `devices` prints one JSON record per Vulkan physical device. On the qualified
  host it reports `"name":"AMD Radeon Graphics"` with
  `"device_type":"INTEGRATED_GPU"`.
- `precision-info` reports the WMMA capability and admission state. A capability
  report is **not** proof that a cooperative dispatch happened; see the
  acceptance report for how an actual production dispatch is evidenced.

## Hub inference and the per-user cache

```powershell
.\bin\hierarchos-native-cli.exe infer `
  --hf-model hf-internal-testing/tiny-random-gpt2 `
  --prompt "Hello" --max-new-tokens 8 --no-do-sample
```

The default cache is `%LOCALAPPDATA%\Hierarchos\hf-cache`. Override it with
`HIERARCHOS_HF_CACHE` or `--hf-cache-dir`. The application does not need write
access to its installation directory.

## Portability and writable state

The extracted directory can be moved or renamed after extraction, including to a
path containing spaces or to a deeply nested directory. Keep the relative layout
intact.

Downloaded model data, Vulkan execution-cache metadata, temporary files, logs,
and training checkpoints use user-selected or per-user writable locations. The
application does not require installation-directory writes. If an explicit
output directory is supplied, that directory is used as requested.

The only external runtime DLL in the measured import closure besides Windows
system components is `vulkan-1.dll`. The GPU software stack supplies it.

## Verification status

The v0.2.6 package is built from the qualified source commit of
`codex/wmma-v026-optimization`; the exact packaging commit is the checked-out
HEAD recorded in `BUILD_INFO.txt` and `BUILD_MANIFEST.json`, together with the
archive and executable hashes.

The archive was extracted into a path containing spaces, relocated to a deeper
path containing spaces, and exercised from the relocated layout. PE closure,
static-runtime linkage, CLI help, diagnostics, inference, one-step Vulkan
training, checkpoint reload, checksum verification and relocation passed.

Both staging routes were verified from the package. The same disposable Llama
geometry with hidden/K = 64 and 4096-wide projections gives 16,777,216 MACs per
admitted projection, above the unchanged 16M floor and inside the 64-element
reduction cap; at `rows=64` it satisfies the C1 contract and the extracted
trainer reported `cooperative_dispatches=8` with the profile naming
`wmma_gemm_f16_f16acc_twowave_p3_c1`. The same fixture at `rows=80` fails the
tile-coverage condition while staying above the floor (20,971,520 MACs), and the
trainer fell back to `wmma_gemm_f16_f16acc_twowave_p3` for its 4 cooperative
dispatches. No variant override or threshold change was used. With WMMA off,
both geometries recorded zero cooperative dispatches. The packaged logits
executable matched FP32 within `max_abs 0.00079345` / `rms 0.00012127` on the C1
route and `0.00079345` / `0.00012087` on the P3 route. Those synthetic geometries
are outside the CLI's frontend-qualified 64-wide training profile; the test uses
the packaged standalone trainer's production path.

Numerical and performance qualification was re-run for this release on the
measured RDNA3 adapter. The full record — seeds, configurations, 37 gates per
receipt, raw paired timings, corrected 95 % confidence intervals, source and
SPIR-V hashes — is `validation/phase_lxxxviii_c1_promotion/` in the repository.
Exact commands, log excerpts and blocked items are in `CLEAN_MACHINE_TEST.md`.

No independent clean Windows installation, second physical CPU/GPU, physical
network-isolation test, or different-drive relocation was available. CPU
compatibility is analytically checked: the build uses the default
`x86_64-pc-windows-msvc` target, not `target-cpu=native`, and retains the
existing runtime feature dispatch.

## Build from source

The qualified kernel source commit is
`b165c2994e9ef5929c9e36fca76192db8dd3df35`. The package is built from the final
PR #9 commit (recorded in `BUILD_INFO.txt` and `BUILD_MANIFEST.json`) with the
checked-in standalone `Cargo.lock` and `--locked`. The static portable build is
produced by:

```powershell
.\hierarchos-vulkan\package-standalone.ps1 `
  -OutDir C:\path\to\hierarchos-v026-stage `
  -StaticCrt
```

The packager sets `-C target-feature=+crt-static`, uses the default Windows x64
target, copies exactly seven release executables, runs the PE import verifier,
and writes `BUILD_INFO.txt` and `DEPENDENCIES.txt`. Normal development builds
remain dynamically linked; static CRT is a release-packaging option. Keep the
default `-SkipDependencyScan`-free invocation: the dependency scan is part of the
release gate.

For a source checkout, the component build commands are documented in the root
[README](https://github.com/necat101/Hierarchos-Native#quick-start). The Vulkan
crate embeds its checked-in SPIR-V shader binaries at compile time; shader
compilers are not required for an ordinary release build.

## License and notices

Hierarchos Native is MIT licensed; see `LICENSE`. Third-party components and
reference-derived shader code retain their own terms in
`THIRD_PARTY_NOTICES.md` and `licenses/`. No model weights are bundled.
