# Physical RDNA3 regression

Use `codex/gen9-final-rdna3-qualification` to test the final Gen9 port on the
RDNA3 device. Gen9 qualification is complete; a fresh physical RDNA3 result is
still pending. Keep this branch as the regression candidate until that run
passes. The earlier verified RDNA3 receipts bind the original handoff.

From the repository on the RDNA3 machine:

```powershell
git fetch origin
git switch --track origin/codex/gen9-final-rdna3-qualification
python -m zipfile -e hierarchos-vulkan/validation/release_parity/gen9_final/20261101/evidence.zip target/rdna-final-reference
```

Use a clean checkout or preserve local changes before switching. If the local
branch already exists, switch to it and fast-forward it from its upstream.
Rust, Vulkan drivers and Python 3.10+ are required; consuming the frozen CPU
references does not require installing Torch or Transformers.

On Windows, identify the AMD adapter's `DEV_XXXX` PCI identifier:

```powershell
Get-CimInstance Win32_VideoController | Select-Object Name, PNPDeviceID
```

Set `RDNA_DEVICE_ID` to `0x` followed by those four hexadecimal digits for the
known RDNA3 adapter. Then run from the repository root:

```powershell
$rdnaOutput = "target/rdna-final-" + [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
python hierarchos-vulkan/validation/run_rdna3_final_regression.py --forward-fixtures target/rdna-final-reference/cpu/forward --generation-fixtures target/rdna-final-reference/cpu/generation --generation-packages target/rdna-final-reference/cpu/generation-packages --expected-device-id $env:RDNA_DEVICE_ID --output $rdnaOutput
```

The runner rebuilds release probes and library tests, verifies the requested
physical AMD adapter in every forward/generation report, and runs GPU tests
sequentially. It consumes the same frozen CPU words at `max_abs < 2e-7`,
`rtol=0`. Both resident and streaming modes are required. It also runs V4.1
default/explicit release gates, native PEFT trajectories, streaming, cache,
Qwen3-Next backward and Vulkan regressions.

Keep the complete output directory, including its `qualification.json` and
logs. A failed gate leaves a failing/incomplete receipt; use a fresh output
directory for another attempt and preserve the earlier result. Do not
regenerate or relax the frozen expectations to obtain a pass.

The [final Gen9 receipt](hierarchos-vulkan/validation/GEN9_FINAL_RECEIPT.md)
records 269 passing native tests, 166 forward runs, 72 generation runs and the
Llama streaming closure. Its evidence ZIP and historical archives are immutable
snapshots taken before publication of this regression branch.
