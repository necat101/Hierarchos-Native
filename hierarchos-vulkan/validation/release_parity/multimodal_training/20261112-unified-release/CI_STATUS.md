# Release delivery status

The qualification target is **AVX512 + RDNA3**. Native AVX2 fixture reproduction is **PENDING_ON_AVX2_HOST**, as assigned by the user. It is a separate host qualification and does not block this target. The local numerical seal remains `PASS_SCOPED_RELEASE`; final hosted workflow conclusions must be checked at the published commit.

Completed numerical qualification: 13/13 packed benchmark/guard pairs; maintained FP32 93/93 images and 166/166 text comparisons; bounded V4.1; 20-update precision training/resume; nine wide Mixed cases; eight actual GUI Start workflows; 1159 library, 50 CLI and 17 GUI tests. Mixed AMP-style WMMA + FP32 remains wired into the GUI. Full FP16 retains its qualified profile/reduction scope. The 194 additional strict intermediate failures and native stress exclusions remain disclosed.

The AVX512 oracle archives retain their independent byte-verification workflow. Existing references keep their original recorded ISA; assigning this executable target to AVX512 does not relabel historical AVX2 goldens.

The workspace provenance gate now uses `fixture_toolchain.py check-all --target-cpu AVX512`. It still checks every existing verified pin, legacy hash and pending-host assignment. Only these exact original fixtures are assigned to the other host:

- `deepseek_v41_native_vision_avx2.json` — SHA-256 `f6bb7f095bb829f1124fd528ed79d36d12e8fc2fe041a89734c9a2ab4cdfc439`.
- `deepseek_v41_native_pipeline_avx2.json` — SHA-256 `9f5592989820c1567caf3fd4ca6b5354f7bb8e78dfb6b38f1ebb43b9b4cc403b`.

The hash-bound assignment file is [fixture_host_qualifications.json](../../../fixtures/fixture_host_qualifications.json). The sealed receipt binds that policy and independently re-derives its audit report. AVX512 scope reports both fixtures as pending, with no verified pin or legacy exception. The unscoped audit and `--target-cpu AVX2` still fail until exact reproduction on the assigned CPU succeeds. Successful reproduction preserves the original fixture bytes, records the verified pin, and removes the pending assignment.

On the AVX2 qualification host, use the existing fixture generators with their recorded inputs and toolchain, then run:

```powershell
python hierarchos-vulkan/validation/fixture_toolchain.py check-all --target-cpu AVX2 --expect-from hierarchos-vulkan/validation/fixtures/deepseek_v41_mixed_block128.json
```

All four required hosted workflows must pass on the final AVX512/RDNA3 commit. PowerShell checks and their unit tests remain separate steps so a successful later command cannot hide an audit failure. The earlier masked result, reproduction attempts and previous blocked assessment remain preserved under `../20261113-phase22-recovery/`. The packaging omission of 72 adapter/optimizer evidence files is fixed and those files remain tracked.
