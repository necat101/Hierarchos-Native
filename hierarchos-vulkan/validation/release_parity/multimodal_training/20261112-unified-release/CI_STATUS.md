# Release delivery status

Delivery is **BLOCKED on fixture provenance**. The local numerical seal remains `PASS_SCOPED_RELEASE`; it is not a claim that all hosted workflows are green.

Completed: 13/13 packed benchmark/guard pairs; maintained FP32 93/93 images and 166/166 text comparisons; bounded V4.1; 20-update precision training/resume; nine wide Mixed cases; eight actual GUI Start workflows; 1159 library, 50 CLI, 17 GUI and 28 unified fault-control tests. Mixed AMP-style WMMA + FP32 remains wired into the GUI. Full FP16 retains its qualified profile/reduction scope.

The first hosted workspace failure exposed 72 adapter/optimizer evidence files excluded by ignore rules. All 72 are now committed, and every release file is tracked. A separate checkout passed 33 of the 34 CPU commands in the actual workspace workflow.

The remaining mandatory command is `fixture_toolchain.py check-all`. These frozen inputs lack verified pins:

- `deepseek_v41_native_vision_avx2.json` — original SHA-256 `f6bb7f095bb829f1124fd528ed79d36d12e8fc2fe041a89734c9a2ab4cdfc439`.
- `deepseek_v41_native_pipeline_avx2.json` — original SHA-256 `9f5592989820c1567caf3fd4ca6b5354f7bb8e78dfb6b38f1ebb43b9b4cc403b`.

Both original files are unchanged. Local/hosted regeneration has not reproduced them exactly, so no verified pin was added, no legacy exception was added, and the gate remains enabled. The original generator environment (including CPU/backend configuration) or a verified original reproduction record is needed. Detailed failed attempts and workflow URLs are preserved under `../20261113-phase22-recovery/native-provenance-blocker/`.

The 194 additional strict intermediate failures and native stress exclusions remain visible in the numerical report. This candidate is not declared release-ready until the provenance gate and all four hosted workflows pass at one final commit.
