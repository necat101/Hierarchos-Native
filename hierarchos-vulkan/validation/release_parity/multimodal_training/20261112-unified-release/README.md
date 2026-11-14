# Unified FP32, Mixed WMMA and scoped Full FP16 qualification

Local numerical and source-bound status: **PASS (scoped)**. The sealed machine-readable record is [release-qualification.json](release-qualification.json); [precision-capabilities.json](precision-capabilities.json) gives the family-by-family support contract. GitHub-hosted CI is a separate final-commit check; its actual run URLs and conclusions belong in the delivery report. Self-hosted hardware availability is never inferred from local PASS.

Sealed source revision: `a7422faa65b47d233bb7e44e26023b9055684e77`. Device: `AMD Radeon Graphics [pci=1002:15bf;driver=00800184]`. Release receipt SHA-256: `3f4fa7928666c3218dfaed106427f509562ed1a5e91134115228d506e804ad2f`. All source, executable, shader, component and raw-evidence digests are recorded in the receipt. No numerical acceptance limit was increased.

## Numerical qualification

| Scope | Status | Actual evidence |
|---|---|---|
| Maintained FP32 images | PASS | 93/93 across six original stages; maximum observed logit error 1.7881393432617188e-07; original loss, update and reload gates retained |
| Maintained FP32 text | PASS | 166/166 resident/streaming comparisons; maximum 1.7881393432617188e-07; strict `<2e-7`, rtol 0 |
| Additional strict intermediate diagnostics | FAIL / unqualified | 194/1116 fail; all raw rows retained |
| Native V4.1 bounded vision | PASS, bounded | 65 strict rows; maximum 1.5646219253540039e-07; 32 Mixed dispatches; exact reload |
| Native V4.1 stress | FAIL / unqualified | First-norm difference 2.384185791015625e-7; independent float64 diagnosis retained; no universal strict-equivalence claim |
| Exploratory geometry | Unqualified | Hidden64/vocab64 maximum 2.2351741790771484e-7; hidden128/vocab32 maxima 2.9802322387695312e-7 and 3.5762786865234375e-7; separate from maintained 93 |
| Mixed primitives | PASS | 37/37 |
| Original Full FP16 primitive cadence | FAIL, preserved | 9/37 fail; not relabeled by the matched-oracle result |
| Matched RDNA3 dot2 Full FP16 | PASS within admitted reductions | 28/28 K<=64; three K=96 versus-FP32 failures retained |

Full training and ordinary LoRA ran 20 optimizer updates in Off, Mixed and Full FP16. Accumulation ran 40 microbatches into 20 updates. Masters and AdamW moments remained FP32; deterministic exact LoRA resume included optimizer and scaler state. Forced global image/text overflow skipped both optimizers without changing weights, moments or clocks: two skips at 2^78, then four after restored state at 2^76. The real GPU underflow test preserves a gradient that otherwise rounds to zero as an FP16 operand.

| Trajectory | Loss maximum / RMS versus FP32 | Master maximum versus FP32 | Accumulation dispatches |
|---|---|---|---|
| full-mixed | 2.21729279e-05 / 1.15093467e-05 | 5.29699028e-05 | [480, 0] |
| full-full-fp16 | 0.000366210938 / 0.000137590513 | 0.000247489661 | [0, 480] |
| lora-mixed | 2.24113464e-05 / 1.17173697e-05 | 0.000102061778 | [320, 0] |
| lora-full-fp16 | 0.000106811523 / 5.2882133e-05 | 0.000203663483 | [0, 320] |

## Wide Mixed matrix

| Family / case | Status | Mixed forward, dW, dX | Worst qualified Mixed maximum / RMS |
|---|---|---|---|
| llava / clip_last_layer_padded | PASS, scoped | [189, 54, 54] | 0.00123545527 / 0.000304668142 |
| gemma3 / siglip_full_grid | PASS, scoped | [189, 54, 54] | 0.0038895905 / 0.0010215329 |
| gemma4 / gemma4_rgb8 | PASS, scoped | [28, 8, 8] | 0.00140762329 / 0.000314929731 |
| seq2seq / t5/unmasked | PASS, scoped | None | 5.10513783e-05 / 6.48748804e-06 |
| seq2seq / t5/padded | PASS, scoped | None | 6.85341656e-05 / 1.00578595e-05 |
| seq2seq / bart/unmasked | PASS, scoped | None | 2.39014626e-05 / 6.78559587e-06 |
| seq2seq / bart/padded | PASS, scoped | None | 2.77161598e-05 / 8.63157629e-06 |
| seq2seq / switch_transformers/unmasked | PASS, scoped | None | 0.00148224831 / 0.00024096831 |
| seq2seq / switch_transformers/padded | PASS, scoped | None | 2.45571136e-05 / 7.99244704e-06 |

Gemma4 Q/K execute cooperative WMMA; V/O/patch/MLP follow the measured FP32 policy (44 cooperative dispatches, 114 explicit policy fallbacks). The unchanged Mixed limits are max <0.02 and RMS <0.003. Shared CLIP/LLaVA transactions and standalone GPT2/Llama full/LoRA have tested GPU loss scaling. T5/BART/Switch, Gemma3, Gemma4 and native V4.1 retain their explicitly bounded unscaled FP32-gradient policy; FP16 GEMM operand quantization still occurs. Complete CUDA autocast/GradScaler equivalence is not advertised for those paths.

## Full FP16 and GUI

Full FP16 uses the genuine FP16 cooperative accumulator for eligible reductions <=64, with FP32 graph suboperations, masters and moments. The qualified local Llama64/shared CLIP64 profile and adapter/driver admission remain enforced. A real K=128 GPU test executes forward/dW/dX FP32 fallback, AdamW and parameter/moment checkpoint restoration with cooperative delta [0,0]. An internal two-update 128-wide image/text test records 144 reduction fallbacks and exact HF export/reload; that test does not admit 128-wide Full FP16 through the frontend. All original primitive failures remain visible.

Actual GUI Start executed eight workflows: Mixed and Full FP16, each vision, full training, full-parameter fine-tune and LoRA. Every workflow completed two optimizer updates with the requested accumulator route. Four real rejection controls cover invalid mode, unqualified Full profile, checkpoint precision mismatch and partial accumulation-window checkpoint. Full-model optimizer resume, image accumulation, dropout accumulation and partial-window resume remain unsupported. These were real Start code-path tests; manual visual clickthrough was not run.

## Current-source qualification and regression

| Gate | Status | Scope |
|---|---|---|
| phase19w | PASS | 146 canonical + 83 aliases; 166 forward, 72 generation, 32 PEFT runs; source `1a54d15d117e7ec6ea187872e891132149a64923` |
| phase19x | PASS | Six newly executed families: attention, training, stability, held-out, drift and activation; source `7d921a2305ec75dc724ae8170b5f757433c21cb5` |
| phase19y | PASS | 16/16 expanded shapes + 74/74 retained shared cases; all 93 entries accounted, with 19 historical entries distinguished; source `7d921a2305ec75dc724ae8170b5f757433c21cb5` |
| phase19z | PASS | All current families, oracle verification, restored shader, source/binary bindings and all thirteen packed pairs; source `7d921a2305ec75dc724ae8170b5f757433c21cb5` |
| Packed-cell | PASS | 13/13 independent current GPU benchmark/guard pairs; full batches 1/4/8/16/32, 1,200 balanced observations, original warmups, eight-step chains and 1%/2% bands |
| Shader provenance | PASS | 727/727 identical training variants; 4/4 WMMA; 45-artifact gate is an overlapping subset, not an extra 45 variants |
| library | PASS | 1159 passed, 0 failed, 40 ignored; ignored tests excluded from passes |
| cli | PASS | 50 passed, 0 failed, 0 ignored; ignored tests excluded from passes |
| gui | PASS | 17 passed, 0 failed, 4 ignored; ignored tests excluded from passes |
| gui_hardware | PASS | 2 passed, 0 failed, 0 ignored; ignored tests excluded from passes |
| oversized_fallback | PASS | 1 passed, 0 failed, 0 ignored; ignored tests excluded from passes |
| Rust packed benchmark controls | PASS | 3 passed, 0 failed, 0 ignored; exact executable hash bound separately |

Training X preserved eight arms, three resume arms and eight performance arms. Stability retained all five original learning rates and 100-step trajectories, selecting 0.001. Held-out/drift used orders 0/37/512: validation gaps were +0.0956068, -0.000362141 and +0.116005; no three-order confidence interval or universal Mixed accuracy equality is claimed. Same-weight maximum logit drift was 0.0551729 in this separate large-model investigation. Activation executed all six cells and ten traces; first divergence was Q projection. Epoch-reuse validation checks both the preceding evaluation and its parameter digest.

Fixture provenance is separated by workload: the maintained 93-image/166-text frozen reference is AVX2; Y expanded shapes use the immutable AVX512/Torch 2.6.0/HF 129e8dc oracle, while the retained shared manifest records AVX2. W forward/generation use Torch 2.14.1/HF 4692303 with AVX2; PEFT uses Torch 2.6.0/HF 129e8dc with AVX512. The exact manifests and environments, rather than a common CPU label, define each reference.

The five Z training families are byte-identical copies of the actually executed current X evidence, independently re-verified by both consumers; their complete copy provenance is retained. CPU fixtures were not regenerated or relabeled: 2,748 frozen files verify against the immutable handoff. The 16-shape and 77-case oracle ZIPs retain their published SHA-256 values. Original failed packed sweeps, before/after policy tables and official calibration logs are archived under `../20261113-phase22-recovery/`. Independent post-recording replay is required; calibration alone is never counted as PASS.

The user reported an active Discord call during the late packed replay. Its exact start time and hardware impact are unknown; the context note is retained in raw diagnostics. Thresholds and calibration were left unchanged, and every actual guard still had to pass.

Forward/rematerialization uses three construction rotations with 400 observations each; backward fusion uses four rotations with 300 each. Each uses the original per-process warmup, pooling 1,200 raw samples per leg. Both allocation placement and execution order are balanced. All raw timing arrays and exact-parity readbacks are retained; 23 allocation fault controls reject missing orders, forged medians, invalid samples and hidden numeric failures. The target production route remains unpinned while other chains are held fixed as in the forced arms.

The retained Phase XX report and summary remain byte-for-byte in `../20261113-phase22-recovery/phase20-handoff/`. [RDNA3_RELEASE_CONTRACT.md](../../../../../RDNA3_RELEASE_CONTRACT.md) describes numerical exclusions; raw component JSON, logs and fixture/toolchain metadata are under `evidence/unified/`. The final delivery report must list the actual hosted CI URLs at the published commit; local qualification does not imply hosted or self-hosted CI success.

## Executable SHA-256 inventory

| Executable | SHA-256 |
|---|---|
| activation_probe | `817a507693d5891f99b2b8bb663debaa9f80be97375a76d44ae3438d567e97d4` |
| classifier_test | `1dd73bb97fbfefc6267feee91bee9189f0e5d5b9398d61ee67dac03cb7c5b83c` |
| cli_test | `d783ab3512cae91b5496e27e34ebf6d4d764123ec5aa01abe2f3383e12ac9e6f` |
| deepseek_v41_vision_precision_probe | `d65a221aba9f38766a42d3ed1346710ca80986d705db3344ee0d7c370aa54b8a` |
| gemma3_vision_probe | `b59ba2b75dd592f8d0a79feb330c1245280767d4a31f99885787ca10969dc859` |
| gemma4_vision_probe | `14c1638fea5e215c9c6f19952061c537847e8a0793cf3ae4ad09082f06c4cee8` |
| gui_app | `980199a3ad11339cc5fc4de44bfa43891e7d5335af2e81008ae0e14e747419d3` |
| gui_test | `af11bcbdd95390ca40f3a72cef748bf1f62420360cef12bcc473dbdd09d29cce` |
| held_out_eval | `e6ec4152e678a245ab4ec3d36224daff7c8a538c1d65070daadf58089f5dd39a` |
| hierarchos-native-cli | `ef9e4f364eecf103aef8a31bd29e4eeb06014cefe109bf790a3233a25c192e06` |
| hierarchos-vulkan-rwkv-key-state-forward-bench | `0d3db02f32125f1b237f2dfdb9ec1912b3895fb5cd7a9c2438a46e938c6742d8` |
| hierarchos-vulkan-rwkv-low-rank-fp16-forward-bench | `ef78065b037ebb4d95951c09c6c7308da6703684be56909b211f314a61b81bf1` |
| hierarchos-vulkan-rwkv-packed-cell-channel-mix-bench | `0c8151c9d182a7570650073827aa346dd19f4206aa1562c4f26f4fedda212a77` |
| hierarchos-vulkan-rwkv-packed-cell-low-rank-forward-bench | `818ecd8cc01048592b2fab6bf20884c6f636a1073141098b9e981a105ae00a81` |
| hierarchos-vulkan-rwkv-packed-cell-step | `0e7278b47a2e41e74f0e8cf400eb90f7aaae5c99f2c862b8a86b1b238f0b076a` |
| hierarchos-vulkan-rwkv-packed-cell-time-mix-bench | `3e8f3cffa4106671f5cc4fd1e2929a519b5fcb79bb4d00d5ab67693dec357531` |
| hierarchos-vulkan-rwkv-tbptt-packed-chains-bench | `2a74b7bc561bb90b03311546a84ca995e528a94d7ab2d8f2640d564eacf40217` |
| hierarchos-vulkan-rwkv-tbptt-time-mix-bench | `c5278403b9862ec187fd04d4a5022f13b4a2699756a1201c5e2573d5f846fd9a` |
| hierarchos-vulkan-training-qualification | `52a7c4566d6672d2c924f38c5794cfcfcda8ad610873b9cef081fce287f02c3c` |
| hierarchos-vulkan-transformer-logits | `f284c880a63e4a715f3bf98ba159b5e54297f7258fa511444108ea2d7a647bf3` |
| hierarchos-vulkan-transformer-train | `5778552a7b3c1e3ae8a56ce00c2ce5c54345e1e5644602f430d7ea3d6b757b3f` |
| image_text_pipeline_probe | `a9ec1cb9edce9285280702cffcb5dad84de7b7f35e0c46e55f94441551bd2926` |
| libtest | `092452e3c555dc46b9abc2ee938bd552b832e5eedc9cb4580131ffc46f539e5b` |
| llava_pipeline_probe | `84c97a7d2f77564a7cf413d64e9dc414120e1b4b4117562d0a583e4de6e98b00` |
| parameter_streaming_bench | `6996e54c28293847eb79aad5b75ebef8223245f9d339be8c443d34cf96241177` |
| peft_bank_parity | `f7602c3e587a902a8ded80c7001b6cec424e2249e60d38a012c71feba0119e31` |
| peft_checkpoint | `27eee613816aaacd4c651c0eff6afc2b6604b7cdd12d3423cb1658bc644c131b` |
| rwkv_tbptt_packed_forward_parity | `fb6be03ad08bd5de9e7fd4cab8fe9ea7f87312a101c11ce0979d132ed087dc84` |
| training-stability | `7554ddfa1ac0d1c5b7b853f90998690a54f8355a56c0b8ddf2dc410a1f1e7e23` |
| transformer_generation_parity | `bb349b6d3daded68fe106347ca2be9a60dba2a307f2b0314ed1f5b1223b779ce` |
| transformer_parity | `f816535778742b85cea1b007bd57cf73ff5aee7521835b027c5f3272ef361dcd` |
| wmma_drift | `a17213553b3d0b99743f2e09c8ad002f490b5fe10fed2a1a88b0256288d3095b` |
| wmma_qualify | `5a873979d43e694185a53ebb047c773d6bb12831e01145f4d625ebd0907fe3f7` |
