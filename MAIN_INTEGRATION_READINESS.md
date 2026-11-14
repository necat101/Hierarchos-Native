# WMMA admission and legacy FP32 compatibility

The integration contract is:

- WMMA is opt-in and restricted to recognised AMD RDNA3, RDNA3.5 and RDNA4 PCI IDs, with all required Vulkan features and tiles. Unknown devices retain FP32.
- RDNA1/2 retain the pre-WMMA AMD FP32 selector. Intel Gen9 retains the pre-WMMA FP32 selector, shader bytes and composition-admission rule.
- Default `off` neither probes nor enables WMMA feature chains. A vendor-name override cannot bypass physical-device admission.
- The RDNA3 key-state arithmetic fix is isolated in `rwkv_key_state_forward_fused_wide_amd_rdna3.spv`. The shared shader is byte-identical to pre-WMMA main `ead6cd3fa5f8943e478dcd0aec27e6014f62d1d5`.

`check_main_compatibility.py` verifies all 1,301 original shader files, the original FP32 vendor selector, the preserved qualified RDNA3 variant and the actual Rust device-policy tests. This guard runs in workspace CI. The table of newer AMD devices follows [AMD GPUOpen device_info](https://github.com/GPUOpen-Tools/device_info/blob/dff4582fe929ae85e6224a5b4a8761eec4872d18/device_info.cpp); Navi 33 `749f` is also recorded by the [PCI ID Project](https://pci-ids.ucw.cz/read/PC/1002/749f).

The earlier qualified release at `92cfa703acb3345b5af5e607ed97f697704b064d` remains preserved in Git. Every prior release-file digest is recorded in `hierarchos-vulkan/validation/release_parity/multimodal_training/20261114-main-compatibility/before-92c-preservation.json`. Runtime changes require fresh current-source qualification; the old receipt is not a claim about the admission patch. Consult the current release summary and the final published commit's CI before merging.

## Gen9 host qualification

Live Gen9 qualification is assigned to the later Gen9-host pull, per the user. These CPU controls do not claim that execution. On that host:

1. Run `python hierarchos-vulkan/validation/check_main_compatibility.py`.
2. Build the normal release and run the library suite with `HIERARCHOS_WMMA_MODE=off` and the normal host ISA detection.
3. Run `hierarchos-vulkan-wmma-probe --wmma-mode off --json`: expect FP32, `rdna3_or_newer=false`, `probed=false`, `enabled=false` and no cooperative dispatches.
4. Request `mixed` and `full-fp16` explicitly: both must fail with the AMD RDNA3-or-newer admission diagnostic.
5. Run the existing Gen9 strict logits, image/text, generation, PEFT/resume and RWKV regression matrices against the original frozen references. Retain the original tolerances and compare the FP32 route/performance with the pre-WMMA baseline.

AVX2 fixture reproduction remains a separate pending qualification on the AVX2 host. Full FP16 retains its existing qualified model/adapter/driver scope even on admitted RDNA3+ hardware.
