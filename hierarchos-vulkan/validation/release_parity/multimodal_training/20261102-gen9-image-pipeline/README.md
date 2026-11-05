# Gen9 native CLIP/LLaVA image pipeline

Base revision: `d41ebeee295d598840438ae03757419d624985ac`, including the RDNA3
GPT-2 fix and updated project README.

The scoped image logit/update gate passes four cases: 22,784 visible logits,
worst `7.450580596923828e-8`, strict `<2e-7`, `rtol=0`. Two complete optimizer
steps train the CLIP tower, projector and Llama backbone. All 130,688 parameter
comparisons pass; native export/reload matches all 5,952 logit words.

* `cpu-fixtures.zip`: independent raw RGB inputs, actual HF CLIP/LLaVA packages,
  frozen CPU logits/gradients/parameters and imported oracle sources.
* `optimizer-cpu-fixtures.zip`: identical-gradient AdamW references, three beta
  configurations, vector blocks/scalar tails, three steps.
* `pipeline-qualification.json`: required logit/update checks and separately
  reported strict intermediate diagnostics. Intermediate bit parity is not
  certified by the logit gate.
* `bridge-qualification.json`: the broader feature bridge remains red at
  63/77 cases, 31/40 profiles fully passing. Four wide cases fail logits;
  ten other cases fail the richer raw-adjoint gate.
* `text-regression.json`: the existing 166-run text forward matrix.
* `evidence.zip`: native outputs/exports, pre-repair failures, tests and the
  source snapshot. `qualification.json` binds artifact/source/binary hashes.

The native image gate covers 8x8 fixed-resolution RGB, two CLIP layers at
width 16/32 and a two-layer width-16 Llama backbone. Padded query rows follow the
existing observable-row CPU contract. Resize/crop, arbitrary production-size
capacity, audio/video, other towers, PEFT and optimizer resume remain pending.

```powershell
python -m zipfile -e hierarchos-vulkan/validation/release_parity/multimodal_training/20261102-gen9-image-pipeline/cpu-fixtures.zip target/llava-repeat/cpu
cargo build --locked --manifest-path hierarchos-vulkan/Cargo.toml -j1 --bin llava_pipeline_probe --bin cpu_adamw_probe
python hierarchos-vulkan/validation/qualify_llava_pipeline.py run --fixtures target/llava-repeat/cpu --binary hierarchos-vulkan/target/debug/llava_pipeline_probe.exe --output target/llava-repeat/gen9 --target gen9
```

RDNA3 replay uses a new output directory, `--target rdna3 --device-id 0x...`
with the physical adapter's PCI device ID. The runner pins the original ATen
vector width from the frozen manifest. Native replay needs standard-library
Python; no Torch, Transformers or Pillow installation is required.
