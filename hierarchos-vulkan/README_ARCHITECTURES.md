# Hierarchos Vulkan supported Transformer architectures

This file is generated from `src/transformer.rs`. Do not hand-maintain the
architecture counts or lists; regenerate it with:

```powershell
python hierarchos-vulkan/validation/generate_supported_architectures.py
```

The current native registry contains **145 canonical Transformer
architectures** plus **83 Hugging Face package/config aliases**
(228 advertised `model_type` spellings total).

## What "end-to-end" means here

The canonical entries below are the native Vulkan Transformer's declared text
graph contracts. They are routed through architecture-specific Hugging Face
configuration parsing, SafeTensors loading, Vulkan forward/backward/training,
generation where the architecture is generative, and native package export.
Unsupported `model_type` values fail closed; the backend does not silently map
an unknown architecture onto a merely similar graph.

This is deliberately narrower than "all Transformers tasks." A text graph being
supported does not imply that every upstream task head, vision tower, audio
tower, processor, custom remote-code model, or multimodal projector is native.
The aliases section is especially important: those names resolve a supported
native **text backbone** from a composite Hugging Face package. They are not a
claim that the package's non-text modalities execute end to end in Vulkan.

For registry coverage against the checked-out Python `transformers` source, run
`python hierarchos-vulkan/validation/audit_transformers_coverage.py`. That audit
is registry overlap, not a numerical-parity certificate. Parity and training
validation live under `hierarchos-vulkan/validation/`.

## User-facing native routes

| Workflow | CLI | Desktop GUI |
| --- | --- | --- |
| Transformer full training | `transformer-train` | Transformer full training |
| Transformer LoRA fine-tuning | `transformer-finetune` | Transformer LoRA fine-tuning |
| Transformer inference/generation | `infer` / `generate` / `transformer-generate` | Transformer inference / generation |
| Hierarchos training | `train` / `finetune` | Hierarchos training / fine-tuning |
| Hierarchos inference | `chat` | Hierarchos inference / chat |

The GUI is a thin native launcher over the same CLI contracts; it does not carry
a second model implementation. Transformer training and generation execute in
the Vulkan backend. Hierarchos training executes in Vulkan and Hierarchos
inference uses the companion pure-Rust inference engine.

## Canonical native architectures (145)

| model_type 1 | model_type 2 | model_type 3 | model_type 4 |
| --- | --- | --- | --- |
| `afmoe` | `esm` | `kimi_linear` | `persimmon` |
| `albert` | `esmc` | `laguna` | `phi` |
| `apertus` | `eurobert` | `llama` | `phi3` |
| `arcee` | `exaone4` | `llama4_text` | `phimoe` |
| `aria_text` | `exaone_moe` | `longcat_flash` | `plbart` |
| `axk1` | `falcon` | `m2m_100` | `qwen2` |
| `axk2` | `flaubert` | `marian` | `qwen2_moe` |
| `bart` | `flex_olmo` | `mbart` | `qwen3` |
| `bert` | `fsmt` | `megatron-bert` | `qwen3_5_moe_text` |
| `bert-generation` | `gemma` | `mellum` | `qwen3_5_text` |
| `big_bird` | `gemma2` | `mimo_v2_flash` | `qwen3_moe` |
| `bigbird_pegasus` | `gemma3_text` | `minicpm3` | `qwen3_next` |
| `biogpt` | `gemma4_text` | `minimax_m2` | `qwen4_exp_text` |
| `bitnet` | `glm` | `minimax_m3_vl_text` | `rembert` |
| `blenderbot` | `glm4` | `ministral` | `roberta` |
| `blenderbot-small` | `glm4_moe` | `ministral3` | `roberta-prelayernorm` |
| `bloom` | `glm5_next_text` | `mistral` | `roc_bert` |
| `camembert` | `gpt-sw3` | `mistral4` | `roformer` |
| `codegen` | `gpt2` | `mixtral` | `seed_oss` |
| `cohere` | `gpt_bigcode` | `modernbert` | `smollm3` |
| `cohere2` | `gpt_neo` | `modernbert-decoder` | `solar_open` |
| `cohere2_moe` | `gpt_neox` | `mpt` | `stablelm` |
| `cohere_compass_text` | `gpt_neox_japanese` | `mt5` | `starcoder2` |
| `ctrl` | `gpt_oss` | `mvp` | `switch_transformers` |
| `cwm` | `gptj` | `nanochat` | `t5` |
| `data2vec-text` | `granite` | `nemotron` | `t5_gemma_module` |
| `dbrx` | `granite_swa` | `nomic_bert` | `trocr` |
| `deepseek_v2` | `granitemoe` | `olmo` | `umt5` |
| `deepseek_v3` | `granitemoe_swa` | `olmo2` | `vaultgemma` |
| `deepseek_v4` | `granitemoeshared` | `olmo3` | `xglm` |
| `distilbert` | `helium` | `olmo_hybrid` | `xlm` |
| `dots1` | `hunyuan_v1_dense` | `olmoe` | `xlm-roberta` |
| `electra` | `hunyuan_v1_moe` | `open-llama` | `xlm-roberta-xl` |
| `emu3_text_model` | `hy_v3` | `openai-gpt` | `youtu` |
| `ernie` | `hyperclovax` | `opt` |  |
| `ernie4_5` | `jais2` | `pegasus` |  |
| `ernie4_5_moe` | `jina_embeddings_v3` | `pegasus_x` |  |

## Hugging Face package/config aliases (83)

These aliases resolve to a canonical native text graph while preserving
unconsumed package tensors on export where supported.

| model_type 1 | model_type 2 | model_type 3 | model_type 4 |
| --- | --- | --- | --- |
| `aria` | `glm4v_moe` | `kimi_k3` | `qwen2_vl` |
| `audioflamingo3` | `glm4v_moe_text` | `llama4` | `qwen2_vl_text` |
| `aya_vision` | `glm4v_text` | `llava` | `qwen3_5` |
| `code_llama` | `glm5_next` | `llava_next` | `qwen3_5_moe` |
| `cohere2_vision` | `glm_moe_dsa` | `llava_next_video` | `qwen3_asr` |
| `cohere_compass` | `glm_ocr` | `llava_onevision` | `qwen3_omni_moe` |
| `cosmos3_omni` | `glm_ocr_text` | `minimax_m3_vl` | `qwen3_omni_moe_text` |
| `deepseek_v32` | `glmasr` | `mistral3` | `qwen3_omni_moe_thinker` |
| `deepseek_vl` | `got_ocr2` | `mllama` | `qwen3_vl` |
| `deepseek_vl_hybrid` | `granite4_vision` | `mllama_text_model` | `qwen3_vl_moe` |
| `emu3` | `granite4_vision_text` | `modernvbert` | `qwen3_vl_moe_text` |
| `encoder-decoder` | `granite_speech` | `musicflamingo` | `qwen3_vl_text` |
| `exaone4_5` | `granite_speech_plus` | `ovis2` | `qwen4_exp` |
| `exaone4_5_text` | `hunyuan_vl` | `paddleocr_vl` | `smolvlm` |
| `fast_vlm` | `hunyuan_vl_text` | `paddleocr_vl_text` | `t5gemma` |
| `fuyu` | `idefics2` | `paligemma` | `vibevoice_asr` |
| `gemma3` | `idefics3` | `phi4_multimodal` | `video_llama_3` |
| `gemma4` | `internvl` | `qwen2_5_omni_text` | `video_llava` |
| `glm46v` | `janus` | `qwen2_5_vl` | `vipllava` |
| `glm4_moe_lite` | `kimi_k2` | `qwen2_5_vl_text` | `voxtral` |
| `glm4v` | `kimi_k25` | `qwen2_audio` |  |

## Shipping checks

Before publishing a standalone bundle, run at least:

```powershell
cargo check --manifest-path hierarchos-vulkan/Cargo.toml --all-targets
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib
cargo test --manifest-path hierarchos-native-cli/Cargo.toml
cargo test --manifest-path hierarchos-gui/Cargo.toml
python -m unittest discover -s hierarchos-vulkan/validation -p "test_*.py"
python hierarchos-vulkan/validation/generate_supported_architectures.py --check
hierarchos-native-cli architectures --json
hierarchos-native-cli doctor
```

`doctor` verifies Vulkan device visibility and reports whether the companion
Hierarchos trainer/device-probe binaries can be found. Transformer training and
generation are implemented directly in `hierarchos-native-cli`; coherent-v9
Hierarchos training uses the companion `hierarchos-vulkan-train` binary.
