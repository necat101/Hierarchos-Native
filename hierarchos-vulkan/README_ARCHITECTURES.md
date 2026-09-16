# Hierarchos Vulkan supported Transformer architectures

This file is generated from `src/transformer.rs`. Do not hand-maintain the
architecture counts or lists; regenerate it with:

```powershell
python hierarchos-vulkan/validation/generate_supported_architectures.py
```

The current native registry contains **143 canonical Transformer
architectures** plus **81 Hugging Face package/config aliases**
(224 advertised `model_type` spellings total).

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

## Canonical native architectures (143)

| model_type 1 | model_type 2 | model_type 3 | model_type 4 |
| --- | --- | --- | --- |
| `afmoe` | `ernie4_5_moe` | `jina_embeddings_v3` | `pegasus_x` |
| `albert` | `esm` | `laguna` | `persimmon` |
| `apertus` | `esmc` | `llama` | `phi` |
| `arcee` | `eurobert` | `llama4_text` | `phi3` |
| `aria_text` | `exaone4` | `longcat_flash` | `phimoe` |
| `axk1` | `exaone_moe` | `m2m_100` | `plbart` |
| `axk2` | `falcon` | `marian` | `qwen2` |
| `bart` | `flaubert` | `mbart` | `qwen2_moe` |
| `bert` | `flex_olmo` | `megatron-bert` | `qwen3` |
| `bert-generation` | `fsmt` | `mellum` | `qwen3_5_moe_text` |
| `big_bird` | `gemma` | `mimo_v2_flash` | `qwen3_5_text` |
| `bigbird_pegasus` | `gemma2` | `minicpm3` | `qwen3_moe` |
| `biogpt` | `gemma3_text` | `minimax_m2` | `qwen3_next` |
| `bitnet` | `glm` | `minimax_m3_vl_text` | `qwen4_exp_text` |
| `blenderbot` | `glm4` | `ministral` | `rembert` |
| `blenderbot-small` | `glm4_moe` | `ministral3` | `roberta` |
| `bloom` | `glm5_next_text` | `mistral` | `roberta-prelayernorm` |
| `camembert` | `gpt-sw3` | `mistral4` | `roc_bert` |
| `codegen` | `gpt2` | `mixtral` | `roformer` |
| `cohere` | `gpt_bigcode` | `modernbert` | `seed_oss` |
| `cohere2` | `gpt_neo` | `modernbert-decoder` | `smollm3` |
| `cohere2_moe` | `gpt_neox` | `mpt` | `solar_open` |
| `cohere_compass_text` | `gpt_neox_japanese` | `mt5` | `stablelm` |
| `ctrl` | `gpt_oss` | `mvp` | `starcoder2` |
| `cwm` | `gptj` | `nanochat` | `switch_transformers` |
| `data2vec-text` | `granite` | `nemotron` | `t5` |
| `dbrx` | `granite_swa` | `nomic_bert` | `t5_gemma_module` |
| `deepseek_v2` | `granitemoe` | `olmo` | `trocr` |
| `deepseek_v3` | `granitemoe_swa` | `olmo2` | `umt5` |
| `deepseek_v4` | `granitemoeshared` | `olmo3` | `vaultgemma` |
| `distilbert` | `helium` | `olmo_hybrid` | `xglm` |
| `dots1` | `hunyuan_v1_dense` | `olmoe` | `xlm` |
| `electra` | `hunyuan_v1_moe` | `open-llama` | `xlm-roberta` |
| `emu3_text_model` | `hy_v3` | `openai-gpt` | `xlm-roberta-xl` |
| `ernie` | `hyperclovax` | `opt` | `youtu` |
| `ernie4_5` | `jais2` | `pegasus` |  |

## Hugging Face package/config aliases (81)

These aliases resolve to a canonical native text graph while preserving
unconsumed package tensors on export where supported.

| model_type 1 | model_type 2 | model_type 3 | model_type 4 |
| --- | --- | --- | --- |
| `aria` | `glm4v_moe_text` | `llava` | `qwen3_5` |
| `audioflamingo3` | `glm4v_text` | `llava_next` | `qwen3_5_moe` |
| `aya_vision` | `glm5_next` | `llava_next_video` | `qwen3_asr` |
| `code_llama` | `glm_moe_dsa` | `llava_onevision` | `qwen3_omni_moe` |
| `cohere2_vision` | `glm_ocr` | `minimax_m3_vl` | `qwen3_omni_moe_text` |
| `cohere_compass` | `glm_ocr_text` | `mistral3` | `qwen3_omni_moe_thinker` |
| `cosmos3_omni` | `glmasr` | `mllama` | `qwen3_vl` |
| `deepseek_v32` | `got_ocr2` | `mllama_text_model` | `qwen3_vl_moe` |
| `deepseek_vl` | `granite4_vision` | `modernvbert` | `qwen3_vl_moe_text` |
| `deepseek_vl_hybrid` | `granite4_vision_text` | `musicflamingo` | `qwen3_vl_text` |
| `emu3` | `granite_speech` | `ovis2` | `qwen4_exp` |
| `encoder-decoder` | `granite_speech_plus` | `paddleocr_vl` | `smolvlm` |
| `exaone4_5` | `hunyuan_vl` | `paddleocr_vl_text` | `t5gemma` |
| `exaone4_5_text` | `hunyuan_vl_text` | `paligemma` | `vibevoice_asr` |
| `fast_vlm` | `idefics2` | `phi4_multimodal` | `video_llama_3` |
| `fuyu` | `idefics3` | `qwen2_5_omni_text` | `video_llava` |
| `gemma3` | `internvl` | `qwen2_5_vl` | `vipllava` |
| `glm46v` | `janus` | `qwen2_5_vl_text` | `voxtral` |
| `glm4_moe_lite` | `kimi_k2` | `qwen2_audio` |  |
| `glm4v` | `kimi_k25` | `qwen2_vl` |  |
| `glm4v_moe` | `llama4` | `qwen2_vl_text` |  |

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
