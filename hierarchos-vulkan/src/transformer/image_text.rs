//! Trainable shared CLIP image adapter composed with an ordinary text package.
//! This composition has its own package identity; it does not impersonate an
//! upstream multimodal architecture or change the text checkpoint configuration.
use super::vision::{LlavaForward, LlavaTrainResult, VulkanLlava};
use super::*;

#[cfg(test)]
#[path = "shared_cpu_training_tests.rs"]
mod cpu_training_tests;

/// A package contains `config.json` with model_type=hierarchos_image_text and
/// schema=1, an unchanged text HF package in `text/`, and a trainable
/// hierarchos_clip_adapter package in `image/`. Images are fixed-resolution NCHW.
pub struct VulkanImageText {
    graph: VulkanLlava,
    decoder: Option<Box<VulkanTransformer>>,
}

impl VulkanImageText {
    pub fn from_hf_package(
        device: VulkanDevice,
        directory: impl AsRef<Path>,
        batch_size: usize,
        seq_len: usize,
        image_count: usize,
    ) -> Result<Self> {
        let directory = directory.as_ref();
        let config: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.join("config.json"))?)?;
        if config["model_type"].as_str() != Some("hierarchos_image_text")
            || config["schema"].as_u64() != Some(1)
        {
            bail!("shared image-text package requires model_type=hierarchos_image_text, schema=1");
        }
        let joint = match config["text_mode"].as_str() {
            None | Some("causal") => false,
            Some("encoder_decoder") => true,
            Some(mode) => bail!("unsupported shared image-text mode {mode}"),
        };
        let mut text = if joint {
            VulkanTransformer::from_hf_dense_text_encoder_package(
                device.clone(),
                &directory.join("text"),
                batch_size,
                seq_len,
            )?
        } else {
            VulkanTransformer::from_hf_package(
                device.clone(),
                directory.join("text"),
                batch_size,
                seq_len,
            )?
        };
        let decoder = if joint {
            if !matches!(
                text.config.architecture,
                VulkanTransformerArchitecture::Bart
                    | VulkanTransformerArchitecture::T5
                    | VulkanTransformerArchitecture::SwitchTransformers
            ) {
                bail!("joint image training currently requires BART, T5 or Switch");
            }
            let mut decoder = Box::new(VulkanTransformer::from_hf_package_with_encoder_seq_len(
                device.clone(),
                directory.join("text"),
                batch_size,
                seq_len,
                seq_len,
            )?);
            decoder.share_encoder_embedding(&text)?;
            if aten_vector_width() == 8 {
                decoder.kernels.bias_grad = vulkan::ComputeKernel::new(
                    &device,
                    include_bytes!("../../shaders/vision_bias_grad_cpu_avx2.spv"),
                    2,
                    8,
                )?;
            }
            Some(decoder)
        } else {
            None
        };
        // CPU arithmetic specializations are admitted only for the independently
        // measured AVX2 geometries. Other packages retain the general kernels.
        let measured_rows = batch_size == 2 && seq_len == 8;
        let wide = measured_rows
            && text.config.hidden_size == 64
            && text.config.num_heads == 4
            && matches!(text.config.num_key_value_heads, 2 | 4)
            && matches!(
                text.config.architecture,
                VulkanTransformerArchitecture::Gpt2 | VulkanTransformerArchitecture::Llama
            );
        let small_gemma4 = measured_rows
            && text.config.architecture == VulkanTransformerArchitecture::Gemma4
            && text.config.hidden_size == 16
            && text.config.num_heads == 2
            && matches!(text.config.num_key_value_heads, 1 | 2);
        if aten_vector_width() == 8 && (wide || small_gemma4) {
            let input_gradient_spv: &[u8] = if wide {
                include_bytes!("../../shaders/vision_linear_input_grad_wide_cpu_avx2.spv")
            } else {
                include_bytes!("../../shaders/vision_linear_input_grad_cpu_avx2.spv")
            };
            text.kernels.cpu_row_gemm = Some((
                vulkan::ComputeKernel::new(
                    &device,
                    include_bytes!("../../shaders/vision_linear_cpu_avx2.spv"),
                    4,
                    16,
                )?,
                vulkan::ComputeKernel::new(&device, input_gradient_spv, 4, 16)?,
            ));
            text.kernels.cpu_wide_row_gemm = wide;
            let weight_gradient_spv: &[u8] = if wide {
                include_bytes!("../../shaders/vision_linear_weight_grad_wide_cpu_avx2.spv")
            } else {
                include_bytes!("../../shaders/vision_linear_weight_grad_cpu_avx2.spv")
            };
            text.kernels.linear_weight_grad =
                vulkan::ComputeKernel::new(&device, weight_gradient_spv, 3, 12)?;
            if text.config.architecture == VulkanTransformerArchitecture::Gpt2 {
                for layer in &mut text.layers {
                    layer.ln1.cpu_welford = true;
                    layer.ln2.cpu_welford = true;
                }
                text.final_norm.cpu_welford = true;
                text.kernels.layer_norm_input_grad = vulkan::ComputeKernel::new(
                    &device,
                    include_bytes!("../../shaders/vision_layer_norm_input_grad_cpu_avx2.spv"),
                    6,
                    8,
                )?;
                text.kernels.layer_norm_param_grad = vulkan::ComputeKernel::new(
                    &device,
                    include_bytes!("../../shaders/vision_layer_norm_param_grad_cpu_avx2.spv"),
                    6,
                    8,
                )?;
                text.kernels.gelu_tanh_forward = vulkan::ComputeKernel::new(
                    &device,
                    include_bytes!("../../shaders/transformer_gelu_tanh_ha_forward.spv"),
                    2,
                    4,
                )?;
                text.kernels.gelu_tanh_backward = vulkan::ComputeKernel::new(
                    &device,
                    include_bytes!("../../shaders/image_text_new_gelu_cpu_avx2_backward.spv"),
                    3,
                    4,
                )?;
            }
            if text.config.architecture == VulkanTransformerArchitecture::Llama {
                text.kernels.rope_forward = vulkan::ComputeKernel::new(
                    &device,
                    include_bytes!("../../shaders/image_text_rope_cpu_avx2_forward.spv"),
                    5,
                    std::mem::size_of::<RopePush>() as u32,
                )?;
                text.kernels.rope_backward = vulkan::ComputeKernel::new(
                    &device,
                    include_bytes!("../../shaders/image_text_rope_cpu_avx2_backward.spv"),
                    5,
                    std::mem::size_of::<RopePush>() as u32,
                )?;
                text.kernels.rms_norm_forward = vulkan::ComputeKernel::new(
                    &device,
                    include_bytes!("../../shaders/image_text_rms_wide_cpu_avx2.spv"),
                    5,
                    std::mem::size_of::<RmsNormForwardPush>() as u32,
                )?;
                text.kernels.rms_norm_input_grad = vulkan::ComputeKernel::new(
                    &device,
                    include_bytes!("../../shaders/image_text_rms_wide_cpu_avx2_backward.spv"),
                    6,
                    std::mem::size_of::<RmsNormBackwardPush>() as u32,
                )?;
                text.kernels.rms_norm_input_grad_residual = vulkan::ComputeKernel::new(
                    &device,
                    include_bytes!("../../shaders/image_text_rms_wide_cpu_avx2_residual.spv"),
                    8,
                    std::mem::size_of::<RmsNormBackwardPush>() as u32,
                )?;
                text.kernels.silu_backward = vulkan::ComputeKernel::new(
                    &device,
                    include_bytes!("../../shaders/silu_backward_cpu_avx2.spv"),
                    3,
                    4,
                )?;
            }
            if text.config.effective_attention_logit_softcap() == 0.0
                && text.config.attention_dropout == 0.0
                && text.config.query_hidden_size() == text.config.hidden_size
            {
                text.kernels.attention_forward = vulkan::ComputeKernel::new(
                    &device,
                    include_bytes!("../../shaders/image_text_sdpa_cpu_avx2.spv"),
                    11,
                    std::mem::size_of::<AttentionPush>() as u32,
                )?;
                text.kernels.attention_backward = vulkan::ComputeKernel::new(
                    &device,
                    include_bytes!("../../shaders/image_text_sdpa_cpu_avx2_backward.spv"),
                    15,
                    std::mem::size_of::<AttentionBackwardPush>() as u32,
                )?;
            }
        }
        let image = directory.join("image");
        let adapter: serde_json::Value =
            serde_json::from_slice(&fs::read(image.join("config.json"))?)?;
        if adapter["model_type"].as_str() != Some("hierarchos_clip_adapter") {
            bail!("shared image-text package requires an explicit hierarchos_clip_adapter");
        }
        Ok(Self {
            graph: VulkanLlava::from_language(device, &image, image_count, text)?,
            decoder,
        })
    }

    pub fn config(&self) -> &VulkanTransformerConfig {
        self.decoder
            .as_ref()
            .map(|decoder| decoder.config())
            .unwrap_or_else(|| self.graph.language.config())
    }

    /// Observe the actual composed text forward without substituting any math.
    /// Intended for CPU localization; ordinary inference does not read tapes.
    pub fn debug_text_stages(
        &self,
        ids: &[u32],
        mask: &[f32],
        pixels: &[f32],
    ) -> Result<serde_json::Value> {
        let forward = self.forward(ids, mask, pixels)?;
        let text = &self.graph.language;
        let config = &text.config;
        let rows = ids.len();
        let dim = config.hidden_size;
        let qdim = config.query_hidden_size();
        let kdim = config.key_value_hidden_size();
        let mid = config.intermediate_size;
        let mut stages = serde_json::Map::new();
        stages.insert(
            "lookup".into(),
            serde_json::json!(text
                .token_embedding_output
                .read_f32(rows * config.effective_embedding_size())?),
        );
        stages.insert(
            "input".into(),
            serde_json::json!(text.input_hidden.read_f32(rows * dim)?),
        );
        for (i, layer) in text.layers.iter().enumerate() {
            let t = &layer.tape;
            for (name, buffer) in [
                ("n1/mean", &t.ln1_mean),
                ("n1/rstd", &t.ln1_rstd),
                ("n3/mean", &t.ln2_mean),
                ("n3/rstd", &t.ln2_rstd),
            ] {
                stages.insert(
                    format!("layer_{i}/{name}"),
                    serde_json::json!(buffer.read_f32(rows)?),
                );
            }
            for (name, buffer, width) in [
                ("n1", &t.ln1, dim),
                ("q", &t.q, qdim),
                ("k", &t.k, kdim),
                ("v", &t.v, kdim),
                ("qn", &t.q_norm, qdim),
                ("kn", &t.k_norm, kdim),
                ("qr", &t.q_rotary, qdim),
                ("kr", &t.k_rotary, kdim),
                ("vn", &t.v_norm, kdim),
                ("a", &t.attention, qdim),
                ("o", &t.attention_projection, dim),
                ("n2", &t.post_attention_norm, dim),
                ("r", &t.residual1, dim),
                ("n3", &t.ln2, dim),
                ("g", &t.mlp_gate_pre, mid),
                ("u", &t.mlp_pre, mid),
                ("activation", &t.mlp_activation, mid),
                ("p", &t.mlp_product, mid),
                ("d", &t.mlp_output, dim),
                ("n4", &t.post_mlp_norm, dim),
                ("out", &t.output, dim),
            ] {
                stages.insert(
                    format!("layer_{i}/{name}"),
                    serde_json::json!(buffer.read_f32(rows * width)?),
                );
            }
            if config.architecture == VulkanTransformerArchitecture::Gpt2 {
                stages.insert(
                    format!("layer_{i}/qkv"),
                    serde_json::json!(t.qkv.read_f32(rows * (qdim + 2 * kdim))?),
                );
            }
        }
        stages.insert(
            "final_norm".into(),
            serde_json::json!(text.final_norm_output.read_f32(rows * dim)?),
        );
        Ok(serde_json::json!({"stages":stages,"logits":forward.logits}))
    }

    pub fn preprocess_rgb8(&self, rgb: &[u8]) -> Result<Vec<f32>> {
        self.graph.preprocess_rgb8(rgb)
    }

    /// Observe a real composed update and its retained native gradients.
    pub fn debug_train_stages(
        &mut self,
        ids: &[u32],
        mask: &[f32],
        pixels: &[f32],
        targets: &[u32],
        weights: &[f32],
        optimizer: crate::CpuAdamWHyperParams,
    ) -> Result<serde_json::Value> {
        let result =
            self.train_step_with_optimizer(ids, mask, pixels, targets, weights, optimizer)?;
        let text = &self.graph.language;
        let rows = ids.len();
        let c = &text.config;
        let mut stages = serde_json::Map::new();
        let mut parameters = serde_json::Map::new();
        fn parameter(
            out: &mut serde_json::Map<String, serde_json::Value>,
            name: String,
            p: &TrainableParameter,
            transpose: Option<(usize, usize)>,
        ) -> Result<()> {
            let values = p.grad_buffer()?.read_f32(p.len)?;
            let values = if let Some((input, output)) = transpose {
                (0..input)
                    .flat_map(|i| {
                        let v = &values;
                        (0..output).map(move |o| v[o * input + i])
                    })
                    .collect::<Vec<_>>()
            } else {
                values
            };
            out.insert(name, serde_json::json!(values));
            Ok(())
        }
        fn linear(
            out: &mut serde_json::Map<String, serde_json::Value>,
            prefix: String,
            l: &VulkanLinear,
            transpose: bool,
        ) -> Result<()> {
            parameter(
                out,
                format!("{prefix}.weight"),
                &l.weight,
                transpose.then_some((l.input_dim, l.output_dim)),
            )?;
            if l.has_bias {
                parameter(out, format!("{prefix}.bias"), &l.bias, None)?;
            }
            Ok(())
        }
        fn norm(
            out: &mut serde_json::Map<String, serde_json::Value>,
            prefix: String,
            n: &VulkanLayerNorm,
        ) -> Result<()> {
            parameter(out, format!("{prefix}.weight"), &n.weight, None)?;
            if n.has_bias {
                parameter(out, format!("{prefix}.bias"), &n.bias, None)?;
            }
            Ok(())
        }
        let gpt2 = c.architecture == VulkanTransformerArchitecture::Gpt2;
        if let Some(head) = text.lm_head.as_ref() {
            linear(&mut parameters, "lm_head".into(), head, false)?;
        }
        norm(
            &mut parameters,
            if gpt2 {
                "transformer.ln_f"
            } else {
                "model.norm"
            }
            .into(),
            &text.final_norm,
        )?;
        if gpt2 {
            if let Some(p) = text.position_embedding.as_ref() {
                parameter(&mut parameters, "transformer.wpe.weight".into(), p, None)?;
            }
        }
        parameters.insert(
            if gpt2 {
                "transformer.wte.weight"
            } else {
                "model.embed_tokens.weight"
            }
            .into(),
            serde_json::json!(text
                .shared_embedding
                .gradient_buffer()?
                .read_f32(c.vocab_size * c.effective_embedding_size())?),
        );
        for (name, buffer, width) in [
            ("final_norm", &text.grad_final_norm_output, c.hidden_size),
            ("logits", &text.grad_logits, c.vocab_size),
        ] {
            stages.insert(
                name.into(),
                serde_json::json!(buffer.read_f32(rows * width)?),
            );
        }
        for (i, layer) in text.layers.iter().enumerate() {
            let t = &layer.tape;
            let d = c.hidden_size;
            let q = c.query_hidden_size();
            let k = c.key_value_hidden_size();
            let m = c.intermediate_size;
            if matches!(
                c.architecture,
                VulkanTransformerArchitecture::Gpt2 | VulkanTransformerArchitecture::Llama
            ) {
                let prefix = if gpt2 {
                    format!("transformer.h.{i}")
                } else {
                    format!("model.layers.{i}")
                };
                norm(
                    &mut parameters,
                    format!("{prefix}.{}", if gpt2 { "ln_1" } else { "input_layernorm" }),
                    &layer.ln1,
                )?;
                norm(
                    &mut parameters,
                    format!(
                        "{prefix}.{}",
                        if gpt2 {
                            "ln_2"
                        } else {
                            "post_attention_layernorm"
                        }
                    ),
                    &layer.ln2,
                )?;
                for (name, l) in [
                    (
                        if gpt2 { "attn.c_attn" } else { "unused" },
                        layer.c_attn.as_ref(),
                    ),
                    ("self_attn.q_proj", layer.q_proj.as_ref()),
                    ("self_attn.k_proj", layer.k_proj.as_ref()),
                    ("self_attn.v_proj", layer.v_proj.as_ref()),
                    ("mlp.gate_proj", layer.c_gate.as_ref()),
                ] {
                    if let Some(l) = l {
                        linear(&mut parameters, format!("{prefix}.{name}"), l, gpt2)?;
                    }
                }
                for (name, l) in [
                    (
                        if gpt2 {
                            "attn.c_proj"
                        } else {
                            "self_attn.o_proj"
                        },
                        &layer.c_proj,
                    ),
                    (if gpt2 { "mlp.c_fc" } else { "mlp.up_proj" }, &layer.c_fc),
                    (
                        if gpt2 { "mlp.c_proj" } else { "mlp.down_proj" },
                        &layer.c_mlp_proj,
                    ),
                ] {
                    linear(&mut parameters, format!("{prefix}.{name}"), l, gpt2)?;
                }
            }
            let incoming = if let Some(next) = text.layers.get(i + 1) {
                &next.tape.grad_input
            } else {
                &text.grad_final_norm_input
            };
            let grad_d = if layer.post_mlp_norm.is_some() {
                &t.grad_mlp_output
            } else {
                incoming
            };
            let grad_o = if layer.post_attention_norm.is_some() {
                &t.grad_attention_projection
            } else {
                &t.grad_residual1
            };
            for (name, buffer, width) in [
                ("n1", &t.grad_ln1, d),
                ("qkv", &t.grad_qkv, q + 2 * k),
                ("q", &t.grad_q, q),
                ("k", &t.grad_k, k),
                ("v", &t.grad_v, k),
                ("qr", &t.grad_q_rotary, q),
                ("kr", &t.grad_k_rotary, k),
                ("a", &t.grad_attention, q),
                ("o", grad_o, d),
                ("n3", &t.grad_ln2, d),
                ("u", &t.grad_mlp_pre, m),
                ("g", &t.grad_mlp_gate_pre, m),
                ("activation", &t.grad_mlp_activation, m),
                ("p", &t.grad_mlp_activation, m),
                ("d", grad_d, d),
                ("r", &t.grad_residual1, d),
                ("input", &t.grad_input, d),
            ] {
                stages.insert(
                    format!("layer_{i}/{name}"),
                    serde_json::json!(buffer.read_f32(rows * width)?),
                );
            }
        }
        Ok(
            serde_json::json!({"gradients":stages,"parameter_gradients":parameters,"result":result,"image":self.graph.debug_image_training_stages()?}),
        )
    }

    pub fn forward(
        &self,
        input_ids: &[u32],
        attention_mask: &[f32],
        pixels: &[f32],
    ) -> Result<LlavaForward> {
        if self.decoder.is_some() {
            bail!("encoder/decoder image packages require explicit decoder inputs");
        }
        self.graph.forward(input_ids, attention_mask, pixels)
    }

    pub fn forward_seq2seq(
        &self,
        input_ids: &[u32],
        attention_mask: &[f32],
        decoder_ids: &[u32],
        decoder_mask: &[f32],
        pixels: &[f32],
    ) -> Result<LlavaForward> {
        let decoder = self
            .decoder
            .as_ref()
            .context("image package has no decoder")?;
        let rows = self.graph.validate(input_ids, attention_mask, pixels)?;
        self.graph.encode_device(pixels)?;
        let hidden = self.graph.language.encode_with_device_modality_embeddings(
            input_ids,
            attention_mask,
            &rows,
            &self.graph.image.projected,
        )?;
        let logits = decoder.forward_with_device_encoder_context(
            decoder_ids,
            decoder_mask,
            hidden,
            attention_mask,
        )?;
        self.graph.forward_result(logits)
    }

    pub fn train_step_with_optimizer(
        &mut self,
        input_ids: &[u32],
        attention_mask: &[f32],
        pixels: &[f32],
        targets: &[u32],
        loss_weights: &[f32],
        optimizer: crate::CpuAdamWHyperParams,
    ) -> Result<LlavaTrainResult> {
        if self.decoder.is_some() {
            bail!("encoder/decoder image packages require explicit decoder inputs");
        }
        self.graph.train_step_with_optimizer(
            input_ids,
            attention_mask,
            pixels,
            targets,
            loss_weights,
            optimizer,
        )
    }

    pub fn train_seq2seq_with_optimizer(
        &mut self,
        input_ids: &[u32],
        attention_mask: &[f32],
        decoder_ids: &[u32],
        decoder_mask: &[f32],
        pixels: &[f32],
        targets: &[u32],
        weights: &[f32],
        optimizer: crate::CpuAdamWHyperParams,
    ) -> Result<LlavaTrainResult> {
        optimizer.validate()?;
        let decoder = self
            .decoder
            .as_mut()
            .context("image package has no decoder")?;
        let rows = self.graph.validate(input_ids, attention_mask, pixels)?;
        self.graph.language.set_cpu_adamw_policy(Some(optimizer))?;
        decoder.set_cpu_adamw_policy(Some(optimizer))?;
        let hyper = optimizer.as_f32();
        self.graph.encode_device(pixels)?;
        let tape = self.graph.language.begin_encoder_image_training(
            input_ids,
            attention_mask,
            &rows,
            &self.graph.image.projected,
            hyper,
        )?;
        let decoder_result = decoder.train_with_device_encoder_context(
            decoder_ids,
            decoder_mask,
            targets,
            weights,
            self.graph.language.training_terminal_output()?,
            attention_mask,
            hyper,
        )?;
        let mut encoder_result = self.graph.language.finish_encoder_image_training(
            tape,
            decoder.device_encoder_gradient()?,
            hyper,
        )?;
        if encoder_result.text.step != decoder_result.text.step {
            bail!("joint encoder and decoder optimizer steps diverged");
        }
        encoder_result.text = decoder_result.text;
        self.graph.backward_image(encoder_result, pixels, hyper)
    }

    /// Export the actual trained tower/projector and the original text model
    /// namespace. Reloading needs no rewritten text model_type or key prefixes.
    pub fn export_hf_package(
        &self,
        source: impl AsRef<Path>,
        destination: impl AsRef<Path>,
    ) -> Result<()> {
        let source = source.as_ref();
        let destination = destination.as_ref();
        fs::create_dir(destination).context("image-text export destination must be new")?;
        fs::copy(source.join("config.json"), destination.join("config.json"))?;
        if let Some(decoder) = self.decoder.as_ref() {
            self.graph.language.export_joint_seq2seq_package(
                decoder,
                &source.join("text"),
                &destination.join("text"),
            )?;
        } else {
            self.graph
                .language
                .export_hf_package(source.join("text"), destination.join("text"))?;
        }
        let image = destination.join("image");
        fs::create_dir(&image)?;
        for name in ["config.json", "model.safetensors"] {
            fs::copy(source.join("image").join(name), image.join(name))?;
        }
        self.graph
            .export_image_parameters(&source.join("image"), &image)
    }
}
