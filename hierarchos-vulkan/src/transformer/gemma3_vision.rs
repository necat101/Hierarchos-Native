//! Native SigLIP -> Gemma3 soft-embedding projector -> Gemma3 text training.
//! The tower reuses the qualified CLIP encoder layers, with the SigLIP
//! differences this file owns: a biased patch projection, no class token, a
//! terminal LayerNorm and the average-pool projector norm.
use super::vision::{
    norm, permute_patch_weights, tensor, ClipLayer, PatchPush, TokenPush, VisionKernels, DI, OUT,
};
use super::*;
use crate::weight_offload::ModelPackage;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AvgPoolPush {
    images: u32,
    grid: u32,
    kernel: u32,
    dim: u32,
    mode: u32,
}

#[derive(Debug, Serialize)]
pub struct Gemma3Forward {
    pub logits: Vec<f32>,
    pub vision_features: Vec<f32>,
    pub projected_features: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct Gemma3TrainResult {
    pub text: VulkanTransformerStepResult,
    pub pixel_gradient: Vec<f32>,
    pub projected_gradient: Vec<f32>,
}

/// Fixed-resolution SigLIP -> Gemma3 projector -> Gemma3 text graph. Image
/// tensors use NCHW layout. The standard HF package supplies the actual tower,
/// the soft-embedding projector and the text backbone. Dynamic positional
/// interpolation, pan-and-scan crops, video and the attention-pool head fail
/// closed; the head is not part of the Gemma3 image path.
pub struct VulkanGemma3Vision {
    device: VulkanDevice,
    language: VulkanTransformer,
    vk: VisionKernels,
    image_count: usize,
    side: usize,
    patch_size: usize,
    patch_order: u32,
    channels: usize,
    dim: usize,
    intermediate: usize,
    heads: u32,
    grid: usize,
    kernel: usize,
    tokens_per_image: usize,
    rows: usize,
    image_token: u32,
    activation: String,
    processor: Option<serde_json::Value>,
    vision_prefix: String,
    projector_prefix: String,
    patch: VulkanLinear,
    patch_projection: vulkan::ComputeKernel,
    position: TrainableParameter,
    post_norm: VulkanLayerNorm,
    layers: Vec<ClipLayer>,
    projector_norm: VulkanLayerNorm,
    projector: VulkanLinear,
    pixels: GpuBuffer,
    patches: GpuBuffer,
    patch_output: GpuBuffer,
    tokens: GpuBuffer,
    post_output: GpuBuffer,
    pooled: GpuBuffer,
    pooled_norm: GpuBuffer,
    projected: GpuBuffer,
    post_mean: GpuBuffer,
    post_rstd: GpuBuffer,
    pool_mean: GpuBuffer,
    pool_rstd: GpuBuffer,
    mask: GpuBuffer,
    dummy: GpuBuffer,
    grad_projected: GpuBuffer,
    grad_pooled_norm: GpuBuffer,
    grad_pooled: GpuBuffer,
    grad_post_output: GpuBuffer,
    grad_tokens: GpuBuffer,
    grad_patch_output: GpuBuffer,
    grad_patches: GpuBuffer,
    grad_pixels: GpuBuffer,
    siglip_tokens: vulkan::ComputeKernel,
    avg_pool: vulkan::ComputeKernel,
}

impl VulkanGemma3Vision {
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
        if config["model_type"].as_str() != Some("gemma3") {
            bail!("native Gemma3 image pipeline requires model_type=gemma3");
        }
        let vision = &config["vision_config"];
        if vision["model_type"].as_str() != Some("siglip_vision_model") {
            bail!("native Gemma3 currently requires a SigLIP vision tower");
        }
        let number = |value: &serde_json::Value, name: &str| -> Result<usize> {
            let x = usize::try_from(
                value[name]
                    .as_u64()
                    .with_context(|| format!("missing positive {name}"))?,
            )?;
            if x == 0 || x > u32::MAX as usize {
                bail!("invalid {name}");
            }
            Ok(x)
        };
        let side = number(vision, "image_size")?;
        let patch_size = number(vision, "patch_size")?;
        let channels = number(vision, "num_channels")?;
        let dim = number(vision, "hidden_size")?;
        let intermediate = number(vision, "intermediate_size")?;
        let num_layers = number(vision, "num_hidden_layers")?;
        let heads = number(vision, "num_attention_heads")?;
        let tokens_per_image = number(&config, "mm_tokens_per_image")?;
        if image_count == 0 || !side.is_multiple_of(patch_size) || !dim.is_multiple_of(heads) {
            bail!("invalid image count, patch grid or SigLIP head geometry");
        }
        if vision["attention_dropout"].as_f64().unwrap_or(0.0) != 0.0 {
            bail!("native SigLIP training requires attention_dropout=0");
        }
        let configured_activation = vision["hidden_act"]
            .as_str()
            .unwrap_or("gelu_pytorch_tanh")
            .to_string();
        if !matches!(
            configured_activation.as_str(),
            "quick_gelu" | "gelu" | "gelu_pytorch_tanh" | "gelu_new"
        ) {
            bail!("unsupported SigLIP activation {configured_activation}");
        }
        // Built-in F.gelu(approximate="tanh") uses the vector SLEEF path.
        // The explicit gelu_new formula calls torch.tanh (MKL VML) instead;
        // those functions differ at observable FP32 rounding boundaries.
        let activation = if configured_activation == "gelu_new" {
            "gelu_pytorch_tanh_ha".to_string()
        } else {
            configured_activation
        };
        let grid = side / patch_size;
        let tokens_per_side = (tokens_per_image as f64).sqrt() as usize;
        if tokens_per_side * tokens_per_side != tokens_per_image
            || tokens_per_image > grid * grid
            || !grid.is_multiple_of(tokens_per_side)
        {
            bail!("Gemma3 mm_tokens_per_image must be a squared divisor of the patch grid");
        }
        let kernel = grid / tokens_per_side;
        let rows = image_count
            .checked_mul(grid * grid)
            .context("SigLIP row overflow")?;
        let hidden = rows.checked_mul(dim).context("SigLIP hidden overflow")?;
        let pixel_len = image_count
            .checked_mul(channels)
            .and_then(|x| x.checked_mul(side))
            .and_then(|x| x.checked_mul(side))
            .context("SigLIP pixel overflow")?;
        for size in [hidden, pixel_len, rows * intermediate] {
            if size > u32::MAX as usize {
                bail!("SigLIP graph exceeds shader indexing range");
            }
        }
        let eps = vision["layer_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        if !eps.is_finite() || eps <= 0.0 {
            bail!("invalid SigLIP layer norm epsilon");
        }
        let mut language =
            VulkanTransformer::from_hf_package(device.clone(), directory, batch_size, seq_len)?;
        if language.config.architecture != VulkanTransformerArchitecture::Gemma3 {
            bail!("native Gemma3 image pipeline requires its qualified Gemma3 backbone");
        }
        language.kernels.install_rdna3_avx512_layer_norm(&device)?;
        language.kernels.install_rdna3_avx512_rope(&device)?;
        language.kernels.install_rdna3_avx512_eager(&device)?;
        let rdna3_avx2 = vendor::rdna3_frozen_avx2_variants(&device, aten_vector_width());
        if rdna3_avx2 {
            // Match ATen's materialized raw-x LayerNorm VJP and bias reduction
            // in the frozen AVX2 track, as already used by the CLIP adapter.
            language.kernels.cpu_layer_norm_adjoint = true;
            language.kernels.bias_grad = vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/vision_bias_grad_cpu_avx2.spv"),
                2,
                8,
            )?;
        }
        if language.config.embedding_dropout != 0.0
            || language.config.attention_dropout != 0.0
            || language.config.mlp_output_dropout != 0.0
            || language.config.attention_output_dropout != 0.0
        {
            bail!("native Gemma3 training requires deterministic zero-dropout text configuration");
        }
        // The composed sequence can have 8/20 rows rather than the text
        // gate's 12. Use the same measured CPU panels for these NN linears;
        // the convolution patch projection retains its oneDNN ordering.
        language.kernels.cpu_row_gemm = Some((
            vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/vision_linear_cpu_avx2.spv"),
                4,
                16,
            )?,
            vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/vision_linear_input_grad_cpu_avx2.spv"),
                4,
                16,
            )?,
        ));
        let text_dim = language.config.effective_embedding_size();
        let image_token = u32::try_from(
            config["image_token_index"]
                .as_u64()
                .context("missing image_token_index")?,
        )?;
        if image_token as usize >= language.config.vocab_size {
            bail!("image token must be in the text vocabulary");
        }
        let mut package = ModelPackage::open_directory(directory)?;
        let names = package.tensor_names()?;
        let vision_prefix = ["model.vision_tower", "vision_tower"]
            .into_iter()
            .find(|p| {
                names.contains(&format!("{p}.embeddings.patch_embedding.bias"))
                    && names.contains(&format!("{p}.post_layernorm.weight"))
            })
            .context("cannot locate the SigLIP tower namespace")?
            .to_string();
        let projector_prefix = ["model.multi_modal_projector", "multi_modal_projector"]
            .into_iter()
            .find(|p| names.contains(&format!("{p}.mm_input_projection_weight")))
            .context("cannot locate the Gemma3 projector")?
            .to_string();
        let patch_input = channels * patch_size * patch_size;
        // The measured AVX2 single-image 8x8/patch-2 case dispatches ATen's
        // native convolution (channel, kh, kw); batch 2 and patch 4 use
        // oneDNN's direct convolution (kh, kw, channel). Preserve the order
        // through load, patchify, backward and export.
        // This Windows AVX512 host's small single-image convolution uses
        // channel/kh/kw reduction order too. Keep that independently measured
        // RDNA3 admission separate from the frozen AVX2 branch.
        let rdna3_native_conv = vendor::rdna3_avx512_cpu_contract(&device, aten_vector_width())
            && image_count == 1
            && side == 8
            && channels == 3
            && (dim, patch_size) == (16, 2);
        let patch_order = u32::from(
            !((aten_vector_width() == 8
                && image_count == 1
                && side == 8
                && channels == 3
                && (dim, patch_size) == (16, 2))
                || rdna3_native_conv),
        );
        let patch_weights = tensor(
            &mut package,
            &format!("{vision_prefix}.embeddings.patch_embedding.weight"),
            &[dim, channels, patch_size, patch_size],
        )?;
        let patch_weights = if patch_order == 0 {
            patch_weights
        } else {
            permute_patch_weights(&patch_weights, dim, channels, patch_size, true)
        };
        let patch_bias = tensor(
            &mut package,
            &format!("{vision_prefix}.embeddings.patch_embedding.bias"),
            &[dim],
        )?;
        let (projector_shape, projector_values) =
            package.read_f32(&format!("{projector_prefix}.mm_input_projection_weight"))?;
        if projector_shape != vec![dim, text_dim] {
            bail!("Gemma3 projector weight must be [vision_hidden, text_hidden]");
        }
        // HF stores the projection as [vision, text] and applies matmul; the
        // native GEMM consumes [text, vision].
        let mut native_projector = vec![0.0f32; dim * text_dim];
        for row in 0..dim {
            for col in 0..text_dim {
                native_projector[col * dim + row] = projector_values[row * text_dim + col];
            }
        }
        let projector_norm = VulkanLayerNorm::new_rms(
            &device,
            dim,
            eps,
            &tensor(
                &mut package,
                &format!("{projector_prefix}.mm_soft_emb_norm.weight"),
                &[dim],
            )?,
            1.0,
        )?
        .use_unfused_rms_products(true)
        // HF's NCHW average pool is transposed to [image,token,hidden]
        // without a contiguous copy. Its hidden-axis mean is source-ordered,
        // unlike the contiguous text RMSNorm's eight-lane reduction. The
        // backward still receives a contiguous matmul adjoint (mode 1).
        .use_vector_cpu_rsqrt(tokens_per_image == 1)
        .use_torch_cpu_rsqrt(tokens_per_image > 1);
        let projector_norm = if vendor::rdna3_avx512_variants(&device, aten_vector_width())
            || (rdna3_avx2 && tokens_per_image == 16 && image_count == 1 && dim == 16)
        {
            // The frozen AVX2 full-grid projector has the same strided
            // four-stream reduction as the independently qualified AVX512 case.
            projector_norm.use_rdna3_avx512_projector_rms(vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/vision_projector_rms_avx512_amd_rdna3.spv"),
                5,
                32,
            )?)
        } else {
            projector_norm
        };
        let selected_rows = image_count * tokens_per_image;
        let buffer = |len| GpuBuffer::zeros_f32(&device, len);
        Ok(Self {
            patch: VulkanLinear::new(&device, patch_input, dim, &patch_weights, &patch_bias)?,
            patch_projection: vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/vision_patch_projection.spv"),
                4,
                16,
            )?,
            position: TrainableParameter::new(
                &device,
                &tensor(
                    &mut package,
                    &format!("{vision_prefix}.embeddings.position_embedding.weight"),
                    &[grid * grid, dim],
                )?,
                true,
            )?,
            post_norm: norm(
                &mut package,
                &device,
                &format!("{vision_prefix}.post_layernorm"),
                dim,
                eps,
            )?
            .use_cpu_welford(true),
            layers: (0..num_layers)
                .map(|i| {
                    ClipLayer::load(
                        &mut package,
                        &device,
                        format!("{vision_prefix}.encoder.layers.{i}"),
                        rows,
                        dim,
                        intermediate,
                        eps,
                        true,
                        rdna3_avx2,
                    )
                })
                .collect::<Result<_>>()?,
            projector: VulkanLinear::new_no_bias(&device, dim, text_dim, &native_projector)?,
            projector_norm,
            pixels: buffer(pixel_len)?,
            patches: buffer(pixel_len)?,
            patch_output: buffer(hidden)?,
            tokens: buffer(hidden)?,
            post_output: buffer(hidden)?,
            pooled: buffer(selected_rows * dim)?,
            pooled_norm: buffer(selected_rows * dim)?,
            projected: buffer(selected_rows * text_dim)?,
            post_mean: buffer(rows)?,
            post_rstd: buffer(rows)?,
            pool_mean: buffer(selected_rows)?,
            pool_rstd: buffer(selected_rows)?,
            mask: GpuBuffer::from_f32(&device, &vec![1.; rows])?,
            dummy: buffer(rows.max(heads))?,
            grad_projected: buffer(selected_rows * text_dim)?,
            grad_pooled_norm: buffer(selected_rows * dim)?,
            grad_pooled: buffer(selected_rows * dim)?,
            grad_post_output: buffer(hidden)?,
            grad_tokens: buffer(hidden)?,
            grad_patch_output: buffer(hidden)?,
            grad_patches: buffer(pixel_len)?,
            grad_pixels: buffer(pixel_len)?,
            siglip_tokens: vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/vision_siglip_tokens.spv"),
                3,
                16,
            )?,
            avg_pool: vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/vision_avg_pool.spv"),
                2,
                std::mem::size_of::<AvgPoolPush>() as u32,
            )?,
            heads: heads as u32,
            vk: VisionKernels::new(&device)?,
            processor: if directory.join("preprocessor_config.json").is_file() {
                Some(serde_json::from_slice(&fs::read(
                    directory.join("preprocessor_config.json"),
                )?)?)
            } else {
                None
            },
            device,
            language,
            image_count,
            side,
            patch_size,
            channels,
            patch_order,
            dim,
            intermediate,
            grid,
            kernel,
            tokens_per_image,
            rows,
            image_token,
            activation,
            vision_prefix,
            projector_prefix,
        })
    }
    /// Gemma3 scales the token lookup by `sqrt(hidden_size)` after embedding
    /// composition. HF replaces the image rows *after* that scale, while the
    /// bridge injects raw lookup rows and therefore returns the gradient with
    /// the scale included. Pre-divide on the way in and undo it on the way out
    /// so the projector sees HF's exact arithmetic.
    fn embedding_scale(&self) -> f32 {
        self.language.config.token_embedding_scale()
    }
    fn attention_push(&self) -> AttentionPush {
        let mut push = AttentionPush::zeroed();
        push.batch_size = self.image_count as u32;
        push.query_seq_len = (self.grid * self.grid) as u32;
        push.key_value_seq_len = push.query_seq_len;
        push.query_hidden_size = self.dim as u32;
        push.value_hidden_size = self.dim as u32;
        push.num_heads = self.heads;
        push.num_key_value_heads = self.heads;
        push.scale = ((self.dim as f64 / self.heads as f64).sqrt().recip()) as f32;
        push.value_scale = 1.;
        push.unfused_qk = 9;
        push.kv_page_size = 1;
        push.kv_page_table_stride = 1;
        push
    }
    /// Rescale and normalize fixed-size interleaved RGB8 images using the
    /// package's processor contract. Gemma3 processing has no center crop, so
    /// resize, pan-and-scan and crop must all be disabled explicitly.
    pub fn preprocess_rgb8(&self, rgb: &[u8]) -> Result<Vec<f32>> {
        let p = self
            .processor
            .as_ref()
            .context("raw RGB input requires preprocessor_config.json")?;
        if self.channels != 3 || rgb.len() != self.image_count * self.side * self.side * 3 {
            bail!("RGB8 input must match the fixed SigLIP image dimensions");
        }
        if p["do_resize"].as_bool() != Some(false)
            || p["do_rescale"].as_bool() != Some(true)
            || p["do_normalize"].as_bool() != Some(true)
            || p["do_pan_and_scan"].as_bool() == Some(true)
            || p["do_center_crop"].as_bool() == Some(true)
        {
            bail!("native Gemma3 RGB processor requires fixed-size rescale and normalize only");
        }
        let factor = p["rescale_factor"]
            .as_f64()
            .context("missing rescale factor")?;
        let means = p["image_mean"].as_array().context("missing RGB mean")?;
        let stds = p["image_std"]
            .as_array()
            .context("missing RGB standard deviation")?;
        if means.len() != 3 || stds.len() != 3 || !factor.is_finite() || factor <= 0. {
            bail!("invalid RGB processor parameters");
        }
        let mut mean = [0.; 3];
        let mut std = [0.; 3];
        for c in 0..3 {
            mean[c] = means[c].as_f64().context("invalid RGB mean")? as f32;
            std[c] = stds[c].as_f64().context("invalid RGB standard deviation")? as f32;
            if !mean[c].is_finite() || !std[c].is_finite() || std[c] <= 0. {
                bail!("invalid RGB normalization");
            }
        }
        let area = self.side * self.side;
        let mut values = vec![0.; rgb.len()];
        for image in 0..self.image_count {
            for pixel in 0..area {
                for channel in 0..3 {
                    let scaled = (rgb[(image * area + pixel) * 3 + channel] as f64 * factor) as f32;
                    values[(image * 3 + channel) * area + pixel] =
                        (scaled - mean[channel]) / std[channel];
                }
            }
        }
        Ok(values)
    }
    fn validate(&self, ids: &[u32], mask: &[f32], pixels: &[f32]) -> Result<Vec<usize>> {
        if ids.len() != self.language.rows || mask.len() != ids.len() {
            bail!("invalid Gemma3 text batch geometry");
        }
        if mask.iter().any(|&v| v != 0. && v != 1.) {
            bail!("Gemma3 attention mask must be binary");
        }
        if ids
            .iter()
            .any(|&id| id as usize >= self.language.config.vocab_size)
        {
            bail!("Gemma3 token outside vocabulary");
        }
        if pixels.len() != self.image_count * self.channels * self.side * self.side
            || pixels.iter().any(|v| !v.is_finite())
        {
            bail!("pixel_values must be finite NCHW at the configured SigLIP resolution");
        }
        let rows = ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| (id == self.image_token).then_some(i))
            .collect::<Vec<_>>();
        if rows.len() != self.image_count * self.tokens_per_image
            || rows.iter().any(|&i| mask[i] != 1.)
        {
            bail!("image placeholders must match every projected feature and be visible");
        }
        Ok(rows)
    }
    fn pool_push(&self, mode: u32) -> AvgPoolPush {
        AvgPoolPush {
            images: self.image_count as u32,
            grid: self.grid as u32,
            kernel: self.kernel as u32,
            dim: self.dim as u32,
            mode,
        }
    }
    fn record_patch_projection(&self, commands: &mut vulkan::ComputeBatch) -> Result<()> {
        let push = [
            self.rows as u32,
            self.patch.input_dim as u32,
            self.dim as u32,
            self.patch_order,
        ];
        self.patch_projection.record_dispatch(
            commands,
            &[
                &self.patches,
                &self.patch.weight.values,
                &self.patch.bias.values,
                &self.patch_output,
            ],
            bytemuck::cast_slice(&push),
            [div_ceil_u32(self.dim, 16), div_ceil_u32(self.rows, 16), 1],
        )
    }
    fn encode(&self, pixels: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let k = &self.language.kernels;
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.enable_watchdog_submission_slicing();
        commands.upload_f32(&self.pixels, pixels)?;
        let patch = PatchPush::new(
            self.image_count as u32,
            self.side as u32,
            self.patch_size as u32,
            self.channels as u32,
            0,
            self.patch_order,
        );
        self.vk.patch.record_dispatch(
            &mut commands,
            &[&self.pixels, &self.patches],
            bytemuck::bytes_of(&patch),
            [div_ceil_u32(pixels.len(), 256), 1, 1],
        )?;
        self.record_patch_projection(&mut commands)?;
        let tokens = TokenPush::new(
            self.image_count as u32,
            (self.grid * self.grid) as u32,
            self.dim as u32,
            0,
        );
        self.siglip_tokens.record_dispatch(
            &mut commands,
            &[&self.patch_output, &self.position.values, &self.tokens],
            bytemuck::bytes_of(&tokens),
            [div_ceil_u32(self.rows * self.dim, 256), 1, 1],
        )?;
        let mut hidden = &self.tokens;
        for layer in &self.layers {
            layer.forward(
                &mut commands,
                k,
                &self.vk,
                hidden,
                &self.mask,
                &self.dummy,
                self.attention_push(),
                &self.activation,
            )?;
            hidden = &layer.tape[OUT];
        }
        self.post_norm.record_forward(
            &mut commands,
            k,
            hidden,
            &self.post_output,
            &self.post_mean,
            &self.post_rstd,
            self.rows,
        )?;
        let pooled_rows = self.image_count * self.tokens_per_image;
        self.avg_pool.record_dispatch(
            &mut commands,
            &[&self.post_output, &self.pooled],
            bytemuck::bytes_of(&self.pool_push(0)),
            [div_ceil_u32(pooled_rows * self.dim, 256), 1, 1],
        )?;
        self.projector_norm.record_forward(
            &mut commands,
            k,
            &self.pooled,
            &self.pooled_norm,
            &self.pool_mean,
            &self.pool_rstd,
            pooled_rows,
        )?;
        self.projector.record_forward(
            &mut commands,
            k,
            &self.pooled_norm,
            &self.projected,
            pooled_rows,
        )?;
        commands.submit()?;
        Ok((
            self.post_output.read_f32(self.rows * self.dim)?,
            self.projected
                .read_f32(pooled_rows * self.projector.output_dim)?,
        ))
    }
    /// Diagnostic dump of every tower boundary, used by
    /// `diagnose_gemma3_vision.py` to locate where native arithmetic first
    /// leaves the oracle. Not part of the qualification path.
    pub fn stage_dump(&self, pixel_values: &[f32]) -> Result<serde_json::Value> {
        let k = &self.language.kernels;
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.enable_watchdog_submission_slicing();
        commands.upload_f32(&self.pixels, pixel_values)?;
        let patch = PatchPush::new(
            self.image_count as u32,
            self.side as u32,
            self.patch_size as u32,
            self.channels as u32,
            0,
            self.patch_order,
        );
        self.vk.patch.record_dispatch(
            &mut commands,
            &[&self.pixels, &self.patches],
            bytemuck::bytes_of(&patch),
            [div_ceil_u32(pixel_values.len(), 256), 1, 1],
        )?;
        self.record_patch_projection(&mut commands)?;
        let tokens = TokenPush::new(
            self.image_count as u32,
            (self.grid * self.grid) as u32,
            self.dim as u32,
            0,
        );
        self.siglip_tokens.record_dispatch(
            &mut commands,
            &[&self.patch_output, &self.position.values, &self.tokens],
            bytemuck::bytes_of(&tokens),
            [div_ceil_u32(self.rows * self.dim, 256), 1, 1],
        )?;
        let mut hidden = &self.tokens;
        for layer in &self.layers {
            layer.forward(
                &mut commands,
                k,
                &self.vk,
                hidden,
                &self.mask,
                &self.dummy,
                self.attention_push(),
                &self.activation,
            )?;
            hidden = &layer.tape[OUT];
        }
        self.post_norm.record_forward(
            &mut commands,
            k,
            hidden,
            &self.post_output,
            &self.post_mean,
            &self.post_rstd,
            self.rows,
        )?;
        let pooled_rows = self.image_count * self.tokens_per_image;
        self.avg_pool.record_dispatch(
            &mut commands,
            &[&self.post_output, &self.pooled],
            bytemuck::bytes_of(&self.pool_push(0)),
            [div_ceil_u32(pooled_rows * self.dim, 256), 1, 1],
        )?;
        self.projector_norm.record_forward(
            &mut commands,
            k,
            &self.pooled,
            &self.pooled_norm,
            &self.pool_mean,
            &self.pool_rstd,
            pooled_rows,
        )?;
        self.projector.record_forward(
            &mut commands,
            k,
            &self.pooled_norm,
            &self.projected,
            pooled_rows,
        )?;
        commands.submit()?;
        let mut stages = serde_json::Map::new();
        stages.insert(
            "patches".into(),
            serde_json::json!(self.patches.read_f32(pixel_values.len())?),
        );
        stages.insert(
            "patch_output".into(),
            serde_json::json!(self.patch_output.read_f32(self.rows * self.dim)?),
        );
        stages.insert(
            "tokens".into(),
            serde_json::json!(self.tokens.read_f32(self.rows * self.dim)?),
        );
        for (index, layer) in self.layers.iter().enumerate() {
            stages.insert(
                format!("layer_{index}"),
                serde_json::json!(layer.tape[OUT].read_f32(self.rows * self.dim)?),
            );
        }
        if let Some(layer) = self.layers.first() {
            for (name, values) in layer.stage_dump(self.rows, self.dim, self.intermediate)? {
                stages.insert(format!("layer_0/{name}"), serde_json::json!(values));
            }
        }
        stages.insert(
            "post_norm".into(),
            serde_json::json!(self.post_output.read_f32(self.rows * self.dim)?),
        );
        stages.insert(
            "pooled".into(),
            serde_json::json!(self.pooled.read_f32(pooled_rows * self.dim)?),
        );
        stages.insert(
            "pooled_norm".into(),
            serde_json::json!(self.pooled_norm.read_f32(pooled_rows * self.dim)?),
        );
        stages.insert(
            "projected".into(),
            serde_json::json!(self
                .projected
                .read_f32(pooled_rows * self.projector.output_dim)?),
        );
        Ok(serde_json::Value::Object(stages))
    }
    /// Diagnostic boundaries from an ordinary composed forward, without
    /// replacing inputs or altering the dispatch policy.
    pub fn text_stage_dump(
        &self,
        input_ids: &[u32],
        attention_mask: &[f32],
        pixel_values: &[f32],
    ) -> Result<serde_json::Value> {
        let forward = self.forward(input_ids, attention_mask, pixel_values)?;
        let rows = input_ids.len();
        let dim = self.language.config.hidden_size;
        let query_dim = self.language.config.query_hidden_size();
        let kv_dim = self.language.config.key_value_hidden_size();
        let intermediate = self.language.config.intermediate_size;
        let mut stages = serde_json::Map::new();
        stages.insert(
            "input".into(),
            serde_json::json!(self.language.input_hidden.read_f32(rows * dim)?),
        );
        for (index, layer) in self.language.layers.iter().enumerate() {
            let t = &layer.tape;
            for (name, buffer, width) in [
                ("ln1", &t.ln1, dim),
                ("q", &t.q, query_dim),
                ("k", &t.k, kv_dim),
                ("v", &t.v, kv_dim),
                ("q_norm", &t.q_norm, query_dim),
                ("k_norm", &t.k_norm, kv_dim),
                ("q_rotary", &t.q_rotary, query_dim),
                ("k_rotary", &t.k_rotary, kv_dim),
                ("attention", &t.attention, query_dim),
                ("attention_projection", &t.attention_projection, dim),
                ("post_attention_norm", &t.post_attention_norm, dim),
                ("residual1", &t.residual1, dim),
                ("ln2", &t.ln2, dim),
                ("mlp_pre", &t.mlp_pre, intermediate),
                ("mlp_gate_pre", &t.mlp_gate_pre, intermediate),
                ("mlp_product", &t.mlp_product, intermediate),
                ("mlp_output", &t.mlp_output, dim),
                ("post_mlp_norm", &t.post_mlp_norm, dim),
                ("output", &t.output, dim),
            ] {
                stages.insert(
                    format!("layer_{index}/{name}"),
                    serde_json::json!(buffer.read_f32(rows * width)?),
                );
            }
        }
        stages.insert(
            "final_norm".into(),
            serde_json::json!(self.language.final_norm_output.read_f32(rows * dim)?),
        );
        Ok(serde_json::json!({"stages":stages,"logits":forward.logits}))
    }
    pub fn forward(
        &self,
        input_ids: &[u32],
        attention_mask: &[f32],
        pixel_values: &[f32],
    ) -> Result<Gemma3Forward> {
        let rows = self.validate(input_ids, attention_mask, pixel_values)?;
        let (vision_features, projected_features) = self.encode(pixel_values)?;
        let scale = self.embedding_scale();
        let injected = projected_features
            .iter()
            .map(|value| value / scale)
            .collect::<Vec<_>>();
        let logits = self.language.forward_logits_with_modality_embeddings(
            input_ids,
            attention_mask,
            ModalityEmbeddingRows {
                rows: &rows,
                values: &injected,
            },
        )?;
        Ok(Gemma3Forward {
            logits,
            vision_features,
            projected_features,
        })
    }
    /// Read retained text training buffers without changing the graph.
    #[doc(hidden)]
    pub fn debug_text_training_states(&self) -> Result<BTreeMap<String, Vec<f32>>> {
        let mut states = self.language.debug_first_layer_norm_states()?;
        for (name, buffer, len) in [
            ("tokens", &self.tokens, self.rows * self.dim),
            ("post_norm", &self.post_output, self.rows * self.dim),
            ("grad_tokens", &self.grad_tokens, self.rows * self.dim),
            (
                "grad_post_output",
                &self.grad_post_output,
                self.rows * self.dim,
            ),
            (
                "pooled",
                &self.pooled,
                self.image_count * self.tokens_per_image * self.dim,
            ),
            (
                "pooled_norm",
                &self.pooled_norm,
                self.image_count * self.tokens_per_image * self.dim,
            ),
            (
                "grad_pooled",
                &self.grad_pooled,
                self.image_count * self.tokens_per_image * self.dim,
            ),
            (
                "grad_pooled_norm",
                &self.grad_pooled_norm,
                self.image_count * self.tokens_per_image * self.dim,
            ),
            (
                "grad_projected",
                &self.grad_projected,
                self.image_count * self.tokens_per_image * self.projector.output_dim,
            ),
        ] {
            states.insert(format!("vision.{name}"), buffer.read_f32(len)?);
        }
        for (index, layer) in self.layers.iter().enumerate() {
            for (name, values) in
                layer.backward_stage_dump(self.rows, self.dim, self.intermediate)?
            {
                states.insert(format!("vision.layer_{index}/{name}"), values);
            }
        }
        Ok(states)
    }

    pub fn train_step_with_optimizer(
        &mut self,
        input_ids: &[u32],
        attention_mask: &[f32],
        pixel_values: &[f32],
        targets: &[u32],
        loss_weights: &[f32],
        optimizer: crate::CpuAdamWHyperParams,
    ) -> Result<Gemma3TrainResult> {
        optimizer.validate()?;
        let hyper = optimizer.as_f32();
        self.language.set_cpu_adamw_policy(Some(optimizer))?;
        hyper.validate()?;
        let rows = self.validate(input_ids, attention_mask, pixel_values)?;
        if targets.len() != input_ids.len()
            || loss_weights.len() != input_ids.len()
            || targets
                .iter()
                .zip(loss_weights)
                .any(|(&id, &w)| id == self.image_token && w > 0.)
        {
            bail!("Gemma3 targets must match text rows and exclude image-token supervision");
        }
        let (_, projected) = self.encode(pixel_values)?;
        let scale = self.embedding_scale();
        let injected = projected
            .iter()
            .map(|value| value / scale)
            .collect::<Vec<_>>();
        let result = self.language.train_step_with_modality_embeddings(
            input_ids,
            targets,
            attention_mask,
            loss_weights,
            hyper,
            ModalityEmbeddingRows {
                rows: &rows,
                values: &injected,
            },
        )?;
        let step = result.text.step;
        let k = &self.language.kernels;
        let projected_gradient = result
            .modality_embedding_grad
            .iter()
            .map(|value| value / scale)
            .collect::<Vec<_>>();
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.enable_watchdog_submission_slicing();
        commands.upload_f32(&self.grad_projected, &projected_gradient)?;
        let pooled_rows = self.image_count * self.tokens_per_image;
        self.projector.record_backward(
            &mut commands,
            k,
            &self.pooled_norm,
            &self.grad_projected,
            &self.grad_pooled_norm,
            pooled_rows,
        )?;
        self.projector_norm.record_backward(
            &mut commands,
            k,
            &self.pooled,
            &self.grad_pooled_norm,
            &self.pool_mean,
            &self.pool_rstd,
            &self.grad_pooled,
            pooled_rows,
        )?;
        self.avg_pool.record_dispatch(
            &mut commands,
            &[&self.grad_pooled, &self.grad_post_output],
            bytemuck::bytes_of(&self.pool_push(1)),
            [div_ceil_u32(self.rows * self.dim, 256), 1, 1],
        )?;
        let last_input = if self.layers.is_empty() {
            &self.tokens
        } else {
            &self.layers[self.layers.len() - 1].tape[OUT]
        };
        self.post_norm.record_backward(
            &mut commands,
            k,
            last_input,
            &self.grad_post_output,
            &self.post_mean,
            &self.post_rstd,
            &self.grad_tokens,
            self.rows,
        )?;
        let mut grad = &self.grad_tokens;
        for index in (0..self.layers.len()).rev() {
            let input = if index == 0 {
                &self.tokens
            } else {
                &self.layers[index - 1].tape[OUT]
            };
            let layer = &self.layers[index];
            layer.backward(
                &mut commands,
                k,
                &self.vk,
                input,
                grad,
                &self.mask,
                &self.dummy,
                self.attention_push(),
                &self.activation,
            )?;
            grad = &layer.tape[DI];
        }
        let tokens = TokenPush::new(
            self.image_count as u32,
            (self.grid * self.grid) as u32,
            self.dim as u32,
            1,
        );
        self.siglip_tokens.record_dispatch(
            &mut commands,
            &[grad, &self.position.values, self.position.grad_buffer()?],
            bytemuck::bytes_of(&tokens),
            [div_ceil_u32(self.grid * self.grid * self.dim, 256), 1, 1],
        )?;
        commands.copy_f32_range(grad, 0, &self.grad_patch_output, 0, self.rows * self.dim)?;
        self.patch.record_backward(
            &mut commands,
            k,
            &self.patches,
            &self.grad_patch_output,
            &self.grad_patches,
            self.rows,
        )?;
        let patch = PatchPush::new(
            self.image_count as u32,
            self.side as u32,
            self.patch_size as u32,
            self.channels as u32,
            1,
            self.patch_order,
        );
        self.vk.patch.record_dispatch(
            &mut commands,
            &[&self.grad_patches, &self.grad_pixels],
            bytemuck::bytes_of(&patch),
            [div_ceil_u32(pixel_values.len(), 256), 1, 1],
        )?;
        self.projector.record_step(&mut commands, k, step, hyper)?;
        self.projector_norm
            .record_step(&mut commands, k, step, hyper)?;
        for layer in &self.layers {
            layer.step(&mut commands, k, step, hyper)?;
        }
        self.post_norm.record_step(&mut commands, k, step, hyper)?;
        self.patch.record_step(&mut commands, k, step, hyper)?;
        self.position.record_step(&mut commands, k, step, hyper)?;
        commands.submit()?;
        Ok(Gemma3TrainResult {
            text: result.text,
            pixel_gradient: self.grad_pixels.read_f32(pixel_values.len())?,
            projected_gradient,
        })
    }
    pub fn export_hf_package(
        &self,
        source: impl AsRef<Path>,
        destination: impl AsRef<Path>,
    ) -> Result<()> {
        self.language
            .export_hf_package(source.as_ref(), destination.as_ref())?;
        if source.as_ref().join("preprocessor_config.json").is_file() {
            fs::copy(
                source.as_ref().join("preprocessor_config.json"),
                destination.as_ref().join("preprocessor_config.json"),
            )?;
        }
        let mut values = BTreeMap::new();
        let prefix = &self.vision_prefix;
        values.insert(
            format!("{prefix}.embeddings.patch_embedding.weight"),
            (
                vec![self.dim, self.channels, self.patch_size, self.patch_size],
                if self.patch_order == 0 {
                    self.patch.weight.read()?
                } else {
                    permute_patch_weights(
                        &self.patch.weight.read()?,
                        self.dim,
                        self.channels,
                        self.patch_size,
                        false,
                    )
                },
            ),
        );
        values.insert(
            format!("{prefix}.embeddings.patch_embedding.bias"),
            (vec![self.dim], self.patch.bias.read()?),
        );
        values.insert(
            format!("{prefix}.embeddings.position_embedding.weight"),
            (vec![self.grid * self.grid, self.dim], self.position.read()?),
        );
        super::vision::export_norm(
            &mut values,
            &format!("{prefix}.post_layernorm"),
            &self.post_norm,
        )?;
        for layer in &self.layers {
            layer.export(&mut values)?;
        }
        let native = self.projector.weight.read()?;
        let text_dim = self.projector.output_dim;
        let mut hf = vec![0.0f32; self.dim * text_dim];
        for row in 0..self.dim {
            for col in 0..text_dim {
                hf[row * text_dim + col] = native[col * self.dim + row];
            }
        }
        values.insert(
            format!("{}.mm_input_projection_weight", self.projector_prefix),
            (vec![self.dim, text_dim], hf),
        );
        values.insert(
            format!("{}.mm_soft_emb_norm.weight", self.projector_prefix),
            (vec![self.dim], self.projector_norm.weight.read()?),
        );
        let replacements = values
            .iter()
            .map(|(name, (shape, values))| (name.as_str(), shape.as_slice(), values.as_slice()))
            .collect::<Vec<_>>();
        let path = destination.as_ref().join("model.safetensors");
        crate::replace_f32_tensors(&path, &path, &replacements)
    }
}
