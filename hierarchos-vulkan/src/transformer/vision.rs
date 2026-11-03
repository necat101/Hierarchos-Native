//! Native CLIP -> LLaVA projector -> language-model training. Model arithmetic
//! stays on Vulkan; the host validates shapes and transfers graph boundaries.
use super::*;
use crate::weight_offload::ModelPackage;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PatchPush {
    images: u32,
    side: u32,
    patch: u32,
    channels: u32,
    inverse: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TokenPush {
    images: u32,
    seq: u32,
    dim: u32,
    mode: u32,
}

fn tensor(package: &mut ModelPackage, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let (actual, values) = package.read_f32(name)?;
    if actual != shape {
        bail!("{name}: expected {shape:?}, found {actual:?}");
    }
    Ok(values)
}
fn linear(
    package: &mut ModelPackage,
    device: &VulkanDevice,
    prefix: &str,
    input: usize,
    output: usize,
    bias: bool,
) -> Result<VulkanLinear> {
    let w = tensor(package, &format!("{prefix}.weight"), &[output, input])?;
    if bias {
        VulkanLinear::new(
            device,
            input,
            output,
            &w,
            &tensor(package, &format!("{prefix}.bias"), &[output])?,
        )
    } else {
        VulkanLinear::new_no_bias(device, input, output, &w)
    }
}
fn norm(
    package: &mut ModelPackage,
    device: &VulkanDevice,
    prefix: &str,
    dim: usize,
    eps: f32,
) -> Result<VulkanLayerNorm> {
    VulkanLayerNorm::new(
        device,
        dim,
        eps,
        &tensor(package, &format!("{prefix}.weight"), &[dim])?,
        &tensor(package, &format!("{prefix}.bias"), &[dim])?,
    )
}
fn export_linear(
    values: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    prefix: &str,
    linear: &VulkanLinear,
) -> Result<()> {
    values.insert(
        format!("{prefix}.weight"),
        (
            vec![linear.output_dim, linear.input_dim],
            linear.weight.read()?,
        ),
    );
    if linear.has_bias {
        values.insert(
            format!("{prefix}.bias"),
            (vec![linear.output_dim], linear.bias.read()?),
        );
    }
    Ok(())
}
fn export_norm(
    values: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    prefix: &str,
    norm: &VulkanLayerNorm,
) -> Result<()> {
    values.insert(
        format!("{prefix}.weight"),
        (vec![norm.dim], norm.weight.read()?),
    );
    values.insert(
        format!("{prefix}.bias"),
        (vec![norm.dim], norm.bias.read()?),
    );
    Ok(())
}

struct VisionKernels {
    patch: vulkan::ComputeKernel,
    tokens: vulkan::ComputeKernel,
    quick_forward: vulkan::ComputeKernel,
    quick_backward: vulkan::ComputeKernel,
}
impl VisionKernels {
    fn new(device: &VulkanDevice) -> Result<Self> {
        Ok(Self {
            patch: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../../shaders/vision_patchify.spv"),
                2,
                20,
            )?,
            tokens: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../../shaders/vision_clip_tokens.spv"),
                4,
                16,
            )?,
            quick_forward: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../../shaders/vision_quick_gelu.spv"),
                2,
                4,
            )?,
            quick_backward: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../../shaders/vision_quick_gelu_backward.spv"),
                3,
                4,
            )?,
        })
    }
    fn activation(
        &self,
        k: &TransformerKernels,
        name: &str,
        commands: &mut vulkan::ComputeBatch,
        x: &GpuBuffer,
        gradient: Option<&GpuBuffer>,
        output: &GpuBuffer,
        len: usize,
    ) -> Result<()> {
        let kernel = match (name, gradient.is_some()) {
            ("quick_gelu", false) => &self.quick_forward,
            ("quick_gelu", true) => &self.quick_backward,
            ("gelu", false) => &k.gelu_erf_forward,
            ("gelu", true) => &k.gelu_erf_backward,
            ("gelu_pytorch_tanh" | "gelu_new", false) => &k.gelu_tanh_forward,
            ("gelu_pytorch_tanh" | "gelu_new", true) => &k.gelu_tanh_backward,
            _ => bail!("unsupported native vision activation {name}"),
        };
        let bindings = if let Some(g) = gradient {
            vec![x, g, output]
        } else {
            vec![x, output]
        };
        kernel.record_dispatch(
            commands,
            &bindings,
            bytemuck::bytes_of(&LenPush { len: len as u32 }),
            [div_ceil_u32(len, 256), 1, 1],
        )
    }
}

const N1: usize = 0;
const Q: usize = 1;
const K: usize = 2;
const V: usize = 3;
const A: usize = 4;
const O: usize = 5;
const R: usize = 6;
const N2: usize = 7;
const F: usize = 8;
const ACT: usize = 9;
const OUT: usize = 10;
const M1: usize = 11;
const S1: usize = 12;
const M2: usize = 13;
const S2: usize = 14;
const DACT: usize = 15;
const DF: usize = 16;
const DN2: usize = 17;
const DR: usize = 18;
const DA: usize = 19;
const DQ: usize = 20;
const DK: usize = 21;
const DV: usize = 22;
const DQI: usize = 23;
const DKI: usize = 24;
const DVI: usize = 25;
const DN1: usize = 26;
const DI: usize = 27;
const TMP: usize = 28;

struct ClipLayer {
    q: VulkanLinear,
    key: VulkanLinear,
    v: VulkanLinear,
    out: VulkanLinear,
    fc1: VulkanLinear,
    fc2: VulkanLinear,
    ln1: VulkanLayerNorm,
    ln2: VulkanLayerNorm,
    tape: Vec<GpuBuffer>,
    prefix: String,
}
impl ClipLayer {
    fn load(
        package: &mut ModelPackage,
        device: &VulkanDevice,
        prefix: String,
        rows: usize,
        dim: usize,
        intermediate: usize,
        eps: f32,
    ) -> Result<Self> {
        let sizes = (0..29).map(|index| match index {
            F | ACT | DACT | DF => rows * intermediate,
            M1 | S1 | M2 | S2 => rows,
            _ => rows * dim,
        });
        Ok(Self {
            q: linear(
                package,
                device,
                &format!("{prefix}.self_attn.q_proj"),
                dim,
                dim,
                true,
            )?,
            key: linear(
                package,
                device,
                &format!("{prefix}.self_attn.k_proj"),
                dim,
                dim,
                true,
            )?,
            v: linear(
                package,
                device,
                &format!("{prefix}.self_attn.v_proj"),
                dim,
                dim,
                true,
            )?,
            out: linear(
                package,
                device,
                &format!("{prefix}.self_attn.out_proj"),
                dim,
                dim,
                true,
            )?,
            fc1: linear(
                package,
                device,
                &format!("{prefix}.mlp.fc1"),
                dim,
                intermediate,
                true,
            )?,
            fc2: linear(
                package,
                device,
                &format!("{prefix}.mlp.fc2"),
                intermediate,
                dim,
                true,
            )?,
            ln1: norm(package, device, &format!("{prefix}.layer_norm1"), dim, eps)?,
            ln2: norm(package, device, &format!("{prefix}.layer_norm2"), dim, eps)?,
            tape: sizes
                .map(|len| GpuBuffer::zeros_f32(device, len))
                .collect::<Result<_>>()?,
            prefix,
        })
    }
    fn forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        k: &TransformerKernels,
        vk: &VisionKernels,
        x: &GpuBuffer,
        mask: &GpuBuffer,
        dummy: &GpuBuffer,
        push: AttentionPush,
        activation: &str,
    ) -> Result<()> {
        let rows = (push.batch_size * push.query_seq_len) as usize;
        let dim = push.query_hidden_size as usize;
        let t = &self.tape;
        self.ln1
            .record_forward(commands, k, x, &t[N1], &t[M1], &t[S1], rows)?;
        self.q.record_forward(commands, k, &t[N1], &t[Q], rows)?;
        self.key.record_forward(commands, k, &t[N1], &t[K], rows)?;
        self.v.record_forward(commands, k, &t[N1], &t[V], rows)?;
        k.attention_forward.record_dispatch(
            commands,
            &[
                &t[Q], &t[K], &t[V], mask, mask, dummy, &t[A], dummy, dummy, mask, mask,
            ],
            bytemuck::bytes_of(&push),
            [push.query_seq_len, push.num_heads, push.batch_size],
        )?;
        self.out.record_forward(commands, k, &t[A], &t[O], rows)?;
        record_add(commands, k, x, &t[O], &t[R], rows * dim)?;
        self.ln2
            .record_forward(commands, k, &t[R], &t[N2], &t[M2], &t[S2], rows)?;
        self.fc1.record_forward(commands, k, &t[N2], &t[F], rows)?;
        vk.activation(
            k,
            activation,
            commands,
            &t[F],
            None,
            &t[ACT],
            rows * self.fc1.output_dim,
        )?;
        self.fc2
            .record_forward(commands, k, &t[ACT], &t[OUT], rows)?;
        record_add(commands, k, &t[R], &t[OUT], &t[OUT], rows * dim)
    }
    fn backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        k: &TransformerKernels,
        vk: &VisionKernels,
        x: &GpuBuffer,
        dy: &GpuBuffer,
        mask: &GpuBuffer,
        dummy: &GpuBuffer,
        push: AttentionPush,
        activation: &str,
    ) -> Result<()> {
        let rows = (push.batch_size * push.query_seq_len) as usize;
        let dim = push.query_hidden_size as usize;
        let t = &self.tape;
        self.fc2
            .record_backward(commands, k, &t[ACT], dy, &t[DACT], rows)?;
        vk.activation(
            k,
            activation,
            commands,
            &t[F],
            Some(&t[DACT]),
            &t[DF],
            rows * self.fc1.output_dim,
        )?;
        self.fc1
            .record_backward(commands, k, &t[N2], &t[DF], &t[DN2], rows)?;
        self.ln2
            .record_backward(commands, k, &t[R], &t[DN2], &t[M2], &t[S2], &t[DR], rows)?;
        record_add(commands, k, dy, &t[DR], &t[DR], rows * dim)?;
        self.out
            .record_backward(commands, k, &t[A], &t[DR], &t[DA], rows)?;
        k.attention_backward.record_dispatch(
            commands,
            &[
                &t[Q], &t[K], &t[V], mask, mask, dummy, &t[DA], &t[DQ], &t[DK], &t[DV], dummy,
                dummy, dummy, dummy, mask,
            ],
            bytemuck::bytes_of(&push.as_backward_push(0, 1)),
            [push.num_heads, push.batch_size, 1],
        )?;
        self.q
            .record_backward(commands, k, &t[N1], &t[DQ], &t[DQI], rows)?;
        self.key
            .record_backward(commands, k, &t[N1], &t[DK], &t[DKI], rows)?;
        self.v
            .record_backward(commands, k, &t[N1], &t[DV], &t[DVI], rows)?;
        record_add(commands, k, &t[DQI], &t[DKI], &t[TMP], rows * dim)?;
        record_add(commands, k, &t[TMP], &t[DVI], &t[DN1], rows * dim)?;
        self.ln1
            .record_backward(commands, k, x, &t[DN1], &t[M1], &t[S1], &t[DI], rows)?;
        record_add(commands, k, &t[DR], &t[DI], &t[DI], rows * dim)
    }
    fn step(
        &self,
        commands: &mut vulkan::ComputeBatch,
        k: &TransformerKernels,
        step: u32,
        hyper: AdamWHyperParams,
    ) -> Result<()> {
        for linear in [&self.q, &self.key, &self.v, &self.out, &self.fc1, &self.fc2] {
            linear.record_step(commands, k, step, hyper)?;
        }
        self.ln1.record_step(commands, k, step, hyper)?;
        self.ln2.record_step(commands, k, step, hyper)
    }
    fn export(&self, values: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>) -> Result<()> {
        for (suffix, linear) in [
            ("self_attn.q_proj", &self.q),
            ("self_attn.k_proj", &self.key),
            ("self_attn.v_proj", &self.v),
            ("self_attn.out_proj", &self.out),
            ("mlp.fc1", &self.fc1),
            ("mlp.fc2", &self.fc2),
        ] {
            export_linear(values, &format!("{}.{suffix}", self.prefix), linear)?;
        }
        export_norm(values, &format!("{}.layer_norm1", self.prefix), &self.ln1)?;
        export_norm(values, &format!("{}.layer_norm2", self.prefix), &self.ln2)
    }
}

#[derive(Debug, Serialize)]
pub struct LlavaForward {
    pub logits: Vec<f32>,
    pub vision_features: Vec<f32>,
    pub projected_features: Vec<f32>,
}
#[derive(Debug, Serialize)]
pub struct LlavaTrainResult {
    pub text: VulkanTransformerStepResult,
    pub pixel_gradient: Vec<f32>,
    pub projected_gradient: Vec<f32>,
}

/// Fixed-resolution CLIP/LLaVA FP32 graph. Image tensors use NCHW layout.
/// The standard HF package supplies the actual tower and two-layer projector.
/// Dynamic positional interpolation, video, multiple selected feature layers,
/// dropout and non-CLIP towers fail closed pending separate implementations.
pub struct VulkanLlava {
    device: VulkanDevice,
    language: VulkanTransformer,
    vk: VisionKernels,
    image_count: usize,
    side: usize,
    patch_size: usize,
    channels: usize,
    dim: usize,
    heads: u32,
    seq: usize,
    features_per_image: usize,
    drop_cls: bool,
    image_token: u32,
    activation: String,
    projector_activation: String,
    processor: Option<serde_json::Value>,
    vision_prefix: String,
    projector_prefix: String,
    patch: VulkanLinear,
    cls: TrainableParameter,
    position: TrainableParameter,
    pre_norm: VulkanLayerNorm,
    layers: Vec<ClipLayer>,
    projector1: VulkanLinear,
    projector2: VulkanLinear,
    pixels: GpuBuffer,
    patches: GpuBuffer,
    patch_output: GpuBuffer,
    tokens: GpuBuffer,
    pre_output: GpuBuffer,
    pre_mean: GpuBuffer,
    pre_rstd: GpuBuffer,
    mask: GpuBuffer,
    dummy: GpuBuffer,
    selected: GpuBuffer,
    projector_pre: GpuBuffer,
    projector_act: GpuBuffer,
    projected: GpuBuffer,
    grad_projected: GpuBuffer,
    grad_act: GpuBuffer,
    grad_projector_pre: GpuBuffer,
    grad_selected: GpuBuffer,
    grad_vision_output: GpuBuffer,
    grad_tokens: GpuBuffer,
    grad_patch_output: GpuBuffer,
    grad_patches: GpuBuffer,
    grad_pixels: GpuBuffer,
}

impl VulkanLlava {
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
        if config["model_type"].as_str() != Some("llava") {
            bail!("native image pipeline requires model_type=llava");
        }
        let vision = &config["vision_config"];
        if vision["model_type"].as_str() != Some("clip_vision_model") {
            bail!("native LLaVA currently requires a CLIP vision tower");
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
        if image_count == 0 || !side.is_multiple_of(patch_size) || !dim.is_multiple_of(heads) {
            bail!("invalid image count, patch grid or CLIP head geometry");
        }
        if vision["attention_dropout"].as_f64().unwrap_or(0.0) != 0.0 {
            bail!("native CLIP training requires attention_dropout=0");
        }
        let activation = vision["hidden_act"]
            .as_str()
            .unwrap_or("quick_gelu")
            .to_string();
        let projector_activation = config["projector_hidden_act"]
            .as_str()
            .unwrap_or("gelu")
            .to_string();
        for name in [&activation, &projector_activation] {
            if !matches!(
                name.as_str(),
                "quick_gelu" | "gelu" | "gelu_pytorch_tanh" | "gelu_new"
            ) {
                bail!("unsupported vision/projector activation {name}");
            }
        }
        let layer = config["vision_feature_layer"]
            .as_i64()
            .context("vision_feature_layer must be one integer")?;
        let active = if layer < 0 {
            (num_layers as i64 + 1).checked_add(layer)
        } else {
            Some(layer)
        }
        .context("vision feature layer overflow")?;
        if active < 0 || active > num_layers as i64 {
            bail!("vision feature layer is outside CLIP hidden states");
        }
        let drop_cls = match config["vision_feature_select_strategy"]
            .as_str()
            .unwrap_or("default")
        {
            "default" => true,
            "full" => false,
            _ => bail!("unsupported vision feature selection"),
        };
        let seq = (side / patch_size)
            .checked_pow(2)
            .and_then(|x| x.checked_add(1))
            .context("vision grid overflow")?;
        let features_per_image = seq - usize::from(drop_cls);
        let rows = image_count
            .checked_mul(seq)
            .context("vision row overflow")?;
        let hidden = rows
            .checked_mul(dim)
            .context("vision hidden size overflow")?;
        let pixel_len = image_count
            .checked_mul(channels)
            .and_then(|x| x.checked_mul(side))
            .and_then(|x| x.checked_mul(side))
            .context("pixel size overflow")?;
        for size in [
            hidden,
            pixel_len,
            image_count * features_per_image,
            rows * intermediate,
        ] {
            if size > u32::MAX as usize {
                bail!("CLIP graph exceeds shader indexing range");
            }
        }
        let eps = vision["layer_norm_eps"].as_f64().unwrap_or(1e-5) as f32;
        if !eps.is_finite() || eps <= 0.0 {
            bail!("invalid CLIP layer norm epsilon");
        }
        let language =
            VulkanTransformer::from_hf_package(device.clone(), directory, batch_size, seq_len)?;
        // The initial public composite contract is exactly CLIP + Llama. Other
        // backbones keep their separate feature-bridge qualification records.
        if language.config.architecture != VulkanTransformerArchitecture::Llama {
            bail!("native LLaVA pipeline currently requires its qualified Llama backbone");
        }
        if language.config.embedding_dropout != 0.0
            || language.config.attention_dropout != 0.0
            || language.config.mlp_output_dropout != 0.0
            || language.config.attention_output_dropout != 0.0
        {
            bail!("native LLaVA training requires deterministic zero-dropout text configuration");
        }
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
        let vision_prefix = [
            "model.vision_tower",
            "model.vision_tower.vision_model",
            "vision_tower.vision_model",
            "vision_tower",
        ]
        .into_iter()
        .find(|p| names.contains(&format!("{p}.embeddings.class_embedding")))
        .context("cannot locate the CLIP tower namespace")?
        .to_string();
        let projector_prefix = ["model.multi_modal_projector", "multi_modal_projector"]
            .into_iter()
            .find(|p| names.contains(&format!("{p}.linear_1.weight")))
            .context("cannot locate LLaVA projector")?
            .to_string();
        let patch_input = channels * patch_size * patch_size;
        let patch_weights = tensor(
            &mut package,
            &format!("{vision_prefix}.embeddings.patch_embedding.weight"),
            &[dim, channels, patch_size, patch_size],
        )?;
        let bias = config["multimodal_projector_bias"]
            .as_bool()
            .unwrap_or(true);
        let selected_rows = image_count * features_per_image;
        let buffer = |len| GpuBuffer::zeros_f32(&device, len);
        Ok(Self {
            patch: VulkanLinear::new_no_bias(&device, patch_input, dim, &patch_weights)?,
            cls: TrainableParameter::new(
                &device,
                &tensor(
                    &mut package,
                    &format!("{vision_prefix}.embeddings.class_embedding"),
                    &[dim],
                )?,
                false,
            )?,
            position: TrainableParameter::new(
                &device,
                &tensor(
                    &mut package,
                    &format!("{vision_prefix}.embeddings.position_embedding.weight"),
                    &[seq, dim],
                )?,
                true,
            )?,
            pre_norm: norm(
                &mut package,
                &device,
                &format!("{vision_prefix}.pre_layrnorm"),
                dim,
                eps,
            )?,
            layers: (0..active as usize)
                .map(|i| {
                    ClipLayer::load(
                        &mut package,
                        &device,
                        format!("{vision_prefix}.encoder.layers.{i}"),
                        rows,
                        dim,
                        intermediate,
                        eps,
                    )
                })
                .collect::<Result<_>>()?,
            projector1: linear(
                &mut package,
                &device,
                &format!("{projector_prefix}.linear_1"),
                dim,
                text_dim,
                bias,
            )?,
            projector2: linear(
                &mut package,
                &device,
                &format!("{projector_prefix}.linear_2"),
                text_dim,
                text_dim,
                bias,
            )?,
            pixels: buffer(pixel_len)?,
            patches: buffer(pixel_len)?,
            patch_output: buffer(image_count * (seq - 1) * dim)?,
            tokens: buffer(hidden)?,
            pre_output: buffer(hidden)?,
            pre_mean: buffer(rows)?,
            pre_rstd: buffer(rows)?,
            mask: GpuBuffer::from_f32(&device, &vec![1.; rows])?,
            dummy: buffer(rows.max(heads))?,
            selected: buffer(selected_rows * dim)?,
            projector_pre: buffer(selected_rows * text_dim)?,
            projector_act: buffer(selected_rows * text_dim)?,
            projected: buffer(selected_rows * text_dim)?,
            grad_projected: buffer(selected_rows * text_dim)?,
            grad_act: buffer(selected_rows * text_dim)?,
            grad_projector_pre: buffer(selected_rows * text_dim)?,
            grad_selected: buffer(selected_rows * dim)?,
            grad_vision_output: buffer(hidden)?,
            grad_tokens: buffer(hidden)?,
            grad_patch_output: buffer(image_count * (seq - 1) * dim)?,
            grad_patches: buffer(pixel_len)?,
            grad_pixels: buffer(pixel_len)?,
            processor: if directory.join("preprocessor_config.json").is_file() {
                Some(serde_json::from_slice(&fs::read(
                    directory.join("preprocessor_config.json"),
                )?)?)
            } else {
                None
            },
            heads: heads as u32,
            vk: VisionKernels::new(&device)?,
            device,
            language,
            image_count,
            side,
            patch_size,
            channels,
            dim,
            seq,
            features_per_image,
            drop_cls,
            image_token,
            activation,
            projector_activation,
            vision_prefix,
            projector_prefix,
        })
    }
    fn attention_push(&self) -> AttentionPush {
        let mut push = AttentionPush::zeroed();
        push.batch_size = self.image_count as u32;
        push.query_seq_len = self.seq as u32;
        push.key_value_seq_len = self.seq as u32;
        push.query_hidden_size = self.dim as u32;
        push.value_hidden_size = self.dim as u32;
        // Every layer has the same validated CLIP head geometry.
        // Stored separately from the text graph, whose GQA may differ.
        push.num_heads = self.vision_heads();
        push.num_key_value_heads = push.num_heads;
        push.scale = ((self.dim as f64 / push.num_heads as f64).sqrt().recip()) as f32;
        push.value_scale = 1.;
        push.unfused_qk = 9;
        push.kv_page_size = 1;
        push.kv_page_table_stride = 1;
        push
    }
    fn vision_heads(&self) -> u32 {
        self.heads
    }
    /// Rescale and normalize fixed-size interleaved RGB8 images using the
    /// package's processor contract. Resize/crop must be disabled explicitly;
    /// arbitrary decoded-image resizing is a separate processor qualification.
    pub fn preprocess_rgb8(&self, rgb: &[u8]) -> Result<Vec<f32>> {
        let p = self
            .processor
            .as_ref()
            .context("raw RGB input requires preprocessor_config.json")?;
        if self.channels != 3 || rgb.len() != self.image_count * self.side * self.side * 3 {
            bail!("RGB8 input must match the fixed CLIP image dimensions");
        }
        if p["do_resize"].as_bool() != Some(false)
            || p["do_center_crop"].as_bool() != Some(false)
            || p["do_rescale"].as_bool() != Some(true)
            || p["do_normalize"].as_bool() != Some(true)
        {
            bail!(
                "native RGB processor requires fixed-size input with rescale and normalize enabled"
            );
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
            bail!("invalid LLaVA text batch geometry");
        }
        if mask.iter().any(|&v| v != 0. && v != 1.) {
            bail!("LLaVA attention mask must be binary");
        }
        if ids
            .iter()
            .any(|&id| id as usize >= self.language.config.vocab_size)
        {
            bail!("LLaVA token outside vocabulary");
        }
        if pixels.len() != self.image_count * self.channels * self.side * self.side
            || pixels.iter().any(|v| !v.is_finite())
        {
            bail!("pixel_values must be finite NCHW at the configured CLIP resolution");
        }
        let rows = ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| (id == self.image_token).then_some(i))
            .collect::<Vec<_>>();
        if rows.len() != self.image_count * self.features_per_image
            || rows.iter().any(|&i| mask[i] != 1.)
        {
            bail!("image placeholders must match every projected feature and be visible");
        }
        Ok(rows)
    }
    fn encode(&self, pixels: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let k = &self.language.kernels;
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.enable_watchdog_submission_slicing();
        commands.upload_f32(&self.pixels, pixels)?;
        let patch = PatchPush {
            images: self.image_count as u32,
            side: self.side as u32,
            patch: self.patch_size as u32,
            channels: self.channels as u32,
            inverse: 0,
        };
        self.vk.patch.record_dispatch(
            &mut commands,
            &[&self.pixels, &self.patches],
            bytemuck::bytes_of(&patch),
            [div_ceil_u32(pixels.len(), 256), 1, 1],
        )?;
        self.patch.record_forward(
            &mut commands,
            k,
            &self.patches,
            &self.patch_output,
            self.image_count * (self.seq - 1),
        )?;
        let tokens = TokenPush {
            images: self.image_count as u32,
            seq: self.seq as u32,
            dim: self.dim as u32,
            mode: 0,
        };
        self.vk.tokens.record_dispatch(
            &mut commands,
            &[
                &self.patch_output,
                &self.cls.values,
                &self.position.values,
                &self.tokens,
            ],
            bytemuck::bytes_of(&tokens),
            [
                div_ceil_u32(self.image_count * self.seq * self.dim, 256),
                1,
                1,
            ],
        )?;
        self.pre_norm.record_forward(
            &mut commands,
            k,
            &self.tokens,
            &self.pre_output,
            &self.pre_mean,
            &self.pre_rstd,
            self.image_count * self.seq,
        )?;
        let mut hidden = &self.pre_output;
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
        for image in 0..self.image_count {
            commands.copy_f32_range(
                hidden,
                (image * self.seq + usize::from(self.drop_cls)) * self.dim,
                &self.selected,
                image * self.features_per_image * self.dim,
                self.features_per_image * self.dim,
            )?;
        }
        let rows = self.image_count * self.features_per_image;
        self.projector1.record_forward(
            &mut commands,
            k,
            &self.selected,
            &self.projector_pre,
            rows,
        )?;
        self.vk.activation(
            k,
            &self.projector_activation,
            &mut commands,
            &self.projector_pre,
            None,
            &self.projector_act,
            rows * self.projector1.output_dim,
        )?;
        self.projector2.record_forward(
            &mut commands,
            k,
            &self.projector_act,
            &self.projected,
            rows,
        )?;
        commands.submit()?;
        Ok((
            self.selected.read_f32(rows * self.dim)?,
            self.projected.read_f32(rows * self.projector2.output_dim)?,
        ))
    }
    pub fn forward(
        &self,
        input_ids: &[u32],
        attention_mask: &[f32],
        pixel_values: &[f32],
    ) -> Result<LlavaForward> {
        let rows = self.validate(input_ids, attention_mask, pixel_values)?;
        let (vision_features, projected_features) = self.encode(pixel_values)?;
        let logits = self.language.forward_logits_with_modality_embeddings(
            input_ids,
            attention_mask,
            ModalityEmbeddingRows {
                rows: &rows,
                values: &projected_features,
            },
        )?;
        Ok(LlavaForward {
            logits,
            vision_features,
            projected_features,
        })
    }
    pub fn train_step(
        &mut self,
        input_ids: &[u32],
        attention_mask: &[f32],
        pixel_values: &[f32],
        targets: &[u32],
        loss_weights: &[f32],
        hyper: AdamWHyperParams,
    ) -> Result<LlavaTrainResult> {
        self.train_step_with_optimizer(
            input_ids,
            attention_mask,
            pixel_values,
            targets,
            loss_weights,
            hyper.into(),
        )
    }

    pub fn train_step_with_optimizer(
        &mut self,
        input_ids: &[u32],
        attention_mask: &[f32],
        pixel_values: &[f32],
        targets: &[u32],
        loss_weights: &[f32],
        optimizer: crate::CpuAdamWHyperParams,
    ) -> Result<LlavaTrainResult> {
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
            bail!("LLaVA targets must match text rows and exclude image-token supervision");
        }
        let (_, projected) = self.encode(pixel_values)?;
        let result = self.language.train_step_with_modality_embeddings(
            input_ids,
            targets,
            attention_mask,
            loss_weights,
            hyper,
            ModalityEmbeddingRows {
                rows: &rows,
                values: &projected,
            },
        )?;
        let step = result.text.step;
        let k = &self.language.kernels;
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.enable_watchdog_submission_slicing();
        commands.upload_f32(&self.grad_projected, &result.modality_embedding_grad)?;
        let feature_rows = self.image_count * self.features_per_image;
        self.projector2.record_backward(
            &mut commands,
            k,
            &self.projector_act,
            &self.grad_projected,
            &self.grad_act,
            feature_rows,
        )?;
        self.vk.activation(
            k,
            &self.projector_activation,
            &mut commands,
            &self.projector_pre,
            Some(&self.grad_act),
            &self.grad_projector_pre,
            feature_rows * self.projector1.output_dim,
        )?;
        self.projector1.record_backward(
            &mut commands,
            k,
            &self.selected,
            &self.grad_projector_pre,
            &self.grad_selected,
            feature_rows,
        )?;
        commands.fill_zero_f32(
            &self.grad_vision_output,
            self.image_count * self.seq * self.dim,
        )?;
        for image in 0..self.image_count {
            commands.copy_f32_range(
                &self.grad_selected,
                image * self.features_per_image * self.dim,
                &self.grad_vision_output,
                (image * self.seq + usize::from(self.drop_cls)) * self.dim,
                self.features_per_image * self.dim,
            )?;
        }
        let mut grad = &self.grad_vision_output;
        for index in (0..self.layers.len()).rev() {
            let input = if index == 0 {
                &self.pre_output
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
        self.pre_norm.record_backward(
            &mut commands,
            k,
            &self.tokens,
            grad,
            &self.pre_mean,
            &self.pre_rstd,
            &self.grad_tokens,
            self.image_count * self.seq,
        )?;
        let mut push = TokenPush {
            images: self.image_count as u32,
            seq: self.seq as u32,
            dim: self.dim as u32,
            mode: 1,
        };
        for (mode, output, len) in [
            (
                1,
                &self.grad_patch_output,
                self.image_count * (self.seq - 1) * self.dim,
            ),
            (2, self.cls.grad_buffer()?, self.dim),
            (3, self.position.grad_buffer()?, self.seq * self.dim),
        ] {
            push.mode = mode;
            self.vk.tokens.record_dispatch(
                &mut commands,
                &[
                    &self.grad_tokens,
                    &self.cls.values,
                    &self.position.values,
                    output,
                ],
                bytemuck::bytes_of(&push),
                [div_ceil_u32(len, 256), 1, 1],
            )?;
        }
        self.patch.record_backward(
            &mut commands,
            k,
            &self.patches,
            &self.grad_patch_output,
            &self.grad_patches,
            self.image_count * (self.seq - 1),
        )?;
        let patch = PatchPush {
            images: self.image_count as u32,
            side: self.side as u32,
            patch: self.patch_size as u32,
            channels: self.channels as u32,
            inverse: 1,
        };
        self.vk.patch.record_dispatch(
            &mut commands,
            &[&self.grad_patches, &self.grad_pixels],
            bytemuck::bytes_of(&patch),
            [div_ceil_u32(pixel_values.len(), 256), 1, 1],
        )?;
        self.projector1.record_step(&mut commands, k, step, hyper)?;
        self.projector2.record_step(&mut commands, k, step, hyper)?;
        for layer in &self.layers {
            layer.step(&mut commands, k, step, hyper)?;
        }
        self.pre_norm.record_step(&mut commands, k, step, hyper)?;
        self.patch.record_step(&mut commands, k, step, hyper)?;
        self.cls.record_step(&mut commands, k, step, hyper)?;
        self.position.record_step(&mut commands, k, step, hyper)?;
        commands.submit()?;
        Ok(LlavaTrainResult {
            text: result.text,
            pixel_gradient: self.grad_pixels.read_f32(pixel_values.len())?,
            projected_gradient: result.modality_embedding_grad,
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
            format!("{prefix}.embeddings.class_embedding"),
            (vec![self.dim], self.cls.read()?),
        );
        values.insert(
            format!("{prefix}.embeddings.position_embedding.weight"),
            (vec![self.seq, self.dim], self.position.read()?),
        );
        values.insert(
            format!("{prefix}.embeddings.patch_embedding.weight"),
            (
                vec![self.dim, self.channels, self.patch_size, self.patch_size],
                self.patch.weight.read()?,
            ),
        );
        export_norm(
            &mut values,
            &format!("{prefix}.pre_layrnorm"),
            &self.pre_norm,
        )?;
        for layer in &self.layers {
            layer.export(&mut values)?;
        }
        export_linear(
            &mut values,
            &format!("{}.linear_1", self.projector_prefix),
            &self.projector1,
        )?;
        export_linear(
            &mut values,
            &format!("{}.linear_2", self.projector_prefix),
            &self.projector2,
        )?;
        let replacements = values
            .iter()
            .map(|(name, (shape, values))| (name.as_str(), shape.as_slice(), values.as_slice()))
            .collect::<Vec<_>>();
        let path = destination.as_ref().join("model.safetensors");
        crate::replace_f32_tensors(&path, &path, &replacements)
    }
}
