//! Released DeepSeek V4.1 ViT and spatial aligner. The retained training graph
//! stays on Vulkan. Patch vectors are channel-major, as in image_processor.py;
//! the patch embedding is Linear, not Conv2d. This module owns no text surrogate.
use super::vision::{cpu_avx2_vision_gemm, VisionKernels};
use super::*;

#[path = "deepseek_v41_image.rs"]
mod image;
pub use image::VulkanDeepseekV41Image;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DeepseekV41VisionConfig {
    pub vision_dim: usize,
    pub vision_n_heads: usize,
    pub vision_inter_dim: usize,
    pub vision_n_layers: usize,
    pub vision_patch_size: usize,
    pub vision_rope_theta: f32,
    pub vision_downsample_ratio: usize,
    pub dim: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DeepseekV41VisionTensor {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LayoutPush {
    rows: u32,
    dim: u32,
    heads: u32,
    mode: u32,
    height: u32,
    width: u32,
    ratio: u32,
    scale: f32,
}

struct Norm {
    parameter: VulkanLayerNorm,
    mean: GpuBuffer,
    rstd: GpuBuffer,
}

type Tensors = BTreeMap<String, DeepseekV41VisionTensor>;
type Tape = BTreeMap<&'static str, GpuBuffer>;

fn values<'a>(tensors: &'a Tensors, name: &str, shape: &[usize]) -> Result<&'a [f32]> {
    let t = tensors
        .get(name)
        .with_context(|| format!("missing V4.1 vision tensor {name}"))?;
    if t.shape != shape
        || t.values.len() != shape.iter().product::<usize>()
        || t.values.iter().any(|v| !v.is_finite())
    {
        bail!("invalid V4.1 vision tensor {name}: expected {shape:?}");
    }
    Ok(&t.values)
}

fn linear(
    device: &VulkanDevice,
    tensors: &Tensors,
    name: &str,
    input: usize,
    output: usize,
    bias: bool,
) -> Result<VulkanLinear> {
    let w = values(tensors, &format!("{name}.weight"), &[output, input])?;
    if bias {
        VulkanLinear::new(
            device,
            input,
            output,
            w,
            values(tensors, &format!("{name}.bias"), &[output])?,
        )
    } else {
        VulkanLinear::new_no_bias(device, input, output, w)
    }
}

impl Norm {
    fn new(
        device: &VulkanDevice,
        tensors: &Tensors,
        name: &str,
        rows: usize,
        dim: usize,
    ) -> Result<Self> {
        Ok(Self {
            parameter: VulkanLayerNorm::new_rms(
                device,
                dim,
                1e-6,
                values(tensors, &format!("{name}.weight"), &[dim])?,
                0.,
            )?
            .use_vector_cpu_rsqrt(true)
            .use_unfused_rms_products(true),
            mean: GpuBuffer::zeros_f32(device, rows)?,
            rstd: GpuBuffer::zeros_f32(device, rows)?,
        })
    }
    fn forward(
        &self,
        b: &mut vulkan::ComputeBatch,
        k: &TransformerKernels,
        x: &GpuBuffer,
        y: &GpuBuffer,
        rows: usize,
    ) -> Result<()> {
        self.parameter
            .record_forward(b, k, x, y, &self.mean, &self.rstd, rows)
    }
    fn backward(
        &self,
        b: &mut vulkan::ComputeBatch,
        k: &TransformerKernels,
        x: &GpuBuffer,
        g: &GpuBuffer,
        dx: &GpuBuffer,
        rows: usize,
    ) -> Result<()> {
        self.parameter
            .record_backward(b, k, x, g, &self.mean, &self.rstd, dx, rows)
    }
    fn backward_with_residual(
        &self,
        b: &mut vulkan::ComputeBatch,
        k: &TransformerKernels,
        x: &GpuBuffer,
        g: &GpuBuffer,
        residual: &GpuBuffer,
        branch: &GpuBuffer,
        dx: &GpuBuffer,
        rows: usize,
    ) -> Result<()> {
        self.parameter.record_backward_with_residual(
            b, k, x, g, &self.mean, &self.rstd, residual, branch, dx, rows,
        )
    }
}

struct Block {
    norm1: Norm,
    norm2: Norm,
    qkv: VulkanLinear,
    output: VulkanLinear,
    gate_up: VulkanLinear,
    down: VulkanLinear,
    tape: Tape,
}

fn tape(device: &VulkanDevice, fields: &[(&'static str, usize)]) -> Result<Tape> {
    fields
        .iter()
        .map(|(name, size)| Ok((*name, GpuBuffer::zeros_f32(device, *size)?)))
        .collect()
}

/// One image's retained graph, including right/bottom aligner padding. A caller
/// supplies the actual released processor's patches; host reads are diagnostic
/// or serialization boundaries only. The language trainer consumes projected().
pub struct VulkanDeepseekV41Vision {
    device: VulkanDevice,
    config: DeepseekV41VisionConfig,
    height: usize,
    width: usize,
    rows: usize,
    aligned_rows: usize,
    kernels: Box<TransformerKernels>,
    vk: VisionKernels,
    layout: vulkan::ComputeKernel,
    flash_forward: vulkan::ComputeKernel,
    flash_backward: vulkan::ComputeKernel,
    gemv: vulkan::ComputeKernel,
    aligner_input_grad: vulkan::ComputeKernel,
    final_rms_input_grad: vulkan::ComputeKernel,
    patch: VulkanLinear,
    blocks: Vec<Block>,
    norm: Norm,
    aligner1: VulkanLinear,
    aligner2: VulkanLinear,
    cos: GpuBuffer,
    sin: GpuBuffer,
    mask: GpuBuffer,
    dummy: GpuBuffer,
    tape: Tape,
    step: u32,
    tape_state: std::cell::Cell<u8>,
}

impl VulkanDeepseekV41Vision {
    pub fn from_tensors(
        device: VulkanDevice,
        config: DeepseekV41VisionConfig,
        height: usize,
        width: usize,
        tensors: &Tensors,
    ) -> Result<Self> {
        let _math = crate::deepseek_v41::native_avx2_math_scope();
        let c = &config;
        if [
            height,
            width,
            c.vision_dim,
            c.vision_n_heads,
            c.vision_inter_dim,
            c.vision_n_layers,
            c.vision_patch_size,
            c.vision_downsample_ratio,
            c.dim,
        ]
        .iter()
        .any(|&n| n > u32::MAX as usize)
        {
            bail!("V4.1 vision dimensions exceed device indexing");
        }
        if [
            height,
            width,
            c.vision_dim,
            c.vision_n_heads,
            c.vision_inter_dim,
            c.vision_n_layers,
            c.vision_patch_size,
            c.vision_downsample_ratio,
            c.dim,
        ]
        .contains(&0)
            || !c.vision_dim.is_multiple_of(c.vision_n_heads)
            || !(c.vision_dim / c.vision_n_heads).is_multiple_of(4)
            || !c.vision_rope_theta.is_finite()
            || c.vision_rope_theta <= 0.
        {
            bail!("invalid released V4.1 vision geometry");
        }
        let rows = height.checked_mul(width).context("vision rows overflow")?;
        let d = c.vision_dim;
        let inter = c.vision_inter_dim;
        let r = c.vision_downsample_ratio;
        let aligned_rows = height
            .div_ceil(r)
            .checked_mul(width.div_ceil(r))
            .context("aligner rows overflow")?;
        let patch_dim = 3usize
            .checked_mul(c.vision_patch_size)
            .and_then(|n| n.checked_mul(c.vision_patch_size))
            .context("patch dimension overflow")?;
        let aligned_dim = d
            .checked_mul(r)
            .and_then(|n| n.checked_mul(r))
            .context("aligner dimension overflow")?;
        for (n, m) in [
            (rows, d * 3),
            (rows, inter * 2),
            (aligned_rows, aligned_dim),
            (aligned_rows, c.dim),
            (rows, patch_dim),
        ] {
            if n.checked_mul(m).is_none_or(|v| v > u32::MAX as usize) {
                bail!("vision dispatch geometry overflow");
            }
        }
        let mut k = Box::new(TransformerKernels::new(&device)?);
        if aten_vector_width() == 8 {
            k.bias_grad = vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/vision_bias_grad_cpu_avx2.spv"),
                2,
                8,
            )?;
            k.linear_weight_grad = vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/vision_linear_weight_grad_cpu_avx2.spv"),
                3,
                12,
            )?;
            k.silu_backward = vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/silu_backward_cpu_avx2.spv"),
                3,
                4,
            )?;
        }
        let vk = VisionKernels::new(&device)?;
        let final_rms_input_grad = vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/vision_deepseek_v41_rms_input_grad.spv"),
            5,
            8,
        )?;
        let gemv = vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/vision_deepseek_v41_gemv.spv"),
            4,
            16,
        )?;
        let aligner_input_grad = vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/vision_deepseek_v41_aligner_input_grad.spv"),
            3,
            12,
        )?;
        let flash_forward = vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/vision_deepseek_v41_attention.spv"),
            4,
            16,
        )?;
        let flash_backward = vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/vision_deepseek_v41_attention_backward.spv"),
            7,
            16,
        )?;
        let layout = vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/vision_deepseek_v41_layout.spv"),
            4,
            32,
        )?;
        let mut blocks = Vec::new();
        for layer in 0..c.vision_n_layers {
            let prefix = format!("vision.blocks.{layer}");
            let fields = [
                ("n1", rows * d),
                ("qkv", rows * d * 3),
                ("q", rows * d),
                ("k", rows * d),
                ("v", rows * d),
                ("qr", rows * d),
                ("kr", rows * d),
                ("qs", rows * d),
                ("ks", rows * d),
                ("a", rows * d),
                ("o", rows * d),
                ("r", rows * d),
                ("n2", rows * d),
                ("gu", rows * inter * 2),
                ("g", rows * inter),
                ("u", rows * inter),
                ("ga", rows * inter),
                ("p", rows * inter),
                ("down", rows * d),
                ("out", rows * d),
                ("ddown", rows * d),
                ("dp", rows * inter),
                ("dga", rows * inter),
                ("du", rows * inter),
                ("dg", rows * inter),
                ("dgu", rows * inter * 2),
                ("dn2", rows * d),
                ("dr", rows * d),
                ("da", rows * d),
                ("dqs", rows * d),
                ("dks", rows * d),
                ("dv", rows * d),
                ("dqr", rows * d),
                ("dkr", rows * d),
                ("dq", rows * d),
                ("dk", rows * d),
                ("dqkv", rows * d * 3),
                ("dn1", rows * d),
                ("dx", rows * d),
            ];
            blocks.push(Block {
                norm1: Norm::new(&device, tensors, &format!("{prefix}.norm1"), rows, d)?,
                norm2: Norm::new(&device, tensors, &format!("{prefix}.norm2"), rows, d)?,
                qkv: linear(
                    &device,
                    tensors,
                    &format!("{prefix}.attn.wqkv"),
                    d,
                    d * 3,
                    true,
                )?,
                output: linear(&device, tensors, &format!("{prefix}.attn.wo"), d, d, true)?,
                gate_up: linear(
                    &device,
                    tensors,
                    &format!("{prefix}.mlp.w1"),
                    d,
                    inter * 2,
                    false,
                )?,
                down: linear(
                    &device,
                    tensors,
                    &format!("{prefix}.mlp.w2"),
                    inter,
                    d,
                    false,
                )?,
                tape: tape(&device, &fields)?,
            });
        }
        let half = (d / c.vision_n_heads) / 2;
        let frequencies = (0..half / 2)
            .map(|j| 1. / c.vision_rope_theta.powf((2 * j) as f32 / half as f32))
            .collect::<Vec<_>>();
        let frequencies = GpuBuffer::from_f32(&device, &frequencies)?;
        let cos = GpuBuffer::zeros_f32(&device, rows * half)?;
        let sin = GpuBuffer::zeros_f32(&device, rows * half)?;
        let table = vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../../shaders/vision_deepseek_v41_rope_tables.spv"),
            3,
            12,
        )?;
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        table.record_dispatch(
            &mut commands,
            &[&frequencies, &cos, &sin],
            bytemuck::cast_slice(&[rows as u32, width as u32, half as u32]),
            [div_ceil_u32(rows * half, 256), 1, 1],
        )?;
        commands.submit()?;
        Ok(Self {
            patch: linear(
                &device,
                tensors,
                "vision.patch_embed.proj",
                patch_dim,
                d,
                true,
            )?,
            norm: Norm::new(&device, tensors, "vision.norm", rows, d)?,
            aligner1: linear(&device, tensors, "aligner.w1", aligned_dim, c.dim, true)?,
            aligner2: linear(&device, tensors, "aligner.w2", c.dim, c.dim, true)?,
            tape: tape(
                &device,
                &[
                    ("patches", rows * patch_dim),
                    ("embedded", rows * d),
                    ("features", rows * d),
                    ("unfold", aligned_rows * aligned_dim),
                    ("pre", aligned_rows * c.dim),
                    ("act", aligned_rows * c.dim),
                    ("projected", aligned_rows * c.dim),
                    ("dact", aligned_rows * c.dim),
                    ("dpre", aligned_rows * c.dim),
                    ("dunfold", aligned_rows * aligned_dim),
                    ("dfeatures", rows * d),
                    ("dlast", rows * d),
                    ("dpatches", rows * patch_dim),
                ],
            )?,
            mask: GpuBuffer::from_f32(&device, &vec![1.; rows])?,
            dummy: GpuBuffer::zeros_f32(&device, 1)?,
            flash_forward,
            flash_backward,
            gemv,
            aligner_input_grad,
            final_rms_input_grad,
            device,
            config,
            height,
            width,
            rows,
            aligned_rows,
            kernels: k,
            vk,
            layout,
            blocks,
            cos,
            sin,
            step: 0,
            tape_state: std::cell::Cell::new(0),
        })
    }

    fn layout(
        &self,
        b: &mut vulkan::ComputeBatch,
        x: &GpuBuffer,
        a: &GpuBuffer,
        e: &GpuBuffer,
        y: &GpuBuffer,
        rows: usize,
        dim: usize,
        mode: u32,
        part: u32,
        scale: f32,
    ) -> Result<()> {
        self.layout.record_dispatch(
            b,
            &[x, a, e, y],
            bytemuck::bytes_of(&LayoutPush {
                rows: rows as u32,
                dim: dim as u32,
                heads: if mode == 5 {
                    2
                } else {
                    self.config.vision_n_heads as u32
                },
                mode,
                height: if mode == 5 { part } else { self.height as u32 },
                width: self.width as u32,
                ratio: self.config.vision_downsample_ratio as u32,
                scale,
            }),
            [div_ceil_u32(rows * dim, 256), 1, 1],
        )
    }
    fn scale(
        &self,
        b: &mut vulkan::ComputeBatch,
        x: &GpuBuffer,
        y: &GpuBuffer,
        scale: f32,
    ) -> Result<()> {
        self.layout(
            b,
            x,
            &self.dummy,
            &self.dummy,
            y,
            self.rows,
            self.config.vision_dim,
            0,
            0,
            scale,
        )
    }
    fn rope(
        &self,
        b: &mut vulkan::ComputeBatch,
        x: &GpuBuffer,
        y: &GpuBuffer,
        inverse: bool,
    ) -> Result<()> {
        self.layout(
            b,
            x,
            &self.cos,
            &self.sin,
            y,
            self.rows,
            self.config.vision_dim,
            if inverse { 2 } else { 1 },
            0,
            1.,
        )
    }
    fn linear_forward(
        &self,
        b: &mut vulkan::ComputeBatch,
        l: &VulkanLinear,
        x: &GpuBuffer,
        y: &GpuBuffer,
        rows: usize,
    ) -> Result<()> {
        if aten_vector_width() == 8 && rows == 1 {
            return self.gemv.record_dispatch(
                b,
                &[x, &l.weight.values, &l.bias.values, y],
                bytemuck::cast_slice(&[
                    l.input_dim as u32,
                    l.output_dim as u32,
                    0,
                    u32::from(l.has_bias),
                ]),
                [div_ceil_u32(l.output_dim, 256), 1, 1],
            );
        }
        if aten_vector_width() == 8
            && (rows <= 3 || cpu_avx2_vision_gemm(rows, l.input_dim, l.output_dim, 8))
        {
            self.vk.cpu_avx2_linear.record_dispatch(
                b,
                &[x, &l.weight.values, &l.bias.values, y],
                bytemuck::cast_slice(&[
                    rows as u32,
                    l.input_dim as u32,
                    l.output_dim as u32,
                    u32::from(l.has_bias),
                ]),
                [div_ceil_u32(l.output_dim, 16), div_ceil_u32(rows, 16), 1],
            )
        } else {
            l.record_forward(b, &self.kernels, x, y, rows)
        }
    }
    fn linear_backward(
        &self,
        b: &mut vulkan::ComputeBatch,
        l: &VulkanLinear,
        x: &GpuBuffer,
        g: &GpuBuffer,
        dx: &GpuBuffer,
        rows: usize,
    ) -> Result<()> {
        self.linear_backward_layout(b, l, x, g, dx, rows, false)
    }
    fn linear_backward_layout(
        &self,
        b: &mut vulkan::ComputeBatch,
        l: &VulkanLinear,
        x: &GpuBuffer,
        g: &GpuBuffer,
        dx: &GpuBuffer,
        rows: usize,
        column_major: bool,
    ) -> Result<()> {
        l.record_backward(b, &self.kernels, x, g, dx, rows)?;
        if aten_vector_width() == 8 && (rows == 1 || (rows == 2 && column_major)) {
            self.aligner_input_grad.record_dispatch(
                b,
                &[g, &l.weight.values, dx],
                bytemuck::cast_slice(&[rows as u32, l.input_dim as u32, l.output_dim as u32]),
                [div_ceil_u32(l.input_dim, 16), div_ceil_u32(rows, 16), 1],
            )?;
        } else if aten_vector_width() == 8 && rows <= 3 {
            self.vk.cpu_avx2_input_grad.record_dispatch(
                b,
                &[g, &l.weight.values, &l.bias.values, dx],
                bytemuck::cast_slice(&[rows as u32, l.output_dim as u32, l.input_dim as u32, 0]),
                [div_ceil_u32(l.input_dim, 16), div_ceil_u32(rows, 16), 1],
            )?;
        }
        Ok(())
    }
    fn attention_push(&self) -> AttentionPush {
        let mut p = AttentionPush::zeroed();
        p.batch_size = 1;
        p.query_seq_len = self.rows as u32;
        p.key_value_seq_len = self.rows as u32;
        p.query_hidden_size = self.config.vision_dim as u32;
        p.value_hidden_size = p.query_hidden_size;
        p.num_heads = self.config.vision_n_heads as u32;
        p.num_key_value_heads = p.num_heads;
        p.scale = 1.;
        p.value_scale = 1.;
        p.unfused_qk = 1;
        p.kv_page_size = 1;
        p.kv_page_table_stride = 1;
        p
    }
    fn native_cpu_flash(&self) -> bool {
        aten_vector_width() == 8
            && matches!(self.rows, 4 | 6)
            && self.config.vision_dim == 16
            && self.config.vision_n_heads == 2
    }
    fn native_flash_push(&self) -> [u32; 4] {
        [
            self.rows as u32,
            self.config.vision_dim as u32,
            self.config.vision_n_heads as u32,
            ((1. / (self.config.vision_dim / self.config.vision_n_heads) as f64).sqrt() as f32)
                .to_bits(),
        ]
    }
    fn qkv(&self, b: &mut vulkan::ComputeBatch, t: &Tape, inverse: bool) -> Result<()> {
        let push = QkvPush {
            rows: self.rows as u32,
            query_hidden_size: self.config.vision_dim as u32,
            kv_hidden_size: self.config.vision_dim as u32,
        };
        let (kernel, buffers) = if inverse {
            (
                &self.kernels.qkv_concat,
                vec![&t["dq"], &t["dk"], &t["dv"], &t["dqkv"]],
            )
        } else {
            (
                &self.kernels.qkv_split,
                vec![&t["qkv"], &t["q"], &t["k"], &t["v"]],
            )
        };
        kernel.record_dispatch(
            b,
            &buffers,
            bytemuck::bytes_of(&push),
            [
                div_ceil_u32(self.config.vision_dim, 16),
                div_ceil_u32(self.rows, 16),
                1,
            ],
        )
    }
    pub fn projected(&self) -> &GpuBuffer {
        &self.tape["projected"]
    }
    pub fn patch_gradient(&self) -> &GpuBuffer {
        &self.tape["dpatches"]
    }
    pub fn projected_rows(&self) -> usize {
        self.aligned_rows
    }
    pub fn config(&self) -> &DeepseekV41VisionConfig {
        &self.config
    }

    pub fn forward(&self, patches: &[f32]) -> Result<()> {
        let _math = crate::deepseek_v41::native_avx2_math_scope();
        if patches.len() != self.rows * self.patch.input_dim
            || patches.iter().any(|v| !v.is_finite())
        {
            bail!("invalid V4.1 image patches");
        }
        self.tape_state.set(0);
        let mut b = vulkan::ComputeBatch::new(&self.device)?;
        b.enable_watchdog_submission_slicing();
        b.upload_f32(&self.tape["patches"], patches)?;
        self.linear_forward(
            &mut b,
            &self.patch,
            &self.tape["patches"],
            &self.tape["embedded"],
            self.rows,
        )?;
        let d = self.config.vision_dim;
        let inter = self.config.vision_inter_dim;
        let scale = (1.0 / (d / self.config.vision_n_heads) as f64)
            .sqrt()
            .sqrt() as f32;
        for (i, block) in self.blocks.iter().enumerate() {
            let t = &block.tape;
            let x = if i == 0 {
                &self.tape["embedded"]
            } else {
                &self.blocks[i - 1].tape["out"]
            };
            block
                .norm1
                .forward(&mut b, &self.kernels, x, &t["n1"], self.rows)?;
            self.linear_forward(&mut b, &block.qkv, &t["n1"], &t["qkv"], self.rows)?;
            self.qkv(&mut b, t, false)?;
            self.rope(&mut b, &t["q"], &t["qr"], false)?;
            self.rope(&mut b, &t["k"], &t["kr"], false)?;
            if self.native_cpu_flash() {
                self.flash_forward.record_dispatch(
                    &mut b,
                    &[&t["qr"], &t["kr"], &t["v"], &t["a"]],
                    bytemuck::cast_slice(&self.native_flash_push()),
                    [self.rows as u32, self.config.vision_n_heads as u32, 1],
                )?;
            } else {
                self.scale(&mut b, &t["qr"], &t["qs"], scale)?;
                self.scale(&mut b, &t["kr"], &t["ks"], scale)?;
                let p = self.attention_push();
                self.vk.attention_forward.record_dispatch(
                    &mut b,
                    &[
                        &t["qs"],
                        &t["ks"],
                        &t["v"],
                        &self.mask,
                        &self.mask,
                        &self.dummy,
                        &t["a"],
                        &self.dummy,
                        &self.dummy,
                        &self.mask,
                        &self.mask,
                    ],
                    bytemuck::bytes_of(&p),
                    [p.query_seq_len, p.num_heads, 1],
                )?;
            }
            self.linear_forward(&mut b, &block.output, &t["a"], &t["o"], self.rows)?;
            record_add(&mut b, &self.kernels, x, &t["o"], &t["r"], self.rows * d)?;
            block
                .norm2
                .forward(&mut b, &self.kernels, &t["r"], &t["n2"], self.rows)?;
            self.linear_forward(&mut b, &block.gate_up, &t["n2"], &t["gu"], self.rows)?;
            for (part, name) in [(0, "g"), (1, "u")] {
                self.layout(
                    &mut b,
                    &t["gu"],
                    &self.dummy,
                    &self.dummy,
                    &t[name],
                    self.rows,
                    inter,
                    5,
                    part,
                    1.,
                )?;
            }
            self.kernels.silu_forward.record_dispatch(
                &mut b,
                &[&t["g"], &t["ga"]],
                bytemuck::bytes_of(&LenPush {
                    len: (self.rows * inter) as u32,
                }),
                [div_ceil_u32(self.rows * inter, 256), 1, 1],
            )?;
            self.kernels.elementwise_mul_forward.record_dispatch(
                &mut b,
                &[&t["ga"], &t["u"], &t["p"]],
                bytemuck::bytes_of(&LenPush {
                    len: (self.rows * inter) as u32,
                }),
                [div_ceil_u32(self.rows * inter, 256), 1, 1],
            )?;
            self.linear_forward(&mut b, &block.down, &t["p"], &t["down"], self.rows)?;
            record_add(
                &mut b,
                &self.kernels,
                &t["r"],
                &t["down"],
                &t["out"],
                self.rows * d,
            )?;
        }
        self.norm.forward(
            &mut b,
            &self.kernels,
            &self.blocks.last().unwrap().tape["out"],
            &self.tape["features"],
            self.rows,
        )?;
        self.layout(
            &mut b,
            &self.tape["features"],
            &self.dummy,
            &self.dummy,
            &self.tape["unfold"],
            self.aligned_rows,
            self.aligner1.input_dim,
            3,
            0,
            1.,
        )?;
        self.linear_forward(
            &mut b,
            &self.aligner1,
            &self.tape["unfold"],
            &self.tape["pre"],
            self.aligned_rows,
        )?;
        let n = self.aligned_rows * self.config.dim;
        self.vk.gelu_cpu_forward.record_dispatch(
            &mut b,
            &[&self.tape["pre"], &self.tape["act"]],
            bytemuck::bytes_of(&LenPush { len: n as u32 }),
            [div_ceil_u32(n, 256), 1, 1],
        )?;
        self.linear_forward(
            &mut b,
            &self.aligner2,
            &self.tape["act"],
            self.projected(),
            self.aligned_rows,
        )?;
        b.submit()?;
        self.tape_state.set(1);
        Ok(())
    }

    pub fn backward(&self, gradient: &GpuBuffer) -> Result<()> {
        let _math = crate::deepseek_v41::native_avx2_math_scope();
        if self.tape_state.get() != 1 {
            bail!("V4.1 vision backward requires a retained forward");
        }
        if !gradient.belongs_to_device(&self.device)
            || gradient.capacity_bytes() < self.aligned_rows * self.config.dim * 4
        {
            bail!("invalid V4.1 projected gradient device or capacity");
        }
        self.tape_state.set(0);
        let mut b = vulkan::ComputeBatch::new(&self.device)?;
        b.enable_watchdog_submission_slicing();
        let t = &self.tape;
        let n = self.aligned_rows * self.config.dim;
        self.linear_backward(
            &mut b,
            &self.aligner2,
            &t["act"],
            gradient,
            &t["dact"],
            self.aligned_rows,
        )?;
        self.vk.gelu_cpu_backward.record_dispatch(
            &mut b,
            &[&t["pre"], &t["dact"], &t["dpre"]],
            bytemuck::bytes_of(&LenPush { len: n as u32 }),
            [div_ceil_u32(n, 256), 1, 1],
        )?;
        self.linear_backward_layout(
            &mut b,
            &self.aligner1,
            &t["unfold"],
            &t["dpre"],
            &t["dunfold"],
            self.aligned_rows,
            true,
        )?;
        let d = self.config.vision_dim;
        let inter = self.config.vision_inter_dim;
        self.layout(
            &mut b,
            &t["dunfold"],
            &self.dummy,
            &self.dummy,
            &t["dfeatures"],
            self.rows,
            d,
            4,
            0,
            1.,
        )?;
        self.norm.backward(
            &mut b,
            &self.kernels,
            &self.blocks.last().unwrap().tape["out"],
            &t["dfeatures"],
            &t["dlast"],
            self.rows,
        )?;
        if self.native_cpu_flash() {
            self.final_rms_input_grad.record_dispatch(
                &mut b,
                &[
                    &t["dfeatures"],
                    &self.blocks.last().unwrap().tape["out"],
                    &self.norm.parameter.weight.values,
                    &self.norm.rstd,
                    &t["dlast"],
                ],
                bytemuck::cast_slice(&[self.rows as u32, d as u32]),
                [div_ceil_u32(self.rows, 64), 1, 1],
            )?;
        }
        let scale = (1.0 / (d / self.config.vision_n_heads) as f64)
            .sqrt()
            .sqrt() as f32;
        for (i, block) in self.blocks.iter().enumerate().rev() {
            let t = &block.tape;
            let x = if i == 0 {
                &self.tape["embedded"]
            } else {
                &self.blocks[i - 1].tape["out"]
            };
            let dy = if i + 1 == self.blocks.len() {
                &self.tape["dlast"]
            } else {
                &self.blocks[i + 1].tape["dx"]
            };
            self.linear_backward(&mut b, &block.down, &t["p"], dy, &t["dp"], self.rows)?;
            self.kernels.elementwise_mul_backward.record_dispatch(
                &mut b,
                &[&t["dp"], &t["ga"], &t["u"], &t["dga"], &t["du"]],
                bytemuck::bytes_of(&LenPush {
                    len: (self.rows * inter) as u32,
                }),
                [div_ceil_u32(self.rows * inter, 256), 1, 1],
            )?;
            self.kernels.silu_backward.record_dispatch(
                &mut b,
                &[&t["dga"], &t["g"], &t["dg"]],
                bytemuck::bytes_of(&LenPush {
                    len: (self.rows * inter) as u32,
                }),
                [div_ceil_u32(self.rows * inter, 256), 1, 1],
            )?;
            self.layout(
                &mut b,
                &t["dg"],
                &t["du"],
                &self.dummy,
                &t["dgu"],
                self.rows,
                inter * 2,
                6,
                0,
                1.,
            )?;
            self.linear_backward(
                &mut b,
                &block.gate_up,
                &t["n2"],
                &t["dgu"],
                &t["dn2"],
                self.rows,
            )?;
            block.norm2.backward_with_residual(
                &mut b,
                &self.kernels,
                &t["r"],
                &t["dn2"],
                dy,
                &t["ddown"],
                &t["dr"],
                self.rows,
            )?;
            self.linear_backward(
                &mut b,
                &block.output,
                &t["a"],
                &t["dr"],
                &t["da"],
                self.rows,
            )?;
            if self.native_cpu_flash() {
                self.flash_backward.record_dispatch(
                    &mut b,
                    &[
                        &t["qr"], &t["kr"], &t["v"], &t["da"], &t["dqr"], &t["dkr"], &t["dv"],
                    ],
                    bytemuck::cast_slice(&self.native_flash_push()),
                    [self.config.vision_n_heads as u32, 1, 1],
                )?;
            } else {
                let p = self.attention_push();
                self.vk.attention_backward.record_dispatch(
                    &mut b,
                    &[
                        &t["qs"],
                        &t["ks"],
                        &t["v"],
                        &self.mask,
                        &self.mask,
                        &self.dummy,
                        &t["da"],
                        &t["dqs"],
                        &t["dks"],
                        &t["dv"],
                        &self.dummy,
                        &self.dummy,
                        &self.dummy,
                        &self.dummy,
                        &self.mask,
                    ],
                    bytemuck::bytes_of(&p.as_backward_push(0, 1)),
                    [p.num_heads, 1, 1],
                )?;
                self.scale(&mut b, &t["dqs"], &t["dqr"], scale)?;
                self.scale(&mut b, &t["dks"], &t["dkr"], scale)?;
            }
            self.rope(&mut b, &t["dqr"], &t["dq"], true)?;
            self.rope(&mut b, &t["dkr"], &t["dk"], true)?;
            self.qkv(&mut b, t, true)?;
            self.linear_backward(
                &mut b, &block.qkv, &t["n1"], &t["dqkv"], &t["dn1"], self.rows,
            )?;
            block.norm1.backward_with_residual(
                &mut b,
                &self.kernels,
                x,
                &t["dn1"],
                &t["dr"],
                &t["dqr"],
                &t["dx"],
                self.rows,
            )?;
        }
        self.linear_backward(
            &mut b,
            &self.patch,
            &self.tape["patches"],
            &self.blocks[0].tape["dx"],
            self.patch_gradient(),
            self.rows,
        )?;
        b.submit()?;
        self.tape_state.set(2);
        Ok(())
    }

    pub fn apply_gradients(&mut self, hyper: crate::CpuAdamWHyperParams) -> Result<()> {
        let _math = crate::deepseek_v41::native_avx2_math_scope();
        if self.tape_state.get() != 2 {
            bail!("V4.1 vision update requires fresh gradients");
        }
        hyper.validate()?;
        *self
            .kernels
            .cpu_adamw_policy
            .lock()
            .map_err(|_| anyhow!("vision AdamW policy lock poisoned"))? = Some(hyper);
        let step = self
            .step
            .checked_add(1)
            .context("vision AdamW step overflow")?;
        let mut b = vulkan::ComputeBatch::new(&self.device)?;
        b.enable_watchdog_submission_slicing();
        let h = AdamWHyperParams::default();
        for l in [&self.patch, &self.aligner1, &self.aligner2] {
            l.record_step(&mut b, &self.kernels, step, h)?;
        }
        self.norm
            .parameter
            .record_step(&mut b, &self.kernels, step, h)?;
        for block in &self.blocks {
            for l in [&block.qkv, &block.output, &block.gate_up, &block.down] {
                l.record_step(&mut b, &self.kernels, step, h)?;
            }
            block
                .norm1
                .parameter
                .record_step(&mut b, &self.kernels, step, h)?;
            block
                .norm2
                .parameter
                .record_step(&mut b, &self.kernels, step, h)?;
        }
        b.submit()?;
        self.step = step;
        self.tape_state.set(0);
        Ok(())
    }

    pub fn parameters(&self) -> Result<Tensors> {
        self.parameter_values(false)
    }
    pub fn gradients(&self) -> Result<Tensors> {
        self.parameter_values(true)
    }
    fn parameter_values(&self, grad: bool) -> Result<Tensors> {
        let mut result = Tensors::new();
        let mut add = |name: String, p: &TrainableParameter, shape: Vec<usize>| -> Result<()> {
            result.insert(
                name,
                DeepseekV41VisionTensor {
                    shape,
                    values: if grad {
                        p.grad_buffer()?.read_f32(p.len)?
                    } else {
                        p.read()?
                    },
                },
            );
            Ok(())
        };
        let mut lines = vec![
            ("vision.patch_embed.proj".to_owned(), &self.patch),
            ("aligner.w1".to_owned(), &self.aligner1),
            ("aligner.w2".to_owned(), &self.aligner2),
        ];
        for (i, block) in self.blocks.iter().enumerate() {
            let p = format!("vision.blocks.{i}");
            lines.extend([
                (format!("{p}.attn.wqkv"), &block.qkv),
                (format!("{p}.attn.wo"), &block.output),
                (format!("{p}.mlp.w1"), &block.gate_up),
                (format!("{p}.mlp.w2"), &block.down),
            ]);
        }
        for (name, l) in lines {
            add(
                format!("{name}.weight"),
                &l.weight,
                vec![l.output_dim, l.input_dim],
            )?;
            if l.has_bias {
                add(format!("{name}.bias"), &l.bias, vec![l.output_dim])?;
            }
        }
        for (i, block) in self.blocks.iter().enumerate() {
            for (name, norm) in [("norm1", &block.norm1), ("norm2", &block.norm2)] {
                add(
                    format!("vision.blocks.{i}.{name}.weight"),
                    &norm.parameter.weight,
                    vec![self.config.vision_dim],
                )?;
            }
        }
        add(
            "vision.norm.weight".to_owned(),
            &self.norm.parameter.weight,
            vec![self.config.vision_dim],
        )?;
        Ok(result)
    }

    pub fn diagnostic_stages(&self) -> Result<BTreeMap<String, Vec<f32>>> {
        let mut result = BTreeMap::new();
        for (name, b) in &self.tape {
            result.insert(name.to_string(), b.read_f32(b.capacity_bytes() / 4)?);
        }
        for (i, block) in self.blocks.iter().enumerate() {
            for (name, b) in &block.tape {
                result.insert(
                    format!("blocks.{i}.{name}"),
                    b.read_f32(b.capacity_bytes() / 4)?,
                );
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
#[path = "deepseek_v41_vision_tests.rs"]
mod tests;
