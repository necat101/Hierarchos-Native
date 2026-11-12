//! Native image-span merge and trainable delimiters around the released ViT.
use super::*;

#[path = "deepseek_v41_image_state.rs"]
mod state;

const MARKERS: [&str; 3] = ["image_start", "image_end", "image_newline"];
struct RetainedMerge {
    rows: usize,
    hc: usize,
    map: GpuBuffer,
    feature_rows: GpuBuffer,
    output: GpuBuffer,
}

/// One complete image span in a native V4.1 prefill. The image may occur in any
/// sample of the batch. Additional spans fail explicitly until a shared-weight
/// multi-image tape is admitted. Image feature and delimiter training use the
/// same retained language loss and update count.
pub struct VulkanDeepseekV41Image {
    vision: VulkanDeepseekV41Vision,
    markers: [TrainableParameter; 3],
    merge: vulkan::ComputeKernel,
    merge_backward: vulkan::ComputeKernel,
    hc_reduce: vulkan::ComputeKernel,
    retained: Option<RetainedMerge>,
    backward_ready: bool,
}

impl DeepseekV41VisionConfig {
    pub fn from_source_config(config: &serde_json::Value) -> Result<Self> {
        let v = &config["vision_config"];
        let number = |name: &str| -> Result<usize> {
            usize::try_from(
                v[name]
                    .as_u64()
                    .with_context(|| format!("missing native V4.1 vision {name}"))?,
            )
            .context("native vision dimension overflow")
        };
        Ok(Self {
            vision_dim: number("hidden_size")?,
            vision_n_heads: number("num_attention_heads")?,
            vision_inter_dim: number("intermediate_size")?,
            vision_n_layers: number("num_hidden_layers")?,
            vision_patch_size: number("patch_size")?,
            vision_downsample_ratio: number("downsample_ratio")?,
            vision_rope_theta: v["rope_theta"]
                .as_f64()
                .context("missing native vision rope_theta")? as f32,
            dim: usize::try_from(
                config["text_config"]["hidden_size"]
                    .as_u64()
                    .context("missing V4.1 text width")?,
            )?,
        })
    }
}

impl VulkanDeepseekV41Image {
    pub fn precision_mode(&self) -> &'static str {
        self.vision.kernels.wmma_mode.label()
    }

    pub fn from_checkpoint(
        device: VulkanDevice,
        checkpoint: &mut crate::deepseek_v41::DeepseekV41Checkpoint,
        height: usize,
        width: usize,
    ) -> Result<Self> {
        let c = DeepseekV41VisionConfig::from_source_config(&checkpoint.config.source_config)?;
        let d = c.vision_dim;
        let inter = c.vision_inter_dim;
        let text = c.dim;
        let patch = 3 * c.vision_patch_size * c.vision_patch_size;
        let mut shapes = BTreeMap::from([
            ("vision.patch_embed.proj.weight".to_owned(), vec![d, patch]),
            ("vision.patch_embed.proj.bias".to_owned(), vec![d]),
            ("vision.norm.weight".to_owned(), vec![d]),
            (
                "aligner.w1.weight".to_owned(),
                vec![
                    text,
                    d * c.vision_downsample_ratio * c.vision_downsample_ratio,
                ],
            ),
            ("aligner.w1.bias".to_owned(), vec![text]),
            ("aligner.w2.weight".to_owned(), vec![text, text]),
            ("aligner.w2.bias".to_owned(), vec![text]),
        ]);
        for i in 0..c.vision_n_layers {
            let p = format!("vision.blocks.{i}");
            for (name, shape) in [
                ("norm1.weight", vec![d]),
                ("norm2.weight", vec![d]),
                ("attn.wqkv.weight", vec![3 * d, d]),
                ("attn.wqkv.bias", vec![3 * d]),
                ("attn.wo.weight", vec![d, d]),
                ("attn.wo.bias", vec![d]),
                ("mlp.w1.weight", vec![2 * inter, d]),
                ("mlp.w2.weight", vec![d, inter]),
            ] {
                shapes.insert(format!("{p}.{name}"), shape);
            }
        }
        for name in MARKERS {
            shapes.insert(name.to_owned(), vec![text]);
        }
        let tensors = shapes
            .into_iter()
            .map(|(name, shape)| {
                let values = checkpoint.read_tensor(&name, &shape)?;
                Ok((name, DeepseekV41VisionTensor { shape, values }))
            })
            .collect::<Result<Tensors>>()?;
        Self::from_tensors(device, c, height, width, &tensors)
    }
    pub fn from_tensors(
        device: VulkanDevice,
        c: DeepseekV41VisionConfig,
        height: usize,
        width: usize,
        tensors: &Tensors,
    ) -> Result<Self> {
        let _math = crate::deepseek_v41::native_avx2_math_scope();
        let dim = c.dim;
        let markers = [
            TrainableParameter::new(&device, values(tensors, MARKERS[0], &[dim])?, false)?,
            TrainableParameter::new(&device, values(tensors, MARKERS[1], &[dim])?, false)?,
            TrainableParameter::new(&device, values(tensors, MARKERS[2], &[dim])?, false)?,
        ];
        Ok(Self {
            merge: vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/vision_deepseek_v41_merge.spv"),
                7,
                12,
            )?,
            merge_backward: vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/vision_deepseek_v41_merge_backward.spv"),
                7,
                12,
            )?,
            hc_reduce: vulkan::ComputeKernel::new(
                &device,
                include_bytes!("../../shaders/transformer_deepseek_v41_hc_reduce.spv"),
                2,
                12,
            )?,
            vision: VulkanDeepseekV41Vision::from_tensors(device, c, height, width, tensors)?,
            markers,
            retained: None,
            backward_ready: false,
        })
    }
    /// `start` is the flattened batch/sequence offset of IMAGE_START. Token
    /// types follow the released processor: TEXT=-1, START=0, IMAGE=1, NL=2, END=3.
    pub fn forward_merge(
        &mut self,
        base: &GpuBuffer,
        patches: &[f32],
        types: &[i32],
        start: usize,
        seq_len: usize,
        hc: usize,
    ) -> Result<GpuBuffer> {
        let dim = self.vision.config.dim;
        let rows = types.len();
        let count = rows
            .checked_mul(hc)
            .and_then(|n| n.checked_mul(dim))
            .context("native image stream overflow")?;
        if rows == 0
            || rows > 65_535
            || hc == 0
            || hc > u32::MAX as usize
            || count > u32::MAX as usize
            || seq_len == 0
            || !rows.is_multiple_of(seq_len)
            || base.capacity_bytes() < count * 4
            || !base.belongs_to_device(&self.vision.device)
        {
            bail!("invalid native V4.1 image embedding stream");
        }
        let r = self.vision.config.vision_downsample_ratio;
        let h = self.vision.height.div_ceil(r);
        let w = self.vision.width.div_ceil(r);
        let mut expected = vec![0];
        for _ in 0..h {
            expected.extend(std::iter::repeat_n(1, w));
            expected.push(2);
        }
        expected.push(3);
        let end = start
            .checked_add(expected.len())
            .context("image span overflow")?;
        if end > rows
            || start / seq_len != (end - 1) / seq_len
            || types[start..end] != expected
            || types[..start].iter().chain(&types[end..]).any(|&t| t != -1)
        {
            bail!("native V4.1 currently requires one complete released image span inside a prefill sample");
        }
        self.retained = None;
        self.backward_ready = false;
        self.vision.forward(patches)?;
        let mut map = vec![u32::MAX; rows];
        let mut feature_rows = Vec::new();
        for (offset, &kind) in expected.iter().enumerate() {
            let row = start + offset;
            map[row] = match kind {
                0 => 0,
                3 => 1,
                2 => 2,
                1 => {
                    let k = feature_rows.len() as u32;
                    feature_rows.push(row as u32);
                    3 + k
                }
                _ => unreachable!(),
            };
        }
        if feature_rows.len() != self.vision.aligned_rows {
            bail!("native aligner/span feature count mismatch");
        }
        let map = GpuBuffer::from_u32(&self.vision.device, &map)?;
        let feature_rows = GpuBuffer::from_u32(&self.vision.device, &feature_rows)?;
        let output = GpuBuffer::zeros_f32(&self.vision.device, count)?;
        let mut b = vulkan::ComputeBatch::new(&self.vision.device)?;
        self.merge.record_dispatch(
            &mut b,
            &[
                base,
                self.vision.projected(),
                &self.markers[0].values,
                &self.markers[1].values,
                &self.markers[2].values,
                &map,
                &output,
            ],
            bytemuck::cast_slice(&[rows as u32, hc as u32, dim as u32]),
            [div_ceil_u32(count, 256), 1, 1],
        )?;
        b.submit()?;
        self.retained = Some(RetainedMerge {
            rows,
            hc,
            map,
            feature_rows,
            output: output.clone(),
        });
        Ok(output)
    }
    pub fn backward(&mut self, stream_gradient: &GpuBuffer) -> Result<()> {
        if self.backward_ready {
            bail!("native image gradients have already been computed");
        }
        let meta = self
            .retained
            .as_ref()
            .context("native image backward requires a retained merge")?;
        let dim = self.vision.config.dim;
        let features = self.vision.aligned_rows;
        if stream_gradient.capacity_bytes() < meta.rows * meta.hc * dim * 4
            || !stream_gradient.belongs_to_device(&self.vision.device)
        {
            bail!("native image stream adjoint geometry/device mismatch");
        }
        let merged = GpuBuffer::zeros_f32(&self.vision.device, meta.rows * dim)?;
        let projected = GpuBuffer::zeros_f32(&self.vision.device, features * dim)?;
        let mut b = vulkan::ComputeBatch::new(&self.vision.device)?;
        self.hc_reduce.record_dispatch(
            &mut b,
            &[stream_gradient, &merged],
            bytemuck::cast_slice(&[meta.rows as u32, meta.hc as u32, dim as u32]),
            [div_ceil_u32(meta.rows * dim, 256), 1, 1],
        )?;
        self.merge_backward.record_dispatch(
            &mut b,
            &[
                &merged,
                &meta.map,
                &meta.feature_rows,
                &projected,
                self.markers[0].grad_buffer()?,
                self.markers[1].grad_buffer()?,
                self.markers[2].grad_buffer()?,
            ],
            bytemuck::cast_slice(&[meta.rows as u32, features as u32, dim as u32]),
            [div_ceil_u32(features.max(1) * dim, 256), 1, 1],
        )?;
        b.submit()?;
        self.vision.backward(&projected)?;
        self.backward_ready = true;
        Ok(())
    }
    pub fn apply_gradients(&mut self, hyper: crate::CpuAdamWHyperParams) -> Result<()> {
        let _math = crate::deepseek_v41::native_avx2_math_scope();
        if !self.backward_ready {
            bail!("native image update requires fresh joint gradients");
        }
        self.vision.apply_gradients(hyper)?;
        let mut b = vulkan::ComputeBatch::new(&self.vision.device)?;
        for marker in &self.markers {
            marker.record_step(
                &mut b,
                &self.vision.kernels,
                self.vision.step,
                hyper.as_f32(),
            )?;
        }
        b.submit()?;
        self.retained = None;
        self.backward_ready = false;
        Ok(())
    }
    pub fn step(&self) -> u32 {
        self.vision.step
    }
    pub fn gradients_ready(&self) -> bool {
        self.backward_ready
    }
    pub fn injected_stream(&self) -> Result<Vec<f32>> {
        let m = self
            .retained
            .as_ref()
            .context("native image has no retained embedding stream")?;
        m.output.read_f32(m.rows * m.hc * self.vision.config.dim)
    }
    pub fn config(&self) -> &DeepseekV41VisionConfig {
        &self.vision.config
    }
    pub fn parameters(&self) -> Result<Tensors> {
        self.parameter_values(false)
    }
    pub fn gradients(&self) -> Result<Tensors> {
        self.parameter_values(true)
    }
    fn parameter_values(&self, grad: bool) -> Result<Tensors> {
        let mut tensors = if grad {
            self.vision.gradients()?
        } else {
            self.vision.parameters()?
        };
        for (name, p) in MARKERS.iter().zip(&self.markers) {
            tensors.insert(
                name.to_string(),
                DeepseekV41VisionTensor {
                    shape: vec![self.vision.config.dim],
                    values: if grad {
                        p.grad_buffer()?.read_f32(self.vision.config.dim)?
                    } else {
                        p.read()?
                    },
                },
            );
        }
        Ok(tensors)
    }
    pub fn patch_gradient(&self) -> Result<Vec<f32>> {
        self.vision
            .patch_gradient()
            .read_f32(self.vision.rows * self.vision.patch.input_dim)
    }
}
