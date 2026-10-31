//! V4.1 rotary embedding: the released YaRN frequency schedule and the
//! adjacent-pair application used by queries, keys, compressed KV, and the
//! inverse rotation of the attention output.
use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::{vulkan, GpuBuffer, VulkanDevice};

const ROTARY: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_rotary.spv");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Push {
    rows: u32,
    width: u32,
    rope_dim: u32,
    position_base: u32,
    position_stride: u32,
    position_period: u32,
    row_repeat: u32,
    inverse: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepseekV41RotaryConfig {
    pub dim: usize,
    pub max_positions: usize,
    /// `original_seq_len` in the released model. Zero disables YaRN, which is
    /// what a sliding-only attention layer does.
    pub original_max_position_embeddings: usize,
    pub base: f32,
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

impl DeepseekV41RotaryConfig {
    /// Sliding-only attention: base theta, no YaRN.
    pub fn sliding(dim: usize, max_positions: usize, base: f32) -> Self {
        Self {
            dim,
            max_positions,
            original_max_position_embeddings: 0,
            base,
            factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }

    /// Compressed attention: compressed theta with YaRN over the original
    /// training context.
    pub fn compressed(
        dim: usize,
        max_positions: usize,
        base: f32,
        original_max_position_embeddings: usize,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> Self {
        Self {
            dim,
            max_positions,
            original_max_position_embeddings,
            base,
            factor,
            beta_fast,
            beta_slow,
        }
    }
}

/// The released `precompute_freqs_cis` table. Angles are computed in FP32 with
/// the same operation order as the reference, so their bit patterns match.
/// Cosines and sines are evaluated in double precision and rounded to FP32:
/// PyTorch's CPU/GPU trig kernels can differ from that by one unit in the last
/// place, which stays below the fixed `2e-7` component ceiling but is recorded
/// as a table-generation caveat rather than claimed as bit-exact.
pub struct DeepseekV41Rotary {
    device: VulkanDevice,
    config: DeepseekV41RotaryConfig,
    pairs: usize,
    angles: Vec<f32>,
    cosine: Vec<f32>,
    sine: Vec<f32>,
    cos_buffer: GpuBuffer,
    sin_buffer: GpuBuffer,
    kernel: vulkan::ComputeKernel,
}

impl DeepseekV41Rotary {
    pub fn new(device: VulkanDevice, config: DeepseekV41RotaryConfig) -> Result<Self> {
        let (angles, cosine, sine) = build_table(config)?;
        let cos_buffer = GpuBuffer::from_f32(&device, &cosine)?;
        let sin_buffer = GpuBuffer::from_f32(&device, &sine)?;
        let kernel =
            vulkan::ComputeKernel::new(&device, ROTARY, 4, std::mem::size_of::<Push>() as u32)?;
        Ok(Self {
            device,
            config,
            pairs: config.dim / 2,
            angles,
            cosine,
            sine,
            cos_buffer,
            sin_buffer,
            kernel,
        })
    }

    /// Installs an externally produced table, with the same shape and range
    /// validation as generation. This isolates application parity from the
    /// published trig implementation when qualifying another component.
    pub fn from_table(
        device: VulkanDevice,
        config: DeepseekV41RotaryConfig,
        cosine: Vec<f32>,
        sine: Vec<f32>,
    ) -> Result<Self> {
        validate_config(config)?;
        let expected = config
            .max_positions
            .checked_mul(config.dim / 2)
            .context("V4.1 rotary table size overflow")?;
        if cosine.len() != expected || sine.len() != expected {
            bail!("V4.1 rotary table length mismatch");
        }
        if cosine
            .iter()
            .chain(&sine)
            .any(|value| !value.is_finite() || value.abs() > 1.000_001)
        {
            bail!("V4.1 rotary table values must be finite unit-magnitude pairs");
        }
        let angles = rebuild_angles(config)?;
        let cos_buffer = GpuBuffer::from_f32(&device, &cosine)?;
        let sin_buffer = GpuBuffer::from_f32(&device, &sine)?;
        let kernel =
            vulkan::ComputeKernel::new(&device, ROTARY, 4, std::mem::size_of::<Push>() as u32)?;
        Ok(Self {
            device,
            config,
            pairs: config.dim / 2,
            angles,
            cosine,
            sine,
            cos_buffer,
            sin_buffer,
            kernel,
        })
    }

    pub fn config(&self) -> DeepseekV41RotaryConfig {
        self.config
    }

    pub fn angles(&self) -> &[f32] {
        &self.angles
    }

    pub fn table(&self) -> (&[f32], &[f32]) {
        (&self.cosine, &self.sine)
    }

    /// Applies the rotation to the trailing `rope_dim` channels of each row.
    /// `position_base + row * position_stride` selects the table row, which
    /// covers token positions (stride 1) and compressed groups (stride ratio).
    pub fn apply(
        &self,
        input: &[f32],
        rows: usize,
        width: usize,
        position_base: usize,
        position_stride: usize,
        inverse: bool,
    ) -> Result<GpuBuffer> {
        self.apply_grouped(
            input,
            rows,
            width,
            rows,
            position_base,
            position_stride,
            inverse,
        )
    }

    /// Batch-local grouping: rows are `[batch, groups]` contiguous, each batch
    /// restarts the table at `position_base` and advances by `position_stride`.
    pub fn apply_grouped(
        &self,
        input: &[f32],
        rows: usize,
        width: usize,
        groups_per_batch: usize,
        position_base: usize,
        position_stride: usize,
        inverse: bool,
    ) -> Result<GpuBuffer> {
        let elements = self.validate_application(
            rows,
            width,
            1,
            groups_per_batch,
            position_base,
            position_stride,
        )?;
        if input.len() != elements {
            bail!("V4.1 rotary input length mismatch");
        }
        if input.iter().any(|value| !value.is_finite()) {
            bail!("V4.1 rotary input must be finite");
        }
        let input = GpuBuffer::from_f32(&self.device, input)?;
        self.apply_grouped_buffer(
            &input,
            rows,
            width,
            groups_per_batch,
            position_base,
            position_stride,
            inverse,
        )
    }

    pub fn apply_buffer(
        &self,
        input: &GpuBuffer,
        rows: usize,
        width: usize,
        position_base: usize,
        position_stride: usize,
        inverse: bool,
    ) -> Result<GpuBuffer> {
        self.apply_grouped_buffer(
            input,
            rows,
            width,
            rows,
            position_base,
            position_stride,
            inverse,
        )
    }

    /// GPU-resident grouped application; no host round trip.
    pub fn apply_grouped_buffer(
        &self,
        input: &GpuBuffer,
        rows: usize,
        width: usize,
        groups_per_batch: usize,
        position_base: usize,
        position_stride: usize,
        inverse: bool,
    ) -> Result<GpuBuffer> {
        let elements = self.validate_application(
            rows,
            width,
            1,
            groups_per_batch,
            position_base,
            position_stride,
        )?;
        if !input.belongs_to_device(&self.device) || input.f32_capacity() < elements {
            bail!("V4.1 rotary input buffer is too small or belongs to another Vulkan device");
        }
        let output = GpuBuffer::zeros_f32(&self.device, elements)?;
        let push = Push {
            rows: rows as u32,
            width: width as u32,
            rope_dim: self.config.dim as u32,
            position_base: position_base as u32,
            position_stride: position_stride as u32,
            position_period: groups_per_batch as u32,
            row_repeat: 1,
            inverse: u32::from(inverse),
        };
        let mut commands = self.device.compute_batch()?;
        self.kernel.record_dispatch(
            &mut commands,
            &[input, &self.cos_buffer, &self.sin_buffer, &output],
            bytemuck::bytes_of(&push),
            row_grid(self.pairs.div_ceil(64) as u32, rows),
        )?;
        commands.submit()?;
        Ok(output)
    }

    /// Per-head rows: `row_repeat` consecutive rows share one position. The
    /// indexer's queries are `[batch, seq, heads, head_dim]`, so each group of
    /// `heads` rows rotates at the same token position.
    pub fn apply_repeated_buffer(
        &self,
        input: &GpuBuffer,
        rows: usize,
        width: usize,
        row_repeat: usize,
        groups_per_batch: usize,
        position_base: usize,
        position_stride: usize,
        inverse: bool,
    ) -> Result<GpuBuffer> {
        let elements = self.validate_application(
            rows,
            width,
            row_repeat,
            groups_per_batch,
            position_base,
            position_stride,
        )?;
        if !input.belongs_to_device(&self.device) || input.f32_capacity() < elements {
            bail!("V4.1 rotary input buffer is too small or belongs to another Vulkan device");
        }
        let output = GpuBuffer::zeros_f32(&self.device, elements)?;
        let push = Push {
            rows: rows as u32,
            width: width as u32,
            rope_dim: self.config.dim as u32,
            position_base: position_base as u32,
            position_stride: position_stride as u32,
            position_period: groups_per_batch as u32,
            row_repeat: row_repeat as u32,
            inverse: u32::from(inverse),
        };
        let mut commands = self.device.compute_batch()?;
        self.kernel.record_dispatch(
            &mut commands,
            &[input, &self.cos_buffer, &self.sin_buffer, &output],
            bytemuck::bytes_of(&push),
            row_grid(self.pairs.div_ceil(64) as u32, rows),
        )?;
        commands.submit()?;
        Ok(output)
    }

    /// Input VJP: complex multiplication's adjoint conjugates the supplied
    /// rotation. Prefix channels are copied, and the forward's batch/head
    /// position mapping is preserved. This is the transpose, which does not
    /// assume that rounded cosine/sine pairs have exactly unit norm.
    #[allow(clippy::too_many_arguments)]
    pub fn input_gradient_repeated(
        &self,
        upstream: &GpuBuffer,
        rows: usize,
        width: usize,
        row_repeat: usize,
        groups_per_batch: usize,
        position_base: usize,
        position_stride: usize,
        inverse_forward: bool,
    ) -> Result<GpuBuffer> {
        self.apply_repeated_buffer(
            upstream,
            rows,
            width,
            row_repeat,
            groups_per_batch,
            position_base,
            position_stride,
            !inverse_forward,
        )
    }

    fn validate_application(
        &self,
        rows: usize,
        width: usize,
        row_repeat: usize,
        groups_per_batch: usize,
        position_base: usize,
        position_stride: usize,
    ) -> Result<usize> {
        if rows == 0 || width < self.config.dim {
            bail!("V4.1 rotary requires rows > 0 and width >= rope dimension");
        }
        if groups_per_batch == 0
            || row_repeat == 0
            || !rows.is_multiple_of(groups_per_batch.saturating_mul(row_repeat))
        {
            bail!("V4.1 rotary rows must be a positive multiple of repeat * group size");
        }
        if self.pairs.div_ceil(64) > 65_535 {
            bail!("V4.1 rotary exceeds dispatch capacity");
        }
        let last = position_base
            .checked_add(
                (groups_per_batch - 1)
                    .checked_mul(position_stride)
                    .context("V4.1 rotary position overflow")?,
            )
            .context("V4.1 rotary position overflow")?;
        if last >= self.config.max_positions {
            bail!("V4.1 rotary position exceeds the prepared table");
        }
        let elements = rows
            .checked_mul(width)
            .context("V4.1 rotary element count overflow")?;
        if elements > u32::MAX as usize {
            bail!("V4.1 rotary exceeds u32 addressing");
        }
        Ok(elements)
    }
}

/// Workgrid for a per-row kernel. Vulkan caps every dispatch axis at 65,535
/// workgroups, so rows past that are tiled on z and the shader adds
/// `z * NumWorkGroups.y` to its row index.
pub(crate) fn row_grid(x: u32, rows: usize) -> [u32; 3] {
    let ygrid = rows.min(65_535) as u32;
    let tiles = rows.div_ceil(ygrid.max(1) as usize) as u32;
    [x, ygrid, tiles]
}

fn validate_config(config: DeepseekV41RotaryConfig) -> Result<()> {
    if config.dim == 0
        || !config.dim.is_multiple_of(2)
        || config.dim > u32::MAX as usize
        || config.max_positions == 0
        || config.max_positions > u32::MAX as usize
    {
        bail!("V4.1 rotary dimension and position count must be positive even u32 values");
    }
    if !config.base.is_finite() || config.base <= 0.0 {
        bail!("V4.1 rotary base must be finite and positive");
    }
    if config.original_max_position_embeddings > 0 {
        if !config.factor.is_finite()
            || config.factor <= 0.0
            || !config.beta_fast.is_finite()
            || !config.beta_slow.is_finite()
            || config.beta_slow <= 0.0
            || config.beta_fast <= config.beta_slow
        {
            bail!("V4.1 rotary YaRN parameters are invalid");
        }
    }
    Ok(())
}

/// Frequency and position-angle computation, shared by generation and table
/// installation. Operation order matches the released FP32 reference.
fn rebuild_angles(config: DeepseekV41RotaryConfig) -> Result<Vec<f32>> {
    validate_config(config)?;
    let pairs = config.dim / 2;
    let (low, high) = yarn_bounds(config);
    let mut frequencies = Vec::with_capacity(pairs);
    for i in 0..pairs {
        let exponent = (2 * i) as f32 / config.dim as f32;
        let mut frequency = 1.0f32 / config.base.powf(exponent);
        if config.original_max_position_embeddings > 0 {
            let ramp = ((i as f32 - low as f32) / (high - low).max(1e-3) as f32).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            frequency = frequency / config.factor * (1.0 - smooth) + frequency * smooth;
        }
        frequencies.push(frequency);
    }
    let mut angles = Vec::with_capacity(config.max_positions * pairs);
    for position in 0..config.max_positions {
        let position = position as f32;
        for &frequency in &frequencies {
            angles.push(position * frequency);
        }
    }
    Ok(angles)
}

fn yarn_bounds(config: DeepseekV41RotaryConfig) -> (f64, f64) {
    if config.original_max_position_embeddings == 0 {
        return (0.0, 0.0);
    }
    let dim = config.dim as f64;
    let log_base = (config.base as f64).ln();
    let corrected = |rotations: f64| {
        dim * ((config.original_max_position_embeddings as f64)
            / (rotations * 2.0 * std::f64::consts::PI))
            .ln()
            / (2.0 * log_base)
    };
    let low = corrected(config.beta_fast as f64).floor().max(0.0);
    let high = corrected(config.beta_slow as f64)
        .ceil()
        .min((config.dim - 1) as f64);
    (low, high)
}

fn build_table(config: DeepseekV41RotaryConfig) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let angles = rebuild_angles(config)?;
    let cosine = angles
        .iter()
        .map(|&angle| (angle as f64).cos() as f32)
        .collect();
    let sine = angles
        .iter()
        .map(|&angle| (angle as f64).sin() as f32)
        .collect();
    Ok((angles, cosine, sine))
}
