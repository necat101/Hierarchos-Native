//! V4.1 activation quantization with the released fused quantize/dequantize
//! semantics. This is a numerical primitive, not the complete checkpoint
//! execution policy: callers decide whether an intermediate is quantized.
use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::{vulkan, GpuBuffer, VulkanDevice};

use super::rotary::row_grid;

const QUANTIZE: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_quantize.spv");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Push {
    rows: u32,
    cols: u32,
    block_size: u32,
    mode: u32,
    scale_mode: u32,
    amax_min: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepseekV41QuantMode {
    /// FP8 E4M3, used for queries/keys and dense activations.
    Fp8E4M3,
    /// FP4 E2M1, used by the indexer and compressed KV caches.
    Fp4E2M1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepseekV41ScaleMode {
    /// Power-of-two scales (`fast_round_scale` / UE8M0 storage).
    Ue8m0,
    /// E4M3 scales, used by compressed KV with 16-element groups.
    E4m3,
}

impl DeepseekV41QuantMode {
    fn code(self) -> u32 {
        match self {
            Self::Fp8E4M3 => 0,
            Self::Fp4E2M1 => 1,
        }
    }
}

impl DeepseekV41ScaleMode {
    fn code(self) -> u32 {
        match self {
            Self::Ue8m0 => 0,
            Self::E4m3 => 1,
        }
    }
}

/// Dequantized values in FP32 plus the stored per-block scales. The reference's
/// `inplace=True` path returns values cast back to the input dtype; the model's
/// BF16 promotion is represented here as FP32 because every caller feeds the
/// values into an FP32 accumulation.
pub struct DeepseekV41QuantizedActivation {
    pub rows: usize,
    pub cols: usize,
    pub block_size: usize,
    pub mode: DeepseekV41QuantMode,
    pub scale_mode: DeepseekV41ScaleMode,
    pub values: GpuBuffer,
    pub scales: GpuBuffer,
}

impl DeepseekV41QuantizedActivation {
    pub fn blocks_per_row(&self) -> usize {
        self.cols.div_ceil(self.block_size)
    }

    pub fn read_values(&self) -> Result<Vec<f32>> {
        self.values.read_f32(self.rows * self.cols)
    }

    /// Stored scale values as FP32 bit patterns. E4M3 and UE8M0 scales are
    /// rounded before storage, so the scale itself is part of the contract.
    pub fn read_scale_bits(&self) -> Result<Vec<u32>> {
        self.scales.read_u32(self.rows * self.blocks_per_row())
    }

    pub fn read_scales(&self) -> Result<Vec<f32>> {
        Ok(self
            .read_scale_bits()?
            .into_iter()
            .map(f32::from_bits)
            .collect())
    }
}

pub struct DeepseekV41ActivationQuantizer {
    device: VulkanDevice,
    kernel: vulkan::ComputeKernel,
}

fn amax_min(mode: DeepseekV41QuantMode, scale_mode: DeepseekV41ScaleMode) -> f32 {
    match (mode, scale_mode) {
        // fp8_max_inv path: `amax = max(amax, 1e-4)`.
        (DeepseekV41QuantMode::Fp8E4M3, DeepseekV41ScaleMode::Ue8m0) => 1e-4,
        // fp4 power-of-two path: `amax = max(amax, 6 * 2**-126)`.
        (DeepseekV41QuantMode::Fp4E2M1, DeepseekV41ScaleMode::Ue8m0) => 6.0 * 2f32.powi(-126),
        // fp4 E4M3 path: `amax = max(amax, 6 * 2**-9)`.
        (DeepseekV41QuantMode::Fp4E2M1, DeepseekV41ScaleMode::E4m3) => 6.0 * 2f32.powi(-9),
        (DeepseekV41QuantMode::Fp8E4M3, DeepseekV41ScaleMode::E4m3) => f32::NAN,
    }
}

impl DeepseekV41ActivationQuantizer {
    pub fn new(device: VulkanDevice) -> Result<Self> {
        Ok(Self {
            kernel: vulkan::ComputeKernel::new(
                &device,
                QUANTIZE,
                3,
                std::mem::size_of::<Push>() as u32,
            )?,
            device,
        })
    }

    pub fn quantize(
        &self,
        input: &[f32],
        rows: usize,
        cols: usize,
        block_size: usize,
        mode: DeepseekV41QuantMode,
        scale_mode: DeepseekV41ScaleMode,
    ) -> Result<DeepseekV41QuantizedActivation> {
        let elements = self.validate(rows, cols, block_size, mode, scale_mode)?;
        if input.len() != elements {
            bail!("V4.1 quantization input length mismatch");
        }
        if input.iter().any(|value| !value.is_finite()) {
            bail!("V4.1 quantization input must be finite");
        }
        let input = GpuBuffer::from_f32(&self.device, input)?;
        self.quantize_buffer(&input, rows, cols, block_size, mode, scale_mode)
    }

    /// Consumes a Vulkan tensor without a host round trip.
    pub fn quantize_buffer(
        &self,
        input: &GpuBuffer,
        rows: usize,
        cols: usize,
        block_size: usize,
        mode: DeepseekV41QuantMode,
        scale_mode: DeepseekV41ScaleMode,
    ) -> Result<DeepseekV41QuantizedActivation> {
        let mut commands = self.device.compute_batch()?;
        let activation = self.record_quantize_buffer(
            &mut commands,
            input,
            rows,
            cols,
            block_size,
            mode,
            scale_mode,
        )?;
        commands.submit()?;
        Ok(activation)
    }

    /// Record-only form of [`Self::quantize_buffer`]: same dispatch and buffers,
    /// with the wait left to the caller's batch.
    pub(crate) fn record_quantize_buffer(
        &self,
        commands: &mut crate::vulkan::ComputeBatch,
        input: &GpuBuffer,
        rows: usize,
        cols: usize,
        block_size: usize,
        mode: DeepseekV41QuantMode,
        scale_mode: DeepseekV41ScaleMode,
    ) -> Result<DeepseekV41QuantizedActivation> {
        let elements = self.validate(rows, cols, block_size, mode, scale_mode)?;
        if !input.belongs_to_device(&self.device) || input.f32_capacity() < elements {
            bail!(
                "V4.1 quantization input buffer is too small or belongs to another Vulkan device"
            );
        }
        let blocks = cols.div_ceil(block_size);
        let values = GpuBuffer::zeros_f32(&self.device, elements)?;
        let scales = GpuBuffer::zeros_u32(&self.device, rows * blocks)?;
        let push = Push {
            rows: rows as u32,
            cols: cols as u32,
            block_size: block_size as u32,
            mode: mode.code(),
            scale_mode: scale_mode.code(),
            amax_min: amax_min(mode, scale_mode),
        };
        self.kernel.record_dispatch(
            commands,
            &[input, &values, &scales],
            bytemuck::bytes_of(&push),
            row_grid(blocks as u32, rows),
        )?;
        Ok(DeepseekV41QuantizedActivation {
            rows,
            cols,
            block_size,
            mode,
            scale_mode,
            values,
            scales,
        })
    }

    fn validate(
        &self,
        rows: usize,
        cols: usize,
        block_size: usize,
        mode: DeepseekV41QuantMode,
        scale_mode: DeepseekV41ScaleMode,
    ) -> Result<usize> {
        if rows == 0 || cols == 0 || block_size == 0 {
            bail!("V4.1 quantization requires positive rows, columns, and block size");
        }
        if !cols.is_multiple_of(block_size) {
            // The released kernel asserts N % block_size == 0.
            bail!("V4.1 quantization columns must be a multiple of the block size");
        }
        if cols.div_ceil(block_size) > 65_535 {
            bail!("V4.1 quantization exceeds dispatch capacity");
        }
        let elements = rows
            .checked_mul(cols)
            .context("V4.1 quantization element count overflow")?;
        if elements > u32::MAX as usize {
            bail!("V4.1 quantization exceeds u32 addressing");
        }
        if mode == DeepseekV41QuantMode::Fp8E4M3 && scale_mode != DeepseekV41ScaleMode::Ue8m0 {
            bail!("V4.1 FP8 quantization has no released E4M3-scale path");
        }
        Ok(elements)
    }
}
