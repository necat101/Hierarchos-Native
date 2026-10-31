//! Explicit packed storage and experimental block-scaled linear primitives.
//! These do not change the FP32 model's precision policy or enable production
//! execution. Component evidence is separate from complete CPU logit parity.
use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use super::{DeepseekV41ActivationQuantizer, DeepseekV41QuantMode, DeepseekV41ScaleMode};
use crate::{vulkan, GpuBuffer, VulkanDevice};

const WEIGHT_DECODE: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_weight_decode.spv");
const PACKED_LINEAR: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_packed_linear.spv");

/// Scale layouts are explicit: they cannot be inferred from a byte dtype alone.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepseekV41WeightFormat {
    Fp8Block32,
    Fp8Block128,
    Fp4Row32,
    EngramFp8Row32,
}
impl DeepseekV41WeightFormat {
    fn code(self) -> u32 {
        match self {
            Self::Fp8Block32 => 0,
            Self::Fp8Block128 => 1,
            Self::Fp4Row32 => 2,
            Self::EngramFp8Row32 => 3,
        }
    }
    pub fn block_size(self) -> usize {
        if self == Self::Fp8Block128 {
            128
        } else {
            32
        }
    }
    pub fn scale_shape(self, rows: usize, cols: usize) -> [usize; 2] {
        let block = self.block_size();
        let scale_rows = if matches!(self, Self::Fp8Block32 | Self::Fp8Block128) {
            rows.div_ceil(block)
        } else {
            rows
        };
        [scale_rows, cols.div_ceil(block)]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepseekV41WeightOutput {
    Fp32,
    Bf16,
}
impl DeepseekV41WeightOutput {
    fn code(self) -> u32 {
        u32::from(self == Self::Bf16)
    }
}

/// Validated byte payloads. FP4 is low-nibble first along the logical K axis;
/// every scale byte is UE8M0, including code zero (2^-127), never NaN code 255.
#[derive(Debug)]
pub struct DeepseekV41PackedWeights {
    rows: usize,
    cols: usize,
    format: DeepseekV41WeightFormat,
    data: Vec<u8>,
    scales: Vec<u8>,
    max_abs_bits: u32,
}
impl DeepseekV41PackedWeights {
    pub fn new(
        rows: usize,
        cols: usize,
        format: DeepseekV41WeightFormat,
        data: Vec<u8>,
        scales: Vec<u8>,
    ) -> Result<Self> {
        let elements = rows
            .checked_mul(cols)
            .context("V4.1 packed weight count overflow")?;
        if rows == 0
            || cols == 0
            || !cols.is_multiple_of(format.block_size())
            || elements > u32::MAX as usize
        {
            bail!("V4.1 packed weight geometry exceeds the supported block/dispatch contract");
        }
        let expected_bytes = if format == DeepseekV41WeightFormat::Fp4Row32 {
            elements / 2
        } else {
            elements
        };
        let [sr, sc] = format.scale_shape(rows, cols);
        if data.len() != expected_bytes || scales.len() != sr * sc {
            bail!("V4.1 packed weight or UE8M0 scale byte count mismatch");
        }
        if scales.contains(&255) {
            bail!("V4.1 UE8M0 scale contains NaN code 255");
        }
        if format != DeepseekV41WeightFormat::Fp4Row32 && data.iter().any(|b| b & 127 == 127) {
            bail!("V4.1 FP8 weight contains a non-finite E4M3FN code");
        }
        // Validate finite materialization on the host; model arithmetic remains
        // on the GPU. Powers of two and these low-bit significands are exact.
        let mut max_abs_bits = 0;
        for i in 0..elements {
            let row = i / cols;
            let col = i % cols;
            let scale_row = if matches!(
                format,
                DeepseekV41WeightFormat::Fp8Block32 | DeepseekV41WeightFormat::Fp8Block128
            ) {
                row / format.block_size()
            } else {
                row
            };
            let scale = scales[scale_row * sc + col / format.block_size()];
            let magnitude = if format == DeepseekV41WeightFormat::Fp4Row32 {
                const LEVELS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
                LEVELS[((data[i / 2] >> ((i % 2) * 4)) & 7) as usize] as f64
            } else {
                let b = data[i] & 127;
                let e = b >> 3;
                if e == 0 {
                    (b & 7) as f64 / 512.0
                } else {
                    (1.0 + (b & 7) as f64 / 8.0) * 2f64.powi(e as i32 - 7)
                }
            };
            let value = (magnitude * 2f64.powi(scale as i32 - 127)) as f32;
            if !value.is_finite() {
                bail!("V4.1 scaled packed weight exceeds finite FP32");
            }
            max_abs_bits = max_abs_bits.max(value.to_bits());
        }
        Ok(Self {
            rows,
            cols,
            format,
            data,
            scales,
            max_abs_bits,
        })
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn format(&self) -> DeepseekV41WeightFormat {
        self.format
    }
    pub fn data(&self) -> &[u8] {
        &self.data
    }
    pub fn scales(&self) -> &[u8] {
        &self.scales
    }

    pub fn upload(&self, device: &VulkanDevice) -> Result<DeepseekV41ResidentPackedWeights> {
        let mut commands = device.compute_batch()?;
        let weight = self.record_upload(device, &mut commands)?;
        commands.submit_and_wait()?;
        Ok(weight)
    }

    pub(crate) fn record_upload(
        &self,
        device: &VulkanDevice,
        commands: &mut vulkan::ComputeBatch,
    ) -> Result<DeepseekV41ResidentPackedWeights> {
        fn words(bytes: &[u8]) -> Vec<u32> {
            bytes
                .chunks(4)
                .map(|chunk| {
                    let mut word = [0u8; 4];
                    word[..chunk.len()].copy_from_slice(chunk);
                    u32::from_le_bytes(word)
                })
                .collect()
        }
        Ok(DeepseekV41ResidentPackedWeights {
            rows: self.rows,
            cols: self.cols,
            format: self.format,
            data: GpuBuffer::from_u32_recorded(device, commands, &words(&self.data))?,
            scales: GpuBuffer::from_u32_recorded(device, commands, &words(&self.scales))?,
            max_abs_bits: self.max_abs_bits,
        })
    }
}

/// Cloning shares the immutable device payload; the packed bytes are never
/// rewritten, so a clone is always the same released matrix.
#[derive(Clone)]
pub struct DeepseekV41ResidentPackedWeights {
    rows: usize,
    cols: usize,
    format: DeepseekV41WeightFormat,
    data: GpuBuffer,
    scales: GpuBuffer,
    max_abs_bits: u32,
}
impl DeepseekV41ResidentPackedWeights {
    pub(super) fn buffers(&self) -> (&GpuBuffer, &GpuBuffer) {
        (&self.data, &self.scales)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WeightDecodePush {
    rows: u32,
    cols: u32,
    format: u32,
    bf16: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PackedLinearPush {
    rows: u32,
    cols: u32,
    inner: u32,
    format: u32,
    act_block: u32,
    bf16: u32,
}

pub struct DeepseekV41PackedWeightKernel {
    device: VulkanDevice,
    decode: vulkan::ComputeKernel,
    linear: vulkan::ComputeKernel,
    quantizer: DeepseekV41ActivationQuantizer,
}
impl DeepseekV41PackedWeightKernel {
    pub fn new(device: VulkanDevice) -> Result<Self> {
        Ok(Self {
            decode: vulkan::ComputeKernel::new(
                &device,
                WEIGHT_DECODE,
                3,
                std::mem::size_of::<WeightDecodePush>() as u32,
            )?,
            linear: vulkan::ComputeKernel::new(
                &device,
                PACKED_LINEAR,
                5,
                std::mem::size_of::<PackedLinearPush>() as u32,
            )?,
            quantizer: DeepseekV41ActivationQuantizer::new(device.clone())?,
            device,
        })
    }
    fn check_device(&self, weight: &DeepseekV41ResidentPackedWeights) -> Result<()> {
        if !weight.data.belongs_to_device(&self.device)
            || !weight.scales.belongs_to_device(&self.device)
        {
            bail!("V4.1 packed weights belong to another Vulkan device");
        }
        Ok(())
    }
    /// Materialize the stored values or the converter/Engram BF16 cast. This
    /// is not a replacement for block-scaled quantized GEMM.
    pub fn dequantize(
        &self,
        weight: &DeepseekV41ResidentPackedWeights,
        output: DeepseekV41WeightOutput,
    ) -> Result<GpuBuffer> {
        self.check_device(weight)?;
        let b = weight.max_abs_bits;
        if output == DeepseekV41WeightOutput::Bf16 && (b + 0x7fff + ((b >> 16) & 1)) >= 0x7f800000 {
            bail!("V4.1 scaled packed weight exceeds finite BF16");
        }
        let result = GpuBuffer::zeros_u32(&self.device, weight.rows * weight.cols)?;
        let push = WeightDecodePush {
            rows: weight.rows as u32,
            cols: weight.cols as u32,
            format: weight.format.code(),
            bf16: output.code(),
        };
        let mut commands = self.device.compute_batch()?;
        let groups = (weight.rows * weight.cols).div_ceil(64);
        self.decode.record_dispatch(
            &mut commands,
            &[&weight.data, &weight.scales, &result],
            bytemuck::bytes_of(&push),
            [groups.min(65_535) as u32, groups.div_ceil(65_535) as u32, 1],
        )?;
        commands.submit()?;
        Ok(result)
    }
    /// Experimental CPU-mirror contract: FP8 activations, per-K-block FP32
    /// dots, scale application, FP32 block accumulation, optional BF16 output.
    /// Qualification uses the pinned CPU PyTorch reference.
    pub fn linear_buffer(
        &self,
        input: &GpuBuffer,
        rows: usize,
        weight: &DeepseekV41ResidentPackedWeights,
        act_block: usize,
        output: DeepseekV41WeightOutput,
    ) -> Result<GpuBuffer> {
        let mut commands = self.device.compute_batch()?;
        let result =
            self.record_linear_buffer(&mut commands, input, rows, weight, act_block, output)?;
        commands.submit()?;
        Ok(result)
    }

    /// Record-only form of [`Self::linear_buffer`]: the activation quantizer and
    /// the packed GEMM land in the caller's batch in the same order.
    pub(crate) fn record_linear_buffer(
        &self,
        commands: &mut vulkan::ComputeBatch,
        input: &GpuBuffer,
        rows: usize,
        weight: &DeepseekV41ResidentPackedWeights,
        act_block: usize,
        output: DeepseekV41WeightOutput,
    ) -> Result<GpuBuffer> {
        self.linear_geometry(rows, weight, act_block)?;
        let act = self.record_activation_input(commands, input, rows, weight.cols, act_block)?;
        self.record_linear_activation(commands, &act, weight, output)
    }

    fn linear_geometry(
        &self,
        rows: usize,
        weight: &DeepseekV41ResidentPackedWeights,
        act_block: usize,
    ) -> Result<usize> {
        self.check_device(weight)?;
        if weight.format == DeepseekV41WeightFormat::EngramFp8Row32 {
            bail!("V4.1 Engram row-scaled tables cannot be used as linear weights");
        }
        if !matches!(act_block, 32 | 128)
            || !weight.cols.is_multiple_of(act_block)
            || (weight.format != DeepseekV41WeightFormat::Fp4Row32
                && act_block != weight.format.block_size())
        {
            bail!("V4.1 activation/weight block geometry mismatch");
        }
        let count = rows
            .checked_mul(weight.rows)
            .context("V4.1 packed linear output overflow")?;
        if rows == 0
            || rows > 65_535
            || weight.rows.div_ceil(64) > 65_535
            || count > u32::MAX as usize
        {
            bail!("V4.1 packed linear exceeds dispatch/addressing capacity");
        }
        Ok(count)
    }

    /// Prepare the exact FP8 activation once for projections sharing an input
    /// and activation block size. Weight storage does not affect this step.
    pub(super) fn record_activation_input(
        &self,
        commands: &mut vulkan::ComputeBatch,
        input: &GpuBuffer,
        rows: usize,
        cols: usize,
        act_block: usize,
    ) -> Result<super::DeepseekV41QuantizedActivation> {
        self.quantizer.record_quantize_buffer(
            commands,
            input,
            rows,
            cols,
            act_block,
            DeepseekV41QuantMode::Fp8E4M3,
            DeepseekV41ScaleMode::Ue8m0,
        )
    }

    pub(super) fn record_linear_activation(
        &self,
        commands: &mut vulkan::ComputeBatch,
        act: &super::DeepseekV41QuantizedActivation,
        weight: &DeepseekV41ResidentPackedWeights,
        output: DeepseekV41WeightOutput,
    ) -> Result<GpuBuffer> {
        let rows = act.rows;
        let act_block = act.block_size;
        let count = self.linear_geometry(rows, weight, act_block)?;
        if act.cols != weight.cols
            || act.mode != DeepseekV41QuantMode::Fp8E4M3
            || act.scale_mode != DeepseekV41ScaleMode::Ue8m0
            || !act.values.belongs_to_device(&self.device)
            || !act.scales.belongs_to_device(&self.device)
            || act.values.f32_capacity() < rows * act.cols
            || act.scales.f32_capacity() < rows * act.blocks_per_row()
        {
            bail!("V4.1 prepared activation geometry, format or device mismatch");
        }
        let result = GpuBuffer::zeros_u32(&self.device, count)?;
        let push = PackedLinearPush {
            rows: rows as u32,
            cols: weight.rows as u32,
            inner: weight.cols as u32,
            format: weight.format.code(),
            act_block: act_block as u32,
            bf16: output.code(),
        };
        self.linear.record_dispatch(
            commands,
            &[
                &act.values,
                &act.scales,
                &weight.data,
                &weight.scales,
                &result,
            ],
            bytemuck::bytes_of(&push),
            [weight.rows.div_ceil(64) as u32, rows as u32, 1],
        )?;
        Ok(result)
    }
}
