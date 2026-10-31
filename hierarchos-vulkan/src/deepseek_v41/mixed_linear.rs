//! Typed linear storage and explicit activation/output precision boundaries.
//! FP8/FP4 results follow the block-scaled CPU PyTorch reference contract.
//! No mixed weight is silently treated as FP32.
use super::indexer::LinearKernel;
use super::peft::DeepseekV41LoraHost;
use super::{
    DeepseekV41PackedWeightKernel, DeepseekV41PackedWeights, DeepseekV41ResidentPackedWeights,
    DeepseekV41WeightFormat, DeepseekV41WeightOutput,
};
use crate::{vulkan, GpuBuffer, VulkanDevice};
use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

#[path = "linear_grad.rs"]
mod gradients;
pub use gradients::{DeepseekV41LinearAdjoints, DeepseekV41LinearGradients, DeepseekV41LinearTape};

const BF16_CAST: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_bf16_cast.spv");
const LORA_ADD: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_lora_add.spv");
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CastPush {
    elements: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LoraAddPush {
    elements: u32,
    scale: f32,
    bf16: u32,
}

pub enum DeepseekV41LinearWeights {
    Fp32 {
        rows: usize,
        cols: usize,
        values: Vec<f32>,
        lora: Option<DeepseekV41LoraHost>,
    },
    Bf16 {
        rows: usize,
        cols: usize,
        values: Vec<f32>,
        lora: Option<DeepseekV41LoraHost>,
    },
    Packed {
        weight: DeepseekV41PackedWeights,
        lora: Option<DeepseekV41LoraHost>,
    },
}
impl DeepseekV41LinearWeights {
    pub fn shape(&self) -> [usize; 2] {
        match self {
            Self::Fp32 { rows, cols, .. } | Self::Bf16 { rows, cols, .. } => [*rows, *cols],
            Self::Packed { weight, .. } => [weight.rows(), weight.cols()],
        }
    }

    /// Attach one validated PEFT module to this projection. The host resolves
    /// the checkpoint tensor name; nothing here rewrites the base payload.
    pub(super) fn attach_lora(
        &mut self,
        module: &super::peft::DeepseekV41LoraModule,
    ) -> Result<()> {
        let [rows, cols] = self.shape();
        if module.a.len() != module.rank * cols || module.b.len() != rows * module.rank {
            bail!(
                "V4.1 LoRA module {:?} has A/B shapes incompatible with [{rows}, {cols}] at rank {}",
                module.target,
                module.rank
            );
        }
        let host = DeepseekV41LoraHost::from_module(module);
        match self {
            Self::Fp32 { lora, .. } | Self::Bf16 { lora, .. } | Self::Packed { lora, .. } => {
                *lora = Some(host)
            }
        }
        Ok(())
    }

    fn lora(&self) -> Option<&DeepseekV41LoraHost> {
        match self {
            Self::Fp32 { lora, .. } | Self::Bf16 { lora, .. } | Self::Packed { lora, .. } => {
                lora.as_ref()
            }
        }
    }

    pub(super) fn validate(&self) -> Result<()> {
        let [rows, cols] = self.shape();
        let count = rows
            .checked_mul(cols)
            .context("V4.1 linear weight count overflow")?;
        if rows == 0 || cols == 0 || count > u32::MAX as usize {
            bail!("V4.1 linear weight geometry is invalid");
        }
        match self {
            Self::Fp32 { values, .. } | Self::Bf16 { values, .. } => {
                if values.len() != count || values.iter().any(|v| !v.is_finite()) {
                    bail!("V4.1 dense linear weight payload must have the exact shape and finite values");
                }
                if matches!(self, Self::Bf16 { .. })
                    && values.iter().any(|v| v.to_bits() & 0xffff != 0)
                {
                    bail!("V4.1 BF16 linear storage contains non-BF16 values");
                }
            }
            Self::Packed { weight, .. }
                if weight.format() == DeepseekV41WeightFormat::EngramFp8Row32 =>
            {
                bail!("V4.1 row-scaled Engram tables are not linear weights");
            }
            _ => {}
        }
        if let Some(lora) = self.lora() {
            if lora.rank == 0
                || !lora.scale.is_finite()
                || lora.a.len() != lora.rank * cols
                || lora.b.len() != rows * lora.rank
                || lora.a.iter().chain(lora.b.iter()).any(|v| !v.is_finite())
            {
                bail!("V4.1 LoRA payload does not match its projection geometry");
            }
        }
        Ok(())
    }
    pub fn upload(self, device: &VulkanDevice) -> Result<DeepseekV41ResidentLinear> {
        let mut commands = device.compute_batch()?;
        let weight = self.record_upload(device, &mut commands)?;
        commands.submit_and_wait()?;
        Ok(weight)
    }

    pub(crate) fn record_upload(
        self,
        device: &VulkanDevice,
        commands: &mut vulkan::ComputeBatch,
    ) -> Result<DeepseekV41ResidentLinear> {
        self.validate()?;
        let [rows, cols] = self.shape();
        let lora = self.lora().cloned();
        let storage = match self {
            Self::Fp32 { values, .. } => LinearStorage::Dense(
                GpuBuffer::from_f32_recorded(device, commands, &values)?,
                false,
            ),
            Self::Bf16 { values, .. } => LinearStorage::Dense(
                GpuBuffer::from_f32_recorded(device, commands, &values)?,
                true,
            ),
            Self::Packed { weight, .. } => LinearStorage::Packed(
                weight.record_upload(device, commands)?,
                weight.format().block_size(),
            ),
        };
        let lora = match lora {
            Some(lora) => Some(DeepseekV41ResidentLora {
                a: GpuBuffer::from_f32_recorded(device, commands, &lora.a)?,
                b: GpuBuffer::from_f32_recorded(device, commands, &lora.b)?,
                rank: lora.rank,
                scale: lora.scale,
            }),
            None => None,
        };
        Ok(DeepseekV41ResidentLinear {
            rows,
            cols,
            storage,
            lora,
        })
    }
}
enum LinearStorage {
    Dense(GpuBuffer, bool),
    Packed(DeepseekV41ResidentPackedWeights, usize),
}
/// One resident PEFT module: FP32 low-rank factors on the device. The base
/// payload is untouched, so the released packed bytes stay bit-identical.
pub struct DeepseekV41ResidentLora {
    a: GpuBuffer,
    b: GpuBuffer,
    rank: usize,
    scale: f32,
}

pub struct DeepseekV41ResidentLinear {
    rows: usize,
    cols: usize,
    storage: LinearStorage,
    lora: Option<DeepseekV41ResidentLora>,
}
impl DeepseekV41ResidentLinear {
    /// Borrow an existing FP32 matrix for a component tape without copying
    /// its payload or changing the original projection's arithmetic.
    pub(super) fn fp32_view(buffer: &GpuBuffer, rows: usize, cols: usize) -> Result<Self> {
        let elements = rows
            .checked_mul(cols)
            .context("V4.1 FP32 projection view overflow")?;
        if rows == 0
            || cols == 0
            || elements > u32::MAX as usize
            || buffer.f32_capacity() < elements
        {
            bail!("V4.1 FP32 projection view geometry is invalid");
        }
        Ok(Self {
            rows,
            cols,
            storage: LinearStorage::Dense(buffer.clone(), false),
            lora: None,
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }

    pub fn lora_rank(&self) -> Option<usize> {
        self.lora.as_ref().map(|lora| lora.rank)
    }

    pub(super) fn output_precision(&self) -> DeepseekV41WeightOutput {
        match &self.storage {
            LinearStorage::Dense(_, false) => DeepseekV41WeightOutput::Fp32,
            _ => DeepseekV41WeightOutput::Bf16,
        }
    }

    /// Dense storage fixes its input dtype. Packed GEMMs preserve the source
    /// stream's rounding before quantizing it.
    pub(super) fn input_precision(
        &self,
        stream: DeepseekV41WeightOutput,
    ) -> DeepseekV41WeightOutput {
        match &self.storage {
            LinearStorage::Dense(_, _) => self.output_precision(),
            LinearStorage::Packed(_, _) => stream,
        }
    }
}

pub struct DeepseekV41MixedLinearKernel {
    device: VulkanDevice,
    dense: LinearKernel,
    packed: DeepseekV41PackedWeightKernel,
    cast: vulkan::ComputeKernel,
    lora_add: vulkan::ComputeKernel,
}
impl DeepseekV41MixedLinearKernel {
    pub fn new(device: VulkanDevice) -> Result<Self> {
        Ok(Self {
            dense: LinearKernel::new(&device)?,
            packed: DeepseekV41PackedWeightKernel::new(device.clone())?,
            cast: vulkan::ComputeKernel::new(
                &device,
                BF16_CAST,
                2,
                std::mem::size_of::<CastPush>() as u32,
            )?,
            lora_add: vulkan::ComputeKernel::new(
                &device,
                LORA_ADD,
                3,
                std::mem::size_of::<LoraAddPush>() as u32,
            )?,
            device,
        })
    }
    /// BF16 values are promoted to FP32 storage after rounding; FP32 boundaries
    /// retain the input bits. No CPU activation readback occurs.
    pub fn cast_buffer(
        &self,
        input: &GpuBuffer,
        elements: usize,
        precision: DeepseekV41WeightOutput,
    ) -> Result<GpuBuffer> {
        if elements == 0
            || elements > u32::MAX as usize
            || input.f32_capacity() < elements
            || !input.belongs_to_device(&self.device)
        {
            bail!("V4.1 precision boundary geometry or device mismatch");
        }
        if precision == DeepseekV41WeightOutput::Fp32 {
            return Ok(input.clone());
        }
        let mut commands = self.device.compute_batch()?;
        let output = self.record_cast_buffer(&mut commands, input, elements, precision)?;
        commands.submit()?;
        Ok(output)
    }

    /// Record-only form of [`Self::cast_buffer`]; an FP32 boundary records no
    /// work and returns the input view unchanged.
    pub fn record_cast_buffer(
        &self,
        commands: &mut vulkan::ComputeBatch,
        input: &GpuBuffer,
        elements: usize,
        precision: DeepseekV41WeightOutput,
    ) -> Result<GpuBuffer> {
        if elements == 0
            || elements > u32::MAX as usize
            || input.f32_capacity() < elements
            || !input.belongs_to_device(&self.device)
        {
            bail!("V4.1 precision boundary geometry or device mismatch");
        }
        if precision == DeepseekV41WeightOutput::Fp32 {
            return Ok(input.clone());
        }
        let output = GpuBuffer::zeros_u32(&self.device, elements)?;
        let push = CastPush {
            elements: elements as u32,
        };
        let groups = elements.div_ceil(64);
        self.cast.record_dispatch(
            commands,
            &[input, &output],
            bytemuck::bytes_of(&push),
            [groups.min(65_535) as u32, groups.div_ceil(65_535) as u32, 1],
        )?;
        Ok(output)
    }
    /// Materialize one resident packed matrix with its UE8M0 weight scales
    /// applied, without rewriting the packed payload. This is the exact matrix
    /// the released GEMM contracts against, and therefore the matrix the
    /// straight-through input adjoint of a frozen packed projection uses.
    pub fn decode_packed(
        &self,
        weight: &DeepseekV41ResidentPackedWeights,
        output: DeepseekV41WeightOutput,
    ) -> Result<GpuBuffer> {
        self.packed.dequantize(weight, output)
    }

    /// Dense F32/BF16 storage requires the corresponding input precision.
    /// Packed linear weights quantize that explicit input and emit BF16, as in
    /// the released fp8_gemm/fp4_gemm default output contract.
    pub fn apply(
        &self,
        input: &GpuBuffer,
        rows: usize,
        weight: &DeepseekV41ResidentLinear,
        precision: DeepseekV41WeightOutput,
    ) -> Result<GpuBuffer> {
        let mut commands = self.device.compute_batch()?;
        let output = self.record_apply(&mut commands, input, rows, weight, precision)?;
        commands.submit()?;
        Ok(output)
    }

    /// Record-only form of [`Self::apply`]: the same cast, projection and PEFT
    /// dispatches in the same order, recorded into the caller's batch.
    pub fn record_apply(
        &self,
        commands: &mut vulkan::ComputeBatch,
        input: &GpuBuffer,
        rows: usize,
        weight: &DeepseekV41ResidentLinear,
        precision: DeepseekV41WeightOutput,
    ) -> Result<GpuBuffer> {
        self.record_apply_prepared(commands, input, rows, weight, precision, None)
    }

    /// Projections of the same tensor share its rounding and, when their
    /// activation block sizes agree, its quantization. LoRA still sees the
    /// original module input; output casts remain separate and unchanged.
    pub(super) fn record_apply_pair(
        &self,
        commands: &mut vulkan::ComputeBatch,
        input: &GpuBuffer,
        rows: usize,
        first: &DeepseekV41ResidentLinear,
        second: &DeepseekV41ResidentLinear,
        precision: DeepseekV41WeightOutput,
    ) -> Result<(GpuBuffer, GpuBuffer)> {
        if first.cols != second.cols {
            bail!("V4.1 paired projections require the same input width");
        }
        let elements = rows
            .checked_mul(first.cols)
            .context("V4.1 paired input overflow")?;
        let rounded = self.record_cast_buffer(commands, input, elements, precision)?;
        let act = match (&first.storage, &second.storage) {
            (LinearStorage::Packed(_, block), LinearStorage::Packed(_, other))
                if block == other =>
            {
                Some(
                    self.packed
                        .record_activation_input(commands, &rounded, rows, first.cols, *block)?,
                )
            }
            _ => None,
        };
        let prepared = Some((&rounded, act.as_ref()));
        let a = self.record_apply_prepared(commands, input, rows, first, precision, prepared)?;
        let b = self.record_apply_prepared(commands, input, rows, second, precision, prepared)?;
        Ok((a, b))
    }

    fn record_apply_prepared(
        &self,
        commands: &mut vulkan::ComputeBatch,
        input: &GpuBuffer,
        rows: usize,
        weight: &DeepseekV41ResidentLinear,
        precision: DeepseekV41WeightOutput,
        prepared: Option<(&GpuBuffer, Option<&super::DeepseekV41QuantizedActivation>)>,
    ) -> Result<GpuBuffer> {
        let elements = rows
            .checked_mul(weight.cols)
            .context("V4.1 mixed projection input overflow")?;
        if rows == 0
            || rows > 65_535
            || rows
                .checked_mul(weight.rows)
                .is_none_or(|n| n > u32::MAX as usize)
        {
            bail!("V4.1 mixed projection output geometry is invalid");
        }
        if let LinearStorage::Dense(_, bf16) = &weight.storage {
            if *bf16 != (precision == DeepseekV41WeightOutput::Bf16) {
                bail!("V4.1 dense linear input/storage precision mismatch");
            }
        }
        let resident = match &weight.storage {
            LinearStorage::Dense(w, _) => w.belongs_to_device(&self.device),
            LinearStorage::Packed(w, _) => {
                let (data, scales) = w.buffers();
                data.belongs_to_device(&self.device) && scales.belongs_to_device(&self.device)
            }
        };
        if !resident {
            bail!("V4.1 mixed linear weights belong to another Vulkan device");
        }
        let x = match prepared {
            Some((rounded, _)) => rounded.clone(),
            None => self.record_cast_buffer(commands, input, elements, precision)?,
        };
        let base = match &weight.storage {
            LinearStorage::Dense(w, bf16) => {
                let result = self.dense.record_apply_with_precision(
                    commands,
                    &self.device,
                    &x,
                    w,
                    rows,
                    weight.cols,
                    weight.rows,
                    *bf16,
                )?;
                if *bf16 {
                    self.record_cast_buffer(
                        commands,
                        &result,
                        rows * weight.rows,
                        DeepseekV41WeightOutput::Bf16,
                    )?
                } else {
                    result
                }
            }
            LinearStorage::Packed(w, block) => match prepared.and_then(|(_, act)| act) {
                Some(act) => self.packed.record_linear_activation(
                    commands,
                    act,
                    w,
                    DeepseekV41WeightOutput::Bf16,
                )?,
                None => self.packed.record_linear_buffer(
                    commands,
                    &x,
                    rows,
                    w,
                    *block,
                    DeepseekV41WeightOutput::Bf16,
                )?,
            },
        };
        let Some(lora) = &weight.lora else {
            return Ok(base);
        };
        // PEFT applies the branch to the module input, not to the quantized
        // activation: A consumes the stream tensor, B produces the delta, and
        // the scaled FP32 sum is rounded once at the existing output boundary.
        let hidden = self.dense.record_apply_with_precision(
            commands,
            &self.device,
            input,
            &lora.a,
            rows,
            weight.cols,
            lora.rank,
            false,
        )?;
        let delta = self.dense.record_apply_with_precision(
            commands,
            &self.device,
            &hidden,
            &lora.b,
            rows,
            lora.rank,
            weight.rows,
            false,
        )?;
        let outputs = rows * weight.rows;
        let push = LoraAddPush {
            elements: outputs as u32,
            scale: lora.scale,
            bf16: u32::from(weight.output_precision() == DeepseekV41WeightOutput::Bf16),
        };
        let output = GpuBuffer::zeros_u32(&self.device, outputs)?;
        let groups = outputs.div_ceil(64);
        self.lora_add.record_dispatch(
            commands,
            &[&base, &delta, &output],
            bytemuck::bytes_of(&push),
            [groups.min(65_535) as u32, groups.div_ceil(65_535) as u32, 1],
        )?;
        Ok(output)
    }
}

#[cfg(test)]
mod pair_tests {
    use super::super::DeepseekV41WeightFormat;
    use super::*;

    fn weight(
        kind: &str,
        output: usize,
        precision: DeepseekV41WeightOutput,
    ) -> Result<DeepseekV41LinearWeights> {
        let cols = 128;
        let lora = Some(DeepseekV41LoraHost {
            rank: 2,
            scale: 0.75,
            a: (0..2 * cols)
                .map(|i| (i as i32 % 19 - 9) as f32 * 0.013)
                .collect::<Vec<_>>()
                .into(),
            b: (0..2 * output)
                .map(|i| (i as i32 % 11 - 5) as f32 * 0.071)
                .collect::<Vec<_>>()
                .into(),
        });
        if kind == "dense" {
            let mut values = (0..output * cols)
                .map(|i| (i as i32 % 31 - 15) as f32 * 0.019)
                .collect::<Vec<_>>();
            if precision == DeepseekV41WeightOutput::Bf16 {
                for v in &mut values {
                    let b = v.to_bits();
                    *v = f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff0000);
                }
                return Ok(DeepseekV41LinearWeights::Bf16 {
                    rows: output,
                    cols,
                    values,
                    lora,
                });
            }
            return Ok(DeepseekV41LinearWeights::Fp32 {
                rows: output,
                cols,
                values,
                lora,
            });
        }
        let format = match kind {
            "fp4" => DeepseekV41WeightFormat::Fp4Row32,
            "fp8_32" => DeepseekV41WeightFormat::Fp8Block32,
            _ => DeepseekV41WeightFormat::Fp8Block128,
        };
        let n = output * cols / if kind == "fp4" { 2 } else { 1 };
        let data = (0..n)
            .map(|i| {
                if kind == "fp4" {
                    (i * 37 + 19) as u8
                } else {
                    ((i * 13 + 9) % 126) as u8
                }
            })
            .collect();
        let [sr, sc] = format.scale_shape(output, cols);
        Ok(DeepseekV41LinearWeights::Packed {
            weight: DeepseekV41PackedWeights::new(output, cols, format, data, vec![120; sr * sc])?,
            lora,
        })
    }

    #[test]
    fn paired_preparation_preserves_projection_bits_lora_and_block_fallbacks() -> Result<()> {
        let device = VulkanDevice::new()?;
        let kernel = DeepseekV41MixedLinearKernel::new(device.clone())?;
        for precision in [DeepseekV41WeightOutput::Fp32, DeepseekV41WeightOutput::Bf16] {
            for (left, right) in [
                ("fp4", "fp4"),
                ("fp8_32", "fp8_32"),
                ("fp8_128", "fp8_128"),
                ("fp4", "fp8_32"),
                ("fp4", "fp8_128"),
                ("dense", "fp4"),
                ("dense", "dense"),
            ] {
                let a = weight(left, 5, precision)?.upload(&device)?;
                let b = weight(right, 7, precision)?.upload(&device)?;
                for rows in [1, 2, 5] {
                    // Deliberately non-BF16 inputs also exercise the raw LoRA branch.
                    let values = (0..rows * 128)
                        .map(|i| (i as i32 % 47 - 23) as f32 * 0.0173)
                        .collect::<Vec<_>>();
                    let input = GpuBuffer::from_f32(&device, &values)?;
                    let mut old = device.compute_batch()?;
                    let ea = kernel.record_apply(&mut old, &input, rows, &a, precision)?;
                    let eb = kernel.record_apply(&mut old, &input, rows, &b, precision)?;
                    let before = old.dispatch_count();
                    old.submit()?;
                    let mut new = device.compute_batch()?;
                    let (aa, ab) =
                        kernel.record_apply_pair(&mut new, &input, rows, &a, &b, precision)?;
                    let saved = usize::from(precision == DeepseekV41WeightOutput::Bf16)
                        + usize::from(
                            left != "dense"
                                && right != "dense"
                                && (left == "fp8_128") == (right == "fp8_128"),
                        );
                    assert_eq!(
                        new.dispatch_count() + saved,
                        before,
                        "{left}/{right} {precision:?} rows={rows}"
                    );
                    new.submit()?;
                    assert_eq!(
                        aa.read_u32(rows * 5)?,
                        ea.read_u32(rows * 5)?,
                        "first {left}/{right} {precision:?} rows={rows}"
                    );
                    assert_eq!(
                        ab.read_u32(rows * 7)?,
                        eb.read_u32(rows * 7)?,
                        "second {left}/{right} {precision:?} rows={rows}"
                    );
                }
            }
        }
        Ok(())
    }
}
