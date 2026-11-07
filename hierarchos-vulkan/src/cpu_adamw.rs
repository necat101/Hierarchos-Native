//! Vulkan implementation of the pinned PyTorch single-tensor CPU AdamW policy.
use crate::{vulkan, AdamWHyperParams, GpuBuffer, VulkanDevice};
use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CpuAdamWHyperParams {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}
impl CpuAdamWHyperParams {
    pub fn validate(self) -> Result<()> {
        if !self.lr.is_finite()
            || self.lr < 0.
            || !self.eps.is_finite()
            || self.eps < 0.
            || !self.weight_decay.is_finite()
            || self.weight_decay < 0.
            || !(0.0..1.0).contains(&self.beta1)
            || !(0.0..1.0).contains(&self.beta2)
        {
            bail!("invalid CPU-policy AdamW hyperparameters");
        }
        self.as_f32().validate()
    }
    pub fn as_f32(self) -> AdamWHyperParams {
        AdamWHyperParams {
            lr: self.lr as f32,
            beta1: self.beta1 as f32,
            beta2: self.beta2 as f32,
            eps: self.eps as f32,
            weight_decay: self.weight_decay as f32,
        }
    }
}
impl From<AdamWHyperParams> for CpuAdamWHyperParams {
    fn from(h: AdamWHyperParams) -> Self {
        Self {
            lr: h.lr as f64,
            beta1: h.beta1 as f64,
            beta2: h.beta2 as f64,
            eps: h.eps as f64,
            weight_decay: h.weight_decay as f64,
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Push {
    len: u32,
    vector_end: u32,
    weight: f32,
    beta2: f32,
    complement2: f32,
    step_size: f32,
    bias2_sqrt: f32,
    eps: f32,
    decay: f32,
}

pub struct CpuAdamW {
    device: VulkanDevice,
    kernel: vulkan::ComputeKernel,
    rdna3_seeds: Option<GpuBuffer>,
}
impl CpuAdamW {
    /// Qualified Windows Zen4/MKL schedule, RDNA3 only.
    pub(crate) fn new_rdna3_avx512(device: &VulkanDevice) -> Result<Self> {
        if !crate::vendor::rdna3_avx512_variants(device, crate::transformer::aten_vector_width())
            || !crate::vendor::rdna3_zen4_cpu_profile()
        {
            bail!("RDNA3 AVX512 AdamW requires the qualified hardware profile");
        }
        Ok(Self {
            device: device.clone(),
            rdna3_seeds: Some(GpuBuffer::from_f32(
                device,
                &include_bytes!("../shaders/rdna3_zen4_rsqrt_seed.bin")
                    .chunks_exact(4)
                    .map(|b| f32::from_bits(u32::from_le_bytes(b.try_into().unwrap())))
                    .collect::<Vec<_>>(),
            )?),
            kernel: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../shaders/image_text_adamw_avx512_amd_rdna3.spv"),
                5,
                std::mem::size_of::<Push>() as u32,
            )?,
        })
    }

    pub fn new(device: &VulkanDevice) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            rdna3_seeds: None,
            kernel: vulkan::ComputeKernel::new(
                device,
                include_bytes!("../shaders/adamw_cpu_single_tensor.spv"),
                4,
                std::mem::size_of::<Push>() as u32,
            )?,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_step(
        &self,
        commands: &mut vulkan::ComputeBatch,
        parameter: &GpuBuffer,
        gradient: &GpuBuffer,
        moment: &GpuBuffer,
        square: &GpuBuffer,
        len: usize,
        step: u32,
        hyper: CpuAdamWHyperParams,
        decay: bool,
    ) -> Result<()> {
        hyper.validate()?;
        let exponent = i32::try_from(step).context("CPU AdamW step exceeds coefficient range")?;
        if step == 0
            || len == 0
            || len > u32::MAX as usize
            || [parameter, gradient, moment, square]
                .iter()
                .any(|b| b.f32_capacity() < len)
        {
            bail!("invalid CPU AdamW step or buffer geometry");
        }
        let block = if self.rdna3_seeds.is_some() {
            16
        } else {
            2 * crate::transformer::aten_vector_width() as usize
        };
        let push = Push {
            len: len as u32,
            vector_end: (len / block * block) as u32,
            weight: (1. - hyper.beta1) as f32,
            beta2: hyper.beta2 as f32,
            complement2: (1. - hyper.beta2) as f32,
            step_size: (hyper.lr / (1. - hyper.beta1.powi(exponent))) as f32,
            bias2_sqrt: (1. - hyper.beta2.powi(exponent)).sqrt() as f32,
            eps: hyper.eps as f32,
            decay: if decay {
                (1. - hyper.lr * hyper.weight_decay) as f32
            } else {
                1.
            },
        };
        if let Some(seeds) = self.rdna3_seeds.as_ref() {
            return self.kernel.record_dispatch(
                commands,
                &[parameter, gradient, moment, square, seeds],
                bytemuck::bytes_of(&push),
                [((len + 255) / 256) as u32, 1, 1],
            );
        }
        self.kernel.record_dispatch(
            commands,
            &[parameter, gradient, moment, square],
            bytemuck::bytes_of(&push),
            [((len + 255) / 256) as u32, 1, 1],
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        parameter: &GpuBuffer,
        gradient: &GpuBuffer,
        moment: &GpuBuffer,
        square: &GpuBuffer,
        len: usize,
        step: u32,
        hyper: CpuAdamWHyperParams,
        decay: bool,
    ) -> Result<()> {
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.record_step(
            &mut commands,
            parameter,
            gradient,
            moment,
            square,
            len,
            step,
            hyper,
            decay,
        )?;
        commands.submit()
    }
}
