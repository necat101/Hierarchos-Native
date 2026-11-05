//! Selected packed experts share one immutable arena and one compute grid.
//! The staged kernels bind the same arena with byte offsets, so A/B and the
//! retained training tapes require neither duplicate weights nor a readback.
use super::{DeepseekV41MixedExpertWeights, DeepseekV41MoeConfig, MixedExpert};
use crate::deepseek_v41::{
    DeepseekV41ActivationQuantizer, DeepseekV41LinearWeights, DeepseekV41QuantMode,
    DeepseekV41ResidentLinear, DeepseekV41ScaleMode, DeepseekV41WeightOutput,
};
use crate::{vulkan, GpuBuffer, VulkanDevice};
use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};

const FUSED: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_expert_fused.spv");
const MAX_INTER: usize = 2304;
// Core Vulkan guarantees this storage-buffer range. Larger resident layers
// retain separate matrices rather than introducing an optional capability.
const MAX_ARENA: usize = 128 * 1024 * 1024;
const SHARED_BYTES: u32 = ((128 + MAX_INTER + MAX_INTER / 32 + 1) * 4) as u32;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FusedExpertPush {
    rows: u32,
    hidden: u32,
    inter: u32,
    topk: u32,
    act_block: u32,
    bf16: u32,
    limit: f32,
}

pub(super) struct PackedExpertBatch {
    arena: GpuBuffer,
    metadata: GpuBuffer,
    quantizer: DeepseekV41ActivationQuantizer,
    kernel: vulkan::ComputeKernel,
    block: usize,
    pub(super) enabled: bool,
}

impl PackedExpertBatch {
    pub(super) fn upload(
        device: &VulkanDevice,
        c: &DeepseekV41MoeConfig,
        weights: &[DeepseekV41MixedExpertWeights],
    ) -> Result<Option<(Vec<MixedExpert>, Self)>> {
        if c.intermediate_size > MAX_INTER
            || device.max_compute_shared_memory_bytes() < SHARED_BYTES
            || !device.supports_compute_work_group_size([64, 1, 1])
        {
            return Ok(None);
        }
        let mut specs = Vec::new();
        let mut total = 0usize;
        let mut block = None;
        for expert in weights {
            for (projection, w) in [&expert.w1, &expert.w3, &expert.w2].into_iter().enumerate() {
                let DeepseekV41LinearWeights::Packed { weight, lora: None } = w else {
                    return Ok(None);
                };
                if projection != 2 {
                    let b = weight.format().block_size();
                    if block.is_some_and(|previous| previous != b) {
                        return Ok(None);
                    }
                    block = Some(b);
                }
                for len in [weight.data().len(), weight.scales().len()] {
                    total = total
                        .checked_add(len.div_ceil(4) * 4)
                        .context("V4.1 expert arena size overflow")?;
                }
                specs.push(weight);
            }
        }
        if total == 0 || total > MAX_ARENA {
            return Ok(None);
        }
        let mut bytes = Vec::with_capacity(total);
        let mut metadata = Vec::with_capacity(specs.len() * 3);
        for weight in &specs {
            metadata.push(bytes.len() as u32);
            bytes.extend_from_slice(weight.data());
            bytes.resize(bytes.len().div_ceil(4) * 4, 0);
            metadata.push(bytes.len() as u32);
            bytes.extend_from_slice(weight.scales());
            bytes.resize(bytes.len().div_ceil(4) * 4, 0);
            metadata.push(weight.format().code());
        }
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let arena = GpuBuffer::from_u32(device, &words)?;
        let mut experts = Vec::with_capacity(weights.len());
        for (expert, projections) in specs.chunks_exact(3).enumerate() {
            let resident = |projection: usize| {
                let offset = expert * 9 + projection * 3;
                DeepseekV41ResidentLinear::packed_arena(
                    projections[projection],
                    &arena,
                    metadata[offset],
                    metadata[offset + 1],
                )
            };
            experts.push(MixedExpert {
                w1: resident(0)?,
                w3: resident(1)?,
                w2: resident(2)?,
            });
        }
        Ok(Some((
            experts,
            Self {
                metadata: GpuBuffer::from_u32(device, &metadata)?,
                arena,
                quantizer: DeepseekV41ActivationQuantizer::new(device.clone())?,
                kernel: vulkan::ComputeKernel::new(
                    device,
                    crate::deepseek_v41::cpu_math::selected(device, FUSED),
                    8,
                    std::mem::size_of::<FusedExpertPush>() as u32,
                )?,
                block: block.unwrap(),
                enabled: match std::env::var("HIERARCHOS_VULKAN_V41_FUSED_EXPERTS").as_deref() {
                    Ok("0") => false,
                    Ok("1") => true,
                    _ => vulkan::is_intel_gen9_pci_id(device.vendor_id(), device.device_id()),
                },
            },
        )))
    }

    pub(super) fn record(
        &self,
        commands: &mut vulkan::ComputeBatch,
        x: &GpuBuffer,
        rows: usize,
        c: &DeepseekV41MoeConfig,
        precision: DeepseekV41WeightOutput,
        indices: &GpuBuffer,
        route: &GpuBuffer,
        counts: &GpuBuffer,
        routed: &GpuBuffer,
    ) -> Result<()> {
        let act = self.quantizer.record_quantize_buffer(
            commands,
            x,
            rows,
            c.hidden_size,
            self.block,
            DeepseekV41QuantMode::Fp8E4M3,
            DeepseekV41ScaleMode::Ue8m0,
        )?;
        let push = FusedExpertPush {
            rows: rows as u32,
            hidden: c.hidden_size as u32,
            inter: c.intermediate_size as u32,
            topk: c.activated as u32,
            act_block: self.block as u32,
            bf16: u32::from(precision == DeepseekV41WeightOutput::Bf16),
            limit: c.swiglu_limit,
        };
        self.kernel.record_dispatch(
            commands,
            &[
                &act.values,
                &act.scales,
                &self.arena,
                &self.metadata,
                indices,
                route,
                counts,
                routed,
            ],
            bytemuck::bytes_of(&push),
            [(rows * c.activated) as u32, 1, 1],
        )
    }
}
