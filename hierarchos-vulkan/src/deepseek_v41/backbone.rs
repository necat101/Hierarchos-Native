//! DeepSeek V4.1 shifted Hyper-Connections and MoE, wired around the attention
//! orchestrator. The released block carries `hc_mult` residual copies, derives
//! the pre/post/comb mix coefficients from the stream itself, and runs a
//! top-k routed plus one shared SwiGLU expert between two Hyper-Connection
//! stages. The default component contract is FP32; checkpoint construction also
//! supports resident packed experts with explicit BF16 GEMM boundaries inside
//! that backbone. Production precision and grouped serving GEMMs are unqualified.
//!
//! The caller supplies the already `hc_mult`-expanded embedding stream, and the
//! final collapse returns one hidden vector per token. Engram is optional;
//! embedding and the vocabulary head stay outside this module. The expert sweep dispatches
//! once per expert and lets unrelated workgroups exit; that is a correctness
//! baseline, not a grouped-GEMM serving path.

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Mutex;

use crate::{vulkan, GpuBuffer, VulkanDevice};

use super::indexer::{LinearKernel, NormKernel};
use super::{
    DeepseekV41AttentionLayerWeights, DeepseekV41AttentionOrchestrator,
    DeepseekV41AttentionRuntimeConfig, DeepseekV41Checkpoint, DeepseekV41Config,
    DeepseekV41EngramRuntime, DeepseekV41LayerPlan, DeepseekV41LinearWeights,
    DeepseekV41MixedLinearKernel, DeepseekV41ResidentLinear, DeepseekV41TerminalAdjoints,
    DeepseekV41WeightOutput,
};

const HC_SPLIT: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_hc_split.spv");
const HC_PRE: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_hc_pre.spv");
const HC_POST: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_hc_post.spv");
const MOE_GATE: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_moe_gate.spv");
const MOE_EXPERT: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_moe_expert.spv");
const MOE_COMBINE: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_moe_combine.spv");
const MOE_PACK: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_moe_pack.spv");
const MOE_SCATTER: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_moe_scatter.spv");
const MIXED_SWIGLU: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_mixed_swiglu.spv");
const HC_PRE_INPUT_GRAD: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_hc_pre_input_grad.spv");
const HC_SPLIT_GRAD: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_hc_split_grad.spv");
const HC_MIXES_INPUT_GRAD: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_hc_mixes_input_grad.spv");
const HC_POST_INPUT_GRAD: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_hc_post_input_grad.spv");
const HC_POST_COMB_GRAD: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_hc_post_comb_grad.spv");
const MOE_COMBINE_GRAD: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_moe_combine_grad.spv");
const MOE_EXPERT_GRAD: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_moe_expert_grad.spv");
const MOE_GATE_GRAD: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_moe_gate_grad.spv");
const MOE_INPUT_GRAD_COMBINE: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_moe_input_grad_combine.spv");

const MAX_HC: usize = 16;
const MAX_EXPERTS: usize = 512;
const MAX_ACTIVATED: usize = 16;

#[path = "backbone_tape.rs"]
mod retained;
pub(super) use retained::ModelAdjoints;

#[path = "mixed_moe_grad.rs"]
mod mixed_moe_grad;
pub use mixed_moe_grad::{
    DeepseekV41MixedMoeAdjoints, DeepseekV41MixedMoeGradients, DeepseekV41MixedMoeTape,
};

#[cfg(test)]
#[path = "backbone_routing_tests.rs"]
mod routing_tests;

#[cfg(test)]
#[path = "block_grad_tests.rs"]
mod block_grad_tests;

#[cfg(test)]
#[path = "backbone_router_width_tests.rs"]
mod router_width_tests;

// Observational GPU handles only, enabled explicitly by the model diagnostic.
// No readbacks or trace allocations exist in the production graph.
macro_rules! trace_stage {
    ($graph:expr, $layer:expr, $stage:expr, $values:expr, $len:expr) => {
        #[cfg(test)]
        if let Some(trace) = &mut $graph.trace {
            trace.push(($layer, $stage, $values.clone(), $len));
        }
    };
}

// A bit-exactness probe for release builds, where `trace_stage!` is compiled
// out: `HIERARCHOS_VULKAN_LAYER_DIGEST=1` prints an FNV-1a digest of each named
// stage so a kernel or submission change can be shown to leave every layer's
// arithmetic identical. Reading a buffer splits the layer session, so this is a
// diagnostic mode and never part of the shipping path.
fn digest_enabled() -> bool {
    std::env::var_os("HIERARCHOS_VULKAN_LAYER_DIGEST").is_some()
}

/// Whether to digest this stage. `HIERARCHOS_VULKAN_DIGEST_ONLY` narrows the
/// probe to one stage name, which also narrows where the read splits the
/// session; that is how a submission-ordering difference is localized.
fn digest_stage_enabled(stage: &str) -> bool {
    if !digest_enabled() {
        return false;
    }
    match std::env::var("HIERARCHOS_VULKAN_DIGEST_ONLY") {
        Ok(only) => only == stage,
        Err(_) => true,
    }
}

fn digest_of(buffer: &GpuBuffer, len: usize) -> Result<u64> {
    let values = buffer.read_f32(len)?;
    let mut hash = 0xcbf29ce484222325_u64;
    for value in &values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    Ok(hash)
}

macro_rules! stage_digest {
    ($graph:expr, $layer:expr, $stage:expr, $values:expr, $len:expr) => {
        if digest_stage_enabled($stage) {
            eprintln!(
                "v41_stage_digest layer={} stage={} digest={:016x}",
                $layer,
                $stage,
                digest_of(&$values, $len)?
            );
        }
    };
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HcSplitPush {
    rows: u32,
    hc: u32,
    dim: u32,
    mix_hc: u32,
    sinkhorn_iters: u32,
    eps: f32,
    norm_eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HcPrePush {
    rows: u32,
    hc: u32,
    dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MoeGatePush {
    rows: u32,
    dim: u32,
    experts: u32,
    topk: u32,
    score_func: u32,
    gate_temp: f32,
    norm_topk: u32,
    route_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MoeExpertPush {
    rows: u32,
    dim: u32,
    inter: u32,
    topk: u32,
    expert: u32,
    weighted: u32,
    swiglu_limit: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MoeCombinePush {
    rows: u32,
    dim: u32,
    topk: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HcMixesGradPush {
    rows: u32,
    hc: u32,
    dim: u32,
    mix_hc: u32,
    norm_eps: f32,
}

fn checked_size(label: &str, sizes: &[usize]) -> Result<usize> {
    let size = sizes
        .iter()
        .try_fold(1usize, |n, &s| n.checked_mul(s))
        .with_context(|| format!("V4.1 {label} size overflow"))?;
    if size == 0 || size > u32::MAX as usize {
        bail!("V4.1 {label} size must fit positive u32 addressing");
    }
    Ok(size)
}

fn validate_finite(label: &str, values: &[f32]) -> Result<()> {
    if !values.iter().all(|value| value.is_finite()) {
        bail!("V4.1 {label} must be finite");
    }
    Ok(())
}

/// The released block's two Hyper-Connection stages. Each stage projects the
/// flattened `hc_mult * hidden` stream into `(2 + hc_mult) * hc_mult`
/// coefficients and splits them into pre (collapse), post (expand), and comb
/// (residual mixing, Sinkhorn-normalized).
pub struct DeepseekV41HyperConnectionWeights {
    pub attn_fn: Vec<f32>,
    pub attn_base: Vec<f32>,
    pub attn_scale: Vec<f32>,
    pub ffn_fn: Vec<f32>,
    pub ffn_base: Vec<f32>,
    pub ffn_scale: Vec<f32>,
}

/// `[rows, hc]` pre/post coefficients and `[rows, hc, hc]` comb.
pub struct DeepseekV41HyperConnectionMixes {
    pub pre: GpuBuffer,
    pub post: GpuBuffer,
    pub comb: GpuBuffer,
    /// The raw `hc_fn` projection, before the per-token `rstd` multiply. The
    /// split adjoint recomputes the coefficients from this buffer, and retains
    /// it instead of trusting a cached post-scale value.
    pub projection: GpuBuffer,
}

impl DeepseekV41HyperConnectionMixes {
    /// Raw projection output, the input of the split stage.
    pub fn raw_projection(&self) -> &GpuBuffer {
        &self.projection
    }
}

/// Which of a block's two Hyper-Connection stages a call refers to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepseekV41HyperStage {
    Attention,
    FeedForward,
}

pub struct DeepseekV41HyperConnections {
    device: VulkanDevice,
    hidden: usize,
    hc: usize,
    mix_hc: usize,
    iterations: usize,
    eps: f32,
    norm_eps: f32,
    attn_fn: GpuBuffer,
    attn_base: GpuBuffer,
    attn_scale: GpuBuffer,
    ffn_fn: GpuBuffer,
    ffn_base: GpuBuffer,
    ffn_scale: GpuBuffer,
    linear: LinearKernel,
    split: vulkan::ComputeKernel,
    pre: vulkan::ComputeKernel,
    post: vulkan::ComputeKernel,
}

impl DeepseekV41HyperConnections {
    pub fn new(
        device: VulkanDevice,
        hidden: usize,
        hc: usize,
        iterations: usize,
        eps: f32,
        norm_eps: f32,
        weights: DeepseekV41HyperConnectionWeights,
    ) -> Result<Self> {
        if hidden == 0 || !(1..=MAX_HC).contains(&hc) {
            bail!("V4.1 hyper-connection dimensions are invalid");
        }
        let mix_hc = (2 + hc) * hc;
        let width = checked_size("hyper-connection stream", &[hc, hidden])?;
        if iterations == 0 || !eps.is_finite() || eps <= 0.0 {
            bail!("V4.1 hyper-connection sinkhorn configuration is invalid");
        }
        if !norm_eps.is_finite() || norm_eps <= 0.0 {
            bail!("V4.1 hyper-connection norm epsilon is invalid");
        }
        for (label, values, expected) in [
            ("attn_fn", &weights.attn_fn, mix_hc * width),
            ("attn_base", &weights.attn_base, mix_hc),
            ("attn_scale", &weights.attn_scale, 3),
            ("ffn_fn", &weights.ffn_fn, mix_hc * width),
            ("ffn_base", &weights.ffn_base, mix_hc),
            ("ffn_scale", &weights.ffn_scale, 3),
        ] {
            if values.len() != expected {
                bail!("V4.1 hyper-connection {label} geometry is invalid");
            }
            validate_finite(label, values)?;
        }
        Ok(Self {
            attn_fn: GpuBuffer::from_f32(&device, &weights.attn_fn)?,
            attn_base: GpuBuffer::from_f32(&device, &weights.attn_base)?,
            attn_scale: GpuBuffer::from_f32(&device, &weights.attn_scale)?,
            ffn_fn: GpuBuffer::from_f32(&device, &weights.ffn_fn)?,
            ffn_base: GpuBuffer::from_f32(&device, &weights.ffn_base)?,
            ffn_scale: GpuBuffer::from_f32(&device, &weights.ffn_scale)?,
            linear: LinearKernel::new(&device)?,
            split: vulkan::ComputeKernel::new(
                &device,
                HC_SPLIT,
                7,
                std::mem::size_of::<HcSplitPush>() as u32,
            )?,
            pre: vulkan::ComputeKernel::new(
                &device,
                HC_PRE,
                3,
                std::mem::size_of::<HcPrePush>() as u32,
            )?,
            post: vulkan::ComputeKernel::new(
                &device,
                HC_POST,
                5,
                std::mem::size_of::<HcPrePush>() as u32,
            )?,
            device,
            hidden,
            hc,
            mix_hc,
            iterations,
            eps,
            norm_eps,
        })
    }

    pub fn hc_mult(&self) -> usize {
        self.hc
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// The frozen `(hc_fn, hc_base, hc_scale)` triple of one stage.
    pub fn stage_weights(
        &self,
        stage: DeepseekV41HyperStage,
    ) -> (&GpuBuffer, &GpuBuffer, &GpuBuffer) {
        match stage {
            DeepseekV41HyperStage::Attention => (&self.attn_fn, &self.attn_base, &self.attn_scale),
            DeepseekV41HyperStage::FeedForward => (&self.ffn_fn, &self.ffn_base, &self.ffn_scale),
        }
    }

    /// Coefficients for the attention stage, from the stream before its collapse.
    pub fn attention_mixes(
        &self,
        stream: &GpuBuffer,
        rows: usize,
    ) -> Result<DeepseekV41HyperConnectionMixes> {
        self.mixes(
            stream,
            rows,
            &self.attn_fn,
            &self.attn_base,
            &self.attn_scale,
        )
    }

    /// Coefficients for the FFN stage, from the stream after attention expands.
    pub fn ffn_mixes(
        &self,
        stream: &GpuBuffer,
        rows: usize,
    ) -> Result<DeepseekV41HyperConnectionMixes> {
        self.mixes(stream, rows, &self.ffn_fn, &self.ffn_base, &self.ffn_scale)
    }

    fn mixes(
        &self,
        stream: &GpuBuffer,
        rows: usize,
        fn_weight: &GpuBuffer,
        base: &GpuBuffer,
        scale: &GpuBuffer,
    ) -> Result<DeepseekV41HyperConnectionMixes> {
        let width = checked_size("hyper-connection stream", &[rows, self.hc, self.hidden])?;
        if rows == 0
            || rows > 65_535
            || stream.f32_capacity() < width
            || !stream.belongs_to_device(&self.device)
        {
            bail!("V4.1 hyper-connection stream geometry or device mismatch");
        }
        let mixes = self.linear.apply(
            &self.device,
            stream,
            fn_weight,
            rows,
            self.hc * self.hidden,
            self.mix_hc,
        )?;
        let pre = GpuBuffer::zeros_f32(&self.device, rows * self.hc)?;
        let post = GpuBuffer::zeros_f32(&self.device, rows * self.hc)?;
        let comb = GpuBuffer::zeros_f32(&self.device, rows * self.hc * self.hc)?;
        let push = HcSplitPush {
            rows: rows as u32,
            hc: self.hc as u32,
            dim: self.hidden as u32,
            mix_hc: self.mix_hc as u32,
            sinkhorn_iters: self.iterations as u32,
            eps: self.eps,
            norm_eps: self.norm_eps,
        };
        let mut commands = self.device.compute_batch()?;
        self.split.record_dispatch(
            &mut commands,
            &[stream, &mixes, scale, base, &pre, &post, &comb],
            bytemuck::bytes_of(&push),
            [rows as u32, 1, 1],
        )?;
        commands.submit()?;
        Ok(DeepseekV41HyperConnectionMixes {
            pre,
            post,
            comb,
            projection: mixes,
        })
    }

    /// Collapses `[rows, hc, hidden]` through `pre` into `[rows, hidden]`.
    pub fn pre(&self, stream: &GpuBuffer, pre: &GpuBuffer, rows: usize) -> Result<GpuBuffer> {
        let width = checked_size("hyper-connection stream", &[rows, self.hc, self.hidden])?;
        if rows == 0
            || rows > 65_535
            || stream.f32_capacity() < width
            || pre.f32_capacity() < rows * self.hc
            || !stream.belongs_to_device(&self.device)
            || !pre.belongs_to_device(&self.device)
        {
            bail!("V4.1 hyper-connection collapse geometry or device mismatch");
        }
        let output = GpuBuffer::zeros_f32(
            &self.device,
            checked_size("collapse", &[rows, self.hidden])?,
        )?;
        let push = HcPrePush {
            rows: rows as u32,
            hc: self.hc as u32,
            dim: self.hidden as u32,
        };
        let mut commands = self.device.compute_batch()?;
        self.pre.record_dispatch(
            &mut commands,
            &[stream, pre, &output],
            bytemuck::bytes_of(&push),
            [self.hidden.div_ceil(64) as u32, rows as u32, 1],
        )?;
        commands.submit()?;
        Ok(output)
    }

    /// Expands `[rows, hidden]` back to `[rows, hc, hidden]`, mixing the
    /// residual stream through `comb`.
    pub fn post(
        &self,
        x: &GpuBuffer,
        residual: &GpuBuffer,
        post: &GpuBuffer,
        comb: &GpuBuffer,
        rows: usize,
    ) -> Result<GpuBuffer> {
        let width = checked_size("hyper-connection stream", &[rows, self.hc, self.hidden])?;
        let copy_rows = rows
            .checked_mul(self.hc)
            .context("V4.1 hyper-connection copy-row overflow")?;
        if rows == 0
            || copy_rows > 65_535
            || x.f32_capacity() < rows * self.hidden
            || residual.f32_capacity() < width
            || post.f32_capacity() < rows * self.hc
            || comb.f32_capacity() < rows * self.hc * self.hc
            || !x.belongs_to_device(&self.device)
            || !residual.belongs_to_device(&self.device)
            || !post.belongs_to_device(&self.device)
            || !comb.belongs_to_device(&self.device)
        {
            bail!("V4.1 hyper-connection expand geometry or device mismatch");
        }
        let output = GpuBuffer::zeros_f32(&self.device, width)?;
        let push = HcPrePush {
            rows: rows as u32,
            hc: self.hc as u32,
            dim: self.hidden as u32,
        };
        let mut commands = self.device.compute_batch()?;
        self.post.record_dispatch(
            &mut commands,
            &[x, residual, post, comb, &output],
            bytemuck::bytes_of(&push),
            [self.hidden.div_ceil(64) as u32, copy_rows as u32, 1],
        )?;
        commands.submit()?;
        Ok(output)
    }

    /// Initial mix: copy 0 is selected, matching `make_identity_pre_mix`.
    pub fn identity_pre_mix(&self, rows: usize) -> Result<GpuBuffer> {
        if rows == 0 || rows > 65_535 {
            bail!("V4.1 hyper-connection pre-mix row count is invalid");
        }
        let mut values = vec![0.0f32; rows * self.hc];
        for row in 0..rows {
            values[row * self.hc] = 1.0;
        }
        GpuBuffer::from_f32(&self.device, &values)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeepseekV41MoeScoreFunc {
    Softmax,
    Sigmoid,
    SqrtSoftplus,
}

impl DeepseekV41MoeScoreFunc {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "softmax" => Ok(Self::Softmax),
            "sigmoid" => Ok(Self::Sigmoid),
            "sqrtsoftplus" => Ok(Self::SqrtSoftplus),
            other => bail!("unsupported V4.1 MoE scoring function {other:?}"),
        }
    }

    fn code(self) -> u32 {
        match self {
            Self::Softmax => 0,
            Self::Sigmoid => 1,
            Self::SqrtSoftplus => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepseekV41MoeConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub experts: usize,
    pub activated: usize,
    pub score_func: DeepseekV41MoeScoreFunc,
    pub gate_temp: f32,
    pub norm_topk_prob: bool,
    pub route_scale: f32,
    pub swiglu_limit: f32,
}

impl DeepseekV41MoeConfig {
    pub fn from_text(text: &super::DeepseekV41TextConfig) -> Result<Self> {
        if text.n_shared_experts != 1 {
            bail!("V4.1 MoE expects exactly one shared expert");
        }
        Ok(Self {
            hidden_size: text.hidden_size,
            intermediate_size: text.moe_intermediate_size,
            experts: text.n_routed_experts,
            activated: text.num_experts_per_tok,
            score_func: DeepseekV41MoeScoreFunc::parse(&text.scoring_func)?,
            gate_temp: 1.0,
            norm_topk_prob: text.norm_topk_prob,
            route_scale: text.routed_scaling_factor as f32,
            swiglu_limit: text.swiglu_limit as f32,
        })
    }
}

pub struct DeepseekV41MoeWeights {
    /// `[experts, hidden]`
    pub gate_weight: Vec<f32>,
    /// `[experts]`; steers selection only.
    pub gate_bias: Vec<f32>,
    /// Per-expert `[intermediate, hidden]`.
    pub expert_w1: Vec<Vec<f32>>,
    /// Per-expert `[hidden, intermediate]`.
    pub expert_w2: Vec<Vec<f32>>,
    /// Per-expert `[intermediate, hidden]`.
    pub expert_w3: Vec<Vec<f32>>,
    pub shared_w1: Vec<f32>,
    pub shared_w2: Vec<f32>,
    pub shared_w3: Vec<f32>,
}

struct ExpertWeights {
    w1: GpuBuffer,
    w2: GpuBuffer,
    w3: GpuBuffer,
}

pub struct DeepseekV41Moe {
    device: VulkanDevice,
    config: DeepseekV41MoeConfig,
    gate_weight: GpuBuffer,
    gate_bias: GpuBuffer,
    cpu_sqrt_seeds: GpuBuffer,
    experts: Vec<ExpertWeights>,
    shared: ExpertWeights,
    gate: vulkan::ComputeKernel,
    expert: vulkan::ComputeKernel,
    combine: vulkan::ComputeKernel,
}

impl DeepseekV41Moe {
    pub fn new(
        device: VulkanDevice,
        config: DeepseekV41MoeConfig,
        weights: DeepseekV41MoeWeights,
    ) -> Result<Self> {
        let c = config;
        if c.hidden_size == 0 || c.intermediate_size == 0 || c.experts == 0 {
            bail!("V4.1 MoE dimensions are invalid");
        }
        if c.experts > MAX_EXPERTS || c.activated == 0 || c.activated > c.experts.min(MAX_ACTIVATED)
        {
            bail!("V4.1 MoE expert counts are invalid");
        }
        if !c.gate_temp.is_finite() || c.gate_temp <= 0.0 {
            bail!("V4.1 MoE gate temperature is invalid");
        }
        if !c.route_scale.is_finite() || !c.swiglu_limit.is_finite() || c.swiglu_limit < 0.0 {
            bail!("V4.1 MoE scaling is invalid");
        }
        let hidden = checked_size("MoE hidden", &[c.hidden_size])?;
        let inter = checked_size("MoE intermediate", &[c.intermediate_size])?;
        if weights.gate_weight.len() != c.experts * hidden
            || weights.gate_bias.len() != c.experts
            || weights.expert_w1.len() != c.experts
            || weights.expert_w2.len() != c.experts
            || weights.expert_w3.len() != c.experts
            || weights.shared_w1.len() != inter * hidden
            || weights.shared_w2.len() != hidden * inter
            || weights.shared_w3.len() != inter * hidden
        {
            bail!("V4.1 MoE parameter geometry is invalid");
        }
        for (index, expert) in weights.expert_w1.iter().enumerate() {
            if expert.len() != inter * hidden
                || weights.expert_w3[index].len() != inter * hidden
                || weights.expert_w2[index].len() != hidden * inter
            {
                bail!("V4.1 MoE expert {index} parameter geometry is invalid");
            }
            for (label, values) in [
                ("w1", expert),
                ("w2", &weights.expert_w2[index]),
                ("w3", &weights.expert_w3[index]),
            ] {
                validate_finite(&format!("MoE expert {index} {label}"), values)?;
            }
        }
        for (label, values) in [
            ("gate_weight", &weights.gate_weight),
            ("gate_bias", &weights.gate_bias),
            ("shared_w1", &weights.shared_w1),
            ("shared_w2", &weights.shared_w2),
            ("shared_w3", &weights.shared_w3),
        ] {
            validate_finite(label, values)?;
        }
        let mut experts = Vec::with_capacity(c.experts);
        for index in 0..c.experts {
            experts.push(ExpertWeights {
                w1: GpuBuffer::from_f32(&device, &weights.expert_w1[index])?,
                w2: GpuBuffer::from_f32(&device, &weights.expert_w2[index])?,
                w3: GpuBuffer::from_f32(&device, &weights.expert_w3[index])?,
            });
        }
        Ok(Self {
            gate_weight: GpuBuffer::from_f32(&device, &weights.gate_weight)?,
            gate_bias: GpuBuffer::from_f32(&device, &weights.gate_bias)?,
            cpu_sqrt_seeds: super::cpu_sqrt_seeds(&device)?,
            shared: ExpertWeights {
                w1: GpuBuffer::from_f32(&device, &weights.shared_w1)?,
                w2: GpuBuffer::from_f32(&device, &weights.shared_w2)?,
                w3: GpuBuffer::from_f32(&device, &weights.shared_w3)?,
            },
            gate: vulkan::ComputeKernel::new(
                &device,
                MOE_GATE,
                7,
                std::mem::size_of::<MoeGatePush>() as u32,
            )?,
            expert: vulkan::ComputeKernel::new(
                &device,
                MOE_EXPERT,
                9,
                std::mem::size_of::<MoeExpertPush>() as u32,
            )?,
            combine: vulkan::ComputeKernel::new(
                &device,
                MOE_COMBINE,
                4,
                std::mem::size_of::<MoeCombinePush>() as u32,
            )?,
            device,
            config: c,
            experts,
        })
    }

    pub fn config(&self) -> DeepseekV41MoeConfig {
        self.config
    }

    fn expert_push(&self, rows: usize, expert: u32, weighted: bool, topk: usize) -> MoeExpertPush {
        MoeExpertPush {
            rows: rows as u32,
            dim: self.config.hidden_size as u32,
            inter: self.config.intermediate_size as u32,
            topk: topk as u32,
            expert,
            weighted: weighted as u32,
            swiglu_limit: self.config.swiglu_limit,
        }
    }

    /// `[rows, hidden]` through one shared and `activated` routed experts.
    pub fn apply(&self, x: &GpuBuffer, rows: usize) -> Result<GpuBuffer> {
        let c = &self.config;
        if rows == 0
            || rows > 65_535
            || x.f32_capacity() < rows * c.hidden_size
            || !x.belongs_to_device(&self.device)
        {
            bail!("V4.1 MoE input geometry or device mismatch");
        }
        let topk = c.activated;
        let slots = rows.checked_mul(topk).context("V4.1 MoE slot overflow")?;
        if slots > 65_535 {
            bail!("V4.1 MoE routed slots exceed dispatch capacity");
        }
        let weights = GpuBuffer::zeros_f32(&self.device, slots)?;
        let indices = GpuBuffer::zeros_u32(&self.device, slots)?;
        let counts = GpuBuffer::zeros_u32(&self.device, c.experts)?;
        let gate_push = MoeGatePush {
            rows: rows as u32,
            dim: c.hidden_size as u32,
            experts: c.experts as u32,
            topk: topk as u32,
            score_func: c.score_func.code(),
            gate_temp: c.gate_temp,
            norm_topk: c.norm_topk_prob as u32,
            route_scale: c.route_scale,
        };
        let mut commands = self.device.compute_batch()?;
        self.gate.record_dispatch(
            &mut commands,
            &[
                x,
                &self.gate_weight,
                &self.gate_bias,
                &weights,
                &indices,
                &counts,
                &self.cpu_sqrt_seeds,
            ],
            bytemuck::bytes_of(&gate_push),
            [rows as u32, 1, 1],
        )?;
        commands.submit()?;

        let routed = GpuBuffer::zeros_f32(
            &self.device,
            checked_size("MoE routed", &[slots, c.hidden_size])?,
        )?;
        let scratch = GpuBuffer::zeros_f32(
            &self.device,
            checked_size("MoE scratch", &[slots, c.intermediate_size])?,
        )?;
        for (index, expert) in self.experts.iter().enumerate() {
            let push = self.expert_push(rows, index as u32, true, topk);
            let mut commands = self.device.compute_batch()?;
            self.expert.record_dispatch(
                &mut commands,
                &[
                    x, &expert.w1, &expert.w3, &expert.w2, &weights, &indices, &scratch, &routed,
                    &counts,
                ],
                bytemuck::bytes_of(&push),
                [slots as u32, 1, 1],
            )?;
            commands.submit()?;
        }

        let shared_out = GpuBuffer::zeros_f32(
            &self.device,
            checked_size("MoE shared", &[rows, c.hidden_size])?,
        )?;
        let shared_scratch = GpuBuffer::zeros_f32(
            &self.device,
            checked_size("MoE shared scratch", &[rows, c.intermediate_size])?,
        )?;
        let shared_push = self.expert_push(rows, u32::MAX, false, 1);
        let mut commands = self.device.compute_batch()?;
        self.expert.record_dispatch(
            &mut commands,
            &[
                x,
                &self.shared.w1,
                &self.shared.w3,
                &self.shared.w2,
                &weights,
                &indices,
                &shared_scratch,
                &shared_out,
                &counts,
            ],
            bytemuck::bytes_of(&shared_push),
            [rows as u32, 1, 1],
        )?;
        commands.submit()?;

        let output = GpuBuffer::zeros_f32(
            &self.device,
            checked_size("MoE output", &[rows, c.hidden_size])?,
        )?;
        let combine_push = MoeCombinePush {
            rows: rows as u32,
            dim: c.hidden_size as u32,
            topk: topk as u32,
        };
        let mut commands = self.device.compute_batch()?;
        self.combine.record_dispatch(
            &mut commands,
            &[&routed, &shared_out, &output, &indices],
            bytemuck::bytes_of(&combine_push),
            [c.hidden_size.div_ceil(64) as u32, rows as u32, 1],
        )?;
        commands.submit()?;
        Ok(output)
    }

    /// Routing weights, indices and per-expert counts. The gate is
    /// deterministic in `x`, so the adjoint chain can regenerate the routing a
    /// forward step used without a cached side buffer.
    pub(super) fn route_slots(
        &self,
        x: &GpuBuffer,
        rows: usize,
    ) -> Result<(GpuBuffer, GpuBuffer, GpuBuffer)> {
        let c = &self.config;
        if rows == 0
            || rows > 65_535
            || x.f32_capacity() < rows * c.hidden_size
            || !x.belongs_to_device(&self.device)
        {
            bail!("V4.1 MoE routing input geometry or device mismatch");
        }
        let slots = rows
            .checked_mul(c.activated)
            .context("V4.1 MoE routing slot overflow")?;
        if slots > 65_535 {
            bail!("V4.1 MoE routing exceeds dispatch capacity");
        }
        let weights = GpuBuffer::zeros_f32(&self.device, slots)?;
        let indices = GpuBuffer::zeros_u32(&self.device, slots)?;
        let counts = GpuBuffer::zeros_u32(&self.device, c.experts)?;
        let push = MoeGatePush {
            rows: rows as u32,
            dim: c.hidden_size as u32,
            experts: c.experts as u32,
            topk: c.activated as u32,
            score_func: c.score_func.code(),
            gate_temp: c.gate_temp,
            norm_topk: c.norm_topk_prob as u32,
            route_scale: c.route_scale,
        };
        let mut commands = self.device.compute_batch()?;
        self.gate.record_dispatch(
            &mut commands,
            &[
                x,
                &self.gate_weight,
                &self.gate_bias,
                &weights,
                &indices,
                &counts,
                &self.cpu_sqrt_seeds,
            ],
            bytemuck::bytes_of(&push),
            [rows as u32, 1, 1],
        )?;
        commands.submit()?;
        Ok((weights, indices, counts))
    }

    /// Routing weights/indices for a step, exposed for qualification.
    #[cfg(test)]
    pub(crate) fn route(&self, x: &GpuBuffer, rows: usize) -> Result<(GpuBuffer, GpuBuffer)> {
        let (weights, indices, _) = self.route_slots(x, rows)?;
        Ok((weights, indices))
    }
}

/// `d loss` through one V4.1 Hyper-Connection collapse.
pub struct DeepseekV41HyperConnectionPreGradients {
    /// `[rows, hc, hidden]`: the residual stream's gradient.
    pub grad_stream: GpuBuffer,
    /// `[rows, hc]`: the collapse coefficient gradient.
    pub grad_pre: GpuBuffer,
}

/// `d loss` through one V4.1 Hyper-Connection expand. The coefficients are
/// inputs of the forward, so their gradients are what the next stage (`hc_split`
/// Sinkhorn and the mix projection) will consume.
pub struct DeepseekV41HyperConnectionPostGradients {
    /// `[rows, hidden]`: the sublayer output's input gradient.
    pub grad_x: GpuBuffer,
    /// `[rows, hc, hidden]`: the residual stream's gradient.
    pub grad_residual: GpuBuffer,
    /// `[rows, hc]`: the post coefficient gradient.
    pub grad_post: GpuBuffer,
    /// `[rows, hc, hc]`: the comb coefficient gradient.
    pub grad_comb: GpuBuffer,
}

/// Combine adjoint: one gradient per routed slot plus the shared expert's.
pub struct DeepseekV41MoeCombineGradients {
    /// `[rows, topk, hidden]`, slot-major like the forward's routed buffer.
    pub grad_routed: GpuBuffer,
    /// `[rows, hidden]`.
    pub grad_shared: GpuBuffer,
}

/// SwiGLU expert adjoint for one expert's routed slots.
pub struct DeepseekV41MoeExpertGradients {
    /// `[rows, topk, hidden]`; groups routed to another expert stay zero.
    pub grad_input: GpuBuffer,
    /// `[rows, topk]`; routing-weight gradients for this expert's groups.
    pub grad_weight: GpuBuffer,
}

/// Kernels for the V4.1 block's Hyper-Connection and MoE backward stages.
///
/// ```text
/// grad_out --combine--> grad_routed, grad_shared
/// grad_routed --expert (per routed expert)--> grad_slots, grad_route_weights
/// grad_shared --shared expert--> grad_tokens
/// grad_route_weights --gate--> grad_gate
/// grad_slots + grad_tokens + grad_gate --> grad_x
/// ```
///
/// The released package's packed/mixed expert path (`DeepseekV41MixedMoe`) has
/// its own quantization, packing and compaction kernels; these adjoints cover
/// the FP32 reference graph only, and the mixed path stays fail-closed for
/// training.
pub struct DeepseekV41BlockAdjoints {
    device: VulkanDevice,
    hc_pre_input: vulkan::ComputeKernel,
    hc_split: vulkan::ComputeKernel,
    hc_mixes_input: vulkan::ComputeKernel,
    hc_post_input: vulkan::ComputeKernel,
    hc_post_comb: vulkan::ComputeKernel,
    combine: vulkan::ComputeKernel,
    expert: vulkan::ComputeKernel,
    gate: vulkan::ComputeKernel,
    input: vulkan::ComputeKernel,
    /// The FFN norm is the same RMSNorm kernel the terminal chain ends with;
    /// its input gradient is reused rather than reimplemented.
    norm: DeepseekV41TerminalAdjoints,
}

impl DeepseekV41BlockAdjoints {
    pub fn new(device: VulkanDevice) -> Result<Self> {
        Ok(Self {
            hc_pre_input: vulkan::ComputeKernel::new(
                &device,
                HC_PRE_INPUT_GRAD,
                5,
                std::mem::size_of::<HcPrePush>() as u32,
            )?,
            hc_split: vulkan::ComputeKernel::new(
                &device,
                HC_SPLIT_GRAD,
                8,
                std::mem::size_of::<HcSplitPush>() as u32,
            )?,
            hc_mixes_input: vulkan::ComputeKernel::new(
                &device,
                HC_MIXES_INPUT_GRAD,
                5,
                std::mem::size_of::<HcMixesGradPush>() as u32,
            )?,
            hc_post_input: vulkan::ComputeKernel::new(
                &device,
                HC_POST_INPUT_GRAD,
                6,
                std::mem::size_of::<HcPrePush>() as u32,
            )?,
            hc_post_comb: vulkan::ComputeKernel::new(
                &device,
                HC_POST_COMB_GRAD,
                5,
                std::mem::size_of::<HcPrePush>() as u32,
            )?,
            combine: vulkan::ComputeKernel::new(
                &device,
                MOE_COMBINE_GRAD,
                3,
                std::mem::size_of::<MoeCombinePush>() as u32,
            )?,
            expert: vulkan::ComputeKernel::new(
                &device,
                MOE_EXPERT_GRAD,
                11,
                std::mem::size_of::<MoeExpertPush>() as u32,
            )?,
            gate: vulkan::ComputeKernel::new(
                &device,
                MOE_GATE_GRAD,
                6,
                std::mem::size_of::<MoeGatePush>() as u32,
            )?,
            input: vulkan::ComputeKernel::new(
                &device,
                MOE_INPUT_GRAD_COMBINE,
                5,
                std::mem::size_of::<MoeCombinePush>() as u32,
            )?,
            norm: DeepseekV41TerminalAdjoints::new(device.clone())?,
            device,
        })
    }

    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }

    /// `d loss / d stream` and `d loss / d pre` for one Hyper-Connection
    /// collapse.
    pub fn hc_pre(
        &self,
        grad_output: &GpuBuffer,
        stream: &GpuBuffer,
        pre: &GpuBuffer,
        rows: usize,
        hc: usize,
        dim: usize,
    ) -> Result<DeepseekV41HyperConnectionPreGradients> {
        if rows == 0 || !(1..=MAX_HC).contains(&hc) || dim == 0 || rows > 65_535 {
            bail!("V4.1 hyper-connection collapse adjoint geometry is invalid");
        }
        let width = checked_size("hyper-connection collapse adjoint", &[rows, hc, dim])?;
        let collapsed = checked_size("hyper-connection collapse adjoint output", &[rows, dim])?;
        for (label, buffer, expected) in [
            ("grad_output", grad_output, collapsed),
            ("stream", stream, width),
            ("pre", pre, rows * hc),
        ] {
            if !buffer.belongs_to_device(&self.device) || buffer.f32_capacity() < expected {
                bail!("V4.1 hyper-connection collapse adjoint {label} geometry or device mismatch");
            }
        }
        let grad_stream = GpuBuffer::zeros_f32(&self.device, width)?;
        let grad_pre = GpuBuffer::zeros_f32(&self.device, rows * hc)?;
        let push = HcPrePush {
            rows: rows as u32,
            hc: hc as u32,
            dim: dim as u32,
        };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.hc_pre_input.record_dispatch(
            &mut commands,
            &[grad_output, stream, pre, &grad_stream, &grad_pre],
            bytemuck::bytes_of(&push),
            [dim.div_ceil(64) as u32, rows as u32, 1],
        )?;
        commands.submit()?;
        Ok(DeepseekV41HyperConnectionPreGradients {
            grad_stream,
            grad_pre,
        })
    }

    /// Coefficient-split adjoint: Sinkhorn and the sigmoid/softmax split back
    /// to the scaled mix value (`raw * rstd`), for one stage.
    ///
    /// `raw` is `DeepseekV41HyperConnectionMixes::raw_projection()`, the buffer
    /// the split forward consumed; `stream` is only needed to recompute the
    /// per-token `rstd` the forward applied. The `rstd` factor itself is
    /// propagated by [`Self::hc_mixes`], because dividing the returned gradient
    /// by `rstd` would not invert the forward multiply exactly.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_split(
        &self,
        hyper: &DeepseekV41HyperConnections,
        stage: DeepseekV41HyperStage,
        raw: &GpuBuffer,
        stream: &GpuBuffer,
        grad_pre: &GpuBuffer,
        grad_post: &GpuBuffer,
        grad_comb: &GpuBuffer,
        rows: usize,
    ) -> Result<GpuBuffer> {
        let hc = hyper.hc;
        let dim = hyper.hidden;
        let mix_hc = hyper.mix_hc;
        if rows == 0 || rows > 65_535 || mix_hc != (2 + hc) * hc {
            bail!("V4.1 hyper-connection split adjoint geometry is invalid");
        }
        let stream_width = checked_size("hyper-connection split adjoint stream", &[rows, hc, dim])?;
        for (label, buffer, expected) in [
            ("raw", raw, rows * mix_hc),
            ("stream", stream, stream_width),
            ("grad_pre", grad_pre, rows * hc),
            ("grad_post", grad_post, rows * hc),
            ("grad_comb", grad_comb, rows * hc * hc),
        ] {
            if !buffer.belongs_to_device(&self.device) || buffer.f32_capacity() < expected {
                bail!("V4.1 hyper-connection split adjoint {label} geometry or device mismatch");
            }
        }
        let (_, base, scale) = hyper.stage_weights(stage);
        let grad_mix = GpuBuffer::zeros_f32(&self.device, rows * mix_hc)?;
        let push = HcSplitPush {
            rows: rows as u32,
            hc: hc as u32,
            dim: dim as u32,
            mix_hc: mix_hc as u32,
            sinkhorn_iters: hyper.iterations as u32,
            eps: hyper.eps,
            norm_eps: hyper.norm_eps,
        };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.hc_split.record_dispatch(
            &mut commands,
            &[
                raw, stream, scale, base, grad_pre, grad_post, grad_comb, &grad_mix,
            ],
            bytemuck::bytes_of(&push),
            [rows as u32, 1, 1],
        )?;
        commands.submit()?;
        Ok(grad_mix)
    }

    /// Mix-projection input adjoint: the frozen `hc_fn` projection plus the
    /// `rstd`/mean-square path back to the flattened stream. `grad_mix` is the
    /// gradient with respect to the scaled mix value, as returned by
    /// [`Self::hc_split`].
    pub fn hc_mixes(
        &self,
        hyper: &DeepseekV41HyperConnections,
        stage: DeepseekV41HyperStage,
        grad_mix: &GpuBuffer,
        raw: &GpuBuffer,
        stream: &GpuBuffer,
        rows: usize,
    ) -> Result<GpuBuffer> {
        let hc = hyper.hc;
        let dim = hyper.hidden;
        let mix_hc = hyper.mix_hc;
        if rows == 0 || rows > 65_535 || mix_hc != (2 + hc) * hc {
            bail!("V4.1 hyper-connection mixes adjoint geometry is invalid");
        }
        let stream_width = checked_size("hyper-connection mixes adjoint stream", &[rows, hc, dim])?;
        for (label, buffer, expected) in [
            ("grad_mix", grad_mix, rows * mix_hc),
            ("raw", raw, rows * mix_hc),
            ("stream", stream, stream_width),
        ] {
            if !buffer.belongs_to_device(&self.device) || buffer.f32_capacity() < expected {
                bail!("V4.1 hyper-connection mixes adjoint {label} geometry or device mismatch");
            }
        }
        let (weight, _, _) = hyper.stage_weights(stage);
        if !weight.belongs_to_device(&self.device) || weight.f32_capacity() < mix_hc * hc * dim {
            bail!("V4.1 hyper-connection mixes adjoint weight geometry or device mismatch");
        }
        let grad_stream = GpuBuffer::zeros_f32(&self.device, stream_width)?;
        let push = HcMixesGradPush {
            rows: rows as u32,
            hc: hc as u32,
            dim: dim as u32,
            mix_hc: mix_hc as u32,
            norm_eps: hyper.norm_eps,
        };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.hc_mixes_input.record_dispatch(
            &mut commands,
            &[grad_mix, raw, stream, weight, &grad_stream],
            bytemuck::bytes_of(&push),
            [rows as u32, 1, 1],
        )?;
        commands.submit()?;
        Ok(grad_stream)
    }

    /// FFN RMSNorm input gradient. The block's attention and FFN norms use the
    /// same `rms_norm` kernel the terminal chain ends with, so the qualified
    /// terminal adjoint is reused verbatim.
    pub fn ffn_norm(
        &self,
        grad_hidden: &GpuBuffer,
        x: &GpuBuffer,
        weight: &GpuBuffer,
        rows: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<GpuBuffer> {
        self.norm
            .norm_input_grad(grad_hidden, x, weight, rows, hidden, eps)
    }

    /// `d loss / d x`, `d loss / d residual`, and the coefficient gradients of
    /// one Hyper-Connection expand.
    pub fn hc_post(
        &self,
        hyper: &DeepseekV41HyperConnections,
        grad_output: &GpuBuffer,
        x: &GpuBuffer,
        residual: &GpuBuffer,
        post: &GpuBuffer,
        comb: &GpuBuffer,
        rows: usize,
    ) -> Result<DeepseekV41HyperConnectionPostGradients> {
        let stream = checked_size("hyper-connection stream", &[rows, hyper.hc, hyper.hidden])?;
        let collapsed = checked_size("hyper-connection collapse", &[rows, hyper.hidden])?;
        if rows == 0 || rows > 65_535 {
            bail!("V4.1 hyper-connection post adjoint row count is invalid");
        }
        for (label, buffer, expected) in [
            ("grad_output", grad_output, stream),
            ("x", x, collapsed),
            ("residual", residual, stream),
            ("post", post, rows * hyper.hc),
            ("comb", comb, rows * hyper.hc * hyper.hc),
        ] {
            if !buffer.belongs_to_device(&self.device) || buffer.f32_capacity() < expected {
                bail!("V4.1 hyper-connection post adjoint {label} geometry or device mismatch");
            }
        }
        let grad_x = GpuBuffer::zeros_f32(&self.device, collapsed)?;
        let grad_residual = GpuBuffer::zeros_f32(&self.device, stream)?;
        let grad_post = GpuBuffer::zeros_f32(&self.device, rows * hyper.hc)?;
        let grad_comb = GpuBuffer::zeros_f32(&self.device, rows * hyper.hc * hyper.hc)?;
        let push = HcPrePush {
            rows: rows as u32,
            hc: hyper.hc as u32,
            dim: hyper.hidden as u32,
        };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.hc_post_input.record_dispatch(
            &mut commands,
            &[grad_output, x, post, comb, &grad_x, &grad_residual],
            bytemuck::bytes_of(&push),
            [hyper.hidden.div_ceil(64) as u32, rows as u32, 1],
        )?;
        self.hc_post_comb.record_dispatch(
            &mut commands,
            &[grad_output, x, residual, &grad_post, &grad_comb],
            bytemuck::bytes_of(&push),
            [(rows * hyper.hc * hyper.hc) as u32, 1, 1],
        )?;
        commands.submit()?;
        Ok(DeepseekV41HyperConnectionPostGradients {
            grad_x,
            grad_residual,
            grad_post,
            grad_comb,
        })
    }

    /// Split an upstream token gradient into the routed-slot and shared-expert
    /// gradients the combine forward summed.
    pub fn moe_combine(
        &self,
        moe: &DeepseekV41Moe,
        grad_output: &GpuBuffer,
        rows: usize,
    ) -> Result<DeepseekV41MoeCombineGradients> {
        let c = &moe.config;
        let slots = checked_size("MoE combine adjoint slots", &[rows, c.activated])?;
        let tokens = checked_size("MoE combine adjoint tokens", &[rows, c.hidden_size])?;
        if rows == 0 || slots > 65_535 {
            bail!("V4.1 MoE combine adjoint geometry is invalid");
        }
        if !grad_output.belongs_to_device(&self.device) || grad_output.f32_capacity() < tokens {
            bail!("V4.1 MoE combine adjoint gradient geometry or device mismatch");
        }
        let grad_routed = GpuBuffer::zeros_f32(&self.device, slots * c.hidden_size)?;
        let grad_shared = GpuBuffer::zeros_f32(&self.device, tokens)?;
        let push = MoeCombinePush {
            rows: rows as u32,
            dim: c.hidden_size as u32,
            topk: c.activated as u32,
        };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.combine.record_dispatch(
            &mut commands,
            &[grad_output, &grad_routed, &grad_shared],
            bytemuck::bytes_of(&push),
            [c.hidden_size.div_ceil(64) as u32, rows as u32, 1],
        )?;
        commands.submit()?;
        Ok(DeepseekV41MoeCombineGradients {
            grad_routed,
            grad_shared,
        })
    }

    /// SwiGLU adjoint for the routed slots of `expert`.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_expert(
        &self,
        moe: &DeepseekV41Moe,
        expert: u32,
        grad_routed: &GpuBuffer,
        x: &GpuBuffer,
        route_weights: &GpuBuffer,
        indices: &GpuBuffer,
        counts: &GpuBuffer,
        rows: usize,
    ) -> Result<DeepseekV41MoeExpertGradients> {
        let c = &moe.config;
        if expert as usize >= c.experts {
            bail!("V4.1 MoE expert adjoint expert id is out of range");
        }
        let slots = checked_size("MoE expert adjoint slots", &[rows, c.activated])?;
        if rows == 0 || slots > 65_535 {
            bail!("V4.1 MoE expert adjoint geometry is invalid");
        }
        let grad_input = GpuBuffer::zeros_f32(&self.device, slots * c.hidden_size)?;
        let grad_weight = GpuBuffer::zeros_f32(&self.device, slots)?;
        self.moe_expert_into(
            moe,
            expert,
            grad_routed,
            x,
            route_weights,
            indices,
            counts,
            rows,
            &grad_input,
            &grad_weight,
        )?;
        Ok(DeepseekV41MoeExpertGradients {
            grad_input,
            grad_weight,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn moe_expert_into(
        &self,
        moe: &DeepseekV41Moe,
        expert: u32,
        grad_routed: &GpuBuffer,
        x: &GpuBuffer,
        route_weights: &GpuBuffer,
        indices: &GpuBuffer,
        counts: &GpuBuffer,
        rows: usize,
        grad_input: &GpuBuffer,
        grad_weight: &GpuBuffer,
    ) -> Result<()> {
        let c = &moe.config;
        let slots = checked_size("MoE expert adjoint slots", &[rows, c.activated])?;
        if rows == 0 || slots > 65_535 {
            bail!("V4.1 MoE expert adjoint geometry is invalid");
        }
        let stream = checked_size("MoE expert adjoint stream", &[rows, c.hidden_size])?;
        let scratch = checked_size(
            "MoE expert adjoint scratch",
            &[slots, 3 * c.intermediate_size],
        )?;
        for (label, buffer, expected) in [
            ("grad_routed", grad_routed, slots * c.hidden_size),
            ("x", x, stream),
            ("route_weights", route_weights, slots),
            ("indices", indices, slots),
            ("counts", counts, c.experts),
            ("grad_input", grad_input, slots * c.hidden_size),
            ("grad_weight", grad_weight, slots),
        ] {
            if !buffer.belongs_to_device(&self.device) || buffer.f32_capacity() < expected {
                bail!("V4.1 MoE expert adjoint {label} geometry or device mismatch");
            }
        }
        let weights = &moe.experts[expert as usize];
        let scratch = GpuBuffer::zeros_f32(&self.device, scratch)?;
        let push = MoeExpertPush {
            rows: rows as u32,
            dim: c.hidden_size as u32,
            inter: c.intermediate_size as u32,
            topk: c.activated as u32,
            expert,
            weighted: 1,
            swiglu_limit: c.swiglu_limit,
        };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.expert.record_dispatch(
            &mut commands,
            &[
                x,
                &weights.w1,
                &weights.w3,
                &weights.w2,
                grad_routed,
                route_weights,
                indices,
                counts,
                &scratch,
                grad_input,
                grad_weight,
            ],
            bytemuck::bytes_of(&push),
            [slots as u32, 1, 1],
        )?;
        commands.submit()?;
        Ok(())
    }

    /// SwiGLU adjoint for the shared expert, which every token visits once.
    pub fn moe_shared_expert(
        &self,
        moe: &DeepseekV41Moe,
        grad_shared: &GpuBuffer,
        x: &GpuBuffer,
        rows: usize,
    ) -> Result<GpuBuffer> {
        let c = &moe.config;
        if rows == 0 || rows > 65_535 {
            bail!("V4.1 MoE shared expert adjoint row count is invalid");
        }
        let stream = checked_size("MoE shared expert adjoint stream", &[rows, c.hidden_size])?;
        if !grad_shared.belongs_to_device(&self.device) || grad_shared.f32_capacity() < stream {
            bail!("V4.1 MoE shared expert adjoint gradient geometry or device mismatch");
        }
        if !x.belongs_to_device(&self.device) || x.f32_capacity() < stream {
            bail!("V4.1 MoE shared expert adjoint input geometry or device mismatch");
        }
        let scratch_size = checked_size(
            "MoE shared expert adjoint scratch",
            &[rows, 3 * c.intermediate_size],
        )?;
        let scratch = GpuBuffer::zeros_f32(&self.device, scratch_size)?;
        let grad_input = GpuBuffer::zeros_f32(&self.device, stream)?;
        // The routing-weight, index and count bindings exist in the kernel's
        // layout but are never read for the shared expert (weighted == 0).
        let unused_weight = GpuBuffer::zeros_f32(&self.device, 1)?;
        let unused_route = GpuBuffer::zeros_f32(&self.device, 1)?;
        let unused_indices = GpuBuffer::zeros_u32(&self.device, 1)?;
        let unused_counts = GpuBuffer::zeros_u32(&self.device, 1)?;
        let push = MoeExpertPush {
            rows: rows as u32,
            dim: c.hidden_size as u32,
            inter: c.intermediate_size as u32,
            topk: 1,
            expert: u32::MAX,
            weighted: 0,
            swiglu_limit: c.swiglu_limit,
        };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.expert.record_dispatch(
            &mut commands,
            &[
                x,
                &moe.shared.w1,
                &moe.shared.w3,
                &moe.shared.w2,
                grad_shared,
                &unused_route,
                &unused_indices,
                &unused_counts,
                &scratch,
                &grad_input,
                &unused_weight,
            ],
            bytemuck::bytes_of(&push),
            [rows as u32, 1, 1],
        )?;
        commands.submit()?;
        Ok(grad_input)
    }

    /// Gate adjoint: routing-weight gradients back to the gate projection's
    /// input.
    pub fn moe_gate(
        &self,
        moe: &DeepseekV41Moe,
        grad_route_weights: &GpuBuffer,
        x: &GpuBuffer,
        indices: &GpuBuffer,
        rows: usize,
    ) -> Result<GpuBuffer> {
        let c = &moe.config;
        let slots = checked_size("MoE gate adjoint slots", &[rows, c.activated])?;
        let stream = checked_size("MoE gate adjoint stream", &[rows, c.hidden_size])?;
        if rows == 0 || rows > 65_535 {
            bail!("V4.1 MoE gate adjoint row count is invalid");
        }
        for (label, buffer, expected) in [
            ("grad_weights", grad_route_weights, slots),
            ("x", x, stream),
            ("indices", indices, slots),
        ] {
            if !buffer.belongs_to_device(&self.device) || buffer.f32_capacity() < expected {
                bail!("V4.1 MoE gate adjoint {label} geometry or device mismatch");
            }
        }
        let grad_x = GpuBuffer::zeros_f32(&self.device, stream)?;
        let push = MoeGatePush {
            rows: rows as u32,
            dim: c.hidden_size as u32,
            experts: c.experts as u32,
            topk: c.activated as u32,
            score_func: c.score_func.code(),
            gate_temp: c.gate_temp,
            norm_topk: c.norm_topk_prob as u32,
            route_scale: c.route_scale,
        };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.gate.record_dispatch(
            &mut commands,
            &[
                x,
                &moe.gate_weight,
                indices,
                grad_route_weights,
                &grad_x,
                &moe.cpu_sqrt_seeds,
            ],
            bytemuck::bytes_of(&push),
            [rows as u32, 1, 1],
        )?;
        commands.submit()?;
        Ok(grad_x)
    }

    /// Full MoE backward: routing is regenerated from `x` (the gate is
    /// deterministic), then combine, every routed expert, the shared expert and
    /// the gate are chained into one token gradient.
    pub fn moe(
        &self,
        moe: &DeepseekV41Moe,
        grad_output: &GpuBuffer,
        x: &GpuBuffer,
        rows: usize,
    ) -> Result<GpuBuffer> {
        let c = &moe.config;
        let slots = checked_size("MoE adjoint slots", &[rows, c.activated])?;
        let stream = checked_size("MoE adjoint stream", &[rows, c.hidden_size])?;
        if rows == 0 || slots > 65_535 {
            bail!("V4.1 MoE adjoint geometry is invalid");
        }
        let (route_weights, indices, counts) = moe.route_slots(x, rows)?;
        let combine = self.moe_combine(moe, grad_output, rows)?;
        let grad_slots = GpuBuffer::zeros_f32(&self.device, slots * c.hidden_size)?;
        let grad_route_weights = GpuBuffer::zeros_f32(&self.device, slots)?;
        for expert in 0..c.experts {
            self.moe_expert_into(
                moe,
                expert as u32,
                &combine.grad_routed,
                x,
                &route_weights,
                &indices,
                &counts,
                rows,
                &grad_slots,
                &grad_route_weights,
            )?;
        }
        let grad_shared = self.moe_shared_expert(moe, &combine.grad_shared, x, rows)?;
        let grad_gate = self.moe_gate(moe, &grad_route_weights, x, &indices, rows)?;
        let grad_x = GpuBuffer::zeros_f32(&self.device, stream)?;
        let push = MoeCombinePush {
            rows: rows as u32,
            dim: c.hidden_size as u32,
            topk: c.activated as u32,
        };
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        self.input.record_dispatch(
            &mut commands,
            &[&grad_slots, &grad_shared, &grad_gate, &indices, &grad_x],
            bytemuck::bytes_of(&push),
            [c.hidden_size.div_ceil(64) as u32, rows as u32, 1],
        )?;
        commands.submit()?;
        Ok(grad_x)
    }
}

pub struct DeepseekV41MixedExpertWeights {
    pub w1: DeepseekV41LinearWeights,
    pub w2: DeepseekV41LinearWeights,
    pub w3: DeepseekV41LinearWeights,
}
pub struct DeepseekV41MixedMoeWeights {
    pub gate_weight: Vec<f32>,
    pub gate_bias: Vec<f32>,
    pub experts: Vec<DeepseekV41MixedExpertWeights>,
    pub shared: DeepseekV41MixedExpertWeights,
}
struct MixedExpert {
    w1: DeepseekV41ResidentLinear,
    w2: DeepseekV41ResidentLinear,
    w3: DeepseekV41ResidentLinear,
}
impl MixedExpert {
    fn upload(device: &VulkanDevice, weights: DeepseekV41MixedExpertWeights) -> Result<Self> {
        let mut commands = device.compute_batch()?;
        let expert = Self::record_upload(device, &mut commands, weights)?;
        commands.submit_and_wait()?;
        Ok(expert)
    }
    fn record_upload(
        device: &VulkanDevice,
        commands: &mut vulkan::ComputeBatch,
        weights: DeepseekV41MixedExpertWeights,
    ) -> Result<Self> {
        Ok(Self {
            w1: weights.w1.record_upload(device, commands)?,
            w2: weights.w2.record_upload(device, commands)?,
            w3: weights.w3.record_upload(device, commands)?,
        })
    }
}

/// Expert payloads may stay resident, or be read/uploaded one at a time.
/// All policies retain gates and non-expert model weights.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeepseekV41ExpertResidency {
    #[default]
    Resident,
    /// Checkpoint files must remain readable and unchanged during execution.
    /// Every routed expert is visited in reference order; routing stays on GPU.
    /// Shared experts remain resident.
    Streamed,
    /// Stage routed experts as in `Streamed`, then stage the shared expert.
    /// No expert payload stays resident between calls. Keep checkpoint files
    /// readable and unchanged for the model's lifetime.
    StreamedAllExperts,
}

/// Whether the per-layer attention, compressor and indexer projections stay
/// resident or are read from the checkpoint for each layer execution.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeepseekV41ProjectionResidency {
    #[default]
    Resident,
    /// Projections are read and uploaded when the orchestrator enters the
    /// layer, then released after it. Window/compressed/key caches, norms and
    /// other layer parameters remain resident. Keep checkpoint files readable
    /// and unchanged for the model's lifetime.
    Streamed,
}

pub(super) struct StreamedMoeWeights {
    pub gate_weight: Vec<f32>,
    pub gate_bias: Vec<f32>,
    pub shared: Option<DeepseekV41MixedExpertWeights>,
    pub checkpoint: DeepseekV41Checkpoint,
    pub layer: usize,
}

enum ExpertSource {
    Resident(Vec<MixedExpert>),
    Streamed {
        checkpoint: Mutex<DeepseekV41Checkpoint>,
        layer: usize,
    },
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MoePackPush {
    rows: u32,
    dim: u32,
    topk: u32,
    expert: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MixedSwigluPush {
    rows: u32,
    inter: u32,
    expert: u32,
    bf16: u32,
    limit: f32,
}

/// Packed routed/shared experts with explicit BF16 boundaries and resident
/// compaction. The enclosing backbone can still use its FP32 diagnostic stream.
/// This is not a qualification of the complete production precision graph.
pub struct DeepseekV41MixedMoe {
    device: VulkanDevice,
    config: DeepseekV41MoeConfig,
    precision: DeepseekV41WeightOutput,
    gate_weight: GpuBuffer,
    gate_bias: GpuBuffer,
    cpu_sqrt_seeds: GpuBuffer,
    experts: ExpertSource,
    // None only when the shared expert is staged from `experts`' checkpoint.
    shared: Option<MixedExpert>,
    linear: DeepseekV41MixedLinearKernel,
    gate: vulkan::ComputeKernel,
    pack: vulkan::ComputeKernel,
    swiglu: vulkan::ComputeKernel,
    scatter: vulkan::ComputeKernel,
    combine: vulkan::ComputeKernel,
    #[cfg(test)]
    pub(super) route_trace: std::cell::RefCell<Option<(GpuBuffer, GpuBuffer, GpuBuffer)>>,
}
impl DeepseekV41MixedMoe {
    pub fn new(
        device: VulkanDevice,
        config: DeepseekV41MoeConfig,
        weights: DeepseekV41MixedMoeWeights,
        precision: DeepseekV41WeightOutput,
    ) -> Result<Self> {
        Self::build(
            device,
            config,
            weights.gate_weight,
            weights.gate_bias,
            Some(weights.shared),
            Some(weights.experts),
            None,
            precision,
        )
    }

    pub(super) fn new_streamed(
        device: VulkanDevice,
        config: DeepseekV41MoeConfig,
        weights: StreamedMoeWeights,
        precision: DeepseekV41WeightOutput,
    ) -> Result<Self> {
        Self::build(
            device,
            config,
            weights.gate_weight,
            weights.gate_bias,
            weights.shared,
            None,
            Some((weights.checkpoint, weights.layer)),
            precision,
        )
    }

    fn build(
        device: VulkanDevice,
        config: DeepseekV41MoeConfig,
        gate_weight: Vec<f32>,
        gate_bias: Vec<f32>,
        shared: Option<DeepseekV41MixedExpertWeights>,
        experts: Option<Vec<DeepseekV41MixedExpertWeights>>,
        streamed: Option<(DeepseekV41Checkpoint, usize)>,
        precision: DeepseekV41WeightOutput,
    ) -> Result<Self> {
        let c = config;
        if c.hidden_size == 0
            || c.intermediate_size == 0
            || c.experts == 0
            || c.experts > MAX_EXPERTS
            || c.activated == 0
            || c.activated > c.experts.min(MAX_ACTIVATED)
            || !c.gate_temp.is_finite()
            || c.gate_temp <= 0.0
            || !c.route_scale.is_finite()
            || !c.swiglu_limit.is_finite()
            || c.swiglu_limit < 0.0
        {
            bail!("V4.1 mixed MoE configuration is invalid");
        }
        checked_size("mixed MoE gate", &[c.experts, c.hidden_size])?;
        checked_size("mixed MoE expert", &[c.intermediate_size, c.hidden_size])?;
        if gate_weight.len() != c.experts * c.hidden_size
            || gate_bias.len() != c.experts
            || experts
                .as_ref()
                .is_some_and(|weights| weights.len() != c.experts)
            || experts.is_some() == streamed.is_some()
            || (shared.is_none() && streamed.is_none())
        {
            bail!("V4.1 mixed MoE parameter counts are invalid");
        }
        validate_finite("mixed MoE gate", &gate_weight)?;
        validate_finite("mixed MoE bias", &gate_bias)?;
        for w in experts.iter().flatten().chain(shared.iter()) {
            for (w, shape) in [
                (&w.w1, [c.intermediate_size, c.hidden_size]),
                (&w.w2, [c.hidden_size, c.intermediate_size]),
                (&w.w3, [c.intermediate_size, c.hidden_size]),
            ] {
                w.validate()?;
                let compatible = match w {
                    DeepseekV41LinearWeights::Fp32 { .. } => {
                        precision == DeepseekV41WeightOutput::Fp32
                    }
                    DeepseekV41LinearWeights::Bf16 { .. } => {
                        precision == DeepseekV41WeightOutput::Bf16
                    }
                    DeepseekV41LinearWeights::Packed { .. } => true,
                };
                if w.shape() != shape || !compatible {
                    bail!("V4.1 mixed MoE projection geometry or dense stream precision mismatch");
                }
            }
        }
        let experts = match (experts, streamed) {
            (Some(weights), None) => ExpertSource::Resident(
                weights
                    .into_iter()
                    .map(|w| MixedExpert::upload(&device, w))
                    .collect::<Result<_>>()?,
            ),
            (None, Some((checkpoint, layer))) => ExpertSource::Streamed {
                checkpoint: Mutex::new(checkpoint),
                layer,
            },
            _ => unreachable!("exactly one expert source validated"),
        };
        let shared = shared
            .map(|w| MixedExpert::upload(&device, w))
            .transpose()?;
        Ok(Self {
            gate_weight: GpuBuffer::from_f32(&device, &gate_weight)?,
            gate_bias: GpuBuffer::from_f32(&device, &gate_bias)?,
            cpu_sqrt_seeds: super::cpu_sqrt_seeds(&device)?,
            experts,
            shared,
            precision,
            config: c,
            linear: DeepseekV41MixedLinearKernel::new(device.clone())?,
            gate: vulkan::ComputeKernel::new(
                &device,
                MOE_GATE,
                7,
                std::mem::size_of::<MoeGatePush>() as u32,
            )?,
            pack: vulkan::ComputeKernel::new(
                &device,
                MOE_PACK,
                6,
                std::mem::size_of::<MoePackPush>() as u32,
            )?,
            swiglu: vulkan::ComputeKernel::new(
                &device,
                MIXED_SWIGLU,
                5,
                std::mem::size_of::<MixedSwigluPush>() as u32,
            )?,
            scatter: vulkan::ComputeKernel::new(
                &device,
                MOE_SCATTER,
                4,
                std::mem::size_of::<MoePackPush>() as u32,
            )?,
            combine: vulkan::ComputeKernel::new(
                &device,
                MOE_COMBINE,
                4,
                std::mem::size_of::<MoeCombinePush>() as u32,
            )?,
            device,
            #[cfg(test)]
            route_trace: std::cell::RefCell::new(None),
        })
    }

    /// Persistent routed experts only; streamed staging and shared experts are excluded.
    pub fn resident_routed_expert_count(&self) -> usize {
        match &self.experts {
            ExpertSource::Resident(experts) => experts.len(),
            ExpertSource::Streamed { .. } => 0,
        }
    }

    /// Persistent shared experts only; temporary checkpoint staging is excluded.
    pub fn resident_shared_expert_count(&self) -> usize {
        usize::from(self.shared.is_some())
    }
    fn record_expert(
        &self,
        commands: &mut vulkan::ComputeBatch,
        x: &GpuBuffer,
        rows: usize,
        expert: u32,
        weights: &GpuBuffer,
        counts: &GpuBuffer,
        w: &MixedExpert,
    ) -> Result<GpuBuffer> {
        let (gate, up) =
            self.linear
                .record_apply_pair(commands, x, rows, &w.w1, &w.w3, self.precision)?;
        let activation = GpuBuffer::zeros_u32(
            &self.device,
            checked_size(
                "mixed MoE activation",
                &[rows, self.config.intermediate_size],
            )?,
        )?;
        let push = MixedSwigluPush {
            rows: rows as u32,
            inter: self.config.intermediate_size as u32,
            expert,
            bf16: u32::from(self.precision == DeepseekV41WeightOutput::Bf16),
            limit: self.config.swiglu_limit,
        };
        self.swiglu.record_dispatch(
            commands,
            &[&gate, &up, weights, counts, &activation],
            bytemuck::bytes_of(&push),
            [
                self.config.intermediate_size.div_ceil(64) as u32,
                rows as u32,
                1,
            ],
        )?;
        self.linear
            .record_apply(commands, &activation, rows, &w.w2, self.precision)
    }
    pub fn apply(&self, input: &GpuBuffer, rows: usize) -> Result<GpuBuffer> {
        let c = &self.config;
        let slots = checked_size("mixed MoE slots", &[rows, c.activated])?;
        if rows == 0
            || slots > 65_535
            || c.hidden_size.div_ceil(64) > 65_535
            || c.intermediate_size.div_ceil(64) > 65_535
        {
            bail!("V4.1 mixed MoE exceeds dispatch capacity");
        }
        let x = self.linear.cast_buffer(
            input,
            checked_size("mixed MoE input", &[rows, c.hidden_size])?,
            self.precision,
        )?;
        let route = GpuBuffer::zeros_f32(&self.device, slots)?;
        let indices = GpuBuffer::zeros_u32(&self.device, slots)?;
        let counts = GpuBuffer::zeros_u32(&self.device, c.experts)?;
        let push = MoeGatePush {
            rows: rows as u32,
            dim: c.hidden_size as u32,
            experts: c.experts as u32,
            topk: c.activated as u32,
            score_func: c.score_func.code(),
            gate_temp: c.gate_temp,
            norm_topk: c.norm_topk_prob as u32,
            route_scale: c.route_scale,
        };
        let mut commands = self.device.compute_batch()?;
        self.gate.record_dispatch(
            &mut commands,
            &[
                &x,
                &self.gate_weight,
                &self.gate_bias,
                &route,
                &indices,
                &counts,
                &self.cpu_sqrt_seeds,
            ],
            bytemuck::bytes_of(&push),
            [rows as u32, 1, 1],
        )?;
        commands.submit()?;
        #[cfg(test)]
        self.route_trace
            .replace(Some((route.clone(), indices.clone(), counts.clone())));
        let routed = GpuBuffer::zeros_f32(
            &self.device,
            checked_size("mixed MoE routed", &[slots, c.hidden_size])?,
        )?;
        // Each expert overwrites its live compact prefix and selected row map.
        // Quantization/projections are row-local, SwiGLU zeros rows beyond the
        // GPU count, and scatter reads only selected rows. Reuse this scratch
        // across experts without clearing or reading routing data on the CPU.
        let compact = GpuBuffer::zeros_f32(&self.device, rows * c.hidden_size)?;
        let weights = GpuBuffer::zeros_f32(&self.device, rows)?;
        let mapping = GpuBuffer::zeros_u32(&self.device, rows)?;
        let visited = counts.read_u32(c.experts)?;
        let moe_trace = std::env::var_os("HIERARCHOS_VULKAN_MOE_TRACE").is_some();
        if moe_trace {
            eprintln!(
                "v41_moe_trace input={:016x} route={:016x} indices={:016x} counts={:016x} active={}",
                digest_of(input, rows * c.hidden_size)?,
                digest_of(&route, slots)?,
                digest_of(&indices, slots)?,
                digest_of(&counts, c.experts)?,
                visited.iter().filter(|&&n| n != 0).count(),
            );
        }
        // One submission per layer for resident experts. The dispatches, their
        // order and every buffer binding are unchanged, so the arithmetic is
        // identical; only the device waits collapse from six per expert to one
        // per layer. Streamed experts keep one submission per expert so only a
        // single staged payload is alive at a time.
        let mut layer_commands = if matches!(self.experts, ExpertSource::Resident(_)) {
            Some(self.device.compute_batch()?)
        } else {
            None
        };
        for id in 0..c.experts {
            if visited[id] == 0 {
                continue;
            }
            let push = MoePackPush {
                rows: rows as u32,
                dim: c.hidden_size as u32,
                topk: c.activated as u32,
                expert: id as u32,
            };
            let dispatch = [c.hidden_size.div_ceil(64) as u32, rows as u32, 1];
            let mut own_commands = None;
            let commands = match layer_commands.as_mut() {
                Some(commands) => commands,
                None => {
                    own_commands = Some(self.device.compute_batch()?);
                    own_commands
                        .as_mut()
                        .expect("streamed expert batch was just created")
                }
            };
            self.pack.record_dispatch(
                commands,
                &[&x, &indices, &route, &compact, &weights, &mapping],
                bytemuck::bytes_of(&push),
                dispatch,
            )?;
            // Finish this expert before dropping its staging buffers or loading
            // the next. No payload vector or GPU expert is cached by the source.
            let staged;
            let w = match &self.experts {
                ExpertSource::Resident(experts) => &experts[id],
                ExpertSource::Streamed { checkpoint, layer } => {
                    let payload = checkpoint
                        .lock()
                        .map_err(|_| anyhow::anyhow!("V4.1 streamed checkpoint lock poisoned"))?
                        .load_mixed_expert(*layer, id)?;
                    staged = MixedExpert::record_upload(&self.device, commands, payload)?;
                    &staged
                }
            };
            // CPU Expert.forward receives only torch.where's live rows. Its
            // GEMM reduction changes below four rows, so padded capacity must
            // never stand in for this logical matrix height.
            let result = self.record_expert(
                commands,
                &compact,
                visited[id] as usize,
                id as u32,
                &weights,
                &counts,
                w,
            )?;
            self.scatter.record_dispatch(
                commands,
                &[&result, &indices, &mapping, &routed],
                bytemuck::bytes_of(&push),
                dispatch,
            )?;
            if let Some(mut batch) = own_commands {
                batch.submit_and_wait()?;
            }
        }
        let shared = if let Some(w) = &self.shared {
            match layer_commands.as_mut() {
                Some(commands) => {
                    self.record_expert(commands, &x, rows, u32::MAX, &route, &counts, w)?
                }
                None => {
                    let mut commands = self.device.compute_batch()?;
                    let shared =
                        self.record_expert(&mut commands, &x, rows, u32::MAX, &route, &counts, w)?;
                    commands.submit()?;
                    shared
                }
            }
        } else {
            let ExpertSource::Streamed { checkpoint, layer } = &self.experts else {
                unreachable!("shared staging requires a checkpoint source");
            };
            let payload = checkpoint
                .lock()
                .map_err(|_| anyhow::anyhow!("V4.1 streamed checkpoint lock poisoned"))?
                .load_mixed_shared_expert(*layer)?;
            let mut commands = self.device.compute_batch()?;
            let staged = MixedExpert::record_upload(&self.device, &mut commands, payload)?;
            let shared =
                self.record_expert(&mut commands, &x, rows, u32::MAX, &route, &counts, &staged)?;
            commands.submit_and_wait()?;
            shared
        };
        if let Some(mut commands) = layer_commands {
            commands.submit()?;
        }
        if moe_trace {
            eprintln!(
                "v41_moe_trace routed={:016x} shared={:016x}",
                digest_of(&routed, slots * c.hidden_size)?,
                digest_of(&shared, rows * c.hidden_size)?,
            );
        }
        let output = GpuBuffer::zeros_f32(&self.device, rows * c.hidden_size)?;
        let push = MoeCombinePush {
            rows: rows as u32,
            dim: c.hidden_size as u32,
            topk: c.activated as u32,
        };
        let mut commands = self.device.compute_batch()?;
        self.combine.record_dispatch(
            &mut commands,
            &[&routed, &shared, &output, &indices],
            bytemuck::bytes_of(&push),
            [c.hidden_size.div_ceil(64) as u32, rows as u32, 1],
        )?;
        commands.submit()?;
        let cast = self
            .linear
            .cast_buffer(&output, rows * c.hidden_size, self.precision)?;
        if moe_trace {
            let cast_values = cast.read_f32(rows * c.hidden_size)?;
            let mut hash = 0xcbf29ce484222325_u64;
            for value in &cast_values {
                for byte in value.to_bits().to_le_bytes() {
                    hash ^= u64::from(byte);
                    hash = hash.wrapping_mul(0x100000001b3);
                }
            }
            let max = cast_values.iter().copied().fold(f32::MIN, f32::max);
            let min = cast_values.iter().copied().fold(f32::MAX, f32::min);
            eprintln!(
                "v41_moe_trace combined={:016x} cast={hash:016x} head={:?} max={max:e} min={min:e}",
                digest_of(&output, rows * c.hidden_size)?,
                &cast_values[..4.min(cast_values.len())],
            );
        }
        Ok(cast)
    }
}

pub struct DeepseekV41BackboneLayerWeights {
    pub attention: DeepseekV41AttentionLayerWeights,
    pub hyper: DeepseekV41HyperConnectionWeights,
    /// RMSNorm applied to the collapsed attention input.
    pub attn_norm: Vec<f32>,
    /// RMSNorm applied to the collapsed FFN input.
    pub ffn_norm: Vec<f32>,
    pub moe: DeepseekV41MoeWeights,
}

pub(super) enum MoeLayerWeights {
    Fp32(DeepseekV41MoeWeights),
    Mixed(DeepseekV41MixedMoeWeights),
    Streamed(StreamedMoeWeights),
}
pub(super) struct BackboneLayerWeights {
    /// `None` when the attention/indexer/compressor projections are streamed
    /// per layer instead of loaded with the other layer parameters.
    pub attention: Option<super::orchestrator::AttentionLayerWeights>,
    pub hyper: DeepseekV41HyperConnectionWeights,
    pub moe: MoeLayerWeights,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
}
impl From<DeepseekV41BackboneLayerWeights> for BackboneLayerWeights {
    fn from(w: DeepseekV41BackboneLayerWeights) -> Self {
        Self {
            attention: Some(w.attention.into()),
            hyper: w.hyper,
            moe: MoeLayerWeights::Fp32(w.moe),
            attn_norm: w.attn_norm,
            ffn_norm: w.ffn_norm,
        }
    }
}
enum MoeExecution {
    Fp32(DeepseekV41Moe),
    Mixed(DeepseekV41MixedMoe),
}
impl MoeExecution {
    fn apply(&self, input: &GpuBuffer, rows: usize) -> Result<GpuBuffer> {
        match self {
            Self::Fp32(m) => m.apply(input, rows),
            Self::Mixed(m) => m.apply(input, rows),
        }
    }
}

/// Runs the released block order around one attention layer at a time:
/// hyper-connection mixes, collapse, attention norm, attention, expand, FFN
/// mixes, collapse through the attention-stage pre mix, FFN norm, MoE, expand.
/// The FFN stage's pre mix is the next layer's attention collapse, exactly as
/// the released `Block.forward` returns it.
pub struct DeepseekV41Backbone {
    device: VulkanDevice,
    runtime: DeepseekV41AttentionRuntimeConfig,
    hidden: usize,
    hc: usize,
    norm_eps: f32,
    plan: Vec<DeepseekV41LayerPlan>,
    attention: DeepseekV41AttentionOrchestrator,
    hyper: Vec<DeepseekV41HyperConnections>,
    attn_norm: Vec<GpuBuffer>,
    ffn_norm: Vec<GpuBuffer>,
    moe: Vec<MoeExecution>,
    norm: NormKernel,
    cast: DeepseekV41MixedLinearKernel,
    precision: DeepseekV41WeightOutput,
    engram: Option<DeepseekV41EngramRuntime>,
    #[cfg(test)]
    pub(super) trace: Option<Vec<(usize, &'static str, GpuBuffer, usize)>>,
}

impl DeepseekV41Backbone {
    pub fn new(
        device: VulkanDevice,
        config: &DeepseekV41Config,
        runtime: DeepseekV41AttentionRuntimeConfig,
        weights: Vec<DeepseekV41BackboneLayerWeights>,
        engram: Option<DeepseekV41EngramRuntime>,
    ) -> Result<Self> {
        let text = &config.text;
        if weights.len() != text.num_hidden_layers || text.num_hidden_layers == 0 {
            bail!("V4.1 backbone layer-weight count is invalid");
        }
        Self::build(
            device,
            config,
            runtime,
            weights.into_iter().map(|w| Ok(w.into())),
            engram,
            DeepseekV41WeightOutput::Fp32,
            None,
        )
    }

    pub fn from_checkpoint(
        device: VulkanDevice,
        checkpoint: &mut DeepseekV41Checkpoint,
        runtime: DeepseekV41AttentionRuntimeConfig,
        engram: Option<DeepseekV41EngramRuntime>,
    ) -> Result<Self> {
        Self::from_checkpoint_with_precision(
            device,
            checkpoint,
            runtime,
            engram,
            DeepseekV41WeightOutput::Fp32,
        )
    }

    pub fn from_checkpoint_with_precision(
        device: VulkanDevice,
        checkpoint: &mut DeepseekV41Checkpoint,
        runtime: DeepseekV41AttentionRuntimeConfig,
        engram: Option<DeepseekV41EngramRuntime>,
        precision: DeepseekV41WeightOutput,
    ) -> Result<Self> {
        Self::from_checkpoint_with_expert_residency(
            device,
            checkpoint,
            runtime,
            engram,
            precision,
            DeepseekV41ExpertResidency::Resident,
        )
    }

    pub fn from_checkpoint_with_expert_residency(
        device: VulkanDevice,
        checkpoint: &mut DeepseekV41Checkpoint,
        runtime: DeepseekV41AttentionRuntimeConfig,
        engram: Option<DeepseekV41EngramRuntime>,
        precision: DeepseekV41WeightOutput,
        residency: DeepseekV41ExpertResidency,
    ) -> Result<Self> {
        Self::from_checkpoint_with_residency(
            device,
            checkpoint,
            runtime,
            engram,
            precision,
            residency,
            DeepseekV41ProjectionResidency::Resident,
        )
    }

    /// Layer weights other than the attention/indexer/compressor projections
    /// load as usual. `Streamed` leaves those projections in the checkpoint and
    /// reads them when the attention orchestrator enters each layer.
    #[allow(clippy::too_many_arguments)]
    pub fn from_checkpoint_with_residency(
        device: VulkanDevice,
        checkpoint: &mut DeepseekV41Checkpoint,
        runtime: DeepseekV41AttentionRuntimeConfig,
        engram: Option<DeepseekV41EngramRuntime>,
        precision: DeepseekV41WeightOutput,
        residency: DeepseekV41ExpertResidency,
        projections: DeepseekV41ProjectionResidency,
    ) -> Result<Self> {
        let config = checkpoint.config.clone();
        let layers = config.text.num_hidden_layers;
        let streamed = projections == DeepseekV41ProjectionResidency::Streamed;
        let mut weights = Vec::with_capacity(layers);
        for layer in 0..layers {
            weights.push(if streamed {
                checkpoint.load_backbone_components_without_attention(layer, residency)?
            } else {
                checkpoint.load_backbone_components_with_residency(layer, residency)?
            });
        }
        let source = streamed.then(|| checkpoint.clone());
        Self::build(
            device,
            &config,
            runtime,
            weights.into_iter().map(Ok),
            engram,
            precision,
            source,
        )
    }

    fn build(
        device: VulkanDevice,
        config: &DeepseekV41Config,
        runtime: DeepseekV41AttentionRuntimeConfig,
        weights: impl Iterator<Item = Result<BackboneLayerWeights>>,
        engram: Option<DeepseekV41EngramRuntime>,
        precision: DeepseekV41WeightOutput,
        projection_source: Option<DeepseekV41Checkpoint>,
    ) -> Result<Self> {
        let text = &config.text;
        if engram
            .as_ref()
            .map_or(&[][..], |runtime| runtime.layer_ids())
            != text.engram_layer_ids
        {
            bail!("V4.1 backbone Engram runtime must match the configured layer schedule");
        }
        if let Some(engram) = &engram {
            if engram.stream_precision() != precision {
                bail!("V4.1 Engram/backbone stream precision mismatch");
            }
            engram.validate_text_geometry(text, runtime.batch_size, runtime.max_positions)?;
        }
        let norm_eps = text.rms_norm_eps as f32;
        if !norm_eps.is_finite() || norm_eps <= 0.0 {
            bail!("V4.1 backbone norm epsilon is invalid");
        }
        let moe_config = DeepseekV41MoeConfig::from_text(text)?;
        let mut hyper = Vec::with_capacity(text.num_hidden_layers);
        let mut attn_norm = Vec::with_capacity(text.num_hidden_layers);
        let mut ffn_norm = Vec::with_capacity(text.num_hidden_layers);
        let mut moe = Vec::with_capacity(text.num_hidden_layers);
        let attention_weights = weights.enumerate().map(|(layer, w)| {
            let w = w?;
            if w.attn_norm.len() != text.hidden_size || w.ffn_norm.len() != text.hidden_size {
                bail!("V4.1 backbone layer {layer} norm geometry is invalid");
            }
            validate_finite(&format!("backbone layer {layer} attn_norm"), &w.attn_norm)?;
            validate_finite(&format!("backbone layer {layer} ffn_norm"), &w.ffn_norm)?;
            hyper.push(DeepseekV41HyperConnections::new(
                device.clone(),
                text.hidden_size,
                text.hc_mult,
                text.hc_sinkhorn_iters,
                text.hc_eps as f32,
                norm_eps,
                w.hyper,
            )?);
            attn_norm.push(GpuBuffer::from_f32(&device, &w.attn_norm)?);
            ffn_norm.push(GpuBuffer::from_f32(&device, &w.ffn_norm)?);
            moe.push(match w.moe {
                MoeLayerWeights::Fp32(weights) => {
                    if precision == DeepseekV41WeightOutput::Bf16 {
                        bail!("V4.1 BF16 backbone requires typed packed MoE weights");
                    }
                    MoeExecution::Fp32(DeepseekV41Moe::new(device.clone(), moe_config, weights)?)
                }
                MoeLayerWeights::Mixed(weights) => MoeExecution::Mixed(DeepseekV41MixedMoe::new(
                    device.clone(),
                    moe_config,
                    weights,
                    precision,
                )?),
                MoeLayerWeights::Streamed(weights) => {
                    MoeExecution::Mixed(DeepseekV41MixedMoe::new_streamed(
                        device.clone(),
                        moe_config,
                        weights,
                        precision,
                    )?)
                }
            });
            Ok(w.attention)
        });
        let attention = if let Some(checkpoint) = projection_source {
            // Streamed layers still load hyper-connections, norms and MoE here;
            // their attention/indexer/compressor projections stay in the
            // checkpoint and are read by the orchestrator per layer.
            for weights in attention_weights {
                if weights?.is_some() {
                    bail!("V4.1 streamed backbone received resident attention projections");
                }
            }
            DeepseekV41AttentionOrchestrator::build_streamed(
                device.clone(),
                config,
                runtime,
                checkpoint,
                precision,
            )?
        } else {
            let weights = attention_weights.map(|weights| {
                weights.and_then(|weights| {
                    weights.context("V4.1 attention layer weights are missing in resident mode")
                })
            });
            DeepseekV41AttentionOrchestrator::build(
                device.clone(),
                config,
                runtime,
                weights,
                precision,
            )?
        };
        let plan = config.plan()?.backbone;
        let norm = NormKernel::new(&device)?;
        let cast = DeepseekV41MixedLinearKernel::new(device.clone())?;
        Ok(Self {
            device,
            runtime,
            hidden: text.hidden_size,
            hc: text.hc_mult,
            norm_eps,
            plan,
            attention,
            hyper,
            attn_norm,
            ffn_norm,
            moe,
            norm,
            cast,
            precision,
            engram,
            #[cfg(test)]
            trace: None,
        })
    }

    pub fn layer_plan(&self) -> &[DeepseekV41LayerPlan] {
        &self.plan
    }
    #[cfg(test)]
    pub(super) fn index_selection(
        &self,
        layer: usize,
    ) -> Result<Option<&super::DeepseekV41IndexSelection>> {
        self.attention.selection(layer)
    }
    #[cfg(test)]
    pub(super) fn raw_attention_trace(&self, layer: usize) -> Option<(GpuBuffer, GpuBuffer)> {
        self.attention.raw_attention_trace(layer)
    }
    /// Number of routed experts kept resident between model calls.
    pub fn resident_routed_expert_count(&self) -> usize {
        self.moe
            .iter()
            .map(|moe| match moe {
                MoeExecution::Fp32(moe) => moe.config.experts,
                MoeExecution::Mixed(moe) => moe.resident_routed_expert_count(),
            })
            .sum()
    }

    pub fn resident_shared_expert_count(&self) -> usize {
        self.moe
            .iter()
            .map(|moe| match moe {
                MoeExecution::Fp32(_) => 1,
                MoeExecution::Mixed(moe) => moe.resident_shared_expert_count(),
            })
            .sum()
    }

    /// Layers whose attention/indexer/compressor projections stay resident.
    pub fn resident_projection_layer_count(&self) -> usize {
        self.attention.resident_projection_layer_count()
    }

    /// Scheduled Engram tables kept resident between calls.
    pub fn resident_engram_table_count(&self) -> usize {
        self.engram
            .as_ref()
            .map_or(0, |engram| engram.resident_engram_table_count())
    }

    #[cfg(test)]
    pub(super) fn moe_route_trace(
        &self,
        layer: usize,
    ) -> Option<(GpuBuffer, GpuBuffer, GpuBuffer)> {
        match self.moe.get(layer)? {
            MoeExecution::Mixed(m) => m.route_trace.borrow().clone(),
            MoeExecution::Fp32(_) => None,
        }
    }
    #[cfg(test)]
    pub(super) fn attention_cache_trace(
        &self,
        layer: usize,
    ) -> Result<Option<(Vec<u32>, Vec<u32>)>> {
        let ratio = self.plan[layer].compress_ratio;
        if ratio == 0 {
            return Ok(None);
        }
        let groups = self
            .attention
            .next_position()
            .context("completed step required for cache trace")?
            / ratio;
        let cache = self.attention.compressed_cache(layer)?.unwrap();
        let keys = self.attention.index_keys(layer)?.unwrap();
        let mut kv_bits = vec![];
        let mut key_bits = vec![];
        for batch in 0..self.runtime.batch_size {
            kv_bits.extend(cache.read_batch(batch, groups)?.iter().map(|v| v.to_bits()));
            key_bits.extend(keys.read_batch(batch, groups)?.iter().map(|v| v.to_bits()));
        }
        Ok(Some((kv_bits, key_bits)))
    }
    pub fn next_layer(&self) -> Option<usize> {
        self.attention.next_layer()
    }
    pub fn next_position(&self) -> Option<usize> {
        self.attention.next_position()
    }
    pub fn begin_prefill(&mut self, seq: usize) -> Result<()> {
        self.attention.begin_prefill(seq)
    }
    pub fn begin_decode(&mut self, start_pos: usize) -> Result<()> {
        self.attention.begin_decode(start_pos)
    }

    fn rows(&self) -> Result<usize> {
        let (seq, _, _) = self
            .attention
            .step_shape()
            .context("V4.1 backbone requires begin_prefill or begin_decode")?;
        self.runtime
            .batch_size
            .checked_mul(seq)
            .context("V4.1 backbone row overflow")
    }

    /// One block, with the previous layer's FFN pre mix as `pre_mix`.
    ///
    /// The block's stages record into one layer session, so the layer retires as
    /// a single queue submission instead of one per kernel family. Stages whose
    /// weights are streamed from disk still retire their own staging work; the
    /// session simply resumes recording after them.
    pub fn forward_layer(
        &mut self,
        layer_id: usize,
        stream: &GpuBuffer,
        pre_mix: &GpuBuffer,
    ) -> Result<(GpuBuffer, GpuBuffer)> {
        let session = self.device.begin_compute_session()?;
        let outcome = self.forward_layer_stages(layer_id, stream, pre_mix);
        session.finish(outcome)
    }

    fn forward_layer_stages(
        &mut self,
        layer_id: usize,
        stream: &GpuBuffer,
        pre_mix: &GpuBuffer,
    ) -> Result<(GpuBuffer, GpuBuffer)> {
        let rows = self.rows()?;
        let width = checked_size("backbone stream", &[rows, self.hc, self.hidden])?;
        if self.attention.next_layer() != Some(layer_id) {
            bail!("V4.1 backbone expected a different next layer");
        }
        if layer_id >= self.hyper.len()
            || !stream.belongs_to_device(&self.device)
            || stream.f32_capacity() < width
            || !pre_mix.belongs_to_device(&self.device)
            || pre_mix.f32_capacity() < rows * self.hc
        {
            bail!("V4.1 backbone input geometry or device mismatch");
        }
        let rounded_stream = self.cast.cast_buffer(stream, width, self.precision)?;
        let stream = &rounded_stream;
        stage_digest!(self, layer_id, "input", stream, width);
        trace_stage!(self, layer_id, "input", stream, width);
        trace_stage!(self, layer_id, "input_pre", pre_mix, rows * self.hc);
        let injected = match &self.engram {
            Some(engram) => engram.apply(layer_id, stream, rows)?,
            None => None,
        };
        let stream =
            self.cast
                .cast_buffer(injected.as_ref().unwrap_or(stream), width, self.precision)?;
        let stream = &stream;
        #[cfg(test)]
        if let Some(kv) = self.engram.as_ref().and_then(|e| e.projected_kv(layer_id)) {
            trace_stage!(
                self,
                layer_id,
                "engram_kv",
                kv,
                rows * (1 + self.hc) * self.hidden
            );
        }
        stage_digest!(self, layer_id, "engram", stream, width);
        trace_stage!(self, layer_id, "engram", stream, width);
        let attn = self.hyper[layer_id].attention_mixes(stream, rows)?;
        trace_stage!(
            self,
            layer_id,
            "attn_projection",
            attn.projection,
            rows * (2 + self.hc) * self.hc
        );
        trace_stage!(self, layer_id, "attn_pre", attn.pre, rows * self.hc);
        trace_stage!(self, layer_id, "attn_post", attn.post, rows * self.hc);
        trace_stage!(
            self,
            layer_id,
            "attn_comb",
            attn.comb,
            rows * self.hc * self.hc
        );
        let collapsed = self.hyper[layer_id].pre(stream, pre_mix, rows)?;
        let collapsed = self
            .cast
            .cast_buffer(&collapsed, rows * self.hidden, self.precision)?;
        trace_stage!(
            self,
            layer_id,
            "attn_collapse",
            collapsed,
            rows * self.hidden
        );
        let collapsed = self.norm.apply(
            &self.device,
            &collapsed,
            &self.attn_norm[layer_id],
            rows,
            self.hidden,
            self.norm_eps,
        )?;
        let collapsed = self
            .cast
            .cast_buffer(&collapsed, rows * self.hidden, self.precision)?;
        stage_digest!(self, layer_id, "attn_norm", collapsed, rows * self.hidden);
        trace_stage!(self, layer_id, "attn_norm", collapsed, rows * self.hidden);
        let attended = self.attention.forward_layer(layer_id, &collapsed)?;
        stage_digest!(self, layer_id, "attention", attended, rows * self.hidden);
        #[cfg(test)]
        if self.precision == DeepseekV41WeightOutput::Bf16 {
            if let Some((q, sparse)) = self.attention.sparse_trace(layer_id) {
                trace_stage!(self, layer_id, "attention_queries", q, q.f32_capacity());
                trace_stage!(
                    self,
                    layer_id,
                    "attention_sparse",
                    sparse,
                    sparse.f32_capacity()
                );
            }
        }
        #[cfg(test)]
        if let Some((input, grouped)) = self.attention.projection_trace(layer_id) {
            trace_stage!(
                self,
                layer_id,
                "attention_projection_input",
                input,
                input.f32_capacity()
            );
            trace_stage!(
                self,
                layer_id,
                "attention_grouped",
                grouped,
                grouped.f32_capacity()
            );
        }
        trace_stage!(self, layer_id, "attention", attended, rows * self.hidden);
        let stream = self.hyper[layer_id].post(&attended, stream, &attn.post, &attn.comb, rows)?;
        let stream = self.cast.cast_buffer(&stream, width, self.precision)?;
        stage_digest!(self, layer_id, "after_attention", stream, width);
        trace_stage!(self, layer_id, "after_attention", stream, width);
        let ffn = self.hyper[layer_id].ffn_mixes(&stream, rows)?;
        trace_stage!(
            self,
            layer_id,
            "ffn_projection",
            ffn.projection,
            rows * (2 + self.hc) * self.hc
        );
        trace_stage!(self, layer_id, "ffn_pre", ffn.pre, rows * self.hc);
        trace_stage!(self, layer_id, "ffn_post", ffn.post, rows * self.hc);
        trace_stage!(
            self,
            layer_id,
            "ffn_comb",
            ffn.comb,
            rows * self.hc * self.hc
        );
        let collapsed = self.hyper[layer_id].pre(&stream, &attn.pre, rows)?;
        let collapsed = self
            .cast
            .cast_buffer(&collapsed, rows * self.hidden, self.precision)?;
        trace_stage!(
            self,
            layer_id,
            "ffn_collapse",
            collapsed,
            rows * self.hidden
        );
        let collapsed = self.norm.apply(
            &self.device,
            &collapsed,
            &self.ffn_norm[layer_id],
            rows,
            self.hidden,
            self.norm_eps,
        )?;
        let collapsed = self
            .cast
            .cast_buffer(&collapsed, rows * self.hidden, self.precision)?;
        stage_digest!(self, layer_id, "ffn_norm", collapsed, rows * self.hidden);
        trace_stage!(self, layer_id, "ffn_norm", collapsed, rows * self.hidden);
        let mixed = self.moe[layer_id].apply(&collapsed, rows)?;
        stage_digest!(self, layer_id, "moe", mixed, rows * self.hidden);
        trace_stage!(self, layer_id, "moe", mixed, rows * self.hidden);
        let stream = self.hyper[layer_id].post(&mixed, &stream, &ffn.post, &ffn.comb, rows)?;
        let stream = self.cast.cast_buffer(&stream, width, self.precision)?;
        stage_digest!(self, layer_id, "output", stream, width);
        trace_stage!(self, layer_id, "output", stream, width);
        Ok((stream, ffn.pre))
    }

    /// Full start-zero prefill, returning the collapsed final hidden state.
    /// Token ids are required when Engram is configured; Engram hashes are built
    /// from the compressed-token cache before any layer runs.
    pub fn prefill(
        &mut self,
        stream: &GpuBuffer,
        seq: usize,
        token_ids: Option<&GpuBuffer>,
        token_mask: Option<&GpuBuffer>,
    ) -> Result<GpuBuffer> {
        self.attention.begin_prefill(seq)?;
        self.begin_engram(token_ids, token_mask, 0, seq)?;
        self.run_layers(stream)
    }

    /// One token per batch at the next position.
    pub fn decode(
        &mut self,
        stream: &GpuBuffer,
        start_pos: usize,
        token_ids: Option<&GpuBuffer>,
        token_mask: Option<&GpuBuffer>,
    ) -> Result<GpuBuffer> {
        self.attention.begin_decode(start_pos)?;
        self.begin_engram(token_ids, token_mask, start_pos, 1)?;
        self.run_layers(stream)
    }

    fn begin_engram(
        &mut self,
        token_ids: Option<&GpuBuffer>,
        token_mask: Option<&GpuBuffer>,
        start_pos: usize,
        seq: usize,
    ) -> Result<()> {
        match &mut self.engram {
            Some(engram) => {
                let ids =
                    token_ids.context("V4.1 prefill/decode with Engram requires token ids")?;
                engram.begin_step(ids, start_pos, seq, token_mask)
            }
            None if token_ids.is_some() => {
                bail!("V4.1 backbone was built without an Engram runtime")
            }
            None => Ok(()),
        }
    }

    fn run_layers(&mut self, stream: &GpuBuffer) -> Result<GpuBuffer> {
        #[cfg(test)]
        if let Some(trace) = &mut self.trace {
            trace.clear();
        }
        let rows = self.rows()?;
        let width = checked_size("backbone stream", &[rows, self.hc, self.hidden])?;
        if !stream.belongs_to_device(&self.device) || stream.f32_capacity() < width {
            bail!("V4.1 backbone stream geometry or device mismatch");
        }
        let mut current = stream.clone();
        let mut pre_mix = self.hyper[0].identity_pre_mix(rows)?;
        for layer in 0..self.hyper.len() {
            let (next, ffn_pre) = self.forward_layer(layer, &current, &pre_mix)?;
            if digest_stage_enabled("layer") {
                // Sampled after the layer retired, so the read cannot split the
                // session it is measuring. This is the bit-exactness probe a
                // kernel change has to keep constant.
                eprintln!(
                    "v41_layer_digest layer={layer} digest={:016x}",
                    digest_of(&next, width)?
                );
            }
            current = next;
            pre_mix = ffn_pre;
        }
        let collapsed = self.hyper[0].pre(&current, &pre_mix, rows)?;
        self.cast
            .cast_buffer(&collapsed, rows * self.hidden, self.precision)
    }

    /// Final collapse for a caller that owns the head.
    pub fn collapse(
        &self,
        stream: &GpuBuffer,
        pre_mix: &GpuBuffer,
        rows: usize,
    ) -> Result<GpuBuffer> {
        let collapsed = self.hyper[0].pre(stream, pre_mix, rows)?;
        self.cast
            .cast_buffer(&collapsed, rows * self.hidden, self.precision)
    }
}
