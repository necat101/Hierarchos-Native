//! Strict gate for the V4.1 Hyper-Connection and MoE adjoints.
//!
//! Every expectation is CPU PyTorch autograd on the same operands, pinned as
//! FP32 bit patterns by `validation/validate_deepseek_v41_block_grad.py`.
//! The stages are checked separately so a failure names the stage, and the MoE
//! input gradient is additionally checked as a chain.

use anyhow::Result;

use super::*;
use crate::{GpuBuffer, VulkanDevice};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    seed: u64,
    tolerance: f32,
    comparison: String,
    rtol: f32,
    ffn_norm: FfnNormCase,
    hc_pre: HcPreCase,
    hc_stages: Vec<HcStageCase>,
    hc_post: HcPostCase,
    moe_cases: Vec<MoeCase>,
}

#[derive(Deserialize)]
struct FfnNormCase {
    rows: usize,
    hidden: usize,
    eps: f32,
    x_bits: Vec<u32>,
    weight_bits: Vec<u32>,
    upstream_bits: Vec<u32>,
    grad_x_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct HcPreCase {
    rows: usize,
    hc: usize,
    dim: usize,
    stream_bits: Vec<u32>,
    pre_bits: Vec<u32>,
    upstream_bits: Vec<u32>,
    grad_stream_bits: Vec<u32>,
    grad_pre_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct HcStageCase {
    name: String,
    rows: usize,
    hc: usize,
    dim: usize,
    mix_hc: usize,
    iterations: usize,
    eps: f32,
    norm_eps: f32,
    raw_bits: Vec<u32>,
    stream_bits: Vec<u32>,
    scale: Vec<f32>,
    base_bits: Vec<u32>,
    weight_bits: Vec<u32>,
    grad_pre_bits: Vec<u32>,
    grad_post_bits: Vec<u32>,
    grad_comb_bits: Vec<u32>,
    grad_mix_bits: Vec<u32>,
    grad_stream_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct HcPostCase {
    rows: usize,
    hc: usize,
    dim: usize,
    x_bits: Vec<u32>,
    residual_bits: Vec<u32>,
    post_bits: Vec<u32>,
    comb_bits: Vec<u32>,
    grad_output_bits: Vec<u32>,
    grad_x_bits: Vec<u32>,
    grad_residual_bits: Vec<u32>,
    grad_post_bits: Vec<u32>,
    grad_comb_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct MoeCase {
    name: String,
    rows: usize,
    hidden: usize,
    inter: usize,
    experts: usize,
    topk: usize,
    score: String,
    gate_temp: f32,
    norm_topk: bool,
    route_scale: f32,
    swiglu_limit: f32,
    x_bits: Vec<u32>,
    gate_weight_bits: Vec<u32>,
    gate_bias_bits: Vec<u32>,
    expert_w1_bits: Vec<Vec<u32>>,
    expert_w2_bits: Vec<Vec<u32>>,
    expert_w3_bits: Vec<Vec<u32>>,
    shared_w1_bits: Vec<u32>,
    shared_w2_bits: Vec<u32>,
    shared_w3_bits: Vec<u32>,
    grad_output_bits: Vec<u32>,
    indices: Vec<u32>,
    weights_bits: Vec<u32>,
    combine_grad_routed_bits: Vec<u32>,
    combine_grad_shared_bits: Vec<u32>,
    stage_expert: usize,
    expert_grad_input_bits: Vec<u32>,
    expert_grad_weight_bits: Vec<u32>,
    shared_grad_x_bits: Vec<u32>,
    gate_grad_weights_bits: Vec<u32>,
    gate_grad_x_bits: Vec<u32>,
    total_grad_x_bits: Vec<u32>,
}

fn values(bits: &[u32]) -> Vec<f32> {
    bits.iter().copied().map(f32::from_bits).collect()
}

fn values_all(bits: &[Vec<u32>]) -> Vec<Vec<f32>> {
    bits.iter().map(|inner| values(inner)).collect()
}

struct Gate<'a> {
    label: &'a str,
    tolerance: f32,
    worst: f64,
    compared: usize,
}

impl<'a> Gate<'a> {
    fn new(label: &'a str, tolerance: f32) -> Self {
        Self {
            label,
            tolerance,
            worst: 0.0,
            compared: 0,
        }
    }

    fn check(&mut self, stage: &str, actual: &[f32], expected_bits: &[u32]) {
        assert_eq!(
            actual.len(),
            expected_bits.len(),
            "{} {stage} count",
            self.label
        );
        let mut worst = (0.0f64, 0usize, 0.0f32, 0.0f32);
        for (index, (&left, &right)) in actual.iter().zip(expected_bits).enumerate() {
            let right = f32::from_bits(right);
            let delta = (left as f64 - right as f64).abs();
            if delta > worst.0 {
                worst = (delta, index, left, right);
            }
        }
        eprintln!(
            "V4.1 {} {stage}: worst |{} - {}| = {:.17e} at {}",
            self.label, worst.2, worst.3, worst.0, worst.1
        );
        assert!(
            worst.0 < self.tolerance as f64,
            "{} {stage} exceeds strict <{:e}: {:.17e} at {} (native {}, CPU {})",
            self.label,
            self.tolerance,
            worst.0,
            worst.1,
            worst.2,
            worst.3
        );
        assert!(
            actual.iter().all(|value| value.is_finite()),
            "{} {stage} has a non-finite value",
            self.label
        );
        self.worst = self.worst.max(worst.0);
        self.compared += actual.len();
    }
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_block_grad.json"
    ))
    .unwrap()
}

#[test]
fn deepseek_v41_ffn_norm_adjoint_matches_the_pinned_cpu_fixture() -> Result<()> {
    let Ok(device) = VulkanDevice::new() else {
        return Ok(());
    };
    let fixture = fixture();
    assert_eq!(fixture.seed, 4141107);
    assert_eq!(fixture.comparison, "strict_less_than");
    assert_eq!(fixture.rtol, 0.0);
    let case = &fixture.ffn_norm;
    let adjoints = DeepseekV41BlockAdjoints::new(device.clone())?;
    let x = GpuBuffer::from_f32(&device, &values(&case.x_bits))?;
    let weight = GpuBuffer::from_f32(&device, &values(&case.weight_bits))?;
    let upstream = GpuBuffer::from_f32(&device, &values(&case.upstream_bits))?;
    let grad_x = adjoints.ffn_norm(&upstream, &x, &weight, case.rows, case.hidden, case.eps)?;
    let mut gate = Gate::new("ffn_norm", fixture.tolerance);
    gate.check(
        "grad_x",
        &grad_x.read_f32(case.rows * case.hidden)?,
        &case.grad_x_bits,
    );
    eprintln!(
        "V4.1 FFN norm adjoint worst drift {:e} over {} values",
        gate.worst, gate.compared
    );
    Ok(())
}

fn stage_hyper(
    device: &VulkanDevice,
    case: &HcStageCase,
) -> Result<(DeepseekV41HyperConnections, DeepseekV41HyperStage)> {
    let width = case.hc * case.dim;
    Ok((
        DeepseekV41HyperConnections::new(
            device.clone(),
            case.dim,
            case.hc,
            case.iterations,
            case.eps,
            case.norm_eps,
            DeepseekV41HyperConnectionWeights {
                attn_fn: values(&case.weight_bits),
                attn_base: values(&case.base_bits),
                attn_scale: case.scale.clone(),
                ffn_fn: vec![0.0; case.mix_hc * width],
                ffn_base: vec![0.0; case.mix_hc],
                ffn_scale: vec![0.0; 3],
            },
        )?,
        DeepseekV41HyperStage::Attention,
    ))
}

#[test]
fn deepseek_v41_hc_collapse_adjoints_match_the_pinned_cpu_fixture() -> Result<()> {
    let Ok(device) = VulkanDevice::new() else {
        return Ok(());
    };
    let fixture = fixture();
    let adjoints = DeepseekV41BlockAdjoints::new(device.clone())?;
    let mut overall = Gate::new("hc", fixture.tolerance);

    let case = &fixture.hc_pre;
    let stream = GpuBuffer::from_f32(&device, &values(&case.stream_bits))?;
    let pre = GpuBuffer::from_f32(&device, &values(&case.pre_bits))?;
    let upstream = GpuBuffer::from_f32(&device, &values(&case.upstream_bits))?;
    let gradients = adjoints.hc_pre(&upstream, &stream, &pre, case.rows, case.hc, case.dim)?;
    let mut gate = Gate::new("hc_pre", fixture.tolerance);
    gate.check(
        "grad_stream",
        &gradients
            .grad_stream
            .read_f32(case.rows * case.hc * case.dim)?,
        &case.grad_stream_bits,
    );
    gate.check(
        "grad_pre",
        &gradients.grad_pre.read_f32(case.rows * case.hc)?,
        &case.grad_pre_bits,
    );
    eprintln!(
        "V4.1 hc_pre collapse adjoint: worst drift {:e} over {} values",
        gate.worst, gate.compared
    );
    overall.worst = overall.worst.max(gate.worst);
    overall.compared += gate.compared;

    for case in &fixture.hc_stages {
        let (hyper, stage) = stage_hyper(&device, case)?;
        let raw = GpuBuffer::from_f32(&device, &values(&case.raw_bits))?;
        let stream = GpuBuffer::from_f32(&device, &values(&case.stream_bits))?;
        let grad_pre = GpuBuffer::from_f32(&device, &values(&case.grad_pre_bits))?;
        let grad_post = GpuBuffer::from_f32(&device, &values(&case.grad_post_bits))?;
        let grad_comb = GpuBuffer::from_f32(&device, &values(&case.grad_comb_bits))?;
        let mut gate = Gate::new(&case.name, fixture.tolerance);

        let grad_mix = adjoints.hc_split(
            &hyper, stage, &raw, &stream, &grad_pre, &grad_post, &grad_comb, case.rows,
        )?;
        gate.check(
            "hc_split_grad_mix",
            &grad_mix.read_f32(case.rows * case.mix_hc)?,
            &case.grad_mix_bits,
        );

        let grad_stream = adjoints.hc_mixes(&hyper, stage, &grad_mix, &raw, &stream, case.rows)?;
        gate.check(
            "hc_mixes_grad_stream",
            &grad_stream.read_f32(case.rows * case.hc * case.dim)?,
            &case.grad_stream_bits,
        );
        eprintln!(
            "V4.1 {} Hyper-Connection collapse/Sinkhorn adjoints: worst drift {:e} over {} values",
            case.name, gate.worst, gate.compared
        );
        overall.worst = overall.worst.max(gate.worst);
        overall.compared += gate.compared;
    }

    eprintln!(
        "V4.1 HC collapse adjoints aggregate worst drift {:e} over {} values",
        overall.worst, overall.compared
    );
    Ok(())
}

#[test]
fn deepseek_v41_hc_post_adjoint_matches_the_pinned_cpu_fixture() -> Result<()> {
    let Ok(device) = VulkanDevice::new() else {
        return Ok(());
    };
    let fixture = fixture();
    assert_eq!(fixture.seed, 4141107);
    assert_eq!(fixture.comparison, "strict_less_than");
    assert_eq!(fixture.rtol, 0.0);
    let case = &fixture.hc_post;
    assert_eq!(case.hc, 3);
    assert_eq!(case.dim, 8);

    let mix_hc = (2 + case.hc) * case.hc;
    let zeros = vec![0.0f32; mix_hc * case.hc * case.dim];
    let hyper = DeepseekV41HyperConnections::new(
        device.clone(),
        case.dim,
        case.hc,
        3,
        1e-6,
        1e-6,
        DeepseekV41HyperConnectionWeights {
            attn_fn: zeros,
            attn_base: vec![0.0; mix_hc],
            attn_scale: vec![0.0; 3],
            ffn_fn: vec![0.0; mix_hc * case.hc * case.dim],
            ffn_base: vec![0.0; mix_hc],
            ffn_scale: vec![0.0; 3],
        },
    )?;
    let adjoints = DeepseekV41BlockAdjoints::new(device.clone())?;
    let x = GpuBuffer::from_f32(&device, &values(&case.x_bits))?;
    let residual = GpuBuffer::from_f32(&device, &values(&case.residual_bits))?;
    let post = GpuBuffer::from_f32(&device, &values(&case.post_bits))?;
    let comb = GpuBuffer::from_f32(&device, &values(&case.comb_bits))?;
    let grad_output = GpuBuffer::from_f32(&device, &values(&case.grad_output_bits))?;
    let gradients =
        adjoints.hc_post(&hyper, &grad_output, &x, &residual, &post, &comb, case.rows)?;

    let mut gate = Gate::new("hc_post", fixture.tolerance);
    gate.check(
        "grad_x",
        &gradients.grad_x.read_f32(case.rows * case.dim)?,
        &case.grad_x_bits,
    );
    gate.check(
        "grad_residual",
        &gradients
            .grad_residual
            .read_f32(case.rows * case.hc * case.dim)?,
        &case.grad_residual_bits,
    );
    gate.check(
        "grad_post",
        &gradients.grad_post.read_f32(case.rows * case.hc)?,
        &case.grad_post_bits,
    );
    gate.check(
        "grad_comb",
        &gradients
            .grad_comb
            .read_f32(case.rows * case.hc * case.hc)?,
        &case.grad_comb_bits,
    );
    eprintln!(
        "V4.1 hc_post adjoint worst drift {:e} over {} values",
        gate.worst, gate.compared
    );
    Ok(())
}

#[test]
fn deepseek_v41_moe_adjoints_match_the_pinned_cpu_fixture() -> Result<()> {
    let Ok(device) = VulkanDevice::new() else {
        return Ok(());
    };
    let fixture = fixture();
    let adjoints = DeepseekV41BlockAdjoints::new(device.clone())?;
    let mut overall = Gate::new("moe", fixture.tolerance);
    for case in &fixture.moe_cases {
        let slots = case.rows * case.topk;
        let moe = DeepseekV41Moe::new(
            device.clone(),
            DeepseekV41MoeConfig {
                hidden_size: case.hidden,
                intermediate_size: case.inter,
                experts: case.experts,
                activated: case.topk,
                score_func: DeepseekV41MoeScoreFunc::parse(&case.score)?,
                gate_temp: case.gate_temp,
                norm_topk_prob: case.norm_topk,
                route_scale: case.route_scale,
                swiglu_limit: case.swiglu_limit,
            },
            DeepseekV41MoeWeights {
                gate_weight: values(&case.gate_weight_bits),
                gate_bias: values(&case.gate_bias_bits),
                expert_w1: values_all(&case.expert_w1_bits),
                expert_w2: values_all(&case.expert_w2_bits),
                expert_w3: values_all(&case.expert_w3_bits),
                shared_w1: values(&case.shared_w1_bits),
                shared_w2: values(&case.shared_w2_bits),
                shared_w3: values(&case.shared_w3_bits),
            },
        )?;
        let mut gate = Gate::new(&case.name, fixture.tolerance);
        let x = GpuBuffer::from_f32(&device, &values(&case.x_bits))?;
        let grad_output = GpuBuffer::from_f32(&device, &values(&case.grad_output_bits))?;

        // The routing the fixture pinned is the routing the adjoint uses.
        let (weights, indices, counts) = moe.route_slots(&x, case.rows)?;
        assert_eq!(
            indices.read_u32(slots)?,
            case.indices,
            "{} routed indices",
            case.name
        );
        gate.check(
            "routing_weights",
            &weights.read_f32(slots)?,
            &case.weights_bits,
        );

        let combine = adjoints.moe_combine(&moe, &grad_output, case.rows)?;
        gate.check(
            "combine_grad_routed",
            &combine.grad_routed.read_f32(slots * case.hidden)?,
            &case.combine_grad_routed_bits,
        );
        gate.check(
            "combine_grad_shared",
            &combine.grad_shared.read_f32(case.rows * case.hidden)?,
            &case.combine_grad_shared_bits,
        );

        let expert = adjoints.moe_expert(
            &moe,
            case.stage_expert as u32,
            &combine.grad_routed,
            &x,
            &weights,
            &indices,
            &counts,
            case.rows,
        )?;
        gate.check(
            "expert_grad_input",
            &expert.grad_input.read_f32(slots * case.hidden)?,
            &case.expert_grad_input_bits,
        );
        gate.check(
            "expert_grad_weight",
            &expert.grad_weight.read_f32(slots)?,
            &case.expert_grad_weight_bits,
        );

        let shared = adjoints.moe_shared_expert(&moe, &combine.grad_shared, &x, case.rows)?;
        gate.check(
            "shared_grad_x",
            &shared.read_f32(case.rows * case.hidden)?,
            &case.shared_grad_x_bits,
        );

        let gate_grad_weights =
            GpuBuffer::from_f32(&device, &values(&case.gate_grad_weights_bits))?;
        let gate_grad = adjoints.moe_gate(&moe, &gate_grad_weights, &x, &indices, case.rows)?;
        gate.check(
            "gate_grad_x",
            &gate_grad.read_f32(case.rows * case.hidden)?,
            &case.gate_grad_x_bits,
        );

        let total = adjoints.moe(&moe, &grad_output, &x, case.rows)?;
        gate.check(
            "chained_grad_x",
            &total.read_f32(case.rows * case.hidden)?,
            &case.total_grad_x_bits,
        );

        overall.worst = overall.worst.max(gate.worst);
        overall.compared += gate.compared;
        eprintln!(
            "V4.1 {} MoE adjoints: worst drift {:e} over {} values",
            case.name, gate.worst, gate.compared
        );
    }
    eprintln!(
        "V4.1 MoE adjoints aggregate worst drift {:e} over {} values",
        overall.worst, overall.compared
    );
    Ok(())
}
