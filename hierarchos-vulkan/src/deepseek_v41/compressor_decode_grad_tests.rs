//! The decode-state chain of the compressor adjoint: one cached step at a time,
//! chained backwards through the pool state the forward carries across steps.

use super::*;
use crate::deepseek_v41::{
    DeepseekV41Compressor, DeepseekV41CompressorDecode, DeepseekV41CompressorDecodeOperands,
    DeepseekV41CompressorWeights, REFERENCE_REVISION,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    revision: String,
    tolerance: f64,
    comparison: String,
    rtol: f64,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    ratio: usize,
    hidden: usize,
    head: usize,
    batch: usize,
    prefill: usize,
    kv_weight_bits: Vec<u32>,
    gate_weight_bits: Vec<u32>,
    norm_weight_bits: Vec<u32>,
    prefill_input_bits: Vec<u32>,
    state0_kv_bits: Vec<u32>,
    state0_score_bits: Vec<u32>,
    final_seed_kv_bits: Vec<u32>,
    final_seed_score_bits: Vec<u32>,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Step {
    start_pos: usize,
    groups: usize,
    input_bits: Vec<u32>,
    projected_kv_bits: Vec<u32>,
    projected_gate_bits: Vec<u32>,
    pooled_bits: Vec<u32>,
    latent_bits: Vec<u32>,
    upstream_bits: Vec<u32>,
    state_kv_bits: Vec<u32>,
    state_score_bits: Vec<u32>,
    expected: Expected,
}

#[derive(Deserialize)]
struct Expected {
    grad_input_bits: Vec<u32>,
    grad_state_kv_bits: Vec<u32>,
    grad_state_score_bits: Vec<u32>,
}

fn fixture() -> Fixture {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_compressor_decode_grad.json"
    ))
    .unwrap();
    assert_eq!(fixture.revision, REFERENCE_REVISION);
    assert_eq!(fixture.tolerance, 2e-7);
    assert_eq!(fixture.comparison, "strict_less_than");
    assert_eq!(fixture.rtol, 0.0);
    fixture
}

fn values(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

fn gpu(device: &VulkanDevice, bits: &[u32]) -> Result<GpuBuffer> {
    GpuBuffer::from_u32(device, bits)
}

fn optional_gpu(device: &VulkanDevice, bits: &[u32]) -> Result<Option<GpuBuffer>> {
    (!bits.is_empty()).then(|| gpu(device, bits)).transpose()
}

fn placeholder_or_gpu(device: &VulkanDevice, bits: &[u32]) -> Result<GpuBuffer> {
    if bits.is_empty() {
        GpuBuffer::zeros_f32(device, 1)
    } else {
        gpu(device, bits)
    }
}

fn read_or_empty(buffer: &GpuBuffer, len: usize) -> Result<Vec<f32>> {
    if len == 0 {
        Ok(Vec::new())
    } else {
        buffer.read_f32(len)
    }
}

/// The reference seeds an untouched score slot with `-inf`, the native state
/// with zero. Neither is ever read: a pooled group always writes both slots
/// first, so only finite reference entries carry a value the forward used.
fn gate_state(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    failures: &mut Vec<String>,
    compared: &mut usize,
) -> f64 {
    assert_eq!(actual.len(), expected.len(), "{label}");
    let (actual, expected): (Vec<_>, Vec<_>) = actual
        .iter()
        .zip(expected)
        .filter(|(_, reference)| reference.is_finite())
        .map(|(&actual, &reference)| (actual, reference.to_bits()))
        .unzip();
    gate(label, &actual, &expected, failures, compared)
}

fn gate(
    label: &str,
    actual: &[f32],
    expected: &[u32],
    failures: &mut Vec<String>,
    compared: &mut usize,
) -> f64 {
    assert_eq!(actual.len(), expected.len(), "{label}");
    let mut worst = 0.0f64;
    let mut at = 0;
    for (index, (&value, &reference)) in actual.iter().zip(expected).enumerate() {
        let reference = f32::from_bits(reference);
        assert!(
            value.is_finite() && reference.is_finite(),
            "{label} non-finite"
        );
        let drift = (value as f64 - reference as f64).abs();
        if drift > worst {
            worst = drift;
            at = index;
        }
    }
    *compared += actual.len();
    eprintln!(
        "V4.1 compressor decode adjoint {label}: {} values, worst {worst:.17e}",
        actual.len()
    );
    if worst >= 2e-7 {
        failures.push(format!(
            "{label}: {worst:.17e} at {at}, native={} cpu={}",
            actual[at],
            f32::from_bits(expected[at])
        ));
    }
    worst
}

fn emits(case: &Case, step: &Step) -> usize {
    if case.ratio == 1 {
        1
    } else {
        usize::from((step.start_pos + 1).is_multiple_of(case.ratio))
    }
}

fn decode(case: &Case, step: &Step) -> DeepseekV41CompressorDecode {
    DeepseekV41CompressorDecode {
        batch: case.batch,
        start_pos: step.start_pos,
        hidden: case.hidden,
        dim: case.head,
        ratio: case.ratio,
        eps: 1e-20,
    }
}

fn seed(device: &VulkanDevice, case: &Case) -> Result<Option<(GpuBuffer, GpuBuffer)>> {
    match (
        optional_gpu(device, &case.final_seed_kv_bits)?,
        optional_gpu(device, &case.final_seed_score_bits)?,
    ) {
        (Some(kv), Some(score)) => Ok(Some((kv, score))),
        (None, None) => Ok(None),
        _ => bail!("{}: half a final state seed", case.name),
    }
}

/// Every step's VJP, chained backwards: each returned pool-state gradient is the
/// incoming state gradient of the step before it.
#[test]
fn deepseek_v41_compressor_decode_adjoint_chains_state_across_steps_matching_cpu() -> Result<()> {
    let device = VulkanDevice::new()?;
    let adjoints = DeepseekV41CompressorAdjoints::new(device.clone())?;
    let mut failures = Vec::new();
    let mut compared = 0usize;
    let mut worst = 0.0f64;

    for case in fixture().cases {
        let kv_weight = gpu(&device, &case.kv_weight_bits)?;
        let gate_weight = optional_gpu(&device, &case.gate_weight_bits)?;
        let norm_weight = gpu(&device, &case.norm_weight_bits)?;
        let mut incoming = seed(&device, &case)?;
        for step in case.steps.iter().rev() {
            assert_eq!(step.groups, emits(&case, step), "{} step shape", case.name);
            let pooled = placeholder_or_gpu(&device, &step.pooled_bits)?;
            let upstream = placeholder_or_gpu(&device, &step.upstream_bits)?;
            let projected_kv = gpu(&device, &step.projected_kv_bits)?;
            let projected_gate = optional_gpu(&device, &step.projected_gate_bits)?;
            let state_kv = optional_gpu(&device, &step.state_kv_bits)?;
            let state_score = optional_gpu(&device, &step.state_score_bits)?;
            let operands = DeepseekV41CompressorDecodeOperands {
                pooled: &pooled,
                projected_kv: &projected_kv,
                projected_gate: projected_gate.as_ref(),
                state_kv: state_kv.as_ref(),
                state_score: state_score.as_ref(),
                kv_weight: &kv_weight,
                gate_weight: gate_weight.as_ref(),
                norm_weight: &norm_weight,
            };
            let gradients = adjoints.decode(
                &decode(&case, step),
                &operands,
                &upstream,
                incoming.as_ref().map(|(kv, score)| (kv, score)),
            )?;
            worst = worst.max(gate(
                &format!("{} pos {} grad_input", case.name, step.start_pos),
                &gradients.grad_input.read_f32(case.batch * case.hidden)?,
                &step.expected.grad_input_bits,
                &mut failures,
                &mut compared,
            ));
            match (&gradients.grad_state_kv, &gradients.grad_state_score) {
                (Some(state_kv_grad), Some(state_score_grad)) => {
                    let len = case.batch * case.ratio * case.head;
                    worst = worst.max(gate(
                        &format!("{} pos {} grad_state_kv", case.name, step.start_pos),
                        &state_kv_grad.read_f32(len)?,
                        &step.expected.grad_state_kv_bits,
                        &mut failures,
                        &mut compared,
                    ));
                    worst = worst.max(gate(
                        &format!("{} pos {} grad_state_score", case.name, step.start_pos),
                        &state_score_grad.read_f32(len)?,
                        &step.expected.grad_state_score_bits,
                        &mut failures,
                        &mut compared,
                    ));
                }
                (None, None) => {
                    assert!(case.ratio == 1, "{} returned no pool state", case.name);
                    assert!(step.expected.grad_state_kv_bits.is_empty());
                    assert!(step.expected.grad_state_score_bits.is_empty());
                }
                _ => failures.push(format!(
                    "{} pos {}: pool-state gradient halves disagree",
                    case.name, step.start_pos
                )),
            }
            incoming = match (gradients.grad_state_kv, gradients.grad_state_score) {
                (Some(kv), Some(score)) => Some((kv, score)),
                _ => None,
            };
        }
    }

    assert!(failures.is_empty(), "{failures:#?}");
    eprintln!(
        "V4.1 compressor decode chain on {}: {compared} frozen gradient values, worst absolute drift {worst:.17e}",
        device.name()
    );
    Ok(())
}

/// The same chain driven from the native compressor's own forward: prefill plus
/// the cached steps supply the projected rows, norm inputs and pool state, so
/// this checks operand plumbing rather than a re-derived fixture layout.
#[test]
fn deepseek_v41_compressor_decode_adjoint_runs_on_native_state() -> Result<()> {
    let device = VulkanDevice::new()?;
    let adjoints = DeepseekV41CompressorAdjoints::new(device.clone())?;
    let mut compared = 0usize;
    let mut worst = 0.0f64;
    let mut operand_worst = 0.0f64;
    let mut failures = Vec::new();
    let mut operand_values = 0usize;

    for case in fixture().cases {
        let weights = DeepseekV41CompressorWeights {
            hidden_size: case.hidden,
            head_dim: case.head,
            compress_ratio: case.ratio,
            eps: 1e-20,
            kv_weight: values(&case.kv_weight_bits),
            gate_weight: (case.ratio == 2).then(|| values(&case.gate_weight_bits)),
            norm_weight: values(&case.norm_weight_bits),
        };
        let mut module =
            DeepseekV41Compressor::new(device.clone(), weights, case.batch, case.prefill)?;
        let prefill_input = gpu(&device, &case.prefill_input_bits)?;
        module.prefill_from_buffer(&prefill_input, case.batch, case.prefill)?;
        let state_len = case.batch * case.ratio * case.head;
        let mut states: Vec<Option<(Vec<f32>, Vec<f32>)>> = Vec::new();
        let mut latents_per_step = Vec::new();
        if case.ratio == 2 {
            let (state_kv, state_score) = module.pool_state();
            for (label, actual, reference) in [
                (
                    "initial KV",
                    state_kv.read_f32(state_len)?,
                    values(&case.state0_kv_bits),
                ),
                (
                    "initial score",
                    state_score.read_f32(state_len)?,
                    values(&case.state0_score_bits),
                ),
            ] {
                let drift = gate_state(
                    &format!("{} native {label}", case.name),
                    &actual,
                    &reference,
                    &mut failures,
                    &mut operand_values,
                );
                operand_worst = operand_worst.max(drift);
            }
        }
        for step in &case.steps {
            let input = gpu(&device, &step.input_bits)?;
            let latents = module.decode_step_from_buffer(&input, case.batch, step.start_pos)?;
            let emitted = latents.read_f32()?;
            for (label, actual, expected) in [
                (
                    "projected_kv",
                    read_or_empty(&latents.projected_kv, step.projected_kv_bits.len())?,
                    &step.projected_kv_bits,
                ),
                (
                    "projected_gate",
                    read_or_empty(&latents.projected_gate, step.projected_gate_bits.len())?,
                    &step.projected_gate_bits,
                ),
                (
                    "pooled",
                    read_or_empty(&latents.pre_norm, step.pooled_bits.len())?,
                    &step.pooled_bits,
                ),
                ("latent", emitted.clone(), &step.latent_bits),
            ] {
                if expected.is_empty() {
                    continue;
                }
                gate(
                    &format!("{} native pos {} {label}", case.name, step.start_pos),
                    &actual,
                    expected,
                    &mut failures,
                    &mut operand_values,
                );
            }
            let drift = emitted
                .iter()
                .zip(values(&step.latent_bits))
                .map(|(&a, b)| (a as f64 - b as f64).abs())
                .fold(0.0f64, f64::max);
            // Gate the forward itself as well as every native-operand VJP.
            operand_worst = operand_worst.max(drift);
            if case.ratio == 2 {
                let (state_kv, state_score) = module.pool_state();
                let state_kv = state_kv.read_f32(state_len)?;
                let state_score = state_score.read_f32(state_len)?;
                for (label, actual, reference) in [
                    ("state KV", state_kv.as_slice(), values(&step.state_kv_bits)),
                    (
                        "state score",
                        state_score.as_slice(),
                        values(&step.state_score_bits),
                    ),
                ] {
                    let drift = gate_state(
                        &format!("{} native pos {} {label}", case.name, step.start_pos),
                        actual,
                        &reference,
                        &mut failures,
                        &mut operand_values,
                    );
                    operand_worst = operand_worst.max(drift);
                }
                states.push(Some((state_kv, state_score)));
            } else {
                states.push(None);
            }
            latents_per_step.push(latents);
        }

        let mut incoming = seed(&device, &case)?;
        let kv_weight = gpu(&device, &case.kv_weight_bits)?;
        let gate_weight = optional_gpu(&device, &case.gate_weight_bits)?;
        let norm_weight = gpu(&device, &case.norm_weight_bits)?;
        for (index, step) in case.steps.iter().enumerate().rev() {
            let latents = &latents_per_step[index];
            let state_kv = match &states[index] {
                Some((kv, _)) => Some(GpuBuffer::from_f32(&device, kv)?),
                None => None,
            };
            let state_score = match &states[index] {
                Some((_, score)) => Some(GpuBuffer::from_f32(&device, score)?),
                None => None,
            };
            let upstream = placeholder_or_gpu(&device, &step.upstream_bits)?;
            let operands = DeepseekV41CompressorDecodeOperands {
                pooled: &latents.pre_norm,
                projected_kv: &latents.projected_kv,
                projected_gate: (case.ratio == 2).then_some(&latents.projected_gate),
                state_kv: state_kv.as_ref(),
                state_score: state_score.as_ref(),
                kv_weight: &kv_weight,
                gate_weight: gate_weight.as_ref(),
                norm_weight: &norm_weight,
            };
            let gradients = adjoints.decode(
                &decode(&case, step),
                &operands,
                &upstream,
                incoming.as_ref().map(|(kv, score)| (kv, score)),
            )?;
            worst = worst.max(gate(
                &format!("{} native pos {} grad_input", case.name, step.start_pos),
                &gradients.grad_input.read_f32(case.batch * case.hidden)?,
                &step.expected.grad_input_bits,
                &mut failures,
                &mut compared,
            ));
            if let (Some(state_kv_grad), Some(state_score_grad)) =
                (&gradients.grad_state_kv, &gradients.grad_state_score)
            {
                worst = worst.max(gate(
                    &format!("{} native pos {} grad_state_kv", case.name, step.start_pos),
                    &state_kv_grad.read_f32(state_len)?,
                    &step.expected.grad_state_kv_bits,
                    &mut failures,
                    &mut compared,
                ));
                worst = worst.max(gate(
                    &format!(
                        "{} native pos {} grad_state_score",
                        case.name, step.start_pos
                    ),
                    &state_score_grad.read_f32(state_len)?,
                    &step.expected.grad_state_score_bits,
                    &mut failures,
                    &mut compared,
                ));
            }
            match (gradients.grad_state_kv, gradients.grad_state_score) {
                (Some(kv), Some(score)) => incoming = Some((kv, score)),
                _ => incoming = None,
            }
        }
        assert_eq!(
            latents_per_step.len(),
            case.steps.len(),
            "{}: step count",
            case.name
        );
    }

    eprintln!(
        "V4.1 compressor decode chain through the native forward: {compared} values, worst absolute drift {worst:.17e}, operand drift {operand_worst:.17e}"
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(
        operand_worst < 2e-7,
        "native forward operands exceed strict gate: {operand_worst:.17e}"
    );
    eprintln!("V4.1 compressor native forward: {operand_values} gated operand values");
    assert!(
        worst < 2e-7,
        "native-operand adjoints exceed strict gate: {worst:.17e}"
    );
    Ok(())
}

#[test]
fn deepseek_v41_compressor_decode_adjoint_rejects_unqualified_calls() -> Result<()> {
    let device = VulkanDevice::new()?;
    let adjoints = DeepseekV41CompressorAdjoints::new(device.clone())?;
    let zeros = |len: usize| GpuBuffer::zeros_f32(&device, len);
    let weights = zeros(16)?;
    let norm = zeros(4)?;
    let pooled = zeros(4)?;
    let upstream = zeros(4)?;
    let state = zeros(8)?;
    let step = DeepseekV41CompressorDecode {
        batch: 1,
        start_pos: 2,
        hidden: 4,
        dim: 4,
        ratio: 2,
        eps: 1e-20,
    };
    let ratio_two = DeepseekV41CompressorDecodeOperands {
        pooled: &pooled,
        projected_kv: &weights,
        projected_gate: Some(&weights),
        state_kv: Some(&state),
        state_score: Some(&state),
        kv_weight: &weights,
        gate_weight: Some(&weights),
        norm_weight: &norm,
    };
    // Position zero is a prefill evaluation.
    let mut prefill_step = step;
    prefill_step.start_pos = 0;
    let err = adjoints
        .decode(&prefill_step, &ratio_two, &upstream, None)
        .err()
        .expect("position zero must be rejected")
        .to_string();
    assert!(err.contains("prefill"), "{err}");
    // A ratio-1 decode has no gate and never reads its state.
    let mut ratio_one = step;
    ratio_one.ratio = 1;
    assert!(adjoints
        .decode(&ratio_one, &ratio_two, &upstream, None)
        .is_err());
    let ratio_one_operands = DeepseekV41CompressorDecodeOperands {
        projected_gate: None,
        state_kv: None,
        state_score: None,
        gate_weight: None,
        ..ratio_two
    };
    assert!(adjoints
        .decode(
            &ratio_one,
            &ratio_one_operands,
            &upstream,
            Some((&state, &state))
        )
        .is_err());
    // A ratio-2 decode needs its gate and the state after the step.
    assert!(adjoints
        .decode(&step, &ratio_one_operands, &upstream, None)
        .is_err());
    let no_state = DeepseekV41CompressorDecodeOperands {
        state_kv: None,
        state_score: None,
        ..ratio_two
    };
    assert!(adjoints.decode(&step, &no_state, &upstream, None).is_err());
    // Foreign-device operands fail before any dispatch.
    let foreign = VulkanDevice::new()?;
    let foreign_state = GpuBuffer::zeros_f32(&foreign, 8)?;
    let foreign_operands = DeepseekV41CompressorDecodeOperands {
        state_kv: Some(&foreign_state),
        ..ratio_two
    };
    assert!(adjoints
        .decode(&step, &foreign_operands, &upstream, None)
        .is_err());
    // A completing step must supply one latent upstream per batch row.
    let completing = DeepseekV41CompressorDecode {
        start_pos: 3,
        ..step
    };
    let short = zeros(1)?;
    assert!(adjoints
        .decode(&completing, &ratio_two, &short, None)
        .is_err());
    Ok(())
}

#[test]
fn deepseek_v41_compressor_pool_simd_and_scalar_tails_match_cpu() -> Result<()> {
    #[derive(Deserialize)]
    struct PoolCase {
        name: String,
        batch: usize,
        seq: usize,
        dim: usize,
        kv_bits: Vec<u32>,
        score_bits: Vec<u32>,
        upstream_bits: Vec<u32>,
        pooled_bits: Vec<u32>,
        grad_kv_bits: Vec<u32>,
        grad_score_bits: Vec<u32>,
    }
    #[derive(Deserialize)]
    struct PoolFixture {
        tolerance: f64,
        rtol: f64,
        comparison: String,
        cases: Vec<PoolCase>,
    }
    let f: PoolFixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_compressor_pool_math.json"
    ))?;
    assert_eq!(f.tolerance, 2e-7);
    assert_eq!(f.rtol, 0.0);
    assert_eq!(f.comparison, "strict_less_than");
    let device = VulkanDevice::new()?;
    let adjoints = DeepseekV41CompressorAdjoints::new(device.clone())?;
    use vulkan::BindingAccess::{ReadOnly as R, ReadWrite as RW, WriteOnly as W};
    let pool = vulkan::ComputeKernel::new_with_access(
        &device,
        include_bytes!("../../shaders/transformer_deepseek_v41_compress.spv"),
        &[R, R, RW, RW, W],
        32,
    )?;
    let mut failures = Vec::new();
    let mut compared = 0;
    for c in f.cases {
        let kv = gpu(&device, &c.kv_bits)?;
        let scores = gpu(&device, &c.score_bits)?;
        let upstream = gpu(&device, &c.upstream_bits)?;
        let groups = c.seq / 2;
        let state_kv = GpuBuffer::zeros_f32(&device, c.batch * 2 * c.dim)?;
        let state_score = GpuBuffer::zeros_f32(&device, c.batch * 2 * c.dim)?;
        let pooled = GpuBuffer::zeros_f32(&device, c.batch * groups * c.dim)?;
        let gkv = GpuBuffer::zeros_f32(&device, c.kv_bits.len())?;
        let gscore = GpuBuffer::zeros_f32(&device, c.score_bits.len())?;
        let empty = GpuBuffer::zeros_f32(&device, 1)?;
        let width = crate::transformer::aten_vector_width() as u32;
        let forward_push = [
            c.batch as u32,
            c.seq as u32,
            c.dim as u32,
            2,
            groups as u32,
            0,
            0,
            width,
        ];
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        pool.record_dispatch(
            &mut commands,
            &[&kv, &scores, &state_kv, &state_score, &pooled],
            bytemuck::cast_slice(&forward_push),
            grid(c.batch * groups * c.dim),
        )?;
        let backward_push = PoolPush {
            batch: c.batch as u32,
            seq: c.seq as u32,
            dim: c.dim as u32,
            ratio: 2,
            groups: groups as u32,
            tail: 0,
            written_slot: 0,
            has_slot_grad: 0,
            vector_width: width,
        };
        adjoints.pool_grad.record_dispatch(
            &mut commands,
            &[&upstream, &kv, &scores, &empty, &empty, &gkv, &gscore],
            bytemuck::bytes_of(&backward_push),
            grid(c.batch * groups * c.dim),
        )?;
        commands.submit()?;
        for (label, actual, expected) in [
            (
                "pooled",
                pooled.read_f32(c.pooled_bits.len())?,
                &c.pooled_bits,
            ),
            (
                "grad_kv",
                gkv.read_f32(c.grad_kv_bits.len())?,
                &c.grad_kv_bits,
            ),
            (
                "grad_score",
                gscore.read_f32(c.grad_score_bits.len())?,
                &c.grad_score_bits,
            ),
        ] {
            gate(
                &format!("SIMD pool {} {label}", c.name),
                &actual,
                expected,
                &mut failures,
                &mut compared,
            );
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    eprintln!("V4.1 compressor SIMD/tail pool: {compared} values");
    Ok(())
}
