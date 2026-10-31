use super::*;
use crate::deepseek_v41::{
    DeepseekV41Compressor, DeepseekV41CompressorWeights, REFERENCE_REVISION,
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
    seq: usize,
    state_seed: bool,
    kv_weight_bits: Vec<u32>,
    gate_weight_bits: Vec<u32>,
    norm_weight_bits: Vec<u32>,
    input_bits: Vec<u32>,
    projected_kv_bits: Vec<u32>,
    projected_gate_bits: Vec<u32>,
    pooled_bits: Vec<u32>,
    upstream_bits: Vec<u32>,
    slot_kv_bits: Vec<u32>,
    slot_score_bits: Vec<u32>,
    latent_bits: Vec<u32>,
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
        "../../validation/fixtures/deepseek_v41_compressor_grad.json"
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
        "V4.1 compressor adjoint {label}: {} values, worst {worst:.17e}",
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

fn prefill(case: &Case, start_pos: usize) -> DeepseekV41CompressorPrefill {
    DeepseekV41CompressorPrefill {
        batch: case.batch,
        seq: case.seq,
        start_pos,
        hidden: case.hidden,
        dim: case.head,
        ratio: case.ratio,
        eps: 1e-20,
    }
}

/// Every stage of the prefill adjoint against frozen CPU PyTorch autograd:
/// the pooling/softmax VJP, the frozen projections' input gradients, the pinned
/// RMSNorm VJP, and the pool-state routing.
#[test]
fn deepseek_v41_compressor_prefill_adjoint_matches_cpu_pytorch() -> Result<()> {
    let device = VulkanDevice::new()?;
    let adjoints = DeepseekV41CompressorAdjoints::new(device.clone())?;
    let mut failures = Vec::new();
    let mut compared = 0usize;
    let mut worst = 0.0f64;

    for case in fixture().cases {
        let pooled = if case.pooled_bits.is_empty() {
            GpuBuffer::zeros_f32(&device, case.head)?
        } else {
            gpu(&device, &case.pooled_bits)?
        };
        let upstream = if case.upstream_bits.is_empty() {
            GpuBuffer::zeros_f32(&device, 1)?
        } else {
            gpu(&device, &case.upstream_bits)?
        };
        let projected_kv = gpu(&device, &case.projected_kv_bits)?;
        let projected_gate = (case.ratio == 2)
            .then(|| gpu(&device, &case.projected_gate_bits))
            .transpose()?;
        let kv_weight = gpu(&device, &case.kv_weight_bits)?;
        let gate_weight = (case.ratio == 2)
            .then(|| gpu(&device, &case.gate_weight_bits))
            .transpose()?;
        let norm_weight = gpu(&device, &case.norm_weight_bits)?;
        let slot_kv = case
            .state_seed
            .then(|| gpu(&device, &case.slot_kv_bits))
            .transpose()?;
        let slot_score = case
            .state_seed
            .then(|| gpu(&device, &case.slot_score_bits))
            .transpose()?;
        let operands = DeepseekV41CompressorOperands {
            pooled: &pooled,
            projected_kv: &projected_kv,
            projected_gate: projected_gate.as_ref(),
            kv_weight: &kv_weight,
            gate_weight: gate_weight.as_ref(),
            norm_weight: &norm_weight,
        };
        let gradients = adjoints.prefill(
            &prefill(&case, 0),
            &operands,
            &upstream,
            slot_kv.as_ref().zip(slot_score.as_ref()),
        )?;
        let grad_input = gradients
            .grad_input
            .read_f32(case.batch * case.seq * case.hidden)?;
        worst = worst.max(gate(
            &format!("{} grad_input", case.name),
            &grad_input,
            &case.expected.grad_input_bits,
            &mut failures,
            &mut compared,
        ));
        match (&gradients.grad_state_kv, &gradients.grad_state_score) {
            (None, None) => {
                assert!(case.ratio == 1, "{} returned no pool state", case.name);
                assert!(case.expected.grad_state_kv_bits.is_empty());
                assert!(case.expected.grad_state_score_bits.is_empty());
            }
            (Some(state_kv), Some(state_score)) => {
                let len = case.batch * case.ratio * case.head;
                worst = worst.max(gate(
                    &format!("{} grad_state_kv", case.name),
                    &state_kv.read_f32(len)?,
                    &case.expected.grad_state_kv_bits,
                    &mut failures,
                    &mut compared,
                ));
                worst = worst.max(gate(
                    &format!("{} grad_state_score", case.name),
                    &state_score.read_f32(len)?,
                    &case.expected.grad_state_score_bits,
                    &mut failures,
                    &mut compared,
                ));
            }
            _ => failures.push(format!(
                "{}: pool-state gradient halves disagree",
                case.name
            )),
        }
    }

    assert!(failures.is_empty(), "{failures:#?}");
    eprintln!(
        "V4.1 compressor adjoint on {}: {compared} frozen gradient values, worst absolute drift {worst:.17e}",
        device.name()
    );
    Ok(())
}

/// The same adjoint driven from the native compressor's own retained operands,
/// which is what a trainer would hand it. This is the plumbing check: the
/// projection, pooled and gating buffers the forward keeps are the adjoint's
/// operands, and the emitted latents come from the same evaluation.
#[test]
fn deepseek_v41_compressor_prefill_adjoint_runs_on_native_operands() -> Result<()> {
    let device = VulkanDevice::new()?;
    let adjoints = DeepseekV41CompressorAdjoints::new(device.clone())?;
    let mut failures = Vec::new();
    let mut compared = 0usize;
    let mut worst = 0.0f64;
    let mut operand_worst = 0.0f64;

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
        let mut module = DeepseekV41Compressor::new(device.clone(), weights, case.batch, case.seq)?;
        let input = gpu(&device, &case.input_bits)?;
        let latents = module.prefill_from_buffer(&input, case.batch, case.seq)?;
        let groups = case.seq / case.ratio;
        assert_eq!(latents.groups_per_batch, groups, "{}", case.name);

        let mut operand_drift = |label: &str, buffer: &GpuBuffer, bits: &[u32]| -> Result<()> {
            if bits.is_empty() {
                return Ok(());
            }
            let actual = buffer.read_f32(bits.len())?;
            let expected = values(bits);
            let drift = actual
                .iter()
                .zip(&expected)
                .map(|(&a, &b)| (a as f64 - b as f64).abs())
                .fold(0.0f64, f64::max);
            assert!(
                drift < 2e-7,
                "{} {label} operand drift {drift:.10e}",
                case.name
            );
            operand_worst = operand_worst.max(drift);
            Ok(())
        };
        operand_drift(
            "projected_kv",
            &latents.projected_kv,
            &case.projected_kv_bits,
        )?;
        operand_drift("pre_norm", &latents.pre_norm, &case.pooled_bits)?;
        if case.ratio == 2 {
            operand_drift(
                "projected_gate",
                &latents.projected_gate,
                &case.projected_gate_bits,
            )?;
        }
        let emitted = latents.read_f32()?;
        let latent = values(&case.latent_bits);
        let latent_drift = emitted
            .iter()
            .zip(&latent)
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .fold(0.0f64, f64::max);
        assert!(
            latent_drift < 2e-7,
            "{} latents drifted {latent_drift:.10e}",
            case.name
        );

        // Feed the adjoint exactly what the forward retained, including the
        // pool state the fixture seeds. An evaluation shorter than one group
        // emits nothing, so its seed is an unused placeholder.
        let upstream = if case.upstream_bits.is_empty() {
            GpuBuffer::zeros_f32(&device, 1)?
        } else {
            gpu(&device, &case.upstream_bits)?
        };
        let slot_kv = case
            .state_seed
            .then(|| gpu(&device, &case.slot_kv_bits))
            .transpose()?;
        let slot_score = case
            .state_seed
            .then(|| gpu(&device, &case.slot_score_bits))
            .transpose()?;
        let kv_weight = gpu(&device, &case.kv_weight_bits)?;
        let gate_weight = (case.ratio == 2)
            .then(|| gpu(&device, &case.gate_weight_bits))
            .transpose()?;
        let norm_weight = gpu(&device, &case.norm_weight_bits)?;
        let projected_gate = (case.ratio == 2)
            .then(|| gpu(&device, &case.projected_gate_bits))
            .transpose()?;
        let operands = DeepseekV41CompressorOperands {
            pooled: &latents.pre_norm,
            projected_kv: &latents.projected_kv,
            projected_gate: projected_gate.as_ref(),
            kv_weight: &kv_weight,
            gate_weight: gate_weight.as_ref(),
            norm_weight: &norm_weight,
        };
        let gradients = adjoints.prefill(
            &prefill(&case, 0),
            &operands,
            &upstream,
            slot_kv.as_ref().zip(slot_score.as_ref()),
        )?;
        let grad_input = gradients
            .grad_input
            .read_f32(case.batch * case.seq * case.hidden)?;
        worst = worst.max(gate(
            &format!("{} native-forward grad_input", case.name),
            &grad_input,
            &case.expected.grad_input_bits,
            &mut failures,
            &mut compared,
        ));
    }

    assert!(failures.is_empty(), "{failures:#?}");
    eprintln!(
        "V4.1 compressor adjoint through the native forward: {compared} values, worst absolute drift {worst:.17e}, operand drift {operand_worst:.17e}"
    );
    Ok(())
}

#[test]
fn deepseek_v41_compressor_adjoint_rejects_unqualified_calls() -> Result<()> {
    let device = VulkanDevice::new()?;
    let adjoints = DeepseekV41CompressorAdjoints::new(device.clone())?;
    let zeros = |len: usize| GpuBuffer::zeros_f32(&device, len);
    let weights = zeros(64)?;
    let norm = zeros(8)?;
    let pooled = zeros(8)?;
    let upstream = zeros(8)?;
    let state = zeros(8)?;

    // A decode evaluation pools from slots earlier steps wrote.
    let operands = DeepseekV41CompressorOperands {
        pooled: &pooled,
        projected_kv: &weights,
        projected_gate: None,
        kv_weight: &weights,
        gate_weight: None,
        norm_weight: &norm,
    };
    let err = adjoints
        .prefill(
            &DeepseekV41CompressorPrefill {
                batch: 1,
                seq: 4,
                start_pos: 3,
                hidden: 8,
                dim: 8,
                ratio: 1,
                eps: 1e-20,
            },
            &operands,
            &upstream,
            None,
        )
        .err()
        .expect("decode must be rejected")
        .to_string();
    assert!(err.contains("prefill"), "{err}");

    let ratio_one = DeepseekV41CompressorPrefill {
        batch: 1,
        seq: 1,
        start_pos: 0,
        hidden: 8,
        dim: 8,
        ratio: 1,
        eps: 1e-20,
    };
    // A ratio-1 evaluation has no gate projection and no pool state.
    let gated = DeepseekV41CompressorOperands {
        pooled: &pooled,
        projected_kv: &weights,
        projected_gate: Some(&weights),
        kv_weight: &weights,
        gate_weight: Some(&weights),
        norm_weight: &norm,
    };
    assert!(adjoints
        .prefill(&ratio_one, &gated, &upstream, None)
        .is_err());
    assert!(adjoints
        .prefill(&ratio_one, &operands, &upstream, Some((&state, &state)))
        .is_err());
    // A ratio-2 evaluation needs both gate buffers.
    let ratio_two = DeepseekV41CompressorPrefill {
        ratio: 2,
        ..ratio_one
    };
    assert!(adjoints
        .prefill(&ratio_two, &operands, &upstream, None)
        .is_err());
    // An operand from another device is refused before any dispatch.
    let foreign = VulkanDevice::new()?;
    let foreign_kv = GpuBuffer::zeros_f32(&foreign, 64)?;
    let foreign_operands = DeepseekV41CompressorOperands {
        projected_kv: &foreign_kv,
        ..operands
    };
    assert!(adjoints
        .prefill(&ratio_one, &foreign_operands, &upstream, None)
        .is_err());
    // The upstream seed must cover every emitted group.
    let short = zeros(1)?;
    let two_groups = DeepseekV41CompressorPrefill {
        batch: 1,
        seq: 2,
        ..ratio_one
    };
    assert!(adjoints
        .prefill(&two_groups, &operands, &short, None)
        .is_err());
    Ok(())
}
