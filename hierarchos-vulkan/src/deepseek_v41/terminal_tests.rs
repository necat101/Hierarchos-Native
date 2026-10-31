//! Strict gate for the terminal training adjoints.
//!
//! The fixture is CPU PyTorch's own autograd on the same norm, head, logits and
//! targets, pinned as FP32 bit patterns by
//! `validation/validate_deepseek_v41_terminal_grad.py`. Nothing here is
//! recomputed from the native kernels, so the comparison is independent.

use anyhow::Result;

use super::DeepseekV41TerminalAdjoints;
use crate::{GpuBuffer, VulkanDevice};

#[derive(serde::Deserialize)]
struct Fixture {
    seed: u64,
    rows: usize,
    hidden: usize,
    vocab: usize,
    targets: Vec<u32>,
    norm_eps: f32,
    tolerance: f32,
    comparison: String,
    rtol: f32,
    x_bits: Vec<u32>,
    norm_weight_bits: Vec<u32>,
    head_bits: Vec<u32>,
    hidden_bits: Vec<u32>,
    logits_bits: Vec<u32>,
    row_loss_bits: Vec<u32>,
    loss_bits: Vec<u32>,
    grad_logits_bits: Vec<u32>,
    grad_hidden_bits: Vec<u32>,
    grad_x_bits: Vec<u32>,
}

fn values(bits: &[u32]) -> Vec<f32> {
    bits.iter().copied().map(f32::from_bits).collect()
}

#[test]
fn reduced_cross_entropy_and_row_mean_keep_their_distinct_cpu_folds() -> Result<()> {
    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        rows: usize,
        vocab: usize,
        targets: Vec<u32>,
        logits_bits: Vec<u32>,
        row_loss_bits: Vec<u32>,
        row_mean_bits: Vec<u32>,
        reduced_mean_bits: Vec<u32>,
    }
    #[derive(serde::Deserialize)]
    struct Contracts {
        tolerance: f32,
        comparison: String,
        rtol: f32,
        cases: Vec<Case>,
    }
    let f: Contracts = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_loss_reductions.json"
    ))?;
    assert_eq!(f.tolerance, 2e-7);
    assert_eq!(f.rtol, 0.0);
    assert_eq!(f.comparison, "strict_less_than");
    let device = VulkanDevice::new()?;
    let adjoints = DeepseekV41TerminalAdjoints::new(device.clone())?;
    for c in f.cases {
        let logits = GpuBuffer::from_u32(&device, &c.logits_bits)?;
        let targets = GpuBuffer::from_u32(&device, &c.targets)?;
        let head = GpuBuffer::zeros_f32(&device, c.vocab)?;
        let row = adjoints.loss_and_input_grad(&logits, &targets, &head, c.rows, c.vocab, 1)?;
        let reduced =
            adjoints.cross_entropy_and_input_grad(&logits, &targets, &head, c.rows, c.vocab, 1)?;
        for (label, actual, expected) in [
            ("rows", row.row_loss.clone(), c.row_loss_bits),
            ("row mean", vec![row.loss], c.row_mean_bits),
            ("reduced mean", vec![reduced.loss], c.reduced_mean_bits),
        ] {
            let delta = worst(&actual, &expected).0;
            eprintln!(
                "V4.1 loss reduction {} {label}: {} values, worst {delta:.17e}",
                c.name,
                actual.len()
            );
            assert!(
                delta < f.tolerance,
                "{} {label} exceeded strict gate",
                c.name
            );
        }
        assert_ne!(
            row.loss.to_bits(),
            reduced.loss.to_bits(),
            "fixture must observe the fold difference"
        );
        assert_eq!(
            row.grad_logits.read_f32(c.rows * c.vocab)?,
            reduced.grad_logits.read_f32(c.rows * c.vocab)?
        );
    }
    Ok(())
}

fn worst(actual: &[f32], expected_bits: &[u32]) -> (f32, usize, f32, f32) {
    assert_eq!(actual.len(), expected_bits.len(), "value count");
    let mut worst = (0.0f32, 0usize, 0.0f32, 0.0f32);
    for (index, (&left, &right)) in actual.iter().zip(expected_bits).enumerate() {
        let right = f32::from_bits(right);
        let delta = (left - right).abs();
        if delta > worst.0 {
            worst = (delta, index, left, right);
        }
    }
    worst
}

#[test]
fn terminal_adjoints_match_the_pinned_cpu_fixture() -> Result<()> {
    let Ok(device) = VulkanDevice::new() else {
        return Ok(());
    };
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_terminal_grad.json"
    ))?;
    assert_eq!(fixture.seed, 4141097);
    assert_eq!(fixture.rows, 4);
    assert_eq!(fixture.hidden, 32);
    assert_eq!(fixture.vocab, 64);
    assert_eq!(fixture.targets, vec![3, 17, 41, 0]);
    assert_eq!(fixture.comparison, "strict_less_than");
    assert_eq!(fixture.rtol, 0.0);
    assert_eq!(fixture.x_bits.len(), fixture.rows * fixture.hidden);
    assert_eq!(fixture.norm_weight_bits.len(), fixture.hidden);
    assert_eq!(fixture.head_bits.len(), fixture.vocab * fixture.hidden);
    assert_eq!(fixture.logits_bits.len(), fixture.rows * fixture.vocab);
    assert_eq!(
        fixture.grad_hidden_bits.len(),
        fixture.rows * fixture.hidden
    );
    assert_eq!(fixture.grad_x_bits.len(), fixture.rows * fixture.hidden);

    let adjoints = DeepseekV41TerminalAdjoints::new(device.clone())?;
    let logits = GpuBuffer::from_f32(&device, &values(&fixture.logits_bits))?;
    let head = GpuBuffer::from_f32(&device, &values(&fixture.head_bits))?;
    let targets = GpuBuffer::from_u32(&device, &fixture.targets)?;
    let gradients = adjoints.loss_and_input_grad(
        &logits,
        &targets,
        &head,
        fixture.rows,
        fixture.vocab,
        fixture.hidden,
    )?;

    for (label, actual, expected) in [
        (
            "row_loss",
            gradients.row_loss.as_slice(),
            fixture.row_loss_bits.as_slice(),
        ),
        (
            "grad_logits",
            gradients
                .grad_logits
                .read_f32(fixture.rows * fixture.vocab)?
                .as_slice(),
            fixture.grad_logits_bits.as_slice(),
        ),
        (
            "grad_hidden",
            gradients
                .grad_hidden
                .read_f32(fixture.rows * fixture.hidden)?
                .as_slice(),
            fixture.grad_hidden_bits.as_slice(),
        ),
    ] {
        let (delta, index, left, right) = worst(actual, expected);
        eprintln!("V4.1 terminal {label}: worst |{left} - {right}| = {delta:.17e} at {index}");
        assert!(
            delta < fixture.tolerance,
            "{label} exceeds strict <{:e}: {delta:.17e} at {index} (native {left}, CPU {right})",
            fixture.tolerance
        );
    }

    let expected_loss = f32::from_bits(fixture.loss_bits[0]);
    let loss_delta = (gradients.loss - expected_loss).abs();
    eprintln!(
        "V4.1 terminal loss: native {:.9} CPU {:.9} delta {loss_delta:.17e}",
        gradients.loss, expected_loss
    );
    assert!(
        loss_delta < fixture.tolerance,
        "loss exceeds strict <{:e}: {loss_delta:.17e} (native {}, CPU {expected_loss})",
        fixture.tolerance,
        gradients.loss
    );

    // The mean is the row losses' mean, not a second reduction path.
    let mut total = 0.0f32;
    for value in &gradients.row_loss {
        total += value;
    }
    assert_eq!(gradients.loss, total / fixture.rows as f32);

    // The final norm's adjoint consumes the head adjoint, so the reported error
    // is the end-to-end error of the two-stage terminal chain.
    let x = GpuBuffer::from_f32(&device, &values(&fixture.x_bits))?;
    let norm_weight = GpuBuffer::from_f32(&device, &values(&fixture.norm_weight_bits))?;
    let grad_x = adjoints.norm_input_grad(
        &gradients.grad_hidden,
        &x,
        &norm_weight,
        fixture.rows,
        fixture.hidden,
        fixture.norm_eps,
    )?;
    let (delta, index, left, right) = worst(
        &grad_x.read_f32(fixture.rows * fixture.hidden)?,
        &fixture.grad_x_bits,
    );
    eprintln!("V4.1 terminal grad_x: worst |{left} - {right}| = {delta:.17e} at {index}");
    assert!(
        delta < fixture.tolerance,
        "grad_x exceeds strict <{:e}: {delta:.17e} at {index} (native {left}, CPU {right})",
        fixture.tolerance
    );

    Ok(())
}
