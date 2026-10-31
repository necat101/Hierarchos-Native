//! Strict gate for the V4.1 adapter optimizer.
//!
//! `validation/validate_deepseek_v41_adapter_training.py` pins CPU PyTorch's own
//! factors and moments after each AdamW step on fixed gradients. Replaying the
//! same steps here qualifies the update rule, the moment layout, the checkpoint
//! format and the resume contract independently of any native gradient kernel.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::Result;
use serde::Deserialize;

use super::{DeepseekV41AdamWConfig, DeepseekV41AdapterGradient, DeepseekV41AdapterTrainer};
use crate::deepseek_v41::DeepseekV41LoraAdapter;

#[derive(Deserialize)]
struct Module {
    target: String,
    #[allow(dead_code)]
    rank: usize,
    #[allow(dead_code)]
    width: usize,
    #[allow(dead_code)]
    out: usize,
    #[allow(dead_code)]
    alpha: f64,
    a_bits: Vec<u32>,
    b_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct Gradients {
    a_bits: Vec<u32>,
    b_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct Parameters {
    a_bits: Vec<u32>,
    b_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct Moments {
    exp_avg_a_bits: Vec<u32>,
    exp_avg_sq_a_bits: Vec<u32>,
    exp_avg_b_bits: Vec<u32>,
    exp_avg_sq_b_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct Step {
    gradients: BTreeMap<String, Gradients>,
    parameters: BTreeMap<String, Parameters>,
    moments: BTreeMap<String, Moments>,
}

#[derive(Deserialize)]
struct Hyper {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
}

#[derive(Deserialize)]
struct Fixture {
    seed: u64,
    adapter: String,
    tolerance: f32,
    comparison: String,
    rtol: f32,
    config: Hyper,
    modules: Vec<Module>,
    steps: Vec<Step>,
}

fn values(bits: &[u32]) -> Vec<f32> {
    bits.iter().copied().map(f32::from_bits).collect()
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

fn assert_gate(label: &str, actual: &[f32], expected: &[u32], tolerance: f32) {
    let (delta, index, left, right) = worst(actual, expected);
    eprintln!("V4.1 adapter {label}: worst |{left} - {right}| = {delta:.17e} at {index}");
    assert!(
        delta < tolerance,
        "{label} exceeds strict <{tolerance:e}: {delta:.17e} at {index} (native {left}, CPU {right})"
    );
}

fn fixture_directory(fixture: &Fixture) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("validation")
        .join("fixtures")
        .join(&fixture.adapter)
}

fn gradients(step: &Step) -> Vec<DeepseekV41AdapterGradient> {
    step.gradients
        .iter()
        .map(|(target, gradient)| DeepseekV41AdapterGradient {
            target: target.clone(),
            grad_a: values(&gradient.a_bits),
            grad_b: values(&gradient.b_bits),
        })
        .collect()
}

fn compare(trainer: &DeepseekV41AdapterTrainer, step: &Step, tolerance: f32) {
    for (target, expected) in &step.parameters {
        let (a, b) = trainer
            .factors(target)
            .unwrap_or_else(|| panic!("missing factors for {target}"));
        assert_gate(&format!("{target} a"), a, &expected.a_bits, tolerance);
        assert_gate(&format!("{target} b"), b, &expected.b_bits, tolerance);
    }
    for (target, expected) in &step.moments {
        let (exp_avg_a, exp_avg_sq_a, exp_avg_b, exp_avg_sq_b) = trainer
            .moments(target)
            .unwrap_or_else(|| panic!("missing moments for {target}"));
        assert_gate(
            &format!("{target} exp_avg_a"),
            exp_avg_a,
            &expected.exp_avg_a_bits,
            tolerance,
        );
        assert_gate(
            &format!("{target} exp_avg_sq_a"),
            exp_avg_sq_a,
            &expected.exp_avg_sq_a_bits,
            tolerance,
        );
        assert_gate(
            &format!("{target} exp_avg_b"),
            exp_avg_b,
            &expected.exp_avg_b_bits,
            tolerance,
        );
        assert_gate(
            &format!("{target} exp_avg_sq_b"),
            exp_avg_sq_b,
            &expected.exp_avg_sq_b_bits,
            tolerance,
        );
    }
}

#[test]
fn adapter_adamw_matches_the_pinned_cpu_trajectory_and_resumes() -> Result<()> {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_adapter_training.json"
    ))?;
    assert_eq!(fixture.seed, 4141201);
    assert_eq!(fixture.comparison, "strict_less_than");
    assert_eq!(fixture.rtol, 0.0);
    assert_eq!(fixture.steps.len(), 3);

    let directory = fixture_directory(&fixture);
    let adapter = DeepseekV41LoraAdapter::load(&directory)?;
    let config = DeepseekV41AdamWConfig {
        learning_rate: fixture.config.learning_rate,
        beta1: fixture.config.beta1,
        beta2: fixture.config.beta2,
        eps: fixture.config.eps,
        weight_decay: fixture.config.weight_decay,
    };

    // The fixture's initial factors must be exactly what the adapter decoded.
    for module in &fixture.modules {
        let loaded = adapter
            .module_for_tensor(&format!("{}.weight", module.target))
            .unwrap_or_else(|| panic!("adapter is missing {}", module.target));
        assert_gate(
            &format!("{} initial a", module.target),
            &module
                .a_bits
                .iter()
                .copied()
                .map(f32::from_bits)
                .collect::<Vec<_>>(),
            &module.a_bits,
            fixture.tolerance,
        );
        assert_eq!(loaded.a.len(), module.a_bits.len());
        assert_eq!(loaded.b.len(), module.b_bits.len());
    }

    let mut trainer = DeepseekV41AdapterTrainer::new(&adapter, config)?;
    for (index, step) in fixture.steps.iter().enumerate() {
        trainer.step(&gradients(step))?;
        assert_eq!(trainer.step_count(), index as u64 + 1);
        compare(&trainer, step, fixture.tolerance);
    }
    let uninterrupted = trainer.clone();

    // Resume: apply the first two steps, publish the trained factors and the
    // optimizer state, then continue from disk. A resumed run must reproduce the
    // uninterrupted trajectory exactly, not merely approximately.
    let mut partial = DeepseekV41AdapterTrainer::new(&adapter, config)?;
    for step in &fixture.steps[..2] {
        partial.step(&gradients(step))?;
    }
    let root = std::env::temp_dir().join(format!(
        "v41-adapter-resume-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(root.clone());
    let adapter_directory = root.join("adapter");
    partial.adapter()?.export(&adapter_directory)?;
    partial.save(&adapter_directory)?;

    let reloaded = DeepseekV41LoraAdapter::load(&adapter_directory)?;
    let mut resumed = DeepseekV41AdapterTrainer::resume(&reloaded, &adapter_directory)?;
    assert_eq!(resumed.step_count(), 2);
    resumed.step(&gradients(&fixture.steps[2]))?;
    compare(&resumed, &fixture.steps[2], fixture.tolerance);

    for target in uninterrupted.targets() {
        let (left_a, left_b) = uninterrupted
            .factors(target)
            .expect("uninterrupted factors");
        let (right_a, right_b) = resumed.factors(target).expect("resumed factors");
        assert_eq!(left_a, right_a, "{target} resumed a differs bitwise");
        assert_eq!(left_b, right_b, "{target} resumed b differs bitwise");
        let (left, right) = (
            uninterrupted
                .moments(target)
                .expect("uninterrupted moments"),
            resumed.moments(target).expect("resumed moments"),
        );
        assert_eq!(left, right, "{target} resumed moments differ bitwise");
    }

    // Exporting the resumed adapter must round-trip through the normal loader
    // with the trained factors, not the source ones.
    let final_directory = root.join("final-adapter");
    resumed.adapter()?.export(&final_directory)?;
    let exported = DeepseekV41LoraAdapter::load(&final_directory)?;
    for target in resumed.targets() {
        let (a, b) = resumed.factors(target).expect("trained factors");
        let module = exported
            .module_for_tensor(&format!("{target}.weight"))
            .expect("exported module");
        assert_eq!(module.a.as_ref(), a, "{target} exported a differs");
        assert_eq!(module.b.as_ref(), b, "{target} exported b differs");
    }

    // A caller that expects every module can still require a complete cohort.
    let mut strict = DeepseekV41AdapterTrainer::new(&adapter, config)?;
    let mut missing = gradients(&fixture.steps[0]);
    missing.pop();
    assert!(strict.step_complete(&missing).is_err());

    // A bad second module must not advance the first module or its moments.
    for nonfinite in [false, true] {
        let before = strict.clone();
        let mut invalid = gradients(&fixture.steps[0]);
        invalid.sort_by(|a, b| a.target.cmp(&b.target));
        let last = invalid.last_mut().unwrap();
        if nonfinite {
            last.grad_b[0] = f32::NAN;
        } else {
            last.grad_a.pop();
        }
        assert!(strict.step(&invalid).is_err());
        assert_eq!(strict.step_count(), before.step_count());
        for target in strict.targets() {
            assert_eq!(strict.factors(target), before.factors(target));
            assert_eq!(strict.moments(target), before.moments(target));
        }
    }

    Ok(())
}
