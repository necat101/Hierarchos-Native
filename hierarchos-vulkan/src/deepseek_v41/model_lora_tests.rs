use super::*;
use crate::deepseek_v41::REFERENCE_REVISION;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct Checkpoint {
    directory: String,
    files: BTreeMap<String, String>,
    tensor_count: usize,
    packed_count: usize,
}

#[derive(Deserialize)]
struct Adapter {
    directory: String,
    files: BTreeMap<String, String>,
    rank: usize,
    alpha: f64,
    targets: Vec<String>,
    worst_logit_delta: f64,
}

#[derive(Deserialize)]
struct Step {
    name: String,
    pos: usize,
    seq: usize,
    ids: Vec<u32>,
    shape: Vec<usize>,
    logits: Vec<f32>,
}

#[derive(Deserialize)]
struct LoraCase {
    seed: u64,
    revision: String,
    source_sha256: String,
    stream_precision: String,
    batch_size: usize,
    max_prefill: usize,
    max_positions: usize,
    checkpoint: Checkpoint,
    adapter: Adapter,
    steps: Vec<Step>,
    base_steps: Vec<Step>,
}

fn hash_file(path: &Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn run_steps(model: &mut DeepseekV41CausalLM, steps: &[Step]) -> Result<Vec<Vec<f32>>> {
    let mut output = Vec::new();
    for step in steps {
        let logits = if step.pos == 0 {
            model.prefill(&step.ids, step.seq, None)?
        } else {
            model.decode(&step.ids, step.pos, None)?
        };
        assert_eq!(
            vec![logits.batch_size, logits.seq_len, logits.vocab_size],
            step.shape,
            "{}",
            step.name
        );
        let values = logits.read_all()?;
        assert_eq!(values.len(), step.logits.len());
        output.push(values);
    }
    Ok(output)
}

fn compare(expected: &[Step], actual: &[Vec<f32>]) -> Result<f64> {
    let mut worst = 0.0f64;
    for (step, values) in expected.iter().zip(actual) {
        let mut step_worst = 0.0f64;
        for (&actual, &expected) in values.iter().zip(&step.logits) {
            assert!(actual.is_finite() && expected.is_finite());
            step_worst = step_worst.max((actual as f64 - expected as f64).abs());
        }
        eprintln!(
            "V4.1 PEFT {}: {} logits, drift {step_worst:.17e}",
            step.name,
            values.len()
        );
        if step_worst >= 2e-7 {
            eprintln!(
                "V4.1 PEFT {} exceeds the strict gate: {step_worst:.17e}",
                step.name
            );
        }
        worst = worst.max(step_worst);
    }
    Ok(worst)
}

/// Production-width additive-PEFT regression through the shipping loader.
///
/// The 247 MB package is one released-width block (5120 hidden, 64x512 heads,
/// Q/O ranks 1280/1024, eight groups, 2304-wide FP4 experts, HC 4) with an
/// adapter on packed FP8 attention, the packed shared expert and one routed FP4
/// expert. Vocabulary is reduced to 512 because the adapter never touches it
/// and the 24-layer production case gates that path. Both the base graph and
/// the adapted graph are compared against their frozen CPU PyTorch logits, so a
/// silently ignored adapter fails instead of matching the base.
///
/// Opt-in because it loads the 247 MB package twice:
///
///     cargo test --lib deepseek_v41_lora_production_width -- --ignored --nocapture
#[test]
#[ignore = "production-width PEFT package (~247 MB, two model loads); run explicitly"]
fn deepseek_v41_lora_production_width_strict_all_logits() -> Result<()> {
    let source = include_str!("../../validation/fixtures/deepseek_v41_lora_model.json");
    let case: LoraCase = serde_json::from_str(source)?;
    assert_eq!(case.seed, 4141095);
    assert_eq!(case.revision, REFERENCE_REVISION);
    assert_eq!(
        case.source_sha256,
        "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
    );
    assert_eq!(case.stream_precision, "bf16");
    assert_eq!(case.batch_size, 1);
    assert_eq!(case.max_prefill, 1);
    assert_eq!(case.adapter.rank, 16);
    assert_eq!(case.adapter.alpha, 16.0);
    assert_eq!(case.adapter.targets.len(), 6);
    assert!(case.adapter.worst_logit_delta > 1e-4);
    assert_eq!(case.steps.len(), 2);
    assert_eq!(case.base_steps.len(), 2);

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("validation/fixtures")
        .join(&case.checkpoint.directory);
    for (name, expected) in &case.checkpoint.files {
        assert_eq!(std::path::Path::new(name).components().count(), 1);
        assert_eq!(hash_file(&root.join(name))?, *expected, "{name}");
    }
    let adapter_root = root.join(&case.adapter.directory);
    for (name, expected) in &case.adapter.files {
        assert_eq!(hash_file(&adapter_root.join(name))?, *expected, "{name}");
    }
    assert!(case.checkpoint.tensor_count > 10);
    assert!(case.checkpoint.packed_count > 4);

    let device = VulkanDevice::new()?;
    eprintln!("V4.1 PEFT device: {}", device.name());

    let mut base = DeepseekV41CausalLM::from_hf_package(
        device.clone(),
        &root,
        case.batch_size,
        case.max_positions,
    )?;
    assert_eq!(base.lora_module_count(), 0);
    let base_values = run_steps(&mut base, &case.base_steps)?;
    let base_worst = compare(&case.base_steps, &base_values)?;
    fs::write(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../target/deepseek-v41-reference/lora-base-native.json"),
        serde_json::to_vec(&serde_json::json!({
            "steps": case.base_steps.iter().zip(&base_values).map(|(s, v)| serde_json::json!({
                "name": s.name, "shape": s.shape, "logits": v,
            })).collect::<Vec<_>>(),
        }))?,
    )?;
    drop(base);

    let mut adapted = DeepseekV41CausalLM::from_hf_package_with_lora(
        device.clone(),
        &root,
        case.batch_size,
        case.max_positions,
        Some(&adapter_root),
    )?;
    assert_eq!(adapted.lora_module_count(), case.adapter.targets.len());
    let adapted_values = run_steps(&mut adapted, &case.steps)?;
    drop(adapted);
    struct ExportCleanup(PathBuf);
    impl Drop for ExportCleanup {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let adapter_export = ExportCleanup(std::env::temp_dir().join(format!(
        "v41-qualified-adapter-{}-{nonce}",
        std::process::id()
    )));
    DeepseekV41LoraAdapter::load(&adapter_root)?.export(&adapter_export.0)?;
    let mut checkpoint = DeepseekV41Checkpoint::open(&root)?;
    checkpoint.set_lora_adapter(std::sync::Arc::new(DeepseekV41LoraAdapter::load(
        &adapter_export.0,
    )?))?;
    let mut resident = DeepseekV41CausalLM::from_checkpoint_with_vocabulary_residency(
        device,
        &mut checkpoint,
        DeepseekV41AttentionRuntimeConfig {
            batch_size: case.batch_size,
            max_prefill: case.max_prefill,
            max_positions: case.max_positions,
        },
        None,
        DeepseekV41WeightOutput::Bf16,
        super::super::DeepseekV41ExpertResidency::Resident,
        super::super::DeepseekV41EngramResidency::Resident,
        super::super::DeepseekV41ProjectionResidency::Resident,
        DeepseekV41VocabularyResidency::Resident,
    )?;
    let resident_values = run_steps(&mut resident, &case.steps)?;
    assert_eq!(
        resident_values, adapted_values,
        "adapter export or expert/projection/vocabulary offloading changed logits"
    );
    drop(resident);
    let worst = compare(&case.steps, &adapted_values)?;
    assert!(
        base_worst < 2e-7 && worst < 2e-7,
        "strict <2e-7, rtol=0: base {base_worst:.17e}, adapted {worst:.17e}"
    );

    let delta = case
        .steps
        .iter()
        .zip(&case.base_steps)
        .flat_map(|(adapted, base)| adapted.logits.iter().zip(&base.logits))
        .map(|(&a, &b)| (a - b).abs() as f64)
        .fold(0.0f64, f64::max);
    assert!(
        delta > 1e-4,
        "the frozen adapter must change the logits: {delta:.17e}"
    );

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/deepseek-v41-reference/lora-native-logits.json");
    fs::create_dir_all(path.parent().unwrap())?;
    let exported = |steps: &[Step], values: &[Vec<f32>]| {
        steps
            .iter()
            .zip(values)
            .map(|(step, values)| {
                serde_json::json!({
                    "name": step.name, "shape": step.shape, "logits": values,
                })
            })
            .collect::<Vec<_>>()
    };
    fs::write(
        path,
        serde_json::to_vec(&serde_json::json!({
            "seed": case.seed,
            "revision": case.revision,
            "fixture_sha256": format!("{:x}", Sha256::digest(source.as_bytes())),
            "checkpoint_files": case.checkpoint.files,
            "adapter_files": case.adapter.files,
            "lora_modules": case.adapter.targets.len(),
            "resident_streamed_export_bit_identical": true,
            "steps": exported(&case.steps, &adapted_values),
            "base_steps": exported(&case.base_steps, &base_values),
            "worst_drift": worst,
            "base_worst_drift": base_worst,
        }))?,
    )?;
    eprintln!(
        "V4.1 PEFT-LoRA: {} logits, worst drift {worst:.17e}; adapter delta {delta:.6}",
        adapted_values.iter().map(Vec::len).sum::<usize>()
    );
    Ok(())
}
