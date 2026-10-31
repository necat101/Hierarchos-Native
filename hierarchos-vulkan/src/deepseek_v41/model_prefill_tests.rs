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
struct Step {
    name: String,
    pos: usize,
    seq: usize,
    ids: Vec<u32>,
    shape: Vec<usize>,
    logits: Vec<f32>,
}

#[derive(Deserialize)]
struct PrefillCase {
    seed: u64,
    revision: String,
    source_sha256: String,
    stream_precision: String,
    batch_size: usize,
    max_positions: usize,
    checkpoint: Checkpoint,
    prefill_steps: Vec<Step>,
}

fn hash_file(path: &Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

/// Production-width multi-token prefill through the normal package loader.
///
/// The CPU bmm behind the grouped output projection uses a different K-block
/// body for one row (512-wide partials) than for several rows (contiguous
/// 1024-wide blocks, ascending partial fold). This case pins the multi-row body:
/// four tokens in one prefill at released width, then a decode that reads the
/// ring those four tokens seeded.
///
/// Opt-in because it loads the 241 MiB package:
///
///     cargo test --lib deepseek_v41_four_token_prefill -- --ignored --nocapture
#[test]
#[ignore = "production-width four-token prefill (~241 MB package); run explicitly"]
fn deepseek_v41_four_token_prefill_strict_all_logits() -> Result<()> {
    let source = include_str!("../../validation/fixtures/deepseek_v41_lora_model.json");
    let case: PrefillCase = serde_json::from_str(source)?;
    assert_eq!(case.seed, 4141095);
    assert_eq!(case.revision, REFERENCE_REVISION);
    assert_eq!(
        case.source_sha256,
        "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
    );
    assert_eq!(case.stream_precision, "bf16");
    assert_eq!(case.batch_size, 1);
    assert!(case.max_positions >= 4);
    assert_eq!(case.prefill_steps.len(), 2);
    assert_eq!(case.prefill_steps[0].seq, 4);
    assert_eq!(case.prefill_steps[0].shape, vec![1, 4, 512]);
    assert_eq!(case.prefill_steps[1].pos, 4);
    assert_eq!(case.checkpoint.tensor_count, 29);
    assert_eq!(case.checkpoint.packed_count, 13);

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("validation/fixtures")
        .join(&case.checkpoint.directory);
    for (name, expected) in &case.checkpoint.files {
        assert_eq!(std::path::Path::new(name).components().count(), 1);
        assert_eq!(hash_file(&root.join(name))?, *expected, "{name}");
    }

    let device = VulkanDevice::new()?;
    eprintln!("V4.1 prefill device: {}", device.name());
    let mut model = DeepseekV41CausalLM::from_hf_package(
        device.clone(),
        &root,
        case.batch_size,
        case.max_positions,
    )?;

    let mut failures = vec![];
    let mut exported = vec![];
    let mut worst = 0.0f64;
    for step in &case.prefill_steps {
        let logits = if step.pos == 0 {
            model.prefill(&step.ids, step.seq, None)?
        } else {
            model.decode(&step.ids, step.pos, None)?
        };
        let shape = vec![logits.batch_size, logits.seq_len, logits.vocab_size];
        assert_eq!(shape, step.shape, "{}", step.name);
        let values = logits.read_all()?;
        assert_eq!(values.len(), step.logits.len());
        let vocab = step.shape[2];
        let mut per_row = vec![0.0f64; step.seq];
        for (index, (&actual, &expected)) in values.iter().zip(&step.logits).enumerate() {
            assert!(actual.is_finite() && expected.is_finite());
            let drift = (actual as f64 - expected as f64).abs();
            per_row[index / vocab] = per_row[index / vocab].max(drift);
        }
        let step_worst = per_row.iter().copied().fold(0.0f64, f64::max);
        eprintln!(
            "V4.1 prefill {}: {} logits, drift {step_worst:.17e} per row {per_row:?}",
            step.name,
            values.len()
        );
        if step_worst >= 2e-7 {
            failures.push(format!("{}: {step_worst:.17e}", step.name));
        }
        worst = worst.max(step_worst);
        exported.push(serde_json::json!({
            "name": step.name, "shape": shape, "logits": values,
        }));
    }
    assert_eq!(model.next_position(), Some(5));

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/deepseek-v41-reference/prefill-native-logits.json");
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(
        path,
        serde_json::to_vec(&serde_json::json!({
            "fixture_sha256": format!("{:x}", Sha256::digest(source.as_bytes())),
            "prefill_steps": exported,
        }))?,
    )?;
    assert!(
        failures.is_empty(),
        "strict <2e-7, rtol=0 failures: {failures:?}"
    );
    eprintln!("V4.1 four-token prefill worst drift {worst:.17e}");
    Ok(())
}
