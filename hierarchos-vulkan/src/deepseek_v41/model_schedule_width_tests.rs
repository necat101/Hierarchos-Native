//! Production arithmetic widths with the sparse/Engram/top-6 schedule enabled.
//! This adds a case without changing the frozen 24-layer accumulation gate.
use super::*;
use crate::deepseek_v41::REFERENCE_REVISION;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::Read, path::PathBuf};

#[derive(Deserialize)]
struct Case {
    seed: u64,
    revision: String,
    source_sha256: String,
    engram_sha256: String,
    stream_precision: String,
    config: serde_json::Value,
    checkpoint: Package,
    steps: Vec<Step>,
}
#[derive(Deserialize)]
struct Package {
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
    mask: Vec<u32>,
    shape: Vec<usize>,
    logits: Vec<f32>,
}

fn frozen_case() -> Result<(Case, String)> {
    let source = fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("validation/fixtures/deepseek_v41_schedule_width_model.json"),
    )?;
    let binding = format!("{:x}", Sha256::digest(&source));
    assert_eq!(
        binding,
        "70d2218429c2d4b8623a8ebbdd6c659eec22f7f64f9eeecd3d95c5d7877fdcf0"
    );
    Ok((serde_json::from_slice(&source)?, binding))
}

/// Keep the oracle contract checked without requiring the 4.26 GB package.
#[test]
fn deepseek_v41_schedule_width_frozen_fixture_contract() -> Result<()> {
    let (case, _) = frozen_case()?;
    assert_eq!(case.seed, 4141211);
    assert_eq!(case.checkpoint.tensor_count, 75);
    assert_eq!(case.checkpoint.packed_count, 107);
    assert_eq!(case.checkpoint.files.len(), 11);
    let text = &case.config["text_config"];
    for (field, expected) in [
        ("hidden_size", 5120),
        ("head_dim", 512),
        ("num_attention_heads", 64),
        ("vocab_size", 129280),
        ("num_hidden_layers", 4),
        ("n_routed_experts", 6),
        ("num_experts_per_tok", 6),
        ("engram_n_heads", 8),
        ("engram_head_dim", 256),
        ("engram_max_ngram_size", 4),
    ] {
        assert_eq!(text[field], expected, "{field}");
    }
    assert_eq!(text["compress_ratios"], serde_json::json!([2, 2, 1, 1]));
    assert_eq!(text["engram_layer_ids"], serde_json::json!([0, 3]));
    assert_eq!(text["rope_scaling"]["factor"], 40);
    assert_eq!(
        case.steps
            .iter()
            .map(|s| (s.pos, s.seq))
            .collect::<Vec<_>>(),
        [(0, 3), (3, 1), (4, 1), (0, 1), (1, 1)]
    );
    assert_eq!(case.steps[0].mask, [1, 0, 1]);
    for step in &case.steps {
        assert_eq!(step.shape, [1, step.seq, 129280]);
        assert_eq!(step.ids.len(), step.seq);
        assert_eq!(step.mask.len(), step.seq);
        assert_eq!(step.logits.len(), step.seq * 129280);
        assert!(step.logits.iter().all(|x| x.is_finite()));
        assert!(step.logits.iter().any(|&x| x != 0.0));
    }
    assert_eq!(
        case.steps.iter().map(|s| s.logits.len()).sum::<usize>(),
        904960
    );
    Ok(())
}

#[test]
#[ignore = "production-width sparse/Engram/top-6 package; run with --ignored"]
fn deepseek_v41_default_package_schedule_width_strict_all_logits() -> Result<()> {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("validation/fixtures");
    let (case, fixture_sha256) = frozen_case()?;
    assert_eq!(case.seed, 4141211);
    assert_eq!(case.revision, REFERENCE_REVISION);
    assert_eq!(
        case.source_sha256,
        "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
    );
    assert_eq!(
        case.engram_sha256,
        "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897"
    );
    assert_eq!(case.stream_precision, "bf16");
    assert_eq!(case.steps.len(), 5);
    assert_eq!(case.checkpoint.packed_count, 107);
    assert_eq!(case.checkpoint.tensor_count, 75);
    let root = fixtures.join(&case.checkpoint.directory);
    for (name, expected) in &case.checkpoint.files {
        assert_eq!(std::path::Path::new(name).components().count(), 1);
        let mut file = fs::File::open(root.join(name))?;
        let mut hash = Sha256::new();
        let mut buffer = vec![0u8; 4 * 1024 * 1024];
        loop {
            let length = file.read(&mut buffer)?;
            if length == 0 {
                break;
            }
            hash.update(&buffer[..length]);
        }
        assert_eq!(format!("{:x}", hash.finalize()), *expected, "{name}");
    }
    let device = VulkanDevice::new()?;
    eprintln!("V4.1 schedule-width device: {}", device.name());
    let mut model = DeepseekV41CausalLM::from_hf_package(device, &root, 1, 16)?;
    let text = &model.config().text;
    assert_eq!(
        (text.hidden_size, text.head_dim, text.num_attention_heads),
        (5120, 512, 64)
    );
    assert_eq!(
        (text.q_lora_rank, text.o_lora_rank, text.o_groups),
        (1280, 1024, 8)
    );
    assert_eq!(
        (text.moe_intermediate_size, text.vocab_size, text.hc_mult),
        (2304, 129280, 4)
    );
    assert_eq!((text.index_n_heads, text.index_head_dim), (32, 128));
    assert_eq!(
        (
            text.engram_n_heads,
            text.engram_head_dim,
            text.engram_max_ngram_size
        ),
        (8, 256, 4)
    );
    assert_eq!((text.n_routed_experts, text.num_experts_per_tok), (6, 6));
    assert_eq!(text.compress_ratios, [2, 2, 1, 1]);
    assert_eq!(text.kv_source_layer_ids, [0, 2]);
    assert_eq!(text.index_source_layer_ids, [0, 2, 3]);
    assert_eq!(text.engram_layer_ids, [0, 3]);
    assert_eq!(text.candidate_source_layer_id, 2);
    assert_eq!(model.stream_precision(), DeepseekV41WeightOutput::Bf16);
    let trace = std::env::var_os("HIERARCHOS_V41_SCHEDULE_WIDTH_TRACE").is_some();
    if trace {
        model.backbone.trace = Some(vec![]);
    }
    let mut total = 0usize;
    let mut worst = 0.0f64;
    let mut exported = Vec::new();
    for step in &case.steps {
        if trace {
            model.backbone.trace.as_mut().unwrap().clear();
        }
        let output = if step.pos == 0 {
            model.prefill(&step.ids, step.seq, Some(&step.mask))?
        } else {
            model.decode(&step.ids, step.pos, Some(&step.mask))?
        };
        assert_eq!(
            [output.batch_size, output.seq_len, output.vocab_size].as_slice(),
            step.shape
        );
        let values = output.read_all()?;
        assert!(!values.is_empty());
        assert_eq!(values.len(), step.logits.len());
        let mut drift = 0.0f64;
        for (&actual, &expected) in values.iter().zip(&step.logits) {
            assert!(actual.is_finite() && expected.is_finite());
            drift = drift.max((actual as f64 - expected as f64).abs());
        }
        eprintln!(
            "V4.1 schedule width {}: {} full logits, drift {drift:.17e}",
            step.name,
            values.len()
        );
        total += values.len();
        worst = worst.max(drift);
        assert_eq!(model.next_position(), Some(step.pos + step.seq));
        assert_eq!(model.resident_routed_expert_count(), 0);
        assert_eq!(model.resident_shared_expert_count(), 0);
        assert_eq!(model.resident_engram_table_count(), 0);
        assert_eq!(model.resident_projection_layer_count(), 0);
        assert_eq!(model.resident_vocabulary_tensor_count(), 0);
        let mut entry = serde_json::json!({"name":step.name,"shape":step.shape,"logits":values});
        if trace {
            let stages = model.backbone.trace.as_ref().unwrap().iter().map(|(layer,stage,buffer,len)| {
                Ok(serde_json::json!({"layer":layer,"stage":stage,"values":buffer.read_f32(*len)?}))
            }).collect::<Result<Vec<_>>>()?;
            entry["stages"] = serde_json::json!(stages);
        }
        exported.push(entry);
        // A failing traced step is enough to locate the first divergence.
        // The total-logit assertion below still rejects this incomplete run.
        if trace && drift >= 2e-7 {
            break;
        }
    }
    let report = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/deepseek-v41-reference/schedule-width-native-logits.json");
    fs::create_dir_all(report.parent().unwrap())?;
    fs::write(
        report,
        serde_json::to_vec(&serde_json::json!({
            "seed":case.seed,"revision":case.revision,"fixture_sha256":fixture_sha256,
            "checkpoint_files":case.checkpoint.files,"loader":"from_hf_package",
            "full_logits":total,"max_abs_drift":worst,"steps":exported,
        }))?,
    )?;
    assert_eq!(total, 904960);
    assert!(
        worst < 2e-7,
        "V4.1 production-width schedule drift {worst:.17e} violates strict <2e-7, rtol=0"
    );
    Ok(())
}
