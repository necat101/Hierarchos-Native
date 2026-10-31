use super::*;
use crate::deepseek_v41::{
    DeepseekV41EngramResidency, DeepseekV41ExpertResidency, DeepseekV41ProjectionResidency,
    REFERENCE_REVISION,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

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
struct ProductionCase {
    seed: u64,
    revision: String,
    source_sha256: String,
    batch_size: usize,
    max_prefill: usize,
    max_positions: usize,
    stream_precision: String,
    checkpoint: Checkpoint,
    steps: Vec<Step>,
}

/// Opt-in qualification of the released widths and vocabulary. The package is
/// ~7.8 GB, so it is not part of the default suite:
///
///     cargo test --lib deepseek_v41_production -- --ignored --nocapture
///
/// Set `V41_PRODUCTION_TRACE=1` to also export every backbone stage to
/// `target/deepseek-v41-reference/production-native-trace.json`.
///
/// It fixes 5120 hidden, 64x512 attention heads, Q/O ranks 1280/1024, eight
/// output groups, 2304-wide experts, and the real 129280 vocabulary in
/// twenty-four layers with two routed experts and one token per step.
/// Twenty-four layers exercise repeated inter-layer accumulation; depth,
/// expert count, context and the compression/index schedule remain reduced and
/// unqualified.
///
/// `deepseek_v41_default_package_production_width_strict_all_logits` runs the
/// same case through `from_hf_package` and is the release parity gate for the
/// shipping loader.
#[test]
#[ignore = "production-width package (~7.8 GB); run with --ignored"]
fn deepseek_v41_production_width_vocabulary_strict_all_logits() -> Result<()> {
    run_production_case(
        DeepseekV41ExpertResidency::Resident,
        DeepseekV41ProjectionResidency::Resident,
    )
}

#[test]
#[ignore = "production-width streamed package (~4.6 GB); run explicitly"]
fn deepseek_v41_streamed_production_width_strict_all_logits() -> Result<()> {
    run_production_case(
        DeepseekV41ExpertResidency::Streamed,
        DeepseekV41ProjectionResidency::Resident,
    )
}

#[test]
#[ignore = "production-width all-expert streamed package (~4.6 GB); run explicitly"]
fn deepseek_v41_streamed_all_production_width_strict_all_logits() -> Result<()> {
    run_production_case(
        DeepseekV41ExpertResidency::StreamedAllExperts,
        DeepseekV41ProjectionResidency::Resident,
    )
}

#[test]
#[ignore = "production-width streamed block package (~4.6 GB); run explicitly"]
fn deepseek_v41_streamed_block_production_width_strict_all_logits() -> Result<()> {
    run_production_case(
        DeepseekV41ExpertResidency::StreamedAllExperts,
        DeepseekV41ProjectionResidency::Streamed,
    )
}

fn run_production_case(
    residency: DeepseekV41ExpertResidency,
    projections: DeepseekV41ProjectionResidency,
) -> Result<()> {
    run_production_case_with_vocabulary(
        residency,
        projections,
        DeepseekV41VocabularyResidency::Resident,
        false,
    )
}

#[test]
#[ignore = "production-width streamed vocabulary package (~4.6 GB); run explicitly"]
fn deepseek_v41_streamed_vocabulary_production_width_strict_all_logits() -> Result<()> {
    run_production_case_with_vocabulary(
        DeepseekV41ExpertResidency::StreamedAllExperts,
        DeepseekV41ProjectionResidency::Streamed,
        DeepseekV41VocabularyResidency::Streamed,
        false,
    )
}

/// The shipping loader for `model_type=deepseek_v41` is
/// `from_hf_package`; this case runs the same production-width package through
/// that default path instead of naming the residency policies explicitly.
/// It is the release parity gate used by the native CLI.
#[test]
#[ignore = "production-width default package (~4.6 GB); run with --ignored"]
fn deepseek_v41_default_package_production_width_strict_all_logits() -> Result<()> {
    run_production_case_with_vocabulary(
        DeepseekV41ExpertResidency::StreamedAllExperts,
        DeepseekV41ProjectionResidency::Streamed,
        DeepseekV41VocabularyResidency::Streamed,
        true,
    )
}

fn run_production_case_with_vocabulary(
    residency: DeepseekV41ExpertResidency,
    projections: DeepseekV41ProjectionResidency,
    vocabulary: DeepseekV41VocabularyResidency,
    default_loader: bool,
) -> Result<()> {
    let prefix = if default_loader {
        ""
    } else if vocabulary == DeepseekV41VocabularyResidency::Streamed {
        "streamed-vocabulary-"
    } else {
        match (residency, projections) {
            (DeepseekV41ExpertResidency::Resident, _) => "",
            (DeepseekV41ExpertResidency::Streamed, _) => "streamed-",
            (
                DeepseekV41ExpertResidency::StreamedAllExperts,
                DeepseekV41ProjectionResidency::Resident,
            ) => "streamed-all-",
            (
                DeepseekV41ExpertResidency::StreamedAllExperts,
                DeepseekV41ProjectionResidency::Streamed,
            ) => "streamed-block-",
        }
    };
    let source = include_str!("../../validation/fixtures/deepseek_v41_production_model.json");
    let case: ProductionCase = serde_json::from_str(source)?;
    assert_eq!(case.seed, 4141090);
    assert_eq!(case.revision, REFERENCE_REVISION);
    assert_eq!(
        case.source_sha256,
        "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
    );
    assert_eq!(case.stream_precision, "bf16");
    assert_eq!(case.steps.len(), 2);
    assert_eq!(case.checkpoint.tensor_count, 339);
    assert_eq!(case.checkpoint.packed_count, 312);

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("validation/fixtures")
        .join(&case.checkpoint.directory);
    for (name, expected) in &case.checkpoint.files {
        assert_eq!(std::path::Path::new(name).components().count(), 1);
        assert_eq!(
            format!("{:x}", Sha256::digest(fs::read(root.join(name))?)),
            *expected,
            "{name}"
        );
    }
    let device = VulkanDevice::new()?;
    eprintln!("V4.1 production device: {}", device.name());
    let mut checkpoint = DeepseekV41Checkpoint::open(&root)?;
    let text = &checkpoint.config.text;
    assert_eq!(text.vocab_size, 129_280);
    assert_eq!(text.hidden_size, 5_120);
    assert_eq!(text.num_hidden_layers, 24);
    assert_eq!(text.num_attention_heads, 64);
    assert_eq!(text.head_dim, 512);
    assert_eq!(text.q_lora_rank, 1_280);
    assert_eq!(text.o_lora_rank, 1_024);
    assert_eq!(text.o_groups, 8);
    assert_eq!(text.moe_intermediate_size, 2_304);
    assert_eq!(text.n_routed_experts, 2);
    assert_eq!(text.hc_mult, 4);
    assert!(case.checkpoint.tensor_count > 10);
    // Residency accessors count persistent weights over the whole backbone, so a
    // twenty-four-layer graph reports per-layer totals. Copy them out before the
    // checkpoint is borrowed mutably for the explicit-residency constructions.
    let layers = text.num_hidden_layers;
    let routed_experts = text.n_routed_experts;

    let runtime = DeepseekV41AttentionRuntimeConfig {
        batch_size: case.batch_size,
        max_prefill: case.max_prefill,
        max_positions: case.max_positions,
    };
    let memory_before = device.memory_stats()?.live_buffer_bytes;
    let mut model = if default_loader {
        drop(checkpoint);
        DeepseekV41CausalLM::from_hf_package(device.clone(), &root, 1, case.max_positions)?
    } else {
        let model = DeepseekV41CausalLM::from_checkpoint_with_vocabulary_residency(
            device.clone(),
            &mut checkpoint,
            runtime,
            None,
            DeepseekV41WeightOutput::Bf16,
            residency,
            DeepseekV41EngramResidency::Resident,
            projections,
            vocabulary,
        )?;
        drop(checkpoint);
        model
    };
    // `Resident` is honored only when both 1.32 GB tables fit beside the loaded
    // backbone; otherwise the loader degrades to bounded staging before the
    // allocation that would fail. The default `Streamed` request always stages.
    let vocabulary_count = model.resident_vocabulary_tensor_count();
    assert_eq!(
        vocabulary_count,
        if model.vocabulary_residency() == DeepseekV41VocabularyResidency::Resident {
            2
        } else {
            0
        }
    );
    if vocabulary == DeepseekV41VocabularyResidency::Streamed {
        assert_eq!(vocabulary_count, 0);
    }
    eprintln!(
        "V4.1 production vocabulary tier: {:?} (requested {:?}, {vocabulary_count} resident tables)",
        model.vocabulary_residency(),
        vocabulary
    );
    let model_buffer_bytes = device.memory_stats()?.live_buffer_bytes - memory_before;
    let persistent_experts = if residency != DeepseekV41ExpertResidency::Resident {
        0
    } else {
        routed_experts * layers
    };
    assert_eq!(model.resident_routed_expert_count(), persistent_experts);
    assert_eq!(
        model.resident_shared_expert_count(),
        if residency != DeepseekV41ExpertResidency::StreamedAllExperts {
            layers
        } else {
            0
        }
    );
    assert_eq!(model.stream_precision(), DeepseekV41WeightOutput::Bf16);
    assert_eq!(
        model.resident_projection_layer_count(),
        if projections == DeepseekV41ProjectionResidency::Resident {
            layers
        } else {
            0
        }
    );

    let trace = std::env::var_os("V41_PRODUCTION_TRACE").is_some();
    if trace {
        model.backbone.trace = Some(vec![]);
    }
    let mut stages = vec![];
    let mut sparse_dumps = vec![];
    let mut exported_logits = vec![];
    let mut failures = vec![];
    for step in &case.steps {
        if trace {
            model.backbone.trace.as_mut().unwrap().clear();
        }
        let logits = if step.pos == 0 {
            model.prefill(&step.ids, step.seq, None)?
        } else {
            model.decode(&step.ids, step.pos, None)?
        };
        let shape = vec![logits.batch_size, logits.seq_len, logits.vocab_size];
        assert_eq!(model.resident_vocabulary_tensor_count(), vocabulary_count);
        assert_eq!(shape, step.shape, "{}", step.name);
        let values = logits.read_all()?;
        assert_eq!(values.len(), step.logits.len());
        assert!(values.iter().any(|&v| v != 0.0));
        let mut worst = 0.0f64;
        let mut per_row = vec![0.0f64; step.seq];
        let vocab = step.shape[2];
        for (index, (&actual, &expected)) in values.iter().zip(&step.logits).enumerate() {
            assert!(actual.is_finite() && expected.is_finite());
            let drift = (actual as f64 - expected as f64).abs();
            worst = worst.max(drift);
            per_row[index / vocab] = per_row[index / vocab].max(drift);
        }
        eprintln!(
            "V4.1 production width {}: {} logits, drift {worst:.17e} per row {per_row:?}",
            step.name,
            values.len()
        );
        if worst >= 2e-7 {
            failures.push(format!("{}: {worst:.17e}", step.name));
        }
        assert_eq!(model.next_position(), Some(step.pos + step.seq));
        assert_eq!(model.resident_routed_expert_count(), persistent_experts);
        if projections == DeepseekV41ProjectionResidency::Streamed {
            assert_eq!(model.resident_projection_layer_count(), 0);
        }
        exported_logits.push(serde_json::json!({
            "name":step.name,"shape":shape,"logits":values,
        }));
        if trace {
            if let Some((weights, raw)) = model.backbone.raw_attention_trace(0) {
                let weight_values: Vec<f32> = weights.read_f32(weights.f32_capacity())?;
                let raw_values: Vec<f32> = raw.read_f32(raw.f32_capacity())?;
                sparse_dumps.push(serde_json::json!({
                    "name": step.name,
                    "weights": weight_values,
                    "raw": raw_values,
                }));
            }
        }
        if trace {
            stages.push(serde_json::json!({
                "name": step.name,
                "stages": model
                    .backbone
                    .trace
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|(layer, stage, buffer, len)| -> Result<serde_json::Value> {
                        Ok(serde_json::json!({
                            "layer": layer,
                            "stage": stage,
                            "values": buffer.read_f32(*len)?,
                        }))
                    })
                    .collect::<Result<Vec<_>>>()?,
            }));
            let exported = exported_logits.last_mut().unwrap();
            exported["stages"] = stages.last().unwrap()["stages"].clone();
            exported["cache_diagnostics"] = serde_json::json!({});
            exported["index_diagnostics"] = serde_json::json!({});
        }
    }
    if trace {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../target/deepseek-v41-reference/{prefix}production-native-trace.json"
        ));
        fs::create_dir_all(path.parent().unwrap())?;
        let sparse_path = path.with_file_name(format!("{prefix}production-native-sparse.json"));
        fs::write(
            path,
            serde_json::to_vec(&serde_json::json!({"steps": stages}))?,
        )?;
        fs::write(
            sparse_path,
            serde_json::to_vec(&serde_json::json!({"steps":sparse_dumps}))?,
        )?;
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../target/deepseek-v41-reference/{prefix}production-native-logits.json"
    ));
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(
        path,
        serde_json::to_vec(&serde_json::json!({
            "seed":case.seed,"revision":case.revision,
            "fixture_sha256":format!("{:x}",Sha256::digest(source.as_bytes())),
            "checkpoint_files":case.checkpoint.files,"steps":exported_logits,
            "loader":if default_loader {"from_hf_package"} else {"explicit_residency"},
            "expert_residency":residency,"resident_shared_experts":model.resident_shared_expert_count(),
            "projection_residency":projections,"resident_projection_layers":model.resident_projection_layer_count(),
            "vocabulary_residency":vocabulary,"resident_vocabulary_tensors":model.resident_vocabulary_tensor_count(),
            "resident_routed_experts":persistent_experts,"initial_live_buffer_bytes":model_buffer_bytes,
        }))?,
    )?;
    assert!(
        failures.is_empty(),
        "strict <2e-7, rtol=0 failures: {failures:?}"
    );
    Ok(())
}
