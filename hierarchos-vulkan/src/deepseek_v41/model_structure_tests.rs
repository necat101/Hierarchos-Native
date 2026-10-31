use super::*;
use crate::deepseek_v41::{
    DeepseekV41EngramResidency, DeepseekV41ExpertResidency, DeepseekV41ProjectionResidency,
};

#[derive(Deserialize)]
struct Checkpoint {
    directory: String,
    files: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct StructureCase {
    #[serde(flatten)]
    model: Fixture,
    checkpoint: Checkpoint,
}

#[derive(Deserialize)]
struct RoutingCase {
    steps: Vec<RoutedStep>,
}

#[derive(Deserialize)]
struct RoutedStep {
    routing: BTreeMap<String, Routing>,
}

#[derive(Deserialize, serde::Serialize, PartialEq, Debug)]
struct Routing {
    indices: Vec<u32>,
    weight_bits: Vec<u32>,
    counts: Vec<u32>,
}

/// PyTorch does not specify top-k tie order, so a tie between two selected
/// experts can swap their slots while every expert still receives the same
/// weight bits. Canonicalize each row by expert id; counts stay exact.
fn routes_match(actual: &Routing, expected: &Routing, topk: usize) -> bool {
    fn canonical(route: &Routing, topk: usize) -> Option<Vec<(u32, u32)>> {
        if topk == 0
            || route.indices.len() != route.weight_bits.len()
            || !route.indices.len().is_multiple_of(topk)
        {
            return None;
        }
        let mut rows = Vec::with_capacity(route.indices.len());
        for (indices, weights) in route
            .indices
            .chunks_exact(topk)
            .zip(route.weight_bits.chunks_exact(topk))
        {
            let mut pairs: Vec<(u32, u32)> = indices
                .iter()
                .copied()
                .zip(weights.iter().copied())
                .collect();
            pairs.sort_unstable();
            if pairs.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                return None;
            }
            rows.extend(pairs);
        }
        Some(rows)
    }
    actual.counts == expected.counts
        && canonical(actual, topk).is_some_and(|actual| {
            canonical(expected, topk).is_some_and(|expected| actual == expected)
        })
}

#[test]
fn deepseek_v41_released_depth_schedule_and_384_expert_strict_all_logits() -> Result<()> {
    run_structure_cases(
        include_str!("../../validation/fixtures/deepseek_v41_structure_model.json"),
        &[(4141101, 40, 8, 64), (4141102, 1, 384, 64)],
        (6, 2, 17),
        "deepseek_v41_structure",
        "structure-model-native-logits.json",
        "HIERARCHOS_V41_STRUCTURE_TRACE",
        &[],
    )
}

#[test]
fn deepseek_v41_streamed_depth_schedule_and_384_expert_strict_all_logits() -> Result<()> {
    run_structure_cases_with_residency(
        include_str!("../../validation/fixtures/deepseek_v41_structure_model.json"),
        &[(4141101, 40, 8, 64), (4141102, 1, 384, 64)],
        (6, 2, 17),
        "deepseek_v41_structure",
        "streamed-structure-model-native-logits.json",
        "HIERARCHOS_V41_STREAMED_TRACE",
        &[],
        DeepseekV41ExpertResidency::Streamed,
    )
}

#[test]
fn deepseek_v41_streamed_all_depth_schedule_and_384_expert_strict_all_logits() -> Result<()> {
    run_structure_cases_with_residency(
        include_str!("../../validation/fixtures/deepseek_v41_structure_model.json"),
        &[(4141101, 40, 8, 64), (4141102, 1, 384, 64)],
        (6, 2, 17),
        "deepseek_v41_structure",
        "streamed-all-structure-model-native-logits.json",
        "HIERARCHOS_V41_STREAMED_TRACE",
        &[],
        DeepseekV41ExpertResidency::StreamedAllExperts,
    )
}

#[test]
fn deepseek_v41_streamed_all_tables_depth_schedule_and_384_expert_strict_all_logits() -> Result<()>
{
    run_structure_cases_with_policies(
        include_str!("../../validation/fixtures/deepseek_v41_structure_model.json"),
        &[(4141101, 40, 8, 64), (4141102, 1, 384, 64)],
        (6, 2, 17),
        "deepseek_v41_structure",
        "streamed-tables-structure-model-native-logits.json",
        "HIERARCHOS_V41_STREAMED_TABLES_TRACE",
        &[],
        DeepseekV41ExpertResidency::StreamedAllExperts,
        DeepseekV41EngramResidency::StreamedTables,
        DeepseekV41ProjectionResidency::Resident,
    )
}

#[test]
fn deepseek_v41_streamed_all_tables_projections_structure_strict_all_logits() -> Result<()> {
    run_structure_cases_with_policies(
        include_str!("../../validation/fixtures/deepseek_v41_structure_model.json"),
        &[(4141101, 40, 8, 64), (4141102, 1, 384, 64)],
        (6, 2, 17),
        "deepseek_v41_structure",
        "streamed-block-structure-model-native-logits.json",
        "HIERARCHOS_V41_STREAMED_BLOCK_TRACE",
        &[],
        DeepseekV41ExpertResidency::StreamedAllExperts,
        DeepseekV41EngramResidency::StreamedTables,
        DeepseekV41ProjectionResidency::Streamed,
    )
}

#[test]
#[ignore = "40-layer/384-expert all-expert streamed graph; run explicitly"]
fn deepseek_v41_streamed_all_joint_strict_all_logits() -> Result<()> {
    run_structure_cases_with_residency(
        include_str!("../../validation/fixtures/deepseek_v41_joint_model.json"),
        &[(4141111, 40, 384, 32)],
        (6, 2, 17),
        "deepseek_v41_joint",
        "streamed-all-joint-model-native-logits.json",
        "HIERARCHOS_V41_STREAMED_TRACE",
        &[],
        DeepseekV41ExpertResidency::StreamedAllExperts,
    )
}

#[test]
#[ignore = "40-layer/384-expert streamed checkpoint graph; run explicitly"]
fn deepseek_v41_streamed_joint_strict_all_logits() -> Result<()> {
    run_structure_cases_with_residency(
        include_str!("../../validation/fixtures/deepseek_v41_joint_model.json"),
        &[(4141111, 40, 384, 32)],
        (6, 2, 17),
        "deepseek_v41_joint",
        "streamed-joint-model-native-logits.json",
        "HIERARCHOS_V41_STREAMED_TRACE",
        &[],
        DeepseekV41ExpertResidency::Streamed,
    )
}

#[test]
fn deepseek_v41_joint_cpu_fixture_remains_frozen() {
    let source = include_str!("../../validation/fixtures/deepseek_v41_joint_model.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(source.as_bytes())),
        "00cef75fe55c2243a6c2134c65e4c0adf224ce76791022c97f75738d4a1608b8",
        "joint CPU expectations were frozen before native measurement"
    );
}

#[test]
#[ignore = "joint 40-layer/384-expert resident graph; run explicitly on the local Vulkan device"]
fn deepseek_v41_joint_depth_and_384_expert_strict_all_logits() -> Result<()> {
    let source = include_str!("../../validation/fixtures/deepseek_v41_joint_model.json");
    run_structure_cases(
        source,
        &[(4141111, 40, 384, 32)],
        (6, 2, 17),
        "deepseek_v41_joint",
        "joint-model-native-logits.json",
        "HIERARCHOS_V41_JOINT_TRACE",
        &[],
    )
}

#[test]
fn deepseek_v41_long_context_cpu_fixture_remains_frozen() {
    let source = include_str!("../../validation/fixtures/deepseek_v41_long_context_model.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(source.as_bytes())),
        "82dcb51beea92833fa8dc382daf30846cd195f5974a17e9eaf0f9c7dd314bbbc",
        "long-context CPU expectations were frozen before native measurement"
    );
}

#[test]
#[ignore = "40-layer/384-expert 2,048-token YaRN graph; run explicitly on the local Vulkan device"]
fn deepseek_v41_long_context_2048_token_strict_all_logits() -> Result<()> {
    let source = include_str!("../../validation/fixtures/deepseek_v41_long_context_model.json");
    run_structure_cases(
        source,
        &[(4141112, 40, 384, 32)],
        (7, 1, 2048),
        "deepseek_v41_long_context",
        "long-context-model-native-logits.json",
        "HIERARCHOS_V41_LONG_TRACE",
        &["long-decode-2049"],
    )
}

#[test]
#[ignore = "joint 40-layer/384-expert streamed experts and tables; run explicitly"]
fn deepseek_v41_streamed_all_tables_joint_strict_all_logits() -> Result<()> {
    run_structure_cases_with_policies(
        include_str!("../../validation/fixtures/deepseek_v41_joint_model.json"),
        &[(4141111, 40, 384, 32)],
        (6, 2, 17),
        "deepseek_v41_joint",
        "streamed-tables-joint-model-native-logits.json",
        "HIERARCHOS_V41_STREAMED_TABLES_TRACE",
        &[],
        DeepseekV41ExpertResidency::StreamedAllExperts,
        DeepseekV41EngramResidency::StreamedTables,
        DeepseekV41ProjectionResidency::Resident,
    )
}

#[test]
#[ignore = "40-layer/384-expert streamed projections joint graph; run explicitly"]
fn deepseek_v41_streamed_block_joint_strict_all_logits() -> Result<()> {
    run_structure_cases_with_policies(
        include_str!("../../validation/fixtures/deepseek_v41_joint_model.json"),
        &[(4141111, 40, 384, 32)],
        (6, 2, 17),
        "deepseek_v41_joint",
        "streamed-block-joint-model-native-logits.json",
        "HIERARCHOS_V41_STREAMED_BLOCK_TRACE",
        &[],
        DeepseekV41ExpertResidency::StreamedAllExperts,
        DeepseekV41EngramResidency::StreamedTables,
        DeepseekV41ProjectionResidency::Streamed,
    )
}

#[test]
#[ignore = "40-layer/384-expert all-expert streamed 2,048-token graph; run explicitly"]
fn deepseek_v41_streamed_all_long_context_2048_strict_all_logits() -> Result<()> {
    run_structure_cases_with_residency(
        include_str!("../../validation/fixtures/deepseek_v41_long_context_model.json"),
        &[(4141112, 40, 384, 32)],
        (7, 1, 2048),
        "deepseek_v41_long_context",
        "streamed-all-long-context-model-native-logits.json",
        "HIERARCHOS_V41_STREAMED_LONG_TRACE",
        &["long-decode-2049"],
        DeepseekV41ExpertResidency::StreamedAllExperts,
    )
}

#[test]
#[ignore = "40-layer/384-expert streamed experts and tables 2,048-token graph; run explicitly"]
fn deepseek_v41_streamed_all_tables_long_context_2048_strict_all_logits() -> Result<()> {
    run_structure_cases_with_policies(
        include_str!("../../validation/fixtures/deepseek_v41_long_context_model.json"),
        &[(4141112, 40, 384, 32)],
        (7, 1, 2048),
        "deepseek_v41_long_context",
        "streamed-tables-long-context-model-native-logits.json",
        "HIERARCHOS_V41_STREAMED_TABLES_TRACE",
        &["long-decode-2049"],
        DeepseekV41ExpertResidency::StreamedAllExperts,
        DeepseekV41EngramResidency::StreamedTables,
        DeepseekV41ProjectionResidency::Resident,
    )
}

#[test]
#[ignore = "40-layer/384-expert streamed projections 2,048-token graph; run explicitly"]
fn deepseek_v41_streamed_block_long_context_2048_strict_all_logits() -> Result<()> {
    run_structure_cases_with_policies(
        include_str!("../../validation/fixtures/deepseek_v41_long_context_model.json"),
        &[(4141112, 40, 384, 32)],
        (7, 1, 2048),
        "deepseek_v41_long_context",
        "streamed-block-long-context-model-native-logits.json",
        "HIERARCHOS_V41_STREAMED_BLOCK_LONG_TRACE",
        &["long-decode-2049"],
        DeepseekV41ExpertResidency::StreamedAllExperts,
        DeepseekV41EngramResidency::StreamedTables,
        DeepseekV41ProjectionResidency::Streamed,
    )
}

#[test]
#[ignore = "40-layer/384-expert streamed 2,048-token graph; run explicitly"]
fn deepseek_v41_streamed_long_context_2048_strict_all_logits() -> Result<()> {
    run_structure_cases_with_residency(
        include_str!("../../validation/fixtures/deepseek_v41_long_context_model.json"),
        &[(4141112, 40, 384, 32)],
        (7, 1, 2048),
        "deepseek_v41_long_context",
        "streamed-long-context-model-native-logits.json",
        "HIERARCHOS_V41_STREAMED_LONG_TRACE",
        &["long-decode-2049"],
        DeepseekV41ExpertResidency::Streamed,
    )
}

fn run_structure_cases(
    source: &str,
    expected: &[(u64, usize, usize, usize)],
    shape: (usize, usize, usize),
    prefix: &str,
    export_name: &str,
    trace_env: &str,
    trace_steps: &[&str],
) -> Result<()> {
    run_structure_cases_with_residency(
        source,
        expected,
        shape,
        prefix,
        export_name,
        trace_env,
        trace_steps,
        DeepseekV41ExpertResidency::Resident,
    )
}

fn run_structure_cases_with_residency(
    source: &str,
    expected: &[(u64, usize, usize, usize)],
    shape: (usize, usize, usize),
    prefix: &str,
    export_name: &str,
    trace_env: &str,
    trace_steps: &[&str],
    residency: DeepseekV41ExpertResidency,
) -> Result<()> {
    run_structure_cases_with_policies(
        source,
        expected,
        shape,
        prefix,
        export_name,
        trace_env,
        trace_steps,
        residency,
        DeepseekV41EngramResidency::Resident,
        DeepseekV41ProjectionResidency::Resident,
    )
}

fn run_structure_cases_with_policies(
    source: &str,
    expected: &[(u64, usize, usize, usize)],
    shape: (usize, usize, usize),
    prefix: &str,
    export_name: &str,
    trace_env: &str,
    trace_steps: &[&str],
    residency: DeepseekV41ExpertResidency,
    engram_residency: DeepseekV41EngramResidency,
    projections: DeepseekV41ProjectionResidency,
) -> Result<()> {
    run_structure_cases_with_vocabulary(
        source,
        expected,
        shape,
        prefix,
        export_name,
        trace_env,
        trace_steps,
        residency,
        engram_residency,
        projections,
        DeepseekV41VocabularyResidency::Resident,
    )
}

#[test]
fn deepseek_v41_streamed_vocabulary_structure_strict_all_logits() -> Result<()> {
    run_structure_cases_with_vocabulary(
        include_str!("../../validation/fixtures/deepseek_v41_structure_model.json"),
        &[(4141101, 40, 8, 64), (4141102, 1, 384, 64)],
        (6, 2, 17),
        "deepseek_v41_structure",
        "streamed-vocabulary-structure-model-native-logits.json",
        "HIERARCHOS_V41_STREAMED_VOCABULARY_TRACE",
        &[],
        DeepseekV41ExpertResidency::StreamedAllExperts,
        DeepseekV41EngramResidency::StreamedTables,
        DeepseekV41ProjectionResidency::Streamed,
        DeepseekV41VocabularyResidency::Streamed,
    )
}

#[test]
#[ignore = "40-layer/384-expert streamed vocabulary joint graph; run explicitly"]
fn deepseek_v41_streamed_vocabulary_joint_strict_all_logits() -> Result<()> {
    run_structure_cases_with_vocabulary(
        include_str!("../../validation/fixtures/deepseek_v41_joint_model.json"),
        &[(4141111, 40, 384, 32)],
        (6, 2, 17),
        "deepseek_v41_joint",
        "streamed-vocabulary-joint-model-native-logits.json",
        "HIERARCHOS_V41_STREAMED_VOCABULARY_TRACE",
        &[],
        DeepseekV41ExpertResidency::StreamedAllExperts,
        DeepseekV41EngramResidency::StreamedTables,
        DeepseekV41ProjectionResidency::Streamed,
        DeepseekV41VocabularyResidency::Streamed,
    )
}

#[test]
#[ignore = "40-layer/384-expert streamed vocabulary 2,048-token graph; run explicitly"]
fn deepseek_v41_streamed_vocabulary_long_context_2048_strict_all_logits() -> Result<()> {
    run_structure_cases_with_vocabulary(
        include_str!("../../validation/fixtures/deepseek_v41_long_context_model.json"),
        &[(4141112, 40, 384, 32)],
        (7, 1, 2048),
        "deepseek_v41_long_context",
        "streamed-vocabulary-long-context-model-native-logits.json",
        "HIERARCHOS_V41_STREAMED_VOCABULARY_LONG_TRACE",
        &["long-decode-2049"],
        DeepseekV41ExpertResidency::StreamedAllExperts,
        DeepseekV41EngramResidency::StreamedTables,
        DeepseekV41ProjectionResidency::Streamed,
        DeepseekV41VocabularyResidency::Streamed,
    )
}

fn run_structure_cases_with_vocabulary(
    source: &str,
    expected: &[(u64, usize, usize, usize)],
    shape: (usize, usize, usize),
    prefix: &str,
    export_name: &str,
    trace_env: &str,
    trace_steps: &[&str],
    residency: DeepseekV41ExpertResidency,
    engram_residency: DeepseekV41EngramResidency,
    projections: DeepseekV41ProjectionResidency,
    vocabulary: DeepseekV41VocabularyResidency,
) -> Result<()> {
    let (step_count, batch_size, max_prefill) = shape;
    #[derive(Deserialize)]
    struct Cases {
        cases: Vec<StructureCase>,
    }
    #[derive(Deserialize)]
    struct Routes {
        cases: Vec<RoutingCase>,
    }
    let fixture: Cases = serde_json::from_str(source)?;
    let routes: Routes = serde_json::from_str(source)?;
    assert_eq!(fixture.cases.len(), expected.len());
    assert_eq!(routes.cases.len(), expected.len());
    let binding = format!("{:x}", Sha256::digest(source.as_bytes()));
    let device = VulkanDevice::new()?;
    let trace = std::env::var_os(trace_env).is_some();
    let mut failures = vec![];
    let mut exported = vec![];
    let output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/deepseek-v41-reference")
        .join(export_name);
    fs::create_dir_all(output_path.parent().unwrap())?;
    for ((case, route_case), &(seed, layers, experts, hidden)) in
        fixture.cases.into_iter().zip(routes.cases).zip(expected)
    {
        let f = case.model;
        let config = &f.config["text_config"];
        assert_eq!(f.seed, seed);
        assert_eq!(f.steps.len(), step_count);
        assert_eq!(route_case.steps.len(), f.steps.len());
        assert_eq!(f.stream_precision, "bf16");
        assert_eq!(f.revision, crate::deepseek_v41::REFERENCE_REVISION);
        assert_eq!(
            f.source_sha256,
            "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
        );
        assert_eq!(config["num_hidden_layers"], layers);
        assert_eq!(config["n_routed_experts"], experts);
        assert_eq!(config["num_experts_per_tok"], 6);
        assert_eq!(config["hidden_size"], hidden);
        assert_eq!(config["moe_intermediate_size"], hidden);
        assert_eq!(config["vocab_size"], 67);
        assert_eq!(f.batch_size, batch_size);
        assert_eq!(f.max_prefill, max_prefill);
        if layers == 40 {
            let released: Value = serde_json::from_str(include_str!(
                "../../validation/fixtures/deepseek_v41_flash_1017_config.json"
            ))?;
            for field in [
                "kv_source_layer_ids",
                "index_source_layer_ids",
                "candidate_source_layer_id",
                "engram_layer_ids",
                "engram_max_ngram_size",
                "engram_n_heads",
            ] {
                assert_eq!(config[field], released["text_config"][field], "{field}");
            }
            assert_eq!(
                config["compress_ratios"],
                json!(
                    released["text_config"]["compress_ratios"]
                        .as_array()
                        .unwrap()[..40]
                )
            );
        }
        assert_eq!(case.checkpoint.directory, format!("{prefix}_{seed}"));
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
        let mut checkpoint = DeepseekV41Checkpoint::open(&root)?;
        let memory_before = device.memory_stats()?.live_buffer_bytes;
        let mut model = DeepseekV41CausalLM::from_checkpoint_with_vocabulary_residency(
            device.clone(),
            &mut checkpoint,
            runtime(&f),
            (layers == 40).then(|| f.tokens.clone()),
            DeepseekV41WeightOutput::Bf16,
            residency,
            engram_residency,
            projections,
            vocabulary,
        )?;
        // The streamed source owns shared header/index metadata and remains
        // usable after the caller releases its reader.
        drop(checkpoint);
        let vocabulary_count = if vocabulary == DeepseekV41VocabularyResidency::Resident {
            2
        } else {
            0
        };
        assert_eq!(model.resident_vocabulary_tensor_count(), vocabulary_count);
        let model_buffer_bytes = device.memory_stats()?.live_buffer_bytes - memory_before;
        if projections == DeepseekV41ProjectionResidency::Streamed {
            assert_eq!(
                model.resident_projection_layer_count(),
                0,
                "streamed block keeps no attention/indexer/compressor projections"
            );
        }
        let persistent_experts = if residency != DeepseekV41ExpertResidency::Resident {
            0
        } else {
            layers * experts
        };
        assert_eq!(model.resident_routed_expert_count(), persistent_experts);
        let shared_count = if residency == DeepseekV41ExpertResidency::StreamedAllExperts {
            0
        } else {
            layers
        };
        assert_eq!(model.resident_shared_expert_count(), shared_count);
        let mut steps = vec![];
        let mut high_expert_seen = false;
        for (step, routed_step) in f.steps.iter().zip(&route_case.steps) {
            model.backbone.trace = (trace
                && (trace_steps.is_empty() || trace_steps.contains(&step.name.as_str())))
            .then(Vec::new);
            let mask = (layers == 40).then_some(step.mask.as_slice());
            let logits = if step.pos == 0 {
                model.prefill(&step.ids, step.seq, mask)?
            } else {
                model.decode(&step.ids, step.pos, mask)?
            };
            let shape = vec![logits.batch_size, logits.seq_len, logits.vocab_size];
            assert_eq!(shape, step.shape);
            let values = logits.read_all()?;
            assert!(!values.is_empty() && values.iter().any(|&v| v != 0.0));
            assert_eq!(values.len(), step.logits.len());
            let mut worst = 0.0f64;
            for (&actual, &expected) in values.iter().zip(&step.logits) {
                assert!(actual.is_finite() && expected.is_finite());
                worst = worst.max((actual as f64 - expected as f64).abs());
            }
            eprintln!(
                "V4.1 structure seed {seed}, {}: {} logits, drift {worst:.17e}",
                step.name,
                values.len()
            );
            if worst >= 2e-7 {
                failures.push(format!("{seed} {} logits: {worst:.17e}", step.name));
            }
            assert_eq!(model.next_position(), Some(step.pos + step.seq));
            assert_eq!(model.resident_vocabulary_tensor_count(), vocabulary_count);
            assert_eq!(model.resident_routed_expert_count(), persistent_experts);
            if projections == DeepseekV41ProjectionResidency::Streamed {
                assert_eq!(
                    model.resident_projection_layer_count(),
                    0,
                    "streamed projections must not be retained between layers"
                );
            }
            let rows = f.batch_size * step.seq;
            let mut routing = BTreeMap::new();
            assert_eq!(routed_step.routing.len(), layers);
            for layer in 0..layers {
                let (weights, indices, counts) = model
                    .backbone
                    .moe_route_trace(layer)
                    .expect("complete routing trace");
                let route = Routing {
                    indices: indices.read_u32(rows * 6)?,
                    weight_bits: weights.read_u32(rows * 6)?,
                    counts: counts.read_u32(experts)?,
                };
                assert_eq!(route.counts.iter().sum::<u32>() as usize, rows * 6);
                assert!(route.indices.iter().all(|&i| (i as usize) < experts));
                high_expert_seen |= route.indices.iter().any(|&i| i > 255);
                if !routes_match(&route, &routed_step.routing[&layer.to_string()], 6) {
                    failures.push(format!("{seed} {} layer {layer} routing bits", step.name));
                }
                routing.insert(layer.to_string(), route);
            }
            let (collapsed, normalized) = model.final_trace.as_ref().unwrap();
            let mut output = json!({"name":step.name, "shape":shape, "logits":values,
                "routing":routing, "collapsed":collapsed.read_f32(rows*hidden)?,
                "normalized":normalized.read_f32(rows*hidden)?});
            if let Some(stages) = &model.backbone.trace {
                let window = f.config["text_config"]["sliding_window"]
                    .as_u64()
                    .context("V4.1 config has no sliding window")?
                    as usize;
                // Compressed indices address `[window rows, compressed rows]`.
                output["index_offset"] = json!(if step.pos == 0 { step.seq } else { window });
                output["stages"] = json!(stages
                    .iter()
                    .map(|(layer, stage, buffer, len)| -> Result<_> {
                        Ok(json!({"layer":layer,"stage":stage,"values":buffer.read_f32(*len)?}))
                    })
                    .collect::<Result<Vec<_>>>()?);
                let mut selections = serde_json::Map::new();
                let mut caches = serde_json::Map::new();
                for layer in 0..layers {
                    if let Some(selection) = model.backbone.index_selection(layer)? {
                        if selection.width > 0 {
                            selections.insert(layer.to_string(), json!({"indices":selection.read_indices()?,
                                "score_bits":selection.scores.read_u32(selection.rows*selection.width)?}));
                        }
                    }
                    if let Some((kv, keys)) = model.backbone.attention_cache_trace(layer)? {
                        caches.insert(layer.to_string(), json!({"kv":kv,"keys":keys}));
                    }
                }
                output["index_diagnostics"] = json!(selections);
                output["cache_diagnostics"] = json!(caches);
            }
            steps.push(output);
        }
        if experts == 384 {
            assert!(high_expert_seen, "exercise indices beyond eight bits");
        }
        exported.push(
            json!({"seed":seed,"revision":f.revision,"fixture_sha256":binding,
            "expert_residency":residency,"resident_shared_experts":model.resident_shared_expert_count(),
            "engram_residency":engram_residency,"resident_engram_tables":model.resident_engram_table_count(),
            "projection_residency":projections,
            "vocabulary_residency":vocabulary,"resident_vocabulary_tensors":model.resident_vocabulary_tensor_count(),
            "resident_projection_layers":model.resident_projection_layer_count(),
            "resident_routed_experts":persistent_experts,
            "initial_live_buffer_bytes":model_buffer_bytes,
            "checkpoint_files":case.checkpoint.files,"steps":steps}),
        );
        // Preserve completed traces if a later package fails to load. Untraced
        // regression runs must not overwrite staged evidence with a partial export.
        if trace {
            fs::write(
                &output_path,
                serde_json::to_vec(&json!({"cases":exported}))?,
            )?;
        }
    }
    assert!(
        failures.is_empty(),
        "strict <2e-7, rtol=0 or exact-routing failures: {failures:?}"
    );
    Ok(())
}
