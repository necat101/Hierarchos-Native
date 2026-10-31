use super::*;

fn released_config() -> Value {
    serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_flash_1017_config.json"
    ))
    .unwrap()
}

#[test]
fn deepseek_v41_released_config_preserves_package_and_builds_sharing_plan() -> Result<()> {
    let value = released_config();
    let config = DeepseekV41Config::from_hf_value(&value)?;
    assert_eq!(config.source_config, value);
    assert_eq!(config.text.rms_norm_eps, 1e-20);
    assert_eq!(config.text.n_routed_experts, 384);
    assert_eq!(config.text.index_n_heads, 32);
    assert_eq!(config.text.q_lora_rank, 1280);
    assert_eq!(
        config.source_config["quantization_config"]["expert_dtype"],
        "fp4"
    );
    assert_eq!(
        config.source_config["quantization_config"]["weight_block_size"],
        serde_json::json!([32, 32])
    );
    assert_eq!(
        config.source_config["vision_config"]["model_type"],
        "deepseek_v41_vision"
    );
    let plan = config.plan()?;
    assert_eq!(plan.backbone.len(), 40);
    assert_eq!(plan.draft_layer_count, 3);
    for layer in 0..40 {
        let p = &plan.backbone[layer];
        assert_eq!(p.layer_id, layer);
        if layer < 2 {
            assert_eq!(p.compress_ratio, 0);
            assert_eq!(p.kv_source_layer, None);
            assert_eq!(p.index_source_layer, None);
        } else {
            let kv = if layer < 8 {
                2
            } else if layer < 14 {
                8
            } else if layer < 20 {
                14
            } else {
                20
            };
            let index = if layer < 24 { kv } else { layer / 4 * 4 };
            assert_eq!(p.compress_ratio, if layer < 20 { 2 } else { 1 });
            assert_eq!(p.kv_source_layer, Some(kv));
            assert_eq!(p.index_source_layer, Some(index));
            assert_eq!(
                p.candidate_source_layer,
                if layer >= 24 { Some(20) } else { None }
            );
        }
        assert_eq!(p.publishes_candidates, layer == 20);
        assert_eq!(
            p.engram_table_index,
            match layer {
                1 => Some(0),
                14 => Some(1),
                _ => None,
            }
        );
    }
    Ok(())
}

#[test]
fn deepseek_v41_standalone_text_is_distinct_and_executor_fails_closed() -> Result<()> {
    let package = released_config();
    let text = &package["text_config"];
    assert_eq!(
        DeepseekV41Config::from_hf_value(text)?.plan()?,
        DeepseekV41Config::from_hf_value(&package)?.plan()?
    );
    for value in [&package, text] {
        let err = crate::VulkanTransformerConfig::from_hf_value(value)
            .unwrap_err()
            .to_string();
        assert!(err.contains("dedicated native graph"), "{err}");
        assert!(
            err.contains("DeepseekV41CausalLM::from_hf_package"),
            "{err}"
        );
        assert!(
            err.contains("instead of the DeepSeek-V4 transformer"),
            "{err}"
        );
    }
    for model_type in ["deepseek_v4", "deepseek_v3", "kimi_linear"] {
        let mut wrong = text.clone();
        wrong["model_type"] = serde_json::json!(model_type);
        assert!(DeepseekV41Config::from_hf_value(&wrong).is_err());
    }
    Ok(())
}

/// The released table row counts are exactly the sum of the generated prime
/// buckets. This binds the Rust layout to the published SafeTensor shapes.
#[test]
fn deepseek_v41_released_engram_prime_ranges_match_table_rows() -> Result<()> {
    let config = DeepseekV41Config::from_hf_value(&released_config())?;
    let layout = DeepseekV41EngramLayout::from_text(&config.text)?.unwrap();
    for (index, &rows) in config.text.engram_num_embeddings.iter().enumerate() {
        let primes: Vec<u32> = layout.primes[index].iter().flatten().copied().collect();
        assert_eq!(layout.offsets[index].len(), primes.len());
        let last = primes.len() - 1;
        let total = layout.offsets[index][last] as u64 + primes[last] as u64;
        assert_eq!(total, rows as u64, "Engram layer {index}");
    }
    Ok(())
}

#[test]
fn deepseek_v41_rejects_invalid_sharing_schedules() {
    for (field, invalid) in [
        ("kv_source_layer_ids", serde_json::json!([8, 14, 20])),
        ("kv_source_layer_ids", serde_json::json!([2, 8, 14])),
        ("kv_source_layer_ids", serde_json::json!([2, 8, 8, 20])),
        ("kv_source_layer_ids", serde_json::json!([2, 14, 8, 20])),
        ("index_source_layer_ids", serde_json::json!([8, 14, 20])),
        (
            "index_source_layer_ids",
            serde_json::json!([2, 8, 14, 20, 40]),
        ),
        ("candidate_source_layer_id", serde_json::json!(21)),
        ("candidate_source_layer_id", serde_json::json!(14)),
        ("candidate_source_layer_id", serde_json::json!(-2)),
        ("candidate_block_size", serde_json::json!(0)),
        ("engram_layer_ids", serde_json::json!([1, 1])),
        ("engram_num_embeddings", serde_json::json!([1])),
        ("dspark_target_layer_ids", serde_json::json!([40])),
    ] {
        let mut value = released_config();
        value["text_config"][field] = invalid;
        assert!(
            DeepseekV41Config::from_hf_value(&value).is_err(),
            "accepted invalid {field}"
        );
    }
    for (layer, ratio) in [(2, 4), (2, 0), (15, 1), (40, 1)] {
        let mut value = released_config();
        value["text_config"]["compress_ratios"][layer] = serde_json::json!(ratio);
        assert!(
            DeepseekV41Config::from_hf_value(&value).is_err(),
            "accepted ratio {ratio} at {layer}"
        );
    }
}

#[test]
fn deepseek_v41_rejects_invalid_dimensions_and_missing_fields() {
    for (field, invalid) in [
        ("hidden_size", serde_json::json!(0)),
        ("q_lora_rank", serde_json::json!(-1)),
        ("num_key_value_heads", serde_json::json!(8)),
        ("o_groups", serde_json::json!(3)),
        ("qk_rope_head_dim", serde_json::json!(63)),
        ("index_head_dim", serde_json::json!(32)),
        ("rms_norm_eps", serde_json::json!(0.0)),
        ("engram_compressed_vocab_size", serde_json::json!(200000)),
        ("num_experts_per_tok", serde_json::json!(385)),
        ("compress_ratios", serde_json::json!([0, 0, 2])),
    ] {
        let mut value = released_config();
        value["text_config"][field] = invalid;
        assert!(
            DeepseekV41Config::from_hf_value(&value).is_err(),
            "accepted invalid {field}"
        );
    }
    let mut value = released_config();
    value["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("q_lora_rank");
    assert!(DeepseekV41Config::from_hf_value(&value).is_err());
    value.as_object_mut().unwrap().remove("text_config");
    assert!(DeepseekV41Config::from_hf_value(&value).is_err());
}

#[test]
fn deepseek_v41_candidate_filter_can_be_disabled() -> Result<()> {
    let mut value = released_config();
    value["text_config"]["candidate_source_layer_id"] = serde_json::json!(-1);
    let plan = DeepseekV41Config::from_hf_value(&value)?.plan()?;
    assert!(plan
        .backbone
        .iter()
        .all(|layer| !layer.publishes_candidates && layer.candidate_source_layer.is_none()));
    Ok(())
}

#[derive(Deserialize)]
struct CandidateFixture {
    revision: String,
    cases: Vec<CandidateCase>,
}

#[derive(Deserialize)]
struct CandidateCase {
    name: String,
    width: usize,
    block_size: usize,
    topk_blocks: usize,
    visible_lengths: Vec<u32>,
    logits: Vec<Option<f32>>,
    expected_mask: Vec<u32>,
}

#[test]
fn deepseek_v41_vulkan_candidates_match_pinned_reference() -> Result<()> {
    let fixture: CandidateFixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_candidates.json"
    ))?;
    assert_eq!(fixture.revision, REFERENCE_REVISION);
    let device = crate::VulkanDevice::new()?;
    eprintln!("V4.1 candidate selector device: {}", device.name());
    let selector = DeepseekV41CandidateSelector::new(device.clone())?;
    for case in fixture.cases {
        let logits: Vec<f32> = case
            .logits
            .into_iter()
            .map(|x| x.unwrap_or(f32::NEG_INFINITY))
            .collect();
        let actual = selector.select(
            &logits,
            &case.visible_lengths,
            case.width,
            case.block_size,
            case.topk_blocks,
        )?;
        assert_eq!(actual, case.expected_mask, "{}", case.name);
        let bits: Vec<u32> = logits.iter().map(|x| x.to_bits()).collect();
        let resident = if bits.is_empty() {
            crate::GpuBuffer::zeros_u32(&device, 1)?
        } else {
            crate::GpuBuffer::from_u32(&device, &bits)?
        };
        let mask = selector.select_buffer(
            &resident,
            &case.visible_lengths,
            case.width,
            case.block_size,
            case.topk_blocks,
        )?;
        if !case.expected_mask.is_empty() {
            assert_eq!(
                mask.read_u32(case.expected_mask.len())?,
                case.expected_mask,
                "resident {}",
                case.name
            );
        }
    }
    // Repeated equal block maxima have a deterministic native tie break.
    for _ in 0..3 {
        assert_eq!(
            selector.select(&[1.0; 6], &[6], 6, 2, 2)?,
            vec![1, 1, 0, 0, 1, 1]
        );
    }
    assert!(selector.select(&[f32::NAN], &[1], 1, 1, 1).is_err());
    assert!(selector.select(&[f32::INFINITY], &[1], 1, 1, 1).is_err());
    assert!(selector.select(&[1.0], &[2], 1, 1, 1).is_err());
    assert!(selector.select(&[1.0], &[1], 1, 0, 1).is_err());
    assert!(selector.select(&[], &[1], 1, 1, 1).is_err());
    let small = crate::GpuBuffer::zeros_f32(&device, 1)?;
    assert!(selector.select_buffer(&small, &[2], 2, 1, 1).is_err());
    assert!(selector.select_buffer(&small, &[2], 1, 1, 1).is_err());
    let foreign_device = crate::VulkanDevice::new()?;
    let foreign = crate::GpuBuffer::zeros_f32(&foreign_device, 1)?;
    assert!(selector.select_buffer(&foreign, &[1], 1, 1, 1).is_err());
    Ok(())
}
