use super::engram::*;
use super::{DeepseekV41Config, DeepseekV41TextConfig, REFERENCE_REVISION};
use crate::{GpuBuffer, VulkanDevice};
use anyhow::Result;
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
struct Fixture {
    revision: String,
    layout_cases: Vec<LayoutCase>,
    hash_cases: Vec<HashCase>,
    gate_cases: Vec<GateCase>,
}

#[derive(Deserialize)]
struct LayoutCase {
    name: String,
    config: EngramConfigJson,
    expected_primes: Vec<Vec<Vec<u32>>>,
    expected_offsets: Vec<Vec<u32>>,
}

#[derive(Deserialize, Clone)]
struct EngramConfigJson {
    layer_ids: Vec<usize>,
    num_embeddings: Vec<usize>,
    max_ngram_size: usize,
    vocab_size: usize,
    n_heads: usize,
    head_dim: usize,
    pad_token_id: usize,
    compressed_vocab_size: usize,
    max_batch_size: usize,
    max_seq_len: usize,
}

#[derive(Deserialize)]
struct HashCase {
    name: String,
    config: EngramConfigJson,
    token_map: Vec<u32>,
    pad_token_id: usize,
    multipliers: Vec<Vec<u64>>,
    steps: Vec<HashStep>,
    expected: Vec<Vec<Vec<i32>>>,
}

#[derive(Deserialize)]
struct HashStep {
    start_pos: usize,
    seq: usize,
    token_ids: Vec<Vec<u32>>,
    token_mask: Vec<Vec<u32>>,
}

#[derive(Deserialize)]
struct GateCase {
    name: String,
    rows: usize,
    hc: usize,
    hidden: usize,
    head_dim: usize,
    num_embeddings: usize,
    cols: usize,
    eps: f32,
    atol: f64,
    stream: Vec<f32>,
    table: Vec<f32>,
    hashes: Vec<i32>,
    wkv: Vec<f32>,
    q_weight: Vec<f32>,
    k_weight: Vec<f32>,
    token_mask: Option<Vec<u32>>,
    expected: Vec<f32>,
}

fn fixture() -> Fixture {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_engram.json"
    ))
    .unwrap();
    assert_eq!(fixture.revision, REFERENCE_REVISION);
    fixture
}

fn text_config(config: &EngramConfigJson) -> Result<DeepseekV41TextConfig> {
    let mut value: Value = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_flash_1017_config.json"
    ))?;
    let layers = config.layer_ids.iter().copied().max().unwrap_or(0) + 2;
    let patch = json!({
        "num_hidden_layers": layers,
        "num_nextn_predict_layers": 0,
        "compress_ratios": vec![0usize; layers],
        "kv_source_layer_ids": [],
        "index_source_layer_ids": [],
        "candidate_source_layer_id": -1,
        "candidate_block_size": 0,
        "candidate_topk_blocks": 0,
        "engram_layer_ids": config.layer_ids,
        "engram_num_embeddings": config.num_embeddings,
        "engram_max_ngram_size": config.max_ngram_size,
        "engram_vocab_size": config.vocab_size,
        "engram_n_heads": config.n_heads,
        "engram_head_dim": config.head_dim,
        "engram_pad_token_id": config.pad_token_id,
        "engram_compressed_vocab_size": config.compressed_vocab_size,
        "dspark_block_size": 0,
        "dspark_target_layer_ids": []
    });
    value["text_config"]
        .as_object_mut()
        .unwrap()
        .extend(patch.as_object().unwrap().clone());
    Ok(DeepseekV41Config::from_hf_value(&value)?.text)
}

fn max_drift(actual: &[f32], expected: &[f32]) -> f64 {
    assert_eq!(actual.len(), expected.len(), "comparison length");
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (*a as f64 - *e as f64).abs())
        .fold(0.0f64, f64::max)
}

#[test]
fn deepseek_v41_engram_layout_matches_the_pinned_reference() -> Result<()> {
    let fixture = fixture();
    for case in &fixture.layout_cases {
        let layout = DeepseekV41EngramLayout::from_text(&text_config(&case.config)?)?
            .expect("engram layout");
        assert_eq!(layout.primes, case.expected_primes, "{} primes", case.name);
        assert_eq!(
            layout.offsets, case.expected_offsets,
            "{} offsets",
            case.name
        );
        assert_eq!(
            layout.n_hash_cols(),
            case.config.max_ngram_size.saturating_sub(1) * case.config.n_heads
        );
    }
    Ok(())
}

#[test]
fn deepseek_v41_engram_hash_matches_the_pinned_reference() -> Result<()> {
    let fixture = fixture();
    let device = VulkanDevice::new()?;
    for case in &fixture.hash_cases {
        let layout = DeepseekV41EngramLayout::from_text(&text_config(&case.config)?)?
            .expect("engram layout");
        let mut hash = DeepseekV41EngramHash::new(
            device.clone(),
            layout,
            DeepseekV41EngramTokens {
                token_map: case.token_map.clone(),
                multipliers: case.multipliers.clone(),
                pad_token_id: case.pad_token_id,
            },
            case.token_map.len(),
            case.config.max_batch_size,
            case.config.max_seq_len,
        )?;
        for (step_index, step) in case.steps.iter().enumerate() {
            let tokens: Vec<u32> = step.token_ids.iter().flatten().copied().collect();
            let mask: Vec<u32> = step.token_mask.iter().flatten().copied().collect();
            let tokens = GpuBuffer::from_u32(&device, &tokens)?;
            let mask = GpuBuffer::from_u32(&device, &mask)?;
            hash.begin_step(&tokens, step.start_pos, step.seq, Some(&mask))?;
            for (layer_index, expected) in case.expected.iter().enumerate() {
                let hashes = hash.hashes(layer_index)?;
                let values = hashes.read_u32(expected[step_index].len())?;
                let expected: Vec<u32> = expected[step_index]
                    .iter()
                    .map(|value| *value as u32)
                    .collect();
                assert_eq!(
                    values, expected,
                    "{} layer {layer_index} step {step_index} hash ids",
                    case.name
                );
            }
        }
    }
    Ok(())
}

#[test]
fn deepseek_v41_engram_gate_matches_the_pinned_reference() -> Result<()> {
    let fixture = fixture();
    let device = VulkanDevice::new()?;
    let mut worst = 0.0f64;
    let mut compared = 0usize;
    for case in &fixture.gate_cases {
        let config = EngramConfigJson {
            layer_ids: vec![0],
            num_embeddings: vec![case.num_embeddings],
            max_ngram_size: 2,
            vocab_size: 50,
            n_heads: case.cols,
            head_dim: case.head_dim,
            pad_token_id: 2,
            compressed_vocab_size: 8,
            max_batch_size: 1,
            max_seq_len: 1,
        };
        let layout =
            DeepseekV41EngramLayout::from_text(&text_config(&config)?)?.expect("engram layout");
        let engram = DeepseekV41Engram::new(
            device.clone(),
            &layout,
            0,
            case.hc,
            case.hidden,
            case.eps,
            DeepseekV41EngramWeights {
                embedding: case.table.clone(),
                wkv: case.wkv.clone(),
                q_weight: case.q_weight.clone(),
                k_weight: case.k_weight.clone(),
            },
        )?;
        let stream = GpuBuffer::from_f32(&device, &case.stream)?;
        let hashes: Vec<u32> = case.hashes.iter().map(|value| *value as u32).collect();
        let hashes = GpuBuffer::from_u32(&device, &hashes)?;
        let mask = case
            .token_mask
            .as_ref()
            .map(|values| GpuBuffer::from_u32(&device, values))
            .transpose()?;
        let output = engram.apply(&stream, &hashes, mask.as_ref(), case.rows)?;
        let output = output.read_f32(case.rows * case.hc * case.hidden)?;
        let drift = max_drift(&output, &case.expected);
        assert!(
            drift <= case.atol,
            "{} drift {drift:e} exceeds {}",
            case.name,
            case.atol
        );
        worst = worst.max(drift);
        compared += output.len();
    }
    assert!(worst > 0.0, "the comparison must exercise the gate");
    eprintln!("Engram gate worst drift {worst:e} over {compared} values");
    Ok(())
}

#[test]
fn deepseek_v41_engram_rejects_invalid_multipliers() -> Result<()> {
    let case = fixture();
    let case = &case.hash_cases[0];
    let layout = DeepseekV41EngramLayout::from_text(&text_config(&case.config)?)?.unwrap();
    let mut multipliers = case.multipliers.clone();
    multipliers[0][0] = multipliers[0][0].wrapping_add(1);
    let result = DeepseekV41EngramHash::new(
        VulkanDevice::new()?,
        layout,
        DeepseekV41EngramTokens {
            token_map: case.token_map.clone(),
            multipliers,
            pad_token_id: case.pad_token_id,
        },
        case.token_map.len(),
        case.config.max_batch_size,
        case.config.max_seq_len,
    );
    assert!(result.is_err(), "even hash multipliers must be rejected");
    Ok(())
}
