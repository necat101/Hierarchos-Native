use super::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct SelectionFixture {
    revision: String,
    cases: Vec<SelectionCase>,
}

#[derive(Deserialize)]
struct SelectionCase {
    name: String,
    rows: usize,
    width: usize,
    seq: usize,
    start_pos: usize,
    ratio: usize,
    topk: usize,
    offset: usize,
    scores: Vec<Option<f32>>,
    expected_indices: Vec<i32>,
}

fn compressor_weights(ratio: usize) -> DeepseekV41CompressorWeights {
    let kv_weight = (0..16 * 8)
        .map(|i| ((i % 9) as f32 - 4.0) * 0.017)
        .collect();
    let gate_weight = (0..16 * 8)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.031)
        .collect();
    DeepseekV41CompressorWeights {
        hidden_size: 8,
        head_dim: 16,
        compress_ratio: ratio,
        eps: 1e-20,
        kv_weight,
        gate_weight: (ratio == 2).then_some(gate_weight),
        norm_weight: (0..16).map(|i| 0.5 + i as f32 * 0.025).collect(),
    }
}

fn indexer_config(owns_k: bool, candidate_source: bool) -> DeepseekV41IndexerConfig {
    DeepseekV41IndexerConfig {
        heads: 2,
        index_head_dim: 32,
        head_dim: 16,
        hidden_size: 8,
        q_lora_rank: 8,
        compress_ratio: 2,
        index_topk: 3,
        owns_k,
        is_candidate_source: candidate_source,
        candidate_topk_blocks: 1,
        candidate_block_size: 1,
        norm_eps: 1e-20,
        rotary: DeepseekV41RotaryConfig::sliding(8, 32, 10000.0),
    }
}

fn indexer_weights(owns_k: bool) -> DeepseekV41IndexerWeights {
    DeepseekV41IndexerWeights {
        wk: if owns_k {
            (0..32 * 16)
                .map(|i| ((i % 7) as f32 - 3.0) * 0.021)
                .collect()
        } else {
            Vec::new()
        },
        k_norm: if owns_k {
            (0..32).map(|i| 0.5 + (i % 5) as f32 * 0.1).collect()
        } else {
            Vec::new()
        },
        wq_b: (0..2 * 32 * 8)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.013)
            .collect(),
        weights_proj: (0..2 * 8).map(|i| 0.2 + ((i % 3) as f32) * 0.05).collect(),
    }
}

fn token_inputs(batch: usize, seq: usize, width: usize) -> Vec<f32> {
    (0..batch * seq * width)
        .map(|i| ((i * 29 % 17) as f32 - 8.0) * 0.037)
        .collect()
}

#[derive(Deserialize)]
struct IndexKeyFixture {
    revision: String,
    atol_exclusive: f64,
    rtol: f64,
    cases: Vec<IndexKeyCase>,
}

#[derive(Deserialize)]
struct IndexKeyCase {
    name: String,
    head_dim: usize,
    index_head_dim: usize,
    rope_dim: usize,
    batch: usize,
    groups: usize,
    ratio: usize,
    first_group: usize,
    start_pos: usize,
    norm_eps: f32,
    wk: Vec<f32>,
    k_norm: Vec<f32>,
    latent: Vec<f32>,
    expected: Vec<f32>,
}

#[test]
fn deepseek_v41_index_key_production_matches_pinned_fp32_reference() -> Result<()> {
    let fixture: IndexKeyFixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_index_keys.json"
    ))?;
    assert_eq!(fixture.revision, REFERENCE_REVISION);
    assert_eq!(fixture.atol_exclusive, 2e-7);
    assert_eq!(fixture.rtol, 0.0);
    let device = crate::VulkanDevice::new()?;
    let mut compared = 0usize;
    let mut worst = 0.0f64;
    for case in fixture.cases {
        let config = DeepseekV41IndexerConfig {
            heads: 2,
            index_head_dim: case.index_head_dim,
            head_dim: case.head_dim,
            hidden_size: 8,
            q_lora_rank: 8,
            compress_ratio: case.ratio,
            index_topk: 1,
            owns_k: true,
            is_candidate_source: false,
            candidate_topk_blocks: 0,
            candidate_block_size: 0,
            norm_eps: case.norm_eps,
            rotary: DeepseekV41RotaryConfig::sliding(case.rope_dim, 64, 10000.0),
        };
        let weights = DeepseekV41IndexerWeights {
            wk: case.wk,
            k_norm: case.k_norm,
            wq_b: vec![0.0; 2 * case.index_head_dim * 8],
            weights_proj: vec![0.0; 2 * 8],
        };
        let indexer = DeepseekV41Indexer::new(device.clone(), config, weights, case.batch, 64)?;
        let latent = crate::GpuBuffer::from_f32(&device, &case.latent)?;
        let keys = indexer
            .produce_index_keys(
                &latent,
                case.batch,
                case.groups,
                case.first_group,
                case.start_pos,
            )?
            .expect("key owner produces keys");
        let actual = keys.read_f32(case.expected.len())?;
        assert_eq!(actual.len(), case.expected.len(), "{} length", case.name);
        for (index, (&a, &b)) in actual.iter().zip(&case.expected).enumerate() {
            let drift = (a as f64 - b as f64).abs();
            assert!(
                drift < 2e-7,
                "{}, value {index}: native={a:.10e}, reference={b:.10e}, drift={drift:.10e}",
                case.name
            );
            worst = worst.max(drift);
        }
        compared += actual.len();
    }
    eprintln!(
        "V4.1 index keys on {}: {compared} reference values, worst absolute drift {worst:.12e}",
        device.name()
    );
    Ok(())
}

#[test]
fn deepseek_v41_index_score_kernel_matches_host_float64_reference() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let scorer = DeepseekV41IndexScorer::new(device.clone())?;
    let rows = 4usize;
    let batch = 2usize;
    let seq = 2usize;
    let heads = 3usize;
    let head_dim = 5usize;
    let width = 4usize;
    let capacity = 6usize;
    let ratio = 2usize;
    let scale = 0.37f32;
    let queries: Vec<f32> = (0..rows * heads * head_dim)
        .map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.25)
        .collect();
    let keys: Vec<f32> = (0..batch * capacity * head_dim)
        .map(|i| ((i * 5 % 9) as f32 - 4.0) * 0.25)
        .collect();
    let weights: Vec<f32> = (0..rows * heads)
        .map(|i| 0.125 * (1 + (i % 4)) as f32)
        .collect();
    let candidate_mask: Vec<u32> = (0..rows * width)
        .map(|i| u32::from((i + i / width) % 3 != 1))
        .collect();
    let q_buffer = crate::GpuBuffer::from_f32(&device, &queries)?;
    let k_buffer = crate::GpuBuffer::from_f32(&device, &keys)?;
    let w_buffer = crate::GpuBuffer::from_f32(&device, &weights)?;
    let c_buffer = crate::GpuBuffer::from_u32(&device, &candidate_mask)?;
    for (start_pos, use_candidates) in [(0usize, false), (3usize, true)] {
        let scores = scorer.score(
            &q_buffer,
            &k_buffer,
            &w_buffer,
            if use_candidates {
                Some(&c_buffer)
            } else {
                None
            },
            rows,
            batch,
            seq,
            heads,
            head_dim,
            width,
            capacity,
            start_pos,
            ratio,
            scale,
        )?;
        let actual = scores.read_f32(rows * width)?;
        let mut worst = 0.0f64;
        for row in 0..rows {
            let batch_index = row / seq;
            let local = row % seq;
            let visible = if start_pos == 0 {
                (local + 1) / ratio
            } else {
                (start_pos + seq) / ratio
            };
            for position in 0..width {
                let finite = position < visible
                    && (!use_candidates || candidate_mask[row * width + position] != 0);
                if !finite {
                    assert_eq!(
                        actual[row * width + position].to_bits(),
                        f32::NEG_INFINITY.to_bits(),
                        "row {row} position {position} must be masked"
                    );
                    continue;
                }
                let mut expected = 0.0f64;
                for head in 0..heads {
                    let mut dot = 0.0f64;
                    for d in 0..head_dim {
                        dot += queries[(row * heads + head) * head_dim + d] as f64
                            * keys[(batch_index * capacity + position) * head_dim + d] as f64;
                    }
                    expected += dot.max(0.0) * weights[row * heads + head] as f64 * scale as f64;
                }
                let drift = (actual[row * width + position] as f64 - expected).abs();
                assert!(
                    drift < 1e-5,
                    "row {row} position {position}: native={} expected={expected}, drift={drift:e}",
                    actual[row * width + position]
                );
                worst = worst.max(drift);
            }
        }
        eprintln!("V4.1 index score diagnostic start_pos={start_pos}: worst drift {worst:.6e}");
    }
    Ok(())
}

#[test]
fn deepseek_v41_index_selection_matches_pinned_reference_topk() -> Result<()> {
    let fixture: SelectionFixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_index_selection.json"
    ))?;
    assert_eq!(fixture.revision, REFERENCE_REVISION);
    let device = crate::VulkanDevice::new()?;
    let topk = DeepseekV41IndexTopk::new(device.clone())?;
    let mut compared = 0usize;
    for case in fixture.cases {
        let bits: Vec<u32> = case
            .scores
            .iter()
            .map(|value| value.unwrap_or(f32::NEG_INFINITY).to_bits())
            .collect();
        let scores = crate::GpuBuffer::from_u32(&device, &bits)?;
        let indices = topk.rank(
            &scores,
            case.rows,
            case.width,
            case.seq,
            case.start_pos,
            case.ratio,
            case.topk,
            case.offset,
        )?;
        let actual: Vec<i32> = indices
            .read_u32(case.rows * case.topk)?
            .into_iter()
            .map(|value| value as i32)
            .collect();
        assert_eq!(actual, case.expected_indices, "{}", case.name);
        compared += actual.len();
    }
    assert!(compared > 0);
    eprintln!(
        "V4.1 index selection on {}: {compared} exact indices",
        device.name()
    );
    Ok(())
}

#[test]
fn deepseek_v41_index_selection_rejects_invalid_topk_and_shapes() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let topk = DeepseekV41IndexTopk::new(device.clone())?;
    let scores = crate::GpuBuffer::from_f32(&device, &[0.0; 8])?;
    assert!(topk.rank(&scores, 1, 8, 1, 0, 2, 0, 0).is_err());
    assert!(topk.rank(&scores, 1, 8, 1, 0, 2, 9, 0).is_err());
    assert!(topk.rank(&scores, 1, 8, 0, 0, 2, 1, 0).is_err());
    assert!(topk.rank(&scores, 3, 8, 2, 0, 2, 1, 0).is_err());
    assert!(topk.rank(&scores, 1, 8, 1, 0, 0, 1, 0).is_err());
    Ok(())
}

#[test]
fn deepseek_v41_fine_indexer_owns_keys_and_returns_sorted_shared_positions() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let mut compressor = DeepseekV41Compressor::new(device.clone(), compressor_weights(2), 1, 8)?;
    let latent = compressor.prefill(&token_inputs(1, 4, 8), 1, 4)?;
    assert_eq!(latent.groups_per_batch, 2);
    let indexer = DeepseekV41Indexer::new(
        device.clone(),
        indexer_config(true, false),
        indexer_weights(true),
        1,
        32,
    )?;
    assert!(indexer.owns_keys());
    assert_eq!(indexer.write_index_keys(&latent, 0)?, Some(0));
    let keys = indexer.index_keys().expect("owner has a shared key cache");
    assert_eq!(keys.capacity(), 16);
    let cache_before = keys.read_batch(0, 2)?;
    assert!(cache_before.iter().any(|value| *value != 0.0));

    let x = crate::GpuBuffer::from_f32(&device, &token_inputs(1, 4, 8))?;
    let qr = crate::GpuBuffer::from_f32(&device, &token_inputs(1, 4, 8))?;
    let selection = indexer.select(&x, &qr, keys, 1, 4, 0, 5, None)?;
    assert_eq!(selection.width, 2);
    assert_eq!(selection.topk, 2);
    let scores = selection.read_scores()?;
    for row in 0..4 {
        let visible = (row + 1) / 2;
        let finite = scores[row * 2..(row + 1) * 2]
            .iter()
            .filter(|value| value.is_finite())
            .count();
        assert_eq!(finite, visible, "row {row} visible compressed positions");
        for value in &scores[row * 2..(row + 1) * 2] {
            assert!(value.is_finite() || (*value == f32::NEG_INFINITY));
        }
    }
    let indices = selection.read_indices()?;
    assert_eq!(indices.len(), 8);
    assert_eq!(&indices[0..2], &[-1, -1]);
    assert!(indices[6] >= 5 && indices[6] < 7, "row 3 first position");
    assert!(indices[7] >= 5 && indices[7] < 7, "row 3 second position");
    // Sorted positions always put unreachable selected slots last.
    for row in 0..4 {
        let row_indices = &indices[row * 2..(row + 1) * 2];
        assert!(row_indices[0] <= row_indices[1] || row_indices[1] == -1);
    }

    // A copy layer reads the source's cache and does not allocate one.
    let consumer = DeepseekV41Indexer::new(
        device.clone(),
        indexer_config(false, false),
        indexer_weights(false),
        1,
        32,
    )?;
    assert!(consumer.write_index_keys(&latent, 0)?.is_none());
    assert!(consumer.index_keys().is_none());
    let copied = consumer.select(&x, &qr, keys, 1, 4, 0, 5, None)?;
    assert_eq!(copied.read_indices()?, indices);

    // Decode appends group two without disturbing groups zero and one.
    let token = token_inputs(1, 1, 8);
    let pending = compressor.decode_step(&token, 1, 4)?;
    assert_eq!(pending.groups_per_batch, 0);
    let completed = compressor.decode_step(&token, 1, 5)?;
    assert_eq!(completed.groups_per_batch, 1);
    assert_eq!(indexer.write_index_keys(&completed, 5)?, Some(2));
    let cache_after = keys.read_batch(0, 3)?;
    assert_eq!(&cache_after[..2 * 32], &cache_before[..]);
    assert!(cache_after[2 * 32..].iter().any(|value| *value != 0.0));

    let repeated = indexer.select(&x, &qr, keys, 1, 4, 0, 5, None)?;
    assert_eq!(repeated.read_indices()?, indices);
    Ok(())
}

#[test]
fn deepseek_v41_candidate_source_publishes_masks_for_consumer_indexers() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let mut compressor = DeepseekV41Compressor::new(device.clone(), compressor_weights(2), 1, 8)?;
    let latent = compressor.prefill(&token_inputs(1, 4, 8), 1, 4)?;
    let source = DeepseekV41Indexer::new(
        device.clone(),
        indexer_config(true, true),
        indexer_weights(true),
        1,
        32,
    )?;
    source.write_index_keys(&latent, 0)?;
    let keys = source.index_keys().expect("source owns keys");
    let x = crate::GpuBuffer::from_f32(&device, &token_inputs(1, 4, 8))?;
    let qr = crate::GpuBuffer::from_f32(&device, &token_inputs(1, 4, 8))?;
    let source_selection = source.select(&x, &qr, keys, 1, 4, 0, 3, None)?;
    let candidates = source_selection
        .candidates
        .as_ref()
        .expect("candidate source publishes a mask");
    assert_eq!(
        candidates
            .read_u32(source_selection.rows * source_selection.width)?
            .len(),
        8
    );

    let consumer = DeepseekV41Indexer::new(
        device.clone(),
        indexer_config(false, false),
        indexer_weights(false),
        1,
        32,
    )?;
    let masked = consumer.select(&x, &qr, keys, 1, 4, 0, 3, Some(candidates))?;
    let source_indices = source_selection.read_indices()?;
    let masked_indices = masked.read_indices()?;
    let reachable_source = source_indices.iter().filter(|value| **value >= 0).count();
    let reachable_masked = masked_indices.iter().filter(|value| **value >= 0).count();
    assert!(reachable_masked <= reachable_source);
    assert!(masked_indices.iter().any(|value| *value == -1));
    Ok(())
}
