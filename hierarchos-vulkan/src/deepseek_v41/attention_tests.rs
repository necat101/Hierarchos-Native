use super::*;

use crate::GpuBuffer;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as f32 / 2_147_483_648.0) - 0.5
    }

    fn vector(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| self.next() * scale).collect()
    }
}

fn rotary_config(dim: usize) -> DeepseekV41RotaryConfig {
    DeepseekV41RotaryConfig::compressed(dim, 64, 10_000.0, 16, 1.0, 32.0, 1.0)
}

fn attention_weights(seed: u64) -> DeepseekV41AttentionWeights {
    let mut rng = Lcg(seed);
    let (hidden, q_lora, heads, head_dim, o_groups, o_lora) = (64, 16, 2, 32, 1, 16);
    let group_width = heads / o_groups * head_dim;
    DeepseekV41AttentionWeights {
        wq_a: rng.vector(q_lora * hidden, 0.5),
        q_norm: (0..q_lora).map(|i| 1.0 + 0.2 * (i as f32).sin()).collect(),
        wq_b: rng.vector(heads * head_dim * q_lora, 0.5),
        wkv: rng.vector(head_dim * hidden, 0.5),
        kv_norm: (0..head_dim)
            .map(|i| 1.0 + 0.2 * (i as f32).cos())
            .collect(),
        wo_a: rng.vector(o_groups * o_lora * group_width, 0.5),
        wo_b: rng.vector(hidden * o_groups * o_lora, 0.5),
        attn_sink: rng.vector(heads, 2.0),
    }
}

fn build_attention(device: crate::VulkanDevice, ratio: usize) -> Result<DeepseekV41Attention> {
    build_attention_with_sink(device, ratio, attention_weights(7).attn_sink)
}

fn build_attention_with_sink(
    device: crate::VulkanDevice,
    ratio: usize,
    sink: Vec<f32>,
) -> Result<DeepseekV41Attention> {
    let mut weights = attention_weights(7);
    weights.attn_sink = sink;
    DeepseekV41Attention::new(
        device,
        DeepseekV41AttentionConfig {
            heads: 2,
            head_dim: 32,
            hidden_size: 64,
            q_lora_rank: 16,
            o_groups: 1,
            o_lora_rank: 16,
            window_size: 4,
            compress_ratio: ratio,
            norm_eps: 1e-20,
            rotary: rotary_config(16),
        },
        weights,
        2,
    )
}

/// The released `get_window_topk_idxs`: each prefill query sees its own causal
/// window inside the chunk; a decode step sees the whole ring, oldest first.
#[test]
fn deepseek_v41_window_indices_match_the_released_ring_rules() -> Result<()> {
    let (indices, topk) = deepseek_v41_window_indices(1, 6, 0, 4)?;
    assert_eq!(topk, 4);
    assert_eq!(
        indices,
        vec![0, -1, -1, -1, 0, 1, -1, -1, 0, 1, 2, -1, 0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5]
    );

    let (short, topk) = deepseek_v41_window_indices(2, 3, 0, 3)?;
    assert_eq!(topk, 3);
    assert_eq!(&short[..9], &[0, -1, -1, 0, 1, -1, 0, 1, 2]);
    assert_eq!(short.len(), 18);
    assert_eq!(
        &short[9..],
        &short[..9],
        "each batch item restarts the window"
    );

    let (decode, topk) = deepseek_v41_window_indices(1, 1, 6, 4)?;
    assert_eq!(topk, 4);
    assert_eq!(decode, vec![3, 0, 1, 2]);

    // A ring that is still filling lists the newest positions last and masks
    // slots the sequence has not reached yet.
    let (filling, topk) = deepseek_v41_window_indices(1, 1, 2, 4)?;
    assert_eq!(topk, 4);
    assert_eq!(filling, vec![-1, 0, 1, 2]);

    assert!(deepseek_v41_window_indices(0, 4, 0, 4).is_err());
    assert!(deepseek_v41_window_indices(1, 0, 0, 4).is_err());
    assert!(deepseek_v41_window_indices(1, 4, 0, 0).is_err());
    Ok(())
}

/// The ring shader must place each token in the slot the index function names.
#[test]
fn deepseek_v41_window_cache_ring_matches_the_index_order() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let cache = DeepseekV41WindowKvCache::new(device.clone(), 2, 4, 2)?;
    assert_eq!((cache.batch(), cache.window(), cache.head_dim()), (2, 4, 2));

    let rows = 6;
    let source: Vec<f32> = (0..2)
        .flat_map(|batch| {
            (0..rows).flat_map(move |token| {
                [
                    batch as f32 * 1000.0 + token as f32 * 10.0,
                    batch as f32 * 1000.0 + 1.0,
                ]
            })
        })
        .collect();
    let source = GpuBuffer::from_f32(&device, &source)?;
    cache.seed_prefill(&source, rows)?;

    let ring = cache.read_all()?;
    for batch in 0..2 {
        let slot = |index: usize, dim: usize| ring[(batch * 4 + index) * 2 + dim];
        // cutoff = 6 % 4 = 2: the newest four tokens are tokens 2..6.
        assert_eq!(slot(2, 0), batch as f32 * 1000.0 + 20.0);
        assert_eq!(slot(3, 0), batch as f32 * 1000.0 + 30.0);
        assert_eq!(slot(0, 0), batch as f32 * 1000.0 + 40.0);
        assert_eq!(slot(1, 0), batch as f32 * 1000.0 + 50.0);
        assert_eq!(slot(3, 1), batch as f32 * 1000.0 + 1.0);
    }

    // Position 6 writes slot 6 % 4 = 2; the other slots are preserved.
    let token: Vec<f32> = (0..2)
        .flat_map(|batch| [batch as f32 * 1000.0 + 60.0, 1.0])
        .collect();
    let token = GpuBuffer::from_f32(&device, &token)?;
    cache.write_decode(&token, 6)?;
    let ring = cache.read_all()?;
    for batch in 0..2 {
        let slot = |index: usize| ring[(batch * 4 + index) * 2];
        assert_eq!(slot(2), batch as f32 * 1000.0 + 60.0);
        assert_eq!(slot(0), batch as f32 * 1000.0 + 40.0);
        assert_eq!(slot(1), batch as f32 * 1000.0 + 50.0);
        assert_eq!(slot(3), batch as f32 * 1000.0 + 30.0);
    }

    // A chunk that fits leaves the trailing slots untouched.
    let short = DeepseekV41WindowKvCache::new(device.clone(), 1, 4, 2)?;
    let source: Vec<f32> = (0..3)
        .flat_map(|token| [token as f32 + 1.0, -1.0])
        .collect();
    let source = GpuBuffer::from_f32(&device, &source)?;
    short.seed_prefill(&source, 3)?;
    let ring = short.read_all()?;
    assert_eq!(&ring[..2], &[1.0, -1.0]);
    assert_eq!(&ring[4..6], &[3.0, -1.0]);
    assert_eq!(&ring[6..], &[0.0, 0.0], "slots past the chunk stay empty");
    Ok(())
}

fn compressor_weights(seed: u64, ratio: usize) -> DeepseekV41CompressorWeights {
    let (hidden, head_dim) = (64, 32);
    let mut rng = Lcg(seed);
    DeepseekV41CompressorWeights {
        hidden_size: hidden,
        head_dim,
        compress_ratio: ratio,
        eps: 1e-20,
        kv_weight: rng.vector(head_dim * hidden, 0.5),
        gate_weight: (ratio == 2).then(|| rng.vector(head_dim * hidden, 0.5)),
        norm_weight: (0..head_dim)
            .map(|i| 1.0 + 0.1 * (i as f32).sin())
            .collect(),
    }
}

fn indexer_weights(seed: u64) -> DeepseekV41IndexerWeights {
    let (hidden, head_dim, index_head_dim, heads, q_lora) = (64, 32, 32, 2, 16);
    let mut rng = Lcg(seed);
    DeepseekV41IndexerWeights {
        wk: rng.vector(index_head_dim * head_dim, 0.5),
        k_norm: (0..index_head_dim)
            .map(|i| 1.0 + 0.1 * (i as f32).cos())
            .collect(),
        wq_b: rng.vector(heads * index_head_dim * q_lora, 0.5),
        weights_proj: rng.vector(heads * hidden, 0.5),
    }
}

fn indexer(device: crate::VulkanDevice, ratio: usize) -> Result<DeepseekV41Indexer> {
    DeepseekV41Indexer::new(
        device,
        DeepseekV41IndexerConfig {
            heads: 2,
            index_head_dim: 32,
            head_dim: 32,
            hidden_size: 64,
            q_lora_rank: 16,
            compress_ratio: ratio,
            index_topk: 3,
            owns_k: true,
            is_candidate_source: false,
            candidate_topk_blocks: 0,
            candidate_block_size: 0,
            norm_eps: 1e-20,
            rotary: rotary_config(16),
        },
        indexer_weights(29),
        2,
        32,
    )
}

fn widen(values: &[f32], start: usize, len: usize) -> Vec<f32> {
    values[start..start + len].to_vec()
}

/// Gathered attention must match the released `sparse_attn` arithmetic: the
/// concatenated window/compressed gather, `-1` skipping, the sink as a virtual
/// zero-value slot, and the softmax normalization.
#[test]
fn deepseek_v41_sparse_attention_matches_a_float64_gather_reference() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let attention = build_attention(device.clone(), 2)?;
    let (batch, seq, heads, dim) = (2, 3, 2, 32);
    let (window_rows, compressed_capacity) = (3, 4);
    let (window_topk, compressed_topk) = (3, 2);

    let mut rng = Lcg(11);
    let queries = rng.vector(batch * seq * heads * dim, 1.5);
    let window_kv = rng.vector(batch * window_rows * dim, 1.5);
    let compressed_kv = rng.vector(batch * compressed_capacity * dim, 1.5);
    // Row 5 sees nothing at all; its output must be zero.
    let window_indices: Vec<i32> =
        vec![0, -1, -1, 0, 1, -1, 1, 2, -1, 0, 1, 2, 0, 1, 2, -1, -1, -1];
    let compressed_indices: Vec<i32> =
        vec![-1, -1, 3, 4, 4, -1, 5, 6, -1, 6, 3, 4, 5, -1, -1, -1, -1];
    let sink = attention_weights(7).attn_sink;

    let query_buffer = GpuBuffer::from_f32(&device, &queries)?;
    let window_buffer = GpuBuffer::from_f32(&device, &window_kv)?;
    let compressed_buffer = GpuBuffer::from_f32(&device, &compressed_kv)?;
    let window_index_buffer = GpuBuffer::from_u32(
        &device,
        &window_indices
            .iter()
            .map(|value| *value as u32)
            .collect::<Vec<_>>(),
    )?;
    let compressed_index_buffer = GpuBuffer::from_u32(
        &device,
        &compressed_indices
            .iter()
            .map(|value| *value as u32)
            .collect::<Vec<_>>(),
    )?;
    let output = attention
        .gathered_attention(
            &query_buffer,
            &window_buffer,
            Some(&compressed_buffer),
            &window_index_buffer,
            &compressed_index_buffer,
            DeepseekV41SparseAttentionParams {
                batch,
                seq,
                window_rows,
                window_topk,
                compressed_topk,
                compressed_capacity,
            },
        )?
        .read_f32(batch * seq * heads * dim)?;

    // The reference concatenates the two caches before gathering, so a
    // compressed index is a row of the combined buffer.
    let rows_per_batch = window_rows + compressed_capacity;
    let mut combined = Vec::with_capacity(batch * rows_per_batch * dim);
    for item in 0..batch {
        combined.extend(widen(
            &window_kv,
            item * window_rows * dim,
            window_rows * dim,
        ));
        combined.extend(widen(
            &compressed_kv,
            item * compressed_capacity * dim,
            compressed_capacity * dim,
        ));
    }
    let total = window_topk + compressed_topk;
    let scale = (dim as f64).powf(-0.5);
    let mut worst = 0.0f64;
    for row in 0..batch * seq {
        let item = row / seq;
        for head in 0..heads {
            let mut slots: Vec<(usize, f64)> = Vec::with_capacity(total);
            for slot in 0..window_topk {
                let index = window_indices[row * window_topk + slot];
                if index < 0 || index as usize >= window_rows {
                    continue;
                }
                slots.push((index as usize, 0.0));
            }
            for slot in 0..compressed_topk {
                let index = compressed_indices[row * compressed_topk + slot];
                if index < window_rows as i32
                    || (index - window_rows as i32) as usize >= compressed_capacity
                {
                    continue;
                }
                slots.push((index as usize, 0.0));
            }
            for (kv_row, logit) in slots.iter_mut() {
                let mut dot = 0.0f64;
                for d in 0..dim {
                    dot += queries[(row * heads + head) * dim + d] as f64
                        * combined[(item * rows_per_batch + *kv_row) * dim + d] as f64;
                }
                *logit = dot * scale;
            }
            if slots.is_empty() {
                for d in 0..dim {
                    let actual = output[(row * heads + head) * dim + d];
                    assert_eq!(actual, 0.0, "an empty gather must produce zeros");
                }
                continue;
            }
            let maximum = slots
                .iter()
                .map(|(_, logit)| *logit)
                .fold(f64::NEG_INFINITY, f64::max);
            let mut denominator = (sink[head] as f64 - maximum).exp();
            for (_, logit) in &slots {
                denominator += (logit - maximum).exp();
            }
            for d in 0..dim {
                let mut expected = 0.0f64;
                for (kv_row, logit) in &slots {
                    let weight = (logit - maximum).exp() / denominator;
                    expected +=
                        weight * combined[(item * rows_per_batch + *kv_row) * dim + d] as f64;
                }
                let actual = output[(row * heads + head) * dim + d] as f64;
                let drift = (actual - expected).abs();
                assert!(
                    drift < 2e-6,
                    "row {row} head {head} channel {d}: native={actual:.10e}, reference={expected:.10e}, drift={drift:.3e}"
                );
                worst = worst.max(drift);
            }
        }
    }
    assert!(
        worst > 0.0,
        "the comparison must actually exercise the gather"
    );
    Ok(())
}

/// The pinned fixture is produced by `validation/validate_deepseek_v41_attention.py`
/// from the released `get_window_topk_idxs` and the `sparse_attn` body.
#[test]
fn deepseek_v41_attention_matches_the_pinned_reference_fixture() -> Result<()> {
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Fixture {
        revision: String,
        window_cases: Vec<WindowCase>,
        attention_cases: Vec<GatherCase>,
    }

    #[derive(Deserialize)]
    struct WindowCase {
        name: String,
        batch: usize,
        seq: usize,
        start_pos: usize,
        window: usize,
        topk: usize,
        expected_indices: Vec<i32>,
    }

    #[derive(Deserialize)]
    struct GatherCase {
        name: String,
        batch: usize,
        seq: usize,
        rows: usize,
        heads: usize,
        head_dim: usize,
        window_rows: usize,
        window_topk: usize,
        compressed_topk: usize,
        compressed_capacity: usize,
        atol: f64,
        window_indices: Vec<i32>,
        compressed_indices: Vec<i32>,
        q: Vec<f32>,
        window_kv: Vec<f32>,
        compressed_kv: Vec<f32>,
        sink: Vec<f32>,
        expected: Vec<f64>,
    }

    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_attention.json"
    ))?;
    assert_eq!(fixture.revision, REFERENCE_REVISION);
    let device = crate::VulkanDevice::new()?;

    for case in &fixture.window_cases {
        let (indices, topk) =
            deepseek_v41_window_indices(case.batch, case.seq, case.start_pos, case.window)?;
        assert_eq!(topk, case.topk, "{} window width", case.name);
        assert_eq!(
            indices, case.expected_indices,
            "{} window indices",
            case.name
        );
    }

    let mut worst = 0.0f64;
    let mut compared = 0usize;
    for case in &fixture.attention_cases {
        assert_eq!(case.heads, 2, "{} heads", case.name);
        assert_eq!(case.head_dim, 32, "{} head width", case.name);
        let attention = build_attention_with_sink(device.clone(), 2, case.sink.clone())?;
        let q = GpuBuffer::from_f32(&device, &case.q)?;
        let window_kv = GpuBuffer::from_f32(&device, &case.window_kv)?;
        let compressed_kv = if case.compressed_topk == 0 {
            None
        } else {
            Some(GpuBuffer::from_f32(&device, &case.compressed_kv)?)
        };
        let bits = |values: &[i32]| -> Vec<u32> { values.iter().map(|v| *v as u32).collect() };
        let window_indices = GpuBuffer::from_u32(&device, &bits(&case.window_indices))?;
        // An unused slot range still needs a bound buffer.
        let compressed_bits = match case.compressed_indices.is_empty() {
            true => vec![0u32],
            false => bits(&case.compressed_indices),
        };
        let compressed_indices = GpuBuffer::from_u32(&device, &compressed_bits)?;
        let output = attention.gathered_attention(
            &q,
            &window_kv,
            compressed_kv.as_ref(),
            &window_indices,
            &compressed_indices,
            DeepseekV41SparseAttentionParams {
                batch: case.batch,
                seq: case.seq,
                window_rows: case.window_rows,
                window_topk: case.window_topk,
                compressed_topk: case.compressed_topk,
                compressed_capacity: case.compressed_capacity,
            },
        )?;
        let output = output.read_f32(case.rows * case.heads * case.head_dim)?;
        assert_eq!(
            output.len(),
            case.expected.len(),
            "{} output shape",
            case.name
        );
        for (index, (&actual, &expected)) in output.iter().zip(&case.expected).enumerate() {
            let drift = (actual as f64 - expected).abs();
            assert!(
                drift <= case.atol,
                "{} value {index}: native={actual:.10e}, reference={expected:.10e}, drift={drift:.3e}",
                case.name
            );
            worst = worst.max(drift);
        }
        compared += output.len();
    }
    assert!(compared > 1000 && worst > 0.0);
    println!("attention fixture: {compared} values, worst drift {worst:.3e}");
    Ok(())
}

/// End to end: one compressor latent feeds the shared index keys and the
/// compressed KV cache, the indexer's selection addresses both the window and
/// compressed sources, and prefill + decode consume the same published state.
#[test]
fn deepseek_v41_attention_wires_the_shared_index_selection_and_key_cache() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let (batch, seq, hidden, window, ratio) = (2, 6, 64, 4, 2);
    let mut rng = Lcg(23);

    let mut compressor =
        DeepseekV41Compressor::new(device.clone(), compressor_weights(23, ratio), batch, 32)?;
    let indexer = indexer(device.clone(), ratio)?;
    let finalizer =
        DeepseekV41CompressedKvFinalizer::new(device.clone(), rotary_config(16), ratio, 32)?;
    let compressed = DeepseekV41CompressedKvCache::new(device.clone(), batch, 16, 32)?;
    let mut attention = build_attention(device.clone(), ratio)?;

    let tokens = rng.vector(batch * seq * hidden, 1.0);
    let x = GpuBuffer::from_f32(&device, &tokens)?;
    let qr = attention.query_latent(&x, batch * seq)?;

    let latents = compressor.prefill(&tokens, batch, seq)?;
    assert_eq!(latents.groups_per_batch, 3);
    // Keys read the pre-RoPE latent; the cache write comes after them.
    assert_eq!(indexer.write_index_keys(&latents, 0)?, Some(0));
    let finalized = finalizer.finalize_prefill(&latents)?;
    compressed.append_groups(&finalized, batch, 3, 0)?;

    // Prefill window indices address the chunk, so the shift is `seq`, the
    // released `kv.size(1)` for this step.
    let keys = indexer.index_keys().expect("prefill owns the shared keys");
    let selection = indexer.select(&x, &qr, keys, batch, seq, 0, seq, None)?;
    assert_eq!((selection.width, selection.topk), (3, 3));
    let indices = selection.read_indices()?;
    assert_eq!(indices.len(), batch * seq * selection.topk);
    for row in 0..batch * seq {
        let visible = (row % seq + 1) / ratio;
        for slot in 0..selection.topk {
            let index = indices[row * selection.topk + slot];
            if slot < visible {
                assert!(
                    index >= seq as i32 && index < (seq + 3) as i32,
                    "row {row} slot {slot}: {index} does not address a compressed position"
                );
            } else {
                assert_eq!(index, -1, "row {row} slot {slot} must be causally masked");
            }
        }
    }

    let output = attention.prefill(&x, Some(&compressed), Some(&selection), batch, seq)?;
    let output = output.read_f32(batch * seq * hidden)?;
    assert!(output.iter().all(|value| value.is_finite()));
    assert!(output.iter().any(|value| *value != 0.0));

    // Position 6 completes no group, so the shared keys and the cache stay put
    // and the step is a plain ring write plus the same index selection.
    let step = rng.vector(batch * hidden, 1.0);
    let empty = compressor.decode_step(&step, batch, seq)?;
    assert_eq!(empty.groups_per_batch, 0);
    assert!(indexer.write_index_keys(&empty, seq)?.is_none());
    let x = GpuBuffer::from_f32(&device, &step)?;
    let qr = attention.query_latent(&x, batch)?;
    let decode = indexer.select(&x, &qr, keys, batch, 1, seq, window, None)?;
    assert_eq!((decode.width, decode.topk), (3, 3));
    let decode_indices = decode.read_indices()?;
    // The last prefill query sees the same three groups; only the shift differs.
    for slot in 0..3 {
        assert_eq!(
            decode_indices[slot] - window as i32,
            indices[(seq - 1) * selection.topk + slot] - seq as i32
        );
    }
    let decoded = attention.decode_step(&x, Some(&compressed), Some(&decode), batch, seq)?;
    let decoded = decoded.read_f32(batch * hidden)?;
    assert!(decoded.iter().all(|value| value.is_finite()));
    assert!(decoded.iter().any(|value| *value != 0.0));

    // Position 7 completes group 3, so the step publishes keys and a compressed
    // row before it selects and attends; the width grows to four groups.
    let step = rng.vector(batch * hidden, 1.0);
    let completed = compressor.decode_step(&step, batch, seq + 1)?;
    assert_eq!((completed.first_group, completed.groups_per_batch), (3, 1));
    assert_eq!(indexer.write_index_keys(&completed, seq + 1)?, Some(3));
    let finalized = finalizer.finalize_decode(&completed, seq + 1)?;
    compressed.append_groups(&finalized, batch, 1, 3)?;
    let x = GpuBuffer::from_f32(&device, &step)?;
    let qr = attention.query_latent(&x, batch)?;
    let grown = indexer.select(&x, &qr, keys, batch, 1, seq + 1, window, None)?;
    assert_eq!((grown.width, grown.topk), (4, 3));
    let grown_indices = grown.read_indices()?;
    for slot in &grown_indices {
        assert!(
            *slot == -1 || (*slot >= window as i32 && *slot < (window + 4) as i32),
            "{slot} does not address a compressed position"
        );
    }
    let grown = attention.decode_step(&x, Some(&compressed), Some(&grown), batch, seq + 1)?;
    let grown = grown.read_f32(batch * hidden)?;
    assert!(grown.iter().all(|value| value.is_finite()));
    assert!(grown.iter().any(|value| *value != 0.0));

    // Position discipline and source requirements are enforced, not assumed.
    assert!(attention
        .decode_step(&x, Some(&compressed), Some(&decode), batch, seq)
        .is_err());
    let mut sliding = build_attention(device.clone(), 0)?;
    assert!(sliding
        .prefill(&x, Some(&compressed), None, batch, seq)
        .is_err());
    let mut orphan = build_attention(device.clone(), ratio)?;
    assert!(orphan.prefill(&x, None, None, batch, seq).is_err());
    Ok(())
}
