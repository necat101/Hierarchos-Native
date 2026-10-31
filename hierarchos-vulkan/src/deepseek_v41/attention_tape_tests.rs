use super::*;
use crate::deepseek_v41::{DeepseekV41AttentionTape, DeepseekV41RotaryConfig};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    tolerance: f64,
    rtol: f64,
    comparison: String,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    batch: usize,
    heads: usize,
    dim: usize,
    caches: Vec<CacheSpec>,
    writes: Vec<WriteSpec>,
    reads: Vec<ReadSpec>,
    events: Vec<EventSpec>,
}
#[derive(Deserialize)]
struct CacheSpec {
    kind: String,
    capacity: usize,
    initial_bits: Vec<u32>,
    grad_initial_bits: Option<Vec<u32>>,
}
#[derive(Deserialize)]
struct WriteSpec {
    cache: usize,
    kind: String,
    rows: usize,
    position: usize,
    raw_bits: Vec<u32>,
    source_bits: Vec<u32>,
    scale_bits: Vec<u32>,
    cache_after_bits: Vec<u32>,
    grad_source_bits: Option<Vec<u32>>,
}
#[derive(Deserialize)]
struct ReadSpec {
    layer: usize,
    seq: usize,
    window_cache: usize,
    prefill_write: Option<usize>,
    compressed_cache: Option<usize>,
    window_rows: usize,
    topk: usize,
    compressed_topk: usize,
    wi: Vec<i32>,
    ci: Vec<i32>,
    query_bits: Vec<u32>,
    sink_bits: Vec<u32>,
    upstream_bits: Vec<u32>,
    output_bits: Vec<u32>,
    seed: bool,
    grad_query_bits: Option<Vec<u32>>,
}
#[derive(Deserialize)]
struct EventSpec {
    kind: String,
    index: usize,
}

fn fixture() -> Fixture {
    let f: Fixture = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_attention_tape.json"
    ))
    .unwrap();
    assert_eq!(
        (f.tolerance, f.rtol, f.comparison.as_str()),
        (2e-7, 0.0, "strict_less_than")
    );
    f
}
fn upload(device: &VulkanDevice, bits: &[u32]) -> Result<GpuBuffer> {
    if bits.is_empty() {
        GpuBuffer::zeros_u32(device, 1)
    } else {
        GpuBuffer::from_u32(device, bits)
    }
}
fn index_buffer(device: &VulkanDevice, values: &[i32]) -> Result<GpuBuffer> {
    upload(
        device,
        &values.iter().map(|&n| n as u32).collect::<Vec<_>>(),
    )
}
fn gate(
    label: &str,
    buffer: &GpuBuffer,
    expected: &[u32],
    failures: &mut Vec<String>,
) -> Result<()> {
    let actual = buffer.read_f32(expected.len())?;
    let mut worst = 0.0f64;
    let mut at = 0;
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let b = f32::from_bits(b);
        assert!(a.is_finite() && b.is_finite(), "{label} nonfinite at {i}");
        let drift = (a as f64 - b as f64).abs();
        if drift > worst {
            worst = drift;
            at = i;
        }
    }
    eprintln!(
        "V4.1 attention tape {label}: {} values, worst {worst:.17e} at {at}",
        expected.len()
    );
    if worst >= 2e-7 {
        failures.push(format!("{label}: {worst:.17e} at {at}"));
    }
    Ok(())
}
fn optional_gate(
    label: &str,
    buffer: Option<&GpuBuffer>,
    expected: &Option<Vec<u32>>,
    failures: &mut Vec<String>,
) -> Result<()> {
    match (buffer, expected) {
        (Some(a), Some(b)) => gate(label, a, b, failures),
        (None, None) => Ok(()),
        _ => panic!("{label} absent-gradient mismatch"),
    }
}
fn attention(
    device: &VulkanDevice,
    batch: usize,
    heads: usize,
    dim: usize,
    sink: &[u32],
) -> Result<DeepseekV41Attention> {
    let c = DeepseekV41AttentionConfig {
        heads,
        head_dim: dim,
        hidden_size: 8,
        q_lora_rank: 4,
        o_groups: 1,
        o_lora_rank: 4,
        window_size: 4,
        compress_ratio: 2,
        norm_eps: 1e-6,
        rotary: DeepseekV41RotaryConfig::sliding(16, 64, 10000.0),
    };
    DeepseekV41Attention::new(
        device.clone(),
        c,
        DeepseekV41AttentionWeights {
            wq_a: vec![0.0; 4 * 8],
            q_norm: vec![1.0; 4],
            wq_b: vec![0.0; heads * dim * 4],
            wkv: vec![0.0; dim * 8],
            kv_norm: vec![1.0; dim],
            wo_a: vec![0.0; 4 * heads * dim],
            wo_b: vec![0.0; 8 * 4],
            attn_sink: sink.iter().map(|&n| f32::from_bits(n)).collect(),
        },
        batch,
    )
}
enum NativeCache {
    Window(DeepseekV41WindowKvCache),
    Compressed(DeepseekV41CompressedKvCache),
}
impl NativeCache {
    fn values(&self) -> &GpuBuffer {
        match self {
            Self::Window(c) => c.values(),
            Self::Compressed(c) => c.values(),
        }
    }
}

#[test]
fn attention_tape_shared_sources_and_decode_chain_match_cpu_autograd() -> Result<()> {
    let device = VulkanDevice::new()?;
    let quantizer = DeepseekV41ActivationQuantizer::new(device.clone())?;
    let mut failures = Vec::new();
    for c in fixture().cases {
        let caches = c
            .caches
            .iter()
            .map(|s| {
                let cache = if s.kind == "window" {
                    NativeCache::Window(DeepseekV41WindowKvCache::new(
                        device.clone(),
                        c.batch,
                        s.capacity,
                        c.dim,
                    )?)
                } else {
                    NativeCache::Compressed(DeepseekV41CompressedKvCache::new(
                        device.clone(),
                        c.batch,
                        s.capacity,
                        c.dim,
                    )?)
                };
                cache.values().write_u32(&s.initial_bits)?;
                Ok(cache)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut tape = DeepseekV41AttentionTape::new(device.clone());
        let ids = caches
            .iter()
            .map(|cache| match cache {
                NativeCache::Window(cache) => tape.register_window(cache),
                NativeCache::Compressed(cache) => tape.register_compressed(cache),
            })
            .collect::<Result<Vec<_>>>()?;
        let mut writes = Vec::new();
        let mut reads = Vec::new();
        let mut seeds = Vec::new();
        let mut expected_bytes = 0;
        for event in &c.events {
            if event.kind == "write" {
                let s = &c.writes[event.index];
                assert_eq!(writes.len(), event.index);
                let raw = upload(&device, &s.raw_bits)?;
                let (block, mode, scale) = if s.kind == "append" {
                    (
                        16,
                        DeepseekV41QuantMode::Fp4E2M1,
                        DeepseekV41ScaleMode::E4m3,
                    )
                } else {
                    (
                        32,
                        DeepseekV41QuantMode::Fp8E4M3,
                        DeepseekV41ScaleMode::Ue8m0,
                    )
                };
                let activation =
                    quantizer.quantize_buffer(&raw, c.batch * s.rows, c.dim, block, mode, scale)?;
                // The forward takes actual native FP8/FP4 quantizer output,
                // rather than uploading a fabricated quantized activation.
                assert_eq!(
                    activation.values.read_u32(s.source_bits.len())?,
                    s.source_bits
                );
                assert_eq!(
                    activation.scales.read_u32(s.scale_bits.len())?,
                    s.scale_bits
                );
                let id = match s.kind.as_str() {
                    "seed" => {
                        expected_bytes += s.source_bits.len() * 4;
                        tape.seed_window(&ids[s.cache], &activation.values, s.rows)?
                    }
                    "decode" => tape.write_window(&ids[s.cache], &activation.values, s.position)?,
                    "append" => {
                        tape.append_compressed(&ids[s.cache], &activation, s.rows, s.position)?
                    }
                    _ => unreachable!(),
                };
                assert_eq!(
                    caches[s.cache]
                        .values()
                        .read_u32(s.cache_after_bits.len())?,
                    s.cache_after_bits,
                    "{} native cache write {}",
                    c.name,
                    event.index
                );
                // Source scratch can be recycled as soon as the write returns.
                activation
                    .values
                    .write_f32(&vec![17.0; s.source_bits.len()])?;
                writes.push(id);
            } else {
                let s = &c.reads[event.index];
                assert_eq!(reads.len(), event.index);
                let attention = attention(&device, c.batch, c.heads, c.dim, &s.sink_bits)?;
                let q = upload(&device, &s.query_bits)?;
                let wi = index_buffer(&device, &s.wi)?;
                let ci = index_buffer(&device, &s.ci)?;
                let view = s.prefill_write.map_or_else(
                    || DeepseekV41AttentionWindowView::Cache(ids[s.window_cache].clone()),
                    |n| DeepseekV41AttentionWindowView::Prefill(writes[n].clone()),
                );
                let read = tape.attend(
                    &attention,
                    &q,
                    view,
                    s.compressed_cache.map(|n| &ids[n]),
                    &wi,
                    &ci,
                    DeepseekV41SparseAttentionParams {
                        batch: c.batch,
                        seq: s.seq,
                        window_rows: s.window_rows,
                        window_topk: s.topk,
                        compressed_topk: s.compressed_topk,
                        compressed_capacity: s.compressed_cache.map_or(0, |n| c.caches[n].capacity),
                    },
                )?;
                gate(
                    &format!("{} read {} layer {} forward", c.name, event.index, s.layer),
                    &read.output,
                    &s.output_bits,
                    &mut failures,
                )?;
                if s.seed {
                    seeds.push((read.id.clone(), upload(&device, &s.upstream_bits)?));
                }
                reads.push(read.id);
                expected_bytes += 4
                    * (s.query_bits.len()
                        + c.batch * s.window_rows * c.dim
                        + s.compressed_cache
                            .map_or(0, |n| c.batch * c.caches[n].capacity * c.dim)
                        + s.wi.len()
                        + s.ci.len().max(1)
                        + c.heads);
                assert_eq!(tape.retained_bytes(), expected_bytes);
                // Mutate caller scratch and the frozen sink, then drop all
                // projection allocations. Backward owns no aliases to them.
                q.write_f32(&vec![17.0; s.query_bits.len()])?;
                wi.write_u32(&vec![u32::MAX; s.wi.len()])?;
                if !s.ci.is_empty() {
                    ci.write_u32(&vec![u32::MAX; s.ci.len()])?;
                }
                attention
                    .attention_sink()?
                    .write_f32(&vec![17.0; c.heads])?;
            }
        }
        assert_eq!(tape.read_count(), c.reads.len());
        assert_eq!(tape.write_count(), c.writes.len());
        eprintln!(
            "V4.1 attention tape {} retained snapshot bytes {expected_bytes}",
            c.name
        );
        let gradients = tape.backward(&seeds)?;
        for (i, s) in c.reads.iter().enumerate() {
            optional_gate(
                &format!("{} read {i} query", c.name),
                gradients.query(&reads[i])?,
                &s.grad_query_bits,
                &mut failures,
            )?;
        }
        for (i, s) in c.writes.iter().enumerate() {
            optional_gate(
                &format!("{} write {i} source", c.name),
                gradients.source(&writes[i])?,
                &s.grad_source_bits,
                &mut failures,
            )?;
        }
        for (i, s) in c.caches.iter().enumerate() {
            optional_gate(
                &format!("{} initial cache {i}", c.name),
                gradients.initial_cache(&ids[i])?,
                &s.grad_initial_bits,
                &mut failures,
            )?;
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn attention_tape_overwrite_and_absent_seed_ownership_is_exact() -> Result<()> {
    let device = VulkanDevice::new()?;
    let attention = attention(&device, 1, 1, 32, &[0])?;
    for seed in [None, Some(0.0), Some(1.0)] {
        let cache = DeepseekV41WindowKvCache::new(device.clone(), 1, 3, 32)?;
        let mut tape = DeepseekV41AttentionTape::new(device.clone());
        let id = tape.register_window(&cache)?;
        let source = GpuBuffer::from_f32(&device, &vec![0.25; 5 * 32])?;
        let write = tape.seed_window(&id, &source, 5)?;
        let q = GpuBuffer::zeros_f32(&device, 32)?;
        let indices = index_buffer(&device, &[0, 1, 2])?;
        let empty = GpuBuffer::zeros_u32(&device, 1)?;
        let read = tape.attend(
            &attention,
            &q,
            DeepseekV41AttentionWindowView::Cache(id.clone()),
            None,
            &indices,
            &empty,
            DeepseekV41SparseAttentionParams {
                batch: 1,
                seq: 1,
                window_rows: 3,
                window_topk: 3,
                compressed_topk: 0,
                compressed_capacity: 0,
            },
        )?;
        let seeds = match seed {
            Some(g) => vec![(read.id.clone(), GpuBuffer::from_f32(&device, &vec![g; 32])?)],
            None => Vec::new(),
        };
        let gradients = tape.backward(&seeds)?;
        if let Some(g) = seed {
            let mut expected = vec![0.0; 2 * 32];
            expected.extend(vec![g * 0.25; 3 * 32]);
            assert_eq!(
                gradients.source(&write)?.unwrap().read_f32(5 * 32)?,
                expected
            );
            assert_eq!(
                gradients.initial_cache(&id)?.unwrap().read_f32(3 * 32)?,
                vec![0.0; 3 * 32]
            );
            assert!(gradients.query(&read.id)?.is_some());
        } else {
            assert!(gradients.source(&write)?.is_none());
            assert!(gradients.initial_cache(&id)?.is_none());
            assert!(gradients.query(&read.id)?.is_none());
        }
    }
    Ok(())
}

#[test]
fn attention_tape_rejects_aliases_foreign_handles_and_bad_geometry() -> Result<()> {
    let device = VulkanDevice::new()?;
    let window = DeepseekV41WindowKvCache::new(device.clone(), 1, 4, 32)?;
    let compressed = DeepseekV41CompressedKvCache::new(device.clone(), 1, 3, 32)?;
    let mut tape = DeepseekV41AttentionTape::new(device.clone());
    let w = tape.register_window(&window)?;
    let c = tape.register_compressed(&compressed)?;
    assert!(tape.register_window(&window).is_err());
    assert!(tape.register_compressed(&compressed).is_err());
    let mut foreign = DeepseekV41AttentionTape::new(device.clone());
    let foreign_id = foreign.register_window(&window)?;
    let source = GpuBuffer::zeros_f32(&device, 32)?;
    for result in [
        tape.seed_window(&foreign_id, &source, 1),
        tape.seed_window(&c, &source, 1),
        tape.seed_window(&w, &source, 0),
        tape.seed_window(&w, &source, 2),
        tape.write_window(&c, &source, 1),
        tape.write_window(&w, window.values(), 1),
        tape.write_window(&w, &source, usize::MAX),
    ] {
        assert!(result.is_err());
    }
    let activation = DeepseekV41QuantizedActivation {
        rows: 1,
        cols: 32,
        block_size: 16,
        mode: DeepseekV41QuantMode::Fp4E2M1,
        scale_mode: DeepseekV41ScaleMode::E4m3,
        values: source,
        scales: GpuBuffer::zeros_f32(&device, 2)?,
    };
    assert!(tape.append_compressed(&w, &activation, 1, 0).is_err());
    assert!(tape
        .append_compressed(&c, &activation, 1, usize::MAX)
        .is_err());
    assert!(tape.append_compressed(&c, &activation, 1, 3).is_err());
    assert!(tape.append_compressed(&c, &activation, 0, 0).is_err());
    assert_eq!(
        (tape.read_count(), tape.write_count(), tape.retained_bytes()),
        (0, 0, 0)
    );
    assert_eq!(window.read_all()?, vec![0.0; 4 * 32]);
    assert_eq!(compressed.read_batch(0, 3)?, vec![0.0; 3 * 32]);
    assert!(size(&[usize::MAX, 2]).is_err());
    assert!(size(&[0, 1]).is_err());
    assert!(size(&[u32::MAX as usize, 2]).is_err());
    Ok(())
}

#[test]
fn attention_tape_invalid_reads_and_backward_seeds_are_rejected() -> Result<()> {
    let device = VulkanDevice::new()?;
    let attention = attention(&device, 1, 1, 32, &[0])?;
    let window = DeepseekV41WindowKvCache::new(device.clone(), 1, 4, 32)?;
    let compressed = DeepseekV41CompressedKvCache::new(device.clone(), 1, 4, 32)?;
    let q = GpuBuffer::zeros_f32(&device, 32)?;
    let wi = index_buffer(&device, &[0, 1, 2, 3])?;
    let ci = GpuBuffer::zeros_u32(&device, 1)?;
    let params = DeepseekV41SparseAttentionParams {
        batch: 1,
        seq: 1,
        window_rows: 4,
        window_topk: 4,
        compressed_topk: 0,
        compressed_capacity: 0,
    };
    for mode in 0..4 {
        let mut tape = DeepseekV41AttentionTape::new(device.clone());
        let w = tape.register_window(&window)?;
        let c = tape.register_compressed(&compressed)?;
        let mut foreign = DeepseekV41AttentionTape::new(device.clone());
        let foreign_id = foreign.register_window(&window)?;
        for view in [
            DeepseekV41AttentionWindowView::Cache(foreign_id.clone()),
            DeepseekV41AttentionWindowView::Cache(c.clone()),
        ] {
            assert!(tape
                .attend(&attention, &q, view, None, &wi, &ci, params)
                .is_err());
        }
        let view = DeepseekV41AttentionWindowView::Cache(w.clone());
        assert!(tape
            .attend(&attention, &q, view.clone(), Some(&w), &wi, &ci, params)
            .is_err());
        assert!(tape
            .attend(
                &attention,
                &q,
                view.clone(),
                None,
                &wi,
                &ci,
                DeepseekV41SparseAttentionParams {
                    window_topk: 0,
                    ..params
                }
            )
            .is_err());
        assert!(tape
            .attend(
                &attention,
                &q,
                view.clone(),
                None,
                &wi,
                &ci,
                DeepseekV41SparseAttentionParams {
                    compressed_capacity: 4,
                    ..params
                }
            )
            .is_err());
        assert!(tape
            .attend(&attention, &q, view.clone(), None, &ci, &ci, params)
            .is_err());
        assert_eq!(
            (tape.read_count(), tape.write_count(), tape.retained_bytes()),
            (0, 0, 0)
        );
        let read = tape.attend(&attention, &q, view, None, &wi, &ci, params)?;
        let seed = GpuBuffer::zeros_f32(&device, 32)?;
        if mode == 3 {
            let gradients = tape.backward(&[(read.id.clone(), seed)])?;
            assert!(gradients.initial_cache(&foreign_id).is_err());
            let foreign_read = DeepseekV41AttentionReadId {
                owner: foreign_id.owner,
                index: 0,
            };
            assert!(gradients.query(&foreign_read).is_err());
        } else {
            let seeds = match mode {
                0 => vec![(read.id.clone(), seed.clone()), (read.id, seed)],
                1 => vec![(
                    DeepseekV41AttentionReadId {
                        owner: Arc::new(()),
                        index: 0,
                    },
                    seed,
                )],
                2 => vec![(read.id, GpuBuffer::zeros_f32(&device, 1)?)],
                _ => unreachable!(),
            };
            assert!(tape.backward(&seeds).is_err());
        }
        assert_eq!(window.read_all()?, vec![0.0; 4 * 32]);
    }
    Ok(())
}
