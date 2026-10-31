use super::*;
use safetensors::{
    serialize_to_file,
    tensor::{Dtype, TensorView},
};
use serde_json::{json, Value};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

const RATIOS: [usize; 8] = [0, 2, 2, 2, 1, 1, 1, 1];
const KV_SOURCES: [usize; 3] = [1, 3, 4];
const INDEX_SOURCES: [usize; 4] = [1, 3, 4, 6];

fn config() -> Result<DeepseekV41Config> {
    let mut value: Value = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_flash_1017_config.json"
    ))?;
    let patch = json!({
        "hidden_size":64, "head_dim":32, "qk_rope_head_dim":16,
        "q_lora_rank":16, "num_attention_heads":2, "o_groups":1, "o_lora_rank":16,
        "index_n_heads":2, "index_head_dim":32, "index_topk":2,
        "sliding_window":4, "max_position_embeddings":32,
        "num_hidden_layers":8, "num_nextn_predict_layers":0,
        "compress_ratios":RATIOS, "kv_source_layer_ids":KV_SOURCES,
        "index_source_layer_ids":INDEX_SOURCES, "candidate_source_layer_id":4,
        "candidate_block_size":2, "candidate_topk_blocks":2,
        "engram_layer_ids":[], "engram_num_embeddings":[],
        "dspark_block_size":0, "dspark_target_layer_ids":[],
        "rope_scaling":{"rope_type":"yarn", "factor":2.0, "beta_fast":32.0,
                        "beta_slow":1.0, "original_max_position_embeddings":16}
    });
    value["text_config"]
        .as_object_mut()
        .unwrap()
        .extend(patch.as_object().unwrap().clone());
    DeepseekV41Config::from_hf_value(&value)
}

fn runtime() -> DeepseekV41AttentionRuntimeConfig {
    DeepseekV41AttentionRuntimeConfig {
        batch_size: 2,
        max_prefill: 12,
        max_positions: 32,
    }
}

struct Lcg(u64);
impl Lcg {
    fn vector(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((self.0 >> 33) as f32 / 2_147_483_648.0 - 0.5) * 0.5
            })
            .collect()
    }
}

fn weights(layer: usize) -> DeepseekV41AttentionLayerWeights {
    let mut rng = Lcg(211 + layer as u64 * 31);
    DeepseekV41AttentionLayerWeights {
        attention: DeepseekV41AttentionWeights {
            wq_a: rng.vector(16 * 64),
            q_norm: vec![1.1; 16],
            wq_b: rng.vector(2 * 32 * 16),
            wkv: rng.vector(32 * 64),
            kv_norm: vec![0.9; 32],
            wo_a: rng.vector(16 * 64),
            wo_b: rng.vector(64 * 16),
            attn_sink: rng.vector(2),
        },
        compressor: KV_SOURCES
            .contains(&layer)
            .then(|| DeepseekV41CompressorWeights {
                hidden_size: 64,
                head_dim: 32,
                compress_ratio: RATIOS[layer],
                eps: 1e-20,
                kv_weight: rng.vector(32 * 64),
                gate_weight: (RATIOS[layer] == 2).then(|| rng.vector(32 * 64)),
                norm_weight: vec![1.05; 32],
            }),
        indexer: INDEX_SOURCES
            .contains(&layer)
            .then(|| DeepseekV41IndexerWeights {
                wk: if KV_SOURCES.contains(&layer) {
                    rng.vector(32 * 32)
                } else {
                    vec![]
                },
                k_norm: if KV_SOURCES.contains(&layer) {
                    vec![0.95; 32]
                } else {
                    vec![]
                },
                wq_b: rng.vector(2 * 32 * 16),
                weights_proj: rng.vector(2 * 64),
            }),
    }
}

fn all_weights() -> Vec<DeepseekV41AttentionLayerWeights> {
    (0..8).map(weights).collect()
}

fn rotary(ratio: usize) -> DeepseekV41RotaryConfig {
    if ratio == 0 {
        DeepseekV41RotaryConfig::sliding(16, 32, 10_000.0)
    } else {
        DeepseekV41RotaryConfig::compressed(16, 32, 160_000.0, 16, 2.0, 32.0, 1.0)
    }
}

/// An explicit, latest-publication driver of the standalone components. It does
/// not consume the orchestrator or its plan/offset/state-machine implementation.
struct Manual {
    layers: Vec<Layer>,
    keys: Vec<Option<DeepseekV41IndexKeyCache>>,
    caches: Vec<Option<DeepseekV41CompressedKvCache>>,
    selections: Vec<Option<DeepseekV41IndexSelection>>,
}

impl Manual {
    fn new(device: &VulkanDevice) -> Result<Self> {
        let mut result = Self {
            layers: vec![],
            keys: vec![],
            caches: vec![],
            selections: vec![None; 8],
        };
        for (id, ratio) in RATIOS.into_iter().enumerate() {
            let w = weights(id);
            let attention = DeepseekV41Attention::new(
                device.clone(),
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
                    rotary: rotary(ratio),
                },
                w.attention,
                2,
            )?;
            let compressor = w
                .compressor
                .map(|w| DeepseekV41Compressor::new(device.clone(), w, 2, 12))
                .transpose()?;
            let indexer = w
                .indexer
                .map(|w| {
                    DeepseekV41Indexer::new(
                        device.clone(),
                        DeepseekV41IndexerConfig {
                            heads: 2,
                            index_head_dim: 32,
                            head_dim: 32,
                            hidden_size: 64,
                            q_lora_rank: 16,
                            compress_ratio: ratio,
                            index_topk: 2,
                            owns_k: KV_SOURCES.contains(&id),
                            is_candidate_source: id == 4,
                            candidate_topk_blocks: 2,
                            candidate_block_size: 2,
                            norm_eps: 1e-20,
                            rotary: rotary(ratio),
                        },
                        w,
                        2,
                        32,
                    )
                })
                .transpose()?;
            result
                .keys
                .push(indexer.as_ref().and_then(|i| i.index_keys()).cloned());
            result.caches.push(if KV_SOURCES.contains(&id) {
                Some(DeepseekV41CompressedKvCache::new(
                    device.clone(),
                    2,
                    32_usize.div_ceil(ratio),
                    32,
                )?)
            } else {
                None
            });
            let finalizer = if KV_SOURCES.contains(&id) {
                Some(DeepseekV41CompressedKvFinalizer::new(
                    device.clone(),
                    rotary(ratio),
                    ratio,
                    32,
                )?)
            } else {
                None
            };
            result.layers.push(Layer {
                attention,
                compressor,
                indexer,
                finalizer,
            });
        }
        Ok(result)
    }

    fn run(&mut self, id: usize, x: &GpuBuffer, seq: usize, pos: usize) -> Result<GpuBuffer> {
        let layer = &mut self.layers[id];
        let qr = layer.attention.query_latent(x, 2 * seq)?;
        if let Some(c) = &mut layer.compressor {
            let latent = if pos == 0 {
                c.prefill_from_buffer(x, 2, seq)?
            } else {
                c.decode_step_from_buffer(x, 2, pos)?
            };
            layer
                .indexer
                .as_ref()
                .unwrap()
                .write_index_keys(&latent, pos)?;
            if latent.groups_per_batch > 0 {
                let f = layer.finalizer.as_ref().unwrap();
                let kv = if pos == 0 {
                    f.finalize_prefill(&latent)?
                } else {
                    f.finalize_decode(&latent, pos)?
                };
                self.caches[id].as_ref().unwrap().append_groups(
                    &kv,
                    2,
                    latent.groups_per_batch,
                    latent.first_group,
                )?;
            }
        }
        let kv_id = KV_SOURCES
            .iter()
            .copied()
            .filter(|&source| source <= id)
            .next_back();
        let index_id = INDEX_SOURCES
            .iter()
            .copied()
            .filter(|&source| source <= id)
            .next_back();
        if let Some(indexer) = &layer.indexer {
            let candidates = if id > 4 {
                self.selections[4]
                    .as_ref()
                    .and_then(|s| s.candidates.as_ref())
            } else {
                None
            };
            self.selections[id] = Some(indexer.select(
                x,
                &qr,
                self.keys[kv_id.unwrap()].as_ref().unwrap(),
                2,
                seq,
                pos,
                if pos == 0 { seq } else { 4 },
                candidates,
            )?);
        }
        let cache = if RATIOS[id] > 0 {
            self.caches[kv_id.unwrap()].as_ref()
        } else {
            None
        };
        let selection = if RATIOS[id] > 0 {
            self.selections[index_id.unwrap()].as_ref()
        } else {
            None
        };
        if pos == 0 {
            layer.attention.prefill(x, cache, selection, 2, seq)
        } else {
            layer.attention.decode_step(x, cache, selection, 2, pos)
        }
    }
}

fn input(device: &VulkanDevice, id: usize, seq: usize, pos: usize) -> Result<GpuBuffer> {
    GpuBuffer::from_f32(
        device,
        &Lcg(991 + 13 * id as u64 + 711 * pos as u64).vector(2 * seq * 64),
    )
}

fn exact(a: &[f32], b: &[f32]) {
    assert_eq!(
        a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}

fn compare_step(
    device: &VulkanDevice,
    graph: &mut DeepseekV41AttentionOrchestrator,
    manual: &mut Manual,
    seq: usize,
    pos: usize,
) -> Result<()> {
    if pos == 0 {
        graph.begin_prefill(seq)?;
    } else {
        graph.begin_decode(pos)?;
    }
    assert!(
        graph.selection(7)?.is_none(),
        "previous-step publications must be invalidated"
    );
    assert!(graph.compressed_cache(7)?.is_none());
    for id in 0..8 {
        assert_eq!(graph.next_layer(), Some(id));
        assert_eq!(graph.next_position(), None);
        let x = input(device, id, seq, pos)?;
        let expected = manual.run(id, &x, seq, pos)?;
        let actual = graph.forward_layer(id, &x)?;
        exact(
            &actual.read_f32(2 * seq * 64)?,
            &expected.read_f32(2 * seq * 64)?,
        );
        if let Some(selection) = graph.selection(id)? {
            let source = INDEX_SOURCES
                .iter()
                .copied()
                .filter(|&source| source <= id)
                .next_back()
                .unwrap();
            let reference = manual.selections[source].as_ref().unwrap();
            assert_eq!(selection.read_indices()?, reference.read_indices()?);
            exact(&selection.read_scores()?, &reference.read_scores()?);
            assert_eq!(selection.read_candidates()?, reference.read_candidates()?);
            assert!(
                std::ptr::eq(selection, graph.selection(source)?.unwrap()),
                "consumers must share the selection"
            );
        }
        if RATIOS[id] > 0 {
            let source = KV_SOURCES
                .iter()
                .copied()
                .filter(|&source| source <= id)
                .next_back()
                .unwrap();
            let width = (pos + seq) / RATIOS[id];
            let cache = graph.compressed_cache(id)?.unwrap();
            assert!(cache
                .values()
                .shares_allocation_with(graph.compressed_cache(source)?.unwrap().values()));
            for batch in 0..2 {
                exact(
                    &cache.read_batch(batch, width)?,
                    &manual.caches[source]
                        .as_ref()
                        .unwrap()
                        .read_batch(batch, width)?,
                );
                exact(
                    &graph.index_keys(id)?.unwrap().read_batch(batch, width)?,
                    &manual.keys[source]
                        .as_ref()
                        .unwrap()
                        .read_batch(batch, width)?,
                );
            }
        }
    }
    assert_eq!(graph.next_layer(), None);
    assert_eq!(graph.next_position(), Some(pos + seq));
    Ok(())
}

#[test]
fn deepseek_v41_orchestrator_matches_explicit_shared_layer_execution_and_reset() -> Result<()> {
    let device = VulkanDevice::new()?;
    let mut graph = DeepseekV41AttentionOrchestrator::new(
        device.clone(),
        &config()?,
        runtime(),
        all_weights(),
    )?;
    let mut manual = Manual::new(&device)?;
    // Odd prefill, group completion, incomplete group, ring wraps, candidate
    // consumers, copy indexers and two batches all go through real dispatches.
    compare_step(&device, &mut graph, &mut manual, 7, 0)?;
    assert!(
        !graph
            .compressed_cache(1)?
            .unwrap()
            .values()
            .shares_allocation_with(graph.compressed_cache(3)?.unwrap().values()),
        "a later KV source must own a distinct cache even when its ratio is unchanged"
    );
    for pos in 7..11 {
        compare_step(&device, &mut graph, &mut manual, 1, pos)?;
    }
    // The shorter restart must never read the old cache tails or candidate mask.
    compare_step(&device, &mut graph, &mut manual, 1, 0)?;
    assert_eq!(graph.selection(1)?.unwrap().width, 0);
    compare_step(&device, &mut graph, &mut manual, 1, 1)?;
    assert_eq!(graph.selection(1)?.unwrap().width, 1);
    Ok(())
}

#[test]
fn deepseek_v41_orchestrator_rejects_invalid_calls_before_mutation_and_recovers_component_errors(
) -> Result<()> {
    let device = VulkanDevice::new()?;
    let mut graph = DeepseekV41AttentionOrchestrator::new(
        device.clone(),
        &config()?,
        runtime(),
        all_weights(),
    )?;
    let x = input(&device, 0, 3, 0)?;
    assert!(graph.begin_decode(0).is_err());
    assert!(graph.forward_layer(0, &x).is_err());
    assert!(graph.begin_prefill(0).is_err());
    assert!(graph.begin_prefill(13).is_err());
    assert!(graph.selection(8).is_err());
    graph.begin_prefill(3)?;
    assert!(graph.begin_decode(3).is_err());
    assert!(graph.forward_layer(1, &x).is_err());
    let short = GpuBuffer::from_f32(&device, &[1.0])?;
    assert!(graph.forward_layer(0, &short).is_err());
    let foreign_device = VulkanDevice::new()?;
    let foreign = input(&foreign_device, 0, 3, 0)?;
    assert!(graph.forward_layer(0, &foreign).is_err());
    assert_eq!(graph.next_layer(), Some(0));
    graph.forward_layer(0, &x)?;
    assert!(graph.forward_layer(0, &x).is_err());
    // Inject a real failure after the compressor/key writes. A retry must not
    // silently continue a partially mutated step; explicit prefill recovers it.
    graph.layers[1].finalizer = Some(DeepseekV41CompressedKvFinalizer::new(
        device.clone(),
        rotary(2),
        1,
        32,
    )?);
    let error = graph.forward_layer(1, &x).err().unwrap().to_string();
    assert!(error.contains("restart with begin_prefill"));
    assert!(graph.forward_layer(1, &x).is_err());
    assert!(graph.begin_decode(3).is_err());
    assert_eq!(graph.next_position(), None);
    assert_eq!(graph.next_layer(), None);
    assert!(graph.selection(1)?.is_none());
    assert!(graph.compressed_cache(1)?.is_none());
    graph.layers[1].finalizer = Some(DeepseekV41CompressedKvFinalizer::new(
        device.clone(),
        rotary(2),
        2,
        32,
    )?);
    let mut manual = Manual::new(&device)?;
    compare_step(&device, &mut graph, &mut manual, 3, 0)?;
    assert!(graph.begin_decode(2).is_err());
    assert!(graph.begin_decode(4).is_err());
    compare_step(&device, &mut graph, &mut manual, 1, 3)?;
    assert!(graph.begin_decode(32).is_err());
    Ok(())
}

#[test]
fn deepseek_v41_orchestrator_rejects_weight_ownership_and_capacity_mismatches() -> Result<()> {
    let device = VulkanDevice::new()?;
    let c = config()?;
    for (layer, remove_compressor) in [(0, false), (1, true)] {
        let mut w = all_weights();
        w[layer].compressor = if remove_compressor {
            None
        } else {
            weights(1).compressor
        };
        assert!(DeepseekV41AttentionOrchestrator::new(device.clone(), &c, runtime(), w).is_err());
    }
    let mut w = all_weights();
    w[3].indexer = None;
    assert!(DeepseekV41AttentionOrchestrator::new(device.clone(), &c, runtime(), w).is_err());
    for r in [
        DeepseekV41AttentionRuntimeConfig {
            batch_size: 0,
            ..runtime()
        },
        DeepseekV41AttentionRuntimeConfig {
            max_prefill: 33,
            ..runtime()
        },
        DeepseekV41AttentionRuntimeConfig {
            max_positions: 33,
            ..runtime()
        },
        DeepseekV41AttentionRuntimeConfig {
            batch_size: 65536,
            ..runtime()
        },
    ] {
        assert!(
            DeepseekV41AttentionOrchestrator::new(device.clone(), &c, r, all_weights()).is_err()
        );
    }
    Ok(())
}

struct Package {
    root: PathBuf,
}
impl Drop for Package {
    fn drop(&mut self) {
        if let Ok(entries) = fs::read_dir(&self.root) {
            for entry in entries.flatten() {
                let _ = fs::remove_file(entry.path());
            }
        }
        let _ = fs::remove_dir(&self.root);
    }
}

#[test]
fn deepseek_v41_orchestrator_loads_exact_owned_checkpoint_tensors() -> Result<()> {
    static ID: AtomicU64 = AtomicU64::new(0);
    let package = Package {
        root: std::env::temp_dir().join(format!(
            "hierarchos-v41-graph-{}-{}",
            std::process::id(),
            ID.fetch_add(1, Ordering::Relaxed)
        )),
    };
    fs::create_dir(&package.root)?;
    let c = config()?;
    fs::write(
        package.root.join("config.json"),
        serde_json::to_vec(&c.source_config)?,
    )?;
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = vec![];
    for id in 0..8 {
        let w = weights(id);
        let prefix = format!("layers.{id}.attn");
        let mut add = |suffix: &str, shape: Vec<usize>, values: Vec<f32>| {
            tensors.push((format!("{prefix}.{suffix}"), shape, values))
        };
        add("wq_a.weight", vec![16, 64], w.attention.wq_a);
        add("q_norm.weight", vec![16], w.attention.q_norm);
        add("wq_b.weight", vec![64, 16], w.attention.wq_b);
        add("wkv.weight", vec![32, 64], w.attention.wkv);
        add("kv_norm.weight", vec![32], w.attention.kv_norm);
        add("wo_a.weight", vec![16, 64], w.attention.wo_a);
        add("wo_b.weight", vec![64, 16], w.attention.wo_b);
        add("attn_sink", vec![2], w.attention.attn_sink);
        if let Some(w) = w.compressor {
            add("compressor.wkv.weight", vec![32, 64], w.kv_weight);
            add("compressor.norm.weight", vec![32], w.norm_weight);
            if let Some(gate) = w.gate_weight {
                add("compressor.wgate.weight", vec![32, 64], gate);
            }
        }
        if let Some(w) = w.indexer {
            if !w.wk.is_empty() {
                add("indexer.wk.weight", vec![32, 32], w.wk);
                add("indexer.k_norm.weight", vec![32], w.k_norm);
            }
            add("indexer.wq_b.weight", vec![64, 16], w.wq_b);
            add("indexer.weights_proj.weight", vec![2, 64], w.weights_proj);
        }
    }
    let views = tensors
        .iter()
        .map(|(name, shape, values)| {
            Ok((
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytemuck::cast_slice(values))?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    serialize_to_file(views, None, &package.root.join("model.safetensors"))?;
    let mut checkpoint = DeepseekV41Checkpoint::open(&package.root)?;
    assert!(checkpoint.load_attention_layer(8).is_err());
    let copy = checkpoint.load_attention_layer(6)?;
    assert!(copy.compressor.is_none());
    assert!(copy.indexer.unwrap().wk.is_empty());
    let owner = checkpoint.load_attention_layer(1)?;
    assert_eq!(owner.attention.wq_a, weights(1).attention.wq_a);
    assert_eq!(
        owner.compressor.unwrap().kv_weight,
        weights(1).compressor.unwrap().kv_weight
    );
    let device = VulkanDevice::new()?;
    let mut graph = DeepseekV41AttentionOrchestrator::from_checkpoint(
        device.clone(),
        &mut checkpoint,
        runtime(),
    )?;
    let mut manual = Manual::new(&device)?;
    compare_step(&device, &mut graph, &mut manual, 5, 0)?;
    compare_step(&device, &mut graph, &mut manual, 1, 5)?;
    Ok(())
}
