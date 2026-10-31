//! Layer-ordered V4.1 attention execution. This owns the shared publications and
//! cache lifetimes; a backbone supplies each layer's normalized attention input.
//! It does not implement the residual/FFN/Engram backbone or qualify model logits.

use anyhow::{bail, Context, Result};

use crate::{GpuBuffer, VulkanDevice};

#[path = "orchestrator_tape.rs"]
mod retained;
use crate::deepseek_v41::{
    DeepseekV41AttentionInputTape, DeepseekV41AttentionProjectionAdjoints,
    DeepseekV41CompressedKvTape, DeepseekV41CompressorTape,
};
pub(super) use retained::{AttentionLayerTape, AttentionRecording};

use super::{
    DeepseekV41Attention, DeepseekV41AttentionConfig, DeepseekV41AttentionWeights,
    DeepseekV41Checkpoint, DeepseekV41CompressedKvCache, DeepseekV41CompressedKvFinalizer,
    DeepseekV41Compressor, DeepseekV41CompressorWeights, DeepseekV41Config,
    DeepseekV41IndexKeyCache, DeepseekV41IndexSelection, DeepseekV41Indexer,
    DeepseekV41IndexerConfig, DeepseekV41IndexerWeights, DeepseekV41LayerPlan,
    DeepseekV41MixedAttentionWeights, DeepseekV41MixedIndexerWeights, DeepseekV41RotaryConfig,
    DeepseekV41WeightOutput,
};

/// Each runtime has a fixed batch shape, matching the shared cache strides.
/// `max_positions` is the allocated context limit, not the package's full limit.
#[derive(Clone, Copy, Debug)]
pub struct DeepseekV41AttentionRuntimeConfig {
    pub batch_size: usize,
    pub max_prefill: usize,
    pub max_positions: usize,
}

pub struct DeepseekV41AttentionLayerWeights {
    pub attention: DeepseekV41AttentionWeights,
    /// Present only at a KV source.
    pub compressor: Option<DeepseekV41CompressorWeights>,
    /// Present only at an index source; copy indexers have no wk/k_norm weights.
    pub indexer: Option<DeepseekV41IndexerWeights>,
}

pub(super) enum AttentionWeights {
    Fp32(DeepseekV41AttentionWeights),
    Mixed(DeepseekV41MixedAttentionWeights),
}
pub(super) enum IndexerWeights {
    Fp32(DeepseekV41IndexerWeights),
    Mixed(DeepseekV41MixedIndexerWeights),
}
pub(super) struct AttentionLayerWeights {
    pub attention: AttentionWeights,
    pub compressor: Option<DeepseekV41CompressorWeights>,
    pub indexer: Option<IndexerWeights>,
}
impl From<DeepseekV41AttentionLayerWeights> for AttentionLayerWeights {
    fn from(w: DeepseekV41AttentionLayerWeights) -> Self {
        Self {
            attention: AttentionWeights::Fp32(w.attention),
            compressor: w.compressor,
            indexer: w.indexer.map(IndexerWeights::Fp32),
        }
    }
}
impl AttentionLayerWeights {
    /// Legacy diagnostic callers deliberately promote dense BF16 storage.
    /// Packed tensors require the checkpoint-backed typed constructor.
    pub(super) fn into_fp32(self) -> Result<DeepseekV41AttentionLayerWeights> {
        let dense = |w| match w {
            super::DeepseekV41LinearWeights::Fp32 { values, .. }
            | super::DeepseekV41LinearWeights::Bf16 { values, .. } => Ok(values),
            super::DeepseekV41LinearWeights::Packed { .. } => bail!(
                "V4.1 packed attention/indexer requires the mixed backbone checkpoint constructor"
            ),
        };
        let attention = match self.attention {
            AttentionWeights::Fp32(w) => w,
            AttentionWeights::Mixed(w) => DeepseekV41AttentionWeights {
                wq_a: dense(w.wq_a)?,
                q_norm: w.q_norm,
                wq_b: dense(w.wq_b)?,
                wkv: dense(w.wkv)?,
                kv_norm: w.kv_norm,
                wo_a: dense(w.wo_a)?,
                wo_b: dense(w.wo_b)?,
                attn_sink: w.attn_sink,
            },
        };
        let indexer = self
            .indexer
            .map(|w| match w {
                IndexerWeights::Fp32(w) => Ok::<_, anyhow::Error>(w),
                IndexerWeights::Mixed(w) => Ok(DeepseekV41IndexerWeights {
                    wk: w.wk.map(dense).transpose()?.unwrap_or_default(),
                    k_norm: w.k_norm,
                    wq_b: dense(w.wq_b)?,
                    weights_proj: dense(w.weights_proj)?,
                }),
            })
            .transpose()?;
        Ok(DeepseekV41AttentionLayerWeights {
            attention,
            compressor: self.compressor,
            indexer,
        })
    }
}

struct Layer {
    attention: DeepseekV41Attention,
    compressor: Option<DeepseekV41Compressor>,
    indexer: Option<DeepseekV41Indexer>,
    finalizer: Option<DeepseekV41CompressedKvFinalizer>,
}

impl Layer {
    fn load_projections(&mut self) -> Result<()> {
        self.attention.load_projections()?;
        if let Some(compressor) = &mut self.compressor {
            compressor.load_weights()?;
        }
        if let Some(indexer) = self.indexer.as_mut() {
            indexer.load_weights()?;
        }
        Ok(())
    }

    fn release_projections(&mut self) {
        self.attention.release_projections();
        if let Some(compressor) = &mut self.compressor {
            compressor.release_weights();
        }
        if let Some(indexer) = self.indexer.as_mut() {
            indexer.release_weights();
        }
    }
}

#[derive(Clone, Copy)]
struct Step {
    seq: usize,
    start_pos: usize,
    prefill: bool,
    next_layer: usize,
}

/// Starts a token/chunk step once, then runs one attention layer at a time.
/// This leaves the caller free to run residual connections and FFNs between
/// layers while keeping KV/key/candidate/selection sharing inside this runtime.
///
/// Layer inputs and outputs remain on Vulkan. Invalid input/order/position calls are
/// rejected before mutation. A component failure after execution begins makes
/// decode unavailable until `begin_prefill` resets the sequence.
pub struct DeepseekV41AttentionOrchestrator {
    device: VulkanDevice,
    runtime: DeepseekV41AttentionRuntimeConfig,
    hidden: usize,
    heads: usize,
    index_heads: usize,
    head_dim: usize,
    window: usize,
    plan: Vec<DeepseekV41LayerPlan>,
    layers: Vec<Layer>,
    compressed: Vec<Option<DeepseekV41CompressedKvCache>>,
    // Clones share the source indexer's physical allocation; no duplicate keys.
    keys: Vec<Option<DeepseekV41IndexKeyCache>>,
    ready_groups: Vec<usize>,
    published_kv: Vec<bool>,
    selections: Vec<Option<DeepseekV41IndexSelection>>,
    step: Option<Step>,
    next_position: Option<usize>,
    needs_prefill: bool,
}

impl DeepseekV41AttentionOrchestrator {
    pub fn new(
        device: VulkanDevice,
        config: &DeepseekV41Config,
        runtime: DeepseekV41AttentionRuntimeConfig,
        weights: Vec<DeepseekV41AttentionLayerWeights>,
    ) -> Result<Self> {
        if weights.len() != config.text.num_hidden_layers {
            bail!("V4.1 attention layer-weight count is invalid");
        }
        Self::build(
            device,
            config,
            runtime,
            weights.into_iter().map(|w| Ok(w.into())),
            DeepseekV41WeightOutput::Fp32,
        )
    }

    pub(super) fn build(
        device: VulkanDevice,
        config: &DeepseekV41Config,
        runtime: DeepseekV41AttentionRuntimeConfig,
        mut weights: impl Iterator<Item = Result<AttentionLayerWeights>>,
        precision: DeepseekV41WeightOutput,
    ) -> Result<Self> {
        let plan = config.plan()?.backbone;
        let text = &config.text;
        if runtime.batch_size == 0
            || runtime.max_prefill == 0
            || runtime.max_positions == 0
            || runtime.max_prefill > runtime.max_positions
            || runtime.max_positions > text.max_position_embeddings
            || runtime.max_positions > 65_535
        {
            bail!("V4.1 attention runtime capacities are invalid");
        }
        Self::validate_geometry(
            runtime,
            text.num_attention_heads,
            text.index_n_heads,
            text.head_dim,
            runtime.max_prefill,
            0,
        )?;
        let mut layers = Vec::with_capacity(plan.len());
        let mut compressed = Vec::with_capacity(plan.len());
        let mut keys = Vec::with_capacity(plan.len());
        for p in &plan {
            // Check a layer before creating its component graph. The checkpoint
            // iterator releases that layer's host tensors after GPU upload.
            let weights = weights
                .next()
                .context("V4.1 attention layer weights are missing")??;
            let owns_kv = p.kv_source_layer == Some(p.layer_id);
            let owns_index = p.index_source_layer == Some(p.layer_id);
            if weights.compressor.is_some() != owns_kv || weights.indexer.is_some() != owns_index {
                bail!(
                    "V4.1 layer {} weights do not match compressor/indexer ownership",
                    p.layer_id
                );
            }
            if let Some(c) = &weights.compressor {
                if c.hidden_size != text.hidden_size
                    || c.head_dim != text.head_dim
                    || c.compress_ratio != p.compress_ratio
                    || c.eps != text.rms_norm_eps as f32
                {
                    bail!(
                        "V4.1 layer {} compressor geometry/epsilon does not match its text config",
                        p.layer_id
                    );
                }
            }
            let rotary = match (p.compress_ratio, &text.rope_scaling) {
                (0, _) => DeepseekV41RotaryConfig::sliding(
                    text.qk_rope_head_dim,
                    runtime.max_positions,
                    text.rope_theta as f32,
                ),
                (_, Some(rope)) => DeepseekV41RotaryConfig::compressed(
                    text.qk_rope_head_dim,
                    runtime.max_positions,
                    text.compress_rope_theta as f32,
                    rope.original_max_position_embeddings,
                    rope.factor as f32,
                    rope.beta_fast as f32,
                    rope.beta_slow as f32,
                ),
                (_, None) => DeepseekV41RotaryConfig::sliding(
                    text.qk_rope_head_dim,
                    runtime.max_positions,
                    text.compress_rope_theta as f32,
                ),
            };
            let attention_config = DeepseekV41AttentionConfig {
                heads: text.num_attention_heads,
                head_dim: text.head_dim,
                hidden_size: text.hidden_size,
                q_lora_rank: text.q_lora_rank,
                o_groups: text.o_groups,
                o_lora_rank: text.o_lora_rank,
                window_size: text.sliding_window,
                compress_ratio: p.compress_ratio,
                norm_eps: text.rms_norm_eps as f32,
                rotary,
            };
            let attention = match weights.attention {
                AttentionWeights::Fp32(w) => {
                    if precision == DeepseekV41WeightOutput::Bf16 {
                        bail!("V4.1 BF16 attention requires typed BF16/packed projections");
                    }
                    DeepseekV41Attention::new(
                        device.clone(),
                        attention_config,
                        w,
                        runtime.batch_size,
                    )
                }
                AttentionWeights::Mixed(w) => DeepseekV41Attention::new_mixed(
                    device.clone(),
                    attention_config,
                    w,
                    runtime.batch_size,
                    precision,
                ),
            }
            .with_context(|| format!("building V4.1 attention layer {}", p.layer_id))?;
            let compressor = weights
                .compressor
                .map(|w| {
                    DeepseekV41Compressor::new_with_precision(
                        device.clone(),
                        w,
                        runtime.batch_size,
                        runtime.max_prefill,
                        precision,
                    )
                })
                .transpose()?;
            let indexer = weights
                .indexer
                .map(|w| {
                    let indexer_config = DeepseekV41IndexerConfig {
                        heads: text.index_n_heads,
                        index_head_dim: text.index_head_dim,
                        head_dim: text.head_dim,
                        hidden_size: text.hidden_size,
                        q_lora_rank: text.q_lora_rank,
                        compress_ratio: p.compress_ratio,
                        index_topk: text.index_topk,
                        owns_k: owns_kv,
                        is_candidate_source: p.publishes_candidates,
                        candidate_topk_blocks: text.candidate_topk_blocks,
                        candidate_block_size: text.candidate_block_size,
                        norm_eps: text.rms_norm_eps as f32,
                        rotary,
                    };
                    match w {
                        IndexerWeights::Fp32(w) => DeepseekV41Indexer::new(
                            device.clone(),
                            indexer_config,
                            w,
                            runtime.batch_size,
                            runtime.max_positions,
                        ),
                        IndexerWeights::Mixed(w) => DeepseekV41Indexer::new_mixed(
                            device.clone(),
                            indexer_config,
                            w,
                            runtime.batch_size,
                            runtime.max_positions,
                            precision,
                            attention.query_precision(),
                        ),
                    }
                })
                .transpose()?;
            keys.push(indexer.as_ref().and_then(|i| i.index_keys()).cloned());
            compressed.push(if owns_kv {
                Some(DeepseekV41CompressedKvCache::new(
                    device.clone(),
                    runtime.batch_size,
                    runtime.max_positions.div_ceil(p.compress_ratio),
                    text.head_dim,
                )?)
            } else {
                None
            });
            let finalizer = if owns_kv {
                Some(DeepseekV41CompressedKvFinalizer::new_with_precision(
                    device.clone(),
                    rotary,
                    p.compress_ratio,
                    text.head_dim,
                    precision,
                )?)
            } else {
                None
            };
            layers.push(Layer {
                attention,
                compressor,
                indexer,
                finalizer,
            });
        }
        let count = plan.len();
        Ok(Self {
            device,
            runtime,
            hidden: text.hidden_size,
            heads: text.num_attention_heads,
            index_heads: text.index_n_heads,
            head_dim: text.head_dim,
            window: text.sliding_window,
            plan,
            layers,
            compressed,
            keys,
            ready_groups: vec![0; count],
            published_kv: vec![false; count],
            selections: vec![None; count],
            step: None,
            next_position: None,
            needs_prefill: false,
        })
    }

    /// Builds every layer without uploading attention, compressor or indexer
    /// projections. The orchestrator reads them when it enters a layer and
    /// releases them after it.
    pub(super) fn build_streamed(
        device: VulkanDevice,
        config: &DeepseekV41Config,
        runtime: DeepseekV41AttentionRuntimeConfig,
        mut checkpoint: DeepseekV41Checkpoint,
        precision: DeepseekV41WeightOutput,
    ) -> Result<Self> {
        let plan = config.plan()?.backbone;
        let text = &config.text;
        if runtime.batch_size == 0
            || runtime.max_prefill == 0
            || runtime.max_positions == 0
            || runtime.max_prefill > runtime.max_positions
            || runtime.max_positions > text.max_position_embeddings
            || runtime.max_positions > 65_535
        {
            bail!("V4.1 attention runtime capacities are invalid");
        }
        Self::validate_geometry(
            runtime,
            text.num_attention_heads,
            text.index_n_heads,
            text.head_dim,
            runtime.max_prefill,
            0,
        )?;
        let mut layers = Vec::with_capacity(plan.len());
        let mut compressed = Vec::with_capacity(plan.len());
        let mut keys = Vec::with_capacity(plan.len());
        for p in &plan {
            let owns_kv = p.kv_source_layer == Some(p.layer_id);
            let owns_index = p.index_source_layer == Some(p.layer_id);
            let rotary = match (p.compress_ratio, &text.rope_scaling) {
                (0, _) => DeepseekV41RotaryConfig::sliding(
                    text.qk_rope_head_dim,
                    runtime.max_positions,
                    text.rope_theta as f32,
                ),
                (_, Some(rope)) => DeepseekV41RotaryConfig::compressed(
                    text.qk_rope_head_dim,
                    runtime.max_positions,
                    text.compress_rope_theta as f32,
                    rope.original_max_position_embeddings,
                    rope.factor as f32,
                    rope.beta_fast as f32,
                    rope.beta_slow as f32,
                ),
                (_, None) => DeepseekV41RotaryConfig::sliding(
                    text.qk_rope_head_dim,
                    runtime.max_positions,
                    text.compress_rope_theta as f32,
                ),
            };
            let attention_config = DeepseekV41AttentionConfig {
                heads: text.num_attention_heads,
                head_dim: text.head_dim,
                hidden_size: text.hidden_size,
                q_lora_rank: text.q_lora_rank,
                o_groups: text.o_groups,
                o_lora_rank: text.o_lora_rank,
                window_size: text.sliding_window,
                compress_ratio: p.compress_ratio,
                norm_eps: text.rms_norm_eps as f32,
                rotary,
            };
            let query_precision = if checkpoint
                .tensor_dtype(&format!("layers.{}.attn.wq_a.weight", p.layer_id))?
                != "F32"
            {
                DeepseekV41WeightOutput::Bf16
            } else {
                DeepseekV41WeightOutput::Fp32
            };
            let attention = DeepseekV41Attention::new_mixed_streamed(
                device.clone(),
                attention_config,
                checkpoint.clone(),
                p.layer_id,
                runtime.batch_size,
                precision,
                query_precision,
            )
            .with_context(|| format!("building streamed V4.1 attention layer {}", p.layer_id))?;
            let compressor = if owns_kv {
                Some(DeepseekV41Compressor::new_streamed(
                    device.clone(),
                    text.hidden_size,
                    text.head_dim,
                    p.compress_ratio,
                    text.rms_norm_eps as f32,
                    runtime.batch_size,
                    runtime.max_prefill,
                    precision,
                    checkpoint.clone(),
                    p.layer_id,
                )?)
            } else {
                None
            };
            let indexer = if owns_index {
                let indexer_config = DeepseekV41IndexerConfig {
                    heads: text.index_n_heads,
                    index_head_dim: text.index_head_dim,
                    head_dim: text.head_dim,
                    hidden_size: text.hidden_size,
                    q_lora_rank: text.q_lora_rank,
                    compress_ratio: p.compress_ratio,
                    index_topk: text.index_topk,
                    owns_k: owns_kv,
                    is_candidate_source: p.publishes_candidates,
                    candidate_topk_blocks: text.candidate_topk_blocks,
                    candidate_block_size: text.candidate_block_size,
                    norm_eps: text.rms_norm_eps as f32,
                    rotary,
                };
                Some(DeepseekV41Indexer::new_mixed_streamed(
                    device.clone(),
                    indexer_config,
                    runtime.batch_size,
                    runtime.max_positions,
                    precision,
                    attention.query_precision(),
                    checkpoint.clone(),
                    p.layer_id,
                )?)
            } else {
                None
            };
            keys.push(indexer.as_ref().and_then(|i| i.index_keys()).cloned());
            compressed.push(if owns_kv {
                Some(DeepseekV41CompressedKvCache::new(
                    device.clone(),
                    runtime.batch_size,
                    runtime.max_positions.div_ceil(p.compress_ratio),
                    text.head_dim,
                )?)
            } else {
                None
            });
            let finalizer = if owns_kv {
                Some(DeepseekV41CompressedKvFinalizer::new_with_precision(
                    device.clone(),
                    rotary,
                    p.compress_ratio,
                    text.head_dim,
                    precision,
                )?)
            } else {
                None
            };
            layers.push(Layer {
                attention,
                compressor,
                indexer,
                finalizer,
            });
        }
        let count = plan.len();
        Ok(Self {
            device,
            runtime,
            hidden: text.hidden_size,
            heads: text.num_attention_heads,
            index_heads: text.index_n_heads,
            head_dim: text.head_dim,
            window: text.sliding_window,
            plan,
            layers,
            compressed,
            keys,
            ready_groups: vec![0; count],
            published_kv: vec![false; count],
            selections: vec![None; count],
            step: None,
            next_position: None,
            needs_prefill: false,
        })
    }

    /// Reads the next layer's streamed projections and guarantees they are
    /// released after its execution, whether it succeeds or fails.
    fn execute_layer(
        &mut self,
        layer_id: usize,
        x: &GpuBuffer,
        step: Step,
        recording: Option<(
            &DeepseekV41AttentionProjectionAdjoints,
            &mut AttentionRecording,
        )>,
    ) -> Result<(GpuBuffer, Option<AttentionLayerTape>)> {
        // Release partial uploads too: a compressor/indexer failure after the
        // attention projections were staged must not pin them between calls.
        let result = (|| {
            self.layers[layer_id].load_projections()?;
            self.execute_layer_inner(layer_id, x, step, recording)
        })();
        self.layers[layer_id].release_projections();
        result
    }

    pub fn from_checkpoint(
        device: VulkanDevice,
        checkpoint: &mut DeepseekV41Checkpoint,
        runtime: DeepseekV41AttentionRuntimeConfig,
    ) -> Result<Self> {
        let config = checkpoint.config.clone();
        let weights = (0..config.text.num_hidden_layers)
            .map(|layer| checkpoint.load_attention_components(layer));
        Self::build(
            device,
            &config,
            runtime,
            weights,
            DeepseekV41WeightOutput::Fp32,
        )
    }

    pub fn layer_plan(&self) -> &[DeepseekV41LayerPlan] {
        &self.plan
    }

    /// Layers holding uploaded attention projections between executions.
    pub fn resident_projection_layer_count(&self) -> usize {
        self.layers
            .iter()
            .filter(|layer| layer.attention.projections_are_resident())
            .count()
    }
    #[cfg(test)]
    pub(super) fn projection_trace(&self, layer: usize) -> Option<(GpuBuffer, GpuBuffer)> {
        self.layers[layer].attention.projection_trace.clone()
    }
    /// Shape of the step in flight, if every layer still has to run.
    pub fn step_shape(&self) -> Option<(usize, usize, bool)> {
        self.step
            .map(|step| (step.seq, step.start_pos, step.prefill))
    }
    pub fn next_layer(&self) -> Option<usize> {
        if self.needs_prefill {
            None
        } else {
            self.step.map(|step| step.next_layer)
        }
    }
    /// Available only after every attention layer has completed the prior step.
    pub fn next_position(&self) -> Option<usize> {
        if self.step.is_none() && !self.needs_prefill {
            self.next_position
        } else {
            None
        }
    }

    /// Explicitly starts/restarts a sequence, including recovery from a failed
    /// component call. Old cache tails are unreachable until rewritten.
    pub fn begin_prefill(&mut self, seq: usize) -> Result<()> {
        self.validate_step_geometry(seq, 0)?;
        if seq > self.runtime.max_prefill {
            bail!("V4.1 prefill exceeds the allocated prefill capacity");
        }
        self.ready_groups.fill(0);
        self.next_position = None;
        self.needs_prefill = false;
        self.begin(Step {
            seq,
            start_pos: 0,
            prefill: true,
            next_layer: 0,
        });
        Ok(())
    }

    pub fn begin_decode(&mut self, start_pos: usize) -> Result<()> {
        if self.needs_prefill {
            bail!("V4.1 attention component failed; restart with begin_prefill");
        }
        if self.step.is_some() || self.next_position != Some(start_pos) {
            bail!("V4.1 decode requires a completed prior step at the next sequence position");
        }
        self.validate_step_geometry(1, start_pos)?;
        self.begin(Step {
            seq: 1,
            start_pos,
            prefill: false,
            next_layer: 0,
        });
        Ok(())
    }

    fn begin(&mut self, step: Step) {
        self.published_kv.fill(false);
        self.selections.fill(None);
        self.step = Some(step);
    }

    fn validate_step_geometry(&self, seq: usize, start_pos: usize) -> Result<()> {
        Self::validate_geometry(
            self.runtime,
            self.heads,
            self.index_heads,
            self.head_dim,
            seq,
            start_pos,
        )
    }

    fn validate_geometry(
        runtime: DeepseekV41AttentionRuntimeConfig,
        heads: usize,
        index_heads: usize,
        head_dim: usize,
        seq: usize,
        start_pos: usize,
    ) -> Result<()> {
        let end = start_pos
            .checked_add(seq)
            .context("V4.1 attention position overflow")?;
        let rows = runtime
            .batch_size
            .checked_mul(seq)
            .context("V4.1 attention row overflow")?;
        let repeated_rows = rows
            .checked_mul(heads.max(index_heads))
            .context("V4.1 attention head-row overflow")?;
        let elements = rows
            .checked_mul(head_dim)
            .context("V4.1 attention element overflow")?;
        // `repeated_rows` is only an overflow check now: per-head-row kernels
        // tile rows past one dispatch axis, so a start-zero prefill can reach
        // the released index-head count at full prompt length.
        let _ = repeated_rows;
        if seq == 0 || end > runtime.max_positions || elements.div_ceil(64) > 65_535 {
            bail!("V4.1 attention step exceeds context or dispatch capacity");
        }
        Ok(())
    }

    /// Runs exactly the next layer. Offsets and source identities are derived
    /// from the validated schedule; callers cannot pass stale selections/caches.
    pub fn forward_layer(&mut self, layer_id: usize, x: &GpuBuffer) -> Result<GpuBuffer> {
        self.forward_layer_recorded(layer_id, x, None)
            .map(|(output, _)| output)
    }

    pub(super) fn forward_layer_recorded(
        &mut self,
        layer_id: usize,
        x: &GpuBuffer,
        recording: Option<(
            &DeepseekV41AttentionProjectionAdjoints,
            &mut AttentionRecording,
        )>,
    ) -> Result<(GpuBuffer, Option<AttentionLayerTape>)> {
        if self.needs_prefill {
            bail!("V4.1 attention component failed; restart with begin_prefill");
        }
        let step = self
            .step
            .context("V4.1 attention requires begin_prefill or begin_decode")?;
        if layer_id != step.next_layer || layer_id >= self.layers.len() {
            bail!(
                "V4.1 attention expected layer {}, got {layer_id}",
                step.next_layer
            );
        }
        let elements = self
            .runtime
            .batch_size
            .checked_mul(step.seq)
            .and_then(|n| n.checked_mul(self.hidden))
            .context("V4.1 attention input overflow")?;
        if !x.belongs_to_device(&self.device) || x.f32_capacity() < elements {
            bail!("V4.1 attention orchestrator input geometry or device mismatch");
        }
        let p = &self.plan[layer_id];
        if let Some(source) = p.kv_source_layer.filter(|&id| id != layer_id) {
            let width = (step.start_pos + step.seq) / p.compress_ratio;
            if !self.published_kv[source] || self.ready_groups[source] != width {
                bail!("V4.1 layer {layer_id} has no current-step KV/key publication from {source}");
            }
        }
        if let Some(source) = p.index_source_layer.filter(|&id| id != layer_id) {
            if self.selections[source].is_none() {
                bail!("V4.1 layer {layer_id} has no current-step selection from {source}");
            }
        }
        if p.index_source_layer == Some(layer_id) {
            if let Some(source) = p.candidate_source_layer {
                let width = (step.start_pos + step.seq) / p.compress_ratio;
                if self.selections[source]
                    .as_ref()
                    .is_none_or(|s| width != 0 && s.candidates.is_none())
                {
                    bail!("V4.1 layer {layer_id} has no current-step candidates from {source}");
                }
            }
        }
        match self.execute_layer(layer_id, x, step, recording) {
            Ok(output) => {
                if layer_id + 1 == self.layers.len() {
                    self.next_position = Some(step.start_pos + step.seq);
                    self.step = None;
                } else {
                    self.step = Some(Step {
                        next_layer: layer_id + 1,
                        ..step
                    });
                }
                Ok(output)
            }
            Err(error) => {
                // GPU cache writes may already have happened; never retry them
                // as if a component failure had rolled back the token step.
                self.needs_prefill = true;
                self.published_kv.fill(false);
                self.selections.fill(None);
                Err(error.context(format!(
                    "executing V4.1 attention layer {layer_id}; restart with begin_prefill"
                )))
            }
        }
    }

    fn execute_layer_inner(
        &mut self,
        layer_id: usize,
        x: &GpuBuffer,
        step: Step,
        mut recording: Option<(
            &DeepseekV41AttentionProjectionAdjoints,
            &mut AttentionRecording,
        )>,
    ) -> Result<(GpuBuffer, Option<AttentionLayerTape>)> {
        let p = &self.plan[layer_id];
        let batch = self.runtime.batch_size;
        let offset = if step.prefill { step.seq } else { self.window };
        let layer = &mut self.layers[layer_id];
        let inputs = recording
            .as_ref()
            .map(|(adjoints, _)| {
                adjoints.forward_inputs(&layer.attention, x, batch, step.seq, step.start_pos)
            })
            .transpose()?;
        let qr = match &inputs {
            Some(tape) => tape.query_latent.clone(),
            None => layer.attention.query_latent(x, batch * step.seq)?,
        };
        let mut compressor_tape = None;
        let mut finalizer_tape = None;
        let mut compressed_write = None;
        let latent = if let Some(compressor) = &mut layer.compressor {
            Some(if recording.is_some() {
                let (latent, tape) = if step.prefill {
                    compressor.prefill_retained(x, batch, step.seq)?
                } else {
                    compressor.decode_retained(x, batch, step.start_pos)?
                };
                compressor_tape = Some(tape);
                latent
            } else if step.prefill {
                compressor.prefill_from_buffer(x, batch, step.seq)?
            } else {
                compressor.decode_step_from_buffer(x, batch, step.start_pos)?
            })
        } else {
            None
        };
        if let Some(latent) = &latent {
            let indexer = layer
                .indexer
                .as_ref()
                .context("V4.1 KV source is missing its indexer")?;
            indexer.write_index_keys(latent, step.start_pos)?;
            let cache = self.compressed[layer_id]
                .as_ref()
                .context("V4.1 KV source has no compressed cache")?;
            let width = (step.start_pos + step.seq) / p.compress_ratio;
            if !step.prefill && self.ready_groups[layer_id] + latent.groups_per_batch != width {
                bail!("V4.1 source {layer_id} compression groups are discontinuous");
            }
            if latent.groups_per_batch > 0 {
                let finalizer = layer
                    .finalizer
                    .as_ref()
                    .context("V4.1 KV source has no finalizer")?;
                let kv = if recording.is_some() {
                    let (kv, tape) = if step.prefill {
                        finalizer.prefill_retained(latent)?
                    } else {
                        finalizer.decode_retained(latent, step.start_pos)?
                    };
                    finalizer_tape = Some(tape);
                    kv
                } else if step.prefill {
                    finalizer.finalize_prefill(latent)?
                } else {
                    finalizer.finalize_decode(latent, step.start_pos)?
                };
                if let Some((_, rec)) = recording.as_mut() {
                    let id = rec.compressed[layer_id]
                        .as_ref()
                        .context("V4.1 retained source has no compressed cache ID")?;
                    compressed_write = Some(rec.cache.append_compressed(
                        id,
                        &kv,
                        latent.groups_per_batch,
                        latent.first_group,
                    )?);
                } else {
                    cache.append_groups(&kv, batch, latent.groups_per_batch, latent.first_group)?;
                }
            }
            self.ready_groups[layer_id] = width;
            self.published_kv[layer_id] = true;
        }
        if let Some(indexer) = &layer.indexer {
            let source = p
                .kv_source_layer
                .context("V4.1 indexer has no KV/key source")?;
            let keys = self.keys[source]
                .as_ref()
                .context("V4.1 indexer has no shared keys")?;
            let candidates = p
                .candidate_source_layer
                .and_then(|source| self.selections[source].as_ref())
                .and_then(|s| s.candidates.as_ref());
            self.selections[layer_id] = Some(indexer.select(
                x,
                &qr,
                keys,
                batch,
                step.seq,
                step.start_pos,
                offset,
                candidates,
            )?);
        }
        let compressed = p
            .kv_source_layer
            .and_then(|source| self.compressed[source].as_ref());
        let selection = p
            .index_source_layer
            .and_then(|source| self.selections[source].as_ref());
        if let Some((adjoints, rec)) = recording {
            return retained::finish_forward(
                rec,
                adjoints,
                &mut layer.attention,
                inputs.unwrap(),
                p,
                step,
                selection,
                compressor_tape,
                finalizer_tape,
                compressed_write,
            );
        }
        layer
            .attention
            .forward_step(
                x,
                Some(&qr),
                compressed,
                selection,
                batch,
                step.seq,
                step.start_pos,
                step.prefill,
            )
            .map(|output| (output, None))
    }

    /// Current-step publications, also retained after a completed step for
    /// inspection. A new begin call invalidates previous selections immediately.
    pub fn selection(&self, layer_id: usize) -> Result<Option<&DeepseekV41IndexSelection>> {
        let p = self
            .plan
            .get(layer_id)
            .context("V4.1 attention layer is out of range")?;
        Ok(p.index_source_layer
            .and_then(|source| self.selections[source].as_ref()))
    }
    #[cfg(test)]
    pub(super) fn sparse_trace(&self, layer: usize) -> Option<&(GpuBuffer, GpuBuffer)> {
        self.layers[layer].attention.sparse_trace.as_ref()
    }

    /// Test-only weights/pre-cast captures for the sparse attention of `layer`.
    #[cfg(test)]
    pub(super) fn raw_attention_trace(&self, layer: usize) -> Option<(GpuBuffer, GpuBuffer)> {
        let attention = &self.layers[layer].attention;
        let weights = attention.weights_trace.borrow().clone()?;
        let raw = attention.raw_trace.borrow().clone()?;
        Some((weights, raw))
    }

    pub fn compressed_cache(
        &self,
        layer_id: usize,
    ) -> Result<Option<&DeepseekV41CompressedKvCache>> {
        let p = self
            .plan
            .get(layer_id)
            .context("V4.1 attention layer is out of range")?;
        Ok(p.kv_source_layer
            .filter(|&source| self.published_kv[source])
            .and_then(|source| self.compressed[source].as_ref()))
    }

    pub fn index_keys(&self, layer_id: usize) -> Result<Option<&DeepseekV41IndexKeyCache>> {
        let p = self
            .plan
            .get(layer_id)
            .context("V4.1 attention layer is out of range")?;
        Ok(p.kv_source_layer
            .filter(|&source| self.published_kv[source])
            .and_then(|source| self.keys[source].as_ref()))
    }
}

#[cfg(test)]
#[path = "orchestrator_tests.rs"]
mod tests;
