//! Native DeepSeek V4.1 text graph, distinct from the DeepSeek V4 graph.
//!
//! Architectural source: deepseek-ai/DeepSeek-V4.1-Flash-1017 at
//! 1bae0e1088993b9b82de08f59de5be5a34e42170 (inference/model.py).
//! `DeepseekV41CausalLM::from_hf_package` is the normal package path: it reads
//! the released config and SafeTensors index, infers F32/BF16 and packed expert
//! storage, derives Engram metadata from `tokenizer.json` or
//! `engram_tokens.json`, streams routed/shared experts, Engram table rows,
//! block projections and vocabulary, and requires no execution-policy flag.
//! The dedicated text graph produces full prefill/decode logits with strict
//! CPU-PyTorch parity for the measured tiny/wider graphs, a 512-token YaRN
//! prefill, a released-width/vocabulary probe, a released-width four-token
//! prefill plus continued decode, separate depth/cardinality probes, and joint
//! 40-layer/384-expert and 2,048-token graphs at reduced widths.
//! The full 510 GB released checkpoint has not been executed locally. Joint
//! released geometry, million-token contexts, DSpark and vision remain
//! unqualified. Standard HF PEFT-LoRA adapters are applied additively through
//! `from_hf_package_with_lora`; the packed base is never rewritten, adapter
//! export round-trips through the normal loader, and replacement-module
//! adapters stay fail-closed. Adapter *training* has its loss-side adjoints
//! (`terminal`), the Hyper-Connection post and FP32 MoE adjoints
//! (`DeepseekV41BlockAdjoints`), the Hyper-Connection split/collapse, FP32 sparse
//! attention, grouped-output/RoPE/cache-write adjoints, and AdamW/checkpoint
//! state (`train`). FP32 projection tapes and LoRA-factor VJPs now train a
//! query-projection subgraph through cross entropy, AdamW and checkpoint resume.
//! The compressor's prefill adjoint and retained mixed MoE input/adapter
//! VJPs are qualified too, including checkpoint-staged frozen experts. Complete
//! Engram stream/key/value and selected-row projection adjoints are gated.
//! Compressor cached-decode state VJPs are gated through the native forward.
//! Conditional factor-local AdamW and resume are gated too. A retained sparse
//! attention/cache tape joins shared readers and chains overwritten rows across
//! decode steps. Retained Q/KV/norm/rotary and grouped output projections now
//! compose with that cache tape, including mixed storage and adapter VJPs.
//! Compressor/finalizer and model integration plus the full-model loss-to-factor gate remain
//! before training can be enabled. Hyper-connections, norms,
//! gates, caches and Engram projections stay resident. The CPU math policy is
//! pinned to the measured Windows runtime. CPU PyTorch is the accepted parity
//! reference; CUDA parity is not required.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

mod attention;
mod attention_grad;
mod backbone;
mod candidates;
mod checkpoint;
mod compressed_kv;
mod compressor;
mod compressor_grad;
mod engram;
mod indexer;
mod mixed_linear;
mod model;
mod orchestrator;
mod packed_weights;
mod peft;
mod quantization;
mod rotary;
mod terminal;
mod tokenizer;
mod train;
#[cfg(test)]
pub(crate) mod training_trace {
    use crate::GpuBuffer;
    thread_local! { static VALUES: std::cell::RefCell<Option<Vec<serde_json::Value>>> = const { std::cell::RefCell::new(None) }; }
    pub fn begin() { VALUES.with(|v| *v.borrow_mut() = Some(Vec::new())); }
    pub fn record(label: &str, value: &GpuBuffer, len: usize) {
        VALUES.with(|v| { if let Some(values) = v.borrow_mut().as_mut() { values.push(serde_json::json!({"label":label,"bits":value.read_f32(len).unwrap().iter().map(|f|f.to_bits()).collect::<Vec<_>>()})); } });
    }
    pub fn finish() -> Vec<serde_json::Value> { VALUES.with(|v| v.borrow_mut().take().unwrap_or_default()) }
}
pub use attention::{
    deepseek_v41_window_indices, DeepseekV41Attention, DeepseekV41AttentionCacheId,
    DeepseekV41AttentionConfig, DeepseekV41AttentionInputTape, DeepseekV41AttentionOutputTape,
    DeepseekV41AttentionProjectionAdjoints, DeepseekV41AttentionProjectionGradients,
    DeepseekV41AttentionRead, DeepseekV41AttentionReadId, DeepseekV41AttentionReverseTape,
    DeepseekV41AttentionTape, DeepseekV41AttentionTapeGradients, DeepseekV41AttentionWeights,
    DeepseekV41AttentionWindowView, DeepseekV41AttentionWriteId, DeepseekV41CompressedKvCache,
    DeepseekV41MixedAttentionWeights, DeepseekV41SparseAttentionParams, DeepseekV41WindowKvCache,
};
pub use attention_grad::{
    DeepseekV41AttentionAdjoints, DeepseekV41CacheWriteGradients,
    DeepseekV41SparseAttentionGradients,
};
pub use backbone::{
    DeepseekV41Backbone, DeepseekV41BackboneLayerWeights, DeepseekV41BlockAdjoints,
    DeepseekV41ExpertResidency, DeepseekV41HyperConnectionMixes,
    DeepseekV41HyperConnectionPostGradients, DeepseekV41HyperConnectionPreGradients,
    DeepseekV41HyperConnectionWeights, DeepseekV41HyperConnections, DeepseekV41HyperStage,
    DeepseekV41MixedExpertWeights, DeepseekV41MixedMoe, DeepseekV41MixedMoeAdjoints,
    DeepseekV41MixedMoeGradients, DeepseekV41MixedMoeTape, DeepseekV41MixedMoeWeights,
    DeepseekV41Moe, DeepseekV41MoeCombineGradients, DeepseekV41MoeConfig,
    DeepseekV41MoeExpertGradients, DeepseekV41MoeScoreFunc, DeepseekV41MoeWeights,
    DeepseekV41ProjectionResidency,
};
pub use candidates::DeepseekV41CandidateSelector;
pub use checkpoint::{DeepseekV41Checkpoint, DeepseekV41CheckpointIoStats};
pub use compressed_kv::{
    DeepseekV41CompressedKvFinalizer, DeepseekV41CompressedKvTape, COMPRESSED_KV_QUANT_BLOCK,
};
pub use compressor::{
    DeepseekV41CompressedLatents, DeepseekV41Compressor, DeepseekV41CompressorTape,
    DeepseekV41CompressorWeights,
};
pub use compressor_grad::{
    DeepseekV41CompressorAdjoints, DeepseekV41CompressorDecode,
    DeepseekV41CompressorDecodeOperands, DeepseekV41CompressorGradients,
    DeepseekV41CompressorOperands, DeepseekV41CompressorPrefill,
};
pub use engram::{
    DeepseekV41Engram, DeepseekV41EngramAdjoints, DeepseekV41EngramGateConfig,
    DeepseekV41EngramGateGradients, DeepseekV41EngramGateTape, DeepseekV41EngramGradients,
    DeepseekV41EngramHash, DeepseekV41EngramLayout, DeepseekV41EngramResidency,
    DeepseekV41EngramRuntime, DeepseekV41EngramTape, DeepseekV41EngramTokens,
    DeepseekV41EngramWeights,
};
pub use indexer::{
    DeepseekV41IndexKeyCache, DeepseekV41IndexScorer, DeepseekV41IndexSelection,
    DeepseekV41IndexTopk, DeepseekV41Indexer, DeepseekV41IndexerConfig, DeepseekV41IndexerWeights,
    DeepseekV41MixedIndexerWeights,
};
pub use mixed_linear::{
    DeepseekV41LinearAdjoints, DeepseekV41LinearGradients, DeepseekV41LinearTape,
    DeepseekV41LinearWeights, DeepseekV41MixedLinearKernel, DeepseekV41ResidentLinear,
};
pub use model::{
    DeepseekV41CausalLM, DeepseekV41Logits, DeepseekV41ModelGradients, DeepseekV41ModelTrainResult,
    DeepseekV41ModelTrainer, DeepseekV41ModelTrainingConfig, DeepseekV41ModelWeights,
    DeepseekV41TrainingBatch, DeepseekV41VocabularyResidency,
};
pub use orchestrator::{
    DeepseekV41AttentionLayerWeights, DeepseekV41AttentionOrchestrator,
    DeepseekV41AttentionRuntimeConfig,
};
pub use packed_weights::{
    DeepseekV41PackedWeightKernel, DeepseekV41PackedWeights, DeepseekV41ResidentPackedWeights,
    DeepseekV41WeightFormat, DeepseekV41WeightOutput,
};
pub use peft::{
    lora_capability, DeepseekV41LoraAdapter, DeepseekV41LoraModule, DeepseekV41PeftCapability,
};
pub use quantization::{
    DeepseekV41ActivationQuantizer, DeepseekV41QuantMode, DeepseekV41QuantizedActivation,
    DeepseekV41ScaleMode,
};
pub use rotary::{DeepseekV41Rotary, DeepseekV41RotaryConfig};
pub use terminal::{DeepseekV41TerminalAdjoints, DeepseekV41TerminalGradients};
pub use train::{
    DeepseekV41AdapterFactorState, DeepseekV41AdapterGradient, DeepseekV41AdapterParameterGradient,
    DeepseekV41AdapterTrainer, DeepseekV41AdamWConfig,
};

pub const REFERENCE_MODEL: &str = "deepseek-ai/DeepSeek-V4.1-Flash-1017";
pub const REFERENCE_REVISION: &str = "1bae0e1088993b9b82de08f59de5be5a34e42170";

pub(super) fn cpu_sqrt_seeds(device: &crate::VulkanDevice) -> Result<crate::GpuBuffer> {
    let bytes = include_bytes!("../shaders/deepseek_v41_rsqrt_seed_amd_zen4.bin");
    let seeds: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    crate::GpuBuffer::from_u32(device, &seeds)
}

/// Preserve the complete package metadata, including quantization and vision
/// fields, while decoding the language graph independently of Transformers.
#[derive(Debug, Clone)]
pub struct DeepseekV41Config {
    pub source_config: Value,
    pub text: DeepseekV41TextConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DeepseekV41RopeScaling {
    pub rope_type: String,
    pub factor: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub original_max_position_embeddings: usize,
}

/// Required shape fields are deliberately not filled with V4 defaults.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DeepseekV41TextConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub hidden_act: String,
    pub swiglu_limit: f64,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rope_scaling: Option<DeepseekV41RopeScaling>,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub scoring_func: String,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f64,
    pub sliding_window: usize,
    pub compress_ratios: Vec<usize>,
    pub compress_rope_theta: f64,
    pub kv_source_layer_ids: Vec<usize>,
    pub index_source_layer_ids: Vec<usize>,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    #[serde(default = "disabled_candidate_source")]
    pub candidate_source_layer_id: i64,
    #[serde(default)]
    pub candidate_topk_blocks: usize,
    #[serde(default)]
    pub candidate_block_size: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f64,
    #[serde(default)]
    pub engram_layer_ids: Vec<usize>,
    #[serde(default)]
    pub engram_num_embeddings: Vec<usize>,
    #[serde(default)]
    pub engram_max_ngram_size: usize,
    #[serde(default)]
    pub engram_vocab_size: usize,
    #[serde(default)]
    pub engram_n_heads: usize,
    #[serde(default)]
    pub engram_head_dim: usize,
    #[serde(default)]
    pub engram_pad_token_id: usize,
    #[serde(default)]
    pub engram_compressed_vocab_size: usize,
    #[serde(default)]
    pub num_nextn_predict_layers: usize,
    #[serde(default)]
    pub dspark_block_size: usize,
    #[serde(default)]
    pub dspark_noise_token_id: usize,
    #[serde(default)]
    pub dspark_target_layer_ids: Vec<usize>,
    #[serde(default)]
    pub dspark_markov_rank: usize,
    #[serde(default)]
    pub dspark_n_routed_experts: usize,
    #[serde(default)]
    pub dspark_num_experts_per_tok: usize,
}

fn disabled_candidate_source() -> i64 {
    -1
}

/// All layer ids are zero based, matching the released inference code.
/// A consumer reads the most recent source, rather than a periodic schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepseekV41LayerPlan {
    pub layer_id: usize,
    pub compress_ratio: usize,
    pub kv_source_layer: Option<usize>,
    pub index_source_layer: Option<usize>,
    pub candidate_source_layer: Option<usize>,
    pub publishes_candidates: bool,
    pub engram_table_index: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepseekV41Plan {
    pub backbone: Vec<DeepseekV41LayerPlan>,
    pub draft_layer_count: usize,
}

impl DeepseekV41Config {
    pub fn from_hf_value(value: &Value) -> Result<Self> {
        let text = match value.get("model_type").and_then(Value::as_str) {
            Some("deepseek_v41") => value
                .get("text_config")
                .context("DeepSeek-V4.1 package is missing text_config")?,
            Some("deepseek_v41_text") => value,
            other => bail!("expected DeepSeek-V4.1 model_type, got {other:?}"),
        };
        let text: DeepseekV41TextConfig =
            serde_json::from_value(text.clone()).context("decoding DeepSeek-V4.1 text_config")?;
        text.plan()?;
        Ok(Self {
            source_config: value.clone(),
            text,
        })
    }

    pub fn plan(&self) -> Result<DeepseekV41Plan> {
        self.text.plan()
    }
}

fn validate_ids(name: &str, ids: &[usize], layers: usize, ordered: bool) -> Result<()> {
    let mut seen = std::collections::BTreeSet::new();
    for (offset, &id) in ids.iter().enumerate() {
        if id >= layers || !seen.insert(id) {
            bail!("DeepSeek-V4.1 {name} contains duplicate or out-of-range layer {id}");
        }
        if ordered && offset > 0 && ids[offset - 1] >= id {
            bail!("DeepSeek-V4.1 {name} must be strictly increasing");
        }
    }
    Ok(())
}

impl DeepseekV41TextConfig {
    pub fn plan(&self) -> Result<DeepseekV41Plan> {
        if self.model_type != "deepseek_v41_text" {
            bail!(
                "expected DeepSeek-V4.1 text model_type, got {:?}",
                self.model_type
            );
        }
        for (name, size) in [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("moe_intermediate_size", self.moe_intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("head_dim", self.head_dim),
            ("q_lora_rank", self.q_lora_rank),
            ("o_lora_rank", self.o_lora_rank),
            ("o_groups", self.o_groups),
            ("max_position_embeddings", self.max_position_embeddings),
            ("n_routed_experts", self.n_routed_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
            ("sliding_window", self.sliding_window),
            ("index_n_heads", self.index_n_heads),
            ("index_head_dim", self.index_head_dim),
            ("index_topk", self.index_topk),
            ("hc_mult", self.hc_mult),
            ("hc_sinkhorn_iters", self.hc_sinkhorn_iters),
        ] {
            if size == 0 || size > u32::MAX as usize {
                bail!("DeepSeek-V4.1 {name} must be a positive u32");
            }
        }
        if self.num_key_value_heads != 1 {
            bail!("DeepSeek-V4.1 requires one shared key/value head");
        }
        if !self.num_attention_heads.is_multiple_of(self.o_groups) {
            bail!("DeepSeek-V4.1 o_groups must divide num_attention_heads");
        }
        self.num_attention_heads
            .checked_mul(self.head_dim)
            .context("DeepSeek-V4.1 query width overflow")?;
        self.o_groups
            .checked_mul(self.o_lora_rank)
            .context("DeepSeek-V4.1 output projection width overflow")?;
        if self.qk_rope_head_dim == 0
            || !self.qk_rope_head_dim.is_multiple_of(2)
            || self.qk_rope_head_dim > self.head_dim
            || self.qk_rope_head_dim > self.index_head_dim
        {
            bail!("DeepSeek-V4.1 qk_rope_head_dim must be positive, even, and fit attention/index heads");
        }
        for (name, value) in [
            ("rms_norm_eps", self.rms_norm_eps),
            ("hc_eps", self.hc_eps),
            ("rope_theta", self.rope_theta),
            ("compress_rope_theta", self.compress_rope_theta),
            ("routed_scaling_factor", self.routed_scaling_factor),
        ] {
            if !value.is_finite() || value <= 0.0 {
                bail!("DeepSeek-V4.1 {name} must be finite and positive");
            }
        }
        if !self.swiglu_limit.is_finite() || self.swiglu_limit < 0.0 || self.hidden_act != "silu" {
            bail!("DeepSeek-V4.1 expects silu and a finite nonnegative swiglu_limit");
        }
        if self.num_experts_per_tok > self.n_routed_experts
            || !matches!(
                self.scoring_func.as_str(),
                "sqrtsoftplus" | "sigmoid" | "softmax"
            )
        {
            bail!("DeepSeek-V4.1 invalid MoE scoring_func or num_experts_per_tok");
        }
        if let Some(rope) = &self.rope_scaling {
            if rope.rope_type != "yarn"
                || rope.original_max_position_embeddings == 0
                || !rope.factor.is_finite()
                || rope.factor <= 0.0
                || !rope.beta_fast.is_finite()
                || !rope.beta_slow.is_finite()
                || rope.beta_slow <= 0.0
                || rope.beta_fast <= rope.beta_slow
            {
                bail!("DeepSeek-V4.1 invalid YaRN rope_scaling");
            }
        }
        let total_layers = self
            .num_hidden_layers
            .checked_add(self.num_nextn_predict_layers)
            .context("DeepSeek-V4.1 total layer count overflow")?;
        if self.compress_ratios.len() != total_layers {
            bail!("DeepSeek-V4.1 compress_ratios must contain {total_layers} entries including draft layers");
        }
        for (layer, &ratio) in self.compress_ratios.iter().enumerate() {
            if !matches!(ratio, 0 | 1 | 2) {
                bail!("DeepSeek-V4.1 compress_ratios[{layer}]={ratio}; expected 0, 1, or 2");
            }
            if layer >= self.num_hidden_layers && ratio != 0 {
                bail!("DeepSeek-V4.1 DSpark draft layers require compress_ratio=0");
            }
        }
        validate_ids(
            "kv_source_layer_ids",
            &self.kv_source_layer_ids,
            self.num_hidden_layers,
            true,
        )?;
        validate_ids(
            "index_source_layer_ids",
            &self.index_source_layer_ids,
            self.num_hidden_layers,
            true,
        )?;
        validate_ids(
            "engram_layer_ids",
            &self.engram_layer_ids,
            self.num_hidden_layers,
            false,
        )?;
        validate_ids(
            "dspark_target_layer_ids",
            &self.dspark_target_layer_ids,
            self.num_hidden_layers,
            false,
        )?;
        for &layer in &self.kv_source_layer_ids {
            if !self.index_source_layer_ids.contains(&layer) {
                bail!("DeepSeek-V4.1 KV source {layer} must also own an indexer to publish index keys");
            }
        }
        let candidate = match self.candidate_source_layer_id {
            -1 => None,
            id if id >= 0 => Some(usize::try_from(id).context("candidate source layer overflow")?),
            _ => bail!("DeepSeek-V4.1 candidate_source_layer_id must be -1 or a backbone layer"),
        };
        if let Some(id) = candidate {
            if !self.index_source_layer_ids.contains(&id)
                || self.candidate_topk_blocks == 0
                || self.candidate_block_size == 0
            {
                bail!("DeepSeek-V4.1 candidate source must own an indexer and have positive block settings");
            }
        }
        if self.engram_layer_ids.len() != self.engram_num_embeddings.len() {
            bail!("DeepSeek-V4.1 engram_num_embeddings must match engram_layer_ids");
        }
        if !self.engram_layer_ids.is_empty()
            && (self.engram_max_ngram_size < 2
                || self.engram_vocab_size == 0
                || self.engram_n_heads == 0
                || self.engram_head_dim == 0
                || self.engram_compressed_vocab_size == 0
                || self.engram_compressed_vocab_size > self.vocab_size
                || self.engram_pad_token_id >= self.vocab_size
                || self.engram_num_embeddings.contains(&0))
        {
            bail!("DeepSeek-V4.1 invalid Engram dimensions, token ids, or table sizes");
        }
        if self.dspark_block_size > 0
            && (self.num_nextn_predict_layers == 0
                || self.dspark_target_layer_ids.is_empty()
                || self.dspark_noise_token_id >= self.vocab_size
                || self.dspark_markov_rank == 0
                || self.dspark_num_experts_per_tok > self.dspark_n_routed_experts)
        {
            bail!("DeepSeek-V4.1 invalid DSpark dimensions, targets, or token ids");
        }

        let mut backbone = Vec::with_capacity(self.num_hidden_layers);
        let mut kv_source = None;
        let mut index_source = None;
        for layer in 0..self.num_hidden_layers {
            let ratio = self.compress_ratios[layer];
            let owns_kv = self.kv_source_layer_ids.contains(&layer);
            let owns_index = self.index_source_layer_ids.contains(&layer);
            if ratio == 0 && (owns_kv || owns_index) {
                bail!("DeepSeek-V4.1 sliding-only layer {layer} cannot publish compressed KV or indices");
            }
            if owns_kv {
                kv_source = Some(layer);
            }
            if owns_index {
                index_source = Some(layer);
            }
            let (kv, index, candidates) = if ratio > 0 {
                let kv = kv_source.with_context(|| {
                    format!("DeepSeek-V4.1 layer {layer} has no preceding KV source")
                })?;
                let index = index_source.with_context(|| {
                    format!("DeepSeek-V4.1 layer {layer} has no preceding index source")
                })?;
                if self.compress_ratios[kv] != ratio || self.compress_ratios[index] != ratio {
                    bail!("DeepSeek-V4.1 layer {layer} cannot reuse a source with a different compress_ratio");
                }
                let candidates = candidate.filter(|&source| source < index);
                if let Some(source) = candidates {
                    if self.compress_ratios[source] != ratio {
                        bail!("DeepSeek-V4.1 layer {layer} candidate source has a different compress_ratio");
                    }
                }
                (Some(kv), Some(index), candidates)
            } else {
                (None, None, None)
            };
            backbone.push(DeepseekV41LayerPlan {
                layer_id: layer,
                compress_ratio: ratio,
                kv_source_layer: kv,
                index_source_layer: index,
                candidate_source_layer: candidates,
                publishes_candidates: candidate == Some(layer),
                engram_table_index: self.engram_layer_ids.iter().position(|&id| id == layer),
            });
        }
        Ok(DeepseekV41Plan {
            backbone,
            draft_layer_count: self.num_nextn_predict_layers,
        })
    }
}

#[cfg(test)]
mod attention_tests;
#[cfg(test)]
mod backbone_tests;
#[cfg(test)]
mod checkpoint_tests;
#[cfg(test)]
mod compressed_kv_tests;
#[cfg(test)]
mod compression_tests;
#[cfg(test)]
mod engram_tests;
#[cfg(test)]
mod indexer_tests;
#[cfg(test)]
mod mixed_tests;
#[cfg(test)]
mod packed_weights_tests;
#[cfg(test)]
mod quantization_tests;
#[cfg(test)]
mod rotary_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod unary_tests;
