//! Dedicated text-only native language-model path. Token lookup/HC expansion,
//! the backbone, final RMSNorm, and every vocabulary logit run on Vulkan.
//! `from_hf_package` is the normal loader for a released package. It exposes
//! full logits for qualification and does not route through the generic
//! DeepSeek-V4 transformer. Explicit checkpoint execution supports F32/hybrid
//! and BF16 streams, with complete-logit CPU PyTorch fixtures for packed
//! attention, experts and Engram projections.

use super::indexer::{HalfLinearKernel, LinearKernel, NormKernel};
use super::{
    DeepseekV41AttentionRuntimeConfig, DeepseekV41Backbone, DeepseekV41BackboneLayerWeights,
    DeepseekV41Checkpoint, DeepseekV41Config, DeepseekV41EngramRuntime, DeepseekV41EngramTokens,
    DeepseekV41LoraAdapter, DeepseekV41MixedLinearKernel, DeepseekV41WeightOutput,
};
use crate::{vulkan, GpuBuffer, VulkanDevice};
use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::{Arc, Mutex};

const EMBEDDING: &[u8] = include_bytes!("../../shaders/transformer_deepseek_v41_embedding.spv");
const EMBEDDING_HALF: &[u8] =
    include_bytes!("../../shaders/transformer_deepseek_v41_embedding_half.spv");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EmbeddingPush {
    rows: u32,
    dim: u32,
    hc: u32,
    vocab: u32,
}

pub struct DeepseekV41ModelWeights {
    pub embedding: Vec<f32>,
    pub norm: Vec<f32>,
    pub head: Vec<f32>,
    pub layers: Vec<DeepseekV41BackboneLayerWeights>,
}

/// Vocabulary payloads are retained by default, or staged for their individual
/// lookup/projection. Streamed checkpoints must stay readable and unchanged.
/// `Resident` is honored only when both released 1.32 GB tables fit beside the
/// already-loaded backbone; otherwise the loader stages them, so a request that
/// cannot be allocated never reaches the allocation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeepseekV41VocabularyResidency {
    #[default]
    Resident,
    Streamed,
}

enum VocabularySource {
    Resident {
        embedding: GpuBuffer,
        head: GpuBuffer,
    },
    /// Bounded staging: each operation reads only the rows it needs from the
    /// package (SSD, through the host page cache) and uploads at most one tile.
    /// The released 129280x5120 tables are 1.32 GB each, so this is the tier
    /// that keeps the vocabulary affordable on a small device.
    Tiered(Mutex<DeepseekV41Checkpoint>),
}

/// Rows of the LM head projected per dispatch in the tiered vocabulary tier:
/// 8192x5120 BF16 is an 84 MB device tile.
const VOCABULARY_TILE_ROWS: usize = 8_192;
/// Device headroom kept free for logits, staging and driver bookkeeping before a
/// resident vocabulary is accepted.
const VOCABULARY_RESERVE_BYTES: u64 = 512 * 1024 * 1024;

#[path = "model_train.rs"]
mod training;
pub use training::{
    DeepseekV41ModelGradients, DeepseekV41ModelTrainResult, DeepseekV41ModelTrainer,
    DeepseekV41ModelTrainingConfig, DeepseekV41TrainingBatch,
};

/// Whether both vocabulary tables fit beside everything already resident, so the
/// `Resident` tier can be honored. An unknown budget is treated as not fitting:
/// the tiered reader produces the same values with a bounded footprint.
fn vocabulary_tables_fit(device: &VulkanDevice, vocab: usize, hidden: usize, half: bool) -> bool {
    let item = if half { 2u64 } else { 4u64 };
    let table = (vocab as u64)
        .checked_mul(hidden as u64)
        .and_then(|n| n.checked_mul(item));
    let tile = (VOCABULARY_TILE_ROWS.min(vocab).max(1) as u64)
        .checked_mul(hidden as u64)
        .and_then(|n| n.checked_mul(item));
    let (Some(table), Some(tile)) = (table, tile) else {
        return false;
    };
    let needed = table
        .saturating_mul(2)
        .saturating_add(tile)
        .saturating_add(VOCABULARY_RESERVE_BYTES);
    let available = match device.memory_budget() {
        Ok(budget) if budget.device_local_available_bytes > 0 => {
            budget.device_local_available_bytes
        }
        Ok(budget) if budget.device_local_heap_size_bytes > budget.device_local_usage_bytes => {
            budget.device_local_heap_size_bytes - budget.device_local_usage_bytes
        }
        _ => return false,
    };
    needed <= available
}

/// Complete `[batch_size, seq_len, vocab_size]` logits, including every prefill
/// position. The released generation wrapper normally selects only the last
/// position; qualification must compare the complete tensor instead.
pub struct DeepseekV41Logits {
    pub values: GpuBuffer,
    pub batch_size: usize,
    pub seq_len: usize,
    pub vocab_size: usize,
}
impl DeepseekV41Logits {
    pub fn read_all(&self) -> Result<Vec<f32>> {
        let count = self
            .batch_size
            .checked_mul(self.seq_len)
            .and_then(|n| n.checked_mul(self.vocab_size))
            .context("V4.1 logit tensor geometry overflow")?;
        if count == 0 {
            bail!("V4.1 logit tensor must be nonempty");
        }
        self.values.read_f32(count)
    }
}

pub struct DeepseekV41CausalLM {
    checkpoint_io: Option<super::checkpoint::CacheHandle>,
    device: VulkanDevice,
    config: DeepseekV41Config,
    runtime: DeepseekV41AttentionRuntimeConfig,
    backbone: DeepseekV41Backbone,
    vocabulary: VocabularySource,
    norm_weight: GpuBuffer,
    /// Vocabulary tensors are BF16-exact; BF16 streams keep them as 16-bit
    /// pairs instead of promoting to FP32 storage.
    half_vocabulary: bool,
    /// Attached PEFT modules, resident or streamed with their projection.
    lora_modules: usize,
    lookup: vulkan::ComputeKernel,
    lookup_half: vulkan::ComputeKernel,
    norm: NormKernel,
    linear: LinearKernel,
    linear_half: HalfLinearKernel,
    cast: DeepseekV41MixedLinearKernel,
    precision: DeepseekV41WeightOutput,
    next_position: Option<usize>,
    needs_prefill: bool,
    #[cfg(test)]
    final_trace: Option<(GpuBuffer, GpuBuffer)>,
}

impl DeepseekV41CausalLM {
    /// Normal package loading: infer storage precision and every residency policy
    /// from the checkpoint itself, and derive Engram's non-persistent metadata
    /// from inline config fields, `engram_tokens.json`, or `tokenizer.json`.
    /// No execution policy requires an opt-in flag.
    ///
    /// The released DeepSeek-V4.1 repository publishes weights, config and prompt
    /// encoding but no tokenizer files. Supply its matching tokenizer.json (or a
    /// precomputed engram_tokens.json) beside the checkpoint.
    pub fn from_hf_package(
        device: VulkanDevice,
        directory: impl AsRef<std::path::Path>,
        batch_size: usize,
        max_positions: usize,
    ) -> Result<Self> {
        Self::from_hf_package_with_lora(device, directory, batch_size, max_positions, None)
    }

    /// Additive PEFT-LoRA on top of the normal package. The adapter is applied
    /// at each targeted projection, so the released packed bytes and the
    /// unadapted loader default stay unchanged.
    pub fn from_hf_package_with_lora(
        device: VulkanDevice,
        directory: impl AsRef<std::path::Path>,
        batch_size: usize,
        max_positions: usize,
        adapter: Option<&std::path::Path>,
    ) -> Result<Self> {
        Self::from_hf_package_with_lora_and_streaming(
            device,
            directory,
            batch_size,
            max_positions,
            adapter,
            true,
        )
    }

    /// Explicit application residency policy. CLI/GUI callers pass false by
    /// default and true only for the user's parameter-streaming opt-in.
    /// The historical package constructor retains its existing tiered policy.
    pub fn from_hf_package_with_lora_and_streaming(
        device: VulkanDevice,
        directory: impl AsRef<std::path::Path>,
        batch_size: usize,
        max_positions: usize,
        adapter: Option<&std::path::Path>,
        streaming: bool,
    ) -> Result<Self> {
        let directory = directory.as_ref();
        let mut checkpoint = DeepseekV41Checkpoint::open(directory)?;
        if let Some(adapter) = adapter {
            checkpoint.set_lora_adapter(Arc::new(DeepseekV41LoraAdapter::load(adapter)?))?;
        }
        let text = &checkpoint.config.text;
        if batch_size == 0 || max_positions == 0 {
            bail!("V4.1 package loading requires positive batch and position capacity");
        }
        let hash_cols = (text.engram_max_ngram_size.saturating_sub(1) * text.engram_n_heads).max(1);
        let max_prefill = max_positions
            .min(2048)
            .min(65535 / hash_cols / batch_size)
            .min(65535 / text.num_experts_per_tok.max(text.hc_mult) / batch_size);
        let tokens = if text.engram_layer_ids.is_empty() {
            None
        } else {
            let source = &checkpoint.config.source_config;
            let source = source.get("text_config").unwrap_or(source);
            if let (Some(map), Some(multipliers)) = (
                source.get("engram_token_map"),
                source.get("engram_hash_multipliers"),
            ) {
                Some(DeepseekV41EngramTokens {
                    token_map: serde_json::from_value(map.clone())?,
                    multipliers: serde_json::from_value(multipliers.clone())?,
                    pad_token_id: text.engram_pad_token_id,
                })
            } else if directory.join("engram_tokens.json").is_file() {
                Some(
                    serde_json::from_slice(&std::fs::read(directory.join("engram_tokens.json"))?)
                        .context("decoding V4.1 engram_tokens.json")?,
                )
            } else {
                let path = directory.join("tokenizer.json");
                if !path.is_file() {
                    bail!(
                        "V4.1 Engram requires tokenizer.json or engram_tokens.json beside the checkpoint; the released package does not publish tokenizer files"
                    );
                }
                let tokenizer = tokenizers::Tokenizer::from_file(&path)
                    .map_err(|e| anyhow::anyhow!("V4.1 tokenizer.json: {e}"))?;
                Some(DeepseekV41EngramTokens::from_tokenizer(&tokenizer, text)?)
            }
        };
        let precision = match checkpoint.tensor_dtype("embed.weight")?.as_str() {
            "BF16" => DeepseekV41WeightOutput::Bf16,
            "F32" => DeepseekV41WeightOutput::Fp32,
            other => bail!("V4.1 embedding requires BF16/F32 storage, got {other}"),
        };
        Self::from_checkpoint_with_vocabulary_residency(
            device,
            &mut checkpoint,
            DeepseekV41AttentionRuntimeConfig {
                batch_size,
                max_prefill,
                max_positions,
            },
            tokens,
            precision,
            if streaming {
                super::DeepseekV41ExpertResidency::StreamedAllExperts
            } else {
                super::DeepseekV41ExpertResidency::Resident
            },
            if streaming {
                super::DeepseekV41EngramResidency::StreamedTables
            } else {
                super::DeepseekV41EngramResidency::Resident
            },
            if streaming {
                super::DeepseekV41ProjectionResidency::Streamed
            } else {
                super::DeepseekV41ProjectionResidency::Resident
            },
            if streaming {
                DeepseekV41VocabularyResidency::Streamed
            } else {
                DeepseekV41VocabularyResidency::Resident
            },
        )
    }

    /// Last-position logits for generation, without allocating a vocabulary
    /// tensor for every prompt position. Long prompts retain the cache and
    /// continue through the released one-token decode path.
    pub fn prefill_last(&mut self, ids: &[u32]) -> Result<DeepseekV41Logits> {
        if self.runtime.batch_size != 1 || ids.is_empty() || ids.len() > self.runtime.max_positions
        {
            bail!("V4.1 generation prefill requires a nonempty single sequence within the context limit");
        }
        let first = ids.len().min(self.runtime.max_prefill);
        self.validate_input(&ids[..first], first, 0, None)?;
        let mut output = self.run(&ids[..first], first, 0, None, true, true)?;
        for (pos, token) in ids.iter().enumerate().skip(first) {
            output = self.decode(std::slice::from_ref(token), pos, None)?;
        }
        Ok(output)
    }

    pub fn generate_sequences_ids(
        &mut self,
        prompt: &[u32],
        config: &crate::VulkanTransformerGenerationConfig,
    ) -> Result<Vec<Vec<u32>>> {
        use crate::transformer::{
            generation_argmax, generation_processed_non_beam_scores,
            generation_sample_processed_scores,
        };
        config.validate()?;
        if config.num_beams != 1
            || config.penalty_alpha.is_some()
            || config.guidance_scale.is_some_and(|x| x != 1.0)
            || config.dola_layers.is_some()
        {
            bail!("V4.1 currently supports native greedy and sampling generation; the requested search mode is unavailable");
        }
        let steps = config.effective_max_new_tokens(prompt.len(), self.runtime.max_positions);
        let started = std::time::Instant::now();
        let mut sequences = Vec::new();
        for sample in 0..config.num_return_sequences {
            let mut history = prompt.to_vec();
            let mut generated = Vec::new();
            if steps != 0 {
                let mut logits = self.prefill_last(prompt)?.read_all()?;
                for step in 0..steps {
                    if config
                        .max_time
                        .is_some_and(|limit| started.elapsed().as_secs_f64() >= limit)
                    {
                        break;
                    }
                    let scores = generation_processed_non_beam_scores(
                        &logits,
                        &history,
                        config,
                        step,
                        step + 1 == steps,
                    )?;
                    let token = if config.do_sample {
                        generation_sample_processed_scores(
                            &scores,
                            config.seed.wrapping_add(sample as u64),
                            step as u64,
                        )?
                    } else {
                        generation_argmax(&scores)?
                    };
                    let stop = config.should_stop_after_token(&history, token);
                    generated.push(token);
                    history.push(token);
                    if stop || step + 1 == steps {
                        break;
                    }
                    logits = if config.use_cache {
                        self.decode(&[token], history.len() - 1, None)?.read_all()?
                    } else {
                        self.prefill_last(&history)?.read_all()?
                    };
                }
            }
            sequences.push(generated);
        }
        Ok(sequences)
    }

    pub fn new(
        device: VulkanDevice,
        config: &DeepseekV41Config,
        runtime: DeepseekV41AttentionRuntimeConfig,
        weights: DeepseekV41ModelWeights,
        engram: Option<DeepseekV41EngramRuntime>,
    ) -> Result<Self> {
        Self::validate_runtime(config, runtime)?;
        let backbone =
            DeepseekV41Backbone::new(device.clone(), config, runtime, weights.layers, engram)?;
        let (embedding, norm, head) = Self::upload_vocabulary(
            &device,
            config,
            weights.embedding,
            weights.norm,
            weights.head,
        )?;
        Self::assemble(
            device,
            config,
            runtime,
            backbone,
            VocabularySource::Resident { embedding, head },
            norm,
            false,
            DeepseekV41WeightOutput::Fp32,
            0,
        )
    }

    /// F32/BF16 checkpoint path. Layer payloads are uploaded individually, and
    /// each Engram table is released on the host after upload. Its tokenizer map
    /// and seeded multipliers are required separately because they are not
    /// persistent tensors in the released checkpoint. Defaults to the measured
    /// F32/hybrid stream contract; BF16 execution must be selected explicitly.
    pub fn from_checkpoint(
        device: VulkanDevice,
        checkpoint: &mut DeepseekV41Checkpoint,
        runtime: DeepseekV41AttentionRuntimeConfig,
        tokens: Option<DeepseekV41EngramTokens>,
    ) -> Result<Self> {
        Self::from_checkpoint_with_precision(
            device,
            checkpoint,
            runtime,
            tokens,
            DeepseekV41WeightOutput::Fp32,
        )
    }

    /// Explicit stream precision, including every residual/HC and norm boundary.
    /// BF16 values use promoted FP32 Vulkan storage after each required rounding.
    pub fn from_checkpoint_with_precision(
        device: VulkanDevice,
        checkpoint: &mut DeepseekV41Checkpoint,
        runtime: DeepseekV41AttentionRuntimeConfig,
        tokens: Option<DeepseekV41EngramTokens>,
        precision: DeepseekV41WeightOutput,
    ) -> Result<Self> {
        Self::from_checkpoint_with_expert_residency(
            device,
            checkpoint,
            runtime,
            tokens,
            precision,
            super::DeepseekV41ExpertResidency::Resident,
        )
    }

    /// Explicit expert residency. Streamed modes read and upload one
    /// packed expert at a time; activation math and routing remain on Vulkan.
    /// `Streamed` retains shared experts; `StreamedAllExperts` stages them too.
    /// Attention, vocabulary and Engram tables stay resident.
    /// Keep checkpoint files readable and unchanged for the model's lifetime.
    pub fn from_checkpoint_with_expert_residency(
        device: VulkanDevice,
        checkpoint: &mut DeepseekV41Checkpoint,
        runtime: DeepseekV41AttentionRuntimeConfig,
        tokens: Option<DeepseekV41EngramTokens>,
        precision: DeepseekV41WeightOutput,
        residency: super::DeepseekV41ExpertResidency,
    ) -> Result<Self> {
        Self::from_checkpoint_with_residency(
            device,
            checkpoint,
            runtime,
            tokens,
            precision,
            residency,
            super::DeepseekV41EngramResidency::Resident,
            super::DeepseekV41ProjectionResidency::Resident,
        )
    }

    /// Explicit expert, Engram-table and layer-projection residency. Streamed
    /// tables are read and uploaded per Engram layer application; streamed
    /// projections are read and uploaded per attention layer execution.
    /// Vocabulary, norms, gates and Hyper-Connection weights remain resident.
    /// Keep checkpoint files readable and unchanged for the model's lifetime.
    #[allow(clippy::too_many_arguments)]
    pub fn from_checkpoint_with_residency(
        device: VulkanDevice,
        checkpoint: &mut DeepseekV41Checkpoint,
        runtime: DeepseekV41AttentionRuntimeConfig,
        tokens: Option<DeepseekV41EngramTokens>,
        precision: DeepseekV41WeightOutput,
        residency: super::DeepseekV41ExpertResidency,
        engram_residency: super::DeepseekV41EngramResidency,
        projections: super::DeepseekV41ProjectionResidency,
    ) -> Result<Self> {
        Self::from_checkpoint_with_vocabulary_residency(
            device,
            checkpoint,
            runtime,
            tokens,
            precision,
            residency,
            engram_residency,
            projections,
            DeepseekV41VocabularyResidency::Resident,
        )
    }

    /// Adds vocabulary residency to the existing independent weight policies.
    /// Streamed mode stages the embedding only for lookup, releases it before
    /// the backbone, then stages the head only for the final Vulkan projection.
    /// Norms, gates, Hyper-Connections, caches and Engram projections remain resident.
    #[allow(clippy::too_many_arguments)]
    pub fn from_checkpoint_with_vocabulary_residency(
        device: VulkanDevice,
        checkpoint: &mut DeepseekV41Checkpoint,
        runtime: DeepseekV41AttentionRuntimeConfig,
        tokens: Option<DeepseekV41EngramTokens>,
        precision: DeepseekV41WeightOutput,
        residency: super::DeepseekV41ExpertResidency,
        engram_residency: super::DeepseekV41EngramResidency,
        projections: super::DeepseekV41ProjectionResidency,
        vocabulary_residency: DeepseekV41VocabularyResidency,
    ) -> Result<Self> {
        let config = checkpoint.config.clone();
        if residency != super::DeepseekV41ExpertResidency::Resident
            || projections != super::DeepseekV41ProjectionResidency::Resident
            || engram_residency != super::DeepseekV41EngramResidency::Resident
            || vocabulary_residency == DeepseekV41VocabularyResidency::Streamed
        {
            checkpoint.enable_payload_cache()?;
        }
        Self::validate_runtime(&config, runtime)?;
        if vocabulary_residency == DeepseekV41VocabularyResidency::Streamed {
            checkpoint.validate_vocabulary_metadata(precision == DeepseekV41WeightOutput::Bf16)?;
        }
        let engram = match (config.text.engram_layer_ids.is_empty(), tokens) {
            (false, Some(tokens)) => Some(
                DeepseekV41EngramRuntime::from_checkpoint_with_table_residency(
                    device.clone(),
                    checkpoint,
                    tokens,
                    runtime.batch_size,
                    runtime.max_positions,
                    precision,
                    engram_residency,
                )?,
            ),
            (false, None) => {
                bail!("V4.1 Engram requires matching converter token-map/hash artifacts")
            }
            (true, Some(_)) => bail!("V4.1 Engram artifacts provided without an Engram schedule"),
            (true, None) => None,
        };
        let backbone = DeepseekV41Backbone::from_checkpoint_with_residency(
            device.clone(),
            checkpoint,
            runtime,
            engram,
            precision,
            residency,
            projections,
        )?;
        let t = &config.text;
        let vocabulary = if vocabulary_residency == DeepseekV41VocabularyResidency::Streamed
            || !vocabulary_tables_fit(
                &device,
                t.vocab_size,
                t.hidden_size,
                precision == DeepseekV41WeightOutput::Bf16,
            ) {
            VocabularySource::Tiered(Mutex::new(checkpoint.clone()))
        } else if precision == DeepseekV41WeightOutput::Bf16 {
            let embedding =
                checkpoint.read_bf16_words("embed.weight", &[t.vocab_size, t.hidden_size])?;
            let head = checkpoint.read_bf16_words("head.weight", &[t.vocab_size, t.hidden_size])?;
            VocabularySource::Resident {
                embedding: GpuBuffer::from_u32(&device, &embedding)?,
                head: GpuBuffer::from_u32(&device, &head)?,
            }
        } else {
            VocabularySource::Resident {
                embedding: GpuBuffer::from_f32(
                    &device,
                    &checkpoint.read_tensor("embed.weight", &[t.vocab_size, t.hidden_size])?,
                )?,
                head: GpuBuffer::from_f32(
                    &device,
                    &checkpoint.read_tensor("head.weight", &[t.vocab_size, t.hidden_size])?,
                )?,
            }
        };
        let norm = GpuBuffer::from_f32(
            &device,
            &checkpoint.read_tensor("norm.weight", &[t.hidden_size])?,
        )?;
        let lora_modules = checkpoint.lora_adapter().map_or(0, |adapter| adapter.len());
        let mut model = Self::assemble(
            device,
            &config,
            runtime,
            backbone,
            vocabulary,
            norm,
            precision == DeepseekV41WeightOutput::Bf16,
            precision,
            lora_modules,
        )?;
        model.checkpoint_io = Some(checkpoint.payload_cache_handle());
        Ok(model)
    }

    fn validate_runtime(
        config: &DeepseekV41Config,
        runtime: DeepseekV41AttentionRuntimeConfig,
    ) -> Result<()> {
        config.plan()?;
        let t = &config.text;
        let rows = runtime
            .batch_size
            .checked_mul(runtime.max_prefill)
            .context("V4.1 model row overflow")?;
        let width = t
            .hidden_size
            .checked_mul(t.hc_mult)
            .context("V4.1 model stream width overflow")?;
        let output = rows
            .checked_mul(t.vocab_size)
            .context("V4.1 model logits overflow")?;
        let stream = rows
            .checked_mul(width)
            .context("V4.1 model stream overflow")?;
        // Per-head-row kernels (rotary application, activation quantization) tile
        // their rows past one 65,535-workgroup dispatch axis, so the released
        // index-head count no longer caps a start-zero prefill. Every other
        // kernel still dispatches one workgroup per row.
        if rows == 0
            || runtime.max_prefill > runtime.max_positions
            || runtime.max_positions > t.max_position_embeddings
            || runtime.max_positions > 65535
            || rows > 65535
            || rows
                .checked_mul(t.num_experts_per_tok.max(t.hc_mult))
                .is_none_or(|v| v > 65535)
            || output > u32::MAX as usize
            || t.vocab_size
                .checked_mul(t.hidden_size)
                .is_none_or(|v| v > u32::MAX as usize)
            || stream > u32::MAX as usize
            || width.div_ceil(16) > 65535
            || t.vocab_size.div_ceil(16) > 65535
        {
            bail!("V4.1 model runtime exceeds context, addressing, or correctness-kernel dispatch capacity");
        }
        Ok(())
    }

    fn upload_vocabulary(
        device: &VulkanDevice,
        config: &DeepseekV41Config,
        embedding: Vec<f32>,
        norm: Vec<f32>,
        head: Vec<f32>,
    ) -> Result<(GpuBuffer, GpuBuffer, GpuBuffer)> {
        let t = &config.text;
        let count = t
            .vocab_size
            .checked_mul(t.hidden_size)
            .context("V4.1 vocabulary geometry overflow")?;
        if embedding.len() != count || head.len() != count || norm.len() != t.hidden_size {
            bail!("V4.1 vocabulary parameter geometry mismatch");
        }
        Ok((
            GpuBuffer::from_f32(device, &embedding)?,
            GpuBuffer::from_f32(device, &norm)?,
            GpuBuffer::from_f32(device, &head)?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble(
        device: VulkanDevice,
        config: &DeepseekV41Config,
        runtime: DeepseekV41AttentionRuntimeConfig,
        backbone: DeepseekV41Backbone,
        vocabulary: VocabularySource,
        norm_weight: GpuBuffer,
        half_vocabulary: bool,
        precision: DeepseekV41WeightOutput,
        lora_modules: usize,
    ) -> Result<Self> {
        Ok(Self {
            checkpoint_io: None,
            lookup: vulkan::ComputeKernel::new(
                &device,
                EMBEDDING,
                3,
                std::mem::size_of::<EmbeddingPush>() as u32,
            )?,
            lookup_half: vulkan::ComputeKernel::new(
                &device,
                EMBEDDING_HALF,
                3,
                std::mem::size_of::<EmbeddingPush>() as u32,
            )?,
            norm: NormKernel::new(&device)?,
            linear: LinearKernel::new(&device)?,
            linear_half: HalfLinearKernel::new(&device)?,
            cast: DeepseekV41MixedLinearKernel::new(device.clone())?,
            half_vocabulary,
            lora_modules,
            precision,
            device,
            config: config.clone(),
            runtime,
            backbone,
            vocabulary,
            norm_weight,
            next_position: None,
            needs_prefill: false,
            #[cfg(test)]
            final_trace: None,
        })
    }

    pub fn config(&self) -> &DeepseekV41Config {
        &self.config
    }
    pub fn checkpoint_io_stats(&self) -> Result<Option<super::DeepseekV41CheckpointIoStats>> {
        self.checkpoint_io
            .as_ref()
            .map(|cache| {
                Ok(cache
                    .lock()
                    .map_err(|_| anyhow::anyhow!("V4.1 payload cache lock poisoned"))?
                    .stats())
            })
            .transpose()
    }
    pub fn stream_precision(&self) -> DeepseekV41WeightOutput {
        self.precision
    }

    /// Validated additive PEFT modules attached to this graph.
    pub fn lora_module_count(&self) -> usize {
        self.lora_modules
    }

    /// Persistent embedding/head tensors; excludes per-operation staging and
    /// tiles, which never survive an operation.
    pub fn resident_vocabulary_tensor_count(&self) -> usize {
        match &self.vocabulary {
            VocabularySource::Resident { .. } => 2,
            VocabularySource::Tiered(_) => 0,
        }
    }

    /// Effective vocabulary tier. A `Resident` request degrades to bounded
    /// staging when both tables would not fit beside the loaded backbone, so the
    /// loader never starts an operation that cannot be allocated.
    pub fn vocabulary_residency(&self) -> DeepseekV41VocabularyResidency {
        match &self.vocabulary {
            VocabularySource::Resident { .. } => DeepseekV41VocabularyResidency::Resident,
            VocabularySource::Tiered(_) => DeepseekV41VocabularyResidency::Streamed,
        }
    }

    /// Persistent routed experts; excludes shared experts and temporary staging.
    pub fn resident_routed_expert_count(&self) -> usize {
        self.backbone.resident_routed_expert_count()
    }
    /// Persistent shared experts; excludes temporary checkpoint staging.
    pub fn resident_shared_expert_count(&self) -> usize {
        self.backbone.resident_shared_expert_count()
    }
    /// Persistent Engram tables; excludes per-lookup checkpoint staging.
    pub fn resident_engram_table_count(&self) -> usize {
        self.backbone.resident_engram_table_count()
    }
    /// Layers holding attention/indexer/compressor projections between executions.
    pub fn resident_projection_layer_count(&self) -> usize {
        self.backbone.resident_projection_layer_count()
    }
    pub fn next_position(&self) -> Option<usize> {
        if self.needs_prefill {
            None
        } else {
            self.next_position
        }
    }

    /// Start or replace a sequence. The optional 0/1 mask affects Engram lookup
    /// and gates only; image encoding and VL attention/router biases are absent.
    pub fn prefill(
        &mut self,
        token_ids: &[u32],
        seq: usize,
        engram_mask: Option<&[u32]>,
    ) -> Result<DeepseekV41Logits> {
        self.validate_input(token_ids, seq, 0, engram_mask)?;
        self.run(token_ids, seq, 0, engram_mask, true, false)
    }

    pub fn decode(
        &mut self,
        token_ids: &[u32],
        start_pos: usize,
        engram_mask: Option<&[u32]>,
    ) -> Result<DeepseekV41Logits> {
        if self.needs_prefill || self.next_position != Some(start_pos) {
            bail!("V4.1 model decode requires a completed prior step; restart with prefill after a component failure");
        }
        self.validate_input(token_ids, 1, start_pos, engram_mask)?;
        self.run(token_ids, 1, start_pos, engram_mask, false, false)
    }

    fn validate_input(
        &self,
        ids: &[u32],
        seq: usize,
        pos: usize,
        mask: Option<&[u32]>,
    ) -> Result<()> {
        let rows = self
            .runtime
            .batch_size
            .checked_mul(seq)
            .context("V4.1 model input overflow")?;
        let end = pos
            .checked_add(seq)
            .context("V4.1 model position overflow")?;
        if seq == 0
            || seq > self.runtime.max_prefill
            || end > self.runtime.max_positions
            || ids.len() != rows
            || ids
                .iter()
                .any(|&id| id as usize >= self.config.text.vocab_size)
        {
            bail!("V4.1 model token count, token id, or sequence position is invalid");
        }
        if let Some(mask) = mask {
            if self.config.text.engram_layer_ids.is_empty()
                || mask.len() != rows
                || mask.iter().any(|&v| v > 1)
            {
                bail!("V4.1 model Engram mask requires a configured runtime and one 0/1 entry per token");
            }
        }
        Ok(())
    }

    fn embed(&self, ids: &GpuBuffer, rows: usize) -> Result<GpuBuffer> {
        let t = &self.config.text;
        let shape = [t.vocab_size, t.hidden_size];
        // A resident table is addressed directly. The tiered tier gathers the
        // rows this call actually uses (one bounded selection, exactly as the
        // Engram lookup does) and remaps the ids onto that compact table, so the
        // lookup kernel copies the same bits without a 1.32 GB resident table.
        let (tokens, embedding, vocab) = match &self.vocabulary {
            VocabularySource::Resident { embedding, .. } => {
                (ids.clone(), embedding.clone(), t.vocab_size)
            }
            VocabularySource::Tiered(checkpoint) => {
                let mut checkpoint = checkpoint
                    .lock()
                    .map_err(|_| anyhow::anyhow!("V4.1 vocabulary checkpoint lock poisoned"))?;
                let requested = ids.read_u32(rows)?;
                let unique: Vec<u32> = requested
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .collect();
                let remapped: Vec<u32> = requested
                    .iter()
                    .map(|id| {
                        unique
                            .binary_search(id)
                            .map(|slot| slot as u32)
                            .unwrap_or(u32::MAX)
                    })
                    .collect();
                let table = if self.half_vocabulary {
                    GpuBuffer::from_u32(
                        &self.device,
                        &checkpoint.read_bf16_rows("embed.weight", shape, &unique)?,
                    )?
                } else {
                    GpuBuffer::from_f32(
                        &self.device,
                        &checkpoint.read_f32_rows("embed.weight", shape, &unique)?,
                    )?
                };
                (
                    GpuBuffer::from_u32(&self.device, &remapped)?,
                    table,
                    unique.len(),
                )
            }
        };
        let stream = GpuBuffer::zeros_f32(&self.device, rows * t.hc_mult * t.hidden_size)?;
        let push = EmbeddingPush {
            rows: rows as u32,
            dim: t.hidden_size as u32,
            hc: t.hc_mult as u32,
            vocab: vocab as u32,
        };
        let mut commands = self.device.compute_batch()?;
        let lookup = if self.half_vocabulary {
            &self.lookup_half
        } else {
            &self.lookup
        };
        lookup.record_dispatch(
            &mut commands,
            &[&tokens, &embedding, &stream],
            bytemuck::bytes_of(&push),
            [
                (t.hc_mult * t.hidden_size).div_ceil(16) as u32,
                rows.div_ceil(16) as u32,
                1,
            ],
        )?;
        commands.submit()?;
        self.cast
            .cast_buffer(&stream, rows * t.hc_mult * t.hidden_size, self.precision)
    }

    fn project_logits(&self, hidden: &GpuBuffer, rows: usize) -> Result<GpuBuffer> {
        let t = &self.config.text;
        if let VocabularySource::Resident { head, .. } = &self.vocabulary {
            return if self.half_vocabulary {
                self.linear_half.apply(
                    &self.device,
                    hidden,
                    head,
                    rows,
                    t.hidden_size,
                    t.vocab_size,
                )
            } else {
                self.linear.apply(
                    &self.device,
                    hidden,
                    head,
                    rows,
                    t.hidden_size,
                    t.vocab_size,
                )
            };
        }
        let VocabularySource::Tiered(checkpoint) = &self.vocabulary else {
            unreachable!("vocabulary source handled by the resident branch")
        };
        let shape = [t.vocab_size, t.hidden_size];
        let mut checkpoint = checkpoint
            .lock()
            .map_err(|_| anyhow::anyhow!("V4.1 vocabulary checkpoint lock poisoned"))?;
        // A table that fits one tile runs the qualified dispatch exactly as before;
        // only wider vocabularies are split, so every existing graph keeps its
        // byte-identical execution shape.
        if t.vocab_size <= VOCABULARY_TILE_ROWS {
            return if self.half_vocabulary {
                let head = GpuBuffer::from_u32(
                    &self.device,
                    &checkpoint.read_bf16_word_range("head.weight", shape, 0, t.vocab_size)?,
                )?;
                self.linear_half.apply(
                    &self.device,
                    hidden,
                    &head,
                    rows,
                    t.hidden_size,
                    t.vocab_size,
                )
            } else {
                let head = GpuBuffer::from_f32(
                    &self.device,
                    &checkpoint.read_f32_row_range("head.weight", shape, 0, t.vocab_size)?,
                )?;
                self.linear.apply(
                    &self.device,
                    hidden,
                    &head,
                    rows,
                    t.hidden_size,
                    t.vocab_size,
                )
            };
        }
        // Each tile is projected by the unchanged kernel over `count` columns,
        // which selects the same dot-product body as the full-width dispatch
        // (`count` stays above eight), and then placed into the full logits. The
        // head therefore never holds more than one tile on the device.
        let outputs = rows
            .checked_mul(t.vocab_size)
            .context("V4.1 vocabulary logit overflow")?;
        let mut full = vec![0.0f32; outputs];
        let mut start = 0usize;
        while start < t.vocab_size {
            let mut count = VOCABULARY_TILE_ROWS.min(t.vocab_size - start);
            if t.vocab_size - (start + count) <= 8 {
                count = t.vocab_size - start;
            }
            let tile = if self.half_vocabulary {
                let weight = GpuBuffer::from_u32(
                    &self.device,
                    &checkpoint.read_bf16_word_range("head.weight", shape, start, count)?,
                )?;
                self.linear_half
                    .apply(&self.device, hidden, &weight, rows, t.hidden_size, count)?
            } else {
                let weight = GpuBuffer::from_f32(
                    &self.device,
                    &checkpoint.read_f32_row_range("head.weight", shape, start, count)?,
                )?;
                self.linear
                    .apply(&self.device, hidden, &weight, rows, t.hidden_size, count)?
            };
            let values = tile.read_f32(rows * count)?;
            for row in 0..rows {
                let destination = row * t.vocab_size + start;
                full[destination..destination + count]
                    .copy_from_slice(&values[row * count..(row + 1) * count]);
            }
            start += count;
        }
        GpuBuffer::from_f32(&self.device, &full)
    }

    fn run(
        &mut self,
        ids: &[u32],
        seq: usize,
        pos: usize,
        mask: Option<&[u32]>,
        prefill: bool,
        last_only: bool,
    ) -> Result<DeepseekV41Logits> {
        let result: Result<DeepseekV41Logits> = (|| {
            let rows = self.runtime.batch_size * seq;
            let ids = GpuBuffer::from_u32(&self.device, ids)?;
            let mask = mask
                .map(|m| GpuBuffer::from_u32(&self.device, m))
                .transpose()?;
            let stream = self.embed(&ids, rows)?;
            let engram_ids = (!self.config.text.engram_layer_ids.is_empty()).then_some(&ids);
            let hidden = if prefill {
                self.backbone
                    .prefill(&stream, seq, engram_ids, mask.as_ref())?
            } else {
                self.backbone
                    .decode(&stream, pos, engram_ids, mask.as_ref())?
            };
            let t = &self.config.text;
            #[cfg(test)]
            let collapsed_trace = hidden.clone();
            let hidden = self.norm.apply(
                &self.device,
                &hidden,
                &self.norm_weight,
                rows,
                t.hidden_size,
                t.rms_norm_eps as f32,
            )?;
            let hidden = self
                .cast
                .cast_buffer(&hidden, rows * t.hidden_size, self.precision)?;
            #[cfg(test)]
            {
                self.final_trace = Some((collapsed_trace, hidden.clone()));
            }
            let last;
            let (hidden, logit_rows) = if last_only {
                last = GpuBuffer::zeros_f32(&self.device, self.runtime.batch_size * t.hidden_size)?;
                let mut commands = self.device.compute_batch()?;
                for batch in 0..self.runtime.batch_size {
                    commands.copy_f32_range(
                        &hidden,
                        ((batch + 1) * seq - 1) * t.hidden_size,
                        &last,
                        batch * t.hidden_size,
                        t.hidden_size,
                    )?;
                }
                commands.submit()?;
                (&last, self.runtime.batch_size)
            } else {
                (&hidden, rows)
            };
            let logits = self.project_logits(hidden, logit_rows)?;
            Ok(DeepseekV41Logits {
                values: logits,
                batch_size: self.runtime.batch_size,
                seq_len: if last_only { 1 } else { seq },
                vocab_size: t.vocab_size,
            })
        })();
        match result {
            Ok(logits) => {
                self.next_position = Some(pos + seq);
                self.needs_prefill = false;
                Ok(logits)
            }
            Err(error) => {
                self.next_position = None;
                self.needs_prefill = true;
                Err(error.context("V4.1 model step failed; restart with prefill"))
            }
        }
    }
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "model_production_tests.rs"]
mod production_tests;

#[cfg(test)]
#[path = "model_lora_tests.rs"]
mod lora_tests;

#[cfg(test)]
#[path = "model_prefill_tests.rs"]
mod prefill_tests;

#[cfg(test)]
#[path = "model_schedule_width_tests.rs"]
mod schedule_width_tests;

#[cfg(test)]
#[path = "model_scalable_release_tests.rs"]
mod scalable_release_tests;
