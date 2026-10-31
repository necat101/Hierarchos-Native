//! Native end-to-end LoRA training. Base payloads stay immutable in the package;
//! the complete model reverse graph produces factors for the existing AdamW.
use super::*;
use crate::deepseek_v41::backbone::ModelAdjoints;
use crate::deepseek_v41::{
    DeepseekV41AdamWConfig, DeepseekV41AdapterGradient, DeepseekV41AdapterTrainer,
    DeepseekV41EngramResidency, DeepseekV41ExpertResidency, DeepseekV41ProjectionResidency,
    DeepseekV41TerminalAdjoints,
};
#[cfg(test)]
use crate::deepseek_v41::DeepseekV41AdapterFactorState;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepseekV41ModelTrainingConfig {
    pub batch_size: usize,
    pub stream_experts: bool,
    pub stream_projections: bool,
    pub stream_engram_tables: bool,
    pub stream_vocabulary: bool,
    pub max_activation_bytes: u64,
}
impl Default for DeepseekV41ModelTrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            stream_experts: true,
            stream_projections: true,
            stream_engram_tables: true,
            stream_vocabulary: true,
            max_activation_bytes: 512 * 1024 * 1024,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepseekV41TrainingBatch {
    pub token_ids: Vec<u32>,
    /// One label per token; u32::MAX is the standard ignored-label boundary.
    pub targets: Vec<u32>,
    pub seq_len: usize,
    pub engram_mask: Option<Vec<u32>>,
}
pub struct DeepseekV41ModelGradients {
    pub loss: f32,
    pub logits: Vec<DeepseekV41Logits>,
    pub adapters: Vec<DeepseekV41AdapterGradient>,
    pub estimated_activation_bytes: u64,
}
pub struct DeepseekV41ModelTrainResult {
    pub loss: f32,
    pub optimizer_step: u64,
    pub active_modules: usize,
    pub logits: Vec<DeepseekV41Logits>,
    pub estimated_activation_bytes: u64,
}
#[derive(Serialize, Deserialize)]
struct Binding {
    schema: u32,
    base_sha256: String,
    config: DeepseekV41ModelTrainingConfig,
}
struct FileStamp {
    path: PathBuf,
    len: u64,
    modified: Option<std::time::SystemTime>,
}

pub struct DeepseekV41ModelTrainer {
    device: VulkanDevice,
    package: DeepseekV41Checkpoint,
    directory: PathBuf,
    optimizer: DeepseekV41AdapterTrainer,
    options: DeepseekV41ModelTrainingConfig,
    tokens: Option<DeepseekV41EngramTokens>,
    precision: DeepseekV41WeightOutput,
    identity: String,
    stamps: Vec<FileStamp>,
    kernels: ModelAdjoints,
    terminal: DeepseekV41TerminalAdjoints,
    head_gradient: vulkan::ComputeKernel,
    head_gradient_short: vulkan::ComputeKernel,
}

impl DeepseekV41ModelTrainer {
    pub fn new(
        device: VulkanDevice,
        directory: impl AsRef<Path>,
        adapter: &DeepseekV41LoraAdapter,
        hyper: DeepseekV41AdamWConfig,
        options: DeepseekV41ModelTrainingConfig,
    ) -> Result<Self> {
        let optimizer = DeepseekV41AdapterTrainer::new(adapter, hyper)?;
        Self::build(device, directory.as_ref(), optimizer, options)
    }
    fn build(
        device: VulkanDevice,
        directory: &Path,
        optimizer: DeepseekV41AdapterTrainer,
        options: DeepseekV41ModelTrainingConfig,
    ) -> Result<Self> {
        if options.batch_size == 0 || options.max_activation_bytes == 0 {
            bail!("V4.1 training requires positive batch size and activation budget");
        }
        if optimizer.adapter()?.dropout() != 0.0 {
            bail!("V4.1 native training currently requires zero adapter dropout");
        }
        let directory = directory.canonicalize()?;
        let mut package = DeepseekV41Checkpoint::open(&directory)?;
        package.set_lora_adapter(Arc::new(optimizer.adapter()?))?;
        let precision = match package.tensor_dtype("embed.weight")?.as_str() {
            "F32" => DeepseekV41WeightOutput::Fp32,
            "BF16" => DeepseekV41WeightOutput::Bf16,
            _ => bail!("V4.1 native training embedding must be F32 or BF16"),
        };
        let tokens = load_tokens(&directory, &package.config)?;
        let (identity, stamps) = fingerprint(&package)?;
        use vulkan::BindingAccess::{ReadOnly as R, ReadWrite as RW};
        Ok(Self {
            kernels: ModelAdjoints::new(device.clone())?,
            terminal: DeepseekV41TerminalAdjoints::new(device.clone())?,
            head_gradient: vulkan::ComputeKernel::new_with_access(
                &device,
                include_bytes!("../../shaders/linear_input_grad_streamed.spv"),
                &[R, R, RW],
                16,
            )?,
            head_gradient_short: vulkan::ComputeKernel::new_with_access(
                &device,
                include_bytes!("../../shaders/linear_input_grad_cpu_short_streamed.spv"),
                &[R, R, RW],
                16,
            )?,
            device,
            package,
            directory,
            optimizer,
            options,
            tokens,
            precision,
            identity,
            stamps,
        })
    }
    pub fn optimizer(&self) -> &DeepseekV41AdapterTrainer {
        &self.optimizer
    }
    /// Qualification replay: pin one projection's factors and optimizer moments
    /// to the state the reference step started from, so every compared step is
    /// that step's own arithmetic rather than a continuation of an already
    /// gated rounding difference. See
    /// [`DeepseekV41AdapterTrainer::load_state_bits`].
    #[cfg(test)]
    pub fn load_adapter_state_bits(
        &mut self,
        target: &str,
        a_bits: &[u32],
        b_bits: &[u32],
        state_a: Option<DeepseekV41AdapterFactorState<'_>>,
        state_b: Option<DeepseekV41AdapterFactorState<'_>>,
    ) -> Result<()> {
        self.optimizer
            .load_state_bits(target, a_bits, b_bits, state_a, state_b)
    }
    pub fn training_config(&self) -> &DeepseekV41ModelTrainingConfig { &self.options }
    pub fn adapter(&self) -> Result<DeepseekV41LoraAdapter> {
        self.optimizer.adapter()
    }
    pub fn base_fingerprint(&self) -> &str {
        &self.identity
    }
    fn unchanged(&self) -> Result<()> {
        for stamp in &self.stamps {
            let meta = fs::metadata(&stamp.path)?;
            if meta.len() != stamp.len || meta.modified().ok() != stamp.modified {
                bail!(
                    "V4.1 frozen package changed during training: {}",
                    stamp.path.display()
                );
            }
        }
        Ok(())
    }
    fn estimate(&self, batches: &[DeepseekV41TrainingBatch]) -> Result<(usize, usize, u64)> {
        let t = &self.package.config.text;
        if batches.is_empty() {
            bail!("V4.1 training sequence has no batches");
        }
        let mut positions = 0usize;
        let mut rows = 0usize;
        let mut bytes = 0u128;
        let mut labels = 0usize;
        for (i, b) in batches.iter().enumerate() {
            if b.seq_len == 0 || (i > 0 && b.seq_len != 1) {
                bail!("V4.1 training uses one prefill followed by single-token decode batches");
            }
            let n = self
                .options
                .batch_size
                .checked_mul(b.seq_len)
                .context("V4.1 training row overflow")?;
            if b.token_ids.len() != n || b.targets.len() != n || n > 65_535 {
                bail!("V4.1 training batch geometry mismatch");
            }
            if b.token_ids.iter().any(|&id| id as usize >= t.vocab_size)
                || b.targets
                    .iter()
                    .any(|&id| id != u32::MAX && id as usize >= t.vocab_size)
            {
                bail!("V4.1 training token/target exceeds vocabulary");
            }
            if let Some(mask) = &b.engram_mask {
                if mask.len() != n || mask.iter().any(|&v| v > 1) {
                    bail!("V4.1 training Engram mask must be 0/1 in batch shape");
                }
            }
            labels += b.targets.iter().filter(|&&v| v != u32::MAX).count();
            positions = positions
                .checked_add(b.seq_len)
                .context("V4.1 training position overflow")?;
            rows = rows.checked_add(n).context("V4.1 training rows overflow")?;
            let visited = t
                .n_routed_experts
                .min(n.saturating_mul(t.num_experts_per_tok));
            bytes += n as u128
                * t.num_hidden_layers as u128
                * 4
                * (12 * t.hc_mult * t.hidden_size
                    + (visited + 1) * (4 * t.hidden_size + 6 * t.moe_intermediate_size))
                    as u128;
            bytes += n as u128 * t.vocab_size as u128 * 4 * 4;
            bytes += self.options.batch_size as u128
                * t.num_hidden_layers as u128
                * t.head_dim as u128
                * 4
                * (t.sliding_window + positions) as u128;
        }
        if labels == 0 {
            bail!("V4.1 training sequence has no non-ignored labels");
        }
        if positions > t.max_position_embeddings.min(65_535) {
            bail!("V4.1 training sequence exceeds native context capacity");
        }
        if bytes > self.options.max_activation_bytes as u128 {
            bail!("V4.1 training sequence estimated activation bytes {bytes} exceed budget {}; reduce the token chunk or increase the explicit budget",self.options.max_activation_bytes);
        }
        Ok((positions, rows, bytes as u64))
    }

    pub fn gradients(
        &mut self,
        batches: &[DeepseekV41TrainingBatch],
    ) -> Result<DeepseekV41ModelGradients> {
        self.unchanged()?;
        let (positions, total_rows, estimate) = self.estimate(batches)?;
        let mut package = self.package.clone();
        package.set_lora_adapter(Arc::new(self.optimizer.adapter()?))?;
        let runtime = DeepseekV41AttentionRuntimeConfig {
            batch_size: self.options.batch_size,
            max_prefill: batches[0].seq_len,
            max_positions: positions,
        };
        let mut model = DeepseekV41CausalLM::from_checkpoint_with_vocabulary_residency(
            self.device.clone(),
            &mut package,
            runtime,
            self.tokens.clone(),
            self.precision,
            if self.options.stream_experts {
                DeepseekV41ExpertResidency::StreamedAllExperts
            } else {
                DeepseekV41ExpertResidency::Resident
            },
            if self.options.stream_engram_tables {
                DeepseekV41EngramResidency::StreamedTables
            } else {
                DeepseekV41EngramResidency::Resident
            },
            if self.options.stream_projections {
                DeepseekV41ProjectionResidency::Streamed
            } else {
                DeepseekV41ProjectionResidency::Resident
            },
            if self.options.stream_vocabulary {
                DeepseekV41VocabularyResidency::Streamed
            } else {
                DeepseekV41VocabularyResidency::Resident
            },
        )?;
        let mut recording = model.backbone.recording()?;
        let mut steps = Vec::new();
        let mut logits = Vec::new();
        let mut targets = Vec::new();
        let mut position = 0;
        let t = model.config.text.clone();
        for batch in batches {
            model.validate_input(
                &batch.token_ids,
                batch.seq_len,
                position,
                batch.engram_mask.as_deref(),
            )?;
            let rows = self.options.batch_size * batch.seq_len;
            let ids = GpuBuffer::from_u32(&self.device, &batch.token_ids)?;
            let mask = batch
                .engram_mask
                .as_ref()
                .map(|m| GpuBuffer::from_u32(&self.device, m))
                .transpose()?;
            let stream = model.embed(&ids, rows)?;
            let (collapsed, backbone) = model.backbone.forward_retained(
                &stream,
                batch.seq_len,
                position,
                (!t.engram_layer_ids.is_empty()).then_some(&ids),
                mask.as_ref(),
                &self.kernels,
                &mut recording,
            )?;
            let hidden = model.norm.apply(
                &self.device,
                &collapsed,
                &model.norm_weight,
                rows,
                t.hidden_size,
                t.rms_norm_eps as f32,
            )?;
            let hidden = model
                .cast
                .cast_buffer(&hidden, rows * t.hidden_size, self.precision)?;
            let output = model.project_logits(&hidden, rows)?;
            logits.push(DeepseekV41Logits {
                values: output,
                batch_size: self.options.batch_size,
                seq_len: batch.seq_len,
                vocab_size: t.vocab_size,
            });
            steps.push((collapsed, backbone, rows));
            targets.extend(&batch.targets);
            position += batch.seq_len;
        }
        let joined = GpuBuffer::zeros_f32(&self.device, total_rows * t.vocab_size)?;
        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        let mut offset = 0;
        for output in &logits {
            let n = output.batch_size * output.seq_len * t.vocab_size;
            commands.copy_f32_range(&output.values, 0, &joined, offset, n)?;
            offset += n;
        }
        commands.submit()?;
        let (loss, grad_logits) =
            self.terminal
                .masked_cross_entropy(&joined, &targets, t.vocab_size)?;
        let mut reverse = recording.cache.into_reverse()?;
        let mut states = (0..t.num_hidden_layers).map(|_| None).collect::<Vec<_>>();
        let mut adapters: BTreeMap<String, DeepseekV41AdapterGradient> = BTreeMap::new();
        let mut end = total_rows;
        for (collapsed, backbone, rows) in steps.into_iter().rev() {
            end -= rows;
            let grad = GpuBuffer::zeros_f32(&self.device, rows * t.vocab_size)?;
            let mut commands = vulkan::ComputeBatch::new(&self.device)?;
            commands.copy_f32_range(
                &grad_logits,
                end * t.vocab_size,
                &grad,
                0,
                rows * t.vocab_size,
            )?;
            commands.submit()?;
            let hidden =
                self.head_input_grad(&mut package, &grad, rows, t.vocab_size, t.hidden_size)?;
            let collapsed_grad = self.terminal.norm_input_grad(
                &hidden,
                &collapsed,
                &model.norm_weight,
                rows,
                t.hidden_size,
                t.rms_norm_eps as f32,
            )?;
            let (_, gradients) = model.backbone.backward_retained(
                backbone,
                &collapsed_grad,
                &mut reverse,
                &mut states,
                &self.kernels,
            )?;
            accumulate(&mut adapters, gradients)?;
        }
        reverse.finish()?;
        self.unchanged()?;
        Ok(DeepseekV41ModelGradients {
            loss,
            logits,
            adapters: adapters.into_values().collect(),
            estimated_activation_bytes: estimate,
        })
    }
    pub fn train_sequence(
        &mut self,
        batches: &[DeepseekV41TrainingBatch],
    ) -> Result<DeepseekV41ModelTrainResult> {
        let gradients = self.gradients(batches)?;
        self.optimizer.step(&gradients.adapters)?;
        Ok(DeepseekV41ModelTrainResult {
            loss: gradients.loss,
            optimizer_step: self.optimizer.step_count(),
            active_modules: gradients.adapters.len(),
            logits: gradients.logits,
            estimated_activation_bytes: gradients.estimated_activation_bytes,
        })
    }
    pub fn train_step(
        &mut self,
        batch: &DeepseekV41TrainingBatch,
    ) -> Result<DeepseekV41ModelTrainResult> {
        self.train_sequence(std::slice::from_ref(batch))
    }
    pub fn export_adapter(&self, directory: impl AsRef<Path>) -> Result<()> {
        self.optimizer.adapter()?.export(directory)
    }
    pub fn save_checkpoint(&self, directory: impl AsRef<Path>) -> Result<()> {
        self.unchanged()?;
        self.optimizer.save_checkpoint(&directory)?;
        let binding = Binding {
            schema: 1,
            base_sha256: self.identity.clone(),
            config: self.options.clone(),
        };
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(directory.as_ref().join("training_model.json"))?;
        file.write_all(&serde_json::to_vec_pretty(&binding)?)?;
        file.sync_all()?;
        Ok(())
    }
    pub fn resume(
        device: VulkanDevice,
        base: impl AsRef<Path>,
        checkpoint: impl AsRef<Path>,
    ) -> Result<Self> {
        let binding: Binding = serde_json::from_slice(
            &fs::read(checkpoint.as_ref().join("training_model.json"))
                .context("V4.1 model checkpoint has no frozen-base binding")?,
        )?;
        if binding.schema != 1 {
            bail!("V4.1 model checkpoint binding schema is unsupported");
        }
        let optimizer = DeepseekV41AdapterTrainer::resume_from_checkpoint(checkpoint)?;
        let trainer = Self::build(device, base.as_ref(), optimizer, binding.config)?;
        if trainer.identity != binding.base_sha256 {
            bail!("V4.1 model checkpoint belongs to different frozen weights/configuration/token artifacts");
        }
        Ok(trainer)
    }
    fn head_input_grad(
        &self,
        package: &mut DeepseekV41Checkpoint,
        grad: &GpuBuffer,
        rows: usize,
        vocab: usize,
        hidden: usize,
    ) -> Result<GpuBuffer> {
        let output = GpuBuffer::zeros_f32(&self.device, rows * hidden)?;
        let tile_rows =
            ((crate::weight_offload::DEFAULT_TILE_BYTES as usize / (hidden * 4)) / 32 * 32).max(32);
        let mut start = 0;
        while start < vocab {
            let count = tile_rows.min(vocab - start);
            let values=match package.tensor_dtype("head.weight")?.as_str() {
                "F32"=>package.read_f32_row_range("head.weight",[vocab,hidden],start,count)?,
                "BF16"=>package.read_bf16_word_range("head.weight",[vocab,hidden],start,count)?.into_iter()
                    .flat_map(|word|[f32::from_bits((word&0xffff)<<16),f32::from_bits(word&0xffff0000)])
                    .take(count*hidden).collect(),
                other=>bail!("V4.1 training head input gradient requires dense F32/BF16 storage, got {other}"),
            };
            let weight = GpuBuffer::from_f32(&self.device,&values)?;
            let tile = GpuBuffer::zeros_f32(&self.device, rows * count)?;
            let mut commands = vulkan::ComputeBatch::new(&self.device)?;
            for row in 0..rows {
                commands.copy_f32_range(grad, row * vocab + start, &tile, row * count, count)?;
            }
            let push = [
                rows as u32,
                hidden as u32,
                count as u32,
                u32::from(start != 0),
            ];
            let kernel = if crate::transformer::aten_vector_width() >= 16 &&
                (rows == 1 || (rows < 4 && hidden.is_multiple_of(4) && vocab.is_multiple_of(4))) {
                &self.head_gradient_short
            } else { &self.head_gradient };
            kernel.record_dispatch(
                &mut commands,
                &[&tile, &weight, &output],
                bytemuck::cast_slice(&push),
                [hidden.div_ceil(16) as u32, rows.div_ceil(16) as u32, 1],
            )?;
            commands.submit()?;
            start += count;
        }
        Ok(output)
    }
}

fn accumulate(
    output: &mut BTreeMap<String, DeepseekV41AdapterGradient>,
    gradients: Vec<DeepseekV41AdapterGradient>,
) -> Result<()> {
    for g in gradients {
        if let Some(current) = output.get_mut(&g.target) {
            if current.grad_a.len() != g.grad_a.len() || current.grad_b.len() != g.grad_b.len() {
                bail!("V4.1 model adapter gradient shape changed");
            }
            for (a, b) in current.grad_a.iter_mut().zip(g.grad_a) {
                *a += b;
            }
            for (a, b) in current.grad_b.iter_mut().zip(g.grad_b) {
                *a += b;
            }
        } else {
            output.insert(g.target.clone(), g);
        }
    }
    Ok(())
}
fn load_tokens(
    directory: &Path,
    config: &DeepseekV41Config,
) -> Result<Option<DeepseekV41EngramTokens>> {
    let t = &config.text;
    if t.engram_layer_ids.is_empty() {
        return Ok(None);
    }
    let source = config
        .source_config
        .get("text_config")
        .unwrap_or(&config.source_config);
    if let (Some(map), Some(multipliers)) = (
        source.get("engram_token_map"),
        source.get("engram_hash_multipliers"),
    ) {
        return Ok(Some(DeepseekV41EngramTokens {
            token_map: serde_json::from_value(map.clone())?,
            multipliers: serde_json::from_value(multipliers.clone())?,
            pad_token_id: t.engram_pad_token_id,
        }));
    }
    if directory.join("engram_tokens.json").is_file() {
        return Ok(Some(serde_json::from_slice(&fs::read(
            directory.join("engram_tokens.json"),
        )?)?));
    }
    let tokenizer = tokenizers::Tokenizer::from_file(directory.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("V4.1 tokenizer for Engram: {e}"))?;
    Ok(Some(DeepseekV41EngramTokens::from_tokenizer(
        &tokenizer, t,
    )?))
}
fn fingerprint(package: &DeepseekV41Checkpoint) -> Result<(String, Vec<FileStamp>)> {
    let mut hash = Sha256::new();
    let mut stamps = Vec::new();
    let mut block = vec![0u8; 1024 * 1024];
    for path in package.source_files()? {
        let name = path
            .file_name()
            .context("V4.1 package source has no file name")?
            .to_string_lossy();
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        let mut file = fs::File::open(&path)?;
        let meta = file.metadata()?;
        hash.update(meta.len().to_le_bytes());
        loop {
            let n = file.read(&mut block)?;
            if n == 0 {
                break;
            }
            hash.update(&block[..n]);
        }
        stamps.push(FileStamp {
            path,
            len: meta.len(),
            modified: meta.modified().ok(),
        });
    }
    Ok((format!("{:x}", hash.finalize()), stamps))
}
