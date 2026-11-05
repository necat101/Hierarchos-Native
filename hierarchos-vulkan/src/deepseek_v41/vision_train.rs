//! Joint native image/delimiter and frozen-language-base LoRA training.
use super::*;
use crate::{CpuAdamWHyperParams, VulkanDeepseekV41Image};

#[path = "vision_state.rs"]
mod state;

pub struct DeepseekV41MultimodalGradients {
    pub text: DeepseekV41ModelGradients,
    pub input_gradient: GpuBuffer,
    step: u64,
    ticket: Arc<()>,
}
pub struct DeepseekV41MultimodalTrainer {
    language: DeepseekV41ModelTrainer,
    image: VulkanDeepseekV41Image,
    hyper: DeepseekV41AdamWConfig,
    image_token_id: u32,
    hc: usize,
    pending: Option<Arc<()>>,
}
impl DeepseekV41MultimodalTrainer {
    pub fn new(
        device: VulkanDevice,
        directory: impl AsRef<Path>,
        adapter: &DeepseekV41LoraAdapter,
        hyper: DeepseekV41AdamWConfig,
        options: DeepseekV41ModelTrainingConfig,
        patch_height: usize,
        patch_width: usize,
    ) -> Result<Self> {
        let _math = crate::deepseek_v41::cpu_math::avx2();
        let mut package = DeepseekV41Checkpoint::open(directory.as_ref())?;
        if package.tensor_dtype("embed.weight")? != "F32" {
            bail!("V4.1 native image training currently requires the FP32 contract");
        }
        let id = package.config.source_config["image_token_id"]
            .as_u64()
            .context("missing V4.1 image_token_id")?;
        if id >= package.config.text.vocab_size as u64 {
            bail!("V4.1 image token is outside the vocabulary");
        }
        let hc = package.config.text.hc_mult;
        for layer in 0..package.config.text.num_hidden_layers {
            if !package.contains_tensor(&format!("layers.{layer}.ffn.gate.bias_vl"))? {
                bail!("native image training requires released VL routing biases");
            }
        }
        let image = VulkanDeepseekV41Image::from_checkpoint(
            device.clone(),
            &mut package,
            patch_height,
            patch_width,
        )?;
        let language = DeepseekV41ModelTrainer::new(device, directory, adapter, hyper, options)?;
        Ok(Self {
            language,
            image,
            hyper,
            image_token_id: id as u32,
            hc,
            pending: None,
        })
    }
    fn batch(
        &self,
        batch: &DeepseekV41TrainingBatch,
        types: &[i32],
    ) -> Result<DeepseekV41TrainingBatch> {
        if batch.token_ids.len() != types.len()
            || types.iter().zip(&batch.token_ids).any(|(&kind, &id)| {
                !(-1..=3).contains(&kind) || (kind >= 0 && id != self.image_token_id)
            })
        {
            bail!("native image token ids/types disagree with the released processor contract");
        }
        let mut batch = batch.clone();
        batch.engram_mask = Some(types.iter().map(|&t| u32::from(t < 0)).collect());
        Ok(batch)
    }
    pub fn gradients(
        &mut self,
        batch: &DeepseekV41TrainingBatch,
        patches: &[f32],
        types: &[i32],
        image_start: usize,
    ) -> Result<DeepseekV41MultimodalGradients> {
        let _math = crate::deepseek_v41::cpu_math::avx2();
        self.pending = None;
        if self.language.optimizer.step_count() != self.image.step() as u64 {
            bail!("native image and adapter optimizer steps disagree");
        }
        let batch = self.batch(batch, types)?;
        let hc = self.hc;
        let image = &mut self.image;
        let mut builder = |base: &GpuBuffer| {
            image.forward_merge(base, patches, types, image_start, batch.seq_len, hc)
        };
        let (text, upstream) = self
            .language
            .gradients_with_modality_builder(std::slice::from_ref(&batch), &mut builder)?;
        let upstream = upstream.context("native language graph did not return an image VJP")?;
        self.image.backward(&upstream)?;
        let ticket = Arc::new(());
        self.pending = Some(ticket.clone());
        Ok(DeepseekV41MultimodalGradients {
            text,
            step: self.language.optimizer.step_count(),
            input_gradient: upstream,
            ticket,
        })
    }
    pub fn apply_gradients(&mut self, gradients: &DeepseekV41MultimodalGradients) -> Result<()> {
        let _math = crate::deepseek_v41::cpu_math::avx2();
        if gradients.step != self.language.optimizer.step_count()
            || gradients.step != self.image.step() as u64
            || !self.image.gradients_ready()
            || !self
                .pending
                .as_ref()
                .is_some_and(|t| Arc::ptr_eq(t, &gradients.ticket))
        {
            bail!("native multimodal update requires fresh gradients from the current joint step");
        }
        let next = self
            .image
            .step()
            .checked_add(1)
            .context("native image optimizer step overflow")?;
        if next > i32::MAX as u32 {
            bail!("native image optimizer step exceeds bias correction range");
        }
        self.pending = None;
        self.language.apply_adapter_gradients(&gradients.text)?;
        self.image.apply_gradients(CpuAdamWHyperParams {
            lr: self.hyper.learning_rate,
            beta1: self.hyper.beta1,
            beta2: self.hyper.beta2,
            eps: self.hyper.eps,
            weight_decay: self.hyper.weight_decay,
        })
    }
    pub fn train_step(
        &mut self,
        batch: &DeepseekV41TrainingBatch,
        patches: &[f32],
        types: &[i32],
        image_start: usize,
    ) -> Result<DeepseekV41ModelTrainResult> {
        let gradients = self.gradients(batch, patches, types, image_start)?;
        self.apply_gradients(&gradients)?;
        Ok(DeepseekV41ModelTrainResult {
            loss: gradients.text.loss,
            optimizer_step: self.language.optimizer.step_count(),
            active_modules: gradients.text.adapters.len(),
            logits: gradients.text.logits,
            estimated_activation_bytes: gradients.text.estimated_activation_bytes,
        })
    }
    pub fn image(&self) -> &VulkanDeepseekV41Image {
        &self.image
    }
    pub fn adapter(&self) -> Result<DeepseekV41LoraAdapter> {
        self.language.adapter()
    }
    pub fn optimizer_step(&self) -> u64 {
        self.language.optimizer.step_count()
    }
    /// Forward-only native model execution, with no retained language tape or
    /// unnecessary reverse graph. Uses this trainer's current LoRA factors.
    pub fn forward_logits(
        &mut self,
        batch: &DeepseekV41TrainingBatch,
        patches: &[f32],
        types: &[i32],
        image_start: usize,
    ) -> Result<DeepseekV41Logits> {
        let _math = crate::deepseek_v41::cpu_math::avx2();
        self.pending = None;
        let batch = self.batch(batch, types)?;
        self.language.unchanged()?;
        let mut package = self.language.package.clone();
        package.set_lora_adapter(Arc::new(self.language.adapter()?))?;
        let o = &self.language.options;
        let runtime = DeepseekV41AttentionRuntimeConfig {
            batch_size: o.batch_size,
            max_prefill: batch.seq_len,
            max_positions: batch.seq_len,
        };
        let mut model = DeepseekV41CausalLM::from_checkpoint_with_vocabulary_residency(
            self.language.device.clone(),
            &mut package,
            runtime,
            self.language.tokens.clone(),
            self.language.precision,
            if o.stream_experts {
                DeepseekV41ExpertResidency::StreamedAllExperts
            } else {
                DeepseekV41ExpertResidency::Resident
            },
            if o.stream_engram_tables {
                DeepseekV41EngramResidency::StreamedTables
            } else {
                DeepseekV41EngramResidency::Resident
            },
            if o.stream_projections {
                DeepseekV41ProjectionResidency::Streamed
            } else {
                DeepseekV41ProjectionResidency::Resident
            },
            if o.stream_vocabulary {
                DeepseekV41VocabularyResidency::Streamed
            } else {
                DeepseekV41VocabularyResidency::Resident
            },
        )?;
        #[cfg(test)]
        if std::env::var_os("HIERARCHOS_V41_NATIVE_DIAGNOSTICS").is_some() {
            model.backbone.trace = Some(Vec::new());
        }
        let hc = self.hc;
        let image = &mut self.image;
        let mut builder = |base: &GpuBuffer| {
            image.forward_merge(base, patches, types, image_start, batch.seq_len, hc)
        };
        let output = model.prefill_with_modality_builder(
            &batch.token_ids,
            batch.seq_len,
            batch.engram_mask.as_deref(),
            &mut builder,
        )?;
        #[cfg(test)]
        if let Some(dir) = std::env::var_os("HIERARCHOS_V41_NATIVE_DIAGNOSTICS") {
            let stages=model.backbone.trace.as_ref().unwrap().iter().map(|(layer,stage,buffer,n)|Ok(serde_json::json!({"layer":layer,"stage":stage,"values":buffer.read_f32(*n)?}))).collect::<Result<Vec<_>>>()?;
            let last = model.final_trace.as_ref().unwrap();
            let n = batch.token_ids.len() * self.image.config().dim;
            fs::create_dir_all(&dir)?;
            fs::write(
                Path::new(&dir).join(format!(
                    "rows{}-step{}-forward.json",
                    batch.token_ids.len(),
                    self.optimizer_step()
                )),
                serde_json::to_vec(
                    &serde_json::json!({"stages":stages,"collapsed":last.0.read_f32(n)?,"normalized":last.1.read_f32(n)?,"logits":output.read_all()?,"injected_stream":self.image.injected_stream()?}),
                )?,
            )?;
        }
        Ok(output)
    }
}
