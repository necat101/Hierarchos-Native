//! Device boundaries and retained encoder tapes for joint seq2seq training.
use super::*;

#[derive(Default)]
pub(super) struct ComposedTrainingOptions<'a> {
    pub(super) encoder_hidden_device: Option<&'a GpuBuffer>,
    pub(super) terminal_gradient: Option<&'a GpuBuffer>,
    pub(super) forward_only: bool,
    pub(super) retained_forward: bool,
    pub(super) defer_shared_embedding_step: bool,
    pub(super) preserve_shared_embedding_gradient: bool,
    pub(super) defer_amp_step: bool,
}

pub(super) struct EncoderTrainingTape {
    step: u32,
    ids: Vec<u32>,
    mask: Vec<f32>,
    modality_rows: Vec<usize>,
    modality_values: GpuBuffer,
}

impl VulkanTransformer {
    pub(super) fn share_encoder_embedding(&mut self, encoder: &VulkanTransformer) -> Result<()> {
        if self.config.vocab_size != encoder.config.vocab_size
            || self.config.effective_embedding_size() != encoder.config.effective_embedding_size()
        {
            bail!("joint encoder/decoder token embedding geometry differs");
        }
        self.shared_embedding = encoder.shared_embedding.clone();
        self.token_embedding = TiedTokenEmbeddingOp::from_shared_parameter_with_padding_idx(
            self.shared_embedding.clone(),
            self.rows,
            self.token_embedding_padding_idx,
        )?;
        Ok(())
    }

    pub(super) fn train_with_device_encoder_context(
        &mut self,
        ids: &[u32],
        mask: &[f32],
        targets: &[u32],
        weights: &[f32],
        hidden: &GpuBuffer,
        encoder_mask: &[f32],
        hyper: AdamWHyperParams,
    ) -> Result<DeviceMultimodalStepResult> {
        self.train_step_with_context_and_modality_embeddings(
            ids,
            None,
            None,
            None,
            None,
            targets,
            mask,
            weights,
            None,
            Some(encoder_mask),
            hyper,
            None,
            ComposedTrainingOptions {
                encoder_hidden_device: Some(hidden),
                defer_shared_embedding_step: true,
                ..Default::default()
            },
        )
    }

    pub(super) fn device_encoder_gradient(&self) -> Result<&GpuBuffer> {
        self.grad_encoder_hidden
            .as_ref()
            .context("decoder omitted its encoder gradient")
    }

    pub(super) fn export_joint_seq2seq_package(
        &self,
        decoder: &VulkanTransformer,
        source: &Path,
        destination: &Path,
    ) -> Result<()> {
        fs::create_dir(destination).context("joint text export destination must be new")?;
        for entry in fs::read_dir(source)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file()
                && path.extension().and_then(|x| x.to_str()) != Some("safetensors")
                && path.file_name().and_then(|x| x.to_str())
                    != Some(GPT2_SAFETENSORS_INDEX_FILENAME)
            {
                fs::copy(&path, destination.join(entry.file_name()))?;
            }
        }
        let mut tensors = Gpt2TensorStore::load(source)?.into_parameter_tensors()?;
        tensors.extend(decoder.hf_parameter_tensors()?);
        if self.config.architecture == VulkanTransformerArchitecture::Bart {
            for (name, tensor) in self.bart_parameter_tensors()? {
                if let Some(suffix) = name.strip_prefix("model.decoder.") {
                    tensors.insert(format!("model.encoder.{suffix}"), tensor);
                }
            }
        } else {
            tensors.extend(self.t5_parameter_tensors()?);
        }
        let shape = vec![
            self.config.vocab_size,
            self.config.effective_embedding_size(),
        ];
        let embedding = self.shared_embedding.weights()?;
        let aliases = if self.config.architecture == VulkanTransformerArchitecture::Bart {
            [
                "model.shared.weight",
                "model.encoder.embed_tokens.weight",
                "model.decoder.embed_tokens.weight",
            ]
        } else {
            [
                "shared.weight",
                "encoder.embed_tokens.weight",
                "decoder.embed_tokens.weight",
            ]
        };
        for alias in aliases {
            tensors.insert(alias.to_owned(), (shape.clone(), embedding.clone()));
        }
        let owned = tensors
            .iter()
            .map(|(name, (shape, values))| (name.clone(), shape.clone(), f32_le_bytes(values)))
            .collect::<Vec<_>>();
        let views = owned
            .iter()
            .map(|(name, shape, bytes)| {
                Ok((
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let metadata = std::collections::HashMap::from([("format".to_owned(), "pt".to_owned())]);
        let path = destination.join(GPT2_SAFETENSORS_FILENAME);
        serialize_to_file(views, Some(metadata), &path)?;
        canonicalize_safetensors_header(&path)?;
        Ok(())
    }

    pub(super) fn forward_with_device_encoder_context(
        &self,
        ids: &[u32],
        mask: &[f32],
        hidden: &GpuBuffer,
        encoder_mask: &[f32],
    ) -> Result<Vec<f32>> {
        self.forward_logits_with_context_and_roc_bert_ids_and_layer_count(
            ids,
            None,
            None,
            None,
            None,
            mask,
            None,
            Some(encoder_mask),
            self.layers.len(),
            false,
            None,
            Some(hidden),
            false,
        )
        .map(|(logits, _)| logits)
    }

    pub(super) fn encode_with_device_modality_embeddings(
        &self,
        ids: &[u32],
        mask: &[f32],
        rows: &[usize],
        values: &GpuBuffer,
    ) -> Result<&GpuBuffer> {
        self.forward_logits_with_context_and_roc_bert_ids_and_layer_count(
            ids,
            None,
            None,
            None,
            None,
            mask,
            None,
            None,
            self.layers.len(),
            false,
            Some(ModalityEmbeddingInput::Device { rows, values }),
            None,
            true,
        )?;
        self.training_terminal_output()
    }

    pub(super) fn validate_device_encoder_context(
        &self,
        hidden: &GpuBuffer,
        mask: Option<&[f32]>,
    ) -> Result<usize> {
        if self.encoder_hidden.is_none() || self.encoder_rows == 0 {
            bail!("device encoder context requires a cross-attention graph");
        }
        let len = self
            .encoder_rows
            .checked_mul(self.encoder_hidden_size)
            .context("encoder hidden size overflow")?;
        if !hidden.belongs_to_device(&self.device) || hidden.f32_capacity() < len {
            bail!("encoder context has the wrong Vulkan device or capacity");
        }
        if let Some(mask) = mask {
            if mask.len() != self.encoder_rows || mask.iter().any(|x| !x.is_finite()) {
                bail!("encoder mask has the wrong row count or non-finite values");
            }
        }
        Ok(self.encoder_rows / self.batch_size)
    }

    pub(super) fn training_terminal_output(&self) -> Result<&GpuBuffer> {
        if self.config.uses_final_norm() {
            Ok(&self.final_norm_output)
        } else {
            Ok(self
                .layers
                .last()
                .context("encoder has no layers")?
                .forward_output())
        }
    }

    pub(super) fn begin_encoder_image_training(
        &mut self,
        ids: &[u32],
        mask: &[f32],
        modality_rows: &[usize],
        modality_values: &GpuBuffer,
        hyper: AdamWHyperParams,
    ) -> Result<EncoderTrainingTape> {
        self.train_step_with_context_and_modality_embeddings(
            ids,
            None,
            None,
            None,
            None,
            &vec![u32::MAX; self.rows],
            mask,
            &vec![0.; self.rows],
            None,
            None,
            hyper,
            Some(ModalityEmbeddingInput::Device {
                rows: modality_rows,
                values: modality_values,
            }),
            ComposedTrainingOptions {
                forward_only: true,
                ..Default::default()
            },
        )?;
        Ok(EncoderTrainingTape {
            step: self.step,
            ids: ids.to_vec(),
            mask: mask.to_vec(),
            modality_rows: modality_rows.to_vec(),
            modality_values: modality_values.clone(),
        })
    }

    pub(super) fn finish_encoder_image_training(
        &mut self,
        tape: EncoderTrainingTape,
        gradient: &GpuBuffer,
        hyper: AdamWHyperParams,
    ) -> Result<DeviceMultimodalStepResult> {
        if self.step != tape.step {
            bail!("encoder parameters changed before its retained backward");
        }
        self.train_step_with_context_and_modality_embeddings(
            &tape.ids,
            None,
            None,
            None,
            None,
            &vec![u32::MAX; self.rows],
            &tape.mask,
            &vec![0.; self.rows],
            None,
            None,
            hyper,
            Some(ModalityEmbeddingInput::Device {
                rows: &tape.modality_rows,
                values: &tape.modality_values,
            }),
            ComposedTrainingOptions {
                terminal_gradient: Some(gradient),
                retained_forward: true,
                preserve_shared_embedding_gradient: true,
                ..Default::default()
            },
        )
    }
}
