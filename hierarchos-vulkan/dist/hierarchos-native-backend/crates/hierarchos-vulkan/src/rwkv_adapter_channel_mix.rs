use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::{
    vulkan, GpuBuffer, RwkvChannelMixOp, RwkvChannelMixResult, SharedTokenAdapterTrainer,
    VulkanDevice,
};

#[derive(Debug)]
pub struct RwkvAdapterChannelMixResult {
    pub channel_mix: RwkvChannelMixResult,
    /// Gradient at the tied-token feature input of SharedTokenAdapter. This is
    /// the tensor that can later be accumulated into the shared embedding.
    pub token_feature_grad: Vec<f32>,
}

/// Single-submit composition of coherent-v9 SharedTokenAdapter -> channel-mix.
///
/// The adapter output is bound directly as channel-mix's `DeepEmbed` input and
/// channel-mix's `grad_deepembed` is bound directly into the adapter backward
/// pass. Neither edge is supplied by the host, so the expensive `4*C`
/// modulation activation stays entirely on Vulkan during the training graph.
pub struct RwkvAdapterChannelMixOp {
    device: VulkanDevice,
    adapter: SharedTokenAdapterTrainer,
    channel_mix: RwkvChannelMixOp,
    max_batch: usize,

    x: GpuBuffer,
    previous: GpuBuffer,
    token_features: GpuBuffer,
    grad_output: GpuBuffer,
}

impl RwkvAdapterChannelMixOp {
    pub fn from_model_package(
        device: VulkanDevice,
        model_dir: impl AsRef<Path>,
        cell_prefix: &str,
        adapter_prefix: &str,
        max_batch: usize,
        key_clamp: f32,
        deepembed_clamp: f32,
        adapter_matrix_weight_decay: f32,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let adapter = SharedTokenAdapterTrainer::from_model_package(
            device.clone(),
            model_dir,
            adapter_prefix,
            max_batch,
            adapter_matrix_weight_decay,
        )?;
        let channel_mix = RwkvChannelMixOp::from_model_package(
            device.clone(),
            model_dir,
            cell_prefix,
            max_batch,
            key_clamp,
            deepembed_clamp,
        )?;
        Self::new(device, adapter, channel_mix)
    }

    pub fn new(
        device: VulkanDevice,
        adapter: SharedTokenAdapterTrainer,
        channel_mix: RwkvChannelMixOp,
    ) -> Result<Self> {
        if adapter.output_dim() != channel_mix.hidden_width() {
            bail!(
                "SharedTokenAdapter output width {} does not match channel-mix DeepEmbed width {}",
                adapter.output_dim(),
                channel_mix.hidden_width()
            );
        }
        let max_batch = adapter.max_rows().min(channel_mix.max_batch());
        if max_batch == 0 {
            bail!("composed adapter/channel-mix capacity must be positive");
        }
        let vector_len = max_batch
            .checked_mul(channel_mix.width())
            .context("adapter/channel-mix vector capacity overflow")?;
        let feature_len = max_batch
            .checked_mul(adapter.input_dim())
            .context("adapter/channel-mix token-feature capacity overflow")?;

        Ok(Self {
            x: GpuBuffer::zeros_f32(&device, vector_len)?,
            previous: GpuBuffer::zeros_f32(&device, vector_len)?,
            token_features: GpuBuffer::zeros_f32(&device, feature_len)?,
            grad_output: GpuBuffer::zeros_f32(&device, vector_len)?,
            device,
            adapter,
            channel_mix,
            max_batch,
        })
    }

    pub fn forward_backward(
        &mut self,
        batch: usize,
        x: &[f32],
        previous: &[f32],
        token_features: &[f32],
        grad_output: &[f32],
    ) -> Result<RwkvAdapterChannelMixResult> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "adapter/channel-mix batch must be in 1..={}; got {batch}",
                self.max_batch
            );
        }
        validate_len("x", x, batch * self.channel_mix.width())?;
        validate_len("previous", previous, batch * self.channel_mix.width())?;
        validate_len(
            "token_features",
            token_features,
            batch * self.adapter.input_dim(),
        )?;
        validate_len("grad_output", grad_output, batch * self.channel_mix.width())?;

        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.upload_f32(&self.x, x)?;
        commands.upload_f32(&self.previous, previous)?;
        commands.upload_f32(&self.token_features, token_features)?;
        commands.upload_f32(&self.grad_output, grad_output)?;

        self.adapter
            .record_forward(&mut commands, batch, &self.token_features)?;
        self.channel_mix.record_forward(
            &mut commands,
            batch,
            &self.x,
            &self.previous,
            self.adapter.output_buffer(),
        )?;
        self.channel_mix.record_backward(
            &mut commands,
            batch,
            &self.x,
            &self.previous,
            self.adapter.output_buffer(),
            &self.grad_output,
        )?;
        self.adapter.record_backward(
            &mut commands,
            batch,
            &self.token_features,
            self.channel_mix.grad_deepembed_buffer(),
        )?;

        self.channel_mix.record_readback(&mut commands, batch)?;
        self.adapter
            .record_grad_input_readback(&mut commands, batch)?;
        commands.submit()?;

        Ok(RwkvAdapterChannelMixResult {
            channel_mix: self.channel_mix.read_result(batch)?,
            token_feature_grad: self.adapter.read_grad_input(batch)?,
        })
    }

    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    pub fn width(&self) -> usize {
        self.channel_mix.width()
    }

    pub fn token_feature_width(&self) -> usize {
        self.adapter.input_dim()
    }

    pub fn deepembed_width(&self) -> usize {
        self.adapter.output_dim()
    }
}

fn validate_len(name: &str, values: &[f32], expected: usize) -> Result<()> {
    if values.len() != expected {
        bail!(
            "adapter/channel-mix {name} has {} values; expected {expected}",
            values.len()
        );
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("adapter/channel-mix {name} contains non-finite values");
    }
    Ok(())
}
