//! Bounded staging for parameters admitted together by a model component.
use super::{DeepseekV41LinearWeights, DeepseekV41ResidentLinear};
use crate::{vulkan, GpuBuffer, VulkanDevice};
use anyhow::{Context, Result};

const MAX_STAGED_BYTES: usize = 16 * 1024 * 1024;

pub(super) struct ParameterUploadBatch {
    device: VulkanDevice,
    commands: Option<vulkan::BatchHandle>,
    staged_bytes: usize,
    budget: usize,
    enabled: bool,
}

impl ParameterUploadBatch {
    pub(super) fn enabled(&self) -> bool {
        self.enabled
    }
    pub(super) fn new(device: &VulkanDevice) -> Self {
        #[cfg(test)]
        let enabled = std::env::var_os("HIERARCHOS_V41_INDIVIDUAL_UPLOADS").is_none();
        #[cfg(not(test))]
        let enabled = true;
        Self::with_policy(device, MAX_STAGED_BYTES, enabled)
    }

    fn with_policy(device: &VulkanDevice, budget: usize, enabled: bool) -> Self {
        Self {
            device: device.clone(),
            commands: None,
            staged_bytes: 0,
            budget,
            enabled,
        }
    }

    fn prepare(&mut self, bytes: usize) -> Result<()> {
        // A single large tensor keeps its established upload boundary. Small
        // tensors share staging, without retaining a whole model's host copy.
        if self.staged_bytes != 0 && self.staged_bytes.saturating_add(bytes) > self.budget {
            self.flush()?;
        }
        if self.commands.is_none() {
            self.commands = Some(self.device.compute_batch()?);
        }
        self.staged_bytes = self
            .staged_bytes
            .checked_add(bytes)
            .context("parameter upload size overflow")?;
        Ok(())
    }

    pub(super) fn f32(&mut self, values: &[f32]) -> Result<GpuBuffer> {
        if !self.enabled {
            return GpuBuffer::from_f32(&self.device, values);
        }
        let bytes = values
            .len()
            .checked_mul(4)
            .context("parameter upload size overflow")?;
        if bytes >= self.budget {
            self.flush()?;
            return GpuBuffer::from_f32(&self.device, values);
        }
        self.prepare(bytes)?;
        let value = GpuBuffer::from_f32_in_upload_arena(
            &self.device,
            self.commands.as_mut().unwrap(),
            values,
        )?;
        Ok(value)
    }

    pub(super) fn linear(
        &mut self,
        weights: DeepseekV41LinearWeights,
    ) -> Result<DeepseekV41ResidentLinear> {
        if !self.enabled {
            return weights.upload(&self.device);
        }
        let [rows, cols] = weights.shape();
        // Dense size is also a conservative upper bound for packed bases.
        let mut elements = rows
            .checked_mul(cols)
            .context("parameter upload geometry overflow")?;
        if let Some(lora) = weights.lora() {
            elements = elements
                .checked_add(lora.a.len())
                .and_then(|n| n.checked_add(lora.b.len()))
                .context("adapter upload geometry overflow")?;
        }
        let bytes = elements
            .checked_mul(4)
            .context("parameter upload size overflow")?;
        if bytes >= self.budget {
            self.flush()?;
            return weights.upload(&self.device);
        }
        self.prepare(bytes)?;
        let value = weights.record_upload_grouped(&self.device, self.commands.as_mut().unwrap())?;
        Ok(value)
    }

    fn flush(&mut self) -> Result<()> {
        if let Some(mut commands) = self.commands.take() {
            // Inside an outer graph session the upload arena is retained by
            // that session, and all copies recorded so far retire together.
            // Waiting here would split the graph into one submission per
            // component and defeat the caller's batching boundary.
            if self.device.has_active_compute_session() {
                commands.submit()?;
            } else {
                commands.submit_and_wait()?;
            }
        }
        self.staged_bytes = 0;
        Ok(())
    }

    pub(super) fn finish(mut self) -> Result<()> {
        self.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_upload_batch_preserves_words_and_bounds_staging() -> Result<()> {
        let device = VulkanDevice::new()?;
        let values = [
            vec![-0., 1.25],
            vec![f32::MIN_POSITIVE, -3.5],
            vec![4., 5., 6.],
        ];
        for (budget, enabled, submissions) in [(64, true, 1), (16, true, 2), (64, false, 3)] {
            let before = device.submission_count();
            let mut upload = ParameterUploadBatch::with_policy(&device, budget, enabled);
            let outputs = values
                .iter()
                .map(|v| upload.f32(v))
                .collect::<Result<Vec<_>>>()?;
            if enabled && budget == 64 {
                assert_eq!(
                    upload
                        .commands
                        .as_ref()
                        .unwrap()
                        .upload_arena_buffer_count(),
                    1
                );
                assert_eq!(device.submission_count(), before);
            }
            upload.finish()?;
            assert_eq!(device.submission_count() - before, submissions);
            for (actual, expected) in outputs.iter().zip(&values) {
                assert_eq!(
                    actual
                        .read_f32(expected.len())?
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
        Ok(())
    }
}
