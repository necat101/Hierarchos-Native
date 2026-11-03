//! Initial multimodal training bridge: already projected modality features at
//! the raw token-embedding boundary. Towers, processors, modality-specific
//! positions and attention masks require separate architecture qualification.

use anyhow::{bail, Context, Result};

use crate::{vulkan::ComputeBatch, GpuBuffer, VulkanDevice, VulkanTransformerStepResult};

/// Replace selected rows of the raw token lookup, before embedding scaling,
/// positions, normalization and dropout. Values have shape `[rows.len(), dim]`
/// in exactly the supplied row order; rows index the flattened batch/sequence.
/// Input IDs remain valid in-vocabulary placeholders and retain their ordinary
/// use by ID-dependent architecture features (for example Qwen4 PLE).
#[derive(Debug, Clone, Copy)]
pub struct ModalityEmbeddingRows<'a> {
    pub rows: &'a [usize],
    pub values: &'a [f32],
}

impl ModalityEmbeddingRows<'_> {
    pub fn validate(&self, sequence_rows: usize, dim: usize) -> Result<()> {
        if self.rows.is_empty() || dim == 0 {
            bail!("modality embedding rows and width must be nonempty");
        }
        let expected = self
            .rows
            .len()
            .checked_mul(dim)
            .context("modality embedding size overflow")?;
        if self.values.len() != expected {
            bail!(
                "modality embeddings contain {} values, expected {expected}",
                self.values.len()
            );
        }
        let mut seen = std::collections::BTreeSet::new();
        for &row in self.rows {
            if row >= sequence_rows || !seen.insert(row) {
                bail!("modality row {row} is out of range or duplicated");
            }
        }
        if self.values.iter().any(|value| !value.is_finite()) {
            bail!("modality embeddings contain non-finite values");
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct MultimodalTransformerStepResult {
    pub text: VulkanTransformerStepResult,
    /// Loss gradient with respect to the supplied raw features, in the same
    /// row order as `ModalityEmbeddingRows::values`, including embedding scale.
    pub modality_embedding_grad: Vec<f32>,
}

pub(crate) struct PreparedModalityRows<'a> {
    rows: &'a [usize],
    dim: usize,
    sequence_rows: usize,
    features: GpuBuffer,
    backward: Option<(GpuBuffer, GpuBuffer, GpuBuffer)>,
}

impl<'a> PreparedModalityRows<'a> {
    pub(crate) fn new(
        device: &VulkanDevice,
        input: ModalityEmbeddingRows<'a>,
        sequence_rows: usize,
        dim: usize,
        training: bool,
    ) -> Result<Self> {
        input.validate(sequence_rows, dim)?;
        let len = sequence_rows
            .checked_mul(dim)
            .context("embedding size overflow")?;
        Ok(Self {
            rows: input.rows,
            dim,
            sequence_rows,
            features: GpuBuffer::from_f32(device, input.values)?,
            backward: if training {
                Some((
                    GpuBuffer::zeros_f32(device, len)?,
                    GpuBuffer::zeros_f32(device, input.values.len())?,
                    GpuBuffer::zeros_f32(device, dim)?,
                ))
            } else {
                None
            },
        })
    }

    pub(crate) fn record_forward(
        &self,
        commands: &mut ComputeBatch,
        token_embeddings: &GpuBuffer,
    ) -> Result<()> {
        for (source, &row) in self.rows.iter().enumerate() {
            commands.copy_f32_range(
                &self.features,
                source * self.dim,
                token_embeddings,
                row * self.dim,
                self.dim,
            )?;
        }
        Ok(())
    }

    pub(crate) fn record_backward<'b>(
        &'b self,
        commands: &mut ComputeBatch,
        gradient: &GpuBuffer,
    ) -> Result<&'b GpuBuffer> {
        let (text_grad, feature_grad, zero_row) = self
            .backward
            .as_ref()
            .context("modality rows were prepared for evaluation only")?;
        commands.copy_f32(gradient, text_grad, self.sequence_rows * self.dim)?;
        for (target, &row) in self.rows.iter().enumerate() {
            commands.copy_f32_range(
                gradient,
                row * self.dim,
                feature_grad,
                target * self.dim,
                self.dim,
            )?;
            commands.copy_f32_range(zero_row, 0, text_grad, row * self.dim, self.dim)?;
        }
        Ok(text_grad)
    }

    pub(crate) fn gradient(&self) -> Result<Vec<f32>> {
        self.backward
            .as_ref()
            .context("modality gradient is unavailable")?
            .1
            .read_f32(self.rows.len() * self.dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modality_rows_reject_ambiguous_or_invalid_layouts() {
        for (rows, values, dim) in [
            (vec![], vec![], 2),
            (vec![1, 1], vec![0.; 4], 2),
            (vec![4], vec![0.; 2], 2),
            (vec![1], vec![0.], 2),
            (vec![1], vec![f32::NAN, 0.], 2),
            (vec![1], vec![], 0),
        ] {
            assert!(ModalityEmbeddingRows {
                rows: &rows,
                values: &values
            }
            .validate(4, dim)
            .is_err());
        }
        ModalityEmbeddingRows {
            rows: &[3, 0],
            values: &[1., 2., 3., 4.],
        }
        .validate(4, 2)
        .unwrap();
    }

    #[test]
    fn modality_scatter_and_adjoint_preserve_order_and_text_rows() -> Result<()> {
        let device = VulkanDevice::new()?;
        let input = ModalityEmbeddingRows {
            rows: &[3, 1],
            values: &[30., 31., 10., 11.],
        };
        let prepared = PreparedModalityRows::new(&device, input, 4, 2, true)?;
        let text = GpuBuffer::from_f32(&device, &[0., 1., 2., 3., 4., 5., 6., 7.])?;
        let grad = GpuBuffer::from_f32(&device, &[1., 2., 3., 4., 5., 6., 7., 8.])?;
        let mut commands = ComputeBatch::new(&device)?;
        prepared.record_forward(&mut commands, &text)?;
        let masked = prepared.record_backward(&mut commands, &grad)?;
        commands.submit()?;
        assert_eq!(text.read_f32(8)?, [0., 1., 10., 11., 4., 5., 30., 31.]);
        assert_eq!(prepared.gradient()?, [7., 8., 3., 4.]);
        assert_eq!(masked.read_f32(8)?, [1., 2., 0., 0., 5., 6., 0., 0.]);
        assert_eq!(grad.read_f32(8)?, [1., 2., 3., 4., 5., 6., 7., 8.]);
        Ok(())
    }
}
