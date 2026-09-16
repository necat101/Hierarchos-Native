use std::path::Path;

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::{read_f32_tensor, vulkan, GpuBuffer, SharedLmHeadParameter, VulkanDevice};

const EMBEDDING_FORWARD_SPV: &[u8] = include_bytes!("../shaders/embedding_forward.spv");
const EMBEDDING_GRAD_ACCUMULATE_SPV: &[u8] =
    include_bytes!("../shaders/embedding_grad_accumulate.spv");
const EMBEDDING_TOKEN_SORT_SPV: &[u8] = include_bytes!("../shaders/embedding_token_sort.spv");
const EMBEDDING_GRAD_SEGMENTED_SPV: &[u8] =
    include_bytes!("../shaders/embedding_grad_segmented.spv");
const EMBEDDING_RADIX_HISTOGRAM_SPV: &[u8] =
    include_bytes!("../shaders/embedding_radix_histogram.spv");
const EMBEDDING_RADIX_PREFIX_SPV: &[u8] = include_bytes!("../shaders/embedding_radix_prefix.spv");
const EMBEDDING_RADIX_SCATTER_SPV: &[u8] = include_bytes!("../shaders/embedding_radix_scatter.spv");
const EMBEDDING_SEGMENTED_SORT_CAPACITY: usize = 1024;
const EMBEDDING_DIRECT_ACCUMULATE_CAPACITY: usize = 32;
const EMBEDDING_RADIX_BLOCK_SIZE: usize = 256;
const EMBEDDING_RADIX_BUCKETS: usize = 16;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EmbeddingPush {
    token_count: u32,
    dim: u32,
    vocab_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EmbeddingBackwardPush {
    token_count: u32,
    dim: u32,
    vocab_size: u32,
    /// `u32::MAX` means the embedding has no Hugging Face / PyTorch
    /// `padding_idx`. Valid token ids are always strictly below vocab_size,
    /// so the sentinel can never alias a real embedding row.
    padding_idx: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EmbeddingRadixPush {
    token_count: u32,
    shift: u32,
    block_count: u32,
    source_is_identity: u32,
}

#[derive(Debug)]
pub struct TiedTokenEmbeddingResult {
    pub output: Vec<f32>,
    pub grad_weight: Vec<f32>,
}

/// Vulkan-native gather/scatter edge for Hierarchos' tied token embedding.
///
/// The parameter is stored exactly as PyTorch's `lm_head.weight` tensor with
/// shape `[vocab_size, context_dim]`. Forward is an embedding gather; backward
/// deterministically accumulates repeated-token gradients into that same matrix.
/// The earliest source occurrence owns each sparse destination and folds later
/// contributors in source order, avoiding order-dependent FP32 atomics. The
/// record API is intentionally caller-owned so recurrent schedulers can bind
/// its output directly into SharedTokenAdapter without a host round-trip.
pub struct TiedTokenEmbeddingOp {
    device: VulkanDevice,
    dim: usize,
    vocab_size: usize,
    max_tokens: usize,
    padding_idx: Option<u32>,

    parameter: SharedLmHeadParameter,
    token_ids: GpuBuffer,
    sorted_token_positions: GpuBuffer,
    radix_scratch_positions: GpuBuffer,
    radix_block_histograms: GpuBuffer,
    radix_block_offsets: GpuBuffer,
    grad_output: GpuBuffer,
    output: GpuBuffer,
    output_readback: GpuBuffer,
    grad_weight_readback: GpuBuffer,

    embedding_forward: vulkan::ComputeKernel,
    embedding_grad_accumulate: vulkan::ComputeKernel,
    embedding_token_sort: vulkan::ComputeKernel,
    embedding_radix_histogram: vulkan::ComputeKernel,
    embedding_radix_prefix: vulkan::ComputeKernel,
    embedding_radix_scatter: vulkan::ComputeKernel,
    embedding_grad_segmented: vulkan::ComputeKernel,
}

impl TiedTokenEmbeddingOp {
    pub fn new(
        device: VulkanDevice,
        dim: usize,
        vocab_size: usize,
        max_tokens: usize,
        weight: &[f32],
    ) -> Result<Self> {
        if dim == 0 || vocab_size == 0 || max_tokens == 0 {
            bail!("tied embedding dim, vocab_size, and max_tokens must be positive");
        }
        let weight_len = vocab_size
            .checked_mul(dim)
            .context("tied embedding weight capacity overflow")?;
        if weight.len() != weight_len {
            bail!(
                "tied embedding weight has {} values; expected {weight_len} for [{vocab_size}, {dim}]",
                weight.len()
            );
        }
        if weight.iter().any(|value| !value.is_finite()) {
            bail!("tied embedding weight contains non-finite values");
        }
        let parameter = SharedLmHeadParameter::new(device, dim, vocab_size, weight)?;
        Self::from_shared_parameter(parameter, max_tokens)
    }

    pub fn from_shared_parameter(
        parameter: SharedLmHeadParameter,
        max_tokens: usize,
    ) -> Result<Self> {
        Self::from_shared_parameter_with_padding_idx(parameter, max_tokens, None)
    }

    /// Construct an embedding edge that mirrors `torch.nn.Embedding`'s
    /// `padding_idx` backward contract. Forward still gathers the stored pad
    /// row normally; only gradient accumulation into that row is suppressed.
    /// This matters when the same parameter is also used by a tied LM head,
    /// because output-head gradients must still be allowed to update the row.
    pub(crate) fn from_shared_parameter_with_padding_idx(
        parameter: SharedLmHeadParameter,
        max_tokens: usize,
        padding_idx: Option<usize>,
    ) -> Result<Self> {
        let dim = parameter.context_dim();
        let vocab_size = parameter.vocab_size();
        if max_tokens == 0 {
            bail!("tied embedding max_tokens must be positive");
        }
        let padding_idx = padding_idx
            .map(|index| {
                if index >= vocab_size {
                    bail!(
                        "tied embedding padding_idx {index} is outside vocabulary size {vocab_size}"
                    );
                }
                u32::try_from(index).context("tied embedding padding_idx exceeds u32")
            })
            .transpose()?;
        let device = parameter.device();
        let weight_len = vocab_size
            .checked_mul(dim)
            .context("tied embedding weight capacity overflow")?;
        let output_len = max_tokens
            .checked_mul(dim)
            .context("tied embedding output capacity overflow")?;
        let radix_block_capacity = max_tokens.div_ceil(EMBEDDING_RADIX_BLOCK_SIZE);
        let radix_scratch_len = radix_block_capacity
            .checked_mul(EMBEDDING_RADIX_BUCKETS)
            .context("tied embedding radix scratch capacity overflow")?;

        Ok(Self {
            embedding_forward: vulkan::ComputeKernel::new(
                &device,
                EMBEDDING_FORWARD_SPV,
                3,
                std::mem::size_of::<EmbeddingPush>() as u32,
            )?,
            embedding_grad_accumulate: vulkan::ComputeKernel::new(
                &device,
                EMBEDDING_GRAD_ACCUMULATE_SPV,
                3,
                std::mem::size_of::<EmbeddingBackwardPush>() as u32,
            )?,
            embedding_token_sort: vulkan::ComputeKernel::new(
                &device,
                EMBEDDING_TOKEN_SORT_SPV,
                2,
                std::mem::size_of::<EmbeddingPush>() as u32,
            )?,
            embedding_radix_histogram: vulkan::ComputeKernel::new_with_access(
                &device,
                EMBEDDING_RADIX_HISTOGRAM_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::WriteOnly,
                ],
                std::mem::size_of::<EmbeddingRadixPush>() as u32,
            )?,
            embedding_radix_prefix: vulkan::ComputeKernel::new_with_access(
                &device,
                EMBEDDING_RADIX_PREFIX_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::WriteOnly,
                ],
                std::mem::size_of::<EmbeddingRadixPush>() as u32,
            )?,
            embedding_radix_scatter: vulkan::ComputeKernel::new_with_access(
                &device,
                EMBEDDING_RADIX_SCATTER_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::WriteOnly,
                ],
                std::mem::size_of::<EmbeddingRadixPush>() as u32,
            )?,
            embedding_grad_segmented: vulkan::ComputeKernel::new(
                &device,
                EMBEDDING_GRAD_SEGMENTED_SPV,
                4,
                std::mem::size_of::<EmbeddingBackwardPush>() as u32,
            )?,
            parameter,
            token_ids: GpuBuffer::zeros_u32(&device, max_tokens)?,
            sorted_token_positions: GpuBuffer::zeros_u32(&device, max_tokens)?,
            radix_scratch_positions: GpuBuffer::zeros_u32(&device, max_tokens)?,
            radix_block_histograms: GpuBuffer::zeros_u32(&device, radix_scratch_len)?,
            radix_block_offsets: GpuBuffer::zeros_u32(&device, radix_scratch_len)?,
            grad_output: GpuBuffer::zeros_f32(&device, output_len)?,
            output: GpuBuffer::zeros_f32(&device, output_len)?,
            output_readback: GpuBuffer::zeros_host_f32(&device, output_len)?,
            grad_weight_readback: GpuBuffer::zeros_host_f32(&device, weight_len)?,
            device,
            dim,
            vocab_size,
            max_tokens,
            padding_idx,
        })
    }

    pub fn from_model_package(
        device: VulkanDevice,
        model_dir: impl AsRef<Path>,
        max_tokens: usize,
    ) -> Result<Self> {
        let path = model_dir.as_ref().join("model.safetensors");
        let (shape, weight) = read_f32_tensor(&path, "lm_head.weight")?;
        let (vocab_size, dim) = match shape.as_slice() {
            [vocab_size, dim] if *vocab_size > 0 && *dim > 0 => (*vocab_size, *dim),
            _ => bail!("lm_head.weight must have shape [vocab_size, context_dim], got {shape:?}"),
        };
        Self::new(device, dim, vocab_size, max_tokens, &weight)
    }

    pub fn forward_backward(
        &mut self,
        token_ids: &[u32],
        grad_output: &[f32],
    ) -> Result<TiedTokenEmbeddingResult> {
        self.validate_token_ids(token_ids)?;
        let token_count = token_ids.len();
        let output_len = token_count * self.dim;
        if grad_output.len() != output_len {
            bail!(
                "tied embedding grad_output has {} values; expected {output_len}",
                grad_output.len()
            );
        }
        if grad_output.iter().any(|value| !value.is_finite()) {
            bail!("tied embedding grad_output contains non-finite values");
        }

        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.upload_u32(&self.token_ids, token_ids)?;
        commands.upload_f32(&self.grad_output, grad_output)?;
        self.record_zero_grad(&mut commands)?;
        self.record_forward(&mut commands, token_count, &self.token_ids, &self.output)?;
        self.record_backward_accumulate(
            &mut commands,
            token_count,
            &self.token_ids,
            &self.grad_output,
        )?;
        commands.readback_f32(&self.output, &self.output_readback, output_len)?;
        commands.readback_f32(
            self.parameter.gradient_buffer(),
            &self.grad_weight_readback,
            self.vocab_size * self.dim,
        )?;
        commands.submit()?;

        Ok(TiedTokenEmbeddingResult {
            output: self.output_readback.read_f32(output_len)?,
            grad_weight: self
                .grad_weight_readback
                .read_f32(self.vocab_size * self.dim)?,
        })
    }

    pub(crate) fn record_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        token_count: usize,
        token_ids: &GpuBuffer,
        output: &GpuBuffer,
    ) -> Result<()> {
        self.validate_token_count(token_count)?;
        let push = self.push(token_count);
        self.embedding_forward.record_dispatch(
            commands,
            &[token_ids, self.parameter.weight_buffer(), output],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(self.dim, 16), div_ceil_u32(token_count, 16), 1],
        )
    }

    pub(crate) fn record_zero_grad(&self, commands: &mut vulkan::ComputeBatch) -> Result<()> {
        self.parameter.record_zero_grad(commands)
    }

    pub(crate) fn record_backward_accumulate(
        &self,
        commands: &mut vulkan::ComputeBatch,
        token_count: usize,
        token_ids: &GpuBuffer,
        grad_output: &GpuBuffer,
    ) -> Result<()> {
        self.validate_token_count(token_count)?;
        let push = self.backward_push(token_count);
        if token_count <= EMBEDDING_DIRECT_ACCUMULATE_CAPACITY {
            return self.embedding_grad_accumulate.record_dispatch(
                commands,
                &[token_ids, grad_output, self.parameter.gradient_buffer()],
                bytemuck::bytes_of(&push),
                [div_ceil_u32(self.dim, 16), div_ceil_u32(token_count, 16), 1],
            );
        }
        let sorted_positions = if token_count <= EMBEDDING_SEGMENTED_SORT_CAPACITY {
            let sort_push = self.push(token_count);
            self.embedding_token_sort.record_dispatch(
                commands,
                &[token_ids, &self.sorted_token_positions],
                bytemuck::bytes_of(&sort_push),
                [1, 1, 1],
            )?;
            &self.sorted_token_positions
        } else {
            self.record_radix_sort(commands, token_count, token_ids)?
        };
        self.embedding_grad_segmented.record_dispatch(
            commands,
            &[
                token_ids,
                sorted_positions,
                grad_output,
                self.parameter.gradient_buffer(),
            ],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(self.dim, 16), div_ceil_u32(token_count, 16), 1],
        )
    }

    fn record_radix_sort<'a>(
        &'a self,
        commands: &mut vulkan::ComputeBatch,
        token_count: usize,
        token_ids: &GpuBuffer,
    ) -> Result<&'a GpuBuffer> {
        let block_count = token_count.div_ceil(EMBEDDING_RADIX_BLOCK_SIZE);
        let pass_count = embedding_radix_pass_count(self.vocab_size);
        for pass in 0..pass_count {
            let source_is_identity = pass == 0;
            let source_positions = if pass == 0 || pass.is_multiple_of(2) {
                &self.sorted_token_positions
            } else {
                &self.radix_scratch_positions
            };
            let destination_positions = if pass.is_multiple_of(2) {
                &self.radix_scratch_positions
            } else {
                &self.sorted_token_positions
            };
            let radix_push = EmbeddingRadixPush {
                token_count: token_count as u32,
                shift: (pass * 4) as u32,
                block_count: block_count as u32,
                source_is_identity: u32::from(source_is_identity),
            };
            self.embedding_radix_histogram.record_dispatch(
                commands,
                &[token_ids, source_positions, &self.radix_block_histograms],
                bytemuck::bytes_of(&radix_push),
                [block_count as u32, 1, 1],
            )?;
            self.embedding_radix_prefix.record_dispatch(
                commands,
                &[&self.radix_block_histograms, &self.radix_block_offsets],
                bytemuck::bytes_of(&radix_push),
                [1, 1, 1],
            )?;
            self.embedding_radix_scatter.record_dispatch(
                commands,
                &[
                    token_ids,
                    source_positions,
                    &self.radix_block_offsets,
                    destination_positions,
                ],
                bytemuck::bytes_of(&radix_push),
                [block_count as u32, 1, 1],
            )?;
        }
        Ok(if pass_count.is_multiple_of(2) {
            &self.sorted_token_positions
        } else {
            &self.radix_scratch_positions
        })
    }

    pub fn weights(&self) -> Result<Vec<f32>> {
        self.parameter.weights()
    }

    pub fn shared_parameter(&self) -> SharedLmHeadParameter {
        self.parameter.clone()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    pub(crate) fn validate_token_ids(&self, token_ids: &[u32]) -> Result<()> {
        self.validate_token_count(token_ids.len())?;
        if let Some(&bad) = token_ids
            .iter()
            .find(|&&token| token as usize >= self.vocab_size)
        {
            bail!(
                "tied embedding token {bad} is outside vocabulary size {}",
                self.vocab_size
            );
        }
        Ok(())
    }

    fn validate_token_count(&self, token_count: usize) -> Result<()> {
        if token_count == 0 || token_count > self.max_tokens {
            bail!(
                "tied embedding token count must be in 1..={}; got {token_count}",
                self.max_tokens
            );
        }
        Ok(())
    }

    fn push(&self, token_count: usize) -> EmbeddingPush {
        EmbeddingPush {
            token_count: token_count as u32,
            dim: self.dim as u32,
            vocab_size: self.vocab_size as u32,
        }
    }

    fn backward_push(&self, token_count: usize) -> EmbeddingBackwardPush {
        EmbeddingBackwardPush {
            token_count: token_count as u32,
            dim: self.dim as u32,
            vocab_size: self.vocab_size as u32,
            padding_idx: self.padding_idx.unwrap_or(u32::MAX),
        }
    }
}

fn div_ceil_u32(value: usize, divisor: usize) -> u32 {
    value.div_ceil(divisor) as u32
}

fn embedding_radix_pass_count(vocab_size: usize) -> usize {
    let highest_token = u32::try_from(vocab_size.saturating_sub(1)).unwrap_or(u32::MAX);
    let significant_bits = (u32::BITS - highest_token.leading_zeros()).max(1) as usize;
    significant_bits.div_ceil(4)
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;

    #[test]
    fn repeated_token_gradient_reduction_is_fixed_order_and_repeatable() -> Result<()> {
        let device = VulkanDevice::new()?;
        let dim = 3usize;
        let vocab_size = 8usize;
        let token_ids = [2u32, 5, 2, 5, 2];
        let grad_output = [
            0.10f32, -0.20, 0.30, 0.40, 0.50, -0.60, -0.70, 0.80, 0.90, 1.00, -1.10, 1.20, 1.30,
            1.40, -1.50,
        ];
        let mut expected = vec![0.0f32; vocab_size * dim];
        for (position, &token) in token_ids.iter().enumerate() {
            for col in 0..dim {
                expected[token as usize * dim + col] += grad_output[position * dim + col];
            }
        }

        let mut op = TiedTokenEmbeddingOp::new(
            device,
            dim,
            vocab_size,
            token_ids.len(),
            &vec![0.0; vocab_size * dim],
        )?;
        let first = op.forward_backward(&token_ids, &grad_output)?;
        let second = op.forward_backward(&token_ids, &grad_output)?;

        assert_eq!(first.grad_weight, expected);
        assert_eq!(second.grad_weight, expected);
        assert_eq!(first.grad_weight, second.grad_weight);
        Ok(())
    }

    #[test]
    fn padding_idx_suppresses_only_embedding_gradient_for_that_row() -> Result<()> {
        let device = VulkanDevice::new()?;
        let dim = 3usize;
        let vocab_size = 8usize;
        let padding_idx = 2usize;
        let token_ids = [2u32, 5, 2, 1];
        let grad_output = [
            0.10f32, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20,
        ];
        let parameter =
            SharedLmHeadParameter::new(device, dim, vocab_size, &vec![0.0; vocab_size * dim])?;
        let mut op = TiedTokenEmbeddingOp::from_shared_parameter_with_padding_idx(
            parameter,
            token_ids.len(),
            Some(padding_idx),
        )?;
        let result = op.forward_backward(&token_ids, &grad_output)?;

        assert_eq!(
            &result.grad_weight[padding_idx * dim..(padding_idx + 1) * dim],
            &[0.0, 0.0, 0.0]
        );
        assert_eq!(&result.grad_weight[5 * dim..6 * dim], &[0.40, 0.50, 0.60]);
        assert_eq!(&result.grad_weight[dim..2 * dim], &[1.00, 1.10, 1.20]);
        Ok(())
    }

    #[test]
    fn segmented_gradient_adds_after_existing_device_gradient_in_one_submission() -> Result<()> {
        let device = VulkanDevice::new()?;
        let dim = 3usize;
        let vocab_size = 8usize;
        let token_count = EMBEDDING_DIRECT_ACCUMULATE_CAPACITY + 7;
        let token_ids = (0..token_count)
            .map(|index| ((index * 3 + index / 5) % vocab_size) as u32)
            .collect::<Vec<_>>();
        let grad_output = (0..token_count * dim)
            .map(|index| ((index % 23) as f32 - 11.0) * (1.0 / 64.0))
            .collect::<Vec<_>>();
        let baseline = vec![0.03125f32; vocab_size * dim];
        let baseline_buffer = GpuBuffer::from_f32(&device, &baseline)?;
        let readback = GpuBuffer::zeros_host_f32(&device, baseline.len())?;
        let op = TiedTokenEmbeddingOp::new(
            device.clone(),
            dim,
            vocab_size,
            token_count,
            &vec![0.0; vocab_size * dim],
        )?;
        let id_buffer = GpuBuffer::from_u32(&device, &token_ids)?;
        let grad_buffer = GpuBuffer::from_f32(&device, &grad_output)?;

        let mut sparse = vec![0.0f32; vocab_size * dim];
        for (position, &token) in token_ids.iter().enumerate() {
            for col in 0..dim {
                sparse[token as usize * dim + col] += grad_output[position * dim + col];
            }
        }
        let expected = baseline
            .iter()
            .zip(&sparse)
            .map(|(base, delta)| base + delta)
            .collect::<Vec<_>>();

        let mut commands = vulkan::ComputeBatch::new(&device)?;
        op.record_zero_grad(&mut commands)?;
        op.parameter
            .record_accumulate_gradient(&mut commands, &baseline_buffer)?;
        op.record_backward_accumulate(&mut commands, token_ids.len(), &id_buffer, &grad_buffer)?;
        commands.readback_f32(op.parameter.gradient_buffer(), &readback, baseline.len())?;
        commands.submit()?;

        assert_eq!(readback.read_f32(baseline.len())?, expected);
        Ok(())
    }

    #[test]
    fn multi_workgroup_radix_gradient_reduction_preserves_source_order() -> Result<()> {
        let device = VulkanDevice::new()?;
        let dim = 3usize;
        // 521 requires three 4-bit LSD passes, exercising the odd-pass
        // ping-pong result path as well as the multi-workgroup scatter itself.
        let vocab_size = 521usize;
        let token_count = EMBEDDING_SEGMENTED_SORT_CAPACITY + 513;
        let token_ids = (0..token_count)
            .map(|index| ((index * 37 + index / 11) % vocab_size) as u32)
            .collect::<Vec<_>>();
        let grad_output = (0..token_count * dim)
            .map(|index| ((index % 31) as f32 - 15.0) * (1.0 / 128.0))
            .collect::<Vec<_>>();
        let mut expected = vec![0.0f32; vocab_size * dim];
        for (position, &token) in token_ids.iter().enumerate() {
            for col in 0..dim {
                expected[token as usize * dim + col] += grad_output[position * dim + col];
            }
        }

        let mut op = TiedTokenEmbeddingOp::new(
            device,
            dim,
            vocab_size,
            token_count,
            &vec![0.0; vocab_size * dim],
        )?;
        let first = op.forward_backward(&token_ids, &grad_output)?;
        let second = op.forward_backward(&token_ids, &grad_output)?;

        assert_eq!(first.grad_weight, expected);
        assert_eq!(second.grad_weight, expected);
        assert_eq!(first.grad_weight, second.grad_weight);
        Ok(())
    }
}
