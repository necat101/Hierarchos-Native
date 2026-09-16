use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::{vulkan, GpuBuffer, VulkanDevice};

const STATE_UNPACK_SPV: &[u8] = include_bytes!("../shaders/rwkv_state_unpack.spv");
const STATE_UNPACK_VECTORS_SPV: &[u8] = include_bytes!("../shaders/rwkv_state_unpack_vectors.spv");
const STATE_PACK_SPV: &[u8] = include_bytes!("../shaders/rwkv_state_pack.spv");
const STATE_PACK_VECTORS_SPV: &[u8] = include_bytes!("../shaders/rwkv_state_pack_vectors.spv");
const STATE_PACK_BACKWARD_SPV: &[u8] = include_bytes!("../shaders/rwkv_state_pack_backward.spv");
const STATE_PACK_BACKWARD_FUSED_ADD_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_state_pack_backward_fused_add.spv");
const STATE_GRAD_PACK_SPV: &[u8] = include_bytes!("../shaders/rwkv_state_grad_pack.spv");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StatePush {
    batch: u32,
    width: u32,
    head_size: u32,
    matrix_offset: u32,
    state_clamp: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RwkvStateReadoutMode {
    LegacyInputCache,
    ExplicitOutput,
}

impl RwkvStateReadoutMode {
    pub fn matrix_offset(self) -> usize {
        match self {
            Self::LegacyInputCache => 3,
            Self::ExplicitOutput => 4,
        }
    }
}

#[derive(Debug)]
pub struct RwkvPackedStateResult {
    pub previous_tm: Vec<f32>,
    pub previous_cm: Vec<f32>,
    pub previous_v_first: Vec<f32>,
    pub matrix_state: Vec<f32>,
    pub packed_new_state: Vec<f32>,
    pub grad_x_norm: Vec<f32>,
    pub grad_x_norm2: Vec<f32>,
    pub grad_v_first: Vec<f32>,
    pub grad_output: Vec<f32>,
    pub grad_matrix_state: Vec<f32>,
}

/// Differentiable Vulkan implementation of Hierarchos' public packed RWKV
/// state contract. It owns both state layouts, finite-preserving state clamps,
/// and the backward mask of that clamp so TBPTT can carry gradients through
/// cache, explicit-output, and matrix-state slots without CPU unpacking.
pub struct RwkvPackedStateOp {
    device: VulkanDevice,
    width: usize,
    head_size: usize,
    max_batch: usize,
    mode: RwkvStateReadoutMode,
    state_clamp: f32,

    packed_state: GpuBuffer,
    previous_tm: GpuBuffer,
    previous_cm: GpuBuffer,
    previous_v_first: GpuBuffer,
    matrix_state: GpuBuffer,

    x_norm: GpuBuffer,
    x_norm2: GpuBuffer,
    v_first: GpuBuffer,
    output: GpuBuffer,
    new_matrix_state: GpuBuffer,
    packed_new_state: GpuBuffer,
    grad_packed_new_state: GpuBuffer,
    grad_x_norm: GpuBuffer,
    grad_x_norm2: GpuBuffer,
    grad_v_first: GpuBuffer,
    grad_output: GpuBuffer,
    grad_matrix_state: GpuBuffer,
    grad_packed_input: GpuBuffer,

    previous_tm_readback: GpuBuffer,
    previous_cm_readback: GpuBuffer,
    previous_v_first_readback: GpuBuffer,
    matrix_state_readback: GpuBuffer,
    packed_new_state_readback: GpuBuffer,
    grad_x_norm_readback: GpuBuffer,
    grad_x_norm2_readback: GpuBuffer,
    grad_v_first_readback: GpuBuffer,
    grad_output_readback: GpuBuffer,
    grad_matrix_state_readback: GpuBuffer,
    grad_packed_input_readback: GpuBuffer,

    unpack: vulkan::ComputeKernel,
    unpack_vectors: vulkan::ComputeKernel,
    pack: vulkan::ComputeKernel,
    pack_vectors: vulkan::ComputeKernel,
    pack_backward: vulkan::ComputeKernel,
    pack_backward_fused_add: vulkan::ComputeKernel,
    grad_pack: vulkan::ComputeKernel,
}

impl RwkvPackedStateOp {
    pub fn new(
        device: VulkanDevice,
        width: usize,
        head_size: usize,
        max_batch: usize,
        mode: RwkvStateReadoutMode,
        state_clamp: f32,
    ) -> Result<Self> {
        if width == 0 || head_size == 0 || max_batch == 0 {
            bail!("packed RWKV state width, head_size, and max_batch must be positive");
        }
        if !width.is_multiple_of(head_size) {
            bail!("packed RWKV state width {width} must be divisible by head_size {head_size}");
        }
        if !state_clamp.is_finite() || state_clamp < 0.0 {
            bail!("packed RWKV state clamp must be finite and non-negative");
        }
        let vector_len = max_batch
            .checked_mul(width)
            .context("packed RWKV vector capacity overflow")?;
        let matrix_len = vector_len
            .checked_mul(head_size)
            .context("packed RWKV matrix capacity overflow")?;
        let state_size = mode.matrix_offset() + head_size;
        let packed_len = vector_len
            .checked_mul(state_size)
            .context("packed RWKV state capacity overflow")?;

        Ok(Self {
            unpack: vulkan::ComputeKernel::new_with_access(
                &device,
                STATE_UNPACK_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<StatePush>() as u32,
            )?,
            unpack_vectors: vulkan::ComputeKernel::new_with_access(
                &device,
                STATE_UNPACK_VECTORS_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<StatePush>() as u32,
            )?,
            pack: vulkan::ComputeKernel::new_with_access(
                &device,
                STATE_PACK_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<StatePush>() as u32,
            )?,
            pack_vectors: vulkan::ComputeKernel::new_with_access(
                &device,
                STATE_PACK_VECTORS_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<StatePush>() as u32,
            )?,
            pack_backward: vulkan::ComputeKernel::new_with_access(
                &device,
                STATE_PACK_BACKWARD_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<StatePush>() as u32,
            )?,
            pack_backward_fused_add: vulkan::ComputeKernel::new_with_access(
                &device,
                STATE_PACK_BACKWARD_FUSED_ADD_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<StatePush>() as u32,
            )?,
            grad_pack: vulkan::ComputeKernel::new_with_access(
                &device,
                STATE_GRAD_PACK_SPV,
                &[
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::ReadOnly,
                    vulkan::BindingAccess::MayWrite,
                ],
                std::mem::size_of::<StatePush>() as u32,
            )?,
            packed_state: GpuBuffer::zeros_f32(&device, packed_len)?,
            previous_tm: GpuBuffer::zeros_f32(&device, vector_len)?,
            previous_cm: GpuBuffer::zeros_f32(&device, vector_len)?,
            previous_v_first: GpuBuffer::zeros_f32(&device, vector_len)?,
            matrix_state: GpuBuffer::zeros_f32(&device, matrix_len)?,
            x_norm: GpuBuffer::zeros_f32(&device, vector_len)?,
            x_norm2: GpuBuffer::zeros_f32(&device, vector_len)?,
            v_first: GpuBuffer::zeros_f32(&device, vector_len)?,
            output: GpuBuffer::zeros_f32(&device, vector_len)?,
            new_matrix_state: GpuBuffer::zeros_f32(&device, matrix_len)?,
            packed_new_state: GpuBuffer::zeros_f32(&device, packed_len)?,
            grad_packed_new_state: GpuBuffer::zeros_f32(&device, packed_len)?,
            grad_x_norm: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_x_norm2: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_v_first: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_output: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_matrix_state: GpuBuffer::zeros_f32(&device, matrix_len)?,
            grad_packed_input: GpuBuffer::zeros_f32(&device, packed_len)?,
            previous_tm_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            previous_cm_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            previous_v_first_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            matrix_state_readback: GpuBuffer::zeros_host_f32(&device, matrix_len)?,
            packed_new_state_readback: GpuBuffer::zeros_host_f32(&device, packed_len)?,
            grad_x_norm_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_x_norm2_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_v_first_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_output_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_matrix_state_readback: GpuBuffer::zeros_host_f32(&device, matrix_len)?,
            grad_packed_input_readback: GpuBuffer::zeros_host_f32(&device, packed_len)?,
            device,
            width,
            head_size,
            max_batch,
            mode,
            state_clamp,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_backward(
        &mut self,
        batch: usize,
        packed_state: &[f32],
        x_norm: &[f32],
        x_norm2: &[f32],
        v_first: &[f32],
        output: &[f32],
        new_matrix_state: &[f32],
        grad_packed_new_state: &[f32],
    ) -> Result<RwkvPackedStateResult> {
        self.validate_batch(batch)?;
        let vector_len = batch * self.width;
        let matrix_len = vector_len * self.head_size;
        let packed_len = vector_len * self.state_size();
        validate_len("packed_state", packed_state, packed_len)?;
        validate_len("x_norm", x_norm, vector_len)?;
        validate_len("x_norm2", x_norm2, vector_len)?;
        validate_len("v_first", v_first, vector_len)?;
        validate_len("output", output, vector_len)?;
        validate_len("new_matrix_state", new_matrix_state, matrix_len)?;
        validate_len("grad_packed_new_state", grad_packed_new_state, packed_len)?;

        let mut commands = vulkan::ComputeBatch::new(&self.device)?;
        commands.upload_f32(&self.packed_state, packed_state)?;
        commands.upload_f32(&self.x_norm, x_norm)?;
        commands.upload_f32(&self.x_norm2, x_norm2)?;
        commands.upload_f32(&self.v_first, v_first)?;
        commands.upload_f32(&self.output, output)?;
        commands.upload_f32(&self.new_matrix_state, new_matrix_state)?;
        commands.upload_f32(&self.grad_packed_new_state, grad_packed_new_state)?;
        self.record_unpack(&mut commands, batch, &self.packed_state)?;
        self.record_pack(
            &mut commands,
            batch,
            &self.x_norm,
            &self.x_norm2,
            &self.v_first,
            &self.output,
            &self.new_matrix_state,
        )?;
        self.record_pack_backward(
            &mut commands,
            batch,
            &self.x_norm,
            &self.x_norm2,
            &self.v_first,
            &self.output,
            &self.new_matrix_state,
            &self.grad_packed_new_state,
        )?;
        self.record_readback(&mut commands, batch)?;
        commands.submit()?;
        self.read_result(batch)
    }

    pub(crate) fn record_unpack(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        packed_state: &GpuBuffer,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        let matrix_len = batch * self.width * self.head_size;
        let push = self.push(batch);
        self.unpack.record_dispatch(
            commands,
            &[
                packed_state,
                &self.previous_tm,
                &self.previous_cm,
                &self.previous_v_first,
                &self.matrix_state,
            ],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(matrix_len, 256), 1, 1],
        )
    }

    /// Unpack only the O(width) vector state used by a forward transition.
    /// Matrix state remains in the packed history buffer and can be consumed
    /// directly by the forward-only recurrent kernel. Backward rematerialization
    /// still uses `record_unpack` so its full training tape is unchanged.
    pub(crate) fn record_unpack_vectors(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        packed_state: &GpuBuffer,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        let vector_len = batch * self.width;
        let push = self.push(batch);
        self.unpack_vectors.record_dispatch(
            commands,
            &[
                packed_state,
                &self.previous_tm,
                &self.previous_cm,
                &self.previous_v_first,
            ],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(vector_len, 256), 1, 1],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_pack(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x_norm: &GpuBuffer,
        x_norm2: &GpuBuffer,
        v_first: &GpuBuffer,
        output: &GpuBuffer,
        new_matrix_state: &GpuBuffer,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        let packed_len = batch * self.width * self.state_size();
        let push = self.push(batch);
        self.pack.record_dispatch(
            commands,
            &[
                x_norm,
                x_norm2,
                v_first,
                output,
                new_matrix_state,
                &self.packed_new_state,
            ],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(packed_len, 256), 1, 1],
        )
    }

    /// Complete an already matrix-packed forward state with only its vector
    /// cache/readout slots. This preserves the matrix slots produced directly
    /// by the packed time-mix transition.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_pack_vectors_into(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x_norm: &GpuBuffer,
        x_norm2: &GpuBuffer,
        v_first: &GpuBuffer,
        output: &GpuBuffer,
        packed_new_state: &GpuBuffer,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        let vector_len = batch * self.width;
        let push = self.push(batch);
        self.pack_vectors.record_dispatch(
            commands,
            &[x_norm, x_norm2, v_first, output, packed_new_state],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(vector_len, 256), 1, 1],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_pack_backward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x_norm: &GpuBuffer,
        x_norm2: &GpuBuffer,
        v_first: &GpuBuffer,
        output: &GpuBuffer,
        new_matrix_state: &GpuBuffer,
        grad_packed_new_state: &GpuBuffer,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        let matrix_len = batch * self.width * self.head_size;
        let push = self.push(batch);
        self.pack_backward.record_dispatch(
            commands,
            &[
                x_norm,
                x_norm2,
                v_first,
                output,
                new_matrix_state,
                grad_packed_new_state,
                &self.grad_x_norm,
                &self.grad_x_norm2,
                &self.grad_v_first,
                &self.grad_output,
                &self.grad_matrix_state,
            ],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(matrix_len, 256), 1, 1],
        )
    }

    /// Backpropagate the packed-state clamp and merge the recurrent output
    /// adjoint with the caller's cell-output gradient in the same dispatch.
    /// This is algebraically identical to `record_pack_backward` followed by
    /// `vector_add(grad_output_external, grad_output, grad_output_total)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_pack_backward_fused_add(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        x_norm: &GpuBuffer,
        x_norm2: &GpuBuffer,
        v_first: &GpuBuffer,
        output: &GpuBuffer,
        new_matrix_state: &GpuBuffer,
        grad_packed_new_state: &GpuBuffer,
        grad_output_external: &GpuBuffer,
        grad_output_total: &GpuBuffer,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        let matrix_len = batch * self.width * self.head_size;
        let push = self.push(batch);
        self.pack_backward_fused_add.record_dispatch(
            commands,
            &[
                x_norm,
                x_norm2,
                v_first,
                output,
                new_matrix_state,
                grad_packed_new_state,
                grad_output_external,
                &self.grad_x_norm,
                &self.grad_x_norm2,
                &self.grad_v_first,
                grad_output_total,
                &self.grad_matrix_state,
            ],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(matrix_len, 256), 1, 1],
        )
    }

    pub(crate) fn record_readback(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        let vector_len = batch * self.width;
        let matrix_len = vector_len * self.head_size;
        let packed_len = vector_len * self.state_size();
        commands.readback_f32(&self.previous_tm, &self.previous_tm_readback, vector_len)?;
        commands.readback_f32(&self.previous_cm, &self.previous_cm_readback, vector_len)?;
        commands.readback_f32(
            &self.previous_v_first,
            &self.previous_v_first_readback,
            vector_len,
        )?;
        commands.readback_f32(&self.matrix_state, &self.matrix_state_readback, matrix_len)?;
        commands.readback_f32(
            &self.packed_new_state,
            &self.packed_new_state_readback,
            packed_len,
        )?;
        commands.readback_f32(&self.grad_x_norm, &self.grad_x_norm_readback, vector_len)?;
        commands.readback_f32(&self.grad_x_norm2, &self.grad_x_norm2_readback, vector_len)?;
        commands.readback_f32(&self.grad_v_first, &self.grad_v_first_readback, vector_len)?;
        commands.readback_f32(&self.grad_output, &self.grad_output_readback, vector_len)?;
        commands.readback_f32(
            &self.grad_matrix_state,
            &self.grad_matrix_state_readback,
            matrix_len,
        )
    }

    pub(crate) fn record_pack_input_grad(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
        grad_previous_tm: &GpuBuffer,
        grad_previous_cm: &GpuBuffer,
        grad_previous_v_first: &GpuBuffer,
        grad_matrix_state: &GpuBuffer,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        let packed_len = batch * self.width * self.state_size();
        let push = self.push(batch);
        self.grad_pack.record_dispatch(
            commands,
            &[
                grad_previous_tm,
                grad_previous_cm,
                grad_previous_v_first,
                grad_matrix_state,
                &self.grad_packed_input,
            ],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(packed_len, 256), 1, 1],
        )
    }

    pub(crate) fn record_grad_packed_input_readback(
        &self,
        commands: &mut vulkan::ComputeBatch,
        batch: usize,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        commands.readback_f32(
            &self.grad_packed_input,
            &self.grad_packed_input_readback,
            batch * self.width * self.state_size(),
        )
    }

    pub(crate) fn read_grad_packed_input(&self, batch: usize) -> Result<Vec<f32>> {
        self.validate_batch(batch)?;
        self.grad_packed_input_readback
            .read_f32(batch * self.width * self.state_size())
    }

    pub(crate) fn read_result(&self, batch: usize) -> Result<RwkvPackedStateResult> {
        self.validate_batch(batch)?;
        let vector_len = batch * self.width;
        let matrix_len = vector_len * self.head_size;
        let packed_len = vector_len * self.state_size();
        Ok(RwkvPackedStateResult {
            previous_tm: self.previous_tm_readback.read_f32(vector_len)?,
            previous_cm: self.previous_cm_readback.read_f32(vector_len)?,
            previous_v_first: self.previous_v_first_readback.read_f32(vector_len)?,
            matrix_state: self.matrix_state_readback.read_f32(matrix_len)?,
            packed_new_state: self.packed_new_state_readback.read_f32(packed_len)?,
            grad_x_norm: self.grad_x_norm_readback.read_f32(vector_len)?,
            grad_x_norm2: self.grad_x_norm2_readback.read_f32(vector_len)?,
            grad_v_first: self.grad_v_first_readback.read_f32(vector_len)?,
            grad_output: self.grad_output_readback.read_f32(vector_len)?,
            grad_matrix_state: self.grad_matrix_state_readback.read_f32(matrix_len)?,
        })
    }

    pub(crate) fn previous_tm_buffer(&self) -> &GpuBuffer {
        &self.previous_tm
    }

    pub(crate) fn previous_cm_buffer(&self) -> &GpuBuffer {
        &self.previous_cm
    }

    pub(crate) fn matrix_state_buffer(&self) -> &GpuBuffer {
        &self.matrix_state
    }

    pub(crate) fn packed_new_state_buffer(&self) -> &GpuBuffer {
        &self.packed_new_state
    }

    pub(crate) fn grad_x_norm_buffer(&self) -> &GpuBuffer {
        &self.grad_x_norm
    }

    pub(crate) fn grad_x_norm2_buffer(&self) -> &GpuBuffer {
        &self.grad_x_norm2
    }

    pub(crate) fn grad_v_first_buffer(&self) -> &GpuBuffer {
        &self.grad_v_first
    }

    pub(crate) fn grad_matrix_state_buffer(&self) -> &GpuBuffer {
        &self.grad_matrix_state
    }

    pub(crate) fn grad_packed_input_buffer(&self) -> &GpuBuffer {
        &self.grad_packed_input
    }

    pub fn state_size(&self) -> usize {
        self.mode.matrix_offset() + self.head_size
    }

    pub fn matrix_offset(&self) -> usize {
        self.mode.matrix_offset()
    }

    pub fn state_clamp(&self) -> f32 {
        self.state_clamp
    }

    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    fn validate_batch(&self, batch: usize) -> Result<()> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "packed RWKV state batch must be in 1..={}; got {batch}",
                self.max_batch
            );
        }
        Ok(())
    }

    fn push(&self, batch: usize) -> StatePush {
        StatePush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
            matrix_offset: self.mode.matrix_offset() as u32,
            state_clamp: self.state_clamp,
        }
    }
}

fn validate_len(name: &str, values: &[f32], expected: usize) -> Result<()> {
    if values.len() != expected {
        bail!(
            "packed RWKV state {name} has {} values; expected {expected}",
            values.len()
        );
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("packed RWKV state {name} contains non-finite values");
    }
    Ok(())
}

fn div_ceil_u32(value: usize, divisor: usize) -> u32 {
    value.div_ceil(divisor) as u32
}
