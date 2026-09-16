use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::{vulkan, GpuBuffer, VulkanDevice};

const RWKV_MATRIX_STATE_FORWARD_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_matrix_state_forward.spv");
const RWKV_MATRIX_STATE_BACKWARD_ROWS_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_matrix_state_backward_rows.spv");
const RWKV_MATRIX_STATE_BACKWARD_COLS_SPV: &[u8] =
    include_bytes!("../shaders/rwkv_matrix_state_backward_cols.spv");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RwkvStatePush {
    batch: u32,
    width: u32,
    head_size: u32,
}

#[derive(Debug)]
pub struct RwkvMatrixStateResult {
    pub new_state: Vec<f32>,
    pub tmix: Vec<f32>,
    pub grad_state: Vec<f32>,
    pub grad_r: Vec<f32>,
    pub grad_k: Vec<f32>,
    pub grad_v: Vec<f32>,
    pub grad_kk: Vec<f32>,
    pub grad_a: Vec<f32>,
    pub grad_w: Vec<f32>,
}

/// Vulkan FP32 implementation of the coherent-v9 / RWKV-v8 matrix-state core.
///
/// The operation consumes already-projected `r`, scaled `k`, `v`, normalized
/// `kk`, in-context rate `a`, and log-decay `w`, matching the explicit FP32
/// autocast-disabled block in `hierarchos/models/rwkv_cell.py`:
///
/// `S' = S * exp(-exp(clamp(w))) + (S @ -kk) outer (kk*a) + v outer k`
/// `tmix = S' @ r`
///
/// Backward accepts gradients from both consumers of this primitive: the next
/// recurrent matrix state and the current `tmix` readout. Row and column
/// reductions are separated so every gradient is deterministic and requires no
/// floating-point atomics or vendor Vulkan extensions.
pub struct RwkvMatrixStateOp {
    device: VulkanDevice,
    width: usize,
    head_size: usize,
    heads: usize,
    max_batch: usize,

    state: GpuBuffer,
    r: GpuBuffer,
    k: GpuBuffer,
    v: GpuBuffer,
    kk: GpuBuffer,
    a: GpuBuffer,
    w: GpuBuffer,
    grad_new_state: GpuBuffer,
    grad_tmix: GpuBuffer,

    new_state: GpuBuffer,
    tmix: GpuBuffer,
    saved_sa: GpuBuffer,
    saved_q: GpuBuffer,
    grad_state: GpuBuffer,
    grad_r: GpuBuffer,
    grad_k: GpuBuffer,
    grad_v: GpuBuffer,
    grad_kk: GpuBuffer,
    grad_a: GpuBuffer,
    grad_w: GpuBuffer,

    new_state_readback: GpuBuffer,
    tmix_readback: GpuBuffer,
    grad_state_readback: GpuBuffer,
    grad_r_readback: GpuBuffer,
    grad_k_readback: GpuBuffer,
    grad_v_readback: GpuBuffer,
    grad_kk_readback: GpuBuffer,
    grad_a_readback: GpuBuffer,
    grad_w_readback: GpuBuffer,

    forward_kernel: vulkan::ComputeKernel,
    backward_rows_kernel: vulkan::ComputeKernel,
    backward_cols_kernel: vulkan::ComputeKernel,
}

impl RwkvMatrixStateOp {
    pub fn new(
        device: VulkanDevice,
        width: usize,
        head_size: usize,
        max_batch: usize,
    ) -> Result<Self> {
        if width == 0 || head_size == 0 || max_batch == 0 {
            bail!("RWKV matrix-state width, head_size, and max_batch must be positive");
        }
        if !width.is_multiple_of(head_size) {
            bail!("RWKV width {width} must be divisible by head_size {head_size}");
        }
        let heads = width / head_size;
        let vector_len = max_batch
            .checked_mul(width)
            .context("RWKV vector capacity overflow")?;
        let state_len = vector_len
            .checked_mul(head_size)
            .context("RWKV matrix-state capacity overflow")?;

        Ok(Self {
            forward_kernel: vulkan::ComputeKernel::new(
                &device,
                RWKV_MATRIX_STATE_FORWARD_SPV,
                10,
                std::mem::size_of::<RwkvStatePush>() as u32,
            )?,
            backward_rows_kernel: vulkan::ComputeKernel::new(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_ROWS_SPV,
                11,
                std::mem::size_of::<RwkvStatePush>() as u32,
            )?,
            backward_cols_kernel: vulkan::ComputeKernel::new(
                &device,
                RWKV_MATRIX_STATE_BACKWARD_COLS_SPV,
                17,
                std::mem::size_of::<RwkvStatePush>() as u32,
            )?,
            state: GpuBuffer::zeros_f32(&device, state_len)?,
            r: GpuBuffer::zeros_f32(&device, vector_len)?,
            k: GpuBuffer::zeros_f32(&device, vector_len)?,
            v: GpuBuffer::zeros_f32(&device, vector_len)?,
            kk: GpuBuffer::zeros_f32(&device, vector_len)?,
            a: GpuBuffer::zeros_f32(&device, vector_len)?,
            w: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_new_state: GpuBuffer::zeros_f32(&device, state_len)?,
            grad_tmix: GpuBuffer::zeros_f32(&device, vector_len)?,
            new_state: GpuBuffer::zeros_f32(&device, state_len)?,
            tmix: GpuBuffer::zeros_f32(&device, vector_len)?,
            saved_sa: GpuBuffer::zeros_f32(&device, vector_len)?,
            saved_q: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_state: GpuBuffer::zeros_f32(&device, state_len)?,
            grad_r: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_k: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_v: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_kk: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_a: GpuBuffer::zeros_f32(&device, vector_len)?,
            grad_w: GpuBuffer::zeros_f32(&device, vector_len)?,
            new_state_readback: GpuBuffer::zeros_host_f32(&device, state_len)?,
            tmix_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_state_readback: GpuBuffer::zeros_host_f32(&device, state_len)?,
            grad_r_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_k_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_v_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_kk_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_a_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            grad_w_readback: GpuBuffer::zeros_host_f32(&device, vector_len)?,
            device,
            width,
            head_size,
            heads,
            max_batch,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_backward(
        &mut self,
        batch: usize,
        state: &[f32],
        r: &[f32],
        k: &[f32],
        v: &[f32],
        kk: &[f32],
        a: &[f32],
        w: &[f32],
        grad_new_state: &[f32],
        grad_tmix: &[f32],
    ) -> Result<RwkvMatrixStateResult> {
        if batch == 0 || batch > self.max_batch {
            bail!(
                "RWKV matrix-state batch must be in 1..={}; got {batch}",
                self.max_batch
            );
        }
        let vector_len = batch
            .checked_mul(self.width)
            .context("RWKV vector size overflow")?;
        let state_len = vector_len
            .checked_mul(self.head_size)
            .context("RWKV state size overflow")?;
        validate_len("state", state, state_len)?;
        validate_len("r", r, vector_len)?;
        validate_len("k", k, vector_len)?;
        validate_len("v", v, vector_len)?;
        validate_len("kk", kk, vector_len)?;
        validate_len("a", a, vector_len)?;
        validate_len("w", w, vector_len)?;
        validate_len("grad_new_state", grad_new_state, state_len)?;
        validate_len("grad_tmix", grad_tmix, vector_len)?;

        let mut batch_commands = vulkan::ComputeBatch::new(&self.device)?;
        batch_commands.upload_f32(&self.state, state)?;
        batch_commands.upload_f32(&self.r, r)?;
        batch_commands.upload_f32(&self.k, k)?;
        batch_commands.upload_f32(&self.v, v)?;
        batch_commands.upload_f32(&self.kk, kk)?;
        batch_commands.upload_f32(&self.a, a)?;
        batch_commands.upload_f32(&self.w, w)?;
        batch_commands.upload_f32(&self.grad_new_state, grad_new_state)?;
        batch_commands.upload_f32(&self.grad_tmix, grad_tmix)?;

        let push = RwkvStatePush {
            batch: batch as u32,
            width: self.width as u32,
            head_size: self.head_size as u32,
        };
        let groups = [div_ceil_u32(vector_len, 64), 1, 1];
        self.forward_kernel.record_dispatch(
            &mut batch_commands,
            &[
                &self.state,
                &self.r,
                &self.k,
                &self.v,
                &self.kk,
                &self.a,
                &self.w,
                &self.new_state,
                &self.tmix,
                &self.saved_sa,
            ],
            bytemuck::bytes_of(&push),
            groups,
        )?;
        self.backward_rows_kernel.record_dispatch(
            &mut batch_commands,
            &[
                &self.state,
                &self.r,
                &self.k,
                &self.kk,
                &self.a,
                &self.w,
                &self.grad_new_state,
                &self.grad_tmix,
                &self.grad_state,
                &self.grad_v,
                &self.saved_q,
            ],
            bytemuck::bytes_of(&push),
            groups,
        )?;
        self.backward_cols_kernel.record_dispatch(
            &mut batch_commands,
            &[
                &self.state,
                &self.new_state,
                &self.r,
                &self.k,
                &self.v,
                &self.kk,
                &self.a,
                &self.w,
                &self.saved_sa,
                &self.saved_q,
                &self.grad_new_state,
                &self.grad_tmix,
                &self.grad_r,
                &self.grad_k,
                &self.grad_kk,
                &self.grad_a,
                &self.grad_w,
            ],
            bytemuck::bytes_of(&push),
            groups,
        )?;

        batch_commands.readback_f32(&self.new_state, &self.new_state_readback, state_len)?;
        batch_commands.readback_f32(&self.tmix, &self.tmix_readback, vector_len)?;
        batch_commands.readback_f32(&self.grad_state, &self.grad_state_readback, state_len)?;
        batch_commands.readback_f32(&self.grad_r, &self.grad_r_readback, vector_len)?;
        batch_commands.readback_f32(&self.grad_k, &self.grad_k_readback, vector_len)?;
        batch_commands.readback_f32(&self.grad_v, &self.grad_v_readback, vector_len)?;
        batch_commands.readback_f32(&self.grad_kk, &self.grad_kk_readback, vector_len)?;
        batch_commands.readback_f32(&self.grad_a, &self.grad_a_readback, vector_len)?;
        batch_commands.readback_f32(&self.grad_w, &self.grad_w_readback, vector_len)?;
        batch_commands.submit()?;

        Ok(RwkvMatrixStateResult {
            new_state: self.new_state_readback.read_f32(state_len)?,
            tmix: self.tmix_readback.read_f32(vector_len)?,
            grad_state: self.grad_state_readback.read_f32(state_len)?,
            grad_r: self.grad_r_readback.read_f32(vector_len)?,
            grad_k: self.grad_k_readback.read_f32(vector_len)?,
            grad_v: self.grad_v_readback.read_f32(vector_len)?,
            grad_kk: self.grad_kk_readback.read_f32(vector_len)?,
            grad_a: self.grad_a_readback.read_f32(vector_len)?,
            grad_w: self.grad_w_readback.read_f32(vector_len)?,
        })
    }

    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    pub fn heads(&self) -> usize {
        self.heads
    }
}

fn validate_len(name: &str, values: &[f32], expected: usize) -> Result<()> {
    if values.len() != expected {
        bail!(
            "RWKV {name} has {} values; expected {expected}",
            values.len()
        );
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("RWKV {name} contains non-finite values");
    }
    Ok(())
}

fn div_ceil_u32(value: usize, divisor: usize) -> u32 {
    value.div_ceil(divisor) as u32
}
