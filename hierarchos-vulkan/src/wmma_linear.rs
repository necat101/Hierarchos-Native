//! Dense-linear WMMA execution for the resident Transformer graph.
//!
//! This is the graph-facing counterpart of the primitive [`crate::wmma`] module:
//! it owns the FP16 execution mirrors a single `VulkanLinear` needs and records
//! the three GEMMs of a training step (forward, weight gradient, input
//! gradient) through the same [`WmmaGemm`] the qualification harness exercises.
//!
//! Mixed and full-FP16 select distinct cooperative kernels. The full kernel
//! stores its FP16 accumulator through shared FP16 storage and explicitly
//! widens each element into the graph's FP32 output buffer on the GPU. Master
//! weights, gradients, and optimizer state therefore retain their FP32 ABI.
//! Full-FP16 accumulation has a separate numerical qualification policy; graph
//! integration alone does not admit it in the desktop GUI.
//!
//! # Weight mirrors
//!
//! A projection is executed twice per optimizer step - once for the forward
//! pass and once for the input gradient - and both read the same master weight.
//! The FP32 master is authoritative, so the FP16 mirror is only an executor
//! input, and it may be reused for as long as the master has not changed.
//!
//! Reuse is keyed by `(allocation, length, epoch)`:
//!
//! * `allocation` is the identity of the master `GpuBuffer`'s allocation, so
//!   two layers that merely share a *geometry* never read each other's mirror
//!   (the executor itself is shared per geometry, the mirrors are not);
//! * `epoch` is a graph-wide counter the training step advances only after a
//!   successful optimizer submission, so a lazy AdamW step, a skipped step or
//!   an accumulated micro-batch can never expose a stale mirror;
//! * a checkpoint, adapter or token-table load bumps the same counter, so
//!   reloaded weights are re-converted before their first dispatch.
//!
//! The counters in [`linear_wmma_conversion_stats`] make the effect auditable:
//! a cache that never hits is visible as `reuses == 0`, and one that serves a
//! stale mirror shows up as a numerical mismatch in the graph's own gates.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};

use crate::vulkan::{self, GpuBuffer, VulkanDevice};
use crate::wmma::{self, GemmLayout, WmmaDispatchCounts, WmmaGemm, WmmaGemmArm, WmmaMode};

static FORWARD_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static WEIGHT_GRAD_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static INPUT_GRAD_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static SHAPE_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static POLICY_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static WEIGHT_CONVERSIONS: AtomicU64 = AtomicU64::new(0);
static WEIGHT_CONVERSION_BYTES: AtomicU64 = AtomicU64::new(0);
static WEIGHT_MIRROR_REUSES: AtomicU64 = AtomicU64::new(0);
static MIXED_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static FULL_FP16_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static FULL_FP16_REDUCTION_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Full FP16 graph policy bounds the number of dot2 rounding stages. Raw
/// primitive qualification remains unrestricted and preserves its failing
/// long-reduction evidence. Longer graph reductions retain their FP32 kernel.
pub use crate::precision_policy::FULL_FP16_MAX_REDUCTION;

pub fn full_fp16_reduction_fallbacks() -> u64 {
    FULL_FP16_REDUCTION_FALLBACKS.load(Ordering::Relaxed)
}

/// Distinguish hardware accumulator types, not merely a requested mode label.
pub fn linear_wmma_precision_counts() -> [u64; 2] {
    [MIXED_DISPATCHES.load(Ordering::Relaxed), FULL_FP16_DISPATCHES.load(Ordering::Relaxed)]
}

/// Projections kept on their FP32 kernel by an explicit numerical policy even
/// though their shape is WMMA-eligible.
///
/// These dispatches are not cooperative and must never be added to
/// [`linear_wmma_dispatch_counts`]. They are counted here so a graph that keeps
/// a sensitive tower in FP32 can document exactly how many projections took
/// that route.
pub fn linear_wmma_policy_fallbacks() -> u64 {
    POLICY_FALLBACKS.load(Ordering::Relaxed)
}

pub(crate) fn note_policy_fallback() {
    POLICY_FALLBACKS.fetch_add(1, Ordering::Relaxed);
}

/// Mirror slots one executor may hold before the stalest is evicted.
///
/// A geometry is shared by every projection of that shape, and the widest real
/// graph uses a handful, so this bounds memory without thrashing.
const MAX_WEIGHT_MIRRORS: usize = 32;

/// Process-wide FP32->FP16 mirror traffic recorded by the graph.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LinearWmmaConversionStats {
    /// Weight mirrors rewritten from their FP32 master.
    pub weight_conversions: u64,
    /// FP32 bytes read by those conversions.
    pub weight_bytes: u64,
    /// Weight dispatches that reused a mirror without converting.
    pub mirror_reuses: u64,
    /// FP32 dispatches retained by an explicit numerical policy.
    pub policy_fallbacks: u64,
}

pub fn linear_wmma_conversion_stats() -> LinearWmmaConversionStats {
    LinearWmmaConversionStats {
        weight_conversions: WEIGHT_CONVERSIONS.load(Ordering::Relaxed),
        weight_bytes: WEIGHT_CONVERSION_BYTES.load(Ordering::Relaxed),
        mirror_reuses: WEIGHT_MIRROR_REUSES.load(Ordering::Relaxed),
        policy_fallbacks: POLICY_FALLBACKS.load(Ordering::Relaxed),
    }
}

/// Process-wide cooperative dispatches recorded by the Transformer graph.
///
/// A zero here with a requested mode means the graph kept its own kernels; the
/// trainer reports that as "requested, not dispatched" rather than as an
/// accelerated run.
pub fn linear_wmma_dispatch_counts() -> WmmaDispatchCounts {
    WmmaDispatchCounts {
        cooperative: FORWARD_DISPATCHES.load(Ordering::Relaxed)
            + WEIGHT_GRAD_DISPATCHES.load(Ordering::Relaxed)
            + INPUT_GRAD_DISPATCHES.load(Ordering::Relaxed),
        plain_f16: 0,
        shape_fallbacks: SHAPE_FALLBACKS.load(Ordering::Relaxed),
    }
}

/// Per-linear receipt for the trainer's end-of-run report.
pub fn linear_wmma_phase_counts() -> [usize; 3] {
    [
        FORWARD_DISPATCHES.load(Ordering::Relaxed) as usize,
        WEIGHT_GRAD_DISPATCHES.load(Ordering::Relaxed) as usize,
        INPUT_GRAD_DISPATCHES.load(Ordering::Relaxed) as usize,
    ]
}

/// One FP16 copy of one FP32 master weight.
struct WeightMirror {
    allocation: usize,
    len: usize,
    /// `u64::MAX` marks a mirror that has never been filled.
    epoch: u64,
    buffer: GpuBuffer,
}

/// One resident projection's WMMA execution state.
pub struct LinearWmma {
    gemm: WmmaGemm,
    device: VulkanDevice,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    input_f16: GpuBuffer,
    grad_output_f16: GpuBuffer,
    epoch: Arc<AtomicU64>,
    mirrors: Mutex<Vec<WeightMirror>>,
}

impl LinearWmma {
    /// Build the executor for one projection.
    ///
    /// The activation mirrors are allocated once at the projection's
    /// construction geometry, so the training loop never allocates on the hot
    /// path. `Off` callers must not construct this at all.
    pub fn new(
        device: &VulkanDevice,
        mode: WmmaMode,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Self> {
        Self::new_with_epoch(
            device,
            mode,
            rows,
            input_dim,
            output_dim,
            Arc::new(AtomicU64::new(0)),
        )
    }

    /// Build the executor against a graph-owned weight epoch.
    ///
    /// The caller must advance `epoch` after every write to a master weight
    /// that a dispatch could observe.
    pub fn new_with_epoch(
        device: &VulkanDevice,
        mode: WmmaMode,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        epoch: Arc<AtomicU64>,
    ) -> Result<Self> {
        match mode {
            WmmaMode::Off => anyhow::bail!("a WMMA linear executor requires an active WMMA mode"),
            WmmaMode::Mixed | WmmaMode::FullFp16 => {}
        }
        if rows == 0 || input_dim == 0 || output_dim == 0 {
            anyhow::bail!("WMMA linear executor requires positive geometry");
        }
        let gemm = WmmaGemm::new(device, mode).context("create WMMA GEMM for a dense projection")?;
        Ok(Self {
            gemm,
            device: device.clone(),
            rows,
            input_dim,
            output_dim,
            input_f16: WmmaGemm::new_half_buffer(device, rows * input_dim)?,
            grad_output_f16: WmmaGemm::new_half_buffer(device, rows * output_dim)?,
            epoch,
            mirrors: Mutex::new(Vec::new()),
        })
    }

    pub fn mode(&self) -> WmmaMode {
        self.gemm.mode()
    }

    fn arm(&self) -> WmmaGemmArm {
        match self.mode() {
            WmmaMode::Mixed => WmmaGemmArm::MixedFp32Accumulate,
            WmmaMode::FullFp16 => WmmaGemmArm::FullFp16Accumulate,
            WmmaMode::Off => unreachable!("inactive executor cannot be constructed"),
        }
    }

    fn count_precision(&self) {
        match self.mode() {
            WmmaMode::Mixed => { MIXED_DISPATCHES.fetch_add(1, Ordering::Relaxed); }
            WmmaMode::FullFp16 => { FULL_FP16_DISPATCHES.fetch_add(1, Ordering::Relaxed); }
            WmmaMode::Off => unreachable!(),
        }
    }

    fn admits(&self, rows: usize, reduction: usize) -> bool {
        if self.gemm.mode() == WmmaMode::FullFp16 && reduction > FULL_FP16_MAX_REDUCTION {
            FULL_FP16_REDUCTION_FALLBACKS.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        rows == self.rows
            && wmma::wmma_shape_eligible(rows, self.output_dim, self.input_dim)
            && self
                .gemm
                .select_arm(rows, self.output_dim, self.input_dim)
                .is_some_and(WmmaGemmArm::uses_cooperative_matrix)
    }

    fn note_fallback(&self) -> bool {
        SHAPE_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        self.gemm.note_shape_fallback();
        false
    }

    /// Record the weight mirror for `weight` if this epoch needs one, then run
    /// `dispatch` against it while the mirror slot is held.
    ///
    /// The slot is held across `dispatch` because the mirror is only guaranteed
    /// live while the recording is sequential, which is the graph's contract.
    fn with_weight_mirror<R>(
        &self,
        commands: &mut vulkan::ComputeBatch,
        weight: &GpuBuffer,
        len: usize,
        dispatch: impl FnOnce(&mut vulkan::ComputeBatch, &GpuBuffer) -> Result<R>,
    ) -> Result<R> {
        let epoch = self.epoch.load(Ordering::Relaxed);
        let allocation = weight.allocation_id();
        let mut mirrors = self
            .mirrors
            .lock()
            .map_err(|_| anyhow::anyhow!("WMMA weight-mirror cache is poisoned"))?;
        let index = match mirrors
            .iter()
            .position(|mirror| mirror.allocation == allocation && mirror.len == len)
        {
            Some(index) => index,
            None => {
                if mirrors.len() >= MAX_WEIGHT_MIRRORS {
                    // Evict the mirror that was refreshed longest ago: it is the
                    // least likely to be reused inside the current step.
                    let stalest = mirrors
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, mirror)| mirror.epoch)
                        .map(|(index, _)| index)
                        .unwrap_or(0);
                    mirrors.remove(stalest);
                }
                mirrors.push(WeightMirror {
                    allocation,
                    len,
                    epoch: u64::MAX,
                    buffer: WmmaGemm::new_half_buffer(&self.device, len)?,
                });
                mirrors.len() - 1
            }
        };
        if mirrors[index].epoch != epoch {
            self.gemm
                .record_conversion(commands, weight, &mirrors[index].buffer, len)?;
            mirrors[index].epoch = epoch;
            WEIGHT_CONVERSIONS.fetch_add(1, Ordering::Relaxed);
            WEIGHT_CONVERSION_BYTES.fetch_add(len as u64 * 4, Ordering::Relaxed);
        } else {
            WEIGHT_MIRROR_REUSES.fetch_add(1, Ordering::Relaxed);
        }
        dispatch(commands, &mirrors[index].buffer)
    }

    /// Record `output = input · weightᵀ` in mixed WMMA.
    ///
    /// Returns `false` when the shape is outside the cooperative contract, in
    /// which case the caller keeps its own kernel and the fallback is counted.
    pub fn record_forward(
        &self,
        commands: &mut vulkan::ComputeBatch,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        output: &GpuBuffer,
        rows: usize,
    ) -> Result<bool> {
        if !self.admits(rows, self.input_dim) {
            return Ok(self.note_fallback());
        }
        self.gemm
            .record_conversion(commands, input, &self.input_f16, rows * self.input_dim)?;
        self.with_weight_mirror(
            commands,
            weight,
            self.output_dim * self.input_dim,
            |commands, mirror| {
                self.gemm.record_gemm(
                    commands,
                    self.arm(),
                    &self.input_f16,
                    mirror,
                    output,
                    rows,
                    self.output_dim,
                    self.input_dim,
                    GemmLayout {
                        transposed_a: false,
                        transposed_b: true,
                    },
                )
            },
        )?;
        FORWARD_DISPATCHES.fetch_add(1, Ordering::Relaxed);
        self.count_precision();
        Ok(true)
    }

    /// Record `grad_weight = grad_outputᵀ · input` in mixed WMMA.
    ///
    /// The output is `[output_dim, input_dim]`, matching the FP32 master's
    /// layout, so gradient accumulation and AdamW are unchanged.
    pub fn record_weight_grad(
        &self,
        commands: &mut vulkan::ComputeBatch,
        grad_output: &GpuBuffer,
        input: &GpuBuffer,
        grad_weight: &GpuBuffer,
        rows: usize,
    ) -> Result<bool> {
        if !self.admits(rows, rows) {
            return Ok(self.note_fallback());
        }
        self.gemm.record_conversion(
            commands,
            grad_output,
            &self.grad_output_f16,
            rows * self.output_dim,
        )?;
        self.gemm
            .record_conversion(commands, input, &self.input_f16, rows * self.input_dim)?;
        self.gemm.record_gemm(
            commands,
            self.arm(),
            &self.grad_output_f16,
            &self.input_f16,
            grad_weight,
            self.output_dim,
            self.input_dim,
            rows,
            GemmLayout {
                transposed_a: true,
                transposed_b: false,
            },
        )?;
        WEIGHT_GRAD_DISPATCHES.fetch_add(1, Ordering::Relaxed);
        self.count_precision();
        Ok(true)
    }

    /// Record `grad_input = grad_output · weight` in mixed WMMA.
    pub fn record_input_grad(
        &self,
        commands: &mut vulkan::ComputeBatch,
        grad_output: &GpuBuffer,
        weight: &GpuBuffer,
        grad_input: &GpuBuffer,
        rows: usize,
    ) -> Result<bool> {
        if !self.admits(rows, self.output_dim) {
            return Ok(self.note_fallback());
        }
        self.gemm.record_conversion(
            commands,
            grad_output,
            &self.grad_output_f16,
            rows * self.output_dim,
        )?;
        self.with_weight_mirror(
            commands,
            weight,
            self.output_dim * self.input_dim,
            |commands, mirror| {
                self.gemm.record_gemm(
                    commands,
                    self.arm(),
                    &self.grad_output_f16,
                    mirror,
                    grad_input,
                    rows,
                    self.input_dim,
                    self.output_dim,
                    GemmLayout {
                        transposed_a: false,
                        transposed_b: false,
                    },
                )
            },
        )?;
        INPUT_GRAD_DISPATCHES.fetch_add(1, Ordering::Relaxed);
        self.count_precision();
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wmma::{MIXED_VS_FP32_MAX_ABS, MIXED_VS_FP32_RMS};

    const ROWS: usize = 32;
    const K: usize = 1024;
    const N: usize = 1024;

    fn fill(len: usize, seed: u64) -> Vec<f32> {
        // Deterministic values in [-0.5, 0.5): the drift gate is about FP16
        // operand narrowing, not about matching a particular distribution.
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect()
    }

    fn matrix_reference(input: &[f32], weight: &[f32]) -> Vec<f32> {
        let mut reference = vec![0.0f32; ROWS * N];
        for m in 0..ROWS {
            for n in 0..N {
                let mut acc = 0.0f32;
                for k in 0..K {
                    acc += input[m * K + k] * weight[n * K + k];
                }
                reference[m * N + n] = acc;
            }
        }
        reference
    }

    /// The three GEMMs the Transformer records per training step, exercised
    /// through the exact graph entry points, against host FP32 references.
    ///
    /// This is the wiring test for the transposed operand layouts: a wrong
    /// layout still produces finite numbers, so only the reference comparison
    /// separates a working dispatch from a silently transposed one.
    #[test]
    fn mixed_linear_graph_path_matches_fp32_in_three_phases() {
        let Ok(device) = VulkanDevice::new_for_wmma(WmmaMode::Mixed) else {
            eprintln!("skipping: no cooperative-matrix capable device");
            return;
        };
        let executor = LinearWmma::new(&device, WmmaMode::Mixed, ROWS, K, N)
            .expect("build mixed linear executor");
        let input = fill(ROWS * K, 1);
        let weight = fill(N * K, 2);
        let grad_out = fill(ROWS * N, 3);

        let input_buf = GpuBuffer::from_f32(&device, &input).unwrap();
        let weight_buf = GpuBuffer::from_f32(&device, &weight).unwrap();
        let grad_out_buf = GpuBuffer::from_f32(&device, &grad_out).unwrap();

        let output = GpuBuffer::zeros_f32(&device, ROWS * N).unwrap();
        let host = GpuBuffer::transient_host_f32(&device, ROWS * N).unwrap();
        let mut batch = vulkan::ComputeBatch::new(&device).unwrap();
        assert!(executor
            .record_forward(&mut batch, &input_buf, &weight_buf, &output, ROWS)
            .unwrap());
        batch.readback_f32(&output, &host, ROWS * N).unwrap();
        batch.submit().unwrap();
        let forward = host.read_f32(ROWS * N).unwrap();

        let reference = matrix_reference(&input, &weight);
        let forward_drift = crate::wmma::drift(&reference, &forward).unwrap();
        assert!(forward_drift.all_finite(), "forward produced non-finite values");
        assert!(
            forward_drift.max_abs <= MIXED_VS_FP32_MAX_ABS,
            "forward max_abs {} exceeds {MIXED_VS_FP32_MAX_ABS}",
            forward_drift.max_abs
        );
        assert!(
            forward_drift.rms <= MIXED_VS_FP32_RMS,
            "forward RMS {} exceeds {MIXED_VS_FP32_RMS}",
            forward_drift.rms
        );

        // Weight gradient: [N, K] = grad_output^T * input.
        let grad_weight = GpuBuffer::zeros_f32(&device, N * K).unwrap();
        let host_w = GpuBuffer::transient_host_f32(&device, N * K).unwrap();
        let mut batch = vulkan::ComputeBatch::new(&device).unwrap();
        assert!(executor
            .record_weight_grad(&mut batch, &grad_out_buf, &input_buf, &grad_weight, ROWS)
            .unwrap());
        batch.readback_f32(&grad_weight, &host_w, N * K).unwrap();
        batch.submit().unwrap();
        let weight_grad = host_w.read_f32(N * K).unwrap();
        let mut reference = vec![0.0f32; N * K];
        for r in 0..ROWS {
            for n in 0..N {
                let g = grad_out[r * N + n];
                for k in 0..K {
                    reference[n * K + k] += g * input[r * K + k];
                }
            }
        }
        let weight_drift = crate::wmma::drift(&reference, &weight_grad).unwrap();
        assert!(weight_drift.all_finite(), "weight gradient produced non-finite values");
        assert!(
            weight_drift.max_abs <= MIXED_VS_FP32_MAX_ABS,
            "weight-grad max_abs {} exceeds {MIXED_VS_FP32_MAX_ABS}",
            weight_drift.max_abs
        );

        // Input gradient: [ROWS, K] = grad_output * weight.
        let grad_input = GpuBuffer::zeros_f32(&device, ROWS * K).unwrap();
        let host_i = GpuBuffer::transient_host_f32(&device, ROWS * K).unwrap();
        let mut batch = vulkan::ComputeBatch::new(&device).unwrap();
        assert!(executor
            .record_input_grad(&mut batch, &grad_out_buf, &weight_buf, &grad_input, ROWS)
            .unwrap());
        batch.readback_f32(&grad_input, &host_i, ROWS * K).unwrap();
        batch.submit().unwrap();
        let input_grad = host_i.read_f32(ROWS * K).unwrap();
        let mut reference = vec![0.0f32; ROWS * K];
        for m in 0..ROWS {
            for k in 0..K {
                let mut acc = 0.0f32;
                for n in 0..N {
                    acc += grad_out[m * N + n] * weight[n * K + k];
                }
                reference[m * K + k] = acc;
            }
        }
        let input_drift = crate::wmma::drift(&reference, &input_grad).unwrap();
        assert!(input_drift.all_finite(), "input gradient produced non-finite values");
        assert!(
            input_drift.max_abs <= MIXED_VS_FP32_MAX_ABS,
            "input-grad max_abs {} exceeds {MIXED_VS_FP32_MAX_ABS}",
            input_drift.max_abs
        );

        // Every phase must have recorded a cooperative dispatch; a silently
        // non-accelerated run is the failure this guards against.
        let phases = linear_wmma_phase_counts();
        assert!(phases[0] >= 1 && phases[1] >= 1 && phases[2] >= 1, "phases {phases:?}");
        assert!(linear_wmma_dispatch_counts().cooperative >= 3);
    }

    /// Forward and input-gradient both read the master weight, so within one
    /// epoch the second dispatch must reuse the mirror instead of converting
    /// again - and after an optimizer step it must convert, because the master
    /// bytes changed.
    #[test]
    fn weight_mirror_is_reused_within_an_epoch_and_refreshed_after_a_step() {
        let Ok(device) = VulkanDevice::new_for_wmma(WmmaMode::Mixed) else {
            eprintln!("skipping: no cooperative-matrix capable device");
            return;
        };
        let epoch = Arc::new(AtomicU64::new(7));
        let executor =
            LinearWmma::new_with_epoch(&device, WmmaMode::Mixed, ROWS, K, N, Arc::clone(&epoch))
                .expect("build mixed linear executor");
        let input = fill(ROWS * K, 11);
        let weight = fill(N * K, 12);
        let grad_out = fill(ROWS * N, 13);
        let input_buf = GpuBuffer::from_f32(&device, &input).unwrap();
        let weight_buf = GpuBuffer::from_f32(&device, &weight).unwrap();
        let grad_out_buf = GpuBuffer::from_f32(&device, &grad_out).unwrap();
        let output = GpuBuffer::zeros_f32(&device, ROWS * N).unwrap();
        let grad_input = GpuBuffer::zeros_f32(&device, ROWS * K).unwrap();
        let host = GpuBuffer::transient_host_f32(&device, ROWS * N).unwrap();

        let before = linear_wmma_conversion_stats();
        let mut batch = vulkan::ComputeBatch::new(&device).unwrap();
        assert!(executor
            .record_forward(&mut batch, &input_buf, &weight_buf, &output, ROWS)
            .unwrap());
        assert!(executor
            .record_input_grad(&mut batch, &grad_out_buf, &weight_buf, &grad_input, ROWS)
            .unwrap());
        batch.readback_f32(&output, &host, ROWS * N).unwrap();
        batch.submit().unwrap();
        let after_step = linear_wmma_conversion_stats();
        assert_eq!(
            after_step.weight_conversions - before.weight_conversions,
            1,
            "one master weight must be converted once per epoch"
        );
        assert!(
            after_step.mirror_reuses > before.mirror_reuses,
            "the input gradient must reuse the forward's mirror"
        );

        // The epoch advances after an optimizer submission; the same master
        // buffer now needs a fresh mirror even though its identity is unchanged.
        epoch.fetch_add(1, Ordering::Relaxed);
        let mut batch = vulkan::ComputeBatch::new(&device).unwrap();
        assert!(executor
            .record_forward(&mut batch, &input_buf, &weight_buf, &output, ROWS)
            .unwrap());
        batch.submit().unwrap();
        let after_update = linear_wmma_conversion_stats();
        assert_eq!(
            after_update.weight_conversions - after_step.weight_conversions,
            1,
            "an epoch change must invalidate the mirror"
        );
        assert_eq!(
            after_update.weight_bytes - before.weight_bytes,
            (N * K * 4) as u64 * 2,
            "each conversion must account for the FP32 master bytes read"
        );

        // A different parameter with the same geometry must never read this
        // mirror: the key is the allocation, not the shape.
        let other = fill(N * K, 14);
        let other_buf = GpuBuffer::from_f32(&device, &other).unwrap();
        let mut batch = vulkan::ComputeBatch::new(&device).unwrap();
        assert!(executor
            .record_forward(&mut batch, &input_buf, &other_buf, &output, ROWS)
            .unwrap());
        batch.readback_f32(&output, &host, ROWS * N).unwrap();
        batch.submit().unwrap();
        let mixed = host.read_f32(ROWS * N).unwrap();
        let expected = matrix_reference(&input, &other);
        let drift = crate::wmma::drift(&expected, &mixed).unwrap();
        assert!(
            drift.max_abs <= MIXED_VS_FP32_MAX_ABS,
            "a sibling parameter read a stale mirror: max_abs {}",
            drift.max_abs
        );
    }

    #[test]
    fn ineligible_shapes_keep_their_own_kernel_and_are_counted() {
        let Ok(device) = VulkanDevice::new_for_wmma(WmmaMode::Mixed) else {
            eprintln!("skipping: no cooperative-matrix capable device");
            return;
        };
        let executor = LinearWmma::new(&device, WmmaMode::Mixed, ROWS, K, N).unwrap();
        let input = GpuBuffer::zeros_f32(&device, ROWS * K).unwrap();
        let weight = GpuBuffer::zeros_f32(&device, N * K).unwrap();
        let output = GpuBuffer::zeros_f32(&device, ROWS * N).unwrap();
        let before = linear_wmma_dispatch_counts().shape_fallbacks;
        let mut batch = vulkan::ComputeBatch::new(&device).unwrap();
        // A single decode row cannot fill a cooperative tile.
        assert!(!executor
            .record_forward(&mut batch, &input, &weight, &output, 1)
            .unwrap());
        assert!(linear_wmma_dispatch_counts().shape_fallbacks > before);
    }

    #[test]
    fn full_fp16_graph_widens_accumulators_in_all_three_phases() {
        const ROWS:usize=64;
        const K:usize=64;
        const N:usize=64;
        let Ok(device) = VulkanDevice::new_for_wmma(WmmaMode::FullFp16) else {
            eprintln!("skipping: no cooperative-matrix capable device");
            return;
        };
        let executor = LinearWmma::new(&device, WmmaMode::FullFp16, ROWS, K, N).unwrap();
        let input = GpuBuffer::from_f32(&device, &vec![1.0/16.0; ROWS*K]).unwrap();
        let weight = GpuBuffer::from_f32(&device, &vec![1.0/64.0; N*K]).unwrap();
        let grad = GpuBuffer::from_f32(&device, &vec![1.0/32.0; ROWS*N]).unwrap();
        let output = GpuBuffer::zeros_f32(&device, ROWS*N).unwrap();
        let dw = GpuBuffer::zeros_f32(&device, N*K).unwrap();
        let dx = GpuBuffer::zeros_f32(&device, ROWS*K).unwrap();
        let before = linear_wmma_precision_counts();
        let mut commands = vulkan::ComputeBatch::new(&device).unwrap();
        assert!(executor.record_forward(&mut commands,&input,&weight,&output,ROWS).unwrap());
        assert!(executor.record_weight_grad(&mut commands,&grad,&input,&dw,ROWS).unwrap());
        assert!(executor.record_input_grad(&mut commands,&grad,&weight,&dx,ROWS).unwrap());
        commands.submit().unwrap();
        assert!(output.read_f32(ROWS*N).unwrap().iter().all(|&x|x==1.0/16.0));
        assert!(dw.read_f32(N*K).unwrap().iter().all(|&x|x==1.0/8.0));
        assert!(dx.read_f32(ROWS*K).unwrap().iter().all(|&x|x==1.0/32.0));
        let after = linear_wmma_precision_counts();
        assert_eq!(after[0],before[0],"full mode must not dispatch mixed accumulation");
        assert_eq!(after[1]-before[1],3);
        let long=LinearWmma::new(&device,WmmaMode::FullFp16,ROWS,128,N).unwrap();
        let x=GpuBuffer::zeros_f32(&device,ROWS*128).unwrap();
        let w=GpuBuffer::zeros_f32(&device,N*128).unwrap();
        let previous=full_fp16_reduction_fallbacks();
        let mut commands=vulkan::ComputeBatch::new(&device).unwrap();
        assert!(!long.record_forward(&mut commands,&x,&w,&output,ROWS).unwrap(),
            "long reductions must retain the caller's FP32 kernel");
        assert_eq!(full_fp16_reduction_fallbacks(),previous+1);
        assert_eq!(linear_wmma_precision_counts(),after,"range fallback must never dispatch Mixed");
    }
}
