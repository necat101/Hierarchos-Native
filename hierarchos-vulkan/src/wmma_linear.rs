//! Dense-linear WMMA execution for the resident Transformer graph.
//!
//! This is the graph-facing counterpart of the primitive [`crate::wmma`] module:
//! it owns the FP16 execution mirrors a single `VulkanLinear` needs and records
//! the three GEMMs of a training step (forward, weight gradient, input
//! gradient) through the same [`WmmaGemm`] the qualification harness exercises.
//!
//! Only the mixed arm (FP16 inputs, FP32 accumulation) is admitted here. The
//! full-FP16 arm accumulates in FP16, so its result must be widened by a host
//! read-back or by a dedicated store kernel; neither exists on the in-graph
//! path, and fabricating one would silently change what the graph computes.
//! [`LinearWmma::new`] therefore refuses the full arm, and callers keep the
//! qualified FP32 kernels for that mode.

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};

use crate::vulkan::{self, GpuBuffer, VulkanDevice};
use crate::wmma::{self, GemmLayout, WmmaDispatchCounts, WmmaGemm, WmmaGemmArm, WmmaMode};

static FORWARD_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static WEIGHT_GRAD_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static INPUT_GRAD_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static SHAPE_FALLBACKS: AtomicU64 = AtomicU64::new(0);

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

/// One resident projection's WMMA execution state.
pub struct LinearWmma {
    gemm: WmmaGemm,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    input_f16: GpuBuffer,
    weight_f16: GpuBuffer,
    grad_output_f16: GpuBuffer,
}

impl LinearWmma {
    /// Build the executor for one projection.
    ///
    /// The FP16 mirrors are allocated once at the projection's construction
    /// geometry, so the training loop never allocates on the hot path. `Off`
    /// callers must not construct this at all.
    pub fn new(
        device: &VulkanDevice,
        mode: WmmaMode,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Self> {
        match mode {
            WmmaMode::Off => anyhow::bail!("a WMMA linear executor requires an active WMMA mode"),
            WmmaMode::FullFp16 => anyhow::bail!(
                "full-FP16 accumulation has no in-graph FP32 output path; \
                 Transformer graphs admit only mixed WMMA"
            ),
            WmmaMode::Mixed => {}
        }
        if rows == 0 || input_dim == 0 || output_dim == 0 {
            anyhow::bail!("WMMA linear executor requires positive geometry");
        }
        let gemm = WmmaGemm::new(device, mode).context("create WMMA GEMM for a dense projection")?;
        Ok(Self {
            gemm,
            rows,
            input_dim,
            output_dim,
            input_f16: WmmaGemm::new_half_buffer(device, rows * input_dim)?,
            weight_f16: WmmaGemm::new_half_buffer(device, output_dim * input_dim)?,
            grad_output_f16: WmmaGemm::new_half_buffer(device, rows * output_dim)?,
        })
    }

    pub fn mode(&self) -> WmmaMode {
        self.gemm.mode()
    }

    fn admits(&self, rows: usize) -> bool {
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
        if !self.admits(rows) {
            return Ok(self.note_fallback());
        }
        self.gemm
            .record_conversion(commands, input, &self.input_f16, rows * self.input_dim)?;
        self.gemm.record_conversion(
            commands,
            weight,
            &self.weight_f16,
            self.output_dim * self.input_dim,
        )?;
        self.gemm.record_gemm(
            commands,
            WmmaGemmArm::MixedFp32Accumulate,
            &self.input_f16,
            &self.weight_f16,
            output,
            rows,
            self.output_dim,
            self.input_dim,
            GemmLayout {
                transposed_a: false,
                transposed_b: true,
            },
        )?;
        FORWARD_DISPATCHES.fetch_add(1, Ordering::Relaxed);
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
        if !self.admits(rows) {
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
            WmmaGemmArm::MixedFp32Accumulate,
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
        if !self.admits(rows) {
            return Ok(self.note_fallback());
        }
        self.gemm.record_conversion(
            commands,
            grad_output,
            &self.grad_output_f16,
            rows * self.output_dim,
        )?;
        self.gemm.record_conversion(
            commands,
            weight,
            &self.weight_f16,
            self.output_dim * self.input_dim,
        )?;
        self.gemm.record_gemm(
            commands,
            WmmaGemmArm::MixedFp32Accumulate,
            &self.grad_output_f16,
            &self.weight_f16,
            grad_input,
            rows,
            self.input_dim,
            self.output_dim,
            GemmLayout {
                transposed_a: false,
                transposed_b: false,
            },
        )?;
        INPUT_GRAD_DISPATCHES.fetch_add(1, Ordering::Relaxed);
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
    fn full_fp16_is_refused_for_the_graph() {
        let Ok(device) = VulkanDevice::new_for_wmma(WmmaMode::Mixed) else {
            eprintln!("skipping: no cooperative-matrix capable device");
            return;
        };
        let error = match LinearWmma::new(&device, WmmaMode::FullFp16, ROWS, K, N) {
            Ok(_) => panic!("full-FP16 must be refused for the graph"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("no in-graph FP32 output path"), "{error}");
    }
}
