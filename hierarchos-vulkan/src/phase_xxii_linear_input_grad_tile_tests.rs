//! Phase XXII — FP32 `linear_input_grad` output-tiling experiment (test builds
//! only).
//!
//! The candidate is a 32 x 32 output tile: four output elements per thread, with
//! each element reducing the vocabulary dimension in strictly ascending order
//! through the same single `fma` the production kernel uses. It is promoted for
//! 32-aligned shapes and keeps the original 16 x 16 kernel as its fallback. The
//! experiment copy of the artifact remains under
//! `validation/phase_xxii_linear_input_grad_tile_20261128/`; the production copy
//! is `shaders/linear_input_grad_output_tile32.spv`.
//!
//! The parity test dispatches the shipped kernel and the candidate on identical
//! operands and compares every output word bitwise. It covers the measured
//! 128 x 1024 x 1024 projection together with a 1024 -> 256 projection,
//! non-multiple-of-32 boundaries, a single row/column and a degenerate
//! 1 x 1 x 1 shape. Every element must be written, finite and exactly equal.

use super::*;
use crate::vulkan::{self, GpuBuffer, VulkanDevice};

/// The production FP32 input-gradient artifact.
const PHASE_XXII_CONTROL: &[u8] = include_bytes!("../shaders/linear_input_grad.spv");
/// The promoted candidate.
const PHASE_XXII_CANDIDATE: &[u8] = include_bytes!("../shaders/linear_input_grad_output_tile32.spv");
const PHASE_XXII_SENTINEL: f32 = -98_765.432_1_f32;

fn phase_xxii_operands(salt: u32, count: usize) -> Vec<f32> {
    let mut state = 0x9e37_79b9_u32 ^ salt.wrapping_mul(0x85eb_ca6b);
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / 16_777_216.0;
            unit * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn phase_xxii_linear_input_grad_output_tile_parity() -> Result<()> {
    let device = VulkanDevice::new()?;
    let control = vulkan::ComputeKernel::new(&device, PHASE_XXII_CONTROL, 3, 12)?;
    let candidate = vulkan::ComputeKernel::new(&device, PHASE_XXII_CANDIDATE, 3, 12)?;

    let shapes = [
        ("measured-128x1024x1024", 128usize, 1024usize, 1024usize),
        ("head-1024x1024x256", 128, 1024, 256),
        ("square-64x128x128", 64, 128, 128),
        ("boundary-33x37x45", 33, 37, 45),
        ("narrow-5x96x160", 5, 96, 160),
        ("degenerate-1x1x1", 1, 1, 1),
    ];

    for (label, rows, input_dim, output_dim) in shapes {
        let grad_output = GpuBuffer::from_f32(
            &device,
            &phase_xxii_operands(0x21, rows * output_dim),
        )?;
        let weight =
            GpuBuffer::from_f32(&device, &phase_xxii_operands(0x22, output_dim * input_dim))?;
        let control_out =
            GpuBuffer::from_f32(&device, &vec![PHASE_XXII_SENTINEL; rows * input_dim])?;
        let candidate_out =
            GpuBuffer::from_f32(&device, &vec![PHASE_XXII_SENTINEL; rows * input_dim])?;
        let push = LinearPush {
            rows: rows as u32,
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
        };

        let mut commands = vulkan::ComputeBatch::new(&device)?;
        control.record_dispatch(
            &mut commands,
            &[&grad_output, &weight, &control_out],
            bytemuck::bytes_of(&push),
            [
                div_ceil_u32(input_dim, 16),
                div_ceil_u32(rows, 16),
                1,
            ],
        )?;
        candidate.record_dispatch(
            &mut commands,
            &[&grad_output, &weight, &candidate_out],
            bytemuck::bytes_of(&push),
            [
                div_ceil_u32(input_dim, 32),
                div_ceil_u32(rows, 32),
                1,
            ],
        )?;
        commands.submit()?;

        let len = rows * input_dim;
        let values = GpuBuffer::read_many_f32(&[(&control_out, len), (&candidate_out, len)])?;
        let expected = &values[0];
        let actual = &values[1];
        let mut bitwise = 0usize;
        let mut first = None;
        let mut sentinel = 0usize;
        let mut non_finite = 0usize;
        for (index, (&expected_value, &actual_value)) in expected.iter().zip(actual).enumerate() {
            if !expected_value.is_finite() || !actual_value.is_finite() {
                non_finite += 1;
            }
            if actual_value.to_bits() == PHASE_XXII_SENTINEL.to_bits() {
                sentinel += 1;
            }
            if expected_value.to_bits() == actual_value.to_bits() {
                bitwise += 1;
            } else if first.is_none() {
                first = Some((index, expected_value.to_bits(), actual_value.to_bits()));
            }
        }
        eprintln!(
            "[phase-xxii-parity] case={label} rows={rows} input_dim={input_dim} \
             output_dim={output_dim} elements={len} bitwise={bitwise} sentinel={sentinel} \
             non_finite={non_finite} first_divergence={first:?} verdict={}",
            if bitwise == len && sentinel == 0 && non_finite == 0 {
                "PASS"
            } else {
                "FAIL"
            }
        );
        assert_eq!(bitwise, len, "{label}: output words are not bitwise equal");
        assert_eq!(sentinel, 0, "{label}: the candidate left outputs unwritten");
        assert_eq!(non_finite, 0, "{label}: non-finite output");
    }
    Ok(())
}

#[test]
fn phase_xxii_linear_input_grad_output_tile_selector() -> Result<()> {
    // The tested shape is eligible; a non-multiple-of-32 shape is not.
    assert!(linear_input_grad_output_tile_eligible(128, 1024, 1024));
    assert!(linear_input_grad_output_tile_eligible(128, 1024, 256));
    assert!(!linear_input_grad_output_tile_eligible(33, 37, 45));
    assert!(!linear_input_grad_output_tile_eligible(0, 1024, 1024));

    // With no opt-out requested, the promoted kernel is the default on the
    // qualified partition; the original kernel remains the fallback elsewhere.
    if std::env::var_os(HIERARCHOS_LINEAR_INPUT_GRAD_OUTPUT_TILE_ENV).is_none() {
        assert!(linear_input_grad_output_tile_selected(128, 1024, 1024));
        assert!(!linear_input_grad_output_tile_selected(33, 37, 45));
    }
    Ok(())
}
