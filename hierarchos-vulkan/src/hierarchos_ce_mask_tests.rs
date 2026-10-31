//! Native CE regression fixtures. These deliberately exercise loss adjoints in
//! isolation: a masked loss does not suppress other recurrent/auxiliary edges.
use crate::{vulkan, GpuBuffer, LinearPush, LmWeightGradPush, VulkanDevice};
use anyhow::{Context, Result};

const WIDTH: usize = 4;
const VOCAB: usize = 17;

fn kernel(
    device: &VulkanDevice,
    stem: &str,
    bindings: usize,
    push: u32,
) -> Result<vulkan::ComputeKernel> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("shaders")
        .join(format!("{stem}.spv"));
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    vulkan::ComputeKernel::new(device, &bytes, bindings, push)
}

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() <= 2e-6,
            "{label}[{index}]: actual={a}, expected={b}"
        );
    }
}

struct Fixture {
    x: GpuBuffer,
    gamma: GpuBuffer,
    beta: GpuBuffer,
    mean: GpuBuffer,
    rstd: GpuBuffer,
    weights: GpuBuffer,
    packed: GpuBuffer,
    logits: GpuBuffer,
    stats: GpuBuffer,
    dx: GpuBuffer,
    dw: GpuBuffer,
    grad_logits: GpuBuffer,
    source: f32,
    z: f32,
}

impl Fixture {
    fn new(device: &VulkanDevice, source: f32, z: f32) -> Result<Self> {
        // Row 0 is supervised. Row 1 has an absent CE source and poisoned
        // activations/statistics, reproducing 0 * NaN in the old adjoint paths.
        let mut x = vec![1.0, 0.0, -1.0, 0.5];
        x.extend([f32::NAN; WIDTH]);
        let mut stats = [
            0.125,
            1.0 / VOCAB as f32,
            f32::from_bits(3),
            0.0,
            source,
            f32::NAN,
            f32::NAN,
            f32::from_bits(0),
            f32::NAN,
            0.0,
        ];
        let mut logits = vec![0.125; VOCAB];
        logits.extend([f32::NAN; VOCAB]);
        if source == 0.0 {
            // A scalar backward cap can remove the entire CE edge while its
            // forward diagnostics remain non-finite. It must still emit zero.
            x[..WIDTH].fill(f32::NAN);
            logits[..VOCAB].fill(f32::NAN);
            stats[0] = f32::NAN;
            stats[1] = f32::NAN;
        }
        Ok(Self {
            x: GpuBuffer::from_u32(device, &x.iter().map(|v| v.to_bits()).collect::<Vec<_>>())?,
            gamma: GpuBuffer::from_f32(device, &[1.0; WIDTH])?,
            beta: GpuBuffer::zeros_f32(device, WIDTH)?,
            mean: GpuBuffer::from_u32(device, &[0.0f32.to_bits(), f32::NAN.to_bits()])?,
            rstd: GpuBuffer::from_u32(device, &[1.0f32.to_bits(), f32::NAN.to_bits()])?,
            weights: GpuBuffer::from_f32(device, &[0.25; VOCAB * WIDTH])?,
            // IEEE half(0.25) = 0x3400; both components have the same value.
            packed: GpuBuffer::from_u32(device, &[0x34003400; VOCAB * WIDTH / 2])?,
            logits: GpuBuffer::from_u32(
                device,
                &logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            )?,
            stats: GpuBuffer::from_u32(
                device,
                &stats.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            )?,
            dx: GpuBuffer::from_f32(device, &[9.0; WIDTH * 2])?,
            dw: GpuBuffer::from_f32(device, &[9.0; VOCAB * WIDTH])?,
            grad_logits: GpuBuffer::from_f32(device, &[9.0; VOCAB * 2])?,
            source,
            z,
        })
    }

    fn linear_push(&self) -> LinearPush {
        LinearPush {
            rows: 2,
            input_dim: WIDTH as u32,
            output_dim: VOCAB as u32,
            z_loss_weight: self.z,
            activation_clamp: 100.0,
        }
    }

    fn weight_push(&self) -> LmWeightGradPush {
        LmWeightGradPush {
            rows: 2,
            input_dim: WIDTH as u32,
            output_dim: VOCAB as u32,
            accumulate: 0,
            z_loss_weight: self.z,
            activation_clamp: 100.0,
        }
    }

    fn expected(&self) -> (Vec<f32>, Vec<f32>) {
        let mut dx = vec![0.0; WIDTH * 2];
        let mut dw = vec![0.0; VOCAB * WIDTH];
        let log_z = 0.125 + (VOCAB as f32).ln();
        for v in 0..VOCAB {
            let probability = 1.0 / VOCAB as f32;
            let grad = (probability - if v == 3 { 1.0 } else { 0.0 }
                + 2.0 * self.z * log_z * probability)
                * self.source;
            for (i, hidden) in [1.0, 0.0, -1.0, 0.5].iter().enumerate() {
                dx[i] += grad * 0.25;
                dw[v * WIDTH + i] = grad * hidden;
            }
        }
        (dx, dw)
    }

    fn check(&self, label: &str) -> Result<()> {
        let (dx, dw) = self.expected();
        assert_close(&self.dx.read_f32(WIDTH * 2)?, &dx, &format!("{label} dX"));
        assert_close(
            &self.dw.read_f32(VOCAB * WIDTH)?,
            &dw,
            &format!("{label} dW"),
        );
        for value in &self.dx.read_f32(WIDTH * 2)?[WIDTH..] {
            assert_eq!(value.to_bits(), 0, "{label}: ignored dX must be exact +0");
        }
        Ok(())
    }
}

#[test]
fn hierarchos_ce_zero_sources_do_not_poison_streaming_adjoints() -> Result<()> {
    let device = VulkanDevice::new()?;
    for source in [1.0, 0.0] {
        for z in [0.0, 0.01] {
            for arm in ["fp32", "packed", "native", "native_compute"] {
                // Native half products intentionally round active derivatives;
                // use FP32-compute arms for the independent positive oracle.
                if arm == "native_compute" && source != 0.0 {
                    continue;
                }
                if arm.starts_with("native")
                    && !device
                        .mixed_precision_capabilities()
                        .native_fp16_storage_compute_ready()
                {
                    continue;
                }
                let f = Fixture::new(&device, source, z)?;
                let input_stem = match arm {
                    "fp32" => "cross_entropy_linear_input_grad_streaming",
                    "packed" => "cross_entropy_linear_input_grad_streaming_fp16_packed",
                    "native" => "cross_entropy_linear_input_grad_streaming_fp16_native",
                    _ => "cross_entropy_linear_input_grad_streaming_fp16_native_compute",
                };
                let mut commands = vulkan::ComputeBatch::new(&device)?;
                let mut buffers = vec![
                    &f.x,
                    &f.gamma,
                    &f.beta,
                    &f.mean,
                    &f.rstd,
                    if arm == "fp32" { &f.weights } else { &f.packed },
                    &f.stats,
                    &f.dx,
                ];
                if arm != "fp32" {
                    buffers.push(&f.grad_logits);
                }
                let input_kernel = kernel(&device, input_stem, buffers.len(), 20)?;
                input_kernel.record_dispatch(
                    &mut commands,
                    &buffers,
                    bytemuck::bytes_of(&f.linear_push()),
                    [2, 1, 1],
                )?;
                let weight_stem = if arm == "fp32" {
                    "cross_entropy_linear_weight_grad_streaming"
                } else {
                    "cross_entropy_linear_weight_grad_streaming_fp16_packed"
                };
                let weight_buffers = if arm == "fp32" {
                    vec![
                        &f.x, &f.gamma, &f.beta, &f.mean, &f.rstd, &f.weights, &f.stats, &f.dw,
                    ]
                } else {
                    vec![
                        &f.x,
                        &f.gamma,
                        &f.beta,
                        &f.mean,
                        &f.rstd,
                        &f.grad_logits,
                        &f.dw,
                    ]
                };
                let weight_kernel = kernel(&device, weight_stem, weight_buffers.len(), 24)?;
                weight_kernel.record_dispatch(
                    &mut commands,
                    &weight_buffers,
                    bytemuck::bytes_of(&f.weight_push()),
                    [if arm == "fp32" { 5 } else { 3 }, 1, 1],
                )?;
                commands.submit()?;
                f.check(arm)?;
            }
        }
    }
    Ok(())
}

#[test]
fn hierarchos_ce_zero_sources_do_not_poison_tape_adjoints() -> Result<()> {
    let device = VulkanDevice::new()?;
    for source in [1.0, 0.0] {
        let f = Fixture::new(&device, source, 0.01)?;
        let input = kernel(
            &device,
            "cross_entropy_linear_input_grad_tape_fp16_packed",
            4,
            16,
        )?;
        let weight = kernel(
            &device,
            "cross_entropy_linear_weight_grad_tape_fp16_packed",
            8,
            24,
        )?;
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        input.record_dispatch(
            &mut commands,
            &[&f.logits, &f.stats, &f.packed, &f.dx],
            &bytemuck::bytes_of(&f.linear_push())[..16],
            [1, 1, 1],
        )?;
        weight.record_dispatch(
            &mut commands,
            &[
                &f.x, &f.gamma, &f.beta, &f.mean, &f.rstd, &f.logits, &f.stats, &f.dw,
            ],
            bytemuck::bytes_of(&f.weight_push()),
            [3, 1, 1],
        )?;
        commands.submit()?;
        f.check("tape")?;

        let f = Fixture::new(&device, source, 0.01)?;
        let fused = kernel(
            &device,
            "cross_entropy_linear_adjoints_tape_fp16_packed_fused",
            10,
            24,
        )?;
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        fused.record_dispatch(
            &mut commands,
            &[
                &f.x, &f.gamma, &f.beta, &f.mean, &f.rstd, &f.logits, &f.stats, &f.packed, &f.dw,
                &f.dx,
            ],
            bytemuck::bytes_of(&f.weight_push()),
            [1, 1, 1],
        )?;
        commands.submit()?;
        f.check("fused tape")?;
    }
    Ok(())
}

#[test]
fn hierarchos_ce_retains_loss_under_large_common_logit_offsets() -> Result<()> {
    let device = VulkanDevice::new()?;
    let target = GpuBuffer::from_u32(&device, &[3])?;
    let push = [1u32, WIDTH as u32, VOCAB as u32];
    for offset in [0.0, 1.0e8] {
        let logits = GpuBuffer::from_f32(&device, &[offset; VOCAB])?;
        let stats = GpuBuffer::zeros_f32(&device, 5)?;
        let shader = kernel(&device, "cross_entropy_row_stats_tape", 3, 12)?;
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        shader.record_dispatch(
            &mut commands,
            &[&logits, &target, &stats],
            bytemuck::cast_slice(&push),
            [1, 1, 1],
        )?;
        commands.submit()?;
        let actual = stats.read_f32(5)?[3];
        assert!(
            (actual - (VOCAB as f32).ln()).abs() <= 1e-6,
            "offset={offset} CE={actual}"
        );
        for tile in [8, 16] {
            let partials = if tile == 8 {
                vec![offset, 8.0, offset, offset, 8.0, 0.0, offset, 1.0, 0.0]
            } else {
                vec![offset, 16.0, offset, offset, 1.0, 0.0]
            };
            let partials = GpuBuffer::from_f32(&device, &partials)?;
            let stem = if tile == 8 {
                "cross_entropy_row_stats_tile_partials"
            } else {
                "cross_entropy_row_stats_tile_partials_rows16"
            };
            let shader = kernel(&device, stem, 3, 12)?;
            let mut commands = vulkan::ComputeBatch::new(&device)?;
            shader.record_dispatch(
                &mut commands,
                &[&partials, &target, &stats],
                bytemuck::cast_slice(&push),
                [1, 1, 1],
            )?;
            commands.submit()?;
            assert_close(&[stats.read_f32(5)?[3]], &[(VOCAB as f32).ln()], stem);
        }
        for packed in [false, true] {
            let f = Fixture::new(&device, 1.0, 0.0)?;
            f.gamma.write_f32(&[0.0; WIDTH])?;
            f.beta.write_f32(&[offset; WIDTH])?;
            let stem = if packed {
                "cross_entropy_linear_row_stats_streaming_fp16_packed"
            } else {
                "cross_entropy_linear_row_stats_streaming"
            };
            let shader = kernel(&device, stem, 8, 20)?;
            let mut commands = vulkan::ComputeBatch::new(&device)?;
            let push = LinearPush {
                rows: 1,
                activation_clamp: f32::MAX,
                ..f.linear_push()
            };
            shader.record_dispatch(
                &mut commands,
                &[
                    &f.x,
                    &f.gamma,
                    &f.beta,
                    &f.mean,
                    &f.rstd,
                    if packed { &f.packed } else { &f.weights },
                    &target,
                    &stats,
                ],
                bytemuck::bytes_of(&push),
                [1, 1, 1],
            )?;
            commands.submit()?;
            assert_close(&[stats.read_f32(5)?[3]], &[(VOCAB as f32).ln()], stem);
        }
    }
    Ok(())
}
