use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{
    variant_dispatch_counts, vendor_kernel_plan, RwkvStateReadoutMode, RwkvTbpttSchedule,
    RwkvTbpttSequenceOp, VendorKernelFamily, VulkanDevice,
};
use serde::Serialize;

/// Mirrors `HIERARCHOS_RWKV_TIME_MIX_FORCE_FORWARD_FUSION_ENV` in
/// `rwkv_time_mix_core`. The env name is not part of the public API surface, so
/// the harness repeats the literal rather than depending on a private constant.
const TIME_MIX_FUSION_FORCE_ENV: &str = "HIERARCHOS_RWKV_TIME_MIX_FORCE_FORWARD_FUSION";
const TIME_MIX_FUSION_DISABLE_ENV: &str = "HIERARCHOS_VULKAN_DISABLE_TIME_MIX_FORWARD_FUSION";
const DEFAULT_STEPS: usize = 8;

#[derive(Serialize)]
struct Timing {
    samples: usize,
    median_ms: f64,
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

#[derive(Serialize)]
struct TensorDiff {
    label: &'static str,
    values: usize,
    max_abs: f64,
    worst_excess: f64,
}

#[derive(Serialize)]
struct LegReport {
    /// The graph the sequence reports it will dispatch for this batch.
    label: &'static str,
    /// Vendor `linear3-forward` dispatches one whole sequence performed.
    linear3_dispatches: u64,
    /// Whole-sequence wall time, and the same divided by `steps`.
    timing: Timing,
    ms_per_step: f64,
}

#[derive(Serialize)]
struct Output {
    device: String,
    width: usize,
    state_size: usize,
    head_size: usize,
    batch: usize,
    steps: usize,
    linear3_variant: String,
    fused: LegReport,
    composed: LegReport,
    default: LegReport,
    composed_over_fused: f64,
    default_over_fused: f64,
    /// Worst per-tensor difference between the fused and composed legs over the
    /// whole sequence. The graphs must agree within tolerance, or the timing
    /// compares different work.
    worst_max_abs: f64,
    tensors: Vec<TensorDiff>,
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|left, right| left.total_cmp(right));
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn timing(mut samples: Vec<f64>) -> Timing {
    let count = samples.len();
    let sum: f64 = samples.iter().sum();
    let median_ms = median(&mut samples);
    Timing {
        samples: count,
        median_ms,
        mean_ms: sum / count as f64,
        min_ms: samples[0],
        max_ms: samples[count - 1],
    }
}

fn linear3_dispatches() -> u64 {
    variant_dispatch_counts()
        .into_iter()
        .find(|(label, _)| *label == "linear3-forward")
        .map(|(_, count)| count)
        .unwrap_or(0)
}

fn tensor_diff(
    label: &'static str,
    fused: &[f32],
    composed: &[f32],
    atol: f64,
    rtol: f64,
) -> TensorDiff {
    let mut max_abs = 0.0f64;
    let mut worst_excess = f64::NEG_INFINITY;
    for (left, right) in fused.iter().zip(composed) {
        let difference = (*left as f64 - *right as f64).abs();
        let bound = atol + rtol * (*right as f64).abs();
        max_abs = max_abs.max(difference);
        worst_excess = worst_excess.max(difference - bound);
    }
    TensorDiff {
        label,
        values: fused.len(),
        max_abs,
        worst_excess,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_sequence(
    device: VulkanDevice,
    model_dir: &Path,
    cell_prefix: &str,
    adapter_prefix: &str,
    head_size: usize,
    batch: usize,
    steps: usize,
) -> Result<RwkvTbpttSequenceOp> {
    RwkvTbpttSequenceOp::from_model_package(
        device,
        model_dir,
        cell_prefix,
        adapter_prefix,
        head_size,
        batch,
        steps,
        12.0,
        4.0,
        RwkvStateReadoutMode::ExplicitOutput,
        50.0,
    )
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut model_dir = None;
    let mut cell_prefix = None;
    let mut adapter_prefix = None;
    let mut head_size: Option<usize> = None;
    let mut input_dim: Option<usize> = None;
    let mut batches: Vec<usize> = Vec::new();
    let mut steps = DEFAULT_STEPS;
    let mut iterations = 10usize;
    let mut warmup = 2usize;
    let mut atol = 1.0e-4f64;
    let mut rtol = 1.0e-5f64;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--model-dir" => model_dir = args.next().map(PathBuf::from),
            "--cell-prefix" => {
                cell_prefix = args
                    .next()
                    .map(|value| value.to_string_lossy().into_owned())
            }
            "--adapter-prefix" => {
                adapter_prefix = args
                    .next()
                    .map(|value| value.to_string_lossy().into_owned())
            }
            "--head-size" => {
                head_size = Some(
                    args.next()
                        .context("--head-size needs a value")?
                        .to_string_lossy()
                        .parse()
                        .context("--head-size must be a positive integer")?,
                )
            }
            "--input-dim" => {
                input_dim = Some(
                    args.next()
                        .context("--input-dim needs a value")?
                        .to_string_lossy()
                        .parse()
                        .context("--input-dim must be a positive integer")?,
                )
            }
            "--batch" => batches.push(
                args.next()
                    .context("--batch needs a value")?
                    .to_string_lossy()
                    .parse()
                    .context("--batch must be a positive integer")?,
            ),
            "--steps" => {
                steps = args
                    .next()
                    .context("--steps needs a value")?
                    .to_string_lossy()
                    .parse()
                    .context("--steps must be a positive integer")?
            }
            "--iterations" => {
                iterations = args
                    .next()
                    .context("--iterations needs a value")?
                    .to_string_lossy()
                    .parse()
                    .context("--iterations must be a positive integer")?
            }
            "--warmup" => {
                warmup = args
                    .next()
                    .context("--warmup needs a value")?
                    .to_string_lossy()
                    .parse()
                    .context("--warmup must be a non-negative integer")?
            }
            "--atol" => {
                atol = args
                    .next()
                    .context("--atol needs a value")?
                    .to_string_lossy()
                    .parse()
                    .context("--atol must be a number")?
            }
            "--rtol" => {
                rtol = args
                    .next()
                    .context("--rtol needs a value")?
                    .to_string_lossy()
                    .parse()
                    .context("--rtol must be a number")?
            }
            other => bail!("unknown argument {other:?}"),
        }
    }
    if iterations == 0 {
        bail!("--iterations must be positive");
    }
    if steps == 0 {
        bail!("--steps must be positive");
    }
    if batches.is_empty() {
        batches.push(1);
    }
    let model_dir = model_dir.context("benchmark requires --model-dir")?;
    let cell_prefix = cell_prefix.context("benchmark requires --cell-prefix")?;
    let adapter_prefix = adapter_prefix.context("benchmark requires --adapter-prefix")?;
    let head_size = head_size.context("benchmark requires --head-size")?;
    let input_dim = input_dim.context("benchmark requires --input-dim")?;

    let device = VulkanDevice::new()?;
    let plan = vendor_kernel_plan(&device);
    let linear3_variant = plan
        .iter()
        .find(|selection| selection.family == VendorKernelFamily::Linear3Forward)
        .context("vendor plan has no linear3-forward entry")?
        .variant
        .to_owned();
    if linear3_variant == "portable" {
        bail!(
            "linear3-forward resolved the portable module on this adapter, so the composition \
             could never reach a tuned module and this A/B would be vacuous"
        );
    }

    for batch in batches {
        // The three legs differ only in the switches resolved at construction;
        // fixture, geometry, seed and weights are identical.
        std::env::set_var(TIME_MIX_FUSION_FORCE_ENV, "1");
        let mut fused = build_sequence(
            device.clone(),
            &model_dir,
            &cell_prefix,
            &adapter_prefix,
            head_size,
            batch,
            steps,
        )?;
        std::env::remove_var(TIME_MIX_FUSION_FORCE_ENV);

        std::env::set_var(TIME_MIX_FUSION_DISABLE_ENV, "1");
        let mut composed = build_sequence(
            device.clone(),
            &model_dir,
            &cell_prefix,
            &adapter_prefix,
            head_size,
            batch,
            steps,
        )?;
        std::env::remove_var(TIME_MIX_FUSION_DISABLE_ENV);

        let mut default = build_sequence(
            device.clone(),
            &model_dir,
            &cell_prefix,
            &adapter_prefix,
            head_size,
            batch,
            steps,
        )?;

        let width = fused.width();
        if fused.state_size() != composed.state_size() || width != composed.width() {
            bail!("the three legs built different state geometry");
        }
        let vector_len = batch * width;
        let state_len = vector_len * fused.state_size();
        let x = deterministic_values(steps * vector_len, 0.017, 3);
        let grad_output = deterministic_values(steps * vector_len, 0.009, 11);
        let initial_state = deterministic_values(state_len, 0.003, 17);
        // The DeepEmbed adapter input is held at zero, exactly as the packed-cell
        // A/B does, so the two harnesses time the same work per step.
        let token_features = vec![0.0f32; steps * batch * input_dim];

        let run = |op: &mut RwkvTbpttSequenceOp| {
            op.run(
                batch,
                steps,
                &x,
                &token_features,
                &initial_state,
                &grad_output,
                None,
                RwkvTbpttSchedule::full_bptt(),
            )
        };

        // Labels first, then one instrumented run per leg so the dispatch
        // counter proves which graph actually ran.
        let fused_label = fused.time_mix_full_forward_projection_label(batch);
        let composed_label = composed.time_mix_full_forward_projection_label(batch);
        let default_label = default.time_mix_full_forward_projection_label(batch);
        if !fused_label.starts_with("fused") || composed_label != "composed" {
            bail!(
                "the switches did not separate the two graphs: fused leg built {fused_label}, \
                 composed leg built {composed_label}"
            );
        }
        let before = linear3_dispatches();
        let mut fused_last = run(&mut fused)?;
        let fused_dispatches = linear3_dispatches() - before;
        let before = linear3_dispatches();
        let mut composed_last = run(&mut composed)?;
        let composed_dispatches = linear3_dispatches() - before;
        let before = linear3_dispatches();
        run(&mut default)?;
        let default_dispatches = linear3_dispatches() - before;
        if fused_dispatches != 0 {
            bail!(
                "the forced-fusion leg dispatched linear3-forward {fused_dispatches} time(s) over \
                 {steps} steps; it did not run the fused projection, so this run is not an A/B"
            );
        }
        // The promotion contract, now read end to end: the default sequence is
        // the composed projection exactly where the tuned module ran for the
        // composed sequence, and the fused kernel everywhere else.
        let expected_default = if composed_dispatches > 0 {
            "composed"
        } else {
            fused_label
        };
        if default_label != expected_default {
            bail!(
                "the default sequence built {default_label} at batch {batch} while the tuned \
                 linear3 module {} run in the composed sequence (expected {expected_default})",
                if composed_dispatches > 0 {
                    "did"
                } else {
                    "did not"
                }
            );
        }

        for _ in 0..warmup {
            run(&mut fused)?;
            run(&mut composed)?;
            run(&mut default)?;
        }

        let mut fused_samples = Vec::with_capacity(iterations);
        let mut composed_samples = Vec::with_capacity(iterations);
        let mut default_samples = Vec::with_capacity(iterations);
        for iteration in 0..iterations {
            // Rotate which sequence runs first so neither warm buffers nor clock
            // drift systematically favors one leg.
            let order = match iteration % 3 {
                0 => [0usize, 1, 2],
                1 => [1, 2, 0],
                _ => [2, 0, 1],
            };
            for leg in order {
                let started = Instant::now();
                match leg {
                    0 => {
                        fused_last = run(&mut fused)?;
                        fused_samples.push(started.elapsed().as_secs_f64() * 1.0e3);
                    }
                    1 => {
                        composed_last = run(&mut composed)?;
                        composed_samples.push(started.elapsed().as_secs_f64() * 1.0e3);
                    }
                    _ => {
                        run(&mut default)?;
                        default_samples.push(started.elapsed().as_secs_f64() * 1.0e3);
                    }
                }
            }
        }

        let tensors = vec![
            tensor_diff(
                "outputs",
                &fused_last.outputs,
                &composed_last.outputs,
                atol,
                rtol,
            ),
            tensor_diff(
                "final_packed_state",
                &fused_last.final_packed_state,
                &composed_last.final_packed_state,
                atol,
                rtol,
            ),
            tensor_diff(
                "grad_x",
                &fused_last.grad_x,
                &composed_last.grad_x,
                atol,
                rtol,
            ),
            tensor_diff(
                "token_feature_grad",
                &fused_last.token_feature_grad,
                &composed_last.token_feature_grad,
                atol,
                rtol,
            ),
            tensor_diff(
                "grad_initial_packed_state",
                &fused_last.grad_initial_packed_state,
                &composed_last.grad_initial_packed_state,
                atol,
                rtol,
            ),
        ];
        if let Some(outside) = tensors.iter().find(|tensor| tensor.worst_excess > 0.0) {
            bail!(
                "the timed sequences disagree on {}: max abs {:e} exceeds atol {atol:e} + rtol \
                 {rtol:e} (worst excess {:e}), so this timing compares different work",
                outside.label,
                outside.max_abs,
                outside.worst_excess
            );
        }
        let worst_max_abs = tensors
            .iter()
            .map(|tensor| tensor.max_abs)
            .fold(0.0f64, f64::max);

        let fused_timing = timing(fused_samples);
        let composed_timing = timing(composed_samples);
        let default_timing = timing(default_samples);
        let output = Output {
            device: fused.device_name().to_string(),
            width,
            state_size: fused.state_size(),
            head_size,
            batch,
            steps,
            linear3_variant: linear3_variant.clone(),
            composed_over_fused: composed_timing.median_ms / fused_timing.median_ms,
            default_over_fused: default_timing.median_ms / fused_timing.median_ms,
            worst_max_abs,
            tensors,
            fused: LegReport {
                label: fused_label,
                linear3_dispatches: fused_dispatches,
                ms_per_step: fused_timing.median_ms / steps as f64,
                timing: fused_timing,
            },
            composed: LegReport {
                label: composed_label,
                linear3_dispatches: composed_dispatches,
                ms_per_step: composed_timing.median_ms / steps as f64,
                timing: composed_timing,
            },
            default: LegReport {
                label: default_label,
                linear3_dispatches: default_dispatches,
                ms_per_step: default_timing.median_ms / steps as f64,
                timing: default_timing,
            },
        };
        println!("{}", serde_json::to_string(&output)?);
    }
    Ok(())
}

fn deterministic_values(len: usize, scale: f32, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|index| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / 16_777_216.0;
            (unit * 2.0 - 1.0) * scale * (1.0 + (index % 7) as f32 * 0.05)
        })
        .collect()
}
