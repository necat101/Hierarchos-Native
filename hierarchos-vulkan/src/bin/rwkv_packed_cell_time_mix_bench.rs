use std::{fs, path::PathBuf, time::Instant};

use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{
    variant_dispatch_counts, vendor_kernel_plan, RwkvPackedCellOp, RwkvStateReadoutMode,
    VendorKernelFamily, VulkanDevice,
};
use serde::{Deserialize, Serialize};

/// Mirrors `HIERARCHOS_RWKV_TIME_MIX_FORCE_FORWARD_FUSION_ENV` in
/// `rwkv_time_mix_core`. The env name is not part of the public API surface, so
/// the harness repeats the literal rather than depending on a private constant.
const TIME_MIX_FUSION_FORCE_ENV: &str = "HIERARCHOS_RWKV_TIME_MIX_FORCE_FORWARD_FUSION";
const TIME_MIX_FUSION_DISABLE_ENV: &str = "HIERARCHOS_VULKAN_DISABLE_TIME_MIX_FORWARD_FUSION";

#[derive(Deserialize)]
struct Case {
    batch: usize,
    width: usize,
    head_size: usize,
    input_dim: usize,
    state_mode: String,
    state_clamp: f32,
    x: Vec<f32>,
    token_features: Vec<f32>,
    packed_state: Vec<f32>,
    grad_output: Vec<f32>,
    grad_packed_new_state: Vec<f32>,
}

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
    /// The graph the op reports it will dispatch for this batch.
    label: &'static str,
    /// Vendor `linear3-forward` dispatches the leg's first run performed.
    linear3_dispatches: u64,
    timing: Timing,
}

#[derive(Serialize)]
struct Output {
    device: String,
    width: usize,
    state_size: usize,
    batch: usize,
    linear3_variant: String,
    fused: LegReport,
    composed: LegReport,
    default: LegReport,
    composed_over_fused: f64,
    default_over_fused: f64,
    /// Worst per-tensor difference between the fused and composed legs. Both
    /// graphs must agree within the harness tolerance, or the timing compares
    /// different work.
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

/// Compare one tensor against the tolerance the harness applies, so a timing
/// that compares different work is rejected instead of reported.
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

fn build_op(
    device: VulkanDevice,
    case: &Case,
    model_dir: &PathBuf,
    cell_prefix: &str,
    adapter_prefix: &str,
    state_mode: RwkvStateReadoutMode,
) -> Result<RwkvPackedCellOp> {
    RwkvPackedCellOp::from_model_package(
        device,
        model_dir,
        cell_prefix,
        adapter_prefix,
        case.head_size,
        case.batch,
        12.0,
        4.0,
        state_mode,
        case.state_clamp,
    )
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut case_path = None;
    let mut model_dir = None;
    let mut cell_prefix = None;
    let mut adapter_prefix = None;
    let mut iterations = 30usize;
    let mut warmup = 5usize;
    let mut atol = 1.0e-4f64;
    let mut rtol = 1.0e-5f64;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--case" => case_path = args.next().map(PathBuf::from),
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
    let case_path = case_path.context(
        "usage: --case CASE.json --model-dir MODEL_DIR --cell-prefix h_rnn \
         --adapter-prefix h_deepembed_adapter [--iterations 30] [--warmup 5] [--atol 1e-4] [--rtol 1e-5]",
    )?;
    let model_dir = model_dir.context("benchmark requires --model-dir")?;
    let cell_prefix = cell_prefix.context("benchmark requires --cell-prefix")?;
    let adapter_prefix = adapter_prefix.context("benchmark requires --adapter-prefix")?;
    let case: Case = serde_json::from_slice(&fs::read(case_path)?)?;
    let state_mode = match case.state_mode.as_str() {
        "legacy-input-cache" => RwkvStateReadoutMode::LegacyInputCache,
        "explicit-output" => RwkvStateReadoutMode::ExplicitOutput,
        other => bail!("unknown packed state mode {other:?}"),
    };

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

    // The three legs differ only in the switches resolved at construction and
    // record time; the fixture, geometry and weights are identical.
    std::env::set_var(TIME_MIX_FUSION_FORCE_ENV, "1");
    let mut fused = build_op(
        device.clone(),
        &case,
        &model_dir,
        &cell_prefix,
        &adapter_prefix,
        state_mode,
    )?;
    std::env::remove_var(TIME_MIX_FUSION_FORCE_ENV);

    std::env::set_var(TIME_MIX_FUSION_DISABLE_ENV, "1");
    let composed = build_op(
        device.clone(),
        &case,
        &model_dir,
        &cell_prefix,
        &adapter_prefix,
        state_mode,
    );
    std::env::remove_var(TIME_MIX_FUSION_DISABLE_ENV);
    let mut composed = composed?;

    let mut default = build_op(
        device,
        &case,
        &model_dir,
        &cell_prefix,
        &adapter_prefix,
        state_mode,
    )?;

    for (name, op) in [
        ("fused", &fused),
        ("composed", &composed),
        ("default", &default),
    ] {
        if op.width() != case.width || op.token_feature_width() != case.input_dim {
            bail!(
                "{name} op geometry width/input_dim={}/{} does not match case {}/{}",
                op.width(),
                op.token_feature_width(),
                case.width,
                case.input_dim
            );
        }
    }

    let run = |op: &mut RwkvPackedCellOp| {
        op.forward_backward(
            case.batch,
            &case.x,
            &case.token_features,
            &case.packed_state,
            &case.grad_output,
            &case.grad_packed_new_state,
        )
    };

    // Two independent markers per leg: the label the op reports for this batch
    // and the vendor dispatch counter, which proves what actually ran.
    let before = linear3_dispatches();
    run(&mut fused)?;
    let fused_dispatches = linear3_dispatches() - before;
    let before = linear3_dispatches();
    run(&mut composed)?;
    let composed_dispatches = linear3_dispatches() - before;
    let before = linear3_dispatches();
    run(&mut default)?;
    let default_dispatches = linear3_dispatches() - before;

    let fused_label = fused.time_mix_full_forward_projection_label(case.batch);
    let composed_label = composed.time_mix_full_forward_projection_label(case.batch);
    let default_label = default.time_mix_full_forward_projection_label(case.batch);
    if !fused_label.starts_with("fused") || composed_label != "composed" {
        bail!(
            "the switches did not separate the two graphs: fused leg built {fused_label}, \
             composed leg built {composed_label}"
        );
    }
    if fused_dispatches != 0 {
        bail!(
            "the forced-fusion leg dispatched linear3-forward {fused_dispatches} time(s); it did \
             not run the fused projection, so this run is not an A/B"
        );
    }
    // The promotion contract: the default graph is the composed projection
    // exactly where the tuned module ran for the composed leg, and the fused
    // kernel everywhere else.
    let expected_default = if composed_dispatches > 0 {
        "composed"
    } else {
        fused_label
    };
    if default_label != expected_default {
        bail!(
            "the default leg built {default_label} at batch {} while the tuned linear3 module {} \
             run in the composed leg (expected {expected_default})",
            case.batch,
            if composed_dispatches > 0 {
                "did"
            } else {
                "did not"
            }
        );
    }
    if default_dispatches != composed_dispatches && default_label == "composed" {
        bail!(
            "the default leg reports {default_label} but dispatched linear3-forward \
             {default_dispatches} time(s) against the composed leg's {composed_dispatches}"
        );
    }

    for _ in 1..warmup {
        run(&mut fused)?;
        run(&mut composed)?;
        run(&mut default)?;
    }

    let mut fused_samples = Vec::with_capacity(iterations);
    let mut composed_samples = Vec::with_capacity(iterations);
    let mut default_samples = Vec::with_capacity(iterations);
    let mut fused_last = None;
    let mut composed_last = None;
    for iteration in 0..iterations {
        // Rotate which graph runs first so neither a warm cache nor clock drift
        // systematically favors one leg.
        let order = match iteration % 3 {
            0 => [0usize, 1, 2],
            1 => [1, 2, 0],
            _ => [2, 0, 1],
        };
        for leg in order {
            let started = Instant::now();
            match leg {
                0 => {
                    fused_last = Some(run(&mut fused)?);
                    fused_samples.push(started.elapsed().as_secs_f64() * 1.0e3);
                }
                1 => {
                    composed_last = Some(run(&mut composed)?);
                    composed_samples.push(started.elapsed().as_secs_f64() * 1.0e3);
                }
                _ => {
                    run(&mut default)?;
                    default_samples.push(started.elapsed().as_secs_f64() * 1.0e3);
                }
            }
        }
    }
    let fused_last = fused_last.context("no fused benchmark sample")?;
    let composed_last = composed_last.context("no composed benchmark sample")?;

    let tensors = vec![
        tensor_diff(
            "output",
            &fused_last.output,
            &composed_last.output,
            atol,
            rtol,
        ),
        tensor_diff(
            "packed_new_state",
            &fused_last.packed_new_state,
            &composed_last.packed_new_state,
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
            "grad_packed_state",
            &fused_last.grad_packed_state,
            &composed_last.grad_packed_state,
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
    ];
    if let Some(outside) = tensors.iter().find(|tensor| tensor.worst_excess > 0.0) {
        bail!(
            "the timed graphs disagree on {}: max abs {:e} exceeds atol {atol:e} + rtol {rtol:e} \
             (worst excess {:e}), so this timing compares different work",
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
        width: fused.width(),
        state_size: fused.state_size(),
        batch: case.batch,
        linear3_variant,
        composed_over_fused: composed_timing.median_ms / fused_timing.median_ms,
        default_over_fused: default_timing.median_ms / fused_timing.median_ms,
        worst_max_abs,
        tensors,
        fused: LegReport {
            label: fused_label,
            linear3_dispatches: fused_dispatches,
            timing: fused_timing,
        },
        composed: LegReport {
            label: composed_label,
            linear3_dispatches: composed_dispatches,
            timing: composed_timing,
        },
        default: LegReport {
            label: default_label,
            linear3_dispatches: default_dispatches,
            timing: default_timing,
        },
    };
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
