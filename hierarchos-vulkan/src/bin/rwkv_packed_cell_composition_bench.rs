use std::{fs, path::PathBuf, time::Instant};

use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{RwkvPackedCellOp, RwkvStateReadoutMode, VulkanDevice};
use serde::{Deserialize, Serialize};

const FUSION_DISABLE_ENV: &str = "HIERARCHOS_VULKAN_DISABLE_CHANNEL_MIX_FORWARD_FUSION";

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
    kernel: &'static str,
    samples: usize,
    median_ms: f64,
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

#[derive(Serialize)]
struct Output {
    device: String,
    width: usize,
    state_size: usize,
    batch: usize,
    fused_kernel_marker: bool,
    composed_kernel_marker: bool,
    max_abs_output: f64,
    max_abs_packed_state: f64,
    fused: Timing,
    composed: Timing,
    composed_over_fused: f64,
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

fn timing(kernel: &'static str, mut samples: Vec<f64>) -> Timing {
    let count = samples.len();
    let sum: f64 = samples.iter().sum();
    let median_ms = median(&mut samples);
    Timing {
        kernel,
        samples: count,
        median_ms,
        mean_ms: sum / count as f64,
        min_ms: samples[0],
        max_ms: samples[count - 1],
    }
}

fn max_abs(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(l, r)| (*l as f64 - *r as f64).abs())
        .fold(0.0, f64::max)
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut case_path = None;
    let mut model_dir = None;
    let mut cell_prefix = None;
    let mut adapter_prefix = None;
    let mut iterations = 30usize;
    let mut warmup = 5usize;
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
            other => bail!("unknown argument {other:?}"),
        }
    }
    if iterations == 0 {
        bail!("--iterations must be positive");
    }
    let case_path = case_path.context(
        "usage: --case CASE.json --model-dir MODEL_DIR --cell-prefix h_rnn \
         --adapter-prefix h_deepembed_adapter [--iterations 30] [--warmup 5]",
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
    let mut fused = RwkvPackedCellOp::from_model_package(
        device.clone(),
        &model_dir,
        &cell_prefix,
        &adapter_prefix,
        case.head_size,
        case.batch,
        12.0,
        4.0,
        state_mode,
        case.state_clamp,
    )?;
    // The switch is resolved once, when the channel-mix op is constructed, so
    // setting it around the second construction selects the composed graph for
    // that op only. The fused op keeps the graph it was built with.
    std::env::set_var(FUSION_DISABLE_ENV, "1");
    let composed = RwkvPackedCellOp::from_model_package(
        device,
        &model_dir,
        &cell_prefix,
        &adapter_prefix,
        case.head_size,
        case.batch,
        12.0,
        4.0,
        state_mode,
        case.state_clamp,
    );
    std::env::remove_var(FUSION_DISABLE_ENV);
    let mut composed = composed?;

    if !fused.packed_channel_mix_forward_fused() {
        bail!("the fused benchmark op did not build the fused packed-cell kernel");
    }
    if composed.packed_channel_mix_forward_fused() {
        bail!(
            "{FUSION_DISABLE_ENV} did not select the composed graph; the comparison would time \
             the same kernel twice"
        );
    }
    for (name, op) in [("fused", &fused), ("composed", &composed)] {
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

    for _ in 0..warmup {
        run(&mut fused)?;
        run(&mut composed)?;
    }

    let mut fused_samples = Vec::with_capacity(iterations);
    let mut composed_samples = Vec::with_capacity(iterations);
    let mut fused_last = None;
    let mut composed_last = None;
    for iteration in 0..iterations {
        // Alternate the order so a steady clock or thermal drift cannot favor
        // whichever graph happens to run first in every sample.
        if iteration % 2 == 0 {
            let start = Instant::now();
            fused_last = Some(run(&mut fused)?);
            fused_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
            let start = Instant::now();
            composed_last = Some(run(&mut composed)?);
            composed_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        } else {
            let start = Instant::now();
            composed_last = Some(run(&mut composed)?);
            composed_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
            let start = Instant::now();
            fused_last = Some(run(&mut fused)?);
            fused_samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        }
    }
    let fused_last = fused_last.context("no fused benchmark sample")?;
    let composed_last = composed_last.context("no composed benchmark sample")?;

    let fused_timing = timing("fused", fused_samples);
    let composed_timing = timing("composed", composed_samples);
    let output = Output {
        device: fused.device_name().to_string(),
        width: fused.width(),
        state_size: fused.state_size(),
        batch: case.batch,
        fused_kernel_marker: fused.packed_channel_mix_forward_fused(),
        composed_kernel_marker: !composed.packed_channel_mix_forward_fused(),
        max_abs_output: max_abs(&fused_last.output, &composed_last.output),
        max_abs_packed_state: max_abs(
            &fused_last.packed_new_state,
            &composed_last.packed_new_state,
        ),
        composed_over_fused: composed_timing.median_ms / fused_timing.median_ms,
        fused: fused_timing,
        composed: composed_timing,
    };
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
