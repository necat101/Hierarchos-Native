use std::{path::PathBuf, time::Instant};

use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{
    RwkvStateReadoutMode, RwkvTbpttSchedule, RwkvTbpttSequenceOp, VulkanDevice,
};
use serde::Serialize;

const DEFAULT_STEPS: usize = 4;
const DEFAULT_HEAD_SIZE: usize = 32;
const DEFAULT_BATCH: usize = 1;
const REPEATS: usize = 7;
const EXACT_PARITY_TOLERANCE: f32 = 0.0;

#[derive(Serialize)]
struct Output {
    device: String,
    width: usize,
    head_size: usize,
    batch: usize,
    steps: usize,
    packed_forward_only_active: bool,
    packed_backward_rematerialization_active: bool,
    output_max_abs_diff: f32,
    final_state_max_abs_diff: f32,
    grad_x_max_abs_diff: f32,
    token_feature_grad_max_abs_diff: f32,
    grad_initial_state_max_abs_diff: f32,
    legacy_median_ms: f64,
    packed_median_ms: f64,
    speedup: f64,
}

fn main() -> Result<()> {
    let (model_dir, cell_prefix, adapter_prefix, head_size, batch, steps) = parse_args()?;
    let device = VulkanDevice::new()?;
    let mut legacy = RwkvTbpttSequenceOp::from_model_package_with_tied_embedding(
        device.clone(),
        &model_dir,
        &cell_prefix,
        &adapter_prefix,
        head_size,
        batch,
        steps,
        12.0,
        4.0,
        RwkvStateReadoutMode::ExplicitOutput,
        50.0,
    )?;
    let mut packed = RwkvTbpttSequenceOp::from_model_package_with_tied_embedding(
        device,
        &model_dir,
        &cell_prefix,
        &adapter_prefix,
        head_size,
        batch,
        steps,
        12.0,
        4.0,
        RwkvStateReadoutMode::ExplicitOutput,
        50.0,
    )?;
    // Keep the true forward sweep identical in both arms so the timing delta
    // isolates reverse-pass matrix-unpack bandwidth. Only reverse
    // rematerialization differs between the two schedulers.
    legacy.set_packed_forward_only_enabled(true);
    packed.set_packed_forward_only_enabled(true);
    legacy.set_packed_backward_rematerialization_enabled(false);
    packed.set_packed_backward_rematerialization_enabled(true);
    let packed_forward_only_active = packed.packed_forward_only_active();
    let packed_backward_rematerialization_active =
        packed.packed_backward_rematerialization_active();
    if !packed_forward_only_active {
        bail!("packed first-forward path is unavailable on this Vulkan target");
    }
    if !packed_backward_rematerialization_active {
        bail!("packed backward rematerialization is unavailable on this Vulkan target");
    }

    let width = legacy.width();
    let vector_len = batch * width;
    let state_len = vector_len * legacy.state_size();
    let x = deterministic_values(steps * vector_len, 0.017, 3);
    let grad_output = deterministic_values(steps * vector_len, 0.009, 11);
    let initial_state = deterministic_values(state_len, 0.003, 17);
    let token_ids = vec![0u32; steps * batch];
    let schedule = RwkvTbpttSchedule::full_bptt();

    let legacy_result = legacy.run_with_token_ids(
        batch,
        steps,
        &x,
        &token_ids,
        &initial_state,
        &grad_output,
        None,
        schedule,
    )?;
    let packed_result = packed.run_with_token_ids(
        batch,
        steps,
        &x,
        &token_ids,
        &initial_state,
        &grad_output,
        None,
        schedule,
    )?;

    let output_max_abs_diff = max_abs_diff(&legacy_result.outputs, &packed_result.outputs)?;
    let final_state_max_abs_diff = max_abs_diff(
        &legacy_result.final_packed_state,
        &packed_result.final_packed_state,
    )?;
    let grad_x_max_abs_diff = max_abs_diff(&legacy_result.grad_x, &packed_result.grad_x)?;
    let token_feature_grad_max_abs_diff = max_abs_diff(
        &legacy_result.token_feature_grad,
        &packed_result.token_feature_grad,
    )?;
    let grad_initial_state_max_abs_diff = max_abs_diff(
        &legacy_result.grad_initial_packed_state,
        &packed_result.grad_initial_packed_state,
    )?;

    for (name, diff) in [
        ("output", output_max_abs_diff),
        ("final packed state", final_state_max_abs_diff),
        ("grad_x", grad_x_max_abs_diff),
        ("token-feature gradient", token_feature_grad_max_abs_diff),
        (
            "initial packed-state gradient",
            grad_initial_state_max_abs_diff,
        ),
    ] {
        if diff > EXACT_PARITY_TOLERANCE {
            bail!(
                "packed reverse rematerialization failed exact {name} parity: max abs diff {diff:e}"
            );
        }
    }

    // Benchmark only after exact output/state/gradient parity has passed.
    let mut legacy_ms = Vec::with_capacity(REPEATS);
    let mut packed_ms = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        legacy_ms.push(time_run(
            &mut legacy,
            batch,
            steps,
            &x,
            &token_ids,
            &initial_state,
            &grad_output,
            schedule,
        )?);
        packed_ms.push(time_run(
            &mut packed,
            batch,
            steps,
            &x,
            &token_ids,
            &initial_state,
            &grad_output,
            schedule,
        )?);
    }
    let legacy_median_ms = median(&mut legacy_ms);
    let packed_median_ms = median(&mut packed_ms);

    println!(
        "{}",
        serde_json::to_string(&Output {
            device: legacy.device_name().to_owned(),
            width,
            head_size,
            batch,
            steps,
            packed_forward_only_active,
            packed_backward_rematerialization_active,
            output_max_abs_diff,
            final_state_max_abs_diff,
            grad_x_max_abs_diff,
            token_feature_grad_max_abs_diff,
            grad_initial_state_max_abs_diff,
            legacy_median_ms,
            packed_median_ms,
            speedup: legacy_median_ms / packed_median_ms,
        })?
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn time_run(
    op: &mut RwkvTbpttSequenceOp,
    batch: usize,
    steps: usize,
    x: &[f32],
    token_ids: &[u32],
    initial_state: &[f32],
    grad_output: &[f32],
    schedule: RwkvTbpttSchedule,
) -> Result<f64> {
    let start = Instant::now();
    let _ = op.run_with_token_ids(
        batch,
        steps,
        x,
        token_ids,
        initial_state,
        grad_output,
        None,
        schedule,
    )?;
    Ok(start.elapsed().as_secs_f64() * 1_000.0)
}

fn deterministic_values(len: usize, scale: f32, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let value = ((index * 37 + seed * 17) % 101) as f32 - 50.0;
            value * scale / 50.0
        })
        .collect()
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.len() != right.len() {
        bail!(
            "parity vectors have different lengths: {} vs {}",
            left.len(),
            right.len()
        );
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max))
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn parse_args() -> Result<(PathBuf, String, String, usize, usize, usize)> {
    let mut args = std::env::args_os().skip(1);
    let mut model = None;
    let mut cell_prefix = "h_rnn".to_owned();
    let mut adapter_prefix = "h_deepembed_adapter".to_owned();
    let mut head_size = DEFAULT_HEAD_SIZE;
    let mut batch = DEFAULT_BATCH;
    let mut steps = DEFAULT_STEPS;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--model" => model = args.next().map(PathBuf::from),
            "--cell-prefix" => {
                cell_prefix = args
                    .next()
                    .context("--cell-prefix requires a value")?
                    .to_string_lossy()
                    .into_owned();
            }
            "--adapter-prefix" => {
                adapter_prefix = args
                    .next()
                    .context("--adapter-prefix requires a value")?
                    .to_string_lossy()
                    .into_owned();
            }
            "--head-size" => {
                head_size = args
                    .next()
                    .context("--head-size requires a value")?
                    .to_string_lossy()
                    .parse()
                    .context("--head-size must be a positive integer")?;
            }
            "--batch" => {
                batch = args
                    .next()
                    .context("--batch requires a value")?
                    .to_string_lossy()
                    .parse()
                    .context("--batch must be a positive integer")?;
            }
            "--steps" => {
                steps = args
                    .next()
                    .context("--steps requires a value")?
                    .to_string_lossy()
                    .parse()
                    .context("--steps must be a positive integer")?;
            }
            other => bail!("unknown argument {other:?}"),
        }
    }
    if head_size == 0 || batch == 0 || steps == 0 {
        bail!("--head-size, --batch, and --steps must be positive");
    }
    Ok((
        model.context("usage: --model MODEL_DIR [--head-size N] [--batch N] [--steps N]")?,
        cell_prefix,
        adapter_prefix,
        head_size,
        batch,
        steps,
    ))
}
