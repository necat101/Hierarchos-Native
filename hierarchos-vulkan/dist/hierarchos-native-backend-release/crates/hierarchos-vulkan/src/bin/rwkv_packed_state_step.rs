use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use hierarchos_vulkan::{RwkvPackedStateOp, RwkvStateReadoutMode, VulkanDevice};
use serde::{Deserialize, Serialize};

const QUALIFICATION_DEVICE_INDEX_ENV: &str = "HIERARCHOS_VULKAN_QUALIFICATION_DEVICE_INDEX";

#[derive(Deserialize)]
struct Case {
    batch: usize,
    width: usize,
    head_size: usize,
    mode: String,
    state_clamp: f32,
    packed_state: Vec<f32>,
    x_norm: Vec<f32>,
    x_norm2: Vec<f32>,
    v_first: Vec<f32>,
    output: Vec<f32>,
    new_matrix_state: Vec<f32>,
    grad_packed_new_state: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    device: String,
    state_size: usize,
    previous_tm: Vec<f32>,
    previous_cm: Vec<f32>,
    previous_v_first: Vec<f32>,
    matrix_state: Vec<f32>,
    packed_new_state: Vec<f32>,
    grad_x_norm: Vec<f32>,
    grad_x_norm2: Vec<f32>,
    grad_v_first: Vec<f32>,
    grad_output: Vec<f32>,
    grad_matrix_state: Vec<f32>,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut case_path = None;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--case" => case_path = args.next().map(PathBuf::from),
            other => anyhow::bail!("unknown argument {other:?}"),
        }
    }
    let case_path = case_path.context("usage: --case CASE.json")?;
    let case: Case = serde_json::from_slice(&fs::read(case_path)?)?;
    let mode = match case.mode.as_str() {
        "legacy-input-cache" => RwkvStateReadoutMode::LegacyInputCache,
        "explicit-output" => RwkvStateReadoutMode::ExplicitOutput,
        other => anyhow::bail!("unknown packed state mode {other:?}"),
    };
    let device = qualification_device()?;
    let mut op = RwkvPackedStateOp::new(
        device,
        case.width,
        case.head_size,
        case.batch,
        mode,
        case.state_clamp,
    )?;
    let result = op.forward_backward(
        case.batch,
        &case.packed_state,
        &case.x_norm,
        &case.x_norm2,
        &case.v_first,
        &case.output,
        &case.new_matrix_state,
        &case.grad_packed_new_state,
    )?;
    println!(
        "{}",
        serde_json::to_string(&Output {
            device: op.device_name().to_string(),
            state_size: op.state_size(),
            previous_tm: result.previous_tm,
            previous_cm: result.previous_cm,
            previous_v_first: result.previous_v_first,
            matrix_state: result.matrix_state,
            packed_new_state: result.packed_new_state,
            grad_x_norm: result.grad_x_norm,
            grad_x_norm2: result.grad_x_norm2,
            grad_v_first: result.grad_v_first,
            grad_output: result.grad_output,
            grad_matrix_state: result.grad_matrix_state,
        })?
    );
    Ok(())
}

fn qualification_device() -> Result<VulkanDevice> {
    match std::env::var(QUALIFICATION_DEVICE_INDEX_ENV) {
        Ok(raw) => {
            let index = raw.parse::<usize>().with_context(|| {
                format!("{QUALIFICATION_DEVICE_INDEX_ENV} must be a non-negative device index")
            })?;
            VulkanDevice::new_with_index(index)
        }
        Err(std::env::VarError::NotPresent) => VulkanDevice::new(),
        Err(err) => Err(err).with_context(|| format!("reading {QUALIFICATION_DEVICE_INDEX_ENV}")),
    }
}
