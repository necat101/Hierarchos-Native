use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use hierarchos_vulkan::{RwkvMatrixStateOp, VulkanDevice};
use serde::{Deserialize, Serialize};

const QUALIFICATION_DEVICE_INDEX_ENV: &str = "HIERARCHOS_VULKAN_QUALIFICATION_DEVICE_INDEX";

#[derive(Deserialize)]
struct Case {
    batch: usize,
    width: usize,
    head_size: usize,
    state: Vec<f32>,
    r: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    kk: Vec<f32>,
    a: Vec<f32>,
    w: Vec<f32>,
    grad_new_state: Vec<f32>,
    grad_tmix: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    device: String,
    heads: usize,
    new_state: Vec<f32>,
    tmix: Vec<f32>,
    grad_state: Vec<f32>,
    grad_r: Vec<f32>,
    grad_k: Vec<f32>,
    grad_v: Vec<f32>,
    grad_kk: Vec<f32>,
    grad_a: Vec<f32>,
    grad_w: Vec<f32>,
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
    let case: Case = serde_json::from_slice(&fs::read(&case_path)?)?;
    let device = qualification_device()?;
    let mut op = RwkvMatrixStateOp::new(device, case.width, case.head_size, case.batch)?;
    let result = op.forward_backward(
        case.batch,
        &case.state,
        &case.r,
        &case.k,
        &case.v,
        &case.kk,
        &case.a,
        &case.w,
        &case.grad_new_state,
        &case.grad_tmix,
    )?;
    println!(
        "{}",
        serde_json::to_string(&Output {
            device: op.device_name().to_string(),
            heads: op.heads(),
            new_state: result.new_state,
            tmix: result.tmix,
            grad_state: result.grad_state,
            grad_r: result.grad_r,
            grad_k: result.grad_k,
            grad_v: result.grad_v,
            grad_kk: result.grad_kk,
            grad_a: result.grad_a,
            grad_w: result.grad_w,
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
