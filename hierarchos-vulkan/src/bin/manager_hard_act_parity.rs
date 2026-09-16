use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use hierarchos_vulkan::{HierarchosManagerHardActInput, HierarchosTrainingGraph, VulkanDevice};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Case {
    batch: usize,
    steps: usize,
    h_residual_input: Vec<f32>,
    h_token_ids: Vec<u32>,
    h_initial_packed_state: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    device: String,
    queue_submissions: u32,
    halt_probabilities: Vec<f32>,
    selected_index: Vec<u32>,
    executed_steps: Vec<f32>,
    selected_output: Vec<f32>,
    selected_packed_state: Vec<f32>,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut case_path = None;
    let mut model_dir = None;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--case" => case_path = args.next().map(PathBuf::from),
            "--model-dir" => model_dir = args.next().map(PathBuf::from),
            other => anyhow::bail!("unknown argument {other:?}"),
        }
    }
    let case_path = case_path.context("manager hard-ACT parity runner requires --case")?;
    let model_dir = model_dir.context("manager hard-ACT parity runner requires --model-dir")?;
    let case: Case = serde_json::from_slice(&fs::read(case_path)?)?;

    let device = VulkanDevice::new()?;
    let mut graph = HierarchosTrainingGraph::from_model_package(
        device,
        &model_dir,
        case.batch,
        case.steps.max(1),
        1,
        case.batch,
    )?;
    let result =
        graph.run_manager_hard_act_candidates_one_submit(HierarchosManagerHardActInput {
            batch: case.batch,
            steps: case.steps,
            h_residual_input: &case.h_residual_input,
            h_token_ids: &case.h_token_ids,
            h_initial_packed_state: &case.h_initial_packed_state,
        })?;

    println!(
        "{}",
        serde_json::to_string(&Output {
            device: graph.summary().device,
            queue_submissions: result.queue_submissions,
            halt_probabilities: result.halt_probabilities,
            selected_index: result.selected_index,
            executed_steps: result.executed_steps,
            selected_output: result.selected_output,
            selected_packed_state: result.selected_packed_state,
        })?
    );
    Ok(())
}
