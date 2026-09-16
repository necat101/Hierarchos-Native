use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use hierarchos_vulkan::{
    AdamWHyperParams, RwkvStateReadoutMode, RwkvTbpttBranchInput, RwkvTbpttSchedule,
    RwkvTbpttSequenceOp, VulkanDevice,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct BranchCase {
    batch: usize,
    steps: usize,
    detach_every_n_steps: Option<usize>,
    x_sequence: Vec<f32>,
    token_id_sequence: Vec<u32>,
    initial_packed_state: Vec<f32>,
    grad_output_sequence: Vec<f32>,
    final_packed_state_grad: Option<Vec<f32>>,
}

#[derive(Deserialize)]
struct OptimizerCase {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
}

#[derive(Deserialize)]
struct Case {
    width: usize,
    head_size: usize,
    input_dim: usize,
    state_mode: String,
    state_clamp: f32,
    branches: Vec<BranchCase>,
    optimizer: OptimizerCase,
}

#[derive(Serialize)]
struct ParameterOutput {
    name: String,
    values: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    device: String,
    branch_count: usize,
    optimizer_step: u32,
    optimizer_tensor_count: usize,
    tied_embedding_optimizer_step: Option<u32>,
    tied_embedding_optimizer_tensor_count: Option<usize>,
    final_outputs: Vec<f32>,
    final_packed_state: Vec<f32>,
    final_grad_x: Vec<f32>,
    final_token_feature_grad: Vec<f32>,
    final_grad_initial_packed_state: Vec<f32>,
    parameters: Vec<ParameterOutput>,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut case_path = None;
    let mut model_dir = None;
    let mut cell_prefix = None;
    let mut adapter_prefix = None;
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
            other => anyhow::bail!("unknown argument {other:?}"),
        }
    }
    let case_path = case_path.context("branch accumulation runner requires --case")?;
    let model_dir = model_dir.context("branch accumulation runner requires --model-dir")?;
    let cell_prefix = cell_prefix.context("branch accumulation runner requires --cell-prefix")?;
    let adapter_prefix =
        adapter_prefix.context("branch accumulation runner requires --adapter-prefix")?;
    let case: Case = serde_json::from_slice(&fs::read(case_path)?)?;
    if case.branches.is_empty() {
        anyhow::bail!("branch accumulation case requires at least one branch");
    }
    if case.input_dim == 0 {
        anyhow::bail!("branch accumulation input_dim must be positive");
    }
    let state_mode = match case.state_mode.as_str() {
        "legacy-input-cache" => RwkvStateReadoutMode::LegacyInputCache,
        "explicit-output" => RwkvStateReadoutMode::ExplicitOutput,
        other => anyhow::bail!("unknown packed state mode {other:?}"),
    };
    let max_batch = case
        .branches
        .iter()
        .map(|branch| branch.batch)
        .max()
        .context("branch accumulation has no max batch")?;
    let max_steps = case
        .branches
        .iter()
        .map(|branch| branch.steps)
        .max()
        .context("branch accumulation has no max steps")?;

    let device = VulkanDevice::new()?;
    let mut op = RwkvTbpttSequenceOp::from_model_package_with_tied_embedding(
        device,
        &model_dir,
        &cell_prefix,
        &adapter_prefix,
        case.head_size,
        max_batch,
        max_steps,
        12.0,
        4.0,
        state_mode,
        case.state_clamp,
    )?;
    if op.width() != case.width {
        anyhow::bail!(
            "branch accumulation case width {} does not match loaded width {}",
            case.width,
            op.width()
        );
    }

    let schedules = case
        .branches
        .iter()
        .map(|branch| RwkvTbpttSchedule::new(branch.detach_every_n_steps))
        .collect::<Result<Vec<_>>>()?;
    let branches = case
        .branches
        .iter()
        .zip(&schedules)
        .map(|(branch, schedule)| RwkvTbpttBranchInput {
            batch: branch.batch,
            steps: branch.steps,
            x_sequence: &branch.x_sequence,
            token_id_sequence: &branch.token_id_sequence,
            initial_packed_state: &branch.initial_packed_state,
            grad_output_sequence: &branch.grad_output_sequence,
            final_packed_state_grad: branch.final_packed_state_grad.as_deref(),
            schedule: *schedule,
        })
        .collect::<Vec<_>>();
    let hyper = AdamWHyperParams {
        lr: case.optimizer.lr,
        beta1: case.optimizer.beta1,
        beta2: case.optimizer.beta2,
        eps: case.optimizer.eps,
        weight_decay: case.optimizer.weight_decay,
    };
    let result = op.train_step_with_token_ids_accumulated_branches(&branches, hyper)?;
    let tied_optimizer = result.tied_embedding_optimizer.as_ref();

    println!(
        "{}",
        serde_json::to_string(&Output {
            device: op.device_name().to_string(),
            branch_count: branches.len(),
            optimizer_step: result.optimizer.step,
            optimizer_tensor_count: result.optimizer.tensor_count,
            tied_embedding_optimizer_step: tied_optimizer.map(|optimizer| optimizer.step),
            tied_embedding_optimizer_tensor_count: tied_optimizer
                .map(|optimizer| optimizer.tensor_count),
            final_outputs: result.sequence.outputs,
            final_packed_state: result.sequence.final_packed_state,
            final_grad_x: result.sequence.grad_x,
            final_token_feature_grad: result.sequence.token_feature_grad,
            final_grad_initial_packed_state: result.sequence.grad_initial_packed_state,
            parameters: result
                .parameters
                .into_iter()
                .map(|parameter| ParameterOutput {
                    name: parameter.name,
                    values: parameter.values,
                })
                .collect(),
        })?
    );
    Ok(())
}
