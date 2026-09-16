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
    state_mode: String,
    state_clamp: f32,
    shadow: BranchCase,
    committed: BranchCase,
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
    optimizer_step: u32,
    optimizer_tensor_count: usize,
    tied_embedding_optimizer_step: Option<u32>,
    committed_outputs: Vec<f32>,
    committed_final_packed_state: Vec<f32>,
    committed_grad_x: Vec<f32>,
    committed_token_feature_grad: Vec<f32>,
    committed_grad_initial_packed_state: Vec<f32>,
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
    let case_path = case_path.context("fork parity runner requires --case")?;
    let model_dir = model_dir.context("fork parity runner requires --model-dir")?;
    let cell_prefix = cell_prefix.context("fork parity runner requires --cell-prefix")?;
    let adapter_prefix = adapter_prefix.context("fork parity runner requires --adapter-prefix")?;
    let case: Case = serde_json::from_slice(&fs::read(case_path)?)?;
    let state_mode = match case.state_mode.as_str() {
        "legacy-input-cache" => RwkvStateReadoutMode::LegacyInputCache,
        "explicit-output" => RwkvStateReadoutMode::ExplicitOutput,
        other => anyhow::bail!("unknown packed state mode {other:?}"),
    };
    let max_batch = case.shadow.batch.max(case.committed.batch);
    let max_steps = case.shadow.steps.max(case.committed.steps);
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
            "fork parity case width {} does not match loaded width {}",
            case.width,
            op.width()
        );
    }

    let shadow_schedule = RwkvTbpttSchedule::new(case.shadow.detach_every_n_steps)?;
    let committed_schedule = RwkvTbpttSchedule::new(case.committed.detach_every_n_steps)?;
    let shadow = RwkvTbpttBranchInput {
        batch: case.shadow.batch,
        steps: case.shadow.steps,
        x_sequence: &case.shadow.x_sequence,
        token_id_sequence: &case.shadow.token_id_sequence,
        initial_packed_state: &case.shadow.initial_packed_state,
        grad_output_sequence: &case.shadow.grad_output_sequence,
        final_packed_state_grad: case.shadow.final_packed_state_grad.as_deref(),
        schedule: shadow_schedule,
    };
    let committed = RwkvTbpttBranchInput {
        batch: case.committed.batch,
        steps: case.committed.steps,
        x_sequence: &case.committed.x_sequence,
        token_id_sequence: &case.committed.token_id_sequence,
        initial_packed_state: &case.committed.initial_packed_state,
        grad_output_sequence: &case.committed.grad_output_sequence,
        final_packed_state_grad: case.committed.final_packed_state_grad.as_deref(),
        schedule: committed_schedule,
    };
    let hyper = AdamWHyperParams {
        lr: case.optimizer.lr,
        beta1: case.optimizer.beta1,
        beta2: case.optimizer.beta2,
        eps: case.optimizer.eps,
        weight_decay: case.optimizer.weight_decay,
    };
    let result = op.train_step_with_token_ids_forked_shadow_commit(shadow, committed, hyper)?;
    let tied_optimizer = result.tied_embedding_optimizer.as_ref();
    println!(
        "{}",
        serde_json::to_string(&Output {
            device: op.device_name().to_string(),
            optimizer_step: result.optimizer.step,
            optimizer_tensor_count: result.optimizer.tensor_count,
            tied_embedding_optimizer_step: tied_optimizer.map(|optimizer| optimizer.step),
            committed_outputs: result.sequence.outputs,
            committed_final_packed_state: result.sequence.final_packed_state,
            committed_grad_x: result.sequence.grad_x,
            committed_token_feature_grad: result.sequence.token_feature_grad,
            committed_grad_initial_packed_state: result.sequence.grad_initial_packed_state,
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
