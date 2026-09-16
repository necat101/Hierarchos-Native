use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use hierarchos_vulkan::{
    AdamWHyperParams, RwkvNumericsPolicy, RwkvStateReadoutMode, RwkvTbpttSchedule,
    RwkvTbpttSequenceOp, VulkanDevice,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Case {
    batch: usize,
    steps: usize,
    width: usize,
    head_size: usize,
    input_dim: usize,
    state_mode: String,
    state_clamp: f32,
    detach_every_n_steps: Option<usize>,
    x_sequence: Vec<f32>,
    #[serde(default)]
    token_feature_sequence: Vec<f32>,
    #[serde(default)]
    token_id_sequence: Option<Vec<u32>>,
    initial_packed_state: Vec<f32>,
    grad_output_sequence: Vec<f32>,
    final_packed_state_grad: Option<Vec<f32>>,
    optimizer: Option<OptimizerCase>,
}

#[derive(Deserialize)]
struct OptimizerCase {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    updates: Option<usize>,
}

#[derive(Serialize)]
struct ParameterOutput {
    name: String,
    values: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    device: String,
    numerics_policy: String,
    backward_kernel_geometry: Option<String>,
    steps: usize,
    batch: usize,
    width: usize,
    state_size: usize,
    outputs: Vec<f32>,
    final_packed_state: Vec<f32>,
    grad_x: Vec<f32>,
    token_feature_grad: Vec<f32>,
    grad_initial_packed_state: Vec<f32>,
    optimizer_step: Option<u32>,
    optimizer_tensor_count: Option<usize>,
    tied_embedding_optimizer_step: Option<u32>,
    tied_embedding_optimizer_tensor_count: Option<usize>,
    parameters: Vec<ParameterOutput>,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut case_path = None;
    let mut model_dir = None;
    let mut output_model = None;
    let mut cell_prefix = None;
    let mut adapter_prefix = None;
    let mut kernel_geometry = None;
    let mut numerics_policy = RwkvNumericsPolicy::StrictParity;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--case" => case_path = args.next().map(PathBuf::from),
            "--model-dir" => model_dir = args.next().map(PathBuf::from),
            "--output-model" => output_model = args.next().map(PathBuf::from),
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
            "--kernel-geometry" => {
                kernel_geometry = args
                    .next()
                    .map(|value| value.to_string_lossy().into_owned())
            }
            "--numerics" => {
                let value = args
                    .next()
                    .context("--numerics requires strict|fast-subgroup|fast-recurrent-tree|fast-recurrent-tiled|fast-recurrent-subgroup")?;
                numerics_policy = match value.to_string_lossy().as_ref() {
                    "strict" | "strict-parity" => RwkvNumericsPolicy::StrictParity,
                    "fast-subgroup" => RwkvNumericsPolicy::FastSubgroup,
                    "fast-recurrent-tree" | "tree" => RwkvNumericsPolicy::FastRecurrentTree,
                    "fast-recurrent-tiled" | "tiled" => RwkvNumericsPolicy::FastRecurrentTiled,
                    "fast-recurrent-subgroup" | "recurrent-subgroup" | "subgroup-recurrent" => {
                        RwkvNumericsPolicy::FastRecurrentSubgroup
                    }
                    other => anyhow::bail!(
                        "unknown --numerics value {other:?}; expected strict|fast-subgroup|fast-recurrent-tree|fast-recurrent-tiled|fast-recurrent-subgroup"
                    ),
                };
            }
            other => anyhow::bail!("unknown argument {other:?}"),
        }
    }
    let case_path = case_path.context(
        "usage: --case CASE.json --model-dir MODEL_DIR --cell-prefix h_rnn --adapter-prefix h_deepembed_adapter",
    )?;
    let model_dir = model_dir.context("TBPTT runner requires --model-dir")?;
    let cell_prefix = cell_prefix.context("TBPTT runner requires --cell-prefix")?;
    let adapter_prefix = adapter_prefix.context("TBPTT runner requires --adapter-prefix")?;
    let case: Case = serde_json::from_slice(&fs::read(case_path)?)?;
    let state_mode = match case.state_mode.as_str() {
        "legacy-input-cache" => RwkvStateReadoutMode::LegacyInputCache,
        "explicit-output" => RwkvStateReadoutMode::ExplicitOutput,
        other => anyhow::bail!("unknown packed state mode {other:?}"),
    };
    let schedule = RwkvTbpttSchedule::new(case.detach_every_n_steps)?;

    let device = VulkanDevice::new()?;
    let use_token_ids = case.token_id_sequence.is_some();
    if use_token_ids && !case.token_feature_sequence.is_empty() {
        anyhow::bail!(
            "TBPTT case must provide token_id_sequence or token_feature_sequence, not both"
        );
    }
    if !use_token_ids && case.token_feature_sequence.is_empty() {
        anyhow::bail!("TBPTT case must provide token_id_sequence or token_feature_sequence");
    }
    let mut op = if use_token_ids {
        RwkvTbpttSequenceOp::from_model_package_with_tied_embedding(
            device,
            &model_dir,
            &cell_prefix,
            &adapter_prefix,
            case.head_size,
            case.batch,
            case.steps,
            12.0,
            4.0,
            state_mode,
            case.state_clamp,
        )?
    } else {
        RwkvTbpttSequenceOp::from_model_package(
            device,
            &model_dir,
            &cell_prefix,
            &adapter_prefix,
            case.head_size,
            case.batch,
            case.steps,
            12.0,
            4.0,
            state_mode,
            case.state_clamp,
        )?
    };
    if op.width() != case.width || case.input_dim == 0 {
        anyhow::bail!(
            "TBPTT case width/input_dim={}/{} does not match loaded width {}",
            case.width,
            case.input_dim,
            op.width()
        );
    }
    if let Some(kernel_geometry) = kernel_geometry.as_deref() {
        op.set_backward_kernel_geometry_label(case.batch, kernel_geometry)?;
    }
    op.set_numerics_policy(numerics_policy)?;
    let (
        result,
        optimizer_step,
        optimizer_tensor_count,
        tied_embedding_optimizer_step,
        tied_embedding_optimizer_tensor_count,
        parameters,
    ) = if let Some(optimizer) = case.optimizer {
        let updates = optimizer.updates.unwrap_or(1);
        if updates == 0 {
            anyhow::bail!("optimizer updates must be positive");
        }
        let hyper = AdamWHyperParams {
            lr: optimizer.lr,
            beta1: optimizer.beta1,
            beta2: optimizer.beta2,
            eps: optimizer.eps,
            weight_decay: optimizer.weight_decay,
        };
        let mut last_result = None;
        for _ in 0..updates {
            last_result = Some(if let Some(token_ids) = case.token_id_sequence.as_deref() {
                op.train_step_with_token_ids(
                    case.batch,
                    case.steps,
                    &case.x_sequence,
                    token_ids,
                    &case.initial_packed_state,
                    &case.grad_output_sequence,
                    case.final_packed_state_grad.as_deref(),
                    schedule,
                    hyper,
                )?
            } else {
                op.train_step(
                    case.batch,
                    case.steps,
                    &case.x_sequence,
                    &case.token_feature_sequence,
                    &case.initial_packed_state,
                    &case.grad_output_sequence,
                    case.final_packed_state_grad.as_deref(),
                    schedule,
                    hyper,
                )?
            });
        }
        let train_result = last_result.context("optimizer update loop produced no result")?;
        let tied_optimizer = train_result.tied_embedding_optimizer.as_ref();
        (
            train_result.sequence,
            Some(train_result.optimizer.step),
            Some(train_result.optimizer.tensor_count),
            tied_optimizer.map(|optimizer| optimizer.step),
            tied_optimizer.map(|optimizer| optimizer.tensor_count),
            train_result
                .parameters
                .into_iter()
                .map(|parameter| ParameterOutput {
                    name: parameter.name,
                    values: parameter.values,
                })
                .collect(),
        )
    } else {
        (
            if let Some(token_ids) = case.token_id_sequence.as_deref() {
                op.run_with_token_ids(
                    case.batch,
                    case.steps,
                    &case.x_sequence,
                    token_ids,
                    &case.initial_packed_state,
                    &case.grad_output_sequence,
                    case.final_packed_state_grad.as_deref(),
                    schedule,
                )?
            } else {
                op.run(
                    case.batch,
                    case.steps,
                    &case.x_sequence,
                    &case.token_feature_sequence,
                    &case.initial_packed_state,
                    &case.grad_output_sequence,
                    case.final_packed_state_grad.as_deref(),
                    schedule,
                )?
            },
            None,
            None,
            None,
            None,
            Vec::new(),
        )
    };
    if let Some(output_model) = output_model {
        op.export_model_package(&model_dir, output_model, &cell_prefix, &adapter_prefix)?;
    }
    println!(
        "{}",
        serde_json::to_string(&Output {
            device: op.device_name().to_string(),
            numerics_policy: op.numerics_policy().label().to_string(),
            backward_kernel_geometry: op
                .backward_kernel_geometry_label(case.batch)
                .map(str::to_owned),
            steps: result.steps,
            batch: result.batch,
            width: op.width(),
            state_size: op.state_size(),
            outputs: result.outputs,
            final_packed_state: result.final_packed_state,
            grad_x: result.grad_x,
            token_feature_grad: result.token_feature_grad,
            grad_initial_packed_state: result.grad_initial_packed_state,
            optimizer_step,
            optimizer_tensor_count,
            tied_embedding_optimizer_step,
            tied_embedding_optimizer_tensor_count,
            parameters,
        })?
    );
    Ok(())
}
