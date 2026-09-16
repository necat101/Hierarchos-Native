use anyhow::{Context, Result};
use hierarchos_vulkan::{ContextDriftVulkanOp, HardActVulkanOp, VulkanDevice};
use serde::Serialize;

const QUALIFICATION_DEVICE_INDEX_ENV: &str = "HIERARCHOS_VULKAN_QUALIFICATION_DEVICE_INDEX";

#[derive(Serialize)]
struct ActOutput {
    halt_probabilities: Vec<f32>,
    selected_index: Vec<u32>,
    executed_steps: Vec<f32>,
    selected_output: Vec<f32>,
    grad_halt_logits: Vec<f32>,
    grad_step_outputs: Vec<f32>,
}

#[derive(Serialize)]
struct ContextOutput {
    output: Vec<f32>,
    grad_enc: Vec<f32>,
    grad_previous: Vec<f32>,
    grad_target: Vec<f32>,
    grad_drift: Vec<f32>,
}

#[derive(Serialize)]
struct DriftOutput {
    output: Vec<f32>,
    grad_current: Vec<f32>,
    grad_projected: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    device: String,
    act: ActOutput,
    context: ContextOutput,
    drift_seed: DriftOutput,
    drift_recurrence: DriftOutput,
}

fn main() -> Result<()> {
    let device = qualification_device()?;
    let mut act = HardActVulkanOp::new(device.clone(), 4, 3, 5)?;
    let halt_logits = vec![
        -4.0, -3.0, -2.0, 1.0, -0.7, 0.2, 0.5, 1.2, -0.5, -0.1, 3.0, 4.0,
    ];
    let step_outputs = deterministic_values(4 * 3 * 5, 0.37, 3);
    let grad_selected = deterministic_values(3 * 5, 0.29, 7);
    let grad_depth = vec![0.7, -0.4, 1.1];
    let act_result = act.forward_backward(
        &halt_logits,
        &step_outputs,
        &grad_selected,
        &grad_depth,
        4,
        3,
        2,
        0.72,
        0.11,
        3.0,
    )?;

    let mut context_op = ContextDriftVulkanOp::new(device.clone(), 5, 2)?;
    let enc = vec![0.2, -0.4, 0.1, 0.7, -0.3, -0.5, 0.6, 0.25, -0.15, 0.4];
    let previous = vec![0.75, -0.9, 0.2, 0.1, -0.4, -0.7, 0.45, 0.95, -0.2, 0.3];
    let target = vec![1.1, 0.6, -0.5, 1.4, 0.2, 0.8, -1.2, 0.3, -1.1, 0.9];
    let drift = vec![
        0.12, -0.08, 0.05, -0.14, 0.09, -0.11, 0.04, 0.13, -0.06, 0.02,
    ];
    let grad_concat = deterministic_values(2 * 10, 0.31, 11);
    let context_result = context_op.lerp_concat_forward_backward(
        &enc,
        &previous,
        &target,
        &drift,
        &grad_concat,
        2,
        0.375,
        0.8,
    )?;

    let current = vec![
        0.42, -0.51, 0.37, -0.28, 0.19, -0.33, 0.26, -0.44, 0.39, -0.21,
    ];
    let projected = vec![1.2, -0.7, 0.4, 1.8, -1.1, -0.9, 1.4, -1.7, 0.8, 0.55];
    let grad_drift_output = deterministic_values(10, 0.43, 17);
    let zeros = vec![0.0; 10];
    let drift_seed = context_op.drift_update_forward_backward(
        &zeros,
        &projected,
        &grad_drift_output,
        2,
        false,
        1.0,
        0.65,
        0.9,
    )?;
    let drift_recurrence = context_op.drift_update_forward_backward(
        &current,
        &projected,
        &grad_drift_output,
        2,
        true,
        0.4,
        0.65,
        0.9,
    )?;

    println!(
        "{}",
        serde_json::to_string(&Output {
            device: device.name().to_string(),
            act: ActOutput {
                halt_probabilities: act_result.halt_probabilities,
                selected_index: act_result.selected_index,
                executed_steps: act_result.executed_steps,
                selected_output: act_result.selected_output,
                grad_halt_logits: act_result.grad_halt_logits,
                grad_step_outputs: act_result.grad_step_outputs,
            },
            context: ContextOutput {
                output: context_result.output,
                grad_enc: context_result.grad_enc,
                grad_previous: context_result.grad_previous,
                grad_target: context_result.grad_target,
                grad_drift: context_result.grad_drift,
            },
            drift_seed: DriftOutput {
                output: drift_seed.output,
                grad_current: drift_seed.grad_current,
                grad_projected: drift_seed.grad_projected,
            },
            drift_recurrence: DriftOutput {
                output: drift_recurrence.output,
                grad_current: drift_recurrence.grad_current,
                grad_projected: drift_recurrence.grad_projected,
            },
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

fn deterministic_values(len: usize, scale: f32, phase: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let centered = ((index * 37 + phase * 19) % 101) as f32 - 50.0;
            centered * scale / 50.0
        })
        .collect()
}
