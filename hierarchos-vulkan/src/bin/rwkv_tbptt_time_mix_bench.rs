use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{
    bench_ab::{
        require_agreement, require_default_matches_decision, require_fused_leg_is_pure,
        require_separated_legs, sample_three_legs, tensor_diff, variant_dispatches, EnvSwitch,
        LegReport, Timing,
    },
    vendor_kernel_plan, RwkvStateReadoutMode, RwkvTbpttSchedule, RwkvTbpttSequenceOp,
    RwkvTbpttSequenceResult, VendorKernelFamily, VulkanDevice,
};
use serde::Serialize;

/// Mirrors `HIERARCHOS_RWKV_TIME_MIX_FORCE_FORWARD_FUSION_ENV` in
/// `rwkv_time_mix_core`. The env name is not part of the public API surface, so
/// the harness repeats the literal rather than depending on a private constant.
const TIME_MIX_FUSION_FORCE_ENV: &str = "HIERARCHOS_RWKV_TIME_MIX_FORCE_FORWARD_FUSION";
const TIME_MIX_FUSION_DISABLE_ENV: &str = "HIERARCHOS_VULKAN_DISABLE_TIME_MIX_FORWARD_FUSION";
const PROBE_FAMILY: &str = "linear3-forward";
const DEFAULT_STEPS: usize = 8;

/// The receipt's leg shape: the shared harness reports a generic
/// `vendor_dispatches` and a per-unit median, and this adapter keeps the
/// family-named key and `ms_per_step` the receipt and its tests already read.
#[derive(Serialize)]
struct Leg {
    label: &'static str,
    linear3_dispatches: u64,
    timing: Timing,
    ms_per_step: f64,
}

impl From<LegReport> for Leg {
    fn from(leg: LegReport) -> Self {
        Self {
            label: leg.label,
            linear3_dispatches: leg.vendor_dispatches,
            ms_per_step: leg.ms_per_unit.unwrap_or(0.0),
            timing: leg.timing,
        }
    }
}

#[derive(Serialize)]
struct Output {
    device: String,
    unmeasured_compose: bool,
    width: usize,
    state_size: usize,
    head_size: usize,
    batch: usize,
    steps: usize,
    linear3_variant: String,
    fused: Leg,
    composed: Leg,
    default: Leg,
    composed_over_fused: f64,
    default_over_fused: f64,
    /// Worst per-tensor difference between the fused and composed legs over the
    /// whole sequence. The graphs must agree within tolerance, or the timing
    /// compares different work.
    worst_max_abs: f64,
    tensors: Vec<hierarchos_vulkan::bench_ab::TensorDiff>,
    /// Which packed TBPTT chains the default sequence built. The projection
    /// receipt is measured on the shipped default, so its timing is only
    /// comparable across runs when these match the chain route records.
    default_packed_forward: bool,
    default_packed_rematerialization: bool,
}

#[allow(clippy::too_many_arguments)]
fn build_sequence(
    device: VulkanDevice,
    model_dir: &Path,
    cell_prefix: &str,
    adapter_prefix: &str,
    head_size: usize,
    batch: usize,
    steps: usize,
) -> Result<RwkvTbpttSequenceOp> {
    RwkvTbpttSequenceOp::from_model_package(
        device,
        model_dir,
        cell_prefix,
        adapter_prefix,
        head_size,
        batch,
        steps,
        12.0,
        4.0,
        RwkvStateReadoutMode::ExplicitOutput,
        50.0,
    )
}

/// One whole TBPTT window on a leg, so the harness closures stay one line each.
#[allow(clippy::too_many_arguments)]
fn step(
    op: &mut RwkvTbpttSequenceOp,
    batch: usize,
    steps: usize,
    x: &[f32],
    token_features: &[f32],
    initial_state: &[f32],
    grad_output: &[f32],
) -> Result<RwkvTbpttSequenceResult> {
    op.run(
        batch,
        steps,
        x,
        token_features,
        initial_state,
        grad_output,
        None,
        RwkvTbpttSchedule::full_bptt(),
    )
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut model_dir = None;
    let mut cell_prefix = None;
    let mut adapter_prefix = None;
    let mut head_size: Option<usize> = None;
    let mut input_dim: Option<usize> = None;
    let mut batches: Vec<usize> = Vec::new();
    let mut steps = DEFAULT_STEPS;
    let mut iterations = 10usize;
    let mut warmup = 2usize;
    let mut atol = 1.0e-4f64;
    let mut rtol = 1.0e-5f64;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
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
            "--head-size" => {
                head_size = Some(
                    args.next()
                        .context("--head-size needs a value")?
                        .to_string_lossy()
                        .parse()
                        .context("--head-size must be a positive integer")?,
                )
            }
            "--input-dim" => {
                input_dim = Some(
                    args.next()
                        .context("--input-dim needs a value")?
                        .to_string_lossy()
                        .parse()
                        .context("--input-dim must be a positive integer")?,
                )
            }
            "--batch" => batches.push(
                args.next()
                    .context("--batch needs a value")?
                    .to_string_lossy()
                    .parse()
                    .context("--batch must be a positive integer")?,
            ),
            "--steps" => {
                steps = args
                    .next()
                    .context("--steps needs a value")?
                    .to_string_lossy()
                    .parse()
                    .context("--steps must be a positive integer")?
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
    if steps == 0 {
        bail!("--steps must be positive");
    }
    if batches.is_empty() {
        batches.push(1);
    }
    let model_dir = model_dir.context("benchmark requires --model-dir")?;
    let cell_prefix = cell_prefix.context("benchmark requires --cell-prefix")?;
    let adapter_prefix = adapter_prefix.context("benchmark requires --adapter-prefix")?;
    let head_size = head_size.context("benchmark requires --head-size")?;
    let input_dim = input_dim.context("benchmark requires --input-dim")?;

    let device = VulkanDevice::new()?;
    let unmeasured_compose = hierarchos_vulkan::unmeasured_composition_is_qualified(device.vendor_id());
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

    for batch in batches {
        // The three legs differ only in the switches resolved at construction;
        // fixture, geometry, seed and weights are identical.
        let mut fused = {
            let _forced = EnvSwitch::set(TIME_MIX_FUSION_FORCE_ENV);
            build_sequence(
                device.clone(),
                &model_dir,
                &cell_prefix,
                &adapter_prefix,
                head_size,
                batch,
                steps,
            )?
        };
        let mut composed = {
            let _disabled = EnvSwitch::set(TIME_MIX_FUSION_DISABLE_ENV);
            build_sequence(
                device.clone(),
                &model_dir,
                &cell_prefix,
                &adapter_prefix,
                head_size,
                batch,
                steps,
            )?
        };
        let mut default = build_sequence(
            device.clone(),
            &model_dir,
            &cell_prefix,
            &adapter_prefix,
            head_size,
            batch,
            steps,
        )?;

        let width = fused.width();
        if fused.state_size() != composed.state_size() || width != composed.width() {
            bail!("the three legs built different state geometry");
        }
        let vector_len = batch * width;
        let state_len = vector_len * fused.state_size();
        let x = deterministic_values(steps * vector_len, 0.017, 3);
        let grad_output = deterministic_values(steps * vector_len, 0.009, 11);
        let initial_state = deterministic_values(state_len, 0.003, 17);
        // The DeepEmbed adapter input is held at zero, exactly as the packed-cell
        // A/B does, so the two harnesses time the same work per step.
        let token_features = vec![0.0f32; steps * batch * input_dim];

        // Labels first, then one instrumented run per leg so the dispatch
        // counter proves which graph actually ran.
        let fused_label = fused.time_mix_full_forward_projection_label(batch);
        let composed_label = composed.time_mix_full_forward_projection_label(batch);
        let default_label = default.time_mix_full_forward_projection_label(batch);
        require_separated_legs(fused_label, composed_label)?;
        if !fused_label.starts_with("fused") || composed_label != "composed" {
            bail!(
                "the switches did not separate the two graphs: fused leg built {fused_label}, \
                 composed leg built {composed_label}"
            );
        }

        let before = variant_dispatches(PROBE_FAMILY);
        let _ = step(
            &mut fused,
            batch,
            steps,
            &x,
            &token_features,
            &initial_state,
            &grad_output,
        )?;
        let fused_dispatches = variant_dispatches(PROBE_FAMILY) - before;
        let before = variant_dispatches(PROBE_FAMILY);
        let _ = step(
            &mut composed,
            batch,
            steps,
            &x,
            &token_features,
            &initial_state,
            &grad_output,
        )?;
        let composed_dispatches = variant_dispatches(PROBE_FAMILY) - before;
        let before = variant_dispatches(PROBE_FAMILY);
        let _ = step(
            &mut default,
            batch,
            steps,
            &x,
            &token_features,
            &initial_state,
            &grad_output,
        )?;
        let default_dispatches = variant_dispatches(PROBE_FAMILY) - before;

        require_fused_leg_is_pure(PROBE_FAMILY, fused_dispatches, steps)?;
        // The composed leg's count may be zero where the vendor gate is not
        // admitted; the label above is the proof it built the composition, and
        // the default is checked against the decision it actually follows.
        require_default_matches_decision(
            PROBE_FAMILY,
            fused_label,
            if unmeasured_compose { composed_dispatches } else { 0 },
            default.time_mix_recorded_projection_decision(batch),
            default_label,
            batch,
        )?;

        // Warm up and sample with the shared rotated order, so a warm cache
        // cannot favor one sequence graph.
        let (fused_timing, composed_timing, default_timing, fused_sampled, composed_sampled, _) =
            sample_three_legs(
                iterations,
                warmup,
                || {
                    step(
                        &mut fused,
                        batch,
                        steps,
                        &x,
                        &token_features,
                        &initial_state,
                        &grad_output,
                    )
                },
                || {
                    step(
                        &mut composed,
                        batch,
                        steps,
                        &x,
                        &token_features,
                        &initial_state,
                        &grad_output,
                    )
                },
                || {
                    step(
                        &mut default,
                        batch,
                        steps,
                        &x,
                        &token_features,
                        &initial_state,
                        &grad_output,
                    )
                },
            )?;
        let fused_last = fused_sampled;
        let composed_last = composed_sampled;

        let tensors = vec![
            tensor_diff(
                "outputs",
                &fused_last.outputs,
                &composed_last.outputs,
                atol,
                rtol,
            ),
            tensor_diff(
                "final_packed_state",
                &fused_last.final_packed_state,
                &composed_last.final_packed_state,
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
                "token_feature_grad",
                &fused_last.token_feature_grad,
                &composed_last.token_feature_grad,
                atol,
                rtol,
            ),
            tensor_diff(
                "grad_initial_packed_state",
                &fused_last.grad_initial_packed_state,
                &composed_last.grad_initial_packed_state,
                atol,
                rtol,
            ),
        ];
        let worst_max_abs = require_agreement(&tensors, atol, rtol)?;

        let output = Output {
            device: fused.decision_device_name().to_owned(),
            unmeasured_compose,
            width,
            state_size: fused.state_size(),
            head_size,
            batch,
            steps,
            linear3_variant: linear3_variant.clone(),
            composed_over_fused: composed_timing.median_ms / fused_timing.median_ms,
            default_over_fused: default_timing.median_ms / fused_timing.median_ms,
            worst_max_abs,
            tensors,
            fused: LegReport::new(fused_label, fused_dispatches, fused_timing, steps as f64).into(),
            composed: LegReport::new(
                composed_label,
                composed_dispatches,
                composed_timing,
                steps as f64,
            )
            .into(),
            default: LegReport::new(
                default_label,
                default_dispatches,
                default_timing,
                steps as f64,
            )
            .into(),
            default_packed_forward: default.packed_forward_only_active(),
            default_packed_rematerialization: default.packed_backward_rematerialization_active(),
        };
        println!("{}", serde_json::to_string(&output)?);
    }
    Ok(())
}

fn deterministic_values(len: usize, scale: f32, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|index| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / 16_777_216.0;
            (unit * 2.0 - 1.0) * scale * (1.0 + (index % 7) as f32 * 0.05)
        })
        .collect()
}
