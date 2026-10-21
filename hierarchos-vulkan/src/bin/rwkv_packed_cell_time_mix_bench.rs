use std::{fs, path::PathBuf};

use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{
    bench_ab::{
        require_agreement, require_default_matches_decision, require_fused_leg_is_pure,
        require_separated_legs, sample_three_legs, tensor_diff, variant_dispatches, EnvSwitch,
        LegReport, Timing,
    },
    vendor_kernel_plan, RwkvPackedCellOp, RwkvPackedCellResult, RwkvStateReadoutMode,
    VendorKernelFamily, VulkanDevice,
};
use serde::{Deserialize, Serialize};

/// Mirrors `HIERARCHOS_RWKV_TIME_MIX_FORCE_FORWARD_FUSION_ENV` in
/// `rwkv_time_mix_core`. The env name is not part of the public API surface, so
/// the harness repeats the literal rather than depending on a private constant.
const TIME_MIX_FUSION_FORCE_ENV: &str = "HIERARCHOS_RWKV_TIME_MIX_FORCE_FORWARD_FUSION";
const TIME_MIX_FUSION_DISABLE_ENV: &str = "HIERARCHOS_VULKAN_DISABLE_TIME_MIX_FORWARD_FUSION";
/// The tuned family only the composed leg may reach.
const PROBE_FAMILY: &str = "linear3-forward";

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

/// The receipt's leg shape. The shared harness reports a generic
/// `vendor_dispatches`; this adapter keeps the family-named key the receipt and
/// its tests already read.
#[derive(Serialize)]
struct Leg {
    label: &'static str,
    linear3_dispatches: u64,
    timing: Timing,
}

impl From<LegReport> for Leg {
    fn from(leg: LegReport) -> Self {
        Self {
            label: leg.label,
            linear3_dispatches: leg.vendor_dispatches,
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
    batch: usize,
    linear3_variant: String,
    /// The compiled-in decision this adapter's record names for this geometry, or
    /// `null` where the record has no entry and the vendor gate decides. The
    /// driver's cross-batch verdict follows this rather than a gate proxy.
    recorded_compose: Option<bool>,
    fused: Leg,
    composed: Leg,
    default: Leg,
    composed_over_fused: f64,
    default_over_fused: f64,
    /// Worst per-tensor difference between the fused and composed legs. Both
    /// graphs must agree within the harness tolerance, or the timing compares
    /// different work.
    worst_max_abs: f64,
    tensors: Vec<hierarchos_vulkan::bench_ab::TensorDiff>,
}

/// One `forward_backward` on a leg, so the harness closures stay one line each.
fn step(op: &mut RwkvPackedCellOp, case: &Case) -> Result<RwkvPackedCellResult> {
    op.forward_backward(
        case.batch,
        &case.x,
        &case.token_features,
        &case.packed_state,
        &case.grad_output,
        &case.grad_packed_new_state,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_op(
    device: VulkanDevice,
    case: &Case,
    model_dir: &PathBuf,
    cell_prefix: &str,
    adapter_prefix: &str,
    state_mode: RwkvStateReadoutMode,
) -> Result<RwkvPackedCellOp> {
    RwkvPackedCellOp::from_model_package(
        device,
        model_dir,
        cell_prefix,
        adapter_prefix,
        case.head_size,
        case.batch,
        12.0,
        4.0,
        state_mode,
        case.state_clamp,
    )
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut case_path = None;
    let mut model_dir = None;
    let mut cell_prefix = None;
    let mut adapter_prefix = None;
    let mut iterations = 30usize;
    let mut warmup = 5usize;
    let mut atol = 1.0e-4f64;
    let mut rtol = 1.0e-5f64;
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
    let case_path = case_path.context(
        "usage: --case CASE.json --model-dir MODEL_DIR --cell-prefix h_rnn \
         --adapter-prefix h_deepembed_adapter [--iterations 30] [--warmup 5] [--atol 1e-4] [--rtol 1e-5]",
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

    // The three legs differ only in the switches resolved at construction and
    // record time; the fixture, geometry and weights are identical.
    let mut fused = {
        let _forced = EnvSwitch::set(TIME_MIX_FUSION_FORCE_ENV);
        build_op(
            device.clone(),
            &case,
            &model_dir,
            &cell_prefix,
            &adapter_prefix,
            state_mode,
        )?
    };
    let mut composed = {
        let _disabled = EnvSwitch::set(TIME_MIX_FUSION_DISABLE_ENV);
        build_op(
            device.clone(),
            &case,
            &model_dir,
            &cell_prefix,
            &adapter_prefix,
            state_mode,
        )?
    };
    let mut default = build_op(
        device,
        &case,
        &model_dir,
        &cell_prefix,
        &adapter_prefix,
        state_mode,
    )?;

    for (name, op) in [
        ("fused", &fused),
        ("composed", &composed),
        ("default", &default),
    ] {
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

    // Two independent markers per leg: the label the op reports for this batch
    // and the vendor dispatch counter, which proves what actually ran.

    let fused_label = fused.time_mix_full_forward_projection_label(case.batch);
    let composed_label = composed.time_mix_full_forward_projection_label(case.batch);
    let default_label = default.time_mix_full_forward_projection_label(case.batch);
    require_separated_legs(fused_label, composed_label)?;
    if !fused_label.starts_with("fused") || composed_label != "composed" {
        bail!(
            "the switches did not separate the two graphs: fused leg built {fused_label}, \
             composed leg built {composed_label}"
        );
    }

    let before = variant_dispatches(PROBE_FAMILY);
    let _ = step(&mut fused, &case)?;
    let fused_dispatches = variant_dispatches(PROBE_FAMILY) - before;
    let before = variant_dispatches(PROBE_FAMILY);
    let _ = step(&mut composed, &case)?;
    let composed_dispatches = variant_dispatches(PROBE_FAMILY) - before;
    let before = variant_dispatches(PROBE_FAMILY);
    let _ = step(&mut default, &case)?;
    let default_dispatches = variant_dispatches(PROBE_FAMILY) - before;

    require_fused_leg_is_pure(PROBE_FAMILY, fused_dispatches, 1)?;
    // The composed leg's dispatch count may legitimately be zero: at geometry
    // where the vendor gate is not admitted, the composition runs the portable
    // module and the fused graph stays. The label (checked above) still proves
    // the leg built the composition, and the receipt records the count so a
    // sweep can require that at least one geometry reached the tuned module.
    // The default is checked against the decision it actually follows.
    require_default_matches_decision(
        PROBE_FAMILY,
        fused_label,
        if unmeasured_compose { composed_dispatches } else { 0 },
        default.time_mix_recorded_projection_decision(case.batch),
        default_label,
        case.batch,
    )?;
    if default_label == "composed" && default_dispatches != composed_dispatches {
        bail!(
            "the default leg reports {default_label} but dispatched {PROBE_FAMILY} \
             {default_dispatches} time(s) against the composed leg's {composed_dispatches}"
        );
    }

    let (fused_timing, composed_timing, default_timing, fused_last, composed_last, _default_last) =
        sample_three_legs(
            iterations,
            warmup,
            || step(&mut fused, &case),
            || step(&mut composed, &case),
            || step(&mut default, &case),
        )?;

    let tensors = vec![
        tensor_diff(
            "output",
            &fused_last.output,
            &composed_last.output,
            atol,
            rtol,
        ),
        tensor_diff(
            "packed_new_state",
            &fused_last.packed_new_state,
            &composed_last.packed_new_state,
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
            "grad_packed_state",
            &fused_last.grad_packed_state,
            &composed_last.grad_packed_state,
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
    ];
    let worst_max_abs = require_agreement(&tensors, atol, rtol)?;

    let output = Output {
        device: fused.decision_device_name().to_owned(),
        unmeasured_compose,
        width: fused.width(),
        state_size: fused.state_size(),
        batch: case.batch,
        linear3_variant,
        recorded_compose: default.time_mix_recorded_projection_decision(case.batch),
        composed_over_fused: composed_timing.median_ms / fused_timing.median_ms,
        default_over_fused: default_timing.median_ms / fused_timing.median_ms,
        worst_max_abs,
        tensors,
        fused: LegReport::new(fused_label, fused_dispatches, fused_timing, 0.0).into(),
        composed: LegReport::new(composed_label, composed_dispatches, composed_timing, 0.0).into(),
        default: LegReport::new(default_label, default_dispatches, default_timing, 0.0).into(),
    };
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
