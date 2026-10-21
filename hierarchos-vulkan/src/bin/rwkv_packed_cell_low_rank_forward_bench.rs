//! Rank 2 of `validation/FUSED_FORWARD_PRODUCER_AUDIT.md` as one three-leg A/B on
//! the shared harness: the fused LN1/low-rank producer against the staged chain it
//! replaces, in the dense packed-cell forward.
//!
//! The two graphs:
//!
//! * **fused** — the LN1/low-rank producer keeps one row's w/a/g rank activations
//!   resident through the layer norm, the activation, the second-stage matmuls,
//!   decay and the final gate, so the whole low-rank forward is one dispatch. This
//!   is the structural default and
//!   `HIERARCHOS_RWKV_LOW_RANK_FORCE_FUSED_FORWARD_CHAIN` holds a leg on it;
//! * **composed** — the staged chain: a standalone layer norm, then the
//!   interpolation, the three first-stage matmuls, the activations and the
//!   second-stage matmuls as separate dispatches. This is what
//!   `HIERARCHOS_RWKV_LOW_RANK_FORCE_STAGED_FORWARD_CHAIN` forces, and it is the
//!   graph the production default falls back to where a measured record entry says
//!   it wins.
//!
//! `bench_ab` owns the measurement shape: three legs that differ only in switches
//! resolved at construction, interleaved rotated sampling, an agreement gate over
//! every returned tensor, a dispatch count proving which graph each leg ran, and a
//! fail-closed check that the shipped default still follows
//! `validation/low_rank_full_forward_decisions.json`.
//!
//! Both legs are complete GPU shader sequences; the record only chooses which one
//! the default dispatches, and no arithmetic moves to the host.

use std::{fs, path::PathBuf};

use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{
    bench_ab::{
        require_agreement, require_default_matches_route_decision, require_forward_arm_dispatches,
        require_separated_legs, sample_three_legs, tensor_diff, EnvSwitch, LegReport, Timing,
    },
    RwkvPackedCellOp, RwkvPackedCellResult, RwkvStateReadoutMode, VulkanDevice,
};
use serde::{Deserialize, Serialize};

/// Mirrors `rwkv_low_rank`'s switch constants. Those are part of the library's
/// documented surface, but the harness repeats the literals so an A/B leg cannot
/// silently follow a renamed switch.
const FORCE_FUSED_ENV: &str = "HIERARCHOS_RWKV_LOW_RANK_FORCE_FUSED_FORWARD_CHAIN";
const FORCE_STAGED_ENV: &str = "HIERARCHOS_RWKV_LOW_RANK_FORCE_STAGED_FORWARD_CHAIN";
/// Turns on the route's dispatch counter, which is off in production.
const COUNT_DISPATCHES_ENV: &str = "HIERARCHOS_RWKV_LOW_RANK_COUNT_FORWARD_DISPATCHES";

/// The producer each leg is expected to dispatch. Recorded in the receipt so a
/// promotion always names the graph it was measured against.
const FUSED_ARM: &str = "layernorm-low-rank-producer";
const COMPOSED_ARM: &str = "staged-layer-norm-interpolation-matmul-chain";

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
/// `vendor_dispatches`; this adapter renames it to what it actually counts, so the
/// receipt and its tests read one meaning.
#[derive(Serialize)]
struct Leg {
    label: &'static str,
    low_rank_forward_dispatches: u64,
    timing: Timing,
}

impl From<LegReport> for Leg {
    fn from(leg: LegReport) -> Self {
        Self {
            label: leg.label,
            low_rank_forward_dispatches: leg.vendor_dispatches,
            timing: leg.timing,
        }
    }
}

#[derive(Serialize)]
struct Output {
    device: String,
    width: usize,
    state_size: usize,
    batch: usize,
    fused_arm: &'static str,
    composed_arm: &'static str,
    /// The compiled-in decision this adapter's record names for this geometry, or
    /// `null` where the record has no entry and the structural default decides.
    recorded_compose: Option<bool>,
    fused: Leg,
    composed: Leg,
    default: Leg,
    composed_over_fused: f64,
    default_over_fused: f64,
    /// Worst per-tensor difference between the fused and composed legs. Unlike the
    /// channel-mix A/B these graphs are not bit-identical: the fused arm normalizes
    /// inside the producer while the staged arm runs the standalone layer-norm
    /// kernel, so the two reductions round differently. The tolerance is the
    /// contract; the tests state it rather than claiming exactness.
    worst_max_abs: f64,
    tensors: Vec<hierarchos_vulkan::bench_ab::TensorDiff>,
}

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

    // The counter is measurement-only and read at construction, so it is set once
    // for the whole process rather than per leg.
    std::env::set_var(COUNT_DISPATCHES_ENV, "1");

    let device = VulkanDevice::new()?;
    let mut fused = {
        let _forced = EnvSwitch::set(FORCE_FUSED_ENV);
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
        let _forced = EnvSwitch::set(FORCE_STAGED_ENV);
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
        device.clone(),
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

    // Two independent markers per leg: the label the op reports for this batch and
    // the dispatch count of the route, which proves what actually ran.
    let fused_label = fused.low_rank_forward_chain_label(case.batch);
    let composed_label = composed.low_rank_forward_chain_label(case.batch);
    let default_label = default.low_rank_forward_chain_label(case.batch);
    require_separated_legs(fused_label, composed_label)?;
    if fused_label != "fused" || composed_label != "composed" {
        bail!(
            "the switches did not separate the two graphs: fused leg built {fused_label}, \
             composed leg built {composed_label}"
        );
    }

    let before = fused.low_rank_forward_dispatches();
    step(&mut fused, &case)?;
    let fused_dispatches = fused.low_rank_forward_dispatches() - before;
    let before = composed.low_rank_forward_dispatches();
    step(&mut composed, &case)?;
    let composed_dispatches = composed.low_rank_forward_dispatches() - before;
    let before = default.low_rank_forward_dispatches();
    step(&mut default, &case)?;
    let default_dispatches = default.low_rank_forward_dispatches() - before;

    require_forward_arm_dispatches(fused_dispatches, composed_dispatches)?;
    // The default is checked against the decision it actually follows: the
    // compiled-in record where this adapter has an entry, the structural default
    // (the fused producer) otherwise.
    let recorded = default.recorded_low_rank_forward_chain_decision(case.batch);
    require_default_matches_route_decision(fused_label, recorded, default_label, case.batch)?;
    let expected_default_dispatches = match recorded {
        Some(true) => composed_dispatches,
        _ => fused_dispatches,
    };
    if default_dispatches != expected_default_dispatches {
        bail!(
            "the default leg reports {default_label} but dispatched the route \
             {default_dispatches} time(s) against the expected {expected_default_dispatches}"
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
        width: fused.width(),
        state_size: fused.state_size(),
        batch: case.batch,
        fused_arm: FUSED_ARM,
        composed_arm: COMPOSED_ARM,
        recorded_compose: recorded,
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
