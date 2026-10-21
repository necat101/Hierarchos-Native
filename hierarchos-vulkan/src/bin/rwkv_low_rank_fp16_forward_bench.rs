//! The rank-2 scope gap from `validation/FUSED_FORWARD_PRODUCER_AUDIT.md`: the
//! *deep* FP16 packed-mirror producer
//! (`low_rank_full_forward_fused_fp16_packed`) against the staged chain, under
//! the training graph's FP16 parameter-storage policy.
//!
//! The audited rank-2 receipt measures the packed cell, whose low-rank parameters
//! are FP32, so its fused arm is the portable first-stage producer. The deep arm
//! binds packed FP16 mirrors that only the training graph's
//! `fp16-storage-fp32-compute` policy installs. This binary therefore measures the
//! op the policy feeds (`RwkvLowRankOp`) with mirrors installed from its own FP32
//! masters, which switches the same route decision onto the deep producer while
//! leaving the FP32 masters as the canonical checkpoint identity.
//!
//! Three legs, one process, on the shared `bench_ab` harness:
//!
//! * **fused** — `record_forward` (pre-normalized input, FP16 mirrors) is one
//!   producer dispatch: the deep FP16 arm when its kernel exists, the FP16
//!   first-stage producer otherwise. `HIERARCHOS_RWKV_LOW_RANK_FORCE_FUSED_FORWARD_CHAIN`
//!   pins it;
//! * **composed** — the staged chain: interpolation, the three first-stage
//!   matmuls (packed FP16 operands), activation, the six second-stage
//!   dispatches. `HIERARCHOS_RWKV_LOW_RANK_FORCE_STAGED_FORWARD_CHAIN` pins it;
//! * **default** — the shipped route for this adapter and geometry, i.e. the
//!   compiled-in `validation/low_rank_full_forward_decisions.json` entry, or the
//!   fused arm where the record has none.
//!
//! The receipt answers the gap's question two ways: the arm name plus the deep
//! producer's own dispatch counter prove *which* FP16 graph ran, and the measured
//! ratio is compared with the record entry for the same geometry by the driver, so
//! "follows the same per-geometry route" is a checked claim rather than an
//! inference from the shared switch.
//!
//! Both legs are complete GPU shader sequences; the decision only chooses which
//! one the default dispatches, and no arithmetic moves to the host.

use std::{fs, path::PathBuf};

use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{
    bench_ab::{
        require_agreement, require_default_matches_route_decision, require_forward_arm_dispatches,
        require_separated_legs, sample_three_legs, tensor_diff, EnvSwitch, LegReport, Timing,
    },
    RwkvLowRankOp, RwkvLowRankResult, VulkanDevice,
};
use serde::{Deserialize, Serialize};

/// Mirrors `rwkv_low_rank`'s switch constants; repeated here so an A/B leg cannot
/// silently follow a renamed switch.
const FORCE_FUSED_ENV: &str = "HIERARCHOS_RWKV_LOW_RANK_FORCE_FUSED_FORWARD_CHAIN";
const FORCE_STAGED_ENV: &str = "HIERARCHOS_RWKV_LOW_RANK_FORCE_STAGED_FORWARD_CHAIN";
const COUNT_DISPATCHES_ENV: &str = "HIERARCHOS_RWKV_LOW_RANK_COUNT_FORWARD_DISPATCHES";

/// The deep arm this binary exists to measure.
const DEEP_ARM: &str = "low-rank-full-forward-fused-fp16-packed";
const COMPOSED_ARM: &str = "staged-low-rank-chain-fp16-packed-operands";

#[derive(Deserialize)]
struct Case {
    batch: usize,
    x_norm: Vec<f32>,
    previous: Vec<f32>,
    grad_a: Vec<f32>,
    grad_w: Vec<f32>,
    grad_g: Vec<f32>,
}

#[derive(Serialize)]
struct Leg {
    label: &'static str,
    /// The producer this leg actually bound. `composed` names the chain.
    arm: &'static str,
    low_rank_forward_dispatches: u64,
    /// Dispatches of the deep FP16 producer specifically; the route counter
    /// cannot tell it from the first-stage producer.
    deep_forward_arm_dispatches: u64,
    timing: Timing,
}

impl From<LegReport> for Leg {
    fn from(leg: LegReport) -> Self {
        Self {
            label: leg.label,
            arm: COMPOSED_ARM,
            low_rank_forward_dispatches: leg.vendor_dispatches,
            deep_forward_arm_dispatches: 0,
            timing: leg.timing,
        }
    }
}

#[derive(Serialize)]
struct Output {
    device: String,
    width: usize,
    w_rank: usize,
    a_rank: usize,
    g_rank: usize,
    batch: usize,
    /// The training-graph policy this run reproduces at the op level.
    precision_policy: &'static str,
    parameter_storage: &'static str,
    fused_arm: &'static str,
    composed_arm: &'static str,
    /// The compiled-in route decision for this geometry, or `null` where the
    /// record has no entry and the structural default (the fusion) decides.
    recorded_compose: Option<bool>,
    fused: Leg,
    composed: Leg,
    default: Leg,
    composed_over_fused: f64,
    default_over_fused: f64,
    /// Worst per-tensor difference between the fused and staged legs. The two
    /// graphs are not bit-identical (the deep producer keeps one row's rank
    /// activations resident and reads packed FP16 operands directly), so the
    /// tolerance is the contract.
    worst_max_abs: f64,
    tensors: Vec<hierarchos_vulkan::bench_ab::TensorDiff>,
}

fn step(op: &mut RwkvLowRankOp, case: &Case) -> Result<RwkvLowRankResult> {
    op.forward_backward(
        case.batch,
        &case.x_norm,
        &case.previous,
        &case.grad_a,
        &case.grad_w,
        &case.grad_g,
    )
}

fn build_op(
    device: VulkanDevice,
    model_dir: &PathBuf,
    prefix: &str,
    case: &Case,
) -> Result<RwkvLowRankOp> {
    let mut op = RwkvLowRankOp::from_model_package(device, model_dir, prefix, case.batch)?;
    op.enable_fp16_parameter_storage()?;
    if !op.fp16_parameter_storage_active() {
        bail!("FP16 parameter storage did not become active on the low-rank op");
    }
    Ok(op)
}

fn leg(leg: LegReport, label_arm: &'static str, deep: u64) -> Leg {
    Leg {
        arm: label_arm,
        deep_forward_arm_dispatches: deep,
        ..Leg::from(leg)
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut case_path = None;
    let mut model_dir = None;
    let mut prefix = None;
    let mut iterations = 30usize;
    let mut warmup = 5usize;
    let mut atol = 1.0e-4f64;
    let mut rtol = 1.0e-5f64;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--case" => case_path = args.next().map(PathBuf::from),
            "--model-dir" => model_dir = args.next().map(PathBuf::from),
            "--prefix" => {
                prefix = args
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
        "usage: --case CASE.json --model-dir MODEL_DIR --prefix h_rnn [--iterations 30] \
         [--warmup 5] [--atol 1e-4] [--rtol 1e-5]",
    )?;
    let model_dir = model_dir.context("benchmark requires --model-dir")?;
    let prefix = prefix.context("benchmark requires --prefix")?;
    let case: Case = serde_json::from_slice(&fs::read(case_path)?)?;

    // Measurement-only counter, read at construction: set once for the process.
    std::env::set_var(COUNT_DISPATCHES_ENV, "1");

    let device = VulkanDevice::new()?;
    let mut fused = {
        let _forced = EnvSwitch::set(FORCE_FUSED_ENV);
        build_op(device.clone(), &model_dir, &prefix, &case)?
    };
    let mut composed = {
        let _forced = EnvSwitch::set(FORCE_STAGED_ENV);
        build_op(device.clone(), &model_dir, &prefix, &case)?
    };
    let mut default = build_op(device.clone(), &model_dir, &prefix, &case)?;

    let fused_label = fused.forward_chain_label(case.batch);
    let composed_label = composed.forward_chain_label(case.batch);
    let default_label = default.forward_chain_label(case.batch);
    require_separated_legs(fused_label, composed_label)?;
    if fused_label != "fused" || composed_label != "composed" {
        bail!(
            "the switches did not separate the two graphs: fused leg built {fused_label}, \
             composed leg built {composed_label}"
        );
    }
    let fused_arm = fused
        .x_norm_forward_arm_label(case.batch)
        .context("the fused leg has no producer arm for this geometry")?;
    if fused_arm == COMPOSED_ARM {
        bail!("the fused leg named the staged chain as its arm");
    }

    let before = fused.low_rank_forward_dispatches();
    let deep_before = fused.deep_forward_arm_dispatches();
    step(&mut fused, &case)?;
    let fused_dispatches = fused.low_rank_forward_dispatches() - before;
    let fused_deep_dispatches = fused.deep_forward_arm_dispatches() - deep_before;
    let before = composed.low_rank_forward_dispatches();
    let deep_before = composed.deep_forward_arm_dispatches();
    step(&mut composed, &case)?;
    let composed_dispatches = composed.low_rank_forward_dispatches() - before;
    let composed_deep_dispatches = composed.deep_forward_arm_dispatches() - deep_before;
    let before = default.low_rank_forward_dispatches();
    let deep_before = default.deep_forward_arm_dispatches();
    step(&mut default, &case)?;
    let default_dispatches = default.low_rank_forward_dispatches() - before;
    let default_deep_dispatches = default.deep_forward_arm_dispatches() - deep_before;

    require_forward_arm_dispatches(fused_dispatches, composed_dispatches)?;
    // The deep arm is the point of this receipt: if the device can build it, the
    // fused leg must have run it, and no other leg may have.
    if fused_arm == DEEP_ARM && fused_deep_dispatches != 1 {
        bail!(
            "the fused leg named the deep FP16 arm but dispatched it {fused_deep_dispatches} \
             time(s) per forward; this run did not measure the arm it reports"
        );
    }
    if fused_arm != DEEP_ARM && fused_deep_dispatches != 0 {
        bail!("the fused leg ran the deep FP16 arm but reported {fused_arm}");
    }
    if composed_deep_dispatches != 0 {
        bail!(
            "the staged leg dispatched the deep FP16 producer {composed_deep_dispatches} time(s); \
             it did not run the chain"
        );
    }
    let recorded = default.recorded_forward_chain_decision(case.batch);
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
    if default_deep_dispatches > 1 {
        bail!("the default leg dispatched the deep FP16 producer {default_deep_dispatches} times");
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
        tensor_diff("output", &fused_last.a, &composed_last.a, atol, rtol),
        tensor_diff("w", &fused_last.w, &composed_last.w, atol, rtol),
        tensor_diff("g", &fused_last.g, &composed_last.g, atol, rtol),
        tensor_diff(
            "grad_x_norm",
            &fused_last.grad_x_norm,
            &composed_last.grad_x_norm,
            atol,
            rtol,
        ),
        tensor_diff(
            "grad_previous",
            &fused_last.grad_previous,
            &composed_last.grad_previous,
            atol,
            rtol,
        ),
        tensor_diff(
            "grad_w1",
            &fused_last.grad_w1,
            &composed_last.grad_w1,
            atol,
            rtol,
        ),
        tensor_diff(
            "grad_w2",
            &fused_last.grad_w2,
            &composed_last.grad_w2,
            atol,
            rtol,
        ),
        tensor_diff(
            "grad_a1",
            &fused_last.grad_a1,
            &composed_last.grad_a1,
            atol,
            rtol,
        ),
        tensor_diff(
            "grad_a2",
            &fused_last.grad_a2,
            &composed_last.grad_a2,
            atol,
            rtol,
        ),
        tensor_diff(
            "grad_g1",
            &fused_last.grad_g1,
            &composed_last.grad_g1,
            atol,
            rtol,
        ),
        tensor_diff(
            "grad_g2",
            &fused_last.grad_g2,
            &composed_last.grad_g2,
            atol,
            rtol,
        ),
    ];
    let worst_max_abs = require_agreement(&tensors, atol, rtol)?;

    let output = Output {
        device: fused.decision_device_name().to_owned(),
        width: fused.width(),
        w_rank: fused.ranks().0,
        a_rank: fused.ranks().1,
        g_rank: fused.ranks().2,
        batch: case.batch,
        precision_policy: "fp16-storage-fp32-compute",
        parameter_storage: "fp16-storage",
        fused_arm,
        composed_arm: COMPOSED_ARM,
        recorded_compose: recorded,
        composed_over_fused: composed_timing.median_ms / fused_timing.median_ms,
        default_over_fused: default_timing.median_ms / fused_timing.median_ms,
        worst_max_abs,
        tensors,
        fused: leg(
            LegReport::new(fused_label, fused_dispatches, fused_timing, 0.0),
            fused_arm,
            fused_deep_dispatches,
        ),
        composed: leg(
            LegReport::new(composed_label, composed_dispatches, composed_timing, 0.0),
            COMPOSED_ARM,
            composed_deep_dispatches,
        ),
        default: leg(
            LegReport::new(default_label, default_dispatches, default_timing, 0.0),
            if default_deep_dispatches > 0 {
                DEEP_ARM
            } else if default_label == "fused" {
                fused_arm
            } else {
                COMPOSED_ARM
            },
            default_deep_dispatches,
        ),
    };
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
