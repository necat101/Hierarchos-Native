//! Rank 3 of `validation/FUSED_FORWARD_PRODUCER_AUDIT.md` as one three-leg A/B on
//! the shared harness: the fused key-transform/state-update kernel against the
//! two-dispatch chain it replaces.
//!
//! The graphs:
//!
//! * **wide** — `rwkv_key_state_forward_fused_wide` is the shipped fused arm: one
//!   workgroup covers `64 / head_size` heads, so the state phase keeps every lane
//!   busy instead of one head's first 16 lanes. It normalizes the key, applies the
//!   in-context scaling and the decay/w gate, and writes `new_state`, `tmix` and
//!   the saved `sa` in one dispatch;
//! * **narrow** — `rwkv_key_state_forward_fused`, the one-workgroup-per-head kernel
//!   the rank-3 measurement first recorded. It lost to the chain at batch 16 on
//!   Gen9, which is the crossover the wider group exists to remove;
//!   `HIERARCHOS_RWKV_FORCE_NARROW_KEY_STATE_FORWARD` pins it;
//! * **composed** — `key_transform_forward` then `state_forward`, two dispatches.
//!   `HIERARCHOS_RWKV_FORCE_STAGED_KEY_STATE_FORWARD` pins it and the record can
//!   withdraw the fusion for a measured geometry;
//! * **default** — the shipped route for this adapter and geometry, i.e.
//!   `validation/key_state_forward_decisions.json` where it has an entry and the
//!   fused arm everywhere else.
//!
//! The stage is measured in isolation rather than inside a whole time-mix step:
//! one dispatch against two is a sub-percent difference of the full core, so a
//! whole-step timing would report a tie for a structural question. Both fused arms
//! run the same shader sequence the shipped route dispatches, `bench_ab` rotates
//! all four legs through every position, and the op's own counter plus the arm
//! label prove which graph each leg ran (`dispatches_per_forward` is 1 for each
//! fused arm against 2 for the chain, measured, not assumed). Narrow and wide are
//! compared against each other as well as against the chain: the widening is only
//! worth shipping if it moves the crossover.
//!
//! No arithmetic moves to the host: the only host work is the fixture upload and
//! the final agreement readback.

use std::{fs, path::PathBuf};

use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{
    bench_ab::{
        require_agreement, require_default_matches_route_decision, require_separated_legs,
        sample_four_legs, tensor_diff, EnvSwitch, LegReport, Timing,
    },
    RwkvKeyStateForwardBenchmark, RwkvTimeMixCoreOp, VulkanDevice,
};
use serde::{Deserialize, Serialize};

/// Mirrors `rwkv_time_mix_core`'s switch constants; repeated so an A/B leg cannot
/// silently follow a renamed switch.
const FORCE_FUSED_ENV: &str = "HIERARCHOS_RWKV_FORCE_FUSED_KEY_STATE_FORWARD";
const FORCE_STAGED_ENV: &str = "HIERARCHOS_RWKV_FORCE_STAGED_KEY_STATE_FORWARD";
const FORCE_NARROW_ENV: &str = "HIERARCHOS_RWKV_FORCE_NARROW_KEY_STATE_FORWARD";
const FORCE_WIDE_ENV: &str = "HIERARCHOS_RWKV_FORCE_WIDE_KEY_STATE_FORWARD";
const COUNT_DISPATCHES_ENV: &str = "HIERARCHOS_RWKV_KEY_STATE_COUNT_FORWARD_DISPATCHES";

const FUSED_ARM: &str = "key-state-forward-fused-wide";
const NARROW_ARM: &str = "key-state-forward-fused";
const COMPOSED_ARM: &str = "key-transform-forward+state-forward";

#[derive(Deserialize)]
struct Case {
    batch: usize,
    head_size: usize,
    state: Vec<f32>,
    raw_k: Vec<f32>,
    a: Vec<f32>,
    w: Vec<f32>,
    r: Vec<f32>,
    v: Vec<f32>,
}

#[derive(Serialize)]
struct Leg {
    label: &'static str,
    arm: &'static str,
    /// Measured dispatches per forward, from the route's own counter.
    key_state_forward_dispatches: u64,
    timing: Timing,
    /// Median milliseconds of one forward, i.e. the repeated sample divided by
    /// the repetition count.
    ms_per_forward: f64,
}

#[derive(Serialize)]
struct Output {
    device: String,
    width: usize,
    head_size: usize,
    state_size: usize,
    batch: usize,
    repetitions: usize,
    fused_arm: &'static str,
    narrow_arm: &'static str,
    composed_arm: &'static str,
    /// The compiled-in route decision for this geometry, in the record's own
    /// vocabulary (`wide-fused`, `narrow-fused`, `staged`), or `null` where the
    /// record has no entry and the structural default (the wider group) decides.
    recorded_route: Option<&'static str>,
    /// The fused arm the record names for this geometry, or `null` where it
    /// measured no arm and the wider-group kernel decides.
    recorded_arm: Option<&'static str>,
    /// The arm the default leg actually built for this geometry, so a receipt can
    /// show the record's arm choice was followed and not merely parsed.
    default_arm: Option<&'static str>,
    /// The shipped fused arm (the wider-group kernel).
    fused: Leg,
    /// The narrow fused kernel it replaces, timed through the same rotation.
    narrow: Leg,
    composed: Leg,
    default: Leg,
    composed_over_fused: f64,
    default_over_fused: f64,
    /// Narrow over wide: above 1 means the wider group is faster at this geometry.
    narrow_over_fused: f64,
    composed_over_narrow: f64,
    /// Worst per-tensor difference between the shipped fused arm and the chain.
    worst_max_abs: f64,
    /// The same against the narrow arm; both fused arms are bit-identical to the
    /// chain, so this pins the widening as a scheduling change.
    worst_max_abs_wide_vs_narrow: f64,
    tensors: Vec<hierarchos_vulkan::bench_ab::TensorDiff>,
}

fn probe(
    op: &RwkvTimeMixCoreOp,
    case: &Case,
    repetitions: usize,
) -> Result<RwkvKeyStateForwardBenchmark> {
    op.benchmark_key_state_forward(
        case.batch,
        &case.state,
        &case.raw_k,
        &case.a,
        &case.w,
        &case.r,
        &case.v,
        1,
        repetitions,
    )
}

fn leg(
    label: &'static str,
    arm: &'static str,
    report: LegReport,
    dispatches: u64,
    repetitions: usize,
) -> Leg {
    Leg {
        label,
        arm,
        key_state_forward_dispatches: dispatches,
        ms_per_forward: report.timing.median_ms / repetitions as f64,
        timing: report.timing,
    }
}

fn build_op(
    device: VulkanDevice,
    model_dir: &PathBuf,
    prefix: &str,
    case: &Case,
) -> Result<RwkvTimeMixCoreOp> {
    RwkvTimeMixCoreOp::from_model_package(device, model_dir, prefix, case.head_size, case.batch)
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut case_path = None;
    let mut model_dir = None;
    let mut prefix = None;
    let mut iterations = 30usize;
    let mut repetitions = 50usize;
    let mut atol = 1.0e-5f64;
    let mut rtol = 1.0e-6f64;
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
            "--repetitions" => {
                repetitions = args
                    .next()
                    .context("--repetitions needs a value")?
                    .to_string_lossy()
                    .parse()
                    .context("--repetitions must be a positive integer")?
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
    if iterations == 0 || repetitions == 0 {
        bail!("--iterations and --repetitions must be positive");
    }
    let case_path = case_path.context(
        "usage: --case CASE.json --model-dir MODEL_DIR --prefix h_rnn [--iterations 30] \
         [--repetitions 50] [--atol 1e-5] [--rtol 1e-6]",
    )?;
    let model_dir = model_dir.context("benchmark requires --model-dir")?;
    let prefix = prefix.context("benchmark requires --prefix")?;
    let case: Case = serde_json::from_slice(&fs::read(case_path)?)?;

    // Measurement-only counter, read at construction: set once for the process.
    std::env::set_var(COUNT_DISPATCHES_ENV, "1");

    let device = VulkanDevice::new()?;
    let fused = {
        let _forced = EnvSwitch::set(FORCE_FUSED_ENV);
        let _wide = EnvSwitch::set(FORCE_WIDE_ENV);
        build_op(device.clone(), &model_dir, &prefix, &case)?
    };
    let narrow = {
        let _forced = EnvSwitch::set(FORCE_FUSED_ENV);
        let _narrow = EnvSwitch::set(FORCE_NARROW_ENV);
        build_op(device.clone(), &model_dir, &prefix, &case)?
    };
    let composed = {
        let _forced = EnvSwitch::set(FORCE_STAGED_ENV);
        build_op(device.clone(), &model_dir, &prefix, &case)?
    };
    let default = build_op(device.clone(), &model_dir, &prefix, &case)?;

    let fused_label = fused.key_state_forward_route_label(case.batch);
    let narrow_label = narrow.key_state_forward_route_label(case.batch);
    let composed_label = composed.key_state_forward_route_label(case.batch);
    let default_label = default.key_state_forward_route_label(case.batch);
    require_separated_legs(fused_label, composed_label)?;
    if fused_label != "fused" || composed_label != "composed" {
        bail!(
            "the switches did not separate the two graphs: fused leg built {fused_label}, \
             composed leg built {composed_label}"
        );
    }

    let fused_probe = probe(&fused, &case, repetitions)?;
    let narrow_probe = probe(&narrow, &case, repetitions)?;
    let composed_probe = probe(&composed, &case, repetitions)?;
    let default_probe = probe(&default, &case, repetitions)?;
    let fused_arm = fused
        .key_state_forward_arm_label(case.batch)
        .context("the fused leg has no key/state arm for this geometry")?;
    if fused_arm != FUSED_ARM {
        bail!("the fused leg named {fused_arm}, not the wider-group kernel {FUSED_ARM}");
    }
    let narrow_arm = narrow
        .key_state_forward_arm_label(case.batch)
        .context("the narrow leg has no key/state arm for this geometry")?;
    if narrow_arm != NARROW_ARM {
        bail!("the narrow leg named {narrow_arm}, not the narrow kernel {NARROW_ARM}");
    }
    let fused_dispatches = fused_probe.dispatches_per_forward.round() as u64;
    let narrow_dispatches = narrow_probe.dispatches_per_forward.round() as u64;
    let composed_dispatches = composed_probe.dispatches_per_forward.round() as u64;
    let default_dispatches = default_probe.dispatches_per_forward.round() as u64;
    if fused_dispatches != 1 || narrow_dispatches != 1 {
        bail!(
            "a fused key/state arm dispatched {fused_dispatches}/{narrow_dispatches} kernel(s) per \
             forward, but each producer is one dispatch; this run is not the graph the receipt names"
        );
    }
    if composed_dispatches != 2 {
        bail!(
            "the composed key/state arm dispatched {composed_dispatches} kernel(s) per forward, \
             but the chain is `key_transform_forward` + `state_forward`"
        );
    }
    let recorded = default.recorded_key_state_forward_decision(case.batch);
    require_default_matches_route_decision(fused_label, recorded, default_label, case.batch)?;
    let recorded_arm = default.recorded_key_state_forward_arm(case.batch);
    let default_arm = default.key_state_forward_arm_label(case.batch);
    if let Some(arm) = recorded_arm {
        let expected = if arm == "narrow" {
            NARROW_ARM
        } else {
            FUSED_ARM
        };
        if default_arm != Some(expected) {
            bail!(
                "the default leg built {default_arm:?} at batch {} but the record names the \
                 {arm:?} fused arm {expected:?}; the shipped default stopped following the \
                 measured arm",
                case.batch
            );
        }
    }
    // The default's own arm label says which fused kernel it built, so a pinned
    // narrow leg and the shipped wide arm are distinguished rather than both
    // accepted as "one dispatch".
    let expected_default_dispatches = if default_label == "composed" {
        composed_dispatches
    } else if default.key_state_forward_arm_label(case.batch) == Some(NARROW_ARM) {
        narrow_dispatches
    } else {
        fused_dispatches
    };
    if default_dispatches != expected_default_dispatches {
        bail!(
            "the default leg reports {default_label} but dispatched the route \
             {default_dispatches} time(s) against the expected {expected_default_dispatches}"
        );
    }

    // Each sample is `repetitions` dispatches in one submission; the rotated order
    // still interleaves whole samples, and the median is divided by the repetition
    // count so the receipt reports one forward. All four graphs rotate together so
    // the wide arm is timed under the same drift as the narrow one it replaced.
    let (
        fused_timing,
        narrow_timing,
        composed_timing,
        default_timing,
        fused_last,
        narrow_last,
        composed_last,
        _default_last,
    ) = sample_four_legs(
        iterations,
        1,
        || probe(&fused, &case, repetitions),
        || probe(&narrow, &case, repetitions),
        || probe(&composed, &case, repetitions),
        || probe(&default, &case, repetitions),
    )?;

    let tensors = vec![
        tensor_diff(
            "new_state",
            &fused_last.new_state,
            &composed_last.new_state,
            atol,
            rtol,
        ),
        tensor_diff("tmix", &fused_last.tmix, &composed_last.tmix, atol, rtol),
        tensor_diff(
            "scaled_k",
            &fused_last.scaled_k,
            &composed_last.scaled_k,
            atol,
            rtol,
        ),
        tensor_diff("kk", &fused_last.kk, &composed_last.kk, atol, rtol),
        tensor_diff(
            "saved_sa",
            &fused_last.saved_sa,
            &composed_last.saved_sa,
            atol,
            rtol,
        ),
    ];
    let worst_max_abs = require_agreement(&tensors, atol, rtol)?;
    // The widening may not change a value either: the wider group only moves which
    // lane advances which row, and the per-row products keep their order.
    let wide_vs_narrow = vec![
        tensor_diff(
            "new_state",
            &fused_last.new_state,
            &narrow_last.new_state,
            atol,
            rtol,
        ),
        tensor_diff("tmix", &fused_last.tmix, &narrow_last.tmix, atol, rtol),
        tensor_diff(
            "scaled_k",
            &fused_last.scaled_k,
            &narrow_last.scaled_k,
            atol,
            rtol,
        ),
        tensor_diff("kk", &fused_last.kk, &narrow_last.kk, atol, rtol),
        tensor_diff(
            "saved_sa",
            &fused_last.saved_sa,
            &narrow_last.saved_sa,
            atol,
            rtol,
        ),
    ];
    let worst_max_abs_wide_vs_narrow = require_agreement(&wide_vs_narrow, atol, rtol)?;

    let output = Output {
        device: fused.device_name().to_string(),
        width: fused.width(),
        head_size: case.head_size,
        state_size: case.batch * fused.width() * case.head_size,
        batch: case.batch,
        repetitions,
        fused_arm: FUSED_ARM,
        narrow_arm: NARROW_ARM,
        composed_arm: COMPOSED_ARM,
        recorded_route: default.recorded_key_state_forward_route_label(case.batch),
        recorded_arm,
        default_arm,
        composed_over_fused: composed_timing.median_ms / fused_timing.median_ms,
        default_over_fused: default_timing.median_ms / fused_timing.median_ms,
        narrow_over_fused: narrow_timing.median_ms / fused_timing.median_ms,
        composed_over_narrow: composed_timing.median_ms / narrow_timing.median_ms,
        worst_max_abs,
        worst_max_abs_wide_vs_narrow,
        tensors,
        narrow: leg(
            narrow_label,
            NARROW_ARM,
            LegReport::new(
                narrow_label,
                narrow_dispatches,
                narrow_timing,
                repetitions as f64,
            ),
            narrow_dispatches,
            repetitions,
        ),
        fused: leg(
            fused_label,
            FUSED_ARM,
            LegReport::new(
                fused_label,
                fused_dispatches,
                fused_timing,
                repetitions as f64,
            ),
            fused_dispatches,
            repetitions,
        ),
        composed: leg(
            composed_label,
            COMPOSED_ARM,
            LegReport::new(
                composed_label,
                composed_dispatches,
                composed_timing,
                repetitions as f64,
            ),
            composed_dispatches,
            repetitions,
        ),
        default: leg(
            default_label,
            if default_label == "fused" {
                default
                    .key_state_forward_arm_label(case.batch)
                    .unwrap_or(FUSED_ARM)
            } else {
                COMPOSED_ARM
            },
            LegReport::new(
                default_label,
                default_dispatches,
                default_timing,
                repetitions as f64,
            ),
            default_dispatches,
            repetitions,
        ),
    };
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
