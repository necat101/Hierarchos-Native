//! A/B the two packed TBPTT chains against their legacy graphs.
//!
//! `rwkv_tbptt_packed_forward_parity` proved both packed chains bit-exact in one
//! run; this binary is the per-geometry measurement that writes their route
//! records. It measures one chain at a time so a receipt names one decision:
//!
//! * `--chain forward` times the packed first-forward transition (packed matrix
//!   history, backward-only tape stores suppressed) against the legacy transition
//!   that unpacks and copies between timesteps. Reverse rematerialization is
//!   held at its structural default (off) in both forced arms.
//! * `--chain remat` times packed reverse rematerialization (vector caches
//!   restored, packed matrix history consumed directly) against the legacy
//!   replay that unpacks the full matrix tape. The first forward stays packed in
//!   both arms.
//! * `--chain backward` times the reverse pass's state-backward fusion depth on
//!   the legacy rematerialization graph: `fused` keeps the autotuner's shipped
//!   schedule, `staged` swaps only that depth for the most decomposed executable
//!   chain (recurrence/add3 kernel, key transform and key-parameter reduction as
//!   separate dispatches). The projection order and low-rank fan-in are held
//!   fixed, so the delta is the fusion depth alone, and the first forward and
//!   reverse chains stay at their structural defaults. The receipt names the
//!   exact schedule each arm dispatched.
//!
//! The third leg is the shipped default built with no pinned switch, so its
//! recorded route is what `RwkvTbpttSequenceOp` selects for this batch. The
//! binary refuses to print a timing unless the forced legs really built
//! different chains, every returned tensor is bit-identical, and the default
//! followed the recorded route (or the structural default where none exists).
//!
//! The backward chain also reads the reduction's own output back after the
//! sequence: `grad_k_k`/`grad_k_a` are what the fold computes and what no
//! returned tensor consumes. The folded rung has to reproduce the two-dispatch
//! chain there exactly, so that comparison carries no allowance, and the staged
//! arm's gradient magnitude is recorded beside it so a zero-vs-zero agreement
//! cannot pass for parity. A single AdamW step on each leg is the second gate.
//!
//! `--negative-control` replaces the timing with a proof of those gates on both of
//! the fold's publication paths. Above batch 1 it seeds one phantom arrival per
//! batch row, so no arrival can complete a head and the fold never reduces; at
//! batch 1 the rung writes the gradients directly, with no counter to seed, so the
//! control redirects those writes into a scratch buffer instead. Each run builds a
//! clean pair and a corrupted pair in one process and requires the clean pair to
//! pass both gates while the corrupted pair fails them. The driver applies its own
//! receipt gates to the measurements.
//!
//! `--parameter-storage fp16-storage` installs the cell's packed FP16 execution
//! mirrors on all three legs before construction-time pins are applied. That is
//! the op-level analogue of the training graph's `fp16-storage-fp32-compute`
//! policy, whose trajectory golden was recorded on the structural packed route;
//! the receipt names the policy so the driver can record the verdict under its
//! own scope, and the run refuses to report a timing if the mirrors did not
//! activate (both arms would otherwise still read the FP32 masters).

use std::{path::PathBuf, time::Instant};

use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{
    bench_ab::{median, require_agreement, tensor_diff, TensorDiff},
    AdamWHyperParams, RwkvParameterSnapshot, RwkvStateReadoutMode, RwkvTbpttSchedule,
    RwkvTbpttSequenceOp, RwkvTbpttSequenceResult, VulkanDevice,
};
use serde::Serialize;

const DEFAULT_STEPS: usize = 8;
const DEFAULT_ITERATIONS: usize = 8;
const DEFAULT_WARMUP: usize = 2;
const FORWARD_ROUTE: &str = "packed";
const REMAT_ROUTE: &str = "legacy";
const FUSION_ROUTE: &str = "fused";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Storage {
    Fp32,
    Fp16,
}

impl Storage {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "fp32" => Ok(Self::Fp32),
            "fp16-storage" | "fp16-storage-fp32-compute" => Ok(Self::Fp16),
            other => bail!("--parameter-storage must be `fp32` or `fp16-storage`; got {other:?}"),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16-storage",
        }
    }

    fn precision_policy(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16-storage-fp32-compute",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Chain {
    Forward,
    Rematerialization,
    BackwardFusion,
}

impl Chain {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "forward" => Ok(Self::Forward),
            "remat" | "rematerialization" => Ok(Self::Rematerialization),
            "backward" | "backward-fusion" => Ok(Self::BackwardFusion),
            other => {
                bail!("--chain must be `forward`, `remat` or `backward`; got {other:?}")
            }
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Rematerialization => "remat",
            Self::BackwardFusion => "backward",
        }
    }

    fn structural_default(self) -> &'static str {
        match self {
            Self::Forward => FORWARD_ROUTE,
            Self::Rematerialization => REMAT_ROUTE,
            Self::BackwardFusion => FUSION_ROUTE,
        }
    }

    /// The arm forced off: the legacy chain, or the staged backward topology.
    fn staged_label(self) -> &'static str {
        match self {
            Self::BackwardFusion => "staged",
            _ => "legacy",
        }
    }

    /// The arm forced on: the packed chain, or the fused backward topology.
    fn packed_label(self) -> &'static str {
        match self {
            Self::BackwardFusion => "fused",
            _ => "packed",
        }
    }

    fn recorded_route(self, op: &RwkvTbpttSequenceOp, batch: usize) -> Option<&'static str> {
        match self {
            Self::Forward => op.recorded_packed_forward_route(batch),
            Self::Rematerialization => op.recorded_packed_rematerialization_route(batch),
            Self::BackwardFusion => op.recorded_packed_backward_fusion_route(batch),
        }
    }

    /// The route the op actually built for this chain, in the record's
    /// vocabulary. The backward-fusion arms are named by their own schedule
    /// labels, so they are classified by the caller.
    fn active_label(self, op: &RwkvTbpttSequenceOp, batch: usize) -> Option<&'static str> {
        match self {
            Self::Forward => Some(if op.packed_forward_only_active() {
                "packed"
            } else {
                "legacy"
            }),
            Self::Rematerialization => Some(if op.packed_backward_rematerialization_active() {
                "packed"
            } else {
                "legacy"
            }),
            Self::BackwardFusion => {
                let _ = batch;
                None
            }
        }
    }
}

#[derive(Serialize)]
struct Leg {
    label: &'static str,
    median_ms: f64,
    ms_per_step: f64,
}

/// A profile leg, which additionally carries the observed sample range. The
/// sweep's `Leg` keeps its shape so no receipt moves: only the profile, which
/// exists to decide whether a delta is measurable, needs the spread.
#[derive(Serialize)]
struct ProfileLeg {
    label: &'static str,
    median_ms: f64,
    ms_per_step: f64,
    min_ms: f64,
    max_ms: f64,
    /// The interquartile range, which ignores the occasional sample the clock or
    /// a context switch stretches. `median` sorts in place, so the quartiles are
    /// nearest-rank positions in the same sorted sample.
    iqr_ms: f64,
}

#[derive(Serialize)]
struct Output {
    device: String,
    width: usize,
    head_size: usize,
    batch: usize,
    steps: usize,
    chain: &'static str,
    parameter_storage: &'static str,
    precision_policy: &'static str,
    // The forward and rematerialization chains name their arms `legacy`/`packed`;
    // the backward-fusion chain names them `staged`/`fused`. Only the pair that
    // this chain measured is serialized, so each receipt names one decision.
    #[serde(skip_serializing_if = "Option::is_none")]
    legacy: Option<Leg>,
    #[serde(skip_serializing_if = "Option::is_none")]
    packed: Option<Leg>,
    #[serde(skip_serializing_if = "Option::is_none")]
    staged: Option<Leg>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fused: Option<Leg>,
    default: Leg,
    default_route: &'static str,
    recorded_route: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    packed_over_legacy: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fused_over_staged: Option<f64>,
    // The deepest compiled fusion rung, timed beside the routing pair. It is
    // evidence about the ladder, not a candidate a record may adopt: its
    // arithmetic is measured against the staged arm, not assumed identical.
    #[serde(skip_serializing_if = "Option::is_none")]
    folded: Option<Leg>,
    #[serde(skip_serializing_if = "Option::is_none")]
    folded_over_staged: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    folded_label: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    folded_schedule: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    folded_max_abs: Option<f64>,
    // The fold's own output, read back directly: `grad_k_k`/`grad_k_a` as the
    // reduction wrote them. No sequence tensor consumes them, so this is the
    // only tensor-level check of the reduction itself rather than of what the
    // loss happened to propagate.
    #[serde(skip_serializing_if = "Option::is_none")]
    folded_key_parameter_max_abs: Option<f64>,
    // The staged arm's own gradient magnitude, so the parity above cannot be a
    // vacuous zero-vs-zero agreement.
    #[serde(skip_serializing_if = "Option::is_none")]
    staged_key_parameter_max_abs: Option<f64>,
    // One AdamW step on the staged and folded legs, compared through the
    // parameter snapshots. A second gate behind the direct readback above, and a
    // weaker one: see `FOLDED_PARAMETER_ATOL`.
    #[serde(skip_serializing_if = "Option::is_none")]
    folded_parameter_delta: Option<f64>,
    // The exact backward-segment labels the two arms dispatched. The record's
    // vocabulary has two routes, but an arm's fusion depth and fan-in are part
    // of what was measured, so the receipt names them literally.
    #[serde(skip_serializing_if = "Option::is_none")]
    staged_schedule: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fused_schedule: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    default_schedule: Option<String>,
    // Every executable fusion depth at this geometry, most decomposed first.
    // Three entries mean the fusion ladder has three rungs here, not two.
    #[serde(skip_serializing_if = "Option::is_none")]
    available_schedules: Option<Vec<String>>,
    worst_max_abs: f64,
    tensors: Vec<TensorDiff>,
}

/// The three state-backward fusion rungs, most decomposed first.
const BACKWARD_FUSION_RUNGS: [&str; 3] = [
    "rkv-add3+key+reduce",
    "rkv-add3-key+reduce",
    "rkv-add3-key-reduce",
];

/// The fusion rung a schedule label names. The rungs share their prefix with
/// the projection/fan-in suffix, whose `+` separators make splitting unsafe.
fn backward_fusion_rung(label: &str) -> Option<&'static str> {
    BACKWARD_FUSION_RUNGS
        .into_iter()
        .find(|rung| label == *rung || label.starts_with(&format!("{rung}+")))
}

fn backward_fusion_suffix<'a>(label: &'a str, rung: &str) -> &'a str {
    &label[rung.len()..]
}

struct Args {
    chain: Chain,
    model_dir: PathBuf,
    cell_prefix: String,
    adapter_prefix: String,
    head_size: usize,
    input_dim: usize,
    batch: usize,
    steps: usize,
    iterations: usize,
    warmup: usize,
    storage: Storage,
    // Instead of timing, inject a broken arrival count into the fold and require
    // the parameter-step guard to reject it.
    negative_control: bool,
    // Time the folded rung against a leg whose reduction never fires, instead of
    // recording a receipt. See `profile_folded`.
    profile_folded: bool,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args_os().skip(1);
        let mut chain = None;
        let mut model_dir = None;
        let mut cell_prefix = "h_rnn".to_owned();
        let mut adapter_prefix = "h_deepembed_adapter".to_owned();
        let mut head_size = None;
        let mut input_dim = None;
        let mut batch = None;
        let mut steps = DEFAULT_STEPS;
        let mut iterations = DEFAULT_ITERATIONS;
        let mut warmup = DEFAULT_WARMUP;
        let mut storage = Storage::Fp32;
        let mut negative_control = false;
        let mut profile_folded = false;
        while let Some(arg) = args.next() {
            match arg.to_string_lossy().as_ref() {
                "--chain" => {
                    let text = next_value(&mut args, "--chain")?;
                    chain = Some(Chain::parse(text.to_string_lossy().as_ref())?);
                }
                "--model-dir" => {
                    model_dir = Some(PathBuf::from(next_value(&mut args, "--model-dir")?))
                }
                "--cell-prefix" => {
                    cell_prefix = next_value(&mut args, "--cell-prefix")?
                        .to_string_lossy()
                        .into_owned()
                }
                "--adapter-prefix" => {
                    adapter_prefix = next_value(&mut args, "--adapter-prefix")?
                        .to_string_lossy()
                        .into_owned()
                }
                "--head-size" => {
                    head_size = Some(parse_usize(
                        next_value(&mut args, "--head-size")?,
                        "--head-size",
                    )?)
                }
                "--input-dim" => {
                    input_dim = Some(parse_usize(
                        next_value(&mut args, "--input-dim")?,
                        "--input-dim",
                    )?)
                }
                "--batch" => {
                    batch = Some(parse_usize(next_value(&mut args, "--batch")?, "--batch")?)
                }
                "--parameter-storage" => {
                    let text = next_value(&mut args, "--parameter-storage")?;
                    storage = Storage::parse(text.to_string_lossy().as_ref())?;
                }
                "--steps" => steps = parse_usize(next_value(&mut args, "--steps")?, "--steps")?,
                "--iterations" => {
                    iterations =
                        parse_usize(next_value(&mut args, "--iterations")?, "--iterations")?
                }
                "--warmup" => warmup = parse_usize(next_value(&mut args, "--warmup")?, "--warmup")?,
                "--negative-control" => negative_control = true,
                "--profile-folded" => profile_folded = true,
                other => bail!("unknown argument {other:?}"),
            }
        }
        if steps == 0 || iterations == 0 {
            bail!("--steps and --iterations must be positive");
        }
        Ok(Self {
            chain: chain.context("--chain is required")?,
            model_dir: model_dir.context("--model-dir is required")?,
            cell_prefix,
            adapter_prefix,
            head_size: head_size.context("--head-size is required")?,
            input_dim: input_dim.context("--input-dim is required")?,
            batch: batch.context("--batch is required")?,
            steps,
            iterations,
            warmup,
            storage,
            negative_control,
            profile_folded,
        })
    }
}

fn next_value(
    args: &mut impl Iterator<Item = std::ffi::OsString>,
    flag: &str,
) -> Result<std::ffi::OsString> {
    args.next().with_context(|| format!("{flag} needs a value"))
}

fn parse_usize(value: std::ffi::OsString, flag: &str) -> Result<usize> {
    value
        .to_string_lossy()
        .parse()
        .with_context(|| format!("{flag} must be a non-negative integer"))
}

fn build(op_args: &Args, device: VulkanDevice) -> Result<RwkvTbpttSequenceOp> {
    RwkvTbpttSequenceOp::from_model_package(
        device,
        &op_args.model_dir,
        &op_args.cell_prefix,
        &op_args.adapter_prefix,
        op_args.head_size,
        op_args.batch,
        op_args.steps,
        12.0,
        4.0,
        RwkvStateReadoutMode::ExplicitOutput,
        50.0,
    )
}

/// Pin the chain under measurement to one forced leg. The other chains are held
/// at their structural defaults, so the delta between the two arms is one graph.
fn pin(op: &mut RwkvTbpttSequenceOp, chain: Chain, packed: bool, batch: usize) -> Result<()> {
    match chain {
        Chain::Forward => {
            op.set_packed_forward_only_enabled(packed);
            op.set_packed_backward_rematerialization_enabled(false);
        }
        Chain::Rematerialization => {
            op.set_packed_forward_only_enabled(true);
            op.set_packed_backward_rematerialization_enabled(packed);
        }
        Chain::BackwardFusion => {
            // Both backward arms keep the structural forward and reverse chains;
            // only the state-backward fusion depth differs.
            op.set_packed_forward_only_enabled(true);
            op.set_packed_backward_rematerialization_enabled(false);
            if packed {
                // The routing arm keeps the shipped depth, which is the only
                // depth a record may name. The deepest rung is timed as a
                // separate leg by `pin_folded`.
                op.set_packed_backward_fusion_enabled(batch, true)?;
            } else {
                op.set_packed_backward_fusion_enabled(batch, false)?;
            }
        }
    }
    Ok(())
}

/// Pin the fully folded fusion rung, which is a routing candidate: a record may
/// name it where the receipt below proves its key-parameter reduction exact. The
/// pin resolves the rung to this geometry's label instead of assuming one, and
/// fails rather than measuring a shallower graph under the folded name.
fn pin_folded(op: &mut RwkvTbpttSequenceOp, batch: usize) -> Result<()> {
    op.set_packed_forward_only_enabled(true);
    op.set_packed_backward_rematerialization_enabled(false);
    let label = op.folded_backward_schedule(batch)?.with_context(|| {
        format!(
            "batch {batch}: this device compiles no fully folded state-backward rung, so the \
             deepest depth cannot be measured"
        )
    })?;
    op.pin_packed_backward_fusion_schedule(batch, &label)
}

/// The folded rung is compared to the staged arm with a few-ulp allowance. On the
/// Intel Gen9 target the folded specialization's machine code contracts one
/// product of the state kernel differently from the two-dispatch chain, which
/// moves `grad_x` by one or two FP32 ulps; every other tensor is bit-identical,
/// and the receipt records the measured difference per batch. The routing pair
/// stays at zero tolerance, so no route is ever chosen on this allowance.
const FOLDED_ATOL: f64 = 1.0e-6;

/// The same allowance on the post-step parameters. AdamW's first step turns a
/// gradient change of a few ulps into a parameter change of the same order.
///
/// It is a coarse gate on purpose, and a weaker one than it looks: AdamW's first
/// update is sign-normalized, so a gradient whose magnitude moved but whose sign
/// held produces nearly the same update. A fold that never publishes its
/// gradients is caught here (its parameters do not move at all, `1.0e-4` away),
/// and a fold that loses one batch row may not be -- which is why the fold's own
/// `grad_k_k`/`grad_k_a` are read back and compared exactly instead.
const FOLDED_PARAMETER_ATOL: f64 = 1.0e-6;

/// The one gate the folded rung must clear: after an identical AdamW step, the
/// two legs' parameters must agree. The sweep and `--negative-control` call this
/// same function, so the control proves the guard the receipt rests on rejects a
/// broken fold rather than a second copy of it.
fn require_folded_parameter_parity(batch: usize, delta: f64) -> Result<()> {
    if delta > FOLDED_PARAMETER_ATOL {
        bail!(
            "batch {batch}: one AdamW step on the folded leg moved a parameter by {delta:.3e}, \
             beyond the {FOLDED_PARAMETER_ATOL:.1e} ulp allowance; the cross-batch fold is not \
             reproducing the shipped reduction"
        );
    }
    Ok(())
}

/// The two key-transform parameters the cross-batch fold reduces, with the
/// gradient tensor each one's reduction writes.
const KEY_PARAMETER_GRADIENTS: [(&str, &str); 2] = [("k_k", "grad_k_k"), ("k_a", "grad_k_a")];

/// Measure the fold's key-parameter reduction against the two-dispatch chain at
/// the tensor it writes, before any optimizer step. Both legs must have just run
/// the identical sequence, so their `grad_k_k`/`grad_k_a` hold what each
/// reduction produced on its last timestep.
fn key_parameter_gradient_delta(
    staged: &RwkvTbpttSequenceOp,
    folded: &RwkvTbpttSequenceOp,
) -> Result<f64> {
    let names: Vec<&str> = KEY_PARAMETER_GRADIENTS
        .iter()
        .map(|(name, _)| *name)
        .collect();
    let staged = staged.parameter_gradient_snapshots(&names)?;
    let folded = folded.parameter_gradient_snapshots(&names)?;
    let mut worst = 0.0f64;
    for (((_, label), staged), folded) in KEY_PARAMETER_GRADIENTS
        .into_iter()
        .zip(&staged)
        .zip(&folded)
    {
        if staged.name != folded.name {
            bail!(
                "the trained legs snapshot different gradients: {:?} and {:?}",
                staged.name,
                folded.name
            );
        }
        worst = worst.max(tensor_diff(label, &staged.values, &folded.values, 0.0, 0.0).max_abs);
    }
    Ok(worst)
}

/// The magnitude of the staging arm's own key-parameter gradients. A fold whose
/// tensor was never written would compare equal to anything, so the control
/// reports this and refuses to read a zero-vs-zero agreement as parity.
fn key_parameter_gradient_magnitude(op: &RwkvTbpttSequenceOp) -> Result<f64> {
    let names: Vec<&str> = KEY_PARAMETER_GRADIENTS
        .iter()
        .map(|(name, _)| *name)
        .collect();
    let snapshots = op.parameter_gradient_snapshots(&names)?;
    Ok(snapshots
        .iter()
        .flat_map(|snapshot| snapshot.values.iter())
        .map(|value| f64::from(value.abs()))
        .fold(0.0f64, f64::max))
}

/// The gate for that measurement. The folded kernel publishes the parameter
/// gradients the shipped reduction produces -- summing the batch rows in its own
/// ascending order above batch 1, writing them directly at batch 1 -- so its
/// `grad_k_k`/`grad_k_a` are the two-dispatch chain's bit for bit: no allowance is
/// needed, and none is given.
///
/// This is the check that catches a broken publication. The post-step parameters
/// cannot: AdamW's first step is sign-normalized, so a fold that drops a batch
/// row changes a gradient's magnitude without changing the update it produces.
const FOLDED_KEY_PARAMETER_ATOL: f64 = 0.0;

fn require_folded_key_parameter_parity(batch: usize, delta: f64) -> Result<()> {
    if delta > FOLDED_KEY_PARAMETER_ATOL {
        bail!(
            "batch {batch}: the folded rung moved grad_k_k/grad_k_a by {delta:.3e} away from \
             the two-dispatch chain; it must publish the gradients that reduction produces"
        );
    }
    Ok(())
}

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

/// Compare the parameter gradients through one identical AdamW step on each leg,
/// then the largest absolute difference between the resulting parameter
/// snapshots. `grad_x` is insensitive to the key-parameter fold, so this reaches
/// a broken `grad_k_k`/`grad_k_a` -- but only when the break moves the update,
/// which AdamW's sign normalization can absorb; `key_parameter_gradient_delta`
/// is the direct check.
#[allow(clippy::too_many_arguments)]
fn trained_parameter_delta(
    staged: &mut RwkvTbpttSequenceOp,
    folded: &mut RwkvTbpttSequenceOp,
    args: &Args,
    x: &[f32],
    token_features: &[f32],
    initial_state: &[f32],
    grad_output: &[f32],
) -> Result<f64> {
    let hyper = AdamWHyperParams::default();
    let staged = staged.train_step(
        args.batch,
        args.steps,
        x,
        token_features,
        initial_state,
        grad_output,
        None,
        RwkvTbpttSchedule::full_bptt(),
        hyper,
    )?;
    let folded = folded.train_step(
        args.batch,
        args.steps,
        x,
        token_features,
        initial_state,
        grad_output,
        None,
        RwkvTbpttSchedule::full_bptt(),
        hyper,
    )?;
    parameter_snapshot_delta(&staged.parameters, &folded.parameters)
}

fn parameter_snapshot_delta(
    staged: &[RwkvParameterSnapshot],
    folded: &[RwkvParameterSnapshot],
) -> Result<f64> {
    if staged.len() != folded.len() {
        bail!(
            "the trained legs snapshot {} and {} parameters",
            staged.len(),
            folded.len()
        );
    }
    let mut worst = 0.0f64;
    for (staged, folded) in staged.iter().zip(folded) {
        if staged.name != folded.name {
            bail!(
                "the trained legs snapshot different parameters: {:?} and {:?}",
                staged.name,
                folded.name
            );
        }
        if staged.values.len() != folded.values.len() {
            bail!(
                "parameter {:?} has {} values in the staged leg and {} in the folded one",
                staged.name,
                staged.values.len(),
                folded.values.len()
            );
        }
        for (staged, folded) in staged.values.iter().zip(&folded.values) {
            worst = worst.max(f64::from((staged - folded).abs()));
        }
    }
    Ok(worst)
}

/// Interleaved rotated sampling: each round starts one position later, so a
/// warm cache or clock drift cannot systematically favor one leg. The legs are
/// passed in one slice so the backward chain can add its folded rung without a
/// second sampling loop.
fn sample(
    ops: &mut [&mut RwkvTbpttSequenceOp],
    args: &Args,
    x: &[f32],
    token_features: &[f32],
    initial_state: &[f32],
    grad_output: &[f32],
) -> Result<Vec<f64>> {
    Ok(
        sample_raw(ops, args, x, token_features, initial_state, grad_output)?
            .into_iter()
            .map(|mut samples| median(&mut samples))
            .collect(),
    )
}

/// The same interleaved sampling without collapsing each leg to its median.
///
/// The folded rung's deltas are a few tenths of a millisecond on a step that
/// takes tens, so the profile needs the spread to say whether a delta is signal
/// at all; the sweep only ever reads medians and keeps using `sample`.
fn sample_raw(
    ops: &mut [&mut RwkvTbpttSequenceOp],
    args: &Args,
    x: &[f32],
    token_features: &[f32],
    initial_state: &[f32],
    grad_output: &[f32],
) -> Result<Vec<Vec<f64>>> {
    for _ in 0..args.warmup {
        for op in ops.iter_mut() {
            step(
                op,
                args.batch,
                args.steps,
                x,
                token_features,
                initial_state,
                grad_output,
            )?;
        }
    }
    let mut samples: Vec<Vec<f64>> = (0..ops.len())
        .map(|_| Vec::with_capacity(args.iterations))
        .collect();
    for round in 0..args.iterations {
        for position in 0..ops.len() {
            let slot = (position + round) % ops.len();
            let start = Instant::now();
            step(
                &mut *ops[slot],
                args.batch,
                args.steps,
                x,
                token_features,
                initial_state,
                grad_output,
            )?;
            samples[slot].push(start.elapsed().as_secs_f64() * 1_000.0);
        }
    }
    Ok(samples)
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

/// The arrival seed the control injects, as phantom arrivals per head: `batch`
/// of them, so no real arrival can be the head's last and its reduction never
/// runs. A smaller seed (one phantom arrival) makes the reduce fire a row early,
/// but whether the not-yet-arrived row's partial is already stored is a race; a
/// full batch of phantoms is deterministic, and it is the same bug taken to its
/// extreme -- the fold never publishes `grad_k_k`/`grad_k_a` at all.
fn negative_control_arrival_seed(batch: usize) -> String {
    batch.to_string()
}

/// How one `--negative-control` run breaks the folded rung, and the switch that
/// does it. The fold has two publication paths and they fail differently.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Corruption {
    /// Batch 1: the rung writes the parameter gradients directly, one row per
    /// channel, so it has no arrival count. Its publication step is the write, and
    /// the corruption redirects both writes into a scratch buffer.
    Batch1DirectWriteSink,
    /// Batch >= 2: a workgroup reduces a head when it observes that head's last
    /// arrival, so the corruption seeds a full batch of phantom arrivals and the
    /// reduction never runs.
    CrossBatchArrivalCount(usize),
}

impl Corruption {
    fn for_batch(batch: usize) -> Self {
        if batch == 1 {
            Self::Batch1DirectWriteSink
        } else {
            Self::CrossBatchArrivalCount(batch)
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Batch1DirectWriteSink => "batch-1-direct-write-sink",
            Self::CrossBatchArrivalCount(_) => "cross-batch-arrival-count",
        }
    }

    /// The environment switch this corruption is injected through. It has to be
    /// set before the corrupted legs are constructed and cleared after them.
    fn switch(self) -> (&'static str, String) {
        match self {
            Self::Batch1DirectWriteSink => (
                "HIERARCHOS_RWKV_KEY_PARAM_DIRECT_WRITE_SINK",
                "1".to_string(),
            ),
            Self::CrossBatchArrivalCount(batch) => (
                "HIERARCHOS_RWKV_CROSS_BATCH_ARRIVAL_SEED",
                negative_control_arrival_seed(batch),
            ),
        }
    }
}

#[derive(Serialize)]
struct NegativeControl {
    result: &'static str,
    negative_control: &'static str,
    device: String,
    width: usize,
    head_size: usize,
    batch: usize,
    steps: usize,
    /// Which publication path this run broke.
    corruption: &'static str,
    /// The phantom arrivals the corrupted fold observed, for the arrival-count
    /// regime. `None` at batch 1, which has no arrival count to seed.
    arrival_seed: Option<usize>,
    parameter_atol: f64,
    staged_schedule: Option<String>,
    folded_schedule: Option<String>,
    staged_key_parameter_max_abs: f64,
    clean_key_parameter_max_abs: f64,
    clean_parameter_delta: f64,
    corrupted_key_parameter_max_abs: f64,
    corrupted_parameter_delta: f64,
    clean_accepted: bool,
    corrupted_rejected: bool,
    /// Whether the post-step parameter gate also saw the corruption. It does not
    /// have to: the direct key-parameter gate is the one that must fire.
    post_step_guard_saw_the_corruption: bool,
    /// The gate's own message for the corrupted delta, quoted so the proof names
    /// the rejection instead of asserting one.
    rejection: String,
}

/// Prove the receipt's guards reject a broken fold, on both of its publication
/// paths.
///
/// Above batch 1 the fold is reachable only through its arrival protocol: a
/// workgroup reduces a head when it observes that head's last batch row, so
/// seeding a full batch of phantom arrivals makes the reduce never run. At batch
/// 1 the rung owns one row per channel and writes `grad_k_k`/`grad_k_a` directly,
/// with no counter at all, so there is nothing to seed: the equivalent bug is a
/// write that never reaches the gradient, and the corruption redirects both
/// bindings into a scratch buffer. Either way the arithmetic is untouched -- the
/// kernel still computes what it can -- and only the publication step is broken.
///
/// The control builds a clean pair and a corrupted pair in one process (each
/// switch is read when a cell is created), then requires the shared gates to
/// accept the clean measurements and reject the corrupted ones. A corruption the
/// guard misses is a failure, not a pass.
fn negative_control(args: &Args) -> Result<()> {
    if args.chain != Chain::BackwardFusion {
        bail!("--negative-control measures the folded rung; pass --chain backward");
    }
    let corruption = Corruption::for_batch(args.batch);
    std::env::set_var("HIERARCHOS_RWKV_CROSS_BATCH_KEY_PARAM_REDUCE", "1");
    std::env::remove_var("HIERARCHOS_RWKV_CROSS_BATCH_ARRIVAL_SEED");
    std::env::remove_var("HIERARCHOS_RWKV_KEY_PARAM_DIRECT_WRITE_SINK");
    let device = VulkanDevice::new()?;
    let mut staged = build(args, device.clone())?;
    let mut folded = build(args, device.clone())?;
    let (switch, value) = corruption.switch();
    std::env::set_var(switch, value);
    let mut corrupted_staged = build(args, device.clone())?;
    let mut corrupted_folded = build(args, device.clone())?;
    std::env::remove_var(switch);

    if args.storage == Storage::Fp16 {
        for op in [
            &mut staged,
            &mut folded,
            &mut corrupted_staged,
            &mut corrupted_folded,
        ] {
            op.enable_low_rank_fp16_parameter_storage()?;
        }
        if !staged.low_rank_fp16_parameter_storage_active() {
            bail!(
                "--parameter-storage fp16-storage did not activate the packed mirrors; the \
                 control would measure the FP32 masters"
            );
        }
    }
    pin(&mut staged, args.chain, false, args.batch)?;
    pin_folded(&mut folded, args.batch)?;
    pin(&mut corrupted_staged, args.chain, false, args.batch)?;
    pin_folded(&mut corrupted_folded, args.batch)?;

    let staged_schedule = staged.packed_backward_fusion_label(args.batch);
    let folded_schedule = folded.packed_backward_fusion_label(args.batch);
    let corrupted_staged_schedule = corrupted_staged.packed_backward_fusion_label(args.batch);
    let corrupted_folded_schedule = corrupted_folded.packed_backward_fusion_label(args.batch);
    for (arm, label) in [
        ("staged", staged_schedule.as_deref()),
        ("corrupted staged", corrupted_staged_schedule.as_deref()),
    ] {
        let label = label.context("the staged arm did not report its schedule")?;
        if backward_fusion_rung(label) != Some(BACKWARD_FUSION_RUNGS[0]) {
            bail!("the {arm} arm pinned {label:?}, not the fully staged chain");
        }
    }
    for (arm, label) in [
        ("folded", folded_schedule.as_deref()),
        ("corrupted folded", corrupted_folded_schedule.as_deref()),
    ] {
        let label = label.context("the folded leg did not report its schedule")?;
        if backward_fusion_rung(label) != Some(BACKWARD_FUSION_RUNGS[2]) {
            bail!("the {arm} leg pinned {label:?}, not the folded rung");
        }
    }
    // Only the arrival seed changed, so the four legs must dispatch the same two
    // graphs; a different schedule would mean the control measured something else.
    if staged_schedule != corrupted_staged_schedule || folded_schedule != corrupted_folded_schedule
    {
        bail!(
            "seeding the arrival counters changed the pinned schedules: staged \
             {staged_schedule:?} vs {corrupted_staged_schedule:?}, folded \
             {folded_schedule:?} vs {corrupted_folded_schedule:?}"
        );
    }

    let width = staged.width();
    let state_size = staged.state_size();
    let vector_len = args.batch * width;
    let state_len = vector_len * state_size;
    let x = deterministic_values(args.steps * vector_len, 0.017, 3);
    let grad_output = deterministic_values(args.steps * vector_len, 0.009, 11);
    let initial_state = deterministic_values(state_len, 0.003, 17);
    let token_features = vec![0.0f32; args.steps * args.batch * args.input_dim];

    for op in [
        &mut staged,
        &mut folded,
        &mut corrupted_staged,
        &mut corrupted_folded,
    ] {
        step(
            op,
            args.batch,
            args.steps,
            &x,
            &token_features,
            &initial_state,
            &grad_output,
        )?;
    }

    let staged_key_parameter_max_abs = key_parameter_gradient_magnitude(&staged)?;
    if staged_key_parameter_max_abs == 0.0 {
        bail!(
            "the staged arm's grad_k_k/grad_k_a are all zero, so a matching fold would prove \
             nothing; the control refuses to read that as parity"
        );
    }
    let clean_key_parameter_max_abs = key_parameter_gradient_delta(&staged, &folded)?;
    require_folded_key_parameter_parity(args.batch, clean_key_parameter_max_abs).with_context(
        || {
            "the clean pair disagrees on grad_k_k/grad_k_a, so the control cannot attribute the \
             rejection to the corrupted arrival count"
        },
    )?;
    let corrupted_key_parameter_max_abs =
        key_parameter_gradient_delta(&corrupted_staged, &corrupted_folded)?;
    let clean_parameter_delta = trained_parameter_delta(
        &mut staged,
        &mut folded,
        args,
        &x,
        &token_features,
        &initial_state,
        &grad_output,
    )?;
    let corrupted_parameter_delta = trained_parameter_delta(
        &mut corrupted_staged,
        &mut corrupted_folded,
        args,
        &x,
        &token_features,
        &initial_state,
        &grad_output,
    )?;

    require_folded_parameter_parity(args.batch, clean_parameter_delta).with_context(|| {
        "the clean pair diverged, so the control cannot attribute the rejection to the \
         corrupted arrival count"
    })?;
    let rejection =
        match require_folded_key_parameter_parity(args.batch, corrupted_key_parameter_max_abs) {
            Ok(()) => bail!(
                "the corrupted publication ({}) left grad_k_k/grad_k_a identical (max abs \
             {corrupted_key_parameter_max_abs:.3e}); the control never exercised the fold",
                corruption.label()
            ),
            Err(rejection) => rejection,
        };
    if corrupted_key_parameter_max_abs <= clean_key_parameter_max_abs {
        bail!(
            "the seeded arrival count did not reach the key-parameter gradients (corrupted \
             {corrupted_key_parameter_max_abs:.3e}, clean {clean_key_parameter_max_abs:.3e}), so \
             the control never exercised the fold"
        );
    }
    let post_step_guard_saw_the_corruption =
        require_folded_parameter_parity(args.batch, corrupted_parameter_delta).is_err();
    if !post_step_guard_saw_the_corruption {
        bail!(
            "the corrupted publication moved a parameter by only \
             {corrupted_parameter_delta:.3e}, inside the {FOLDED_PARAMETER_ATOL:.1e} allowance; \
             the post-step guard would accept a fold that never publishes its \
             key-parameter gradients"
        );
    }

    println!(
        "{}",
        serde_json::to_string(&NegativeControl {
            result: "pass",
            negative_control: "packed-fold-publication",
            device: staged.decision_device_name().to_owned(),
            width,
            head_size: args.head_size,
            batch: args.batch,
            steps: args.steps,
            corruption: corruption.label(),
            arrival_seed: match corruption {
                Corruption::Batch1DirectWriteSink => None,
                Corruption::CrossBatchArrivalCount(seed) => Some(seed),
            },
            parameter_atol: FOLDED_PARAMETER_ATOL,
            staged_schedule,
            folded_schedule,
            staged_key_parameter_max_abs,
            clean_key_parameter_max_abs,
            clean_parameter_delta,
            corrupted_key_parameter_max_abs,
            corrupted_parameter_delta,
            clean_accepted: true,
            corrupted_rejected: true,
            post_step_guard_saw_the_corruption,
            rejection: rejection.to_string(),
        })?
    );
    Ok(())
}

/// The folded rung's cost split into the two things the fold adds to the state
/// kernel: the arrival counters with their device-scope barriers, and the
/// batch-order reduction the last-arriving workgroup performs.
///
/// The reduction is removable at runtime in exactly one way, and it is the way
/// the fold itself decides it: only the workgroup that observes a head's last
/// arrival runs it, so seeding a full batch of phantom arrivals removes that
/// workgroup's loop and nothing else -- every workgroup still stores its
/// partials, publishes its arrival and passes the same barriers. The difference
/// against the clean folded leg is therefore the reduction tail. The difference
/// of that leg against the autotuner's `fused` rung is what the counters cost
/// once the separate reduce dispatch they replace is credited back, which is why
/// it is reported as a signed number rather than a cost.
///
/// Both legs are gated: the clean folded rung must still publish the two-dispatch
/// chain's `grad_k_k`/`grad_k_a` exactly, and the phantom leg must fail that same
/// gate. A profile where either check does not hold is measuring some other
/// graph and is refused.
#[derive(Serialize)]
struct FoldedProfile {
    result: &'static str,
    profile: &'static str,
    device: String,
    width: usize,
    head_size: usize,
    batch: usize,
    steps: usize,
    samples: usize,
    warmup: usize,
    staged_schedule: Option<String>,
    fused_schedule: Option<String>,
    folded_schedule: Option<String>,
    /// The gates the legs cleared before they were timed, recorded so the receipt
    /// proves its own timings rather than trusting the binary's exit status. The
    /// clean fold must publish the staged arm's `grad_k_k`/`grad_k_a` exactly, the
    /// staged magnitude must be non-zero (or the agreement is zero-vs-zero), and
    /// the phantom-arrival leg must fail that same gate.
    key_parameter_atol: f64,
    staged_key_parameter_max_abs: f64,
    folded_key_parameter_max_abs: f64,
    without_reduction_key_parameter_max_abs: Option<f64>,
    legs: Vec<ProfileLeg>,
    /// The widest leg's interquartile range: the smallest delta this run can
    /// call a measurement rather than noise. `range_resolution_ms` is the far
    /// more conservative half-range, reported so the reader can see both.
    resolution_ms: f64,
    range_resolution_ms: f64,
    folded_over_fused: f64,
    folded_minus_fused_ms: f64,
    /// Whether the folded rung leads by more than that resolution.
    folded_beats_fused_beyond_resolution: bool,
    folded_minus_without_reduction_ms: Option<f64>,
    without_reduction_minus_fused_ms: Option<f64>,
    /// Whether the reduction tail, and separately the counter/dispatch credit,
    /// is larger than the resolution. `false` means the component is below what
    /// this run can see, not that it is free.
    reduction_tail_beyond_resolution: Option<bool>,
    counters_beyond_resolution: Option<bool>,
    note: &'static str,
}

fn profile_folded(args: &Args) -> Result<()> {
    if args.chain != Chain::BackwardFusion {
        bail!("--profile-folded measures the folded rung; pass --chain backward");
    }
    std::env::set_var("HIERARCHOS_RWKV_CROSS_BATCH_KEY_PARAM_REDUCE", "1");
    std::env::remove_var("HIERARCHOS_RWKV_CROSS_BATCH_ARRIVAL_SEED");
    std::env::remove_var("HIERARCHOS_RWKV_KEY_PARAM_DIRECT_WRITE_SINK");
    let device = VulkanDevice::new()?;
    let mut staged = build(args, device.clone())?;
    let mut fused = build(args, device.clone())?;
    let mut folded = build(args, device.clone())?;
    // Batch 1 has no arrival counter at all: the rung owns one row per channel and
    // writes the gradients directly, so the fold there already runs without a
    // reduction and there is no second leg to measure.
    let mut without_reduction = if args.batch > 1 {
        let (switch, value) = Corruption::for_batch(args.batch).switch();
        std::env::set_var(switch, value);
        let built = build(args, device.clone());
        std::env::remove_var(switch);
        Some(built?)
    } else {
        None
    };
    if args.storage == Storage::Fp16 {
        let mut legs: Vec<&mut RwkvTbpttSequenceOp> = vec![&mut staged, &mut fused, &mut folded];
        if let Some(op) = without_reduction.as_mut() {
            legs.push(op);
        }
        for op in legs {
            op.enable_low_rank_fp16_parameter_storage()?;
        }
        if !staged.low_rank_fp16_parameter_storage_active() {
            bail!(
                "--parameter-storage fp16-storage did not activate the packed mirrors; the \
                 profile would time the FP32 masters"
            );
        }
    }
    pin(&mut staged, Chain::BackwardFusion, false, args.batch)?;
    pin(&mut fused, Chain::BackwardFusion, true, args.batch)?;
    pin_folded(&mut folded, args.batch)?;
    if let Some(op) = without_reduction.as_mut() {
        pin_folded(op, args.batch)?;
    }

    let width = staged.width();
    let state_size = staged.state_size();
    let vector_len = args.batch * width;
    let state_len = vector_len * state_size;
    let x = deterministic_values(args.steps * vector_len, 0.017, 3);
    let grad_output = deterministic_values(args.steps * vector_len, 0.009, 11);
    let initial_state = deterministic_values(state_len, 0.003, 17);
    let token_features = vec![0.0f32; args.steps * args.batch * args.input_dim];

    // One untimed sequence on every leg first, so the samples below run on a warm
    // graph and the parity gates read the gradients of one identical step.
    {
        let mut legs: Vec<&mut RwkvTbpttSequenceOp> = vec![&mut staged, &mut fused, &mut folded];
        if let Some(op) = without_reduction.as_mut() {
            legs.push(op);
        }
        for op in legs {
            step(
                op,
                args.batch,
                args.steps,
                &x,
                &token_features,
                &initial_state,
                &grad_output,
            )?;
        }
    }
    let magnitude = key_parameter_gradient_magnitude(&staged)?;
    if magnitude == 0.0 {
        bail!(
            "the staged arm's grad_k_k/grad_k_a are all zero, so the folded rung's agreement \
             with them would prove nothing"
        );
    }
    let clean = key_parameter_gradient_delta(&staged, &folded)?;
    require_folded_key_parameter_parity(args.batch, clean)?;
    let phantom_delta = match without_reduction.as_ref() {
        Some(op) => {
            let delta = key_parameter_gradient_delta(&staged, op)?;
            if delta <= clean {
                bail!(
                    "batch {}: the phantom-arrival leg published the key-parameter gradients \
                     anyway ({delta:.3e} against {clean:.3e}), so its timing is not the fold \
                     without its reduction",
                    args.batch
                );
            }
            Some(delta)
        }
        None => None,
    };

    let (mut raw, labels): (Vec<Vec<f64>>, &[&'static str]) = match without_reduction.as_mut() {
        Some(op) => (
            sample_raw(
                &mut [&mut staged, &mut fused, &mut folded, op],
                args,
                &x,
                &token_features,
                &initial_state,
                &grad_output,
            )?,
            &["staged", "fused", "folded", "folded-without-reduction"],
        ),
        None => (
            sample_raw(
                &mut [&mut staged, &mut fused, &mut folded],
                args,
                &x,
                &token_features,
                &initial_state,
                &grad_output,
            )?,
            &["staged", "fused", "folded"],
        ),
    };
    let legs: Vec<ProfileLeg> = raw
        .iter_mut()
        .zip(labels)
        .map(|(samples, label)| {
            let median_ms = median(samples);
            let last = samples.len().saturating_sub(1);
            let q1 = samples.get(samples.len() / 4).copied().unwrap_or(median_ms);
            let q3 = samples
                .get(samples.len() * 3 / 4)
                .copied()
                .unwrap_or(median_ms);
            ProfileLeg {
                label,
                median_ms,
                ms_per_step: median_ms / args.steps as f64,
                min_ms: samples.first().copied().unwrap_or(median_ms),
                max_ms: samples.get(last).copied().unwrap_or(median_ms),
                iqr_ms: q3 - q1,
            }
        })
        .collect();
    let fused_ms = legs[1].median_ms;
    let folded_ms = legs[2].median_ms;
    let without_reduction_ms = legs.get(3).map(|leg| leg.median_ms);
    // The interleaved legs are timed identically, so the widest half-range any of
    // them shows is the resolution this profile can attribute a delta to. The
    // deltas it reports are a few tenths of a millisecond, so without this number
    // a signed difference would look like a cost when it is a sample of noise.
    let resolution_ms = legs.iter().map(|leg| leg.iqr_ms).fold(0.0, f64::max);
    let range_resolution_ms = legs
        .iter()
        .map(|leg| (leg.max_ms - leg.min_ms) / 2.0)
        .fold(0.0, f64::max);
    let note = if without_reduction.is_some() {
        "folded-minus-without-reduction is the reduction tail: one workgroup per head, one \
         dependent add chain per channel, run after every other workgroup of the dispatch \
         has finished. without-reduction-minus-fused is the atomics, barriers and partial \
         stores credited against the separate reduce dispatch they replace, so its value is \
         what the counters cost *after* that credit and may legitimately be negative. \
         resolution_ms is the widest leg's interquartile range and range_resolution_ms its \
         half-range; a delta at or below the former is inside this run's noise, which is why \
         each component also carries a *_beyond_resolution flag rather than only a sign. At \
         batch 1 there is no counter and no reduction; the folded leg is the whole fold."
    } else {
        "batch 1 has no arrival counter: the folded rung writes the parameter gradients \
         directly, so this row measures the fold with neither counters nor a reduction."
    };
    let profile = FoldedProfile {
        result: "pass",
        profile: "folded-publication",
        device: staged.decision_device_name().to_owned(),
        width,
        head_size: args.head_size,
        batch: args.batch,
        steps: args.steps,
        samples: args.iterations,
        warmup: args.warmup,
        staged_schedule: staged.packed_backward_fusion_label(args.batch),
        fused_schedule: fused.packed_backward_fusion_label(args.batch),
        folded_schedule: folded.packed_backward_fusion_label(args.batch),
        key_parameter_atol: FOLDED_KEY_PARAMETER_ATOL,
        staged_key_parameter_max_abs: magnitude,
        folded_key_parameter_max_abs: clean,
        without_reduction_key_parameter_max_abs: phantom_delta,
        legs,
        resolution_ms,
        range_resolution_ms,
        folded_over_fused: folded_ms / fused_ms,
        folded_minus_fused_ms: folded_ms - fused_ms,
        folded_beats_fused_beyond_resolution: fused_ms - folded_ms > resolution_ms,
        folded_minus_without_reduction_ms: without_reduction_ms.map(|ms| folded_ms - ms),
        without_reduction_minus_fused_ms: without_reduction_ms.map(|ms| ms - fused_ms),
        reduction_tail_beyond_resolution: without_reduction_ms
            .map(|ms| (folded_ms - ms).abs() > resolution_ms),
        counters_beyond_resolution: without_reduction_ms
            .map(|ms| (ms - fused_ms).abs() > resolution_ms),
        note,
    };
    println!("{}", serde_json::to_string(&profile)?);
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    if args.profile_folded && args.negative_control {
        bail!("--profile-folded and --negative-control are separate measurements");
    }
    if args.negative_control {
        return negative_control(&args);
    }
    if args.profile_folded {
        return profile_folded(&args);
    }
    // The deepest rung of the fusion ladder is guarded: the cell only compiles
    // the cross-batch fold in a process that asks for it. The A/B is that kind
    // of caller, and every leg -- including the shipped default -- then sees the
    // same kernel set, so the measured default is the same graph training builds
    // once its record asks for that depth.
    std::env::set_var("HIERARCHOS_RWKV_CROSS_BATCH_KEY_PARAM_REDUCE", "1");
    let device = VulkanDevice::new()?;

    let mut legacy = build(&args, device.clone())?;
    let mut packed = build(&args, device.clone())?;
    let mut default = build(&args, device.clone())?;
    let mut folded = if args.chain == Chain::BackwardFusion {
        Some(build(&args, device.clone())?)
    } else {
        None
    };
    if args.storage == Storage::Fp16 {
        for op in [&mut legacy, &mut packed, &mut default] {
            op.enable_low_rank_fp16_parameter_storage()?;
        }
        if let Some(op) = folded.as_mut() {
            op.enable_low_rank_fp16_parameter_storage()?;
        }
        if !legacy.low_rank_fp16_parameter_storage_active() {
            bail!(
                "--parameter-storage fp16-storage did not activate the packed mirrors; \
                 both arms would still read the FP32 masters"
            );
        }
    }
    pin(&mut legacy, args.chain, false, args.batch)?;
    pin(&mut packed, args.chain, true, args.batch)?;
    if let Some(op) = folded.as_mut() {
        pin_folded(op, args.batch)?;
    }

    let width = legacy.width();
    let state_size = legacy.state_size();
    let vector_len = args.batch * width;
    let state_len = vector_len * state_size;
    let x = deterministic_values(args.steps * vector_len, 0.017, 3);
    let grad_output = deterministic_values(args.steps * vector_len, 0.009, 11);
    let initial_state = deterministic_values(state_len, 0.003, 17);
    let token_features = vec![0.0f32; args.steps * args.batch * args.input_dim];

    let (fused_schedule, staged_schedule, folded_schedule) = if args.chain == Chain::BackwardFusion
    {
        (
            packed.packed_backward_fusion_label(args.batch),
            legacy.packed_backward_fusion_label(args.batch),
            folded
                .as_ref()
                .and_then(|op| op.packed_backward_fusion_label(args.batch)),
        )
    } else {
        (None, None, None)
    };
    let arm_a = args.chain.active_label(&legacy, args.batch);
    let arm_b = args.chain.active_label(&packed, args.batch);
    let available_schedules = if args.chain == Chain::BackwardFusion {
        Some(legacy.packed_backward_fusion_labels(args.batch)?)
    } else {
        None
    };
    if args.chain == Chain::BackwardFusion {
        if fused_schedule.is_none() || staged_schedule.is_none() {
            bail!(
                "the pinned backward arms did not report their schedules: fused {:?}, staged {:?}",
                fused_schedule,
                staged_schedule
            );
        }
        let available = available_schedules
            .as_ref()
            .context("the backward-fusion chain did not enumerate its depths")?;
        let rungs: Vec<&str> = BACKWARD_FUSION_RUNGS
            .into_iter()
            .filter(|rung| {
                available
                    .iter()
                    .any(|label| backward_fusion_rung(label) == Some(*rung))
            })
            .collect();
        if rungs.len() < 2 {
            bail!(
                "batch {}: the device compiles {} backward fusion rung(s) ({rungs:?}), so \
                 there is no fusion A/B to measure",
                args.batch,
                rungs.len()
            );
        }
        let staged_label = staged_schedule.as_deref().unwrap_or_default();
        let fused_label = fused_schedule.as_deref().unwrap_or_default();
        let staged_rung = backward_fusion_rung(staged_label)
            .context("the staged arm did not pin one of the fusion rungs")?;
        let fused_rung = backward_fusion_rung(fused_label)
            .context("the fused arm did not pin one of the fusion rungs")?;
        if staged_rung != BACKWARD_FUSION_RUNGS[0] {
            bail!(
                "batch {}: the staged arm pinned {staged_rung:?}, not the fully staged rung \
                 {:?}",
                args.batch,
                BACKWARD_FUSION_RUNGS[0]
            );
        }
        if fused_rung == staged_rung {
            bail!(
                "batch {}: both routing arms pinned the {fused_rung:?} rung, so there is no                  fusion-depth difference to measure",
                args.batch
            );
        }
        if backward_fusion_suffix(staged_label, staged_rung)
            != backward_fusion_suffix(fused_label, fused_rung)
        {
            bail!(
                "batch {}: the arms differ in projection order or fan-in as well as depth: \
                 staged {staged_label:?}, fused {fused_label:?}",
                args.batch
            );
        }
        let folded_label = folded_schedule
            .as_deref()
            .context("the folded leg did not report its schedule")?;
        let folded_rung = backward_fusion_rung(folded_label)
            .context("the folded leg did not pin one of the fusion rungs")?;
        if folded_rung != rungs[rungs.len() - 1] {
            bail!(
                "batch {}: the folded leg pinned {folded_rung:?}, not the deepest rung {:?} \
                 (compiled: {rungs:?})",
                args.batch,
                rungs[rungs.len() - 1]
            );
        }
        if backward_fusion_suffix(staged_label, staged_rung)
            != backward_fusion_suffix(folded_label, folded_rung)
        {
            bail!(
                "batch {}: the folded leg differs in projection order or fan-in as well as \
                 depth: staged {staged_label:?}, folded {folded_label:?}",
                args.batch
            );
        }
    } else if arm_a != Some(args.chain.staged_label()) || arm_b != Some(args.chain.packed_label()) {
        bail!(
            "the pinned arms did not separate the two {} routes: forced {} built {:?}, \
             forced {} built {:?}",
            args.chain.label(),
            args.chain.staged_label(),
            arm_a,
            args.chain.packed_label(),
            arm_b
        );
    }

    let (legacy_ms, packed_ms, default_ms, folded_ms) = match folded.as_mut() {
        Some(folded) => {
            let samples = sample(
                &mut [&mut legacy, &mut packed, &mut default, folded],
                &args,
                &x,
                &token_features,
                &initial_state,
                &grad_output,
            )?;
            (samples[0], samples[1], samples[2], Some(samples[3]))
        }
        None => {
            let samples = sample(
                &mut [&mut legacy, &mut packed, &mut default],
                &args,
                &x,
                &token_features,
                &initial_state,
                &grad_output,
            )?;
            (samples[0], samples[1], samples[2], None)
        }
    };

    let legacy_last = step(
        &mut legacy,
        args.batch,
        args.steps,
        &x,
        &token_features,
        &initial_state,
        &grad_output,
    )?;
    let packed_last = step(
        &mut packed,
        args.batch,
        args.steps,
        &x,
        &token_features,
        &initial_state,
        &grad_output,
    )?;
    let tensors = vec![
        tensor_diff(
            "outputs",
            &legacy_last.outputs,
            &packed_last.outputs,
            0.0,
            0.0,
        ),
        tensor_diff(
            "final_packed_state",
            &legacy_last.final_packed_state,
            &packed_last.final_packed_state,
            0.0,
            0.0,
        ),
        tensor_diff("grad_x", &legacy_last.grad_x, &packed_last.grad_x, 0.0, 0.0),
        tensor_diff(
            "token_feature_grad",
            &legacy_last.token_feature_grad,
            &packed_last.token_feature_grad,
            0.0,
            0.0,
        ),
        tensor_diff(
            "grad_initial_packed_state",
            &legacy_last.grad_initial_packed_state,
            &packed_last.grad_initial_packed_state,
            0.0,
            0.0,
        ),
    ];
    let worst_max_abs = require_agreement(&tensors, 0.0, 0.0)?;

    let folded_last = match folded.as_mut() {
        Some(op) => Some(step(
            op,
            args.batch,
            args.steps,
            &x,
            &token_features,
            &initial_state,
            &grad_output,
        )?),
        None => None,
    };
    let folded_max_abs = match folded_last.as_ref() {
        Some(folded_last) => {
            let folded_tensors = vec![
                tensor_diff(
                    "outputs",
                    &legacy_last.outputs,
                    &folded_last.outputs,
                    FOLDED_ATOL,
                    0.0,
                ),
                tensor_diff(
                    "final_packed_state",
                    &legacy_last.final_packed_state,
                    &folded_last.final_packed_state,
                    FOLDED_ATOL,
                    0.0,
                ),
                tensor_diff(
                    "grad_x",
                    &legacy_last.grad_x,
                    &folded_last.grad_x,
                    FOLDED_ATOL,
                    0.0,
                ),
                tensor_diff(
                    "token_feature_grad",
                    &legacy_last.token_feature_grad,
                    &folded_last.token_feature_grad,
                    FOLDED_ATOL,
                    0.0,
                ),
                tensor_diff(
                    "grad_initial_packed_state",
                    &legacy_last.grad_initial_packed_state,
                    &folded_last.grad_initial_packed_state,
                    FOLDED_ATOL,
                    0.0,
                ),
            ];
            Some(require_agreement(&folded_tensors, FOLDED_ATOL, 0.0)?)
        }
        None => None,
    };

    // The fold's own output, read before the AdamW step below touches the
    // gradient buffers: both legs have just run the identical sequence, so their
    // `grad_k_k`/`grad_k_a` hold what each reduction produced on its last
    // timestep.
    let (folded_key_parameter_max_abs, staged_key_parameter_max_abs) = match folded.as_ref() {
        Some(folded) => {
            let magnitude = key_parameter_gradient_magnitude(&legacy)?;
            if magnitude == 0.0 {
                bail!(
                    "batch {}: the staged arm's grad_k_k/grad_k_a are all zero, so a matching \
                     fold would prove nothing",
                    args.batch
                );
            }
            let delta = key_parameter_gradient_delta(&legacy, folded)?;
            require_folded_key_parameter_parity(args.batch, delta)?;
            (Some(delta), Some(magnitude))
        }
        None => (None, None),
    };

    // One identical AdamW step on the two legs reaches the parameter gradients,
    // which the returned tensors cannot: `grad_x` never consumes `grad_k_k` or
    // `grad_k_a`, so only the post-step snapshots can catch a broken cross-batch
    // fold. It runs after every timing sample and readback comparison.
    let folded_parameter_delta = match folded.as_mut() {
        Some(folded) => Some(trained_parameter_delta(
            &mut legacy,
            folded,
            &args,
            &x,
            &token_features,
            &initial_state,
            &grad_output,
        )?),
        None => None,
    };
    if let Some(delta) = folded_parameter_delta {
        require_folded_parameter_parity(args.batch, delta)?;
    }

    let recorded_route = args.chain.recorded_route(&default, args.batch);
    let default_schedule = default.packed_backward_fusion_label(args.batch);
    let default_route = if args.chain == Chain::BackwardFusion {
        // The shipped default is the autotuner's schedule, which is the fused arm
        // unless a record replaced it; when the two arms coincide there is no
        // distinct staged graph and the default is the fused route by construction.
        let available = available_schedules
            .as_ref()
            .context("the backward-fusion chain did not enumerate its depths")?;
        if !available
            .iter()
            .any(|label| Some(label) == default_schedule.as_ref())
        {
            bail!(
                "the default backward schedule {default_schedule:?} is not one of the \
                 compiled depths {available:?}"
            );
        }
        // `fused` names the autotuner's shipped depth, which need not be the
        // extreme the fused arm pins; a recorded staged or folded verdict is what
        // moves it, and the route is classified by the rung the default built
        // rather than by which arm it happens to match.
        match default_schedule.as_deref().and_then(backward_fusion_rung) {
            Some(rung) if Some(rung) == Some(BACKWARD_FUSION_RUNGS[0]) => "staged",
            Some(rung) if Some(rung) == Some(BACKWARD_FUSION_RUNGS[2]) => "folded",
            _ => "fused",
        }
    } else {
        args.chain
            .active_label(&default, args.batch)
            .unwrap_or("intermediate")
    };
    if let Some(route) = recorded_route {
        if default_route != route {
            bail!(
                "the default TBPTT {} route built {default_route}{:?}, but the record names \
                 {route}",
                args.chain.label(),
                default_schedule
            );
        }
    } else if args.chain != Chain::BackwardFusion
        && default_route != args.chain.structural_default()
    {
        bail!(
            "the default TBPTT {} route built {default_route}, but the structural default is \
             {}",
            args.chain.label(),
            args.chain.structural_default()
        );
    }

    let arm_a_leg = Leg {
        label: args.chain.staged_label(),
        median_ms: legacy_ms,
        ms_per_step: legacy_ms / args.steps as f64,
    };
    let arm_b_leg = Leg {
        label: args.chain.packed_label(),
        median_ms: packed_ms,
        ms_per_step: packed_ms / args.steps as f64,
    };
    let backward_fusion = args.chain == Chain::BackwardFusion;
    let (legacy_leg, staged_leg) = if backward_fusion {
        (None, Some(arm_a_leg))
    } else {
        (Some(arm_a_leg), None)
    };
    let (packed_leg, fused_leg) = if backward_fusion {
        (None, Some(arm_b_leg))
    } else {
        (Some(arm_b_leg), None)
    };

    let output = Output {
        device: legacy.decision_device_name().to_owned(),
        width,
        head_size: args.head_size,
        batch: args.batch,
        steps: args.steps,
        chain: args.chain.label(),
        parameter_storage: args.storage.label(),
        precision_policy: args.storage.precision_policy(),
        legacy: legacy_leg,
        packed: packed_leg,
        staged: staged_leg,
        fused: fused_leg,
        default: Leg {
            label: default_route,
            median_ms: default_ms,
            ms_per_step: default_ms / args.steps as f64,
        },
        default_route,
        recorded_route,
        packed_over_legacy: (!backward_fusion).then_some(packed_ms / legacy_ms),
        fused_over_staged: backward_fusion.then_some(packed_ms / legacy_ms),
        folded: folded_ms.map(|median_ms| Leg {
            label: "folded",
            median_ms,
            ms_per_step: median_ms / args.steps as f64,
        }),
        folded_over_staged: folded_ms.map(|median_ms| median_ms / legacy_ms),
        folded_label: folded_ms.map(|_| "folded"),
        folded_schedule,
        folded_max_abs,
        folded_key_parameter_max_abs,
        staged_key_parameter_max_abs,
        folded_parameter_delta,
        staged_schedule: backward_fusion
            .then(|| legacy.packed_backward_fusion_label(args.batch))
            .flatten(),
        fused_schedule: backward_fusion
            .then(|| packed.packed_backward_fusion_label(args.batch))
            .flatten(),
        default_schedule: backward_fusion
            .then(|| default.packed_backward_fusion_label(args.batch))
            .flatten(),
        available_schedules,
        worst_max_abs,
        tensors,
    };
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
