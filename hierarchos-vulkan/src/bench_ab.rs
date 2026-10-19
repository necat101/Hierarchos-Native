//! Shared measurement harness for the three-leg kernel A/B receipts.
//!
//! Both the packed-cell time-mix benchmark and the packed-cell channel-mix
//! benchmark measure the same way, and the next producer measurement should be
//! configuration rather than another copy of the same loop:
//!
//! * build three legs that differ only in construction-time switches (forced
//!   first graph, forced second graph, production default);
//! * prove which graph each leg really ran, with the graph's own label and with
//!   the vendor dispatch counter, so a vacuous A/B fails instead of reporting;
//! * sample the legs **interleaved with a rotated order**, so neither a warm
//!   cache nor clock drift systematically favors one leg;
//! * require every returned tensor to agree, so a timing cannot compare
//!   different work;
//! * fail closed when the production default stops building the graph the
//!   decision record names.
//!
//! Host-side code here only *measures and selects* complete Vulkan graphs. No
//! arithmetic moves off the GPU: each leg dispatches its own full shader
//! sequence, and the decision only picks which sequence the default runs.

use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde::Serialize;

/// Median of the samples. Sorts in place; the caller keeps the original order
/// only for min/max, which are taken from the sorted array.
pub fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|left, right| left.total_cmp(right));
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct Timing {
    pub samples: usize,
    pub median_ms: f64,
    pub mean_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

pub fn timing(mut samples: Vec<f64>) -> Timing {
    let count = samples.len();
    let sum: f64 = samples.iter().sum();
    let median_ms = median(&mut samples);
    Timing {
        samples: count,
        median_ms,
        mean_ms: sum / count as f64,
        min_ms: samples[0],
        max_ms: samples[count - 1],
    }
}

#[derive(Serialize, Debug)]
pub struct TensorDiff {
    pub label: &'static str,
    pub values: usize,
    pub max_abs: f64,
    pub worst_excess: f64,
}

/// Compare one returned tensor against the tolerance the harness applies.
pub fn tensor_diff(
    label: &'static str,
    left: &[f32],
    right: &[f32],
    atol: f64,
    rtol: f64,
) -> TensorDiff {
    let mut max_abs = 0.0f64;
    let mut worst_excess = f64::NEG_INFINITY;
    for (left, right) in left.iter().zip(right) {
        let difference = (*left as f64 - *right as f64).abs();
        let bound = atol + rtol * (*right as f64).abs();
        max_abs = max_abs.max(difference);
        worst_excess = worst_excess.max(difference - bound);
    }
    TensorDiff {
        label,
        values: left.len(),
        max_abs,
        worst_excess,
    }
}

/// Fail unless every compared tensor agrees, and return the worst absolute
/// difference for the receipt. A timing that compared different work must not be
/// reported.
pub fn require_agreement(tensors: &[TensorDiff], atol: f64, rtol: f64) -> Result<f64> {
    if let Some(outside) = tensors.iter().find(|tensor| tensor.worst_excess > 0.0) {
        bail!(
            "the timed graphs disagree on {}: max abs {:e} exceeds atol {atol:e} + rtol {rtol:e} \
             (worst excess {:e}), so this timing compares different work",
            outside.label,
            outside.max_abs,
            outside.worst_excess
        );
    }
    Ok(tensors
        .iter()
        .map(|tensor| tensor.max_abs)
        .fold(0.0f64, f64::max))
}

/// Sets an environment switch for the lifetime of the guard, so a leg cannot
/// leak its construction-time switch into the next leg even on an error path.
pub struct EnvSwitch {
    name: &'static str,
}

impl EnvSwitch {
    pub fn set(name: &'static str) -> Self {
        std::env::set_var(name, "1");
        Self { name }
    }
}

impl Drop for EnvSwitch {
    fn drop(&mut self) {
        std::env::remove_var(self.name);
    }
}

/// Vendor dispatches of one family since process start, or zero when the family
/// never ran. This is the second, independent proof of which graph a leg ran.
pub fn variant_dispatches(family_label: &str) -> u64 {
    crate::variant_dispatch_counts()
        .into_iter()
        .find(|(label, _)| *label == family_label)
        .map(|(_, count)| count)
        .unwrap_or(0)
}

/// The forced legs must really have built different graphs, or the A/B is not an
/// A/B. Both labels come from the op, not from this harness.
pub fn require_separated_legs(first_label: &str, second_label: &str) -> Result<()> {
    if first_label.is_empty() || second_label.is_empty() || first_label == second_label {
        bail!(
            "the switches did not separate the two graphs: first leg built {first_label:?}, \
             second leg built {second_label:?}"
        );
    }
    Ok(())
}

/// The forced-fusion leg must not have dispatched the module only the
/// composition uses.
pub fn require_fused_leg_is_pure(
    family_label: &str,
    fused_dispatches: u64,
    units: usize,
) -> Result<()> {
    if fused_dispatches != 0 {
        bail!(
            "the forced-fusion leg dispatched {family_label} {fused_dispatches} time(s) over \
             {units} unit(s); it did not run the fused graph, so this run is not an A/B"
        );
    }
    Ok(())
}

/// The promotion contract: the production default builds the composition exactly
/// where the composed leg reached the tuned module, and the fused graph
/// everywhere else. `fused_label` is the label the fused leg reported for this
/// geometry, which the default must reproduce when it keeps the fused graph.
pub fn require_default_follows_record(
    family_label: &str,
    fused_label: &str,
    composed_dispatches: u64,
    default_label: &str,
    batch: usize,
) -> Result<()> {
    let expected = if composed_dispatches > 0 {
        "composed"
    } else {
        fused_label
    };
    if default_label != expected {
        bail!(
            "the default leg built {default_label} at batch {batch} while the tuned {family_label} \
             module {} run in the composed leg (expected {expected})",
            if composed_dispatches > 0 { "did" } else { "did not" }
        );
    }
    Ok(())
}

/// The fail-closed check the measurement exists to protect, using what the
/// shipped binary itself believes.
///
/// `recorded` is the decision the compiled-in record names for this adapter and
/// geometry, read through the op, so this compares the default against the
/// rule that actually decided it rather than against a gate proxy. Where the
/// record has no entry the gate decides, and the composed leg's dispatch count
/// is the proof of what the gate did.
pub fn require_default_matches_decision(
    family_label: &str,
    fused_label: &str,
    composed_dispatches: u64,
    recorded: Option<bool>,
    default_label: &str,
    batch: usize,
) -> Result<()> {
    match recorded {
        None => require_default_follows_record(
            family_label,
            fused_label,
            composed_dispatches,
            default_label,
            batch,
        ),
        Some(compose) => {
            let expected = if compose { "composed" } else { fused_label };
            if default_label != expected {
                bail!(
                    "the default leg built {default_label} at batch {batch} but the compiled-in \
                     decision record for this adapter names {expected}; the shipped default \
                     stopped following the measured record"
                );
            }
            Ok(())
        }
    }
}

/// The rank-2 fail-closed check, using what the shipped binary itself believes.
///
/// Where the record has an entry the default must follow it. Where it has none the
/// expectation is the *structural* default rather than a gate: the fused producer
/// is what the shipped graph builds when nothing measured says otherwise, so a
/// device with no entry keeps the fusion and the receipt says so.
pub fn require_default_matches_route_decision(
    fused_label: &str,
    recorded: Option<bool>,
    default_label: &str,
    batch: usize,
) -> Result<()> {
    let expected = match recorded {
        Some(true) => "composed",
        Some(false) => fused_label,
        None => fused_label,
    };
    if default_label != expected {
        bail!(
            "the default leg built {default_label} at batch {batch} but the compiled-in route \
             record (or, with no entry, the structural default) names {expected}; the shipped \
             default stopped following the measured record"
        );
    }
    Ok(())
}

/// The rank-2 dispatch accounting: the fused arm is the single producer dispatch
/// and the staged arm is the chain it replaces, so the counts must be one against
/// at least the layer norm, the interpolation and the first stage.
pub fn require_forward_arm_dispatches(fused: u64, composed: u64) -> Result<()> {
    if fused != 1 {
        bail!(
            "the fused LN1/low-rank arm dispatched {fused} kernel(s) per forward, but the \
             producer is one dispatch; this run is not the graph the receipt names"
        );
    }
    if composed < 6 {
        bail!(
            "the staged chain dispatched {composed} kernel(s) per forward, which cannot cover \
             the layer norm, the interpolation, the three first-stage matmuls and the second stage"
        );
    }
    if fused >= composed {
        bail!(
            "the fused arm dispatched {fused} kernel(s) and the staged arm {composed}; the \
             fusion replaced nothing, so this A/B is vacuous"
        );
    }
    Ok(())
}

/// Interleaved, rotated sampling of three legs.
///
/// Each leg is a closure returning the op's result for one invocation: run it
/// once per sample and hand back the op's own output, exactly as the callers do
/// today. Returns the three timings and the last result of each leg, so the
/// caller can compare the two timed graphs without running either outside the
/// rotated order.
#[allow(clippy::type_complexity)]
pub fn sample_three_legs<A, B, C, RA, RB, RC>(
    iterations: usize,
    warmup: usize,
    mut leg0: A,
    mut leg1: B,
    mut leg2: C,
) -> Result<(Timing, Timing, Timing, RA, RB, RC)>
where
    A: FnMut() -> Result<RA>,
    B: FnMut() -> Result<RB>,
    C: FnMut() -> Result<RC>,
{
    if iterations == 0 {
        bail!("sample_three_legs needs at least one iteration");
    }
    let mut last0 = None;
    let mut last1 = None;
    let mut last2 = None;
    for _ in 0..warmup {
        last0 = Some(leg0()?);
        last1 = Some(leg1()?);
        last2 = Some(leg2()?);
    }
    let mut samples = [
        Vec::with_capacity(iterations),
        Vec::with_capacity(iterations),
        Vec::with_capacity(iterations),
    ];
    for iteration in 0..iterations {
        // Rotate which graph runs first so neither a warm cache nor clock drift
        // systematically favors one leg.
        let order = match iteration % 3 {
            0 => [0usize, 1, 2],
            1 => [1, 2, 0],
            _ => [2, 0, 1],
        };
        for leg in order {
            let started = Instant::now();
            match leg {
                0 => last0 = Some(leg0()?),
                1 => last1 = Some(leg1()?),
                _ => last2 = Some(leg2()?),
            }
            samples[leg].push(started.elapsed().as_secs_f64() * 1.0e3);
        }
    }
    let first = timing(std::mem::take(&mut samples[0]));
    let second = timing(std::mem::take(&mut samples[1]));
    let third = timing(std::mem::take(&mut samples[2]));
    Ok((
        first,
        second,
        third,
        last0.context("the first leg produced no sample")?,
        last1.context("the second leg produced no sample")?,
        last2.context("the third leg produced no sample")?,
    ))
}

/// Interleaved, rotated sampling of four legs.
///
/// The rank-3 key/state A/B compares four graphs: the wider-group fused kernel,
/// the narrow one it replaces, the two-dispatch chain, and the production
/// default. Widening the group is a scheduling claim
/// that must be measured against both neighbours, not inferred from the chain's
/// timing, so all four legs rotate through every position instead of one arm
/// timing outside the rotated order.
#[allow(clippy::type_complexity)]
pub fn sample_four_legs<A, B, C, D, RA, RB, RC, RD>(
    iterations: usize,
    warmup: usize,
    mut leg0: A,
    mut leg1: B,
    mut leg2: C,
    mut leg3: D,
) -> Result<(Timing, Timing, Timing, Timing, RA, RB, RC, RD)>
where
    A: FnMut() -> Result<RA>,
    B: FnMut() -> Result<RB>,
    C: FnMut() -> Result<RC>,
    D: FnMut() -> Result<RD>,
{
    if iterations == 0 {
        bail!("sample_four_legs needs at least one iteration");
    }
    let mut last0 = None;
    let mut last1 = None;
    let mut last2 = None;
    let mut last3 = None;
    for _ in 0..warmup {
        last0 = Some(leg0()?);
        last1 = Some(leg1()?);
        last2 = Some(leg2()?);
        last3 = Some(leg3()?);
    }
    let mut samples = [
        Vec::with_capacity(iterations),
        Vec::with_capacity(iterations),
        Vec::with_capacity(iterations),
        Vec::with_capacity(iterations),
    ];
    for iteration in 0..iterations {
        // A rotation of the four legs, so each leg runs first, second, third and
        // last equally often: neither a warm cache nor clock drift can favor one.
        for offset in 0..4usize {
            let leg = (iteration + offset) % 4;
            let started = Instant::now();
            match leg {
                0 => last0 = Some(leg0()?),
                1 => last1 = Some(leg1()?),
                2 => last2 = Some(leg2()?),
                _ => last3 = Some(leg3()?),
            }
            samples[leg].push(started.elapsed().as_secs_f64() * 1.0e3);
        }
    }
    let first = timing(std::mem::take(&mut samples[0]));
    let second = timing(std::mem::take(&mut samples[1]));
    let third = timing(std::mem::take(&mut samples[2]));
    let fourth = timing(std::mem::take(&mut samples[3]));
    Ok((
        first,
        second,
        third,
        fourth,
        last0.context("the first leg produced no sample")?,
        last1.context("the second leg produced no sample")?,
        last2.context("the third leg produced no sample")?,
        last3.context("the fourth leg produced no sample")?,
    ))
}

/// A receipt leg: the graph it reported, the vendor dispatches its first run
/// performed, and its timing (optionally per unit, for sequence harnesses).
#[derive(Serialize)]
pub struct LegReport {
    pub label: &'static str,
    pub vendor_dispatches: u64,
    pub timing: Timing,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ms_per_unit: Option<f64>,
}

impl LegReport {
    pub fn new(label: &'static str, vendor_dispatches: u64, timing: Timing, unit: f64) -> Self {
        let ms_per_unit = if unit > 0.0 {
            Some(timing.median_ms / unit)
        } else {
            None
        };
        Self {
            label,
            vendor_dispatches,
            timing,
            ms_per_unit,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn median_handles_both_parities() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
    }

    #[test]
    fn rotated_order_covers_every_leg_every_position() {
        let mut first = [0usize; 3];
        let mut last = [0usize; 3];
        for iteration in 0..3 {
            let order = match iteration % 3 {
                0 => [0usize, 1, 2],
                1 => [1, 2, 0],
                _ => [2, 0, 1],
            };
            first[order[0]] += 1;
            last[order[2]] += 1;
        }
        assert_eq!(
            first,
            [1, 1, 1],
            "each leg must run first once in three samples"
        );
        assert_eq!(
            last,
            [1, 1, 1],
            "each leg must run last once in three samples"
        );
    }

    #[test]
    fn recorded_decision_overrides_the_gate_proxy() {
        // The record says `fused` where the gate would compose: the default must
        // follow the record, not the gate.
        assert!(require_default_matches_decision(
            "linear3-forward",
            "fused",
            8,
            Some(false),
            "fused",
            8
        )
        .is_ok());
        assert!(require_default_matches_decision(
            "linear3-forward",
            "fused",
            8,
            Some(false),
            "composed",
            8
        )
        .is_err());
        // No entry: the gate proxy still applies.
        assert!(require_default_matches_decision(
            "linear3-forward",
            "fused",
            8,
            None,
            "composed",
            8
        )
        .is_ok());
        assert!(require_default_matches_decision(
            "linear3-forward",
            "fused",
            0,
            None,
            "composed",
            4
        )
        .is_err());
    }

    #[test]
    fn rank_two_default_follows_the_record_and_otherwise_keeps_the_fusion() {
        // No entry: the structural default is the fused producer.
        assert!(require_default_matches_route_decision("fused", None, "fused", 8).is_ok());
        assert!(require_default_matches_route_decision("fused", None, "composed", 8).is_err());
        // A recorded staged win must be dispatched by the default.
        assert!(require_default_matches_route_decision("fused", Some(true), "composed", 8).is_ok());
        assert!(require_default_matches_route_decision("fused", Some(true), "fused", 8).is_err());
        // A recorded fused win keeps the fusion even though the chain exists.
        assert!(require_default_matches_route_decision("fused", Some(false), "fused", 8).is_ok());
        assert!(
            require_default_matches_route_decision("fused", Some(false), "composed", 8).is_err()
        );
    }

    #[test]
    fn rank_two_arms_must_actually_differ() {
        assert!(require_forward_arm_dispatches(1, 12).is_ok());
        assert!(
            require_forward_arm_dispatches(2, 12).is_err(),
            "the fused arm is one dispatch"
        );
        assert!(
            require_forward_arm_dispatches(1, 3).is_err(),
            "the chain needs its parts"
        );
        assert!(
            require_forward_arm_dispatches(1, 1).is_err(),
            "equal counts mean the fusion replaced nothing"
        );
        assert!(
            require_forward_arm_dispatches(1, 0).is_err(),
            "an arm that dispatched nothing is not an arm"
        );
    }

    #[test]
    fn promotion_predicate_rejects_a_wrong_default() {
        // Tuned module ran: the default must compose.
        assert!(
            require_default_follows_record("linear3-forward", "fused", 8, "composed", 8).is_ok()
        );
        assert!(require_default_follows_record("linear3-forward", "fused", 8, "fused", 8).is_err());
        // Tuned module did not run: the default must keep the fused graph.
        assert!(require_default_follows_record("linear3-forward", "fused", 0, "fused", 4).is_ok());
        assert!(
            require_default_follows_record("linear3-forward", "fused", 0, "composed", 4).is_err()
        );
    }

    #[test]
    fn agreement_gate_rejects_out_of_tolerance_tensors() {
        let agreeing = vec![tensor_diff(
            "output",
            &[1.0, 2.0],
            &[1.0, 2.0],
            1.0e-4,
            1.0e-5,
        )];
        assert!(require_agreement(&agreeing, 1.0e-4, 1.0e-5).is_ok());
        let disagreeing = vec![tensor_diff(
            "output",
            &[1.0, 2.0],
            &[1.0, 2.1],
            1.0e-4,
            1.0e-5,
        )];
        assert!(require_agreement(&disagreeing, 1.0e-4, 1.0e-5).is_err());
    }

    #[test]
    fn separated_legs_need_two_labels() {
        assert!(require_separated_legs("fused", "composed").is_ok());
        assert!(require_separated_legs("composed", "composed").is_err());
        assert!(require_separated_legs("", "composed").is_err());
    }

    #[test]
    fn four_leg_sampling_rotates_every_leg_through_every_position() {
        // Record the order the harness actually invokes the legs in, so the test
        // checks the rotation rather than re-deriving it.
        let order = std::cell::RefCell::new(Vec::new());
        let counts = [Cell::new(0usize), Cell::new(0), Cell::new(0), Cell::new(0)];
        let record = |leg: usize| {
            counts[leg].set(counts[leg].get() + 1);
            order.borrow_mut().push(leg);
            Ok(leg as u32)
        };
        let (first, second, third, fourth, a, b, c, d) =
            sample_four_legs(4, 0, || record(0), || record(1), || record(2), || record(3))
                .expect("sampling succeeds");
        assert_eq!((a, b, c, d), (0, 1, 2, 3));
        assert_eq!(
            counts.map(|count| count.get()),
            [4, 4, 4, 4],
            "four rotated samples per leg"
        );
        assert_eq!(
            [first.samples, second.samples, third.samples, fourth.samples],
            [4, 4, 4, 4]
        );
        let order = order.into_inner();
        for position in 0..4usize {
            let mut at_position: Vec<usize> =
                (0..4).map(|sample| order[sample * 4 + position]).collect();
            at_position.sort_unstable();
            assert_eq!(
                at_position,
                vec![0, 1, 2, 3],
                "each leg must run in position {position} exactly once"
            );
        }
    }

    #[test]
    fn sampling_rotates_and_returns_last_results() {
        let counts = [Cell::new(0usize), Cell::new(0), Cell::new(0)];
        let (first, second, third, a, b, c) = sample_three_legs(
            3,
            1,
            || {
                counts[0].set(counts[0].get() + 1);
                Ok(0u32)
            },
            || {
                counts[1].set(counts[1].get() + 1);
                Ok(1u32)
            },
            || {
                counts[2].set(counts[2].get() + 1);
                Ok(2u32)
            },
        )
        .expect("sampling succeeds");
        assert_eq!((a, b, c), (0, 1, 2));
        assert_eq!(
            counts.map(|count| count.get()),
            [4, 4, 4],
            "one warmup plus three samples per leg"
        );
        assert_eq!(first.samples, 3);
        assert_eq!(second.samples, 3);
        assert_eq!(third.samples, 3);
    }
}
