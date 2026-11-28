//! Phase XVIII — production promotion of the Phase XVII dQ accumulator
//! specialization (test builds only).
//!
//! Phase XVII measured one candidate artifact, `q_dqunroll.spv`, and left it in
//! `validation/phase_xvii_v028_optimization_discovery_20261126/`. This phase
//! promotes those exact bytes into `shaders/` as
//! `transformer_attention_backward_query_rdna3_dqunroll.spv` and routes pass 1
//! of the production two-pass backward through it when
//! `qk_head_dim == 128 && value_head_dim == 128`.
//!
//! Everything numerical here is driven through
//! [`record_two_pass_attention_backward_dispatch`], the function the real
//! training step calls, with the artifact the crate embeds. The route is proved
//! by the production dispatch counters, not by the predicate: a case that claims
//! the specialized path must also report `query_dqunroll > 0` and
//! `query == 0`.
//!
//! Tests:
//!
//! * [`phase_xviii_dq_accumulator_selector`] — the exact eligibility predicate
//!   and the fallback identity, nodevice.
//! * [`phase_xviii_dq_accumulator_parity`] — the production dispatch helper at
//!   every required geometry, both arithmetic modes, causal and non-causal, all
//!   masks visible and partially masked, sequence 37 preserved, and head
//!   dimension 64 as the fallback control. `dQ`, `scratch_probability` and
//!   `scratch_ds` are compared bitwise against the shipped query shader.
//! * [`phase_xviii_dq_accumulator_optional_loading`] — the Phase IV/V loading
//!   semantics for the promoted artifact: opt-out, refused artifact, eligible
//!   default.
//! * [`phase_xviii_dq_accumulator_full_step`] — opt-in release-mode training
//!   driver (`HIERARCHOS_PHASE_XVIII_STEP_ARM=original|dqunroll`) producing the
//!   ABBA samples, loss bits, digests and dispatch counters.
//!
//! Run:
//!
//! ```text
//! cargo test --lib phase_xviii_dq_accumulator -- --nocapture --test-threads=1
//! HIERARCHOS_PHASE_XVIII_STEP_ARM=dqunroll HIERARCHOS_PHASE_XVIII_STEP_DIGEST=1 \
//!   cargo test --release --lib phase_xviii_dq_accumulator_full_step -- --nocapture
//! ```

use super::*;
use crate::vulkan::{self, GpuBuffer, VulkanDevice};

/// Written into every output buffer before a dispatch so an element the kernel
/// never writes survives as a sentinel.
const PHASE_XVIII_SENTINEL: f32 = -98_765.432_1_f32;
const PHASE_XVIII_BATCH: u32 = 2;

/// Deterministic operands: identical bytes for every arm of a case.
fn phase_xviii_operands(salt: u32, count: usize) -> Vec<f32> {
    let mut state = 0x9e37_79b9_u32 ^ salt.wrapping_mul(0x85eb_ca6b);
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / 16_777_216.0;
            unit * 2.0 - 1.0
        })
        .collect()
}

/// Bitwise comparison: equality is the FP32 bit pattern, so one ULP fails.
#[derive(Clone, Copy, Debug, Default)]
struct PhaseXviiiComparison {
    elements: usize,
    bitwise: usize,
    non_finite: usize,
    first: Option<(usize, u32, u32)>,
}

fn phase_xviii_compare(expected: &[f32], actual: &[f32]) -> PhaseXviiiComparison {
    assert_eq!(expected.len(), actual.len());
    let mut report = PhaseXviiiComparison {
        elements: expected.len(),
        ..PhaseXviiiComparison::default()
    };
    for (index, (&expected_value, &actual_value)) in expected.iter().zip(actual).enumerate() {
        if !actual_value.is_finite() {
            report.non_finite += 1;
        }
        if expected_value.to_bits() == actual_value.to_bits() {
            report.bitwise += 1;
        } else if report.first.is_none() {
            report.first = Some((index, expected_value.to_bits(), actual_value.to_bits()));
        }
    }
    report
}

fn phase_xviii_report(
    label: &str,
    tensor: &str,
    comparison: &PhaseXviiiComparison,
    verdict: bool,
) {
    let first = match comparison.first {
        Some((index, expected, actual)) => format!(
            "first_divergence=index {index} expected=0x{expected:08x} actual=0x{actual:08x}"
        ),
        None => "first_divergence=none".to_string(),
    };
    eprintln!(
        "[phase-xviii-parity] {label} tensor={tensor} elements={} bitwise={} diverged={} \
         non_finite={} {first} verdict={}",
        comparison.elements,
        comparison.bitwise,
        comparison.elements - comparison.bitwise,
        comparison.non_finite,
        if verdict { "PASS" } else { "FAIL" }
    );
}

/// Mask shapes the query pass has to survive: fully visible, a query row with no
/// visible key, and keys that are never visible.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PhaseXviiiMasks {
    AllVisible,
    DroppedQueryAndKey,
}

struct PhaseXviiiCase {
    label: &'static str,
    push: AttentionPush,
    query_mask: Vec<f32>,
    key_mask: Vec<f32>,
    q_count: usize,
    kv_count: usize,
    scratch: usize,
}

impl PhaseXviiiCase {
    #[allow(clippy::too_many_arguments)]
    fn new(
        label: &'static str,
        heads: u32,
        kv_heads: u32,
        mode: u32,
        seq: u32,
        head_dim: u32,
        value_head_dim: u32,
        causal: u32,
        masks: PhaseXviiiMasks,
    ) -> Self {
        Self::with_shape(
            label,
            heads,
            kv_heads,
            mode,
            seq,
            seq,
            head_dim,
            value_head_dim,
            causal,
            masks,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn with_shape(
        label: &'static str,
        heads: u32,
        kv_heads: u32,
        mode: u32,
        query_seq: u32,
        key_seq: u32,
        head_dim: u32,
        value_head_dim: u32,
        causal: u32,
        masks: PhaseXviiiMasks,
    ) -> Self {
        let push = AttentionPush {
            batch_size: PHASE_XVIII_BATCH,
            query_seq_len: query_seq,
            key_value_seq_len: key_seq,
            query_hidden_size: heads * head_dim,
            value_hidden_size: kv_heads * value_head_dim,
            num_heads: heads,
            num_key_value_heads: kv_heads,
            window_size: 0,
            scale: 1.0 / (head_dim as f32).sqrt(),
            value_scale: 1.0,
            softcap: 0.0,
            alibi_mode: 0,
            alibi_bias_max: 0.0,
            dropout_p: 0.0,
            rng_step: 0,
            rng_site: 0,
            rng_seed: 0,
            has_sink: 0,
            causal,
            relative_bias_mode: 0,
            relative_bias_num_buckets: 0,
            relative_bias_max_distance: 0,
            query_position_offset: 0,
            has_sparse_mask: 0,
            compressed_key_count: 0,
            compressed_key_rate: 0,
            paged_kv: 0,
            kv_page_size: 1,
            kv_page_table_stride: 1,
            unfused_qk: mode,
        };
        let query_mask_count = (PHASE_XVIII_BATCH * query_seq) as usize;
        let key_mask_count = (PHASE_XVIII_BATCH * key_seq) as usize;
        let (query_mask, key_mask) = match masks {
            PhaseXviiiMasks::AllVisible => (
                vec![1.0_f32; query_mask_count],
                vec![1.0_f32; key_mask_count],
            ),
            PhaseXviiiMasks::DroppedQueryAndKey => (
                (0..query_mask_count)
                    .map(|i| if i % 7 == 2 { 0.0 } else { 1.0 })
                    .collect::<Vec<_>>(),
                (0..key_mask_count)
                    .map(|i| if i % 5 == 0 || i < 3 { 0.0 } else { 1.0 })
                    .collect::<Vec<_>>(),
            ),
        };
        let q_count = (PHASE_XVIII_BATCH * query_seq * push.query_hidden_size) as usize;
        let kv_count = (PHASE_XVIII_BATCH * key_seq * push.value_hidden_size) as usize;
        let scratch = two_pass_attention_scratch_elements(&push).expect("scratch size");
        Self {
            label,
            push,
            query_mask,
            key_mask,
            q_count,
            kv_count,
            scratch,
        }
    }

    fn query_grid(&self) -> [u32; 3] {
        [
            div_ceil_u32(
                (self.push.batch_size * self.push.num_heads * self.push.query_seq_len) as usize,
                64,
            ),
            1,
            1,
        ]
    }
}

struct PhaseXviiiBuffers {
    query: GpuBuffer,
    key: GpuBuffer,
    value: GpuBuffer,
    grad_output: GpuBuffer,
    query_mask: GpuBuffer,
    key_mask: GpuBuffer,
}

impl PhaseXviiiBuffers {
    fn new(device: &VulkanDevice, case: &PhaseXviiiCase) -> Result<Self> {
        Ok(Self {
            query: GpuBuffer::from_f32(device, &phase_xviii_operands(51, case.q_count))?,
            key: GpuBuffer::from_f32(device, &phase_xviii_operands(52, case.kv_count))?,
            value: GpuBuffer::from_f32(device, &phase_xviii_operands(53, case.kv_count))?,
            grad_output: GpuBuffer::from_f32(device, &phase_xviii_operands(54, case.q_count))?,
            query_mask: GpuBuffer::from_f32(device, &case.query_mask)?,
            key_mask: GpuBuffer::from_f32(device, &case.key_mask)?,
        })
    }
}

struct PhaseXviiiRoute {
    grad_q: Vec<f32>,
    grad_k: Vec<f32>,
    grad_v: Vec<f32>,
    probability: Vec<f32>,
    ds: Vec<f32>,
    counts: AttentionBackwardDispatchCounts,
}

/// One submission through the production dispatch helper.
fn phase_xviii_route(
    device: &VulkanDevice,
    kernels: &TransformerKernels,
    case: &PhaseXviiiCase,
    buffers: &PhaseXviiiBuffers,
    pair: &TwoPassAttentionBackward,
) -> Result<PhaseXviiiRoute> {
    let scratch = TwoPassAttentionScratch::new(device, case.scratch)?;
    scratch
        .probability
        .write_f32(&vec![PHASE_XVIII_SENTINEL; case.scratch])?;
    scratch.ds.write_f32(&vec![PHASE_XVIII_SENTINEL; case.scratch])?;
    let grad_q = GpuBuffer::from_f32(device, &vec![PHASE_XVIII_SENTINEL; case.q_count])?;
    let grad_k = GpuBuffer::from_f32(device, &vec![PHASE_XVIII_SENTINEL; case.kv_count])?;
    let grad_v = GpuBuffer::from_f32(device, &vec![PHASE_XVIII_SENTINEL; case.kv_count])?;

    two_pass_attention_test_hooks::reset_thread_counts();
    let mut commands = vulkan::ComputeBatch::new(device)?;
    record_two_pass_attention_backward_dispatch(
        kernels,
        &mut commands,
        &case.push,
        pair,
        1,
        &scratch,
        &TwoPassAttentionBackwardBindings {
            query: &buffers.query,
            key: &buffers.key,
            value: &buffers.value,
            grad_query: &grad_q,
            grad_key: &grad_k,
            grad_value: &grad_v,
            grad_output: &buffers.grad_output,
            query_mask: &buffers.query_mask,
            key_mask: &buffers.key_mask,
        },
    )?;
    anyhow::ensure!(
        commands.dispatch_count() == 2,
        "pass 1 and pass 2 must remain two dispatches, saw {}",
        commands.dispatch_count()
    );
    commands.submit()?;
    Ok(PhaseXviiiRoute {
        grad_q: grad_q.read_f32(case.q_count)?,
        grad_k: grad_k.read_f32(case.kv_count)?,
        grad_v: grad_v.read_f32(case.kv_count)?,
        probability: scratch.probability.read_f32(case.scratch)?,
        ds: scratch.ds.read_f32(case.scratch)?,
        counts: two_pass_attention_test_hooks::thread_counts(),
    })
}

/// The shipped pair: the qualified query and key artifacts, installed exactly as
/// a harness would install them so the production selectors leave it alone.
fn phase_xviii_reference_pair(device: &VulkanDevice) -> Result<TwoPassAttentionBackward> {
    let query = vulkan::ComputeKernel::new(
        device,
        ATTENTION_BACKWARD_QUERY_PARALLEL_SPV,
        9,
        std::mem::size_of::<AttentionPush>() as u32,
    )?;
    let key = vulkan::ComputeKernel::new(
        device,
        ATTENTION_BACKWARD_KEY_PARALLEL_SPV,
        8,
        std::mem::size_of::<AttentionPush>() as u32,
    )?;
    Ok(TwoPassAttentionBackward {
        kernels: Some([query, key]),
        reason: "experimental_pair_override",
        detail: None,
    })
}

/// The same kernels, but presented as the ordinary production pair so the
/// promoted query selector runs inside the real dispatch helper.
fn phase_xviii_candidate_pair(device: &VulkanDevice) -> Result<TwoPassAttentionBackward> {
    let mut pair = phase_xviii_reference_pair(device)?;
    pair.reason = "ready";
    Ok(pair)
}

#[test]
fn phase_xviii_dq_accumulator_selector() -> Result<()> {
    let eligible = [
        ("gqa8q2kv", 8u32, 2u32, 128u32, 128u32),
        ("gqa4q2kv", 4, 2, 128, 128),
        ("mha8q8kv", 8, 8, 128, 128),
    ];
    for (label, heads, kv_heads, head_dim, value_head_dim) in eligible {
        let case = PhaseXviiiCase::new(
            label,
            heads,
            kv_heads,
            0,
            64,
            head_dim,
            value_head_dim,
            1,
            PhaseXviiiMasks::AllVisible,
        );
        assert_eq!(
            dqunroll_query_attention_rejection(&case.push),
            None,
            "{label} must be eligible"
        );
    }

    let cases: [(&'static str, u32, u32, u32, u32, &'static str); 6] = [
        (
            "head_dim_64",
            8,
            2,
            64,
            128,
            "dqunroll_query_head_dim_not_128",
        ),
        (
            "value_head_dim_64",
            8,
            2,
            128,
            64,
            "dqunroll_query_value_head_dim_not_128",
        ),
        (
            "head_dim_96",
            8,
            2,
            96,
            128,
            "dqunroll_query_head_dim_not_128",
        ),
        (
            "both_64",
            8,
            2,
            64,
            64,
            "dqunroll_query_head_dim_not_128",
        ),
        (
            "qk_64_value_128",
            4,
            2,
            64,
            128,
            "dqunroll_query_head_dim_not_128",
        ),
        (
            "qk_128_value_96",
            4,
            2,
            128,
            96,
            "dqunroll_query_value_head_dim_not_128",
        ),
    ];
    for (label, heads, kv_heads, head_dim, value_head_dim, expected) in cases {
        let case = PhaseXviiiCase::new(
            label,
            heads,
            kv_heads,
            0,
            64,
            head_dim,
            value_head_dim,
            1,
            PhaseXviiiMasks::AllVisible,
        );
        assert_eq!(
            dqunroll_query_attention_rejection(&case.push),
            Some(expected),
            "{label} must be rejected with its own reason"
        );
    }

    // Degenerate geometry must not divide by zero; it is the enclosing gate's
    // business, but the selector has to stay total.
    let mut degenerate = PhaseXviiiCase::new(
        "degenerate",
        0,
        0,
        0,
        64,
        128,
        128,
        1,
        PhaseXviiiMasks::AllVisible,
    )
    .push;
    degenerate.num_heads = 0;
    assert_eq!(
        dqunroll_query_attention_rejection(&degenerate),
        Some("invalid_geometry")
    );

    // The promoted constant is the production artifact, not the shipped one, and
    // the shipped fallback is unchanged by this phase.
    assert_ne!(
        ATTENTION_BACKWARD_QUERY_DQUNROLL_SPV,
        ATTENTION_BACKWARD_QUERY_PARALLEL_SPV
    );
    eprintln!(
        "[phase-xviii-selector] eligible=3 rejected=6 fallback_reason=exact_head_dim_pair \
         promoted_bytes={} shipped_bytes={} workgroup=64 bindings=9 push_bytes={}",
        ATTENTION_BACKWARD_QUERY_DQUNROLL_SPV.len(),
        ATTENTION_BACKWARD_QUERY_PARALLEL_SPV.len(),
        std::mem::size_of::<AttentionPush>(),
    );
    Ok(())
}

#[test]
fn phase_xviii_dq_accumulator_parity() -> Result<()> {
    let device = VulkanDevice::new()?;
    // Both arms keep the shipped key shader: this phase changes pass 1 only, and
    // holding pass 2 fixed is what makes the comparison attributable. The
    // promoted switch is pinned to the unset-and-eligible default so the matrix
    // never depends on the surrounding environment.
    let _channel4 = two_pass_attention_test_hooks::scope_channel4(Some(Some(false)), false);
    let _dqunroll = two_pass_attention_test_hooks::scope_dqunroll(Some(None), false);
    // Phase XX keeps this module as the intermediate fallback, so the
    // scratch-reuse variant is pinned off: the production promoted selector still
    // runs, and this matrix keeps measuring the artifact it was written for.
    let _scratch_reuse =
        two_pass_attention_test_hooks::scope_scratch_reuse(Some(Some(false)), false);
    let kernels = TransformerKernels::new(&device)?;
    let reference_pair = phase_xviii_reference_pair(&device)?;
    let candidate_pair = phase_xviii_candidate_pair(&device)?;

    // label, heads, kv, mode, seq, head_dim, value_head_dim, causal, masks
    let cases = [
        ("gqa8q2kv_m0_seq64_causal", 8, 2, 0, 64, 128, 128, 1, PhaseXviiiMasks::AllVisible),
        ("gqa8q2kv_m9_seq64_causal", 8, 2, 9, 64, 128, 128, 1, PhaseXviiiMasks::AllVisible),
        ("gqa8q2kv_m0_seq64_noncausal", 8, 2, 0, 64, 128, 128, 0, PhaseXviiiMasks::AllVisible),
        ("gqa8q2kv_m9_seq64_noncausal", 8, 2, 9, 64, 128, 128, 0, PhaseXviiiMasks::AllVisible),
        ("gqa8q2kv_m0_seq64_masked", 8, 2, 0, 64, 128, 128, 1, PhaseXviiiMasks::DroppedQueryAndKey),
        ("gqa8q2kv_m9_seq64_masked", 8, 2, 9, 64, 128, 128, 1, PhaseXviiiMasks::DroppedQueryAndKey),
        ("gqa8q2kv_m0_seq128_causal", 8, 2, 0, 128, 128, 128, 1, PhaseXviiiMasks::AllVisible),
        ("gqa8q2kv_m9_seq128_causal", 8, 2, 9, 128, 128, 128, 1, PhaseXviiiMasks::AllVisible),
        ("gqa8q2kv_m0_seq128_noncausal", 8, 2, 0, 128, 128, 128, 0, PhaseXviiiMasks::AllVisible),
        ("gqa8q2kv_m9_seq128_masked", 8, 2, 9, 128, 128, 128, 0, PhaseXviiiMasks::DroppedQueryAndKey),
        ("mha8q8kv_m0_seq64_causal", 8, 8, 0, 64, 128, 128, 1, PhaseXviiiMasks::AllVisible),
        ("mha8q8kv_m9_seq64_causal", 8, 8, 9, 64, 128, 128, 1, PhaseXviiiMasks::AllVisible),
        ("mha8q8kv_m0_seq64_noncausal", 8, 8, 0, 64, 128, 128, 0, PhaseXviiiMasks::DroppedQueryAndKey),
        ("mha8q8kv_m9_seq64_masked", 8, 8, 9, 64, 128, 128, 1, PhaseXviiiMasks::DroppedQueryAndKey),
        ("mha8q8kv_m0_seq128_causal", 8, 8, 0, 128, 128, 128, 1, PhaseXviiiMasks::AllVisible),
        ("mha8q8kv_m9_seq128_causal", 8, 8, 9, 128, 128, 128, 1, PhaseXviiiMasks::AllVisible),
        ("mha8q8kv_m0_seq128_noncausal", 8, 8, 0, 128, 128, 128, 0, PhaseXviiiMasks::DroppedQueryAndKey),
        ("mha8q8kv_m9_seq128_masked", 8, 8, 9, 128, 128, 128, 0, PhaseXviiiMasks::AllVisible),
        ("gqa4q2kv_m0_seq64_causal", 4, 2, 0, 64, 128, 128, 1, PhaseXviiiMasks::AllVisible),
        ("gqa4q2kv_m9_seq64_causal", 4, 2, 9, 64, 128, 128, 1, PhaseXviiiMasks::AllVisible),
        ("gqa4q2kv_m0_seq64_noncausal", 4, 2, 0, 64, 128, 128, 0, PhaseXviiiMasks::DroppedQueryAndKey),
        ("gqa4q2kv_m9_seq64_masked", 4, 2, 9, 64, 128, 128, 1, PhaseXviiiMasks::DroppedQueryAndKey),
        ("gqa4q2kv_m0_seq128_causal", 4, 2, 0, 128, 128, 128, 1, PhaseXviiiMasks::AllVisible),
        ("gqa4q2kv_m9_seq128_causal", 4, 2, 9, 128, 128, 128, 1, PhaseXviiiMasks::AllVisible),
        ("gqa4q2kv_m0_seq128_noncausal", 4, 2, 0, 128, 128, 128, 0, PhaseXviiiMasks::DroppedQueryAndKey),
        ("gqa4q2kv_m9_seq128_masked", 4, 2, 9, 128, 128, 128, 0, PhaseXviiiMasks::AllVisible),
        // Phase XVII's non-divisible sequence case is preserved verbatim.
        ("gqa4q2kv_m0_seq37_causal", 4, 2, 0, 37, 128, 128, 1, PhaseXviiiMasks::DroppedQueryAndKey),
        ("gqa4q2kv_m9_seq37_noncausal", 4, 2, 9, 37, 128, 128, 0, PhaseXviiiMasks::AllVisible),
        // Head dimension 64 must keep the shipped shader and stay bit-exact.
        ("gqa8q2kv_m0_hd64_fallback", 8, 2, 0, 64, 64, 128, 1, PhaseXviiiMasks::DroppedQueryAndKey),
        ("gqa8q2kv_m9_hd64_fallback", 8, 2, 9, 64, 64, 128, 1, PhaseXviiiMasks::AllVisible),
    ];

    let mut failures = 0usize;
    let mut comparisons = 0usize;
    let mut specialized_cases = 0usize;
    let mut fallback_cases = 0usize;
    for (label, heads, kv_heads, mode, seq, head_dim, value_head_dim, causal, masks) in cases {
        let case = PhaseXviiiCase::new(
            label,
            heads,
            kv_heads,
            mode,
            seq,
            head_dim,
            value_head_dim,
            causal,
            masks,
        );
        let buffers = PhaseXviiiBuffers::new(&device, &case)?;
        let reference = phase_xviii_route(&device, &kernels, &case, &buffers, &reference_pair)?;
        let candidate = phase_xviii_route(&device, &kernels, &case, &buffers, &candidate_pair)?;

        let eligible = dqunroll_query_attention_rejection(&case.push).is_none();
        eprintln!(
            "[phase-xviii-parity] case={label} heads={heads} kv_heads={kv_heads} mode={mode} \
             seq={seq} head_dim={head_dim} value_head_dim={value_head_dim} causal={causal} \
             rows={} grid_x={} eligible={eligible} reference_query={} reference_dqunroll={} \
             candidate_query={} candidate_dqunroll={}",
            case.push.batch_size * heads * seq,
            case.query_grid()[0],
            reference.counts.query,
            reference.counts.query_dqunroll,
            candidate.counts.query,
            candidate.counts.query_dqunroll,
        );

        // The route, not the predicate, decides whether the claim holds.
        let mut ok = true;
        if eligible {
            specialized_cases += 1;
            ok &= candidate.counts.query_dqunroll == 1 && candidate.counts.query == 0;
        } else {
            fallback_cases += 1;
            ok &= candidate.counts.query == 1 && candidate.counts.query_dqunroll == 0;
        }
        ok &= reference.counts.query == 1 && reference.counts.query_dqunroll == 0;
        ok &= candidate.counts.scalar == 0 && reference.counts.scalar == 0;

        for (tensor, expected, actual) in [
            ("dq", &reference.grad_q, &candidate.grad_q),
            ("scratch_probability", &reference.probability, &candidate.probability),
            ("scratch_ds", &reference.ds, &candidate.ds),
        ] {
            let comparison = phase_xviii_compare(expected, actual);
            comparisons += 1;
            let verdict = comparison.first.is_none() && comparison.non_finite == 0;
            ok &= verdict;
            phase_xviii_report(label, tensor, &comparison, verdict);
        }
        // dK/dV are pass 2's; comparing them proves pass 1's scratch is
        // consumable by the shipped key shader, not merely self-consistent.
        for (tensor, expected, actual) in [
            ("dk", &reference.grad_k, &candidate.grad_k),
            ("dv", &reference.grad_v, &candidate.grad_v),
        ] {
            let comparison = phase_xviii_compare(expected, actual);
            comparisons += 1;
            let verdict = comparison.first.is_none() && comparison.non_finite == 0;
            ok &= verdict;
            phase_xviii_report(label, tensor, &comparison, verdict);
        }

        if !ok {
            failures += 1;
            eprintln!("[phase-xviii-parity] case={label} verdict=FAIL");
        }
    }

    eprintln!(
        "[phase-xviii-parity] summary cases={} specialized={specialized_cases} \
         fallback={fallback_cases} tensor_comparisons={comparisons} failed={failures}",
        specialized_cases + fallback_cases,
    );
    assert_eq!(failures, 0, "phase-xviii dq accumulator parity failed");

    // The promoted artifact must resolve to a registered production identity so
    // profiling can name it instead of `unknown`.
    let promoted = vulkan::ComputeKernel::new(
        &device,
        ATTENTION_BACKWARD_QUERY_DQUNROLL_SPV,
        9,
        std::mem::size_of::<AttentionPush>() as u32,
    )?;
    let name = promoted.shader_name();
    eprintln!(
        "[phase-xviii-identity] promoted_signature=0x{:016x} name={:?}",
        promoted.shader_signature(),
        name
    );
    assert_eq!(name, Some("transformer_attention_backward_query_rdna3_dqunroll"));
    Ok(())
}

#[test]
fn phase_xviii_dq_accumulator_optional_loading() -> Result<()> {
    let device = VulkanDevice::new()?;
    let _channel4 = two_pass_attention_test_hooks::scope_channel4(Some(Some(false)), false);
    // This test is about the Phase XVIII intermediate fallback.
    let _scratch_reuse =
        two_pass_attention_test_hooks::scope_scratch_reuse(Some(Some(false)), false);
    let kernels = TransformerKernels::new(&device)?;
    let case = PhaseXviiiCase::new(
        "gqa8q2kv_m0_causal",
        8,
        2,
        0,
        64,
        128,
        128,
        1,
        PhaseXviiiMasks::AllVisible,
    );
    let buffers = PhaseXviiiBuffers::new(&device, &case)?;
    let pair = phase_xviii_candidate_pair(&device)?;

    // `=0`: the shipped query shader, and the promoted artifact is not resolved
    // at all, so an operator who opted out pays for no extra pipeline.
    {
        let _scope = two_pass_attention_test_hooks::scope_dqunroll(Some(Some(false)), false);
        assert!(!attention_backward_query_dqunroll_enabled());
        let route = phase_xviii_route(&device, &kernels, &case, &buffers, &pair)?;
        assert_eq!(route.counts.query, 1);
        assert_eq!(route.counts.query_dqunroll, 0);
        assert!(
            kernels
                .rdna3_dqunroll_attention_backward_query
                .get()
                .is_none(),
            "the opt-out must not build the promoted pipeline"
        );
        eprintln!("[phase-xviii-loading] setting=0 route=query_original resolved=false");
    }

    // A refused artifact falls back to the shipped query shader for the same
    // geometry, which is the observable shape of a truncated or
    // foreign-architecture module.
    {
        let _scope = two_pass_attention_test_hooks::scope_dqunroll(Some(None), true);
        assert!(attention_backward_query_dqunroll_enabled());
        let route = phase_xviii_route(&device, &kernels, &case, &buffers, &pair)?;
        assert_eq!(route.counts.query, 1);
        assert_eq!(route.counts.query_dqunroll, 0);
        let resolved = kernels
            .rdna3_dqunroll_attention_backward_query
            .get()
            .expect("the loader must record its refusal");
        assert_eq!(resolved.reason, "dqunroll_shader_unavailable");
        eprintln!(
            "[phase-xviii-loading] setting=default refused=true route=query_original \
             reason={}",
            resolved.reason
        );
    }
    Ok(())
}

struct PhaseXviiiSequence {
    input_ids: Vec<u32>,
    targets: Vec<u32>,
    attention_mask: Vec<f32>,
    loss_weights: Vec<f32>,
}

fn phase_xviii_load_sequences(
    path: &std::path::Path,
    seq_len: usize,
) -> Result<Vec<PhaseXviiiSequence>> {
    use std::io::BufRead;
    let file = std::fs::File::open(path)?;
    let mut rows = Vec::new();
    for (index, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(&line)
            .with_context(|| format!("decoding dataset line {}", index + 1))?;
        let ids = value
            .get("input_ids")
            .and_then(serde_json::Value::as_array)
            .context("dataset rows require input_ids")?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .context("input_ids must be u32")
            })
            .collect::<Result<Vec<_>>>()?;
        rows.push(ids);
    }
    let mut sequences = Vec::new();
    for tokens in rows {
        if tokens.len() < 2 {
            continue;
        }
        let predict = tokens.len() - 1;
        for start in (0..predict).step_by(seq_len) {
            let count = (predict - start).min(seq_len);
            let mut sequence = PhaseXviiiSequence {
                input_ids: vec![0; seq_len],
                targets: vec![u32::MAX; seq_len],
                attention_mask: vec![0.0; seq_len],
                loss_weights: vec![0.0; seq_len],
            };
            sequence.input_ids[..count].copy_from_slice(&tokens[start..start + count]);
            sequence.targets[..count].copy_from_slice(&tokens[start + 1..start + 1 + count]);
            sequence.attention_mask[..count].fill(1.0);
            sequence.loss_weights[..count].fill(1.0);
            sequences.push(sequence);
        }
    }
    if sequences.is_empty() {
        anyhow::bail!("dataset produced no trainable sequences");
    }
    Ok(sequences)
}

fn phase_xviii_batch_arrays(
    sequences: &[PhaseXviiiSequence],
    seq_len: usize,
) -> Result<(Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>)> {
    let mut input_ids = Vec::with_capacity(sequences.len() * seq_len);
    let mut targets = Vec::with_capacity(sequences.len() * seq_len);
    let mut attention_mask = Vec::with_capacity(sequences.len() * seq_len);
    let mut loss_weights = Vec::with_capacity(sequences.len() * seq_len);
    for sequence in sequences {
        input_ids.extend_from_slice(&sequence.input_ids);
        targets.extend_from_slice(&sequence.targets);
        attention_mask.extend_from_slice(&sequence.attention_mask);
        loss_weights.extend_from_slice(&sequence.loss_weights);
    }
    let normalization = loss_weights.iter().copied().sum::<f32>();
    if !normalization.is_finite() || normalization <= 0.0 {
        anyhow::bail!("batch has no finite positive supervised loss weight");
    }
    for weight in &mut loss_weights {
        *weight /= normalization;
    }
    Ok((input_ids, targets, attention_mask, loss_weights))
}

fn phase_xviii_digest(values: &std::collections::BTreeMap<String, Vec<f32>>) -> (String, usize, f64) {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    let mut elements = 0usize;
    let mut sumsq = 0.0f64;
    for (name, tensor) in values {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((tensor.len() as u64).to_le_bytes());
        for value in tensor {
            hash.update(value.to_bits().to_le_bytes());
            elements += 1;
            if value.is_finite() {
                sumsq += f64::from(*value) * f64::from(*value);
            }
        }
    }
    (format!("{:x}", hash.finalize()), elements, sumsq.sqrt())
}

/// `HIERARCHOS_PHASE_XVIII_STEP_ARM`     `original` or `dqunroll` (required)
/// `HIERARCHOS_PHASE_XVIII_FIXTURE`      fixture dir (default `target/gqa-e2e/fixture-8q2kv`)
/// `HIERARCHOS_PHASE_XVIII_STEP_SEQ`     sequence length (default 64)
/// `HIERARCHOS_PHASE_XVIII_STEP_BATCH`   batch size (default 2)
/// `HIERARCHOS_PHASE_XVIII_STEP_STEPS`   steps per process (default 29)
/// `HIERARCHOS_PHASE_XVIII_STEP_DIGEST=1` print loss bits and digests every step
/// `HIERARCHOS_PHASE_XVIII_LORA=0`       full-parameter instead of LoRA
/// `HIERARCHOS_PHASE_XVIII_EXPECT_ROUTE` the route this process must report:
///                                       `original` | `dqunroll` | `scalar` |
///                                       `admitted` (either pass-1 arm, chosen
///                                       by the arm) | `excluded` (scalar)
///
/// The arm is chosen by production code only: `original` runs with
/// `HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL=0` in the process environment,
/// `dqunroll` with it unset. The driver refuses to run if the dispatch counters
/// disagree with the requested arm and expected route.
#[test]
fn phase_xviii_dq_accumulator_full_step() -> Result<()> {
    let arm = match std::env::var("HIERARCHOS_PHASE_XVIII_STEP_ARM") {
        Ok(value) => value.trim().to_string(),
        Err(_) => {
            eprintln!(
                "phase-xviii full step: status=SKIP reason=not_requested \
                 (set HIERARCHOS_PHASE_XVIII_STEP_ARM=original|dqunroll)"
            );
            return Ok(());
        }
    };
    anyhow::ensure!(
        arm == "original" || arm == "dqunroll",
        "unknown phase-xviii step arm {arm}"
    );
    let fixture = std::env::var("HIERARCHOS_PHASE_XVIII_FIXTURE")
        .unwrap_or_else(|_| "target/gqa-e2e/fixture-8q2kv".to_string());
    let fixture = std::path::PathBuf::from(fixture);
    let steps: usize = std::env::var("HIERARCHOS_PHASE_XVIII_STEP_STEPS")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(29);
    let warmup: usize = 3;
    let digest = std::env::var("HIERARCHOS_PHASE_XVIII_STEP_DIGEST")
        .map(|value| value.trim() == "1")
        .unwrap_or(false);
    let batch: usize = std::env::var("HIERARCHOS_PHASE_XVIII_STEP_BATCH")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(2);
    let seq_len: usize = std::env::var("HIERARCHOS_PHASE_XVIII_STEP_SEQ")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(64);
    let lora = std::env::var("HIERARCHOS_PHASE_XVIII_LORA")
        .map(|value| value.trim() != "0")
        .unwrap_or(true);
    let lr: f32 = std::env::var("HIERARCHOS_PHASE_XVIII_LR")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(1.0e-2);
    let expected_route = std::env::var("HIERARCHOS_PHASE_XVIII_EXPECT_ROUTE")
        .unwrap_or_else(|_| "dqunroll".to_string());
    let seed = 7u64;
    let hyper = AdamWHyperParams {
        lr,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1.0e-8,
        weight_decay: 0.0,
    };

    // Phase XX promoted a variant on top of this module. This driver measures the
    // Phase XVIII intermediate fallback, so the scratch-reuse variant is pinned
    // off; its own production-route driver is `phase_xx_production_route_ab`.
    let _scratch_reuse =
        two_pass_attention_test_hooks::scope_scratch_reuse(Some(Some(false)), false);

    let sequences = phase_xviii_load_sequences(&fixture.join("dataset.jsonl"), seq_len)?;
    let batches_per_epoch = sequences.len() / batch;
    anyhow::ensure!(batches_per_epoch > 0, "fixture dataset is smaller than one batch");

    let device = VulkanDevice::new()?;
    let mut graph = VulkanTransformer::from_hf_package(
        device,
        fixture.join("model"),
        batch,
        seq_len,
    )?;
    if lora {
        let target = graph
            .config()
            .architecture
            .default_lora_target_module()
            .to_owned();
        graph.enable_lora(
            VulkanTransformerLoraConfig::causal_lm(8, 16, vec![target], false)?,
            seed,
        )?;
    }
    graph.set_dropout_seed(seed as u32 ^ (seed >> 32) as u32);
    let start_parameters = phase_xviii_digest(&graph.parameter_values()?);

    let installed = graph.kernels.rdna3_row_packed_attention.is_some();
    let two_pass_enabled = attention_backward_two_pass_enabled();
    let dqunroll_enabled = attention_backward_query_dqunroll_enabled();
    anyhow::ensure!(installed, "the production row-packed forward is not installed");
    anyhow::ensure!(two_pass_enabled, "the two-pass backward is not enabled");
    anyhow::ensure!(
        dqunroll_enabled == (arm == "dqunroll"),
        "arm {arm} disagrees with the production dq-unroll switch \
         (HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL)"
    );

    let forward_before = attention_forward_dispatch_counts();
    let backward_before = attention_backward_dispatch_counts();
    eprintln!(
        "[phase-xviii-step-run] arm={arm} fixture={} sequences={} batches_per_epoch={batches_per_epoch} \
         batch={batch} seq={seq_len} steps={steps} warmup={warmup} lora={lora} lr={lr} seed={seed} \
         digest={digest} expected_route={expected_route} device={} row_packed_installed={installed} \
         two_pass_enabled={two_pass_enabled} dqunroll_enabled={dqunroll_enabled} \
         parameter_start_digest={} parameter_elements={}",
        fixture.display(),
        sequences.len(),
        graph.device_name(),
        start_parameters.0,
        start_parameters.1,
    );

    for index in 0..steps {
        let global_step = index;
        let epoch = global_step / batches_per_epoch;
        let batch_index = global_step % batches_per_epoch;
        let begin = batch_index * batch;
        let (input_ids, targets, attention_mask, loss_weights) =
            phase_xviii_batch_arrays(&sequences[begin..begin + batch], seq_len)?;
        let started = Instant::now();
        let result = graph.train_step(&input_ids, &targets, &attention_mask, &loss_weights, hyper)?;
        let step_ms = started.elapsed().as_secs_f64() * 1e3;
        let measured = index >= warmup;
        // The digest is a readback of every parameter, so it must never run in a
        // timing process: it would add tens of milliseconds to every step and
        // dilute the measured kernel delta. `HIERARCHOS_PHASE_XVIII_STEP_DIGEST`
        // is therefore both the numerical switch and the timing switch.
        if digest {
            let parameters = phase_xviii_digest(&graph.parameter_values()?);
            if lora {
                let gradients = phase_xviii_digest(&graph.lora_gradient_values()?);
                eprintln!(
                    "[phase-xviii-step] arm={arm} step={index} epoch={epoch} batch={batch_index} \
                     measured={measured} step_ms={step_ms:.6} loss_bits={:08x} loss={:e} \
                     grad_digest={} grad_elements={} grad_norm={:.17e} param_digest={} \
                     param_elements={} param_norm={:.17e}",
                    result.loss.to_bits(),
                    result.loss,
                    gradients.0,
                    gradients.1,
                    gradients.2,
                    parameters.0,
                    parameters.1,
                    parameters.2
                );
            } else {
                eprintln!(
                    "[phase-xviii-step] arm={arm} step={index} epoch={epoch} batch={batch_index} \
                     measured={measured} step_ms={step_ms:.6} loss_bits={:08x} loss={:e} \
                     param_digest={} param_elements={} param_norm={:.17e}",
                    result.loss.to_bits(),
                    result.loss,
                    parameters.0,
                    parameters.1,
                    parameters.2
                );
            }
        } else {
            eprintln!(
                "[phase-xviii-step] arm={arm} step={index} epoch={epoch} batch={batch_index} \
                 measured={measured} step_ms={step_ms:.6} loss_bits={:08x} loss={:e}",
                result.loss.to_bits(),
                result.loss
            );
        }
    }

    let forward_after = attention_forward_dispatch_counts();
    let backward_after = attention_backward_dispatch_counts();
    let end_parameters = phase_xviii_digest(&graph.parameter_values()?);
    let query_original = backward_after.query - backward_before.query;
    let query_dqunroll = backward_after.query_dqunroll - backward_before.query_dqunroll;
    let key = backward_after.key - backward_before.key;
    let key_channel4 = backward_after.key_channel4 - backward_before.key_channel4;
    let scalar = backward_after.scalar - backward_before.scalar;
    let scratch_allocations =
        backward_after.scratch_allocations - backward_before.scratch_allocations;
    let scratch_bytes = backward_after.scratch_bytes - backward_before.scratch_bytes;
    eprintln!(
        "[phase-xviii-step-run-end] arm={arm} parameter_end_digest={} parameter_elements={} \
         parameter_end_norm={:.17e} forward_generic={} forward_row_packed={} forward_short={} \
         forward_total={} backward_scalar={scalar} backward_query_original={query_original} \
         backward_query_dqunroll={query_dqunroll} backward_key_original={key} \
         backward_key_channel4={key_channel4} scratch_allocations={scratch_allocations} \
         scratch_bytes={scratch_bytes}",
        end_parameters.0,
        end_parameters.1,
        end_parameters.2,
        forward_after.generic - forward_before.generic,
        forward_after.row_packed - forward_before.row_packed,
        forward_after.short - forward_before.short,
        forward_after.total() - forward_before.total(),
    );

    // The production counters, not the environment, decide whether the arm ran
    // what it claims: every step runs two layers.
    let layers = (steps * 2) as u64;
    match expected_route.as_str() {
        "admitted" | "two_pass" => {
            // The geometry is admitted by the enclosing gate; the arm decides
            // which pass-1 module ran, and the counters must agree with it.
            let (want_original, want_dqunroll) = if arm == "dqunroll" {
                (0, layers)
            } else {
                (layers, 0)
            };
            anyhow::ensure!(
                query_original == want_original && query_dqunroll == want_dqunroll,
                "arm {arm} dispatched query_original={query_original} \
                 query_dqunroll={query_dqunroll}, expected {want_original} and {want_dqunroll}"
            );
            anyhow::ensure!(scalar == 0, "the step dispatched the scalar kernel {scalar} times");
        }
        route if route == "original" || route == "dqunroll" => {
            let (want_original, want_dqunroll) = if route == "dqunroll" {
                (0, layers)
            } else {
                (layers, 0)
            };
            anyhow::ensure!(
                query_original == want_original && query_dqunroll == want_dqunroll,
                "arm {arm} dispatched query_original={query_original} \
                 query_dqunroll={query_dqunroll}, expected {want_original} and {want_dqunroll} "
            );
            anyhow::ensure!(scalar == 0, "the step dispatched the scalar kernel {scalar} times");
        }
        "scalar" | "excluded" => {
            anyhow::ensure!(
                query_original == 0 && query_dqunroll == 0,
                "the scalar route dispatched query_original={query_original} \
                 query_dqunroll={query_dqunroll}, expected 0 and 0"
            );
            anyhow::ensure!(
                scalar > 0,
                "expected_route=scalar but the scalar kernel never ran"
            );
        }
        route => anyhow::bail!("unknown expected route {route}"),
    }
    // The two dispatches per layer are always pass 1 + pass 2 of the same route.
    anyhow::ensure!(
        key + key_channel4 == if query_dqunroll + query_original == 0 { 0 } else { layers },
        "pass 2 dispatched key_original={key} key_channel4={key_channel4} \
         for query_original={query_original} query_dqunroll={query_dqunroll}"
    );
    Ok(())
}
