//! Phase XIX — Candidate A: query-backward intermediate score / softmax-term
//! reuse through the two scratch planes (test builds only).
//!
//! Phase XVII measured a register-resident dQ accumulator and declined this
//! candidate on the grounds that the query shader's dynamic-index `dq[128]`
//! access was the dominant bottleneck. Phase XVIII promoted that accumulator, so
//! the query kernel's resource profile changed and the candidate is re-measured
//! here against the *promoted* shader, not against the pre-promotion one.
//!
//! Arm A is `shaders/transformer_attention_backward_query_rdna3_dqunroll.spv`
//! (the Phase XVIII production artifact, `f0214b60…`). Arm B is
//! `validation/phase_xix_query_scratch_reuse_20261127/shaders/q_scratchreuse.spv`
//! (`a6b3d6c6…`), which is that same module's source with the score, the
//! probability and `dp` retained in `scratch_probability` / `scratch_ds` instead
//! of being recomputed in later traversals.
//!
//! The candidate was promoted in Phase XX; these tests keep their historical
//! helper-level shape and read the *sealed* Phase XIX bytes, which Phase XX
//! asserts are byte-identical to the promoted production artifact
//! (`shaders/transformer_attention_backward_query_rdna3_dqunroll_scratchreuse.spv`,
//! identity `transformer_attention_backward_query_rdna3_dqunroll_scratchreuse`).
//!
//! Nothing here is reachable from a production build through this module:
//!
//! * the module is declared `#[cfg(test)]`,
//! * the candidate SPIR-V is embedded with `include_bytes!` from `validation/`,
//!   not from `shaders/`,
//! * arm B is installed through a `#[cfg(test)]` seam that substitutes the bytes
//!   the *promoted* loader builds, so the production selector, the device
//!   qualification, the dispatch counters, the pass-1 grid, the bindings, the
//!   push block and the pass-2 key route are all exactly the production ones and
//!   the only difference between the arms is the module.
//!
//! Tests:
//!
//! * [`phase_xix_scratch_reuse_parity`] — bitwise comparison of `dQ`,
//!   `scratch_probability`, `scratch_ds`, `dK` and `dV` between the two modules
//!   through the production dispatch helper, across both arithmetic modes,
//!   causal and non-causal ranges, masked and ordinary inputs, GQA and MHA
//!   groupings, sequence 64/128 and the non-divisible sequence 37.
//! * [`phase_xix_scratch_reuse_kernel_timing`] — opt-in isolated pass-1 timing
//!   from GPU timestamps, ABBA per mode. No profiler.
//! * [`phase_xix_scratch_reuse_full_step`] — opt-in release-mode complete-step
//!   driver (`HIERARCHOS_PHASE_XIX_STEP_ARM=promoted|scratch`) producing the
//!   ABBA samples, loss bits, digests and production dispatch counters.
//!
//! Run:
//!
//! ```text
//! cargo test --lib phase_xix_scratch_reuse -- --nocapture --test-threads=1
//! HIERARCHOS_PHASE_XIX_KERNEL_TIMING=1 \
//!   cargo test --release --lib phase_xix_scratch_reuse_kernel_timing -- --nocapture
//! HIERARCHOS_PHASE_XIX_STEP_ARM=scratch HIERARCHOS_PHASE_XIX_STEP_DIGEST=1 \
//!   cargo test --release --lib phase_xix_scratch_reuse_full_step -- --nocapture
//! ```

use super::*;
use crate::vulkan::{self, GpuBuffer, VulkanDevice};

/// Arm A: the Phase XVIII production artifact, embedded exactly as production
/// embeds it.
const PHASE_XIX_PROMOTED_QUERY: &[u8] =
    include_bytes!("../shaders/transformer_attention_backward_query_rdna3_dqunroll.spv");

/// Arm B: the same source plus the scratch-backed intermediate reuse. It lives
/// in `validation/` and has no production identity.
const PHASE_XIX_CANDIDATE_QUERY: &[u8] = include_bytes!(
    "../validation/phase_xix_query_scratch_reuse_20261127/shaders/q_scratchreuse.spv"
);

/// Written into every output buffer before a dispatch so an element the kernel
/// never writes survives as a sentinel.
const PHASE_XIX_SENTINEL: f32 = -98_765.432_1_f32;
const PHASE_XIX_BATCH: u32 = 2;

/// Deterministic operands: identical bytes for every arm of a case.
fn phase_xix_operands(salt: u32, count: usize) -> Vec<f32> {
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
struct PhaseXixComparison {
    elements: usize,
    bitwise: usize,
    non_finite: usize,
    first: Option<(usize, u32, u32)>,
}

fn phase_xix_compare(expected: &[f32], actual: &[f32]) -> PhaseXixComparison {
    assert_eq!(expected.len(), actual.len());
    let mut report = PhaseXixComparison {
        elements: expected.len(),
        ..PhaseXixComparison::default()
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

fn phase_xix_report(
    label: &str,
    tensor: &str,
    comparison: &PhaseXixComparison,
    verdict: bool,
) {
    let first = match comparison.first {
        Some((index, expected, actual)) => format!(
            "first_divergence=index {index} expected=0x{expected:08x} actual=0x{actual:08x}"
        ),
        None => "first_divergence=none".to_string(),
    };
    eprintln!(
        "[phase-xix-parity] {label} tensor={tensor} elements={} bitwise={} diverged={} \
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
enum PhaseXixMasks {
    AllVisible,
    DroppedQueryAndKey,
}

struct PhaseXixCase {
    label: &'static str,
    push: AttentionPush,
    query_mask: Vec<f32>,
    key_mask: Vec<f32>,
    q_count: usize,
    kv_count: usize,
    scratch: usize,
}

impl PhaseXixCase {
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
        masks: PhaseXixMasks,
    ) -> Self {
        let push = AttentionPush {
            batch_size: PHASE_XIX_BATCH,
            query_seq_len: seq,
            key_value_seq_len: seq,
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
        let query_mask_count = (PHASE_XIX_BATCH * seq) as usize;
        let key_mask_count = (PHASE_XIX_BATCH * seq) as usize;
        let (query_mask, key_mask) = match masks {
            PhaseXixMasks::AllVisible => (
                vec![1.0_f32; query_mask_count],
                vec![1.0_f32; key_mask_count],
            ),
            PhaseXixMasks::DroppedQueryAndKey => (
                (0..query_mask_count)
                    .map(|i| if i % 7 == 2 { 0.0 } else { 1.0 })
                    .collect::<Vec<_>>(),
                (0..key_mask_count)
                    .map(|i| if i % 5 == 0 || i < 3 { 0.0 } else { 1.0 })
                    .collect::<Vec<_>>(),
            ),
        };
        let q_count = (PHASE_XIX_BATCH * seq * push.query_hidden_size) as usize;
        let kv_count = (PHASE_XIX_BATCH * seq * push.value_hidden_size) as usize;
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

struct PhaseXixBuffers {
    query: GpuBuffer,
    key: GpuBuffer,
    value: GpuBuffer,
    grad_output: GpuBuffer,
    query_mask: GpuBuffer,
    key_mask: GpuBuffer,
}

impl PhaseXixBuffers {
    fn new(device: &VulkanDevice, case: &PhaseXixCase) -> Result<Self> {
        Ok(Self {
            query: GpuBuffer::from_f32(device, &phase_xix_operands(51, case.q_count))?,
            key: GpuBuffer::from_f32(device, &phase_xix_operands(52, case.kv_count))?,
            value: GpuBuffer::from_f32(device, &phase_xix_operands(53, case.kv_count))?,
            grad_output: GpuBuffer::from_f32(device, &phase_xix_operands(54, case.q_count))?,
            query_mask: GpuBuffer::from_f32(device, &case.query_mask)?,
            key_mask: GpuBuffer::from_f32(device, &case.key_mask)?,
        })
    }
}

struct PhaseXixRoute {
    grad_q: Vec<f32>,
    grad_k: Vec<f32>,
    grad_v: Vec<f32>,
    probability: Vec<f32>,
    ds: Vec<f32>,
    counts: AttentionBackwardDispatchCounts,
}

/// One submission through the production dispatch helper.
fn phase_xix_route(
    device: &VulkanDevice,
    kernels: &TransformerKernels,
    case: &PhaseXixCase,
    buffers: &PhaseXixBuffers,
    pair: &TwoPassAttentionBackward,
) -> Result<PhaseXixRoute> {
    let scratch = TwoPassAttentionScratch::new(device, case.scratch)?;
    scratch
        .probability
        .write_f32(&vec![PHASE_XIX_SENTINEL; case.scratch])?;
    scratch.ds.write_f32(&vec![PHASE_XIX_SENTINEL; case.scratch])?;
    let grad_q = GpuBuffer::from_f32(device, &vec![PHASE_XIX_SENTINEL; case.q_count])?;
    let grad_k = GpuBuffer::from_f32(device, &vec![PHASE_XIX_SENTINEL; case.kv_count])?;
    let grad_v = GpuBuffer::from_f32(device, &vec![PHASE_XIX_SENTINEL; case.kv_count])?;

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
    Ok(PhaseXixRoute {
        grad_q: grad_q.read_f32(case.q_count)?,
        grad_k: grad_k.read_f32(case.kv_count)?,
        grad_v: grad_v.read_f32(case.kv_count)?,
        probability: scratch.probability.read_f32(case.scratch)?,
        ds: scratch.ds.read_f32(case.scratch)?,
        counts: two_pass_attention_test_hooks::thread_counts(),
    })
}

fn phase_xix_build_kernel(device: &VulkanDevice, spirv: &'static [u8]) -> Result<vulkan::ComputeKernel> {
    vulkan::ComputeKernel::new(
        device,
        spirv,
        9,
        std::mem::size_of::<AttentionPush>() as u32,
    )
}

/// A harness pair whose pass 1 is `query` and whose pass 2 is the shipped key
/// shader. `reason != "ready"` makes the helper dispatch `kernels[0]` verbatim,
/// so both arms take the identical route and only the module differs.
fn phase_xix_pair(
    device: &VulkanDevice,
    query: &'static [u8],
) -> Result<TwoPassAttentionBackward> {
    let query = phase_xix_build_kernel(device, query)?;
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

#[test]
fn phase_xix_scratch_reuse_parity() -> Result<()> {
    let device = VulkanDevice::new()?;
    // Both arms keep the shipped key shader, so the comparison is attributable to
    // pass 1, and both keep the promoted switch at its unset-and-eligible default
    // so the matrix never depends on the surrounding environment.
    let _channel4 = two_pass_attention_test_hooks::scope_channel4(Some(Some(false)), false);
    let _dqunroll = two_pass_attention_test_hooks::scope_dqunroll(Some(None), false);
    let kernels = TransformerKernels::new(&device)?;
    let control_pair = phase_xix_pair(&device, PHASE_XIX_PROMOTED_QUERY)?;
    let candidate_pair = phase_xix_pair(&device, PHASE_XIX_CANDIDATE_QUERY)?;

    eprintln!(
        "[phase-xix-parity] device={} control_bytes={} candidate_bytes={} \
         control_kernel=transformer_attention_backward_query_rdna3_dqunroll \
         candidate_kernel=phase_xix_candidate_scratchreuse",
        device.name(),
        ATTENTION_BACKWARD_QUERY_DQUNROLL_SPV.len(),
        PHASE_XIX_CANDIDATE_QUERY.len(),
    );

    // label, heads, kv, mode, seq, head_dim, value_head_dim, causal, masks
    let cases = [
        ("gqa8q2kv_m0_seq64_causal", 8, 2, 0, 64, 128, 128, 1, PhaseXixMasks::AllVisible),
        ("gqa8q2kv_m9_seq64_causal", 8, 2, 9, 64, 128, 128, 1, PhaseXixMasks::AllVisible),
        ("gqa8q2kv_m0_seq64_noncausal", 8, 2, 0, 64, 128, 128, 0, PhaseXixMasks::AllVisible),
        ("gqa8q2kv_m9_seq64_noncausal", 8, 2, 9, 64, 128, 128, 0, PhaseXixMasks::AllVisible),
        ("gqa8q2kv_m0_seq64_masked", 8, 2, 0, 64, 128, 128, 1, PhaseXixMasks::DroppedQueryAndKey),
        ("gqa8q2kv_m9_seq64_masked", 8, 2, 9, 64, 128, 128, 1, PhaseXixMasks::DroppedQueryAndKey),
        ("gqa8q2kv_m0_seq128_causal", 8, 2, 0, 128, 128, 128, 1, PhaseXixMasks::AllVisible),
        ("gqa8q2kv_m9_seq128_causal", 8, 2, 9, 128, 128, 128, 1, PhaseXixMasks::AllVisible),
        ("gqa8q2kv_m0_seq128_noncausal", 8, 2, 0, 128, 128, 128, 0, PhaseXixMasks::AllVisible),
        ("gqa8q2kv_m9_seq128_masked", 8, 2, 9, 128, 128, 128, 0, PhaseXixMasks::DroppedQueryAndKey),
        ("mha8q8kv_m0_seq64_causal", 8, 8, 0, 64, 128, 128, 1, PhaseXixMasks::AllVisible),
        ("mha8q8kv_m9_seq64_causal", 8, 8, 9, 64, 128, 128, 1, PhaseXixMasks::AllVisible),
        ("mha8q8kv_m0_seq64_noncausal", 8, 8, 0, 64, 128, 128, 0, PhaseXixMasks::DroppedQueryAndKey),
        ("mha8q8kv_m9_seq64_masked", 8, 8, 9, 64, 128, 128, 1, PhaseXixMasks::DroppedQueryAndKey),
        ("mha8q8kv_m0_seq128_causal", 8, 8, 0, 128, 128, 128, 1, PhaseXixMasks::AllVisible),
        ("mha8q8kv_m9_seq128_causal", 8, 8, 9, 128, 128, 128, 1, PhaseXixMasks::AllVisible),
        ("mha8q8kv_m0_seq128_noncausal", 8, 8, 0, 128, 128, 128, 0, PhaseXixMasks::DroppedQueryAndKey),
        ("mha8q8kv_m9_seq128_masked", 8, 8, 9, 128, 128, 128, 0, PhaseXixMasks::AllVisible),
        ("gqa4q2kv_m0_seq64_causal", 4, 2, 0, 64, 128, 128, 1, PhaseXixMasks::AllVisible),
        ("gqa4q2kv_m9_seq64_causal", 4, 2, 9, 64, 128, 128, 1, PhaseXixMasks::AllVisible),
        ("gqa4q2kv_m0_seq64_masked", 4, 2, 0, 64, 128, 128, 0, PhaseXixMasks::DroppedQueryAndKey),
        ("gqa4q2kv_m9_seq64_masked", 4, 2, 9, 64, 128, 128, 1, PhaseXixMasks::DroppedQueryAndKey),
        ("gqa4q2kv_m0_seq128_causal", 4, 2, 0, 128, 128, 128, 1, PhaseXixMasks::AllVisible),
        ("gqa4q2kv_m9_seq128_masked", 4, 2, 9, 128, 128, 128, 0, PhaseXixMasks::AllVisible),
        // Non-divisible sequence, preserved from Phase XVII through Phase XVIII.
        ("gqa4q2kv_m0_seq37_causal", 4, 2, 0, 37, 128, 128, 1, PhaseXixMasks::DroppedQueryAndKey),
        ("gqa4q2kv_m9_seq37_noncausal", 4, 2, 9, 37, 128, 128, 0, PhaseXixMasks::AllVisible),
        // Head dimension 64: both arms take the same generic body.
        ("gqa8q2kv_m0_hd64_generic", 8, 2, 0, 64, 64, 128, 1, PhaseXixMasks::DroppedQueryAndKey),
        ("gqa8q2kv_m9_hd64_generic", 8, 2, 9, 64, 64, 128, 1, PhaseXixMasks::AllVisible),
    ];

    let mut failures = 0usize;
    let mut comparisons = 0usize;
    let mut specialized_cases = 0usize;
    let mut generic_cases = 0usize;
    for (label, heads, kv_heads, mode, seq, head_dim, value_head_dim, causal, masks) in cases {
        let case = PhaseXixCase::new(
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
        let buffers = PhaseXixBuffers::new(&device, &case)?;
        let control = phase_xix_route(&device, &kernels, &case, &buffers, &control_pair)?;
        let candidate = phase_xix_route(&device, &kernels, &case, &buffers, &candidate_pair)?;

        // NOTE: this is only the head-dimension half of the production gate. The
        // enclosing two-pass gate (MHA bypass, `QUALIFIED_GROUPED_QUERY_GROUPINGS`,
        // sequence, causal) is deliberately bypassed by the harness pair, so this
        // value does not mean "a production step reaches this case".
        let head_dim_matches = dqunroll_query_attention_rejection(&case.push).is_none();
        eprintln!(
            "[phase-xix-parity] case={label} heads={heads} kv_heads={kv_heads} mode={mode} \
             seq={seq} head_dim={head_dim} value_head_dim={value_head_dim} causal={causal} \
             rows={} grid_x={} dqunroll_head_dim_selector_admits={head_dim_matches} \
             control_query={} control_dqunroll={} candidate_query={} candidate_dqunroll={}",
            case.push.batch_size * heads * seq,
            case.query_grid()[0],
            control.counts.query,
            control.counts.query_dqunroll,
            candidate.counts.query,
            candidate.counts.query_dqunroll,
        );

        // Both arms run the same route: one pass-1 dispatch of the harness's own
        // kernel plus the shipped pass 2. Neither selector may fire, because a
        // harness pair is never silently replaced by a production artifact.
        let mut ok = control.counts.query == 1
            && control.counts.query_dqunroll == 0
            && candidate.counts.query == 1
            && candidate.counts.query_dqunroll == 0
            && control.counts.scalar == 0
            && candidate.counts.scalar == 0
            && control.counts.key == 1
            && candidate.counts.key == 1;
        if head_dim_matches {
            specialized_cases += 1;
        } else {
            generic_cases += 1;
        }

        for (tensor, expected, actual) in [
            ("dq", &control.grad_q, &candidate.grad_q),
            ("scratch_probability", &control.probability, &candidate.probability),
            ("scratch_ds", &control.ds, &candidate.ds),
        ] {
            let comparison = phase_xix_compare(expected, actual);
            comparisons += 1;
            let verdict = comparison.first.is_none() && comparison.non_finite == 0;
            ok &= verdict;
            phase_xix_report(label, tensor, &comparison, verdict);
        }
        // dK/dV are pass 2's; comparing them proves the candidate's scratch is
        // consumable by the shipped key shader, not merely self-consistent.
        for (tensor, expected, actual) in [
            ("dk", &control.grad_k, &candidate.grad_k),
            ("dv", &control.grad_v, &candidate.grad_v),
        ] {
            let comparison = phase_xix_compare(expected, actual);
            comparisons += 1;
            let verdict = comparison.first.is_none() && comparison.non_finite == 0;
            ok &= verdict;
            phase_xix_report(label, tensor, &comparison, verdict);
        }

        if !ok {
            failures += 1;
            eprintln!("[phase-xix-parity] case={label} verdict=FAIL");
        }
    }

    eprintln!(
        "[phase-xix-parity] summary cases={} specialized={specialized_cases} \
         generic={generic_cases} tensor_comparisons={comparisons} failed={failures}",
        specialized_cases + generic_cases,
    );

    // The candidate must be a different module from the Phase XVIII one, or the
    // comparison above would be vacuous.
    assert_ne!(PHASE_XIX_PROMOTED_QUERY, PHASE_XIX_CANDIDATE_QUERY);
    // Promotion identity: the sealed Phase XIX bytes are exactly the Phase XX
    // production artifact, so the evidence in this module covers that binary.
    assert_eq!(
        PHASE_XIX_CANDIDATE_QUERY,
        ATTENTION_BACKWARD_QUERY_DQUNROLL_SCRATCH_REUSE_SPV,
        "the sealed Phase XIX candidate is not the promoted production artifact"
    );
    let candidate = phase_xix_build_kernel(&device, PHASE_XIX_CANDIDATE_QUERY)?;
    let promoted = phase_xix_build_kernel(&device, ATTENTION_BACKWARD_QUERY_DQUNROLL_SPV)?;
    eprintln!(
        "[phase-xix-identity] promoted_signature=0x{:016x} promoted_name={:?} \
         candidate_signature=0x{:016x} candidate_name={:?}",
        promoted.shader_signature(),
        promoted.shader_name(),
        candidate.shader_signature(),
        candidate.shader_name()
    );
    assert_eq!(
        promoted.shader_name(),
        Some("transformer_attention_backward_query_rdna3_dqunroll")
    );
    assert_eq!(
        candidate.shader_name(),
        Some("transformer_attention_backward_query_rdna3_dqunroll_scratchreuse"),
        "the promoted artifact must resolve to its own production profiler identity"
    );
    assert_ne!(
        candidate.shader_signature(),
        promoted.shader_signature(),
        "the promoted artifact must not share the Phase XVIII module signature"
    );

    assert_eq!(failures, 0, "phase-xix scratch-reuse parity failed");
    Ok(())
}

/// Median GPU-timestamp milliseconds for one pass-1 module at one case.
fn phase_xix_median_ms(
    device: &VulkanDevice,
    kernel: &vulkan::ComputeKernel,
    case: &PhaseXixCase,
    buffers: &PhaseXixBuffers,
    repeats: usize,
) -> Result<f64> {
    let grad_q = GpuBuffer::from_f32(device, &vec![0.0_f32; case.q_count])?;
    let grad_k = GpuBuffer::from_f32(device, &vec![0.0_f32; case.kv_count])?;
    let grad_v = GpuBuffer::from_f32(device, &vec![0.0_f32; case.kv_count])?;
    let probability = GpuBuffer::from_f32(device, &vec![0.0_f32; case.scratch])?;
    let ds = GpuBuffer::from_f32(device, &vec![0.0_f32; case.scratch])?;
    let total_ms = device.time_compute_batch_ms(|commands| {
        for _ in 0..repeats {
            kernel.record_dispatch(
                commands,
                &[
                    &buffers.query,
                    &buffers.key,
                    &buffers.value,
                    &buffers.query_mask,
                    &buffers.key_mask,
                    &buffers.grad_output,
                    &grad_q,
                    &probability,
                    &ds,
                ],
                bytemuck::bytes_of(&case.push),
                case.query_grid(),
            )?;
        }
        Ok(())
    })?;
    Ok(total_ms / repeats as f64)
}

#[test]
fn phase_xix_scratch_reuse_kernel_timing() -> Result<()> {
    let requested = std::env::var("HIERARCHOS_PHASE_XIX_KERNEL_TIMING")
        .map(|value| value.trim() == "1")
        .unwrap_or(false);
    if !requested {
        eprintln!(
            "phase-xix kernel timing: status=SKIP reason=not_requested \
             (set HIERARCHOS_PHASE_XIX_KERNEL_TIMING=1)"
        );
        return Ok(());
    }
    let device = VulkanDevice::new()?;
    eprintln!(
        "[phase-xix-timing] device={} note=isolated_pass1_only_no_profiler",
        device.name()
    );
    let promoted = phase_xix_build_kernel(&device, PHASE_XIX_PROMOTED_QUERY)?;
    let candidate = phase_xix_build_kernel(&device, PHASE_XIX_CANDIDATE_QUERY)?;
    let rounds: usize = std::env::var("HIERARCHOS_PHASE_XIX_KERNEL_ROUNDS")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(5);
    let repeats: usize = std::env::var("HIERARCHOS_PHASE_XIX_KERNEL_REPEATS")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(8);

    for mode in [0u32, 9u32] {
        let case = PhaseXixCase::new(
            "gqa8q2kv_seq64_causal",
            8,
            2,
            mode,
            64,
            128,
            128,
            1,
            PhaseXixMasks::AllVisible,
        );
        let buffers = PhaseXixBuffers::new(&device, &case)?;
        // Warm both pipelines so the first measured submission is not a compile.
        let _ = phase_xix_median_ms(&device, &promoted, &case, &buffers, repeats)?;
        let _ = phase_xix_median_ms(&device, &candidate, &case, &buffers, repeats)?;
        let mut ratios = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let (a, b) = if round % 2 == 0 {
                (
                    phase_xix_median_ms(&device, &promoted, &case, &buffers, repeats)?,
                    phase_xix_median_ms(&device, &candidate, &case, &buffers, repeats)?,
                )
            } else {
                let b = phase_xix_median_ms(&device, &candidate, &case, &buffers, repeats)?;
                let a = phase_xix_median_ms(&device, &promoted, &case, &buffers, repeats)?;
                (a, b)
            };
            eprintln!(
                "[phase-xix-timing] mode={mode} round={round} promoted_ms={a:.4} \
                 candidate_ms={b:.4} ratio={:.4}",
                b / a
            );
            ratios.push(b / a);
        }
        ratios.sort_by(f64::total_cmp);
        let median = ratios[ratios.len() / 2];
        eprintln!(
            "[phase-xix-timing] mode={mode} rounds={rounds} repeats={repeats} \
             median_ratio={median:.4} speedup={:.4} rows={} workgroups={}",
            1.0 / median,
            case.push.batch_size * case.push.num_heads * case.push.query_seq_len,
            case.query_grid()[0]
        );
    }
    Ok(())
}

pub(super) struct PhaseXixSequence {
    input_ids: Vec<u32>,
    targets: Vec<u32>,
    attention_mask: Vec<f32>,
    loss_weights: Vec<f32>,
}

pub(super) fn phase_xix_load_sequences(
    path: &std::path::Path,
    seq_len: usize,
) -> Result<Vec<PhaseXixSequence>> {
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
            let mut sequence = PhaseXixSequence {
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

pub(super) fn phase_xix_batch_arrays(
    sequences: &[PhaseXixSequence],
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

pub(super) fn phase_xix_digest(
    values: &std::collections::BTreeMap<String, Vec<f32>>,
) -> (String, usize, f64) {
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

/// `HIERARCHOS_PHASE_XIX_STEP_ARM`      `promoted` or `scratch` (required)
/// `HIERARCHOS_PHASE_XIX_FIXTURE`       fixture dir (default `target/gqa-e2e/fixture-8q2kv`)
/// `HIERARCHOS_PHASE_XIX_STEP_SEQ`      sequence length (default 64)
/// `HIERARCHOS_PHASE_XIX_STEP_BATCH`    batch size (default 2)
/// `HIERARCHOS_PHASE_XIX_STEP_STEPS`    steps per process (default 29)
/// `HIERARCHOS_PHASE_XIX_STEP_DIGEST=1` print loss bits and digests every step
/// `HIERARCHOS_PHASE_XIX_LORA=0`        full-parameter instead of LoRA
///
/// Both arms run the *production* route: the promoted selector, its device
/// qualification, its dispatch counter and the promoted pass-2 key artifact. The
/// only difference is the SPIR-V the promoted loader builds, which arm `scratch`
/// substitutes through the `#[cfg(test)]` seam. The driver refuses to run if the
/// production counters do not report one promoted pass-1 dispatch per layer.
#[test]
fn phase_xix_scratch_reuse_full_step() -> Result<()> {
    let arm = match std::env::var("HIERARCHOS_PHASE_XIX_STEP_ARM") {
        Ok(value) => value.trim().to_string(),
        Err(_) => {
            eprintln!(
                "phase-xix full step: status=SKIP reason=not_requested \
                 (set HIERARCHOS_PHASE_XIX_STEP_ARM=promoted|scratch)"
            );
            return Ok(());
        }
    };
    anyhow::ensure!(
        arm == "promoted" || arm == "scratch",
        "unknown phase-xix step arm {arm}"
    );
    // Phase XX promoted this candidate, so the production selector would pick the
    // scratch-reuse module by default in *both* arms. This historical driver
    // installs arm B through the Phase XIX test seam, so the Phase XX route is
    // pinned off and the two arms remain the two modules it was written to
    // compare. The promotion's own driver is `phase_xx_production_route_ab`.
    let _phase_xx_route =
        two_pass_attention_test_hooks::scope_scratch_reuse(Some(Some(false)), false);
    let _seam = if arm == "scratch" {
        eprintln!(
            "[phase-xix-step-seam] arm=scratch module=phase_xix_candidate_scratchreuse \
             bytes={} sha256_prefix=a6b3d6c6 production_selector_unchanged=true",
            PHASE_XIX_CANDIDATE_QUERY.len()
        );
        Some(two_pass_attention_test_hooks::scope_dqunroll_spirv(Some(
            PHASE_XIX_CANDIDATE_QUERY,
        )))
    } else {
        eprintln!(
            "[phase-xix-step-seam] arm=promoted \
             module=transformer_attention_backward_query_rdna3_dqunroll bytes={} \
             sha256_prefix=f0214b60 production_selector_unchanged=true",
            ATTENTION_BACKWARD_QUERY_DQUNROLL_SPV.len()
        );
        None
    };

    let fixture = std::env::var("HIERARCHOS_PHASE_XIX_FIXTURE")
        .unwrap_or_else(|_| "target/gqa-e2e/fixture-8q2kv".to_string());
    let fixture = std::path::PathBuf::from(fixture);
    let steps: usize = std::env::var("HIERARCHOS_PHASE_XIX_STEP_STEPS")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(29);
    let warmup: usize = 3;
    let digest = std::env::var("HIERARCHOS_PHASE_XIX_STEP_DIGEST")
        .map(|value| value.trim() == "1")
        .unwrap_or(false);
    let batch: usize = std::env::var("HIERARCHOS_PHASE_XIX_STEP_BATCH")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(2);
    let seq_len: usize = std::env::var("HIERARCHOS_PHASE_XIX_STEP_SEQ")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(64);
    let lora = std::env::var("HIERARCHOS_PHASE_XIX_LORA")
        .map(|value| value.trim() != "0")
        .unwrap_or(true);
    let lr: f32 = std::env::var("HIERARCHOS_PHASE_XIX_LR")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(1.0e-2);
    let seed = 7u64;
    let hyper = AdamWHyperParams {
        lr,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1.0e-8,
        weight_decay: 0.0,
    };

    let sequences = phase_xix_load_sequences(&fixture.join("dataset.jsonl"), seq_len)?;
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
    let start_parameters = phase_xix_digest(&graph.parameter_values()?);

    let installed = graph.kernels.rdna3_row_packed_attention.is_some();
    let two_pass_enabled = attention_backward_two_pass_enabled();
    let dqunroll_enabled = attention_backward_query_dqunroll_enabled();
    anyhow::ensure!(installed, "the production row-packed forward is not installed");
    anyhow::ensure!(two_pass_enabled, "the two-pass backward is not enabled");
    anyhow::ensure!(
        dqunroll_enabled,
        "the promoted dq-unroll selector must be at its production default in both arms"
    );

    let forward_before = attention_forward_dispatch_counts();
    let backward_before = attention_backward_dispatch_counts();
    eprintln!(
        "[phase-xix-step-run] arm={arm} fixture={} sequences={} batches_per_epoch={batches_per_epoch} \
         batch={batch} seq={seq_len} steps={steps} warmup={warmup} lora={lora} lr={lr} seed={seed} \
         digest={digest} device={} row_packed_installed={installed} \
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
            phase_xix_batch_arrays(&sequences[begin..begin + batch], seq_len)?;
        let started = Instant::now();
        let result = graph.train_step(&input_ids, &targets, &attention_mask, &loss_weights, hyper)?;
        let step_ms = started.elapsed().as_secs_f64() * 1e3;
        let measured = index >= warmup;
        // The digest reads every parameter back, so it must never run in a timing
        // process: `HIERARCHOS_PHASE_XIX_STEP_DIGEST` is both switches.
        if digest {
            let parameters = phase_xix_digest(&graph.parameter_values()?);
            if lora {
                let gradients = phase_xix_digest(&graph.lora_gradient_values()?);
                eprintln!(
                    "[phase-xix-step] arm={arm} step={index} epoch={epoch} batch={batch_index} \
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
                    "[phase-xix-step] arm={arm} step={index} epoch={epoch} batch={batch_index} \
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
                "[phase-xix-step] arm={arm} step={index} epoch={epoch} batch={batch_index} \
                 measured={measured} step_ms={step_ms:.6} loss_bits={:08x} loss={:e}",
                result.loss.to_bits(),
                result.loss
            );
        }
    }

    let forward_after = attention_forward_dispatch_counts();
    let backward_after = attention_backward_dispatch_counts();
    let end_parameters = phase_xix_digest(&graph.parameter_values()?);
    let query_original = backward_after.query - backward_before.query;
    let query_dqunroll = backward_after.query_dqunroll - backward_before.query_dqunroll;
    let key = backward_after.key - backward_before.key;
    let key_channel4 = backward_after.key_channel4 - backward_before.key_channel4;
    let scalar = backward_after.scalar - backward_before.scalar;
    let scratch_allocations =
        backward_after.scratch_allocations - backward_before.scratch_allocations;
    let scratch_bytes = backward_after.scratch_bytes - backward_before.scratch_bytes;
    eprintln!(
        "[phase-xix-step-run-end] arm={arm} parameter_end_digest={} parameter_elements={} \
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

    // The production counters decide whether each arm ran what it claims: two
    // layers per step, one pass 1 and one pass 2 each, and the promoted selector
    // in both arms. Only the module behind the promoted identity differs.
    let layers = (steps * 2) as u64;
    anyhow::ensure!(
        query_dqunroll == layers && query_original == 0,
        "arm {arm} dispatched query_original={query_original} query_dqunroll={query_dqunroll}, \
         expected 0 and {layers}"
    );
    anyhow::ensure!(scalar == 0, "the step dispatched the scalar kernel {scalar} times");
    anyhow::ensure!(
        key + key_channel4 == layers && key == 0,
        "pass 2 dispatched key_original={key} key_channel4={key_channel4}, expected 0 and {layers}"
    );
    Ok(())
}
