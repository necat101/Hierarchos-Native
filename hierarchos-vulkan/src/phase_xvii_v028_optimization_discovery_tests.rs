//! Phase XVII v0.2.8 optimization discovery — query-pass register-resident dQ
//! accumulator (test builds only).
//!
//! This module is the whole dispatch path of the experiment. Nothing here is
//! reachable from a production build: the module is declared `#[cfg(test)]`, the
//! candidate SPIR-V is a `#[cfg(test)]` artifact embedded with `include_bytes!`,
//! and the production query shader, the selectors, the two-pass gate and the
//! default route are untouched. The candidate source and artifact live in
//! `validation/phase_xvii_v028_optimization_discovery_20261126/` and are not in
//! `shaders/`.
//!
//! Two tests:
//!
//! * [`phase_xvii_query_dqunroll_parity`] — bitwise parity of `dQ`, the
//!   probability scratch and the dS scratch against the shipped v0.2.7 artifact,
//!   for both arithmetic modes, MHA and GQA groupings, causal and non-causal
//!   ranges, masked queries and keys, a partial-workgroup/boundary key length,
//!   and a non-128 head dimension that must fall through to the generic body.
//! * [`phase_xvii_query_dqunroll_timing`] — GPU-timestamp timing of the shipped
//!   artifact against the candidate at the production GQA geometry. Opt-in
//!   (`HIERARCHOS_PHASE_XVII_KERNEL_TIMING=1`), because a timing run must not be
//!   confused with a qualification run.
//!
//! Run:
//!
//! ```text
//! cargo test --lib phase_xvii_query_dqunroll_parity -- --nocapture --test-threads=1
//! HIERARCHOS_PHASE_XVII_KERNEL_TIMING=1 \
//!   cargo test --lib phase_xvii_query_dqunroll_timing -- --nocapture --test-threads=1
//! ```

use super::*;
use crate::vulkan::{self, GpuBuffer, VulkanDevice};

/// The shipped v0.2.7 pass-1 query artifact, embedded exactly as production
/// embeds it. This is the arm every measurement is taken against.
const PHASE_XVII_PRODUCTION_QUERY: &[u8] =
    include_bytes!("../shaders/transformer_attention_backward_query_rdna3.spv");

/// The production source rebuilt from this tree with the experiment's compiler
/// invocation. It is byte-identical in SPIR-V to the shipped artifact, which is
/// what makes the candidate's delta attributable to the source change rather
/// than to different toolchain flags.
const PHASE_XVII_REBUILT_QUERY: &[u8] = include_bytes!(
    "../validation/phase_xvii_v028_optimization_discovery_20261126/shaders/q_rebuild.spv"
);

/// The candidate: production arithmetic, dQ accumulator indexed only by literals
/// on the head-dimension-128 geometry.
const PHASE_XVII_CANDIDATE_QUERY: &[u8] = include_bytes!(
    "../validation/phase_xvii_v028_optimization_discovery_20261126/shaders/q_dqunroll.spv"
);

const PHASE_XVII_BATCH: u32 = 2;
const PHASE_XVII_SEQ: u32 = 64;
const PHASE_XVII_HEAD_DIM: u32 = 128;

/// Deterministic operands: identical bytes for every arm of a case.
fn phase_xvii_operands(salt: u32, count: usize) -> Vec<f32> {
    let mut state = 0x9e37_79b9_u32 ^ salt.wrapping_mul(0x85eb_ca6b);
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / 16_777_216.0;
            unit * 2.0 - 1.0
        })
        .collect()
}

struct PhaseXviiCase {
    label: &'static str,
    query_mask: Vec<f32>,
    key_mask: Vec<f32>,
    push: AttentionPush,
    q_count: usize,
    kv_count: usize,
    scratch: usize,
}

/// Mask shapes the query pass has to survive: fully visible, a query row with no
/// visible key, a first key that is never visible, and a sparse key mask.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PhaseXviiMasks {
    AllVisible,
    DroppedQueryAndKey,
}

impl PhaseXviiCase {
    fn new(
        label: &'static str,
        heads: u32,
        kv_heads: u32,
        mode: u32,
        seq: u32,
        head_dim: u32,
        causal: u32,
        masks: PhaseXviiMasks,
    ) -> Self {
        let push = AttentionPush {
            batch_size: PHASE_XVII_BATCH,
            query_seq_len: seq,
            key_value_seq_len: seq,
            query_hidden_size: heads * head_dim,
            value_hidden_size: kv_heads * head_dim,
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
        let mask_count = (PHASE_XVII_BATCH * seq) as usize;
        let (query_mask, key_mask) = match masks {
            PhaseXviiMasks::AllVisible => (vec![1.0_f32; mask_count], vec![1.0_f32; mask_count]),
            PhaseXviiMasks::DroppedQueryAndKey => (
                (0..mask_count)
                    .map(|i| if i % 7 == 2 { 0.0 } else { 1.0 })
                    .collect::<Vec<_>>(),
                (0..mask_count)
                    .map(|i| if i % 5 == 0 || i < 3 { 0.0 } else { 1.0 })
                    .collect::<Vec<_>>(),
            ),
        };
        let q_count = (PHASE_XVII_BATCH * seq * heads * head_dim) as usize;
        let kv_count = (PHASE_XVII_BATCH * seq * kv_heads * head_dim) as usize;
        let scratch = two_pass_attention_scratch_elements(&push).unwrap();
        Self {
            label,
            query_mask,
            key_mask,
            push,
            q_count,
            kv_count,
            scratch,
        }
    }

    fn query_grid(&self) -> [u32; 3] {
        [
            div_ceil_u32((PHASE_XVII_BATCH * self.push.num_heads * self.push.query_seq_len) as usize, 64),
            1,
            1,
        ]
    }

    fn rows(&self) -> usize {
        (PHASE_XVII_BATCH * self.push.num_heads * self.push.query_seq_len) as usize
    }
}

struct PhaseXviiBuffers {
    query: GpuBuffer,
    key: GpuBuffer,
    value: GpuBuffer,
    query_mask: GpuBuffer,
    key_mask: GpuBuffer,
    grad_output: GpuBuffer,
}

impl PhaseXviiBuffers {
    fn new(device: &VulkanDevice, case: &PhaseXviiCase) -> Result<Self> {
        Ok(Self {
            query: GpuBuffer::from_f32(device, &phase_xvii_operands(11, case.q_count))?,
            key: GpuBuffer::from_f32(device, &phase_xvii_operands(12, case.kv_count))?,
            value: GpuBuffer::from_f32(device, &phase_xvii_operands(13, case.kv_count))?,
            query_mask: GpuBuffer::from_f32(device, &case.query_mask)?,
            key_mask: GpuBuffer::from_f32(device, &case.key_mask)?,
            grad_output: GpuBuffer::from_f32(device, &phase_xvii_operands(14, case.q_count))?,
        })
    }
}

/// One query-pass submission. Returns the three outputs the pass writes.
fn phase_xvii_run(
    device: &VulkanDevice,
    kernel: &vulkan::ComputeKernel,
    case: &PhaseXviiCase,
    buffers: &PhaseXviiBuffers,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let grad_q = GpuBuffer::from_f32(device, &vec![0.0_f32; case.q_count])?;
    let probability = GpuBuffer::from_f32(device, &vec![0.0_f32; case.scratch])?;
    let ds = GpuBuffer::from_f32(device, &vec![0.0_f32; case.scratch])?;
    let mut commands = vulkan::ComputeBatch::new(device)?;
    kernel.record_dispatch(
        &mut commands,
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
    commands.submit()?;
    Ok((
        grad_q.read_f32(case.q_count)?,
        probability.read_f32(case.scratch)?,
        ds.read_f32(case.scratch)?,
    ))
}

/// Bitwise verdict for one output tensor.
fn phase_xvii_bitwise(label: &str, tensor: &str, expected: &[f32], actual: &[f32]) -> bool {
    assert_eq!(expected.len(), actual.len());
    let mut diverged = 0usize;
    let mut first: Option<(usize, u32, u32)> = None;
    let mut non_finite = 0usize;
    for (index, (&e, &a)) in expected.iter().zip(actual).enumerate() {
        if !a.is_finite() {
            non_finite += 1;
        }
        if e.to_bits() != a.to_bits() {
            diverged += 1;
            if first.is_none() {
                first = Some((index, e.to_bits(), a.to_bits()));
            }
        }
    }
    let detail = match first {
        Some((index, e, a)) => {
            format!("first_divergence=index {index} expected=0x{e:08x} actual=0x{a:08x}")
        }
        None => "first_divergence=none".to_string(),
    };
    eprintln!(
        "[phase-xvii-parity] {label} tensor={tensor} elements={} bitwise_equal={} \
         diverged={} non_finite={} {detail} verdict={}",
        expected.len(),
        expected.len() - diverged,
        diverged,
        non_finite,
        if diverged == 0 && non_finite == 0 { "PASS" } else { "FAIL" }
    );
    diverged == 0 && non_finite == 0
}

fn phase_xvii_build_kernel(device: &VulkanDevice, spirv: &'static [u8]) -> Result<vulkan::ComputeKernel> {
    vulkan::ComputeKernel::new(
        device,
        spirv,
        9,
        std::mem::size_of::<AttentionPush>() as u32,
    )
}

#[test]
fn phase_xvii_query_dqunroll_parity() -> Result<()> {
    let device = VulkanDevice::new()?;
    let production = phase_xvii_build_kernel(&device, PHASE_XVII_PRODUCTION_QUERY)?;
    let rebuilt = phase_xvii_build_kernel(&device, PHASE_XVII_REBUILT_QUERY)?;
    let candidate = phase_xvii_build_kernel(&device, PHASE_XVII_CANDIDATE_QUERY)?;

    // (label, heads, kv_heads, mode, seq, head_dim, causal, masks)
    let cases = [
        ("gqa8q2kv_m0_causal", 8u32, 2u32, 0u32, 64u32, 128u32, 1u32, PhaseXviiMasks::AllVisible),
        ("gqa8q2kv_m9_causal", 8, 2, 9, 64, 128, 1, PhaseXviiMasks::AllVisible),
        ("mha8q8kv_m9_causal", 8, 8, 9, 64, 128, 1, PhaseXviiMasks::AllVisible),
        ("mha8q8kv_m0_noncausal", 8, 8, 0, 64, 128, 0, PhaseXviiMasks::AllVisible),
        ("gqa8q2kv_m9_masked", 8, 2, 9, 64, 128, 1, PhaseXviiMasks::DroppedQueryAndKey),
        ("gqa8q2kv_m0_masked_noncausal", 8, 2, 0, 64, 128, 0, PhaseXviiMasks::DroppedQueryAndKey),
        ("gqa4q2kv_m0_seq37", 4, 2, 0, 37, 128, 1, PhaseXviiMasks::DroppedQueryAndKey),
        ("gqa4q2kv_m9_seq37", 4, 2, 9, 37, 128, 0, PhaseXviiMasks::AllVisible),
        // Head dimension 64: the candidate must fall through to the generic body.
        ("gqa8q2kv_hd64_m0", 8, 2, 0, 64, 64, 1, PhaseXviiMasks::DroppedQueryAndKey),
        ("gqa8q2kv_hd64_m9", 8, 2, 9, 64, 64, 1, PhaseXviiMasks::AllVisible),
    ];

    let mut failures = 0usize;
    for (label, heads, kv_heads, mode, seq, head_dim, causal, masks) in cases {
        let case = PhaseXviiCase::new(label, heads, kv_heads, mode, seq, head_dim, causal, masks);
        let buffers = PhaseXviiBuffers::new(&device, &case)?;
        let (gq_ref, p_ref, ds_ref) = phase_xvii_run(&device, &production, &case, &buffers)?;
        let (gq_reb, p_reb, ds_reb) = phase_xvii_run(&device, &rebuilt, &case, &buffers)?;
        let (gq_can, p_can, ds_can) = phase_xvii_run(&device, &candidate, &case, &buffers)?;

        eprintln!(
            "[phase-xvii-parity] case={label} heads={heads} kv_heads={kv_heads} mode={mode} \
             seq={seq} head_dim={head_dim} causal={causal} rows={} grid_x={}",
            case.rows(),
            case.query_grid()[0]
        );

        // The rebuilt control must reproduce the shipped artifact bit for bit,
        // otherwise the candidate's delta is not attributable to its source.
        let mut ok = true;
        ok &= phase_xvii_bitwise(label, "control_dq_rebuild_vs_shipped", &gq_ref, &gq_reb);
        ok &= phase_xvii_bitwise(label, "control_prob_rebuild_vs_shipped", &p_ref, &p_reb);
        ok &= phase_xvii_bitwise(label, "control_ds_rebuild_vs_shipped", &ds_ref, &ds_reb);

        ok &= phase_xvii_bitwise(label, "dq_candidate_vs_shipped", &gq_ref, &gq_can);
        ok &= phase_xvii_bitwise(label, "prob_candidate_vs_shipped", &p_ref, &p_can);
        ok &= phase_xvii_bitwise(label, "ds_candidate_vs_shipped", &ds_ref, &ds_can);

        if !ok {
            failures += 1;
        }
    }

    eprintln!("[phase-xvii-parity] summary cases={} failed={failures}", 10);
    assert_eq!(failures, 0, "phase-xvii query dq-unroll parity failed");
    Ok(())
}

/// Median GPU-timestamp milliseconds for one arm at one case.
fn phase_xvii_median_ms(
    device: &VulkanDevice,
    kernel: &vulkan::ComputeKernel,
    case: &PhaseXviiCase,
    buffers: &PhaseXviiBuffers,
    repeats: usize,
    rounds: usize,
) -> Result<f64> {
    // Warm the pipeline once so the first measured submission is not a compile.
    let _ = phase_xvii_run(device, kernel, case, buffers)?;
    let grad_q = GpuBuffer::from_f32(device, &vec![0.0_f32; case.q_count])?;
    let probability = GpuBuffer::from_f32(device, &vec![0.0_f32; case.scratch])?;
    let ds = GpuBuffer::from_f32(device, &vec![0.0_f32; case.scratch])?;
    let mut samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
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
        samples.push(total_ms / repeats as f64);
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples[samples.len() / 2])
}

#[test]
fn phase_xvii_query_dqunroll_timing() -> Result<()> {
    let requested = std::env::var("HIERARCHOS_PHASE_XVII_KERNEL_TIMING")
        .map(|value| value.trim() == "1")
        .unwrap_or(false);
    if !requested {
        eprintln!(
            "phase-xvii kernel timing: status=SKIP reason=not_requested \
             (set HIERARCHOS_PHASE_XVII_KERNEL_TIMING=1)"
        );
        return Ok(());
    }
    let device = VulkanDevice::new()?;
    eprintln!("[phase-xvii-timing] device={}", device.name());
    let production = phase_xvii_build_kernel(&device, PHASE_XVII_PRODUCTION_QUERY)?;
    let candidate = phase_xvii_build_kernel(&device, PHASE_XVII_CANDIDATE_QUERY)?;

    let repeats: usize = std::env::var("HIERARCHOS_PHASE_XVII_KERNEL_REPEATS")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(8);
    let rounds: usize = std::env::var("HIERARCHOS_PHASE_XVII_KERNEL_ROUNDS")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(5);

    for mode in [0u32, 9u32] {
        let case = PhaseXviiCase::new(
            "gqa8q2kv_production",
            8,
            2,
            mode,
            PHASE_XVII_SEQ,
            PHASE_XVII_HEAD_DIM,
            1,
            PhaseXviiMasks::AllVisible,
        );
        let buffers = PhaseXviiBuffers::new(&device, &case)?;
        // ABBA within each round so a slow drift affects both arms equally.
        let mut ratios = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let (a, b) = if round % 2 == 0 {
                (
                    phase_xvii_median_ms(&device, &production, &case, &buffers, repeats, 1)?,
                    phase_xvii_median_ms(&device, &candidate, &case, &buffers, repeats, 1)?,
                )
            } else {
                let b = phase_xvii_median_ms(&device, &candidate, &case, &buffers, repeats, 1)?;
                let a = phase_xvii_median_ms(&device, &production, &case, &buffers, repeats, 1)?;
                (a, b)
            };
            eprintln!(
                "[phase-xvii-timing] mode={mode} round={round} production_ms={a:.4} \
                 candidate_ms={b:.4} ratio={:.4}",
                b / a
            );
            ratios.push(b / a);
        }
        ratios.sort_by(f64::total_cmp);
        let median = ratios[ratios.len() / 2];
        eprintln!(
            "[phase-xvii-timing] mode={mode} rounds={rounds} repeats={repeats} \
             median_ratio={median:.4} speedup={:.4} rows={} workgroups={}",
            1.0 / median,
            case.rows(),
            case.query_grid()[0]
        );
    }
    Ok(())
}
