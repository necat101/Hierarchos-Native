//! Phase VIII — RDNA3 GQA key-pass key-block experiment (test builds only).
//!
//! This module is the experiment's entire dispatch path. Nothing here is
//! reachable from a production build: the module is declared `#[cfg(test)]`, the
//! candidate SPIR-V is a `#[cfg(test)]` artifact embedded with `include_bytes!`,
//! and the production selector, the production eligibility predicate, the
//! production key shader and the default route are untouched. The guest
//! candidate lives in `validation/phase_viii_key_blocks/shaders/` and is not in
//! `shaders/`.
//!
//! Two tests:
//!
//! * [`phase_viii_key_block_parity`] — bitwise dK/dV parity against the scalar
//!   oracle and against the production two-pass key pass, for every candidate
//!   configuration, both arithmetic modes and every production-qualified
//!   grouping including the MHA control. It also measures key *ownership*:
//!   each block is dispatched alone into a sentinel buffer and the written
//!   element sets are checked to be pairwise disjoint and to cover every
//!   destination key position exactly once.
//! * [`phase_viii_key_block_timing`] — GPU-timestamp timing of the production
//!   key pass against every candidate configuration at the primary 8q/2kv
//!   geometry. Opt-in (`HIERARCHOS_PHASE_VIII_KERNEL_TIMING=1`), because a
//!   timing run must not be confused with a qualification run.
//!
//! Run:
//!
//! ```text
//! cargo test --lib phase_viii_key_block_parity -- --nocapture --test-threads=1
//! HIERARCHOS_PHASE_VIII_KERNEL_TIMING=1 HIERARCHOS_VULKAN_PROFILE_KERNELS=1 \
//!   HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
//!   cargo test --lib phase_viii_key_block_timing -- --nocapture --test-threads=1
//! ```

use super::*;

/// The four workgroup-width builds of the single experimental source.
///
/// `KEY_WORKGROUP_LANES` is the only difference between them; the partitioning
/// itself is a runtime push-constant decision, so one build serves every block
/// count at its width.
const CANDIDATE_L64: &[u8] = include_bytes!(
    "../validation/phase_viii_key_blocks/shaders/transformer_attention_backward_key_blockexp_l64_rdna3.spv"
);
const CANDIDATE_L32: &[u8] = include_bytes!(
    "../validation/phase_viii_key_blocks/shaders/transformer_attention_backward_key_blockexp_l32_rdna3.spv"
);
const CANDIDATE_L16: &[u8] = include_bytes!(
    "../validation/phase_viii_key_blocks/shaders/transformer_attention_backward_key_blockexp_l16_rdna3.spv"
);
const CANDIDATE_L8: &[u8] = include_bytes!(
    "../validation/phase_viii_key_blocks/shaders/transformer_attention_backward_key_blockexp_l8_rdna3.spv"
);

const PRIMARY_BATCH: u32 = 2;
const PRIMARY_SEQ: u32 = 64;
const PRIMARY_HEAD_DIM: u32 = 128;

/// One candidate configuration: `(blocks, lanes)`.
///
/// Every entry is a full-utilization configuration at the primary geometry:
/// `blocks * lanes` divides the 256 destination key rows exactly, so no lane is
/// created that has nothing to own. The block counts are the experiment's
/// 1/2/4/8 series; the widths exist because at this geometry the workgroup
/// *count* is a function of the width, not of the partition.
const CANDIDATE_CONFIGS: [(u32, u32); 10] = [
    (1, 64),
    (2, 64),
    (4, 64),
    (8, 64),
    (1, 32),
    (1, 16),
    (1, 8),
    (2, 32),
    (4, 16),
    (8, 8),
];

fn candidate_spirv(lanes: u32) -> &'static [u8] {
    match lanes {
        64 => CANDIDATE_L64,
        32 => CANDIDATE_L32,
        16 => CANDIDATE_L16,
        8 => CANDIDATE_L8,
        other => panic!("no Phase VIII candidate was built for {other} lanes"),
    }
}

/// Deterministic phase_viii_operands: identical bytes for every arm of a case.
fn phase_viii_operands(salt: u32, count: usize) -> Vec<f32> {
    let mut state = 0x9e37_79b9_u32 ^ salt.wrapping_mul(0x85eb_ca6b);
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / 16_777_216.0;
            unit * 2.0 - 1.0
        })
        .collect()
}

/// Bitwise comparison identical in spirit to the production qualification's
/// `gradient_report`: equality is defined on the FP32 bit pattern, so a
/// one-ULP difference is a failure and `max_abs == 0` is the only pass.
struct PhaseViiiComparison {
    elements: usize,
    bitwise: usize,
    max_abs: f32,
    max_ulp: u32,
    first: Option<(usize, u32, u32)>,
}

fn phase_viii_ulp_order(value: f32) -> i64 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 != 0 {
        -(bits as i64)
    } else {
        bits as i64
    }
}

fn phase_viii_compare(expected: &[f32], actual: &[f32]) -> PhaseViiiComparison {
    assert_eq!(expected.len(), actual.len());
    let mut report = PhaseViiiComparison {
        elements: expected.len(),
        bitwise: 0,
        max_abs: 0.0,
        max_ulp: 0,
        first: None,
    };
    for (index, (&expected_value, &actual_value)) in expected.iter().zip(actual).enumerate() {
        if expected_value.to_bits() == actual_value.to_bits() {
            report.bitwise += 1;
            continue;
        }
        if report.first.is_none() {
            report.first = Some((index, expected_value.to_bits(), actual_value.to_bits()));
        }
        let difference = (expected_value - actual_value).abs();
        if difference.is_finite() {
            report.max_abs = report.max_abs.max(difference);
        }
        let ulp = (phase_viii_ulp_order(expected_value) - phase_viii_ulp_order(actual_value)).unsigned_abs() as u32;
        report.max_ulp = report.max_ulp.max(ulp);
    }
    report
}

fn phase_viii_report_line(
    label: &str,
    tensor: &str,
    comparison: &PhaseViiiComparison,
    unwritten: usize,
    verdict: bool,
) {
    let first = match comparison.first {
        Some((index, expected, actual)) => format!(
            "first_divergence=index {index} expected=0x{expected:08x} actual=0x{actual:08x}"
        ),
        None => "first_divergence=none".to_string(),
    };
    eprintln!(
        "[phase-viii-parity] {label} tensor={tensor} elements={} bitwise={} max_abs={:.9e} \
         max_ulp={} unwritten={} {first} verdict={}",
        comparison.elements,
        comparison.bitwise,
        comparison.max_abs,
        comparison.max_ulp,
        unwritten,
        if verdict { "PASS" } else { "FAIL" }
    );
}

const PHASE_VIII_SENTINEL: f32 = -98_765.432_1_f32;

struct PhaseViiiBuffers {
    query: GpuBuffer,
    grad_output: GpuBuffer,
    gradk_ref: GpuBuffer,
    gradv_ref: GpuBuffer,
    query_mask: GpuBuffer,
    key_mask: GpuBuffer,
    /// k/v are needed by the scalar oracle only; the two-pass pair reads pass-1
    /// scratch instead.
    #[allow(dead_code)]
    key: GpuBuffer,
    #[allow(dead_code)]
    value: GpuBuffer,
    dummy: GpuBuffer,
}

struct PhaseViiiCase {
    label: &'static str,
    heads: u32,
    kv_heads: u32,
    mode: u32,
    push: AttentionPush,
    q_count: usize,
    kv_count: usize,
    scratch: usize,
}

impl PhaseViiiCase {
    fn new(label: &'static str, heads: u32, kv_heads: u32, mode: u32) -> Self {
        let push = AttentionPush {
            batch_size: PRIMARY_BATCH,
            query_seq_len: PRIMARY_SEQ,
            key_value_seq_len: PRIMARY_SEQ,
            query_hidden_size: heads * PRIMARY_HEAD_DIM,
            value_hidden_size: kv_heads * PRIMARY_HEAD_DIM,
            num_heads: heads,
            num_key_value_heads: kv_heads,
            window_size: 0,
            scale: 1.0 / (PRIMARY_HEAD_DIM as f32).sqrt(),
            value_scale: 1.0,
            softcap: 0.0,
            alibi_mode: 0,
            alibi_bias_max: 0.0,
            dropout_p: 0.0,
            rng_step: 0,
            rng_site: 0,
            rng_seed: 0,
            has_sink: 0,
            causal: 1,
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
        let q_count = (PRIMARY_BATCH * PRIMARY_SEQ * heads * PRIMARY_HEAD_DIM) as usize;
        let kv_count = (PRIMARY_BATCH * PRIMARY_SEQ * kv_heads * PRIMARY_HEAD_DIM) as usize;
        let scratch = two_pass_attention_scratch_elements(&push).unwrap();
        Self {
            label,
            heads,
            kv_heads,
            mode,
            push,
            q_count,
            kv_count,
            scratch,
        }
    }

    /// Destination key rows: the row domain the pass owns, in production order
    /// `(batch, KV head, key)`.
    fn key_rows(&self) -> u32 {
        PRIMARY_BATCH * self.kv_heads * PRIMARY_SEQ
    }

    fn query_grid(&self) -> [u32; 3] {
        [
            (PRIMARY_BATCH * self.heads * PRIMARY_SEQ + 63) / 64,
            1,
            1,
        ]
    }

    fn key_grid(&self) -> [u32; 3] {
        [(self.key_rows() + 63) / 64, 1, 1]
    }

    fn block_span(&self, blocks: u32) -> u32 {
        // The candidate's own partition, mirrored here for the grid: block 0
        // spans [0, seq/blocks), every block has this size when blocks divides
        // the sequence, and the grid below is sized from the largest block.
        (PRIMARY_SEQ + blocks - 1) / blocks
    }

    fn block_rows(&self, blocks: u32) -> u32 {
        PRIMARY_BATCH * self.kv_heads * self.block_span(blocks)
    }

    fn block_grid(&self, blocks: u32, lanes: u32) -> [u32; 3] {
        [(self.block_rows(blocks) + lanes - 1) / lanes, 1, 1]
    }

    /// Workgroups actually launched across every block of the pass.
    fn total_workgroups(&self, blocks: u32, lanes: u32) -> u32 {
        self.block_grid(blocks, lanes)[0] * blocks
    }

    /// Lane slots the device must schedule, and the ones that own a key.
    fn lane_slots(&self, blocks: u32, lanes: u32) -> (u32, u32) {
        (
            self.total_workgroups(blocks, lanes) * lanes,
            self.key_rows(),
        )
    }
}

fn build_buffers(device: &VulkanDevice, case: &PhaseViiiCase) -> Result<PhaseViiiBuffers> {
    let mask_count = (PRIMARY_BATCH * PRIMARY_SEQ) as usize;
    Ok(PhaseViiiBuffers {
        query: GpuBuffer::from_f32(device, &phase_viii_operands(11, case.q_count))?,
        grad_output: GpuBuffer::from_f32(device, &phase_viii_operands(14, case.q_count))?,
        gradk_ref: GpuBuffer::from_f32(device, &vec![0.0_f32; case.kv_count])?,
        gradv_ref: GpuBuffer::from_f32(device, &vec![0.0_f32; case.kv_count])?,
        query_mask: GpuBuffer::from_f32(device, &vec![1.0_f32; mask_count])?,
        key_mask: GpuBuffer::from_f32(device, &vec![1.0_f32; mask_count])?,
        key: GpuBuffer::from_f32(device, &phase_viii_operands(12, case.kv_count))?,
        value: GpuBuffer::from_f32(device, &phase_viii_operands(13, case.kv_count))?,
        dummy: GpuBuffer::from_f32(device, &[0.0_f32])?,
    })
}

/// Scalar oracle: the production generic backward, one invocation per
/// `(batch, KV head)`, unsplit. Returns `(dQ, dK, dV)`.
fn run_scalar(
    device: &VulkanDevice,
    kernels: &TransformerKernels,
    case: &PhaseViiiCase,
    buffers: &PhaseViiiBuffers,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let grad_q = GpuBuffer::from_f32(device, &vec![PHASE_VIII_SENTINEL; case.q_count])?;
    let grad_k = GpuBuffer::from_f32(device, &vec![PHASE_VIII_SENTINEL; case.kv_count])?;
    let grad_v = GpuBuffer::from_f32(device, &vec![PHASE_VIII_SENTINEL; case.kv_count])?;
    let mut commands = vulkan::ComputeBatch::new(device)?;
    kernels.attention_backward.record_dispatch(
        &mut commands,
        &[
            &buffers.query,
            &buffers.key,
            &buffers.value,
            &buffers.query_mask,
            &buffers.key_mask,
            &buffers.dummy,
            &buffers.grad_output,
            &grad_q,
            &grad_k,
            &grad_v,
            &buffers.dummy,
            &buffers.dummy,
            &buffers.dummy,
            &buffers.dummy,
            &buffers.dummy,
        ],
        bytemuck::bytes_of(&case.push.as_backward_push(0, 1)),
        [case.kv_heads, PRIMARY_BATCH, 1],
    )?;
    commands.submit()?;
    Ok((
        grad_q.read_f32(case.q_count)?,
        grad_k.read_f32(case.kv_count)?,
        grad_v.read_f32(case.kv_count)?,
    ))
}

/// Pass 1 (query-owned) of the production two-pass pair, into `grad_q`.
fn run_query_pass(
    device: &VulkanDevice,
    kernel: &vulkan::ComputeKernel,
    case: &PhaseViiiCase,
    buffers: &PhaseViiiBuffers,
    grad_q: &GpuBuffer,
    probability: &GpuBuffer,
    ds: &GpuBuffer,
) -> Result<()> {
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
            grad_q,
            probability,
            ds,
        ],
        bytemuck::bytes_of(&case.push),
        case.query_grid(),
    )?;
    commands.submit()
}

/// The production key pass, one dispatch, unchanged artifact.
fn run_production_key(
    device: &VulkanDevice,
    kernel: &vulkan::ComputeKernel,
    case: &PhaseViiiCase,
    buffers: &PhaseViiiBuffers,
    grad_k: &GpuBuffer,
    grad_v: &GpuBuffer,
    probability: &GpuBuffer,
    ds: &GpuBuffer,
) -> Result<()> {
    let mut commands = vulkan::ComputeBatch::new(device)?;
    kernel.record_dispatch(
        &mut commands,
        &[
            &buffers.query,
            &buffers.grad_output,
            probability,
            ds,
            grad_k,
            grad_v,
            &buffers.key_mask,
            &buffers.query_mask,
        ],
        bytemuck::bytes_of(&case.push),
        case.key_grid(),
    )?;
    commands.submit()
}

/// One candidate submission: all `blocks` key blocks, in block order.
///
/// `separate_submissions` makes each block its own submission, which is the
/// form the generic split uses; the default keeps the whole pass in one
/// submission, which is how a promotion would dispatch it (the blocks write
/// disjoint keys, so no ordering exists between them).
fn run_candidate_key(
    device: &VulkanDevice,
    kernel: &vulkan::ComputeKernel,
    case: &PhaseViiiCase,
    buffers: &PhaseViiiBuffers,
    grad_k: &GpuBuffer,
    grad_v: &GpuBuffer,
    probability: &GpuBuffer,
    ds: &GpuBuffer,
    blocks: u32,
    lanes: u32,
    separate_submissions: bool,
) -> Result<()> {
    let mut commands = vulkan::ComputeBatch::new(device)?;
    for block in 0..blocks {
        kernel.record_dispatch(
            &mut commands,
            &[
                &buffers.query,
                &buffers.grad_output,
                probability,
                ds,
                grad_k,
                grad_v,
                &buffers.key_mask,
                &buffers.query_mask,
            ],
            bytemuck::bytes_of(&case.push.as_backward_push(block, blocks)),
            case.block_grid(blocks, lanes),
        )?;
        if separate_submissions && block + 1 < blocks {
            commands.submit_and_restart()?;
        }
    }
    commands.submit()
}

fn build_candidate_kernels(device: &VulkanDevice) -> Result<Vec<vulkan::ComputeKernel>> {
    let mut kernels = Vec::new();
    for lanes in [64u32, 32, 16, 8] {
        kernels.push(vulkan::ComputeKernel::new(
            device,
            candidate_spirv(lanes),
            8,
            std::mem::size_of::<AttentionBackwardPush>() as u32,
        )?);
    }
    Ok(kernels)
}

fn candidate_kernel<'a>(
    kernels: &'a [vulkan::ComputeKernel],
    lanes: u32,
) -> &'a vulkan::ComputeKernel {
    let index = match lanes {
        64 => 0,
        32 => 1,
        16 => 2,
        8 => 3,
        other => panic!("no candidate kernel for {other} lanes"),
    };
    &kernels[index]
}

/// Every destination key position is owned by exactly one block, measured by
/// dispatching each block alone into a sentinel buffer.
///
/// The written set is the set of elements whose value is no longer the
/// sentinel; the union must be every destination element and the pairwise
/// intersection must be empty.
fn measure_key_ownership(
    device: &VulkanDevice,
    kernel: &vulkan::ComputeKernel,
    case: &PhaseViiiCase,
    buffers: &PhaseViiiBuffers,
    probability: &GpuBuffer,
    ds: &GpuBuffer,
    blocks: u32,
    lanes: u32,
) -> Result<(usize, usize, usize)> {
    let mut owned = vec![0u32; case.kv_count * 2];
    for block in 0..blocks {
        let grad_k = GpuBuffer::from_f32(device, &vec![PHASE_VIII_SENTINEL; case.kv_count])?;
        let grad_v = GpuBuffer::from_f32(device, &vec![PHASE_VIII_SENTINEL; case.kv_count])?;
        let mut commands = vulkan::ComputeBatch::new(device)?;
        kernel.record_dispatch(
            &mut commands,
            &[
                &buffers.query,
                &buffers.grad_output,
                probability,
                ds,
                &grad_k,
                &grad_v,
                &buffers.key_mask,
                &buffers.query_mask,
            ],
            bytemuck::bytes_of(&case.push.as_backward_push(block, blocks)),
            case.block_grid(blocks, lanes),
        )?;
        commands.submit()?;
        let written_k = grad_k.read_f32(case.kv_count)?;
        let written_v = grad_v.read_f32(case.kv_count)?;
        for (index, &value) in written_k.iter().enumerate() {
            if value.to_bits() != PHASE_VIII_SENTINEL.to_bits() {
                owned[index] += 1;
            }
        }
        for (index, &value) in written_v.iter().enumerate() {
            if value.to_bits() != PHASE_VIII_SENTINEL.to_bits() {
                owned[case.kv_count + index] += 1;
            }
        }
    }
    let missing = owned.iter().filter(|&&count| count == 0).count();
    let overlapping = owned.iter().filter(|&&count| count > 1).count();
    Ok((owned.len(), missing, overlapping))
}

#[test]
fn phase_viii_key_block_parity() -> Result<()> {
    let Ok(device) = VulkanDevice::new() else {
        eprintln!("phase-viii parity: status=BLOCKED reason=no_vulkan_device");
        return Ok(());
    };
    let kernels = TransformerKernels::new(&device)?;
    let query_kernel = vulkan::ComputeKernel::new(
        &device,
        TwoPassAttentionPass::Query.spirv(),
        TwoPassAttentionPass::Query.binding_count(),
        std::mem::size_of::<AttentionPush>() as u32,
    )?;
    let production_key_kernel = vulkan::ComputeKernel::new(
        &device,
        TwoPassAttentionPass::Key.spirv(),
        TwoPassAttentionPass::Key.binding_count(),
        std::mem::size_of::<AttentionPush>() as u32,
    )?;
    let candidates = build_candidate_kernels(&device)?;

    let geometries: [(&'static str, u32, u32); 5] = [
        ("8q-2kv", 8, 2),
        ("4q-2kv", 4, 2),
        ("8q-4kv", 8, 4),
        ("16q-4kv", 16, 4),
        ("8q-8kv-mha", 8, 8),
    ];

    let mut failures = 0usize;
    let mut cases = 0usize;
    let mut tensors = 0usize;
    let mut elements = 0usize;
    let mut written_passes = 0usize;
    let mut missing_total = 0usize;
    let mut overlapping_total = 0usize;

    for mode in [0u32, 9u32] {
        for (label, heads, kv_heads) in geometries {
            let case = PhaseViiiCase::new(label, heads, kv_heads, mode);
            let buffers = build_buffers(&device, &case)?;
            let (oracle_q, oracle_k, oracle_v) = run_scalar(&device, &kernels, &case, &buffers)?;

            // Production two-pass pair, on the same phase_viii_operands.
            let prod_q = GpuBuffer::from_f32(&device, &vec![PHASE_VIII_SENTINEL; case.q_count])?;
            let prod_k = GpuBuffer::from_f32(&device, &vec![PHASE_VIII_SENTINEL; case.kv_count])?;
            let prod_v = GpuBuffer::from_f32(&device, &vec![PHASE_VIII_SENTINEL; case.kv_count])?;
            let probability =
                GpuBuffer::from_f32(&device, &vec![0.0_f32; case.scratch])?;
            let ds = GpuBuffer::from_f32(&device, &vec![0.0_f32; case.scratch])?;
            run_query_pass(
                &device,
                &query_kernel,
                &case,
                &buffers,
                &prod_q,
                &probability,
                &ds,
            )?;
            run_production_key(
                &device,
                &production_key_kernel,
                &case,
                &buffers,
                &prod_k,
                &prod_v,
                &probability,
                &ds,
            )?;
            let prod_q = prod_q.read_f32(case.q_count)?;
            let prod_k = prod_k.read_f32(case.kv_count)?;
            let prod_v = prod_v.read_f32(case.kv_count)?;

            // The production pair must itself be the oracle's bit pattern; if it
            // is not, the candidate comparison below would be against the wrong
            // baseline and every number after it would be meaningless.
            for (name, oracle, production) in [
                ("dQ", &oracle_q, &prod_q),
                ("dK", &oracle_k, &prod_k),
                ("dV", &oracle_v, &prod_v),
            ] {
                let comparison = phase_viii_compare(oracle, production);
                let ok = comparison.bitwise == comparison.elements;
                if !ok {
                    failures += 1;
                }
                phase_viii_report_line(
                    &format!("geometry={label} mode={mode} arm=production-key-pair baseline=oracle"),
                    name,
                    &comparison,
                    0,
                    ok,
                );
            }

            for (blocks, lanes) in CANDIDATE_CONFIGS {
                let label_full = format!(
                    "geometry={label} mode={mode} lanes={lanes} blocks={blocks} \
                     workgroups={} lanes_slots={}",
                    case.total_workgroups(blocks, lanes),
                    case.lane_slots(blocks, lanes).0
                );
                cases += 1;
                let cand_q = GpuBuffer::from_f32(&device, &vec![PHASE_VIII_SENTINEL; case.q_count])?;
                let cand_k = GpuBuffer::from_f32(&device, &vec![PHASE_VIII_SENTINEL; case.kv_count])?;
                let cand_v = GpuBuffer::from_f32(&device, &vec![PHASE_VIII_SENTINEL; case.kv_count])?;
                // A fresh pair of scratch buffers, so a stale read from the
                // production pass cannot hide a candidate that never wrote one.
                let cand_probability =
                    GpuBuffer::from_f32(&device, &vec![0.0_f32; case.scratch])?;
                let cand_ds = GpuBuffer::from_f32(&device, &vec![0.0_f32; case.scratch])?;
                run_query_pass(
                    &device,
                    &query_kernel,
                    &case,
                    &buffers,
                    &cand_q,
                    &cand_probability,
                    &cand_ds,
                )?;
                run_candidate_key(
                    &device,
                    candidate_kernel(&candidates, lanes),
                    &case,
                    &buffers,
                    &cand_k,
                    &cand_v,
                    &cand_probability,
                    &cand_ds,
                    blocks,
                    lanes,
                    true,
                )?;
                let cand_q = cand_q.read_f32(case.q_count)?;
                let cand_k = cand_k.read_f32(case.kv_count)?;
                let cand_v = cand_v.read_f32(case.kv_count)?;

                let unwritten_k = cand_k
                    .iter()
                    .filter(|value| value.to_bits() == PHASE_VIII_SENTINEL.to_bits())
                    .count();
                let unwritten_v = cand_v
                    .iter()
                    .filter(|value| value.to_bits() == PHASE_VIII_SENTINEL.to_bits())
                    .count();
                let unwritten = unwritten_k + unwritten_v;

                for (name, oracle, candidate, production, unwritten_tensor) in [
                    ("dQ", &oracle_q, &cand_q, &prod_q, 0usize),
                    ("dK", &oracle_k, &cand_k, &prod_k, unwritten_k),
                    ("dV", &oracle_v, &cand_v, &prod_v, unwritten_v),
                ] {
                    let vs_oracle = phase_viii_compare(oracle, candidate);
                    let vs_production = phase_viii_compare(production, candidate);
                    let ok = vs_oracle.bitwise == vs_oracle.elements
                        && vs_production.bitwise == vs_production.elements
                        && unwritten_tensor == 0;
                    if !ok {
                        failures += 1;
                    }
                    tensors += 1;
                    elements += vs_oracle.elements;
                    phase_viii_report_line(
                        &format!("{label_full} baseline=oracle"),
                        name,
                        &vs_oracle,
                        unwritten_tensor,
                        ok,
                    );
                    phase_viii_report_line(
                        &format!("{label_full} baseline=production-key-pass"),
                        name,
                        &vs_production,
                        unwritten_tensor,
                        ok,
                    );
                }
            }

            // Ownership is a property of the lane mapping, not of the
            // arithmetic mode, so it is measured once per geometry at the
            // primary mode with the widest and narrowest candidates.
            if mode == 9 {
                for (blocks, lanes) in CANDIDATE_CONFIGS {
                    let (total, missing, overlapping) = measure_key_ownership(
                        &device,
                        candidate_kernel(&candidates, lanes),
                        &case,
                        &buffers,
                        &probability,
                        &ds,
                        blocks,
                        lanes,
                    )?;
                    written_passes += 1;
                    missing_total += missing;
                    overlapping_total += overlapping;
                    let ok = missing == 0 && overlapping == 0 && total > 0;
                    if !ok {
                        failures += 1;
                    }
                    eprintln!(
                        "[phase-viii-ownership] geometry={label} lanes={lanes} blocks={blocks} \
                         destination_elements={total} missing={missing} overlapping={overlapping} \
                         workgroups={} verdict={}",
                        case.total_workgroups(blocks, lanes),
                        if ok { "PASS" } else { "FAIL" }
                    );
                }
            }
        }
    }

    eprintln!(
        "[phase-viii-parity-summary] status={} cases={cases} tensors={tensors} \
         elements={elements} failures={failures} ownership_passes={written_passes} \
         missing={missing_total} overlapping={overlapping_total}",
        if failures == 0 { "PASS" } else { "FAIL" }
    );
    assert_eq!(failures, 0, "the Phase VIII candidate is not bit-exact");
    Ok(())
}

/// Opt-in GPU-timestamp timing of the key pass at the two qualified grouped
/// geometries, and of its reference peers.
///
/// Each arm is `warmup + iterations` submissions of the same dispatch set; the
/// marker line before each submission is what the shipped profiler's block is
/// attributed to (`analyze_timing.py`). Host-observed submission latency is
/// recorded too, but the GPU numbers are the ones that are reported: an
/// integrated GPU's fence wait is a different quantity from its kernel time.
#[test]
fn phase_viii_key_block_timing() -> Result<()> {
    let requested = std::env::var("HIERARCHOS_PHASE_VIII_KERNEL_TIMING")
        .map(|value| value.trim() == "1")
        .unwrap_or(false);
    if !requested {
        eprintln!(
            "phase-viii kernel timing: status=SKIP reason=not_requested \
             (set HIERARCHOS_PHASE_VIII_KERNEL_TIMING=1)"
        );
        return Ok(());
    }
    let profiled = std::env::var_os("HIERARCHOS_VULKAN_PROFILE_KERNELS").is_some();
    eprintln!(
        "phase-viii kernel timing: status=RUN timestamp_profile={profiled} \
         (set HIERARCHOS_VULKAN_PROFILE_KERNELS=1 and \
         HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 for per-submission GPU time)"
    );
    let device = VulkanDevice::new()?;
    let query_kernel = vulkan::ComputeKernel::new(
        &device,
        TwoPassAttentionPass::Query.spirv(),
        TwoPassAttentionPass::Query.binding_count(),
        std::mem::size_of::<AttentionPush>() as u32,
    )?;
    let production_key_kernel = vulkan::ComputeKernel::new(
        &device,
        TwoPassAttentionPass::Key.spirv(),
        TwoPassAttentionPass::Key.binding_count(),
        std::mem::size_of::<AttentionPush>() as u32,
    )?;
    let candidates = build_candidate_kernels(&device)?;

    let iterations: u32 = std::env::var("HIERARCHOS_PHASE_VIII_KERNEL_ITERATIONS")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(15);
    let warmup: u32 = 3;
    let mode: u32 = std::env::var("HIERARCHOS_PHASE_VIII_KERNEL_MODE")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(9);

    for (label, heads, kv_heads) in [("8q-2kv", 8u32, 2u32), ("4q-2kv", 4u32, 2u32)] {
        let case = PhaseViiiCase::new(label, heads, kv_heads, mode);
        let buffers = build_buffers(&device, &case)?;
        let probability = GpuBuffer::from_f32(&device, &vec![0.0_f32; case.scratch])?;
        let ds = GpuBuffer::from_f32(&device, &vec![0.0_f32; case.scratch])?;
        let grad_q = GpuBuffer::from_f32(&device, &vec![0.0_f32; case.q_count])?;
        let grad_k = GpuBuffer::from_f32(&device, &vec![0.0_f32; case.kv_count])?;
        let grad_v = GpuBuffer::from_f32(&device, &vec![0.0_f32; case.kv_count])?;
        // Pass 1 must run once so the key arms read real probabilities instead
        // of a zeroed scratch; the numbers are irrelevant, the access pattern is
        // not.
        run_query_pass(
            &device,
            &query_kernel,
            &case,
            &buffers,
            &grad_q,
            &probability,
            &ds,
        )?;

        // (arm, blocks, lanes) — `blocks == 0` means the production artifact.
        let mut arms: Vec<(String, u32, u32)> = Vec::new();
        arms.push(("key_production".to_string(), 0, 64));
        for (blocks, lanes) in CANDIDATE_CONFIGS {
            arms.push((format!("key_blocks{blocks}_lanes{lanes}"), blocks, lanes));
        }
        arms.push(("query_reference".to_string(), 0, 0));
        for (blocks, lanes) in CANDIDATE_CONFIGS {
            arms.push((format!("combined_blocks{blocks}_lanes{lanes}"), blocks, lanes));
        }
        arms.push(("combined_production".to_string(), 0, 64));
        // Arm order is a control, not a preference: the production arm runs
        // first by default, so a second process measures it last and the two
        // runs together show whether a difference between two arms is a
        // property of the arm or of its position in the session.
        if std::env::var("HIERARCHOS_PHASE_VIII_REVERSE")
            .map(|value| value.trim() == "1")
            .unwrap_or(false)
        {
            arms.reverse();
        }

        for (arm, blocks, lanes) in arms {
            let key_grid = if lanes == 0 {
                [0, 0, 0]
            } else if blocks == 0 {
                case.key_grid()
            } else {
                case.block_grid(blocks, lanes)
            };
            let workgroups = if lanes == 0 {
                case.query_grid()[0]
            } else if blocks == 0 {
                case.key_grid()[0]
            } else {
                case.total_workgroups(blocks, lanes)
            };
            let key_only = arm.starts_with("key_");
            let query_only = lanes == 0;
            let submit = |marker: &str| -> Result<()> {
                eprintln!(
                    "[phase-viii-bench{marker}] geometry={label} mode={mode} arm={arm} \
                     blocks={blocks} lanes={lanes} workgroups={workgroups}"
                );
                let mut commands = vulkan::ComputeBatch::new(&device)?;
                if query_only || !key_only {
                    query_kernel.record_dispatch(
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
                }
                if query_only {
                    commands.submit()?;
                    return Ok(());
                }
                if blocks == 0 {
                    production_key_kernel.record_dispatch(
                        &mut commands,
                        &[
                            &buffers.query,
                            &buffers.grad_output,
                            &probability,
                            &ds,
                            &grad_k,
                            &grad_v,
                            &buffers.key_mask,
                            &buffers.query_mask,
                        ],
                        bytemuck::bytes_of(&case.push),
                        key_grid,
                    )?;
                } else {
                    for block in 0..blocks {
                        candidate_kernel(&candidates, lanes).record_dispatch(
                            &mut commands,
                            &[
                                &buffers.query,
                                &buffers.grad_output,
                                &probability,
                                &ds,
                                &grad_k,
                                &grad_v,
                                &buffers.key_mask,
                                &buffers.query_mask,
                            ],
                            bytemuck::bytes_of(&case.push.as_backward_push(block, blocks)),
                            key_grid,
                        )?;
                    }
                }
                commands.submit()
            };
            for _ in 0..warmup {
                submit("-warmup")?;
            }
            let mut samples = Vec::with_capacity(iterations as usize);
            for _ in 0..iterations {
                let start = Instant::now();
                submit("-sample")?;
                samples.push(start.elapsed().as_secs_f64() * 1e3);
            }
            let mut sorted = samples.clone();
            sorted.sort_by(|left, right| left.partial_cmp(right).unwrap());
            let median = sorted[sorted.len() / 2];
            let p25 = sorted[sorted.len() / 4];
            eprintln!(
                "[phase-viii-bench-result] geometry={label} mode={mode} arm={arm} \
                 blocks={blocks} lanes={lanes} workgroups={workgroups} \
                 dispatches_per_submission={} iterations={iterations} \
                 median_ms={median:.6} p25_ms={p25:.6} min_ms={:.6} max_ms={:.6} \
                 samples_ms={:?}",
                if query_only {
                    1
                } else if key_only {
                    if blocks == 0 { 1 } else { blocks }
                } else if blocks == 0 {
                    2
                } else {
                    1 + blocks
                },
                sorted[0],
                sorted[sorted.len() - 1],
                sorted
            );
        }
    }
    Ok(())
}
