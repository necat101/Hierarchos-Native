//! Phase XI — RDNA3 GQA key-pass channel tiling (test builds only).
//!
//! This module is the experiment's entire dispatch path. Nothing here is
//! reachable from a production build: the module is declared `#[cfg(test)]`, the
//! candidate SPIR-V is a `#[cfg(test)]` artifact embedded with `include_bytes!`,
//! and the production key shader, the production selectors, the `key_blocks`
//! gate and the default route are untouched. The candidate sources and
//! artifacts live in `validation/phase_xi_channel_tiling_20261125/` and are not
//! in `shaders/`.
//!
//! The candidate gives one lane one destination key row and one contiguous
//! channel chunk of it, instead of the production mapping's whole key row. Every
//! output element is still owned by exactly one lane and still reduced over
//! query heads (ascending) and query positions (ascending) in the production
//! order, so no output element is split across lanes and no cross-lane
//! reduction exists.
//!
//! Two tests:
//!
//! * [`phase_xi_channel_tiling_parity`] — bitwise dK/dV/dQ parity against the
//!   scalar oracle and against the production two-pass key pass, for every
//!   candidate configuration, both arithmetic modes and every
//!   production-qualified grouping including the MHA control. It also measures
//!   ownership three ways: no destination element left at the sentinel, a
//!   host-side bijection over the exact launched lane set, and value equality
//!   (a double-accumulated element cannot reproduce the scalar bits).
//! * [`phase_xi_channel_tiling_timing`] — GPU-timestamp timing of the production
//!   key pass against every candidate configuration at the two grouped
//!   geometries. Opt-in (`HIERARCHOS_PHASE_XI_KERNEL_TIMING=1`), because a
//!   timing run must not be confused with a qualification run.
//!
//! Run:
//!
//! ```text
//! cargo test --lib phase_xi_channel_tiling_parity -- --nocapture --test-threads=1
//! HIERARCHOS_PHASE_XI_KERNEL_TIMING=1 HIERARCHOS_VULKAN_PROFILE_KERNELS=1 \
//!   HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
//!   cargo test --lib phase_xi_channel_tiling_timing -- --nocapture --test-threads=1
//! ```

use super::*;

/// The nine builds of the single experimental source.
///
/// `XI_CHANNEL_CHUNK` is the channel count one lane owns, `XI_UNROLL_CHANNEL`
/// selects the compile-time-expanded channel body (`1`, constant indices, no
/// function-scope accumulator array in the SPIR-V) or the runtime loop (`0`,
/// the shape the production shader compiles to). `c128_u0` is the mapping
/// control: the production lane geometry and the production loop shape, with
/// only the new lane-arithmetic prologue changed.
const CANDIDATE_C128_U0: &[u8] = include_bytes!(
    "../validation/phase_xi_channel_tiling_20261125/shaders/key_chantile_c128_u0.spv"
);
const CANDIDATE_C32_U0: &[u8] = include_bytes!(
    "../validation/phase_xi_channel_tiling_20261125/shaders/key_chantile_c32_u0.spv"
);
const CANDIDATE_C32_U1: &[u8] = include_bytes!(
    "../validation/phase_xi_channel_tiling_20261125/shaders/key_chantile_c32_u1.spv"
);
const CANDIDATE_C16_U0: &[u8] = include_bytes!(
    "../validation/phase_xi_channel_tiling_20261125/shaders/key_chantile_c16_u0.spv"
);
const CANDIDATE_C16_U1: &[u8] = include_bytes!(
    "../validation/phase_xi_channel_tiling_20261125/shaders/key_chantile_c16_u1.spv"
);
const CANDIDATE_C8_U0: &[u8] = include_bytes!(
    "../validation/phase_xi_channel_tiling_20261125/shaders/key_chantile_c8_u0.spv"
);
const CANDIDATE_C8_U1: &[u8] = include_bytes!(
    "../validation/phase_xi_channel_tiling_20261125/shaders/key_chantile_c8_u1.spv"
);
const CANDIDATE_C4_U0: &[u8] = include_bytes!(
    "../validation/phase_xi_channel_tiling_20261125/shaders/key_chantile_c4_u0.spv"
);
const CANDIDATE_C4_U1: &[u8] = include_bytes!(
    "../validation/phase_xi_channel_tiling_20261125/shaders/key_chantile_c4_u1.spv"
);

const PRIMARY_BATCH: u32 = 2;
const PRIMARY_SEQ: u32 = 64;
const PRIMARY_HEAD_DIM: u32 = 128;

/// One candidate configuration.
#[derive(Clone, Copy)]
struct PhaseXiCandidate {
    label: &'static str,
    /// Channels owned by one lane.
    chunk: u32,
    /// Lanes that share one destination key row (`128 / chunk`).
    blocks: u32,
    /// 1 = compile-time-expanded channel body, 0 = runtime loop.
    unroll: u32,
    spirv: &'static [u8],
}

const XI_CANDIDATES: [PhaseXiCandidate; 9] = [
    PhaseXiCandidate { label: "c128_u0", chunk: 128, blocks: 1, unroll: 0, spirv: CANDIDATE_C128_U0 },
    PhaseXiCandidate { label: "c32_u0", chunk: 32, blocks: 4, unroll: 0, spirv: CANDIDATE_C32_U0 },
    PhaseXiCandidate { label: "c32_u1", chunk: 32, blocks: 4, unroll: 1, spirv: CANDIDATE_C32_U1 },
    PhaseXiCandidate { label: "c16_u0", chunk: 16, blocks: 8, unroll: 0, spirv: CANDIDATE_C16_U0 },
    PhaseXiCandidate { label: "c16_u1", chunk: 16, blocks: 8, unroll: 1, spirv: CANDIDATE_C16_U1 },
    PhaseXiCandidate { label: "c8_u0", chunk: 8, blocks: 16, unroll: 0, spirv: CANDIDATE_C8_U0 },
    PhaseXiCandidate { label: "c8_u1", chunk: 8, blocks: 16, unroll: 1, spirv: CANDIDATE_C8_U1 },
    PhaseXiCandidate { label: "c4_u0", chunk: 4, blocks: 32, unroll: 0, spirv: CANDIDATE_C4_U0 },
    PhaseXiCandidate { label: "c4_u1", chunk: 4, blocks: 32, unroll: 1, spirv: CANDIDATE_C4_U1 },
];

/// Deterministic operands: identical bytes for every arm of a case.
fn phase_xi_operands(salt: u32, count: usize) -> Vec<f32> {
    let mut state = 0x9e37_79b9_u32 ^ salt.wrapping_mul(0x85eb_ca6b);
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / 16_777_216.0;
            unit * 2.0 - 1.0
        })
        .collect()
}

/// Bitwise comparison: equality is defined on the FP32 bit pattern, so a
/// one-ULP difference is a failure and `max_abs == 0` is the only pass.
struct PhaseXiComparison {
    elements: usize,
    bitwise: usize,
    max_abs: f32,
    max_ulp: u32,
    first: Option<(usize, u32, u32)>,
}

fn phase_xi_ulp_order(value: f32) -> i64 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 != 0 {
        -(bits as i64)
    } else {
        bits as i64
    }
}

fn phase_xi_compare(expected: &[f32], actual: &[f32]) -> PhaseXiComparison {
    assert_eq!(expected.len(), actual.len());
    let mut report = PhaseXiComparison {
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
        let ulp = (phase_xi_ulp_order(expected_value) - phase_xi_ulp_order(actual_value))
            .unsigned_abs() as u32;
        report.max_ulp = report.max_ulp.max(ulp);
    }
    report
}

fn phase_xi_report_line(
    label: &str,
    tensor: &str,
    comparison: &PhaseXiComparison,
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
        "[phase-xi-parity] {label} tensor={tensor} elements={} bitwise={} max_abs={:.9e} \
         max_ulp={} unwritten={} {first} verdict={}",
        comparison.elements,
        comparison.bitwise,
        comparison.max_abs,
        comparison.max_ulp,
        unwritten,
        if verdict { "PASS" } else { "FAIL" }
    );
}

const PHASE_XI_SENTINEL: f32 = -98_765.432_1_f32;

struct PhaseXiBuffers {
    query: GpuBuffer,
    grad_output: GpuBuffer,
    query_mask: GpuBuffer,
    key_mask: GpuBuffer,
    key: GpuBuffer,
    value: GpuBuffer,
    dummy: GpuBuffer,
}

struct PhaseXiCase {
    label: &'static str,
    heads: u32,
    kv_heads: u32,
    mode: u32,
    push: AttentionPush,
    q_count: usize,
    kv_count: usize,
    scratch: usize,
}

impl PhaseXiCase {
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
            div_ceil_u32((PRIMARY_BATCH * self.heads * PRIMARY_SEQ) as usize, 64),
            1,
            1,
        ]
    }

    fn production_key_grid(&self) -> [u32; 3] {
        [div_ceil_u32(self.key_rows() as usize, 64), 1, 1]
    }

    /// One candidate submission: every `(row, chunk)` lane slot, 64 per
    /// workgroup, so the grid is derived from the lane count the mapping needs.
    fn candidate_key_grid(&self, candidate: &PhaseXiCandidate) -> [u32; 3] {
        [
            div_ceil_u32((self.key_rows() * candidate.blocks) as usize, 64),
            1,
            1,
        ]
    }

    fn candidate_workgroups(&self, candidate: &PhaseXiCandidate) -> u32 {
        self.candidate_key_grid(candidate)[0]
    }

    fn candidate_invocations(&self, candidate: &PhaseXiCandidate) -> u32 {
        self.candidate_workgroups(candidate) * 64
    }
}

fn build_buffers(device: &VulkanDevice, case: &PhaseXiCase) -> Result<PhaseXiBuffers> {
    let mask_count = (PRIMARY_BATCH * PRIMARY_SEQ) as usize;
    Ok(PhaseXiBuffers {
        query: GpuBuffer::from_f32(device, &phase_xi_operands(11, case.q_count))?,
        grad_output: GpuBuffer::from_f32(device, &phase_xi_operands(14, case.q_count))?,
        query_mask: GpuBuffer::from_f32(device, &vec![1.0_f32; mask_count])?,
        key_mask: GpuBuffer::from_f32(device, &vec![1.0_f32; mask_count])?,
        key: GpuBuffer::from_f32(device, &phase_xi_operands(12, case.kv_count))?,
        value: GpuBuffer::from_f32(device, &phase_xi_operands(13, case.kv_count))?,
        dummy: GpuBuffer::from_f32(device, &[0.0_f32])?,
    })
}

/// Scalar oracle: the production generic backward, one invocation per
/// `(batch, KV head)`, unsplit. Returns `(dQ, dK, dV)`.
fn run_scalar(
    device: &VulkanDevice,
    kernels: &TransformerKernels,
    case: &PhaseXiCase,
    buffers: &PhaseXiBuffers,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let grad_q = GpuBuffer::from_f32(device, &vec![PHASE_XI_SENTINEL; case.q_count])?;
    let grad_k = GpuBuffer::from_f32(device, &vec![PHASE_XI_SENTINEL; case.kv_count])?;
    let grad_v = GpuBuffer::from_f32(device, &vec![PHASE_XI_SENTINEL; case.kv_count])?;
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
    case: &PhaseXiCase,
    buffers: &PhaseXiBuffers,
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
    case: &PhaseXiCase,
    buffers: &PhaseXiBuffers,
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
        case.production_key_grid(),
    )?;
    commands.submit()
}

/// One candidate submission: the whole channel-tiled mapping, one dispatch.
fn run_candidate_key(
    device: &VulkanDevice,
    kernel: &vulkan::ComputeKernel,
    case: &PhaseXiCase,
    buffers: &PhaseXiBuffers,
    grad_k: &GpuBuffer,
    grad_v: &GpuBuffer,
    probability: &GpuBuffer,
    ds: &GpuBuffer,
    candidate: &PhaseXiCandidate,
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
        case.candidate_key_grid(candidate),
    )?;
    commands.submit()
}

fn build_candidate_kernels(device: &VulkanDevice) -> Result<Vec<vulkan::ComputeKernel>> {
    XI_CANDIDATES
        .iter()
        .map(|candidate| {
            vulkan::ComputeKernel::new(
                device,
                candidate.spirv,
                8,
                std::mem::size_of::<AttentionPush>() as u32,
            )
        })
        .collect()
}

/// Host-side bijection over the exact launched lane set.
///
/// The candidate's own mapping is `real index = chunk + row * blocks` over
/// `[0, rows * blocks)`, and a lane writes channel `c0 + i` when that index is
/// inside the output dimension. Replaying it here proves the three properties a
/// sentinel pass cannot see: every `(row, channel)` pair is produced by exactly
/// one lane slot, no channel of a row is skipped, and no channel is produced
/// twice. Returns `(covered, missing, duplicated, lane_slots, useful_lanes)`.
fn phase_xi_bijection(
    rows: u32,
    chunk: u32,
    blocks: u32,
    qk_dim: u32,
    value_dim: u32,
) -> (usize, usize, usize, u32, u32) {
    let lane_slots = rows * blocks;
    let mut owners = vec![0u8; (rows as usize) * (qk_dim.max(value_dim) as usize) * 2];
    let mut useful = 0u32;
    for slot in 0..lane_slots {
        let chunk_index = slot % blocks;
        let row = slot / blocks;
        let c0 = chunk_index * chunk;
        if c0 >= qk_dim && c0 >= value_dim {
            continue;
        }
        useful += 1;
        for i in 0..chunk {
            let d = c0 + i;
            if d < qk_dim {
                let index = (row as usize) * (qk_dim as usize) + d as usize;
                owners[index] = owners[index].saturating_add(1);
            }
            if d < value_dim {
                let offset = (rows as usize) * (qk_dim as usize);
                let index = offset + (row as usize) * (value_dim as usize) + d as usize;
                owners[index] = owners[index].saturating_add(1);
            }
        }
    }
    let k_len = (rows as usize) * (qk_dim as usize);
    let v_len = (rows as usize) * (value_dim as usize);
    let mut covered = 0usize;
    let mut missing = 0usize;
    let mut duplicated = 0usize;
    for &count in &owners[..k_len] {
        if count == 0 {
            missing += 1;
        } else {
            covered += 1;
        }
        if count > 1 {
            duplicated += 1;
        }
    }
    for &count in &owners[k_len..k_len + v_len] {
        if count == 0 {
            missing += 1;
        } else {
            covered += 1;
        }
        if count > 1 {
            duplicated += 1;
        }
    }
    (covered, missing, duplicated, lane_slots, useful)
}

#[test]
fn phase_xi_channel_tiling_ownership() -> Result<()> {
    // Pure mapping arithmetic, no device: the lane -> (row, chunk) replay that
    // backs the device-side sentinel check below.
    let mut failures = 0usize;
    for (label, heads, kv_heads) in [("8q-2kv", 8u32, 2u32), ("8q-8kv-mha", 8u32, 8u32)] {
        let case = PhaseXiCase::new(label, heads, kv_heads, 0);
        let rows = case.key_rows();
        for candidate in XI_CANDIDATES {
            let (covered, missing, duplicated, lane_slots, useful) = phase_xi_bijection(
                rows,
                candidate.chunk,
                candidate.blocks,
                PRIMARY_HEAD_DIM,
                PRIMARY_HEAD_DIM,
            );
            let expected = (rows as usize) * (PRIMARY_HEAD_DIM as usize) * 2;
            let ok = covered == expected && missing == 0 && duplicated == 0;
            if !ok {
                failures += 1;
            }
            eprintln!(
                "[phase-xi-ownership] geometry={label} variant={} chunk={} blocks={} \
                 lane_slots={lane_slots} useful_lanes={useful} output_elements={expected} \
                 covered={covered} missing={missing} duplicated={duplicated} \
                 workgroups={} verdict={}",
                candidate.label,
                candidate.chunk,
                candidate.blocks,
                div_ceil_u32(lane_slots as usize, 64),
                if ok { "PASS" } else { "FAIL" }
            );
        }
    }
    assert_eq!(failures, 0, "a candidate does not cover the output exactly once");
    Ok(())
}

/// The complete Phase VII/VIII/IX/X training step, with the key pass swapped.
///
/// `HIERARCHOS_PHASE_XI_STEP_ARM`   `production` or a candidate label (required)
/// `HIERARCHOS_PHASE_XI_FIXTURE`    fixture dir (default `target/gqa-e2e/fixture-8q2kv`)
/// `HIERARCHOS_PHASE_XI_STEP_STEPS` steps per process (default 29)
/// `HIERARCHOS_PHASE_XI_STEP_DIGEST=1`  print loss bits and gradient digests
/// `HIERARCHOS_PHASE_XI_STEP_CHECKPOINT=<dir>`  write adapter + optimizer state
///
/// The swap is a test-module operation, not a production one: the two-pass pair
/// is cached in a `OnceCell` with the kernel set, and this test pre-sets that
/// cell before the first step, so `resolve_two_pass_attention_backward` finds it
/// already resolved and the dispatch path uses the pair it is given. The
/// production selector, the production key shader and the `key_blocks` gate are
/// untouched, and an unset environment variable runs the production arm.
/// Channel blocks per destination key row for the arm this process is running.
///
/// One means the production mapping, so a test that never installs an override
/// submits exactly the production grid. The step driver is single-threaded, so a
/// thread-local is the narrowest place to carry this.
thread_local! {
    static STEP_KEY_BLOCKS: std::cell::Cell<u32> = const { std::cell::Cell::new(1) };
}

/// Grid the two-pass key dispatch uses. Only a test build reaches this.
pub(super) fn step_key_grid(key_rows: u32) -> [u32; 3] {
    let blocks = STEP_KEY_BLOCKS.with(|value| value.get()).max(1);
    [
        div_ceil_u32((key_rows * blocks) as usize, 64),
        1,
        1,
    ]
}

struct PhaseXiSequence {
    input_ids: Vec<u32>,
    targets: Vec<u32>,
    attention_mask: Vec<f32>,
    loss_weights: Vec<f32>,
}

fn phase_xi_load_sequences(
    path: &std::path::Path,
    seq_len: usize,
) -> Result<Vec<PhaseXiSequence>> {
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
            let mut sequence = PhaseXiSequence {
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

fn phase_xi_batch_arrays(
    sequences: &[PhaseXiSequence],
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

fn phase_xi_digest(values: &std::collections::BTreeMap<String, Vec<f32>>) -> (String, usize, f64) {
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

#[test]
fn phase_xi_channel_tiling_full_step() -> Result<()> {
    let arm = match std::env::var("HIERARCHOS_PHASE_XI_STEP_ARM") {
        Ok(value) => value.trim().to_string(),
        Err(_) => {
            eprintln!(
                "phase-xi full step: status=SKIP reason=not_requested \
                 (set HIERARCHOS_PHASE_XI_STEP_ARM=production|<candidate label>)"
            );
            return Ok(());
        }
    };
    let candidate = if arm == "production" {
        None
    } else {
        Some(
            *XI_CANDIDATES
                .iter()
                .find(|entry| entry.label == arm)
                .with_context(|| format!("unknown phase-xi step arm {arm}"))?,
        )
    };
    let fixture = std::env::var("HIERARCHOS_PHASE_XI_FIXTURE")
        .unwrap_or_else(|_| "target/gqa-e2e/fixture-8q2kv".to_string());
    let fixture = std::path::PathBuf::from(fixture);
    let steps: usize = std::env::var("HIERARCHOS_PHASE_XI_STEP_STEPS")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(29);
    let warmup: usize = 3;
    let digest = std::env::var("HIERARCHOS_PHASE_XI_STEP_DIGEST")
        .map(|value| value.trim() == "1")
        .unwrap_or(false);
    let (batch, seq_len) = (2usize, 64usize);
    let seed = 7u64;
    let hyper = AdamWHyperParams {
        lr: 1.0e-2,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1.0e-8,
        weight_decay: 0.0,
    };

    let sequences = phase_xi_load_sequences(&fixture.join("dataset.jsonl"), seq_len)?;
    let batches_per_epoch = sequences.len() / batch;
    anyhow::ensure!(batches_per_epoch > 0, "fixture dataset is smaller than one batch");

    let device = VulkanDevice::new()?;
    let mut graph = VulkanTransformer::from_hf_package(
        device.clone(),
        fixture.join("model"),
        batch,
        seq_len,
    )?;
    let target = graph
        .config()
        .architecture
        .default_lora_target_module()
        .to_owned();
    graph.enable_lora(
        VulkanTransformerLoraConfig::causal_lm(8, 16, vec![target], false)?,
        seed,
    )?;
    graph.set_dropout_seed(seed as u32 ^ (seed >> 32) as u32);
    let start_parameters = phase_xi_digest(&graph.parameter_values()?);

    let production_key_signature = match candidate {
        None => None,
        Some(_) => {
            let query = vulkan::ComputeKernel::new(
                &device,
                TwoPassAttentionPass::Query.spirv(),
                TwoPassAttentionPass::Query.binding_count(),
                std::mem::size_of::<AttentionPush>() as u32,
            )?;
            let key = vulkan::ComputeKernel::new(
                &device,
                candidate.expect("candidate arm").spirv,
                8,
                std::mem::size_of::<AttentionPush>() as u32,
            )?;
            let probe = vulkan::ComputeKernel::new(
                &device,
                TwoPassAttentionPass::Key.spirv(),
                TwoPassAttentionPass::Key.binding_count(),
                std::mem::size_of::<AttentionPush>() as u32,
            )?;
            let production_signature = probe.shader_signature();
            let candidate_signature = key.shader_signature();
            anyhow::ensure!(
                candidate_signature != production_signature,
                "the step arm's key kernel is the production artifact"
            );
            graph
                .kernels
                .rdna3_two_pass_attention_backward
                .set(TwoPassAttentionBackward {
                    kernels: Some([query, key]),
                    reason: "phase_xi_experimental_override",
                    detail: None,
                })
                .map_err(|_| anyhow!("the two-pass pair was already resolved"))?;
            // The candidate owns one channel chunk per lane, so the dispatch
            // needs `blocks` times the lane slots for the same key rows.
            STEP_KEY_BLOCKS.with(|value| value.set(candidate.expect("candidate arm").blocks));
            eprintln!(
                "[phase-xi-step-arm] arm={arm} candidate_key_signature={candidate_signature:016x} \
                 production_key_signature={production_signature:016x} blocks={} override=installed",
                candidate.expect("candidate arm").blocks
            );
            Some(production_signature)
        }
    };
    let _ = production_key_signature;

    let installed = graph.kernels.rdna3_row_packed_attention.is_some();
    let forward_before = attention_forward_dispatch_counts();
    let backward_before = attention_backward_dispatch_counts();
    eprintln!(
        "[phase-xi-step-run] arm={arm} fixture={} sequences={} batches_per_epoch={batches_per_epoch} \
         batch={batch} seq={seq_len} steps={steps} warmup={warmup} lora_rank=8 lora_alpha=16 \
         lr={} seed={seed} digest={digest} device={} row_packed_installed={installed} \
         parameter_start_digest={} parameter_elements={}",
        fixture.display(),
        sequences.len(),
        hyper.lr,
        graph.device_name(),
        start_parameters.0,
        start_parameters.1
    );
    anyhow::ensure!(installed, "the production row-packed forward is not installed");

    for index in 0..steps {
        let global_step = index;
        let epoch = global_step / batches_per_epoch;
        let batch_index = global_step % batches_per_epoch;
        let begin = batch_index * batch;
        let (input_ids, targets, attention_mask, loss_weights) =
            phase_xi_batch_arrays(&sequences[begin..begin + batch], seq_len)?;
        let started = Instant::now();
        let result = graph.train_step(&input_ids, &targets, &attention_mask, &loss_weights, hyper)?;
        let step_ms = started.elapsed().as_secs_f64() * 1e3;
        let measured = index >= warmup;
        if digest {
            let gradients = graph.lora_gradient_values()?;
            let gradient_digest = phase_xi_digest(&gradients);
            let parameters = phase_xi_digest(&graph.parameter_values()?);
            eprintln!(
                "[phase-xi-step] arm={arm} step={index} epoch={epoch} batch={batch_index} \
                 measured={measured} step_ms={step_ms:.6} loss_bits={:08x} loss={:e} \
                 grad_digest={} grad_elements={} grad_norm={:.17e} param_digest={} \
                 param_elements={}",
                result.loss.to_bits(),
                result.loss,
                gradient_digest.0,
                gradient_digest.1,
                gradient_digest.2,
                parameters.0,
                parameters.1
            );
        } else {
            eprintln!(
                "[phase-xi-step] arm={arm} step={index} epoch={epoch} batch={batch_index} \
                 measured={measured} step_ms={step_ms:.6} loss_bits={:08x} loss={:e}",
                result.loss.to_bits(),
                result.loss
            );
        }
    }
    let forward_after = attention_forward_dispatch_counts();
    let backward_after = attention_backward_dispatch_counts();
    let end_parameters = phase_xi_digest(&graph.parameter_values()?);
    eprintln!(
        "[phase-xi-step-run-end] arm={arm} parameter_end_digest={} parameter_elements={} \
         forward_generic={} forward_row_packed={} forward_short={} forward_total={} \
         backward_scalar={} backward_query={} backward_key={}",
        end_parameters.0,
        end_parameters.1,
        forward_after.generic - forward_before.generic,
        forward_after.row_packed - forward_before.row_packed,
        forward_after.short - forward_before.short,
        (forward_after.total() - forward_before.total()),
        backward_after.scalar - backward_before.scalar,
        backward_after.query - backward_before.query,
        backward_after.key - backward_before.key
    );

    if let Ok(directory) = std::env::var("HIERARCHOS_PHASE_XI_STEP_CHECKPOINT") {
        let directory = std::path::PathBuf::from(directory);
        anyhow::ensure!(
            !directory.exists(),
            "checkpoint destination already exists: {}",
            directory.display()
        );
        graph.save_lora_training_checkpoint(&directory, hyper)?;
        let mut entries = std::fs::read_dir(&directory)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<Result<Vec<_>, _>>()?;
        entries.sort();
        for path in entries {
            if !path.is_file() {
                continue;
            }
            use sha2::{Digest, Sha256};
            let name = path
                .file_name()
                .map(|value| value.to_string_lossy().into_owned())
                .unwrap_or_default();
            let digest = format!("{:x}", Sha256::digest(std::fs::read(&path)?));
            eprintln!(
                "[phase-xi-checkpoint] arm={arm} file={name} bytes={} sha256={digest}",
                std::fs::metadata(&path)?.len()
            );
        }
    }
    Ok(())
}

#[test]
fn phase_xi_channel_tiling_parity() -> Result<()> {
    let Ok(device) = VulkanDevice::new() else {
        eprintln!("phase-xi parity: status=BLOCKED reason=no_vulkan_device");
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
    let mut unwritten_total = 0usize;

    for mode in [0u32, 9u32] {
        for (label, heads, kv_heads) in geometries {
            let case = PhaseXiCase::new(label, heads, kv_heads, mode);
            let buffers = build_buffers(&device, &case)?;
            let (oracle_q, oracle_k, oracle_v) = run_scalar(&device, &kernels, &case, &buffers)?;

            // The scratch is filled with the sentinel *before* pass 1, so a
            // candidate that reads a slot pass 1 never wrote for this causal
            // geometry reads the sentinel and cannot reproduce the scalar bits.
            let probability = GpuBuffer::from_f32(&device, &vec![PHASE_XI_SENTINEL; case.scratch])?;
            let ds = GpuBuffer::from_f32(&device, &vec![PHASE_XI_SENTINEL; case.scratch])?;
            let prod_q = GpuBuffer::from_f32(&device, &vec![PHASE_XI_SENTINEL; case.q_count])?;
            run_query_pass(
                &device,
                &query_kernel,
                &case,
                &buffers,
                &prod_q,
                &probability,
                &ds,
            )?;
            let prod_q = prod_q.read_f32(case.q_count)?;
            let prod_k = GpuBuffer::from_f32(&device, &vec![PHASE_XI_SENTINEL; case.kv_count])?;
            let prod_v = GpuBuffer::from_f32(&device, &vec![PHASE_XI_SENTINEL; case.kv_count])?;
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
            let prod_k = prod_k.read_f32(case.kv_count)?;
            let prod_v = prod_v.read_f32(case.kv_count)?;

            // The production pair must itself be the oracle's bit pattern; if it
            // is not, every candidate comparison below is against the wrong
            // baseline.
            for (name, oracle, production) in [
                ("dQ", &oracle_q, &prod_q),
                ("dK", &oracle_k, &prod_k),
                ("dV", &oracle_v, &prod_v),
            ] {
                let comparison = phase_xi_compare(oracle, production);
                let ok = comparison.bitwise == comparison.elements;
                if !ok {
                    failures += 1;
                }
                phase_xi_report_line(
                    &format!("geometry={label} mode={mode} arm=production-key-pair baseline=oracle"),
                    name,
                    &comparison,
                    0,
                    ok,
                );
            }

            for (index, candidate) in XI_CANDIDATES.iter().enumerate() {
                let label_full = format!(
                    "geometry={label} mode={mode} variant={} chunk={} blocks={} unroll={} \
                     workgroups={} invocations={} channels_per_lane={}",
                    candidate.label,
                    candidate.chunk,
                    candidate.blocks,
                    candidate.unroll,
                    case.candidate_workgroups(candidate),
                    case.candidate_invocations(candidate),
                    candidate.chunk,
                );
                cases += 1;
                let cand_k = GpuBuffer::from_f32(&device, &vec![PHASE_XI_SENTINEL; case.kv_count])?;
                let cand_v = GpuBuffer::from_f32(&device, &vec![PHASE_XI_SENTINEL; case.kv_count])?;
                run_candidate_key(
                    &device,
                    &candidates[index],
                    &case,
                    &buffers,
                    &cand_k,
                    &cand_v,
                    &probability,
                    &ds,
                    candidate,
                )?;
                let cand_k = cand_k.read_f32(case.kv_count)?;
                let cand_v = cand_v.read_f32(case.kv_count)?;

                let unwritten_k = cand_k
                    .iter()
                    .filter(|value| value.to_bits() == PHASE_XI_SENTINEL.to_bits())
                    .count();
                let unwritten_v = cand_v
                    .iter()
                    .filter(|value| value.to_bits() == PHASE_XI_SENTINEL.to_bits())
                    .count();
                let unwritten = unwritten_k + unwritten_v;
                unwritten_total += unwritten;

                for (name, oracle, candidate_values, production, unwritten_tensor) in [
                    ("dK", &oracle_k, &cand_k, &prod_k, unwritten_k),
                    ("dV", &oracle_v, &cand_v, &prod_v, unwritten_v),
                ] {
                    let vs_oracle = phase_xi_compare(oracle, candidate_values);
                    let vs_production = phase_xi_compare(production, candidate_values);
                    let ok = vs_oracle.bitwise == vs_oracle.elements
                        && vs_production.bitwise == vs_production.elements
                        && unwritten_tensor == 0;
                    if !ok {
                        failures += 1;
                    }
                    tensors += 1;
                    elements += vs_oracle.elements;
                    phase_xi_report_line(
                        &format!("{label_full} baseline=oracle"),
                        name,
                        &vs_oracle,
                        unwritten_tensor,
                        ok,
                    );
                    phase_xi_report_line(
                        &format!("{label_full} baseline=production-key-pass"),
                        name,
                        &vs_production,
                        unwritten_tensor,
                        ok,
                    );
                }
            }
        }
    }

    eprintln!(
        "[phase-xi-parity-summary] status={} cases={cases} tensors={tensors} \
         elements={elements} unwritten={unwritten_total} failures={failures}",
        if failures == 0 { "PASS" } else { "FAIL" }
    );
    assert_eq!(failures, 0, "the Phase XI candidate is not bit-exact");
    Ok(())
}

/// Opt-in GPU-timestamp timing of the key pass at the two qualified grouped
/// geometries.
///
/// Each arm is `warmup + iterations` submissions of one key-only dispatch set.
/// The marker line before each submission is what the shipped profiler's block
/// is attributed to (`analyze_timing.py`). Host-observed submission latency is
/// recorded too, but the GPU numbers are the ones that are reported: an
/// integrated GPU's fence wait is a different quantity from its kernel time.
///
/// The candidate module is not in the binary's shader-name registry, so the
/// profiler labels it `unknown`. Key-only arms keep that unambiguous: the only
/// dispatch in the submission is the candidate's.
#[test]
fn phase_xi_channel_tiling_timing() -> Result<()> {
    let requested = std::env::var("HIERARCHOS_PHASE_XI_KERNEL_TIMING")
        .map(|value| value.trim() == "1")
        .unwrap_or(false);
    if !requested {
        eprintln!(
            "phase-xi kernel timing: status=SKIP reason=not_requested \
             (set HIERARCHOS_PHASE_XI_KERNEL_TIMING=1)"
        );
        return Ok(());
    }
    let profiled = std::env::var_os("HIERARCHOS_VULKAN_PROFILE_KERNELS").is_some();
    eprintln!(
        "phase-xi kernel timing: status=RUN timestamp_profile={profiled} \
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

    let iterations: u32 = std::env::var("HIERARCHOS_PHASE_XI_KERNEL_ITERATIONS")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(15);
    let warmup: u32 = 3;
    let mode: u32 = std::env::var("HIERARCHOS_PHASE_XI_KERNEL_MODE")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(9);

    for (label, heads, kv_heads) in [("8q-2kv", 8u32, 2u32), ("4q-2kv", 4u32, 2u32)] {
        let case = PhaseXiCase::new(label, heads, kv_heads, mode);
        let buffers = build_buffers(&device, &case)?;
        let probability = GpuBuffer::from_f32(&device, &vec![0.0_f32; case.scratch])?;
        let ds = GpuBuffer::from_f32(&device, &vec![0.0_f32; case.scratch])?;
        let grad_q = GpuBuffer::from_f32(&device, &vec![0.0_f32; case.q_count])?;
        let grad_k = GpuBuffer::from_f32(&device, &vec![0.0_f32; case.kv_count])?;
        let grad_v = GpuBuffer::from_f32(&device, &vec![0.0_f32; case.kv_count])?;
        // Pass 1 runs once so the key arms read real probabilities instead of a
        // zeroed scratch; the numbers are irrelevant, the access pattern is not.
        run_query_pass(
            &device,
            &query_kernel,
            &case,
            &buffers,
            &grad_q,
            &probability,
            &ds,
        )?;

        // (arm, candidate index; `None` is the production artifact)
        let mut arms: Vec<(String, Option<usize>)> =
            vec![("key_production".to_string(), None)];
        // `HIERARCHOS_PHASE_XI_ARMS` restricts the set (for example
        // `production,c8_u0,c4_u0`), so a head-to-head between finalists is not
        // diluted by seven arms that are already measured. The production arm is
        // always present, which is what the paired ratio needs.
        let only: Option<Vec<String>> = std::env::var("HIERARCHOS_PHASE_XI_ARMS")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .map(|entry| entry.trim().to_string())
                    .filter(|entry| !entry.is_empty())
                    .collect()
            });
        let wanted = |label: &str| -> bool {
            only.as_ref()
                .map(|set| set.iter().any(|entry| entry == label))
                .unwrap_or(true)
        };
        for (index, candidate) in XI_CANDIDATES.iter().enumerate() {
            if wanted(candidate.label) {
                arms.push((format!("key_{}", candidate.label), Some(index)));
            }
        }
        if wanted("query_reference") {
            arms.push(("query_reference".to_string(), Some(usize::MAX)));
        }
        // Arm order is a control, not a preference: the production arm runs
        // first by default, so a second process measures it last and the two
        // runs together show whether a difference between two arms is a
        // property of the arm or of its position in the session.
        if std::env::var("HIERARCHOS_PHASE_XI_REVERSE")
            .map(|value| value.trim() == "1")
            .unwrap_or(false)
        {
            arms.reverse();
        }

        for (arm, selection) in arms.iter() {
            let query_only = *selection == Some(usize::MAX);
            let candidate = selection.and_then(|index| XI_CANDIDATES.get(index));
            let workgroups = match (query_only, candidate) {
                (true, _) => case.query_grid()[0],
                (false, None) => case.production_key_grid()[0],
                (false, Some(candidate)) => case.candidate_workgroups(candidate),
            };
            let submit = |marker: &str| -> Result<()> {
                eprintln!(
                    "[phase-xi-bench{marker}] geometry={label} mode={mode} arm={arm} \
                     chunk={} blocks={} unroll={} workgroups={workgroups}",
                    candidate.map(|entry| entry.chunk).unwrap_or(128),
                    candidate.map(|entry| entry.blocks).unwrap_or(1),
                    candidate.map(|entry| entry.unroll).unwrap_or(0),
                );
                let mut commands = vulkan::ComputeBatch::new(&device)?;
                if query_only {
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
                    return commands.submit();
                }
                match candidate {
                    None => production_key_kernel.record_dispatch(
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
                        case.production_key_grid(),
                    )?,
                    Some(candidate) => {
                        let index = XI_CANDIDATES
                            .iter()
                            .position(|entry| entry.label == candidate.label)
                            .expect("candidate index");
                        candidates[index].record_dispatch(
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
                            case.candidate_key_grid(candidate),
                        )?
                    }
                }
                commands.submit()
            };
            // Arm-major sampling was measured to be order-dominated: on this
            // integrated device the first arms of a session run at lower clocks,
            // so an arm's number depends on when it ran rather than on what it
            // does. The protocol is therefore round-major: every round submits
            // every arm once, in the listed order, and only the per-arm series
            // across rounds is reduced. `HIERARCHOS_PHASE_XI_REVERSE` reverses
            // the within-round order, so a position effect shows up as a
            // difference between the two sessions instead of as a ranking.
            let mut samples = Vec::with_capacity(iterations as usize);
            let mut warmup_samples = Vec::new();
            for round in 0..(warmup + iterations) {
                let marker = if round < warmup { "-warmup" } else { "-sample" };
                let start = Instant::now();
                submit(marker)?;
                let elapsed = start.elapsed().as_secs_f64() * 1e3;
                if round < warmup {
                    warmup_samples.push(elapsed);
                } else {
                    samples.push(elapsed);
                }
            }
            let _ = warmup_samples;
            let mut sorted = samples.clone();
            sorted.sort_by(|left, right| left.partial_cmp(right).unwrap());
            let median = sorted[sorted.len() / 2];
            eprintln!(
                "[phase-xi-bench-result] geometry={label} mode={mode} arm={arm} \
                 workgroups={workgroups} iterations={iterations} median_ms={median:.6} \
                 min_ms={:.6} max_ms={:.6} samples_ms={:?}",
                sorted[0],
                sorted[sorted.len() - 1],
                sorted
            );
        }
    }
    Ok(())
}
