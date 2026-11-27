//! Phase XII — production promotion of the four-channel key pass (test builds
//! only).
//!
//! Phase XI qualified one experimental artifact, `c4_u0`, and left it in
//! `validation/phase_xi_channel_tiling_20261125/`. This phase promotes that
//! exact artifact into `shaders/` and routes the production two-pass backward
//! through it for the one geometry it was qualified on.
//!
//! Every numerical comparison here is driven through
//! [`record_two_pass_attention_backward_dispatch`], the function the real
//! training step calls, with the artifact the crate embeds
//! (`shaders/transformer_attention_backward_key_channel4_rdna3.spv`,
//! `f5480cce…`, byte-identical to the Phase XI `c4_u0` SPIR-V). The route is
//! proved by the production dispatch counters, not by the predicate: a case
//! that reports channel-tiled parity must also have `key_channel4 > 0` and
//! `key == 0`.
//!
//! Tests:
//!
//! * [`phase_xii_channel4_key_parity`] — the complete Phase XI matrix
//!   (8q/2kv, 4q/2kv, 8q/4kv, 16q/4kv, 8q/8kv MHA control; modes 0 and 9)
//!   against the scalar oracle and the original two-pass key pass, with dK, dV
//!   and dQ bitwise equality, zero absolute error, zero ULP difference and zero
//!   unwritten elements.
//! * [`phase_xii_channel4_key_ownership`] — the host lane -> (row, chunk)
//!   bijection and the production grid, including the Phase XI workgroup count.
//! * [`phase_xii_channel4_key_head_dim_gate`] — the narrow eligibility
//!   predicate, nodevice required.
//! * [`phase_xii_channel4_key_original_fallback`] — an otherwise-supported MHA
//!   geometry at head dimension 64: the tiled route is rejected with a reason,
//!   the original two-pass key shader dispatches, the scalar shader does not,
//!   no channel-tiled artifact is resolved, and the gradients are bit-identical
//!   to the scalar oracle.
//! * [`phase_xii_channel4_key_optional_loading`] — the Phase IV loading
//!   semantics for the new optional artifact: `=0`, the eligible default, an
//!   unavailable tiled artifact, an unavailable required pair and `=1`.
//! * [`phase_xii_channel4_key_full_step`] — opt-in release-mode training driver
//!   (`HIERARCHOS_PHASE_XII_STEP_ARM=original|channel4`) producing the ABBA
//!   samples, the digests and the dispatch counters.
//!
//! Run:
//!
//! ```text
//! cargo test --lib phase_xii_channel4 -- --nocapture --test-threads=1
//! HIERARCHOS_PHASE_XII_STEP_ARM=channel4 HIERARCHOS_PHASE_XII_STEP_DIGEST=1 \
//!   cargo test --release --lib phase_xii_channel4_key_full_step -- --nocapture
//! ```

use super::*;

/// Written into every output buffer before a dispatch so an element the kernel
/// never writes survives as a sentinel.
const PHASE_XII_SENTINEL: f32 = -98_765.432_1_f32;
const PHASE_XII_BATCH: u32 = 2;
const PHASE_XII_SEQ: u32 = 64;
const PHASE_XII_HEAD_DIM: u32 = 128;

/// Deterministic operands: identical bytes for every arm of a case.
fn phase_xii_operands(salt: u32, count: usize) -> Vec<f32> {
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
struct PhaseXiiComparison {
    elements: usize,
    bitwise: usize,
    max_abs: f32,
    max_ulp: u32,
    first: Option<(usize, u32, u32)>,
}

fn phase_xii_ulp_order(value: f32) -> i64 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 != 0 {
        -(bits as i64)
    } else {
        bits as i64
    }
}

fn phase_xii_compare(expected: &[f32], actual: &[f32]) -> PhaseXiiComparison {
    assert_eq!(expected.len(), actual.len());
    let mut report = PhaseXiiComparison {
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
        let ulp = (phase_xii_ulp_order(expected_value) - phase_xii_ulp_order(actual_value))
            .unsigned_abs() as u32;
        report.max_ulp = report.max_ulp.max(ulp);
    }
    report
}

fn phase_xii_report_line(
    label: &str,
    tensor: &str,
    comparison: &PhaseXiiComparison,
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
        "[phase-xii-parity] {label} tensor={tensor} elements={} bitwise={} max_abs={:.9e} \
         max_ulp={} unwritten={} {first} verdict={}",
        comparison.elements,
        comparison.bitwise,
        comparison.max_abs,
        comparison.max_ulp,
        unwritten,
        if verdict { "PASS" } else { "FAIL" }
    );
}

fn phase_xii_push(
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    batch: u32,
    seq: u32,
    mode: u32,
) -> AttentionPush {
    AttentionPush {
        batch_size: batch,
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
    }
}

struct PhaseXiiCase {
    label: &'static str,
    push: AttentionPush,
    q_count: usize,
    kv_count: usize,
    scratch: usize,
}

impl PhaseXiiCase {
    fn new(label: &'static str, heads: u32, kv_heads: u32, mode: u32) -> Self {
        Self::with_shape(
            label,
            heads,
            kv_heads,
            PHASE_XII_HEAD_DIM,
            PHASE_XII_BATCH,
            PHASE_XII_SEQ,
            mode,
        )
    }

    fn with_shape(
        label: &'static str,
        heads: u32,
        kv_heads: u32,
        head_dim: u32,
        batch: u32,
        seq: u32,
        mode: u32,
    ) -> Self {
        let push = phase_xii_push(heads, kv_heads, head_dim, batch, seq, mode);
        let q_count = (batch * seq * heads * head_dim) as usize;
        let kv_count = (batch * seq * kv_heads * head_dim) as usize;
        let scratch = two_pass_attention_scratch_elements(&push).expect("scratch size");
        Self {
            label,
            push,
            q_count,
            kv_count,
            scratch,
        }
    }

    fn key_rows(&self) -> u32 {
        self.push.batch_size * self.push.num_key_value_heads * self.push.key_value_seq_len
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

    fn scalar_grid(&self) -> [u32; 3] {
        [self.push.num_key_value_heads, self.push.batch_size, 1]
    }

    fn original_key_grid(&self) -> [u32; 3] {
        [div_ceil_u32(self.key_rows() as usize, 64), 1, 1]
    }

    fn channel4_grid(&self) -> [u32; 3] {
        channel4_key_dispatch_grid(self.key_rows())
    }
}

struct PhaseXiiBuffers {
    query: GpuBuffer,
    key: GpuBuffer,
    value: GpuBuffer,
    grad_output: GpuBuffer,
    query_mask: GpuBuffer,
    key_mask: GpuBuffer,
    dummy: GpuBuffer,
}

fn phase_xii_buffers(device: &VulkanDevice, case: &PhaseXiiCase) -> Result<PhaseXiiBuffers> {
    let mask_count =
        (case.push.batch_size * case.push.query_seq_len.max(case.push.key_value_seq_len)) as usize;
    Ok(PhaseXiiBuffers {
        query: GpuBuffer::from_f32(device, &phase_xii_operands(41, case.q_count))?,
        grad_output: GpuBuffer::from_f32(device, &phase_xii_operands(44, case.q_count))?,
        key: GpuBuffer::from_f32(device, &phase_xii_operands(42, case.kv_count))?,
        value: GpuBuffer::from_f32(device, &phase_xii_operands(43, case.kv_count))?,
        query_mask: GpuBuffer::from_f32(device, &vec![1.0_f32; mask_count])?,
        key_mask: GpuBuffer::from_f32(device, &vec![1.0_f32; mask_count])?,
        dummy: GpuBuffer::from_f32(device, &[0.0_f32])?,
    })
}

/// The scalar oracle: the production generic backward, one invocation per
/// `(batch, KV head)`, unsplit. Returns `(dQ, dK, dV)`.
fn phase_xii_scalar(
    device: &VulkanDevice,
    kernels: &TransformerKernels,
    case: &PhaseXiiCase,
    buffers: &PhaseXiiBuffers,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let grad_q = GpuBuffer::from_f32(device, &vec![PHASE_XII_SENTINEL; case.q_count])?;
    let grad_k = GpuBuffer::from_f32(device, &vec![PHASE_XII_SENTINEL; case.kv_count])?;
    let grad_v = GpuBuffer::from_f32(device, &vec![PHASE_XII_SENTINEL; case.kv_count])?;
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
        case.scalar_grid(),
    )?;
    commands.submit()?;
    Ok((
        grad_q.read_f32(case.q_count)?,
        grad_k.read_f32(case.kv_count)?,
        grad_v.read_f32(case.kv_count)?,
    ))
}

/// The production two-pass pair as the loader builds it, for dispatch tests
/// that need an already-resolved pair.
fn phase_xii_pair(device: &VulkanDevice) -> Result<TwoPassAttentionBackward> {
    let query = vulkan::ComputeKernel::new(
        device,
        TwoPassAttentionPass::Query.spirv(),
        TwoPassAttentionPass::Query.binding_count(),
        std::mem::size_of::<AttentionPush>() as u32,
    )?;
    let key = vulkan::ComputeKernel::new(
        device,
        TwoPassAttentionPass::Key.spirv(),
        TwoPassAttentionPass::Key.binding_count(),
        std::mem::size_of::<AttentionPush>() as u32,
    )?;
    Ok(TwoPassAttentionBackward {
        kernels: Some([query, key]),
        reason: "ready",
        detail: None,
    })
}

struct PhaseXiiRoute {
    commands_dispatches: usize,
    counts: AttentionBackwardDispatchCounts,
    grad_q: Vec<f32>,
    grad_k: Vec<f32>,
    grad_v: Vec<f32>,
    channel4_reason: Option<&'static str>,
}

/// Run the real production dispatch helper for one case.
fn phase_xii_route(
    device: &VulkanDevice,
    kernels: &TransformerKernels,
    case: &PhaseXiiCase,
    buffers: &PhaseXiiBuffers,
    pair: &TwoPassAttentionBackward,
) -> Result<PhaseXiiRoute> {
    // Scratch is sentinel-filled before pass 1, so a slot pass 1 never wrote
    // cannot reproduce the oracle's bits.
    let scratch = TwoPassAttentionScratch::new(device, case.scratch)?;
    // The scratch is filled with the sentinel *before* pass 1 runs, so a key
    // lane reading a slot pass 1 never wrote for this causal geometry reads the
    // sentinel and cannot reproduce the oracle's bits.
    scratch
        .probability
        .write_f32(&vec![PHASE_XII_SENTINEL; case.scratch])?;
    scratch.ds.write_f32(&vec![PHASE_XII_SENTINEL; case.scratch])?;

    let grad_q = GpuBuffer::from_f32(device, &vec![PHASE_XII_SENTINEL; case.q_count])?;
    let grad_k = GpuBuffer::from_f32(device, &vec![PHASE_XII_SENTINEL; case.kv_count])?;
    let grad_v = GpuBuffer::from_f32(device, &vec![PHASE_XII_SENTINEL; case.kv_count])?;

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
    let commands_dispatches = commands.dispatch_count();
    commands.submit()?;
    Ok(PhaseXiiRoute {
        commands_dispatches,
        counts: two_pass_attention_test_hooks::thread_counts(),
        grad_q: grad_q.read_f32(case.q_count)?,
        grad_k: grad_k.read_f32(case.kv_count)?,
        grad_v: grad_v.read_f32(case.kv_count)?,
        channel4_reason: kernels
            .rdna3_channel4_attention_backward_key
            .get()
            .map(|candidate| candidate.reason),
    })
}

#[allow(clippy::too_many_arguments)]
fn phase_xii_assert_bitwise(
    label: &str,
    oracle: &[f32],
    actual: &[f32],
    unwritten_label: &str,
    failures: &mut usize,
    elements: &mut usize,
    tensors: &mut usize,
    unwritten_total: &mut usize,
) {
    let unwritten = actual
        .iter()
        .filter(|value| value.to_bits() == PHASE_XII_SENTINEL.to_bits())
        .count();
    *unwritten_total += unwritten;
    let comparison = phase_xii_compare(oracle, actual);
    let ok = comparison.bitwise == comparison.elements && unwritten == 0;
    if !ok {
        *failures += 1;
    }
    *tensors += 1;
    *elements += comparison.elements;
    phase_xii_report_line(
        &format!("{label} baseline=scalar-oracle"),
        unwritten_label,
        &comparison,
        unwritten,
        ok,
    );
}

#[test]
fn phase_xii_channel4_key_parity() -> Result<()> {
    let Ok(device) = VulkanDevice::new() else {
        eprintln!("phase-xii parity: status=BLOCKED reason=no_vulkan_device");
        return Ok(());
    };
    let _fixed = two_pass_attention_test_hooks::scope(Some(None), None);
    // Phase XVIII promotes the dq-unroll query pass on by default, which changes
    // which query counter this route records. This test is about the Phase XII
    // channel-4 key path, so pin the query pass back to the shipped shader for
    // the duration of the test instead of relying on the process environment.
    let _dq = two_pass_attention_test_hooks::scope_dqunroll(Some(Some(false)), false);
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

    // Three kernel sets for the whole matrix: the oracle, a set whose optional
    // tiled artifact is resolved while the production opt-out is in force, and
    // a set that resolves it under the eligible default. Each set caches its
    // answer, so a later case cannot silently change its arm.
    let oracle_kernels = TransformerKernels::new(&device)?;
    let original_kernels = TransformerKernels::new(&device)?;
    let tiled_kernels = TransformerKernels::new(&device)?;
    let pair = phase_xii_pair(&device)?;

    for mode in [0u32, 9u32] {
        for (label, heads, kv_heads) in geometries {
            let case = PhaseXiiCase::new(label, heads, kv_heads, mode);
            let buffers = phase_xii_buffers(&device, &case)?;

            // Scalar oracle from the generic kernel, one dispatch.
            let (oracle_q, oracle_k, oracle_v) =
                phase_xii_scalar(&device, &oracle_kernels, &case, &buffers)?;

            // Arm A: the original two-pass key shader, selected by the
            // production opt-out. Proves the fallback still owns the geometry.
            let original = {
                let _opt_out =
                    two_pass_attention_test_hooks::scope_channel4(Some(Some(false)), false);
                phase_xii_route(&device, &original_kernels, &case, &buffers, &pair)?
            };
            assert_eq!(
                original.counts.query, 1,
                "[{label} mode={mode}] the query pass did not dispatch exactly once"
            );
            assert_eq!(
                original.counts.key, 1,
                "[{label} mode={mode}] the original key pass did not dispatch exactly once"
            );
            assert_eq!(
                original.counts.key_channel4, 0,
                "[{label} mode={mode}] the opt-out still dispatched the tiled key pass"
            );
            assert_eq!(
                original.channel4_reason,
                Some("disabled_by_environment"),
                "[{label} mode={mode}] the opt-out built the tiled artifact"
            );
            assert_eq!(
                original.commands_dispatches, 2,
                "[{label} mode={mode}] the original route recorded {} dispatches",
                original.commands_dispatches
            );

            // Arm B: the promoted four-channel key pass, selected by the
            // production default with no test override of the pair.
            let tiled = {
                let _default = two_pass_attention_test_hooks::scope_channel4(Some(None), false);
                phase_xii_route(&device, &tiled_kernels, &case, &buffers, &pair)?
            };
            assert_eq!(
                tiled.counts.query, 1,
                "[{label} mode={mode}] the query pass did not dispatch exactly once"
            );
            assert_eq!(
                tiled.counts.key_channel4, 1,
                "[{label} mode={mode}] the production route did not dispatch the tiled key pass"
            );
            assert_eq!(
                tiled.counts.key, 0,
                "[{label} mode={mode}] the production route dispatched the original key pass"
            );
            assert_eq!(
                tiled.counts.scalar, 0,
                "[{label} mode={mode}] the production route dispatched the scalar kernel"
            );
            assert_eq!(
                tiled.channel4_reason,
                Some("ready"),
                "[{label} mode={mode}] the tiled artifact did not initialize"
            );
            assert_eq!(
                tiled.commands_dispatches, 2,
                "[{label} mode={mode}] the tiled route recorded {} dispatches",
                tiled.commands_dispatches
            );
            if (label, mode) == ("8q-2kv", 9) {
                // The Phase XI measured geometry: the production grid must be
                // the same lane set the candidate was measured with.
                assert_eq!(case.channel4_grid(), [128, 1, 1]);
                assert_eq!(case.key_rows(), 256);
                assert_eq!(case.channel4_grid()[0] * 64, 8_192);
            }

            for (arm, route) in [("key_original", &original), ("key_channel4", &tiled)] {
                cases += 1;
                for (name, oracle, actual) in [
                    ("dQ", &oracle_q, &route.grad_q),
                    ("dK", &oracle_k, &route.grad_k),
                    ("dV", &oracle_v, &route.grad_v),
                ] {
                    phase_xii_assert_bitwise(
                        &format!("geometry={label} mode={mode} arm={arm}"),
                        oracle,
                        actual,
                        name,
                        &mut failures,
                        &mut elements,
                        &mut tensors,
                        &mut unwritten_total,
                    );
                }
            }
        }
    }

    eprintln!(
        "[phase-xii-parity-summary] status={} cases={cases} tensors={tensors} \
         elements={elements} unwritten={unwritten_total} failures={failures}",
        if failures == 0 { "PASS" } else { "FAIL" }
    );
    assert_eq!(failures, 0, "the promoted key pass is not bit-exact");
    Ok(())
}

/// Replay the promoted lane mapping over the exact launched lane set.
fn phase_xii_bijection(
    rows: u32,
    chunk: u32,
    blocks: u32,
    qk_dim: u32,
    value_dim: u32,
) -> (usize, usize, usize, u32) {
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
    for &count in owners[..k_len].iter().chain(owners[k_len..k_len + v_len].iter()) {
        if count == 0 {
            missing += 1;
        } else {
            covered += 1;
        }
        if count > 1 {
            duplicated += 1;
        }
    }
    (covered, missing, duplicated, useful)
}

#[test]
fn phase_xii_channel4_key_ownership() -> Result<()> {
    assert_eq!(CHANNEL4_KEY_CHUNK, 4);
    assert_eq!(CHANNEL4_KEY_BLOCKS, 32);
    assert_eq!(
        CHANNEL4_KEY_BLOCKS * CHANNEL4_KEY_CHUNK,
        CHANNEL4_KEY_HEAD_DIM
    );
    let mut failures = 0usize;
    for (label, heads, kv_heads) in [("8q-2kv", 8u32, 2u32), ("8q-8kv-mha", 8u32, 8u32)] {
        let case = PhaseXiiCase::new(label, heads, kv_heads, 9);
        let rows = case.key_rows();
        let (covered, missing, duplicated, useful) = phase_xii_bijection(
            rows,
            CHANNEL4_KEY_CHUNK,
            CHANNEL4_KEY_BLOCKS,
            PHASE_XII_HEAD_DIM,
            PHASE_XII_HEAD_DIM,
        );
        let expected = (rows as usize) * (PHASE_XII_HEAD_DIM as usize) * 2;
        let grid = case.channel4_grid();
        let lane_slots = rows * CHANNEL4_KEY_BLOCKS;
        let ok = covered == expected
            && missing == 0
            && duplicated == 0
            && useful == lane_slots
            && grid == [lane_slots / 64, 1, 1];
        if !ok {
            failures += 1;
        }
        eprintln!(
            "[phase-xii-ownership] geometry={label} rows={rows} chunk={} blocks={} \
             lane_slots={lane_slots} useful_lanes={useful} output_elements={expected} \
             covered={covered} missing={missing} duplicated={duplicated} \
             workgroups={} invocations={} verdict={}",
            CHANNEL4_KEY_CHUNK,
            CHANNEL4_KEY_BLOCKS,
            grid[0],
            grid[0] * 64,
            if ok { "PASS" } else { "FAIL" }
        );
    }
    assert_eq!(failures, 0, "the promoted lane mapping is not a partition");
    Ok(())
}

#[test]
fn phase_xii_channel4_key_head_dim_gate() {
    for (label, heads, kv_heads) in [
        ("8q-2kv", 8u32, 2u32),
        ("4q-2kv", 4, 2),
        ("8q-4kv", 8, 4),
        ("16q-4kv", 16, 4),
        ("8q-8kv-mha", 8, 8),
    ] {
        let push = phase_xii_push(heads, kv_heads, 128, 2, 64, 9);
        assert_eq!(
            two_pass_attention_rejection(&push),
            None,
            "[{label}] the two-pass gate no longer admits the qualified geometry"
        );
        assert_eq!(
            channel4_key_attention_rejection(&push),
            None,
            "[{label}] the tiled gate refuses a qualified head dimension"
        );
    }
    for (head_dim, expected) in [
        (64u32, "channel4_head_dim_not_128"),
        (32, "channel4_head_dim_not_128"),
        (16, "channel4_head_dim_not_128"),
    ] {
        let push = phase_xii_push(1, 1, head_dim, 1, 8, 9);
        assert_eq!(
            two_pass_attention_rejection(&push),
            None,
            "head_dim {head_dim} is not an otherwise-supported MHA geometry"
        );
        assert_eq!(
            channel4_key_attention_rejection(&push),
            Some(expected),
            "head_dim {head_dim} must not reach the tiled route"
        );
    }
    // A grouped geometry at 128 is admitted; the value width is a separate
    // condition because the candidate guards dK and dV independently.
    let mut mismatched = phase_xii_push(8, 2, 128, 2, 64, 9);
    mismatched.value_hidden_size = 2 * 64;
    assert_eq!(
        channel4_key_attention_rejection(&mismatched),
        Some("channel4_value_head_dim_not_128")
    );
    eprintln!("[phase-xii-head-dim-gate] status=PASS head_dims=128 admitted,16/32/64 rejected");
}

/// An otherwise-supported MHA geometry at head dimension 64: the tiled route is
/// rejected, the original two-pass key shader dispatches, the scalar shader does
/// not, and the result stays bit-identical to the scalar oracle.
#[test]
fn phase_xii_channel4_key_original_fallback() -> Result<()> {
    let Ok(device) = VulkanDevice::new() else {
        eprintln!("phase-xii fallback: status=BLOCKED reason=no_vulkan_device");
        return Ok(());
    };
    let _fixed = two_pass_attention_test_hooks::scope(Some(None), None);
    let case = PhaseXiiCase::with_shape("1q-1kv-hd64-mha", 1, 1, 64, 1, 8, 9);
    assert_eq!(two_pass_attention_rejection(&case.push), None);
    assert_eq!(
        channel4_key_attention_rejection(&case.push),
        Some("channel4_head_dim_not_128")
    );

    let buffers = phase_xii_buffers(&device, &case)?;
    let oracle_kernels = TransformerKernels::new(&device)?;
    let (oracle_q, oracle_k, oracle_v) =
        phase_xii_scalar(&device, &oracle_kernels, &case, &buffers)?;

    let kernels = TransformerKernels::new(&device)?;
    let pair = phase_xii_pair(&device)?;
    let route = {
        let _default = two_pass_attention_test_hooks::scope_channel4(Some(None), false);
        phase_xii_route(&device, &kernels, &case, &buffers, &pair)?
    };
    assert_eq!(
        route.counts.query, 1,
        "the fallback geometry did not dispatch the query pass"
    );
    assert_eq!(
        route.counts.key, 1,
        "the original key shader did not dispatch for head_dim 64"
    );
    assert_eq!(
        route.counts.key_channel4, 0,
        "the tiled route claimed a head dimension it was not qualified on"
    );
    assert_eq!(route.counts.scalar, 0, "the scalar kernel dispatched");
    assert_eq!(
        route.channel4_reason, None,
        "the tiled artifact was resolved for an ineligible geometry"
    );
    assert_eq!(route.commands_dispatches, 2);
    assert_eq!(case.original_key_grid(), [1, 1, 1]);
    assert_eq!(case.scratch, 1 * 1 * 8 * 8);

    for (name, oracle, actual) in [
        ("dQ", &oracle_q, &route.grad_q),
        ("dK", &oracle_k, &route.grad_k),
        ("dV", &oracle_v, &route.grad_v),
    ] {
        let comparison = phase_xii_compare(oracle, actual);
        assert_eq!(
            comparison.bitwise, comparison.elements,
            "[fallback head_dim=64] {name} is not bit-identical"
        );
        assert_eq!(comparison.max_ulp, 0, "[fallback head_dim=64] {name} ULP drift");
    }
    eprintln!(
        "[phase-xii-fallback] status=PASS geometry=1q-1kv-hd64-mha reason=channel4_head_dim_not_128 \
         query=1 key_original=1 key_channel4=0 scalar=0 channel4_resolved=false scratch_elements={}",
        case.scratch
    );
    Ok(())
}

/// The tiled artifact exists in the build but this device cannot initialize it:
/// the production dispatch must fall back to the original qualified two-pass key
/// shader, not to the scalar kernel, and stay bit-exact.
#[test]
fn phase_xii_channel4_key_unavailable_falls_back_to_original() -> Result<()> {
    let Ok(device) = VulkanDevice::new() else {
        eprintln!("phase-xii unavailable: status=BLOCKED reason=no_vulkan_device");
        return Ok(());
    };
    let _fixed = two_pass_attention_test_hooks::scope(Some(None), None);
    // The promoted dq-unroll query pass is the production default after Phase
    // XVIII; pin it off here so the assertions below keep describing the Phase
    // XII channel-4 key route rather than depending on the caller's environment.
    let _dq = two_pass_attention_test_hooks::scope_dqunroll(Some(Some(false)), false);
    let case = PhaseXiiCase::new("8q-2kv", 8, 2, 9);
    assert_eq!(channel4_key_attention_rejection(&case.push), None);
    let buffers = phase_xii_buffers(&device, &case)?;
    let oracle_kernels = TransformerKernels::new(&device)?;
    let (_, oracle_k, oracle_v) = phase_xii_scalar(&device, &oracle_kernels, &case, &buffers)?;

    let kernels = TransformerKernels::new(&device)?;
    let pair = phase_xii_pair(&device)?;
    let route = {
        let _faulted = two_pass_attention_test_hooks::scope_channel4(Some(None), true);
        phase_xii_route(&device, &kernels, &case, &buffers, &pair)?
    };
    assert_eq!(
        route.channel4_reason,
        Some("channel4_shader_unavailable"),
        "the loader did not record the unavailable artifact"
    );
    assert_eq!(route.counts.query, 1);
    assert_eq!(
        route.counts.key, 1,
        "the original key shader did not take over an unavailable tiled artifact"
    );
    assert_eq!(route.counts.key_channel4, 0);
    assert_eq!(route.counts.scalar, 0, "the fallback dropped to the scalar kernel");
    assert_eq!(route.commands_dispatches, 2);
    for (name, oracle, actual) in [("dK", &oracle_k, &route.grad_k), ("dV", &oracle_v, &route.grad_v)] {
        let comparison = phase_xii_compare(oracle, actual);
        assert_eq!(comparison.bitwise, comparison.elements, "{name} is not bit-identical");
        assert_eq!(comparison.max_ulp, 0);
    }
    eprintln!(
        "[phase-xii-unavailable] status=PASS geometry=8q-2kv-hd128 reason=channel4_shader_unavailable \
         query=1 key_original=1 key_channel4=0 scalar=0 scratch_elements={}",
        case.scratch
    );
    Ok(())
}

#[test]
fn phase_xii_channel4_key_optional_loading() -> Result<()> {
    let Ok(device) = VulkanDevice::new() else {
        eprintln!("phase-xii loading: status=BLOCKED reason=no_vulkan_device");
        return Ok(());
    };
    // 1. `=0` loads neither the pair nor the tiled artifact and builds nothing.
    {
        let _off = two_pass_attention_test_hooks::scope(Some(Some(false)), None);
        let pair = TwoPassAttentionBackward::load(&device)?;
        let tiled = Channel4KeyBackward::load(&device)?;
        assert_eq!(pair.reason, "disabled_by_environment");
        assert!(pair.kernels.is_none());
        assert_eq!(tiled.reason, "disabled_by_environment");
        assert!(tiled.kernel.is_none());
        assert_eq!(
            two_pass_attention_test_hooks::optional_kernel_loads(),
            0,
            "=0 built an optional kernel"
        );
    }
    // 2. The eligible default resolves all three artifacts.
    {
        let _default = two_pass_attention_test_hooks::scope(Some(None), None);
        let pair = TwoPassAttentionBackward::load(&device)?;
        let tiled = Channel4KeyBackward::load(&device)?;
        assert_eq!(pair.reason, "ready");
        assert_eq!(tiled.reason, "ready");
        assert!(tiled.kernel.is_some());
        assert_eq!(two_pass_attention_test_hooks::optional_kernel_loads(), 3);
        eprintln!(
            "[phase-xii-loading] default=rda3-ready pair=ready channel4=ready loads=3"
        );
    }
    // 3. An unavailable tiled artifact leaves the original key pass in charge.
    {
        let _faulted = two_pass_attention_test_hooks::scope_channel4(Some(None), true);
        let pair = TwoPassAttentionBackward::load(&device)?;
        let tiled = Channel4KeyBackward::load(&device)?;
        assert_eq!(pair.reason, "ready");
        assert_eq!(tiled.reason, "channel4_shader_unavailable");
        assert!(tiled.detail.is_some());
        assert!(tiled.kernel.is_none());
        assert_eq!(two_pass_attention_test_hooks::optional_kernel_loads(), 3);
    }
    // 4. The opt-out builds no tiled pipeline at all.
    {
        let _opt_out = two_pass_attention_test_hooks::scope_channel4(Some(Some(false)), false);
        let tiled = Channel4KeyBackward::load(&device)?;
        assert_eq!(tiled.reason, "disabled_by_environment");
        assert_eq!(two_pass_attention_test_hooks::optional_kernel_loads(), 0);
    }
    // 5. An unavailable required pair is a visible failure under `=1` and an
    //    ordinary scalar fallback otherwise; either way the tiled artifact is
    //    never resolved.
    {
        let _both = two_pass_attention_test_hooks::scope(Some(None), Some(
            two_pass_attention_test_hooks::ShaderFault::Both,
        ));
        let pair = TwoPassAttentionBackward::load(&device)?;
        assert_eq!(pair.reason, "both_shaders_unavailable");
        assert!(pair.kernels.is_none());
        assert_eq!(two_pass_attention_test_hooks::optional_kernel_loads(), 2);
    }
    {
        let _qualified =
            two_pass_attention_test_hooks::scope(Some(Some(true)), Some(
                two_pass_attention_test_hooks::ShaderFault::Query,
            ));
        let blocked = match TwoPassAttentionBackward::load(&device) {
            Ok(_) => panic!("=1 must not fall back silently"),
            Err(error) => error,
        };
        let message = format!("{blocked:#}");
        assert!(message.contains("BLOCKED"), "{message}");
        assert!(message.contains("query_shader_unavailable"), "{message}");
    }
    eprintln!(
        "[phase-xii-loading-summary] status=PASS cases=6 disabled_off=0 default=3 \
         channel4_fault=3 channel4_opt_out=0 pair_fault=2 explicit_one=BLOCKED"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Release-mode complete-step driver (ABBA sample source).
// ---------------------------------------------------------------------------

struct PhaseXiiSequence {
    input_ids: Vec<u32>,
    targets: Vec<u32>,
    attention_mask: Vec<f32>,
    loss_weights: Vec<f32>,
}

fn phase_xii_load_sequences(
    path: &std::path::Path,
    seq_len: usize,
) -> Result<Vec<PhaseXiiSequence>> {
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
            let mut sequence = PhaseXiiSequence {
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

fn phase_xii_batch_arrays(
    sequences: &[PhaseXiiSequence],
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

fn phase_xii_digest(values: &std::collections::BTreeMap<String, Vec<f32>>) -> (String, usize, f64) {
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

/// `HIERARCHOS_PHASE_XII_STEP_ARM`   `original` or `channel4` (required)
/// `HIERARCHOS_PHASE_XII_FIXTURE`    fixture dir (default `target/gqa-e2e/fixture-8q2kv`)
/// `HIERARCHOS_PHASE_XII_STEP_STEPS` steps per process (default 29)
/// `HIERARCHOS_PHASE_XII_STEP_DIGEST=1` print loss bits and digests every step
/// `HIERARCHOS_PHASE_XII_STEP_CHECKPOINT=<dir>` write adapter + optimizer state
///
/// The arm is chosen by production code only: `original` runs with
/// `HIERARCHOS_ATTENTION_BACKWARD_KEY_CHANNEL4=0` in the process environment,
/// `channel4` with it unset. The driver refuses to run if the dispatch counters
/// disagree with the requested arm.
#[test]
fn phase_xii_channel4_key_full_step() -> Result<()> {
    let arm = match std::env::var("HIERARCHOS_PHASE_XII_STEP_ARM") {
        Ok(value) => value.trim().to_string(),
        Err(_) => {
            eprintln!(
                "phase-xii full step: status=SKIP reason=not_requested \
                 (set HIERARCHOS_PHASE_XII_STEP_ARM=original|channel4)"
            );
            return Ok(());
        }
    };
    anyhow::ensure!(
        arm == "original" || arm == "channel4",
        "unknown phase-xii step arm {arm}"
    );
    let fixture = std::env::var("HIERARCHOS_PHASE_XII_FIXTURE")
        .unwrap_or_else(|_| "target/gqa-e2e/fixture-8q2kv".to_string());
    let fixture = std::path::PathBuf::from(fixture);
    let steps: usize = std::env::var("HIERARCHOS_PHASE_XII_STEP_STEPS")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(29);
    let warmup: usize = 3;
    let digest = std::env::var("HIERARCHOS_PHASE_XII_STEP_DIGEST")
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

    let sequences = phase_xii_load_sequences(&fixture.join("dataset.jsonl"), seq_len)?;
    let batches_per_epoch = sequences.len() / batch;
    anyhow::ensure!(batches_per_epoch > 0, "fixture dataset is smaller than one batch");

    let device = VulkanDevice::new()?;
    let mut graph =
        VulkanTransformer::from_hf_package(device, fixture.join("model"), batch, seq_len)?;
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
    let start_parameters = phase_xii_digest(&graph.parameter_values()?);

    let installed = graph.kernels.rdna3_row_packed_attention.is_some();
    let two_pass_enabled = attention_backward_two_pass_enabled();
    let channel4_enabled = attention_backward_key_channel4_enabled();
    anyhow::ensure!(installed, "the production row-packed forward is not installed");
    anyhow::ensure!(two_pass_enabled, "the two-pass backward is not enabled");
    anyhow::ensure!(
        channel4_enabled == (arm == "channel4"),
        "arm {arm} disagrees with the production channel4 switch \
         (HIERARCHOS_ATTENTION_BACKWARD_KEY_CHANNEL4)"
    );

    let forward_before = attention_forward_dispatch_counts();
    let backward_before = attention_backward_dispatch_counts();
    eprintln!(
        "[phase-xii-step-run] arm={arm} fixture={} sequences={} batches_per_epoch={batches_per_epoch} \
         batch={batch} seq={seq_len} steps={steps} warmup={warmup} lora_rank=8 lora_alpha=16 \
         lr={} seed={seed} digest={digest} device={} row_packed_installed={installed} \
         two_pass_enabled={two_pass_enabled} channel4_enabled={channel4_enabled} \
         parameter_start_digest={} parameter_elements={}",
        fixture.display(),
        sequences.len(),
        hyper.lr,
        graph.device_name(),
        start_parameters.0,
        start_parameters.1
    );

    for index in 0..steps {
        let global_step = index;
        let epoch = global_step / batches_per_epoch;
        let batch_index = global_step % batches_per_epoch;
        let begin = batch_index * batch;
        let (input_ids, targets, attention_mask, loss_weights) =
            phase_xii_batch_arrays(&sequences[begin..begin + batch], seq_len)?;
        let started = Instant::now();
        let result = graph.train_step(&input_ids, &targets, &attention_mask, &loss_weights, hyper)?;
        let step_ms = started.elapsed().as_secs_f64() * 1e3;
        let measured = index >= warmup;
        if digest {
            let gradients = graph.lora_gradient_values()?;
            let gradient_digest = phase_xii_digest(&gradients);
            let parameters = phase_xii_digest(&graph.parameter_values()?);
            eprintln!(
                "[phase-xii-step] arm={arm} step={index} epoch={epoch} batch={batch_index} \
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
                "[phase-xii-step] arm={arm} step={index} epoch={epoch} batch={batch_index} \
                 measured={measured} step_ms={step_ms:.6} loss_bits={:08x} loss={:e}",
                result.loss.to_bits(),
                result.loss
            );
        }
    }
    let forward_after = attention_forward_dispatch_counts();
    let backward_after = attention_backward_dispatch_counts();
    let end_parameters = phase_xii_digest(&graph.parameter_values()?);
    let query = backward_after.query - backward_before.query;
    let key = backward_after.key - backward_before.key;
    let key_channel4 = backward_after.key_channel4 - backward_before.key_channel4;
    let scalar = backward_after.scalar - backward_before.scalar;
    eprintln!(
        "[phase-xii-step-run-end] arm={arm} parameter_end_digest={} parameter_elements={} \
         forward_generic={} forward_row_packed={} forward_short={} forward_total={} \
         backward_scalar={scalar} backward_query={query} backward_key_original={key} \
         backward_key_channel4={key_channel4}",
        end_parameters.0,
        end_parameters.1,
        forward_after.generic - forward_before.generic,
        forward_after.row_packed - forward_before.row_packed,
        forward_after.short - forward_before.short,
        forward_after.total() - forward_before.total(),
    );
    // The production counters, not the environment, decide whether the arm ran
    // what it claims: every step runs two layers, so two key dispatches.
    match arm.as_str() {
        "channel4" => {
            anyhow::ensure!(
                key_channel4 == (steps * 2) as u64 && key == 0,
                "the channel4 arm dispatched key_channel4={key_channel4} key_original={key}, \
                 expected {} and 0",
                steps * 2
            );
        }
        _ => {
            anyhow::ensure!(
                key == (steps * 2) as u64 && key_channel4 == 0,
                "the original arm dispatched key_original={key} key_channel4={key_channel4}, \
                 expected {} and 0",
                steps * 2
            );
        }
    }
    anyhow::ensure!(scalar == 0, "the step dispatched the scalar kernel {scalar} times");

    if let Ok(directory) = std::env::var("HIERARCHOS_PHASE_XII_STEP_CHECKPOINT") {
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
                "[phase-xii-checkpoint] arm={arm} file={name} bytes={} sha256={digest}",
                std::fs::metadata(&path)?.len()
            );
        }
    }
    Ok(())
}
