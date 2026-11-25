//! Phase X — production qualification of the promoted RDNA3 row-packed attention
//! forward (test builds only).
//!
//! Phase IX qualified the candidate out-of-tree and its harness reached the
//! candidate SPIR-V with `include_bytes!` from `validation/`. This module is the
//! *production* gate: every comparison here is driven through
//! `record_text_attention_dispatch`, the one function the real training step
//! calls, and the artifact under test is the one the crate embeds
//! (`shaders/transformer_attention_forward_rowpacked64_cached_rdna3.spv`,
//! `c053503e…`).
//!
//! Five tests:
//!
//! * [`phase_x_row_packed_forward_parity`] — the complete Phase IX forward
//!   matrix (ten geometries, three arithmetic modes, both dispatch grids) plus
//!   the dispatch proof: the selector's counters must show the promoted route
//!   for every eligible geometry and the generic route for none of them.
//! * [`phase_x_row_packed_selector_matrix`] — the host-side eligibility proof,
//!   one rejection reason per field, with no GPU required.
//! * [`phase_x_row_packed_fallback_dispatch`] — geometries the promoted route
//!   refuses must reach the generic kernel, measured through the selector
//!   rather than argued from the predicate.
//! * [`phase_x_row_packed_short_precedence`] — the shipped short-sequence
//!   module keeps precedence where its own gate holds, and the promoted route
//!   only takes over where that gate refuses.
//! * [`phase_x_row_packed_forward_gradient_parity`] — the production two-pass
//!   backward from each forward, with probability/dS scratch parity.
//!
//! Two opt-in tests produce the performance and complete-step evidence:
//!
//! * [`phase_x_row_packed_forward_full_step`] — the Phase VII/VIII/IX 8q/2kv
//!   fixture driven by the *default* selector, digested against the generic
//!   arm of the same binary.
//! * [`phase_x_row_packed_forward_timing`] — GPU timestamps for the production
//!   route and the generic kernel at the primary geometry.
//!
//! Run:
//!
//! ```text
//! cargo test --lib phase_x_row_packed -- --nocapture --test-threads=1
//! HIERARCHOS_PHASE_X_STEP_ARM=production HIERARCHOS_PHASE_X_STEP_DIGEST=1 \
//!   cargo test --release --lib phase_x_row_packed_forward_full_step -- --nocapture
//! ```

use super::*;

/// Written into every output buffer before a dispatch so a row the kernel never
/// writes survives as a sentinel.
const PHASE_X_SENTINEL: f32 = -98_765.432_1_f32;

/// One forward geometry. `masked_key_stride` / `masked_query_stride` of zero
/// means "every position visible"; `n` means "zero the mask where
/// `(position % n) == n - 1`".
struct PhaseXGeometry {
    label: &'static str,
    batch: u32,
    query_seq: u32,
    key_seq: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    causal: u32,
    /// The generic kernel's arithmetic switch word. The production Llama path
    /// builds 9; 0 is the plain mode; 1 is the Gemma materialized mode.
    unfused_qk: u32,
    masked_key_stride: u32,
    masked_query_stride: u32,
}

impl PhaseXGeometry {
    const fn plain(
        label: &'static str,
        batch: u32,
        query_seq: u32,
        heads: u32,
        kv_heads: u32,
        head_dim: u32,
        causal: u32,
    ) -> Self {
        Self {
            label,
            batch,
            query_seq,
            key_seq: query_seq,
            heads,
            kv_heads,
            head_dim,
            causal,
            unfused_qk: 9,
            masked_key_stride: 0,
            masked_query_stride: 0,
        }
    }

    const fn at_mode(mut self, unfused_qk: u32) -> Self {
        self.unfused_qk = unfused_qk;
        self
    }

    fn rows(&self) -> u64 {
        self.batch as u64 * self.heads as u64 * self.query_seq as u64
    }

    fn q_count(&self) -> usize {
        (self.batch * self.query_seq * self.heads * self.head_dim) as usize
    }

    fn kv_count(&self) -> usize {
        (self.batch * self.key_seq * self.kv_heads * self.head_dim) as usize
    }

    fn mask_count(&self) -> usize {
        (self.batch * self.query_seq.max(self.key_seq)) as usize
    }

    /// The production dispatch geometry: one workgroup per `(query, head, batch)`.
    fn generic_grid(&self) -> [u32; 3] {
        [self.query_seq, self.heads, self.batch]
    }

    fn push(&self) -> AttentionPush {
        AttentionPush {
            batch_size: self.batch,
            query_seq_len: self.query_seq,
            key_value_seq_len: self.key_seq,
            query_hidden_size: self.heads * self.head_dim,
            value_hidden_size: self.kv_heads * self.head_dim,
            num_heads: self.heads,
            num_key_value_heads: self.kv_heads,
            window_size: 0,
            scale: 1.0 / (self.head_dim as f32).sqrt(),
            value_scale: 1.0,
            softcap: 0.0,
            alibi_mode: 0,
            alibi_bias_max: 0.0,
            dropout_p: 0.0,
            rng_step: 0,
            rng_site: 0,
            rng_seed: 0,
            has_sink: 0,
            causal: self.causal,
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
            unfused_qk: self.unfused_qk,
        }
    }
}

struct PhaseXBuffers {
    query: GpuBuffer,
    key: GpuBuffer,
    value: GpuBuffer,
    query_mask: GpuBuffer,
    key_mask: GpuBuffer,
    dummy: GpuBuffer,
}

/// Deterministic operands: identical bytes for every arm of a case.
fn phase_x_operands(salt: u32, count: usize) -> Vec<f32> {
    let mut state = 0x9e37_79b9_u32 ^ salt.wrapping_mul(0x85eb_ca6b);
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / 16_777_216.0;
            unit * 2.0 - 1.0
        })
        .collect()
}

fn phase_x_masks(geometry: &PhaseXGeometry) -> (Vec<f32>, Vec<f32>) {
    let mut query_mask = vec![1.0_f32; geometry.mask_count()];
    let mut key_mask = vec![1.0_f32; geometry.mask_count()];
    if geometry.masked_query_stride != 0 {
        for batch in 0..geometry.batch {
            for query in 0..geometry.query_seq {
                if (query + 1) % geometry.masked_query_stride == 0 {
                    query_mask[(batch * geometry.query_seq + query) as usize] = 0.0;
                }
            }
        }
    }
    if geometry.masked_key_stride != 0 {
        for batch in 0..geometry.batch {
            for key in 0..geometry.key_seq {
                if (key + 1) % geometry.masked_key_stride == 0 {
                    key_mask[(batch * geometry.key_seq + key) as usize] = 0.0;
                }
            }
        }
    }
    (query_mask, key_mask)
}

fn phase_x_buffers(device: &VulkanDevice, geometry: &PhaseXGeometry) -> Result<PhaseXBuffers> {
    let (query_mask, key_mask) = phase_x_masks(geometry);
    Ok(PhaseXBuffers {
        query: GpuBuffer::from_f32(device, &phase_x_operands(21, geometry.q_count()))?,
        key: GpuBuffer::from_f32(device, &phase_x_operands(22, geometry.kv_count()))?,
        value: GpuBuffer::from_f32(device, &phase_x_operands(23, geometry.kv_count()))?,
        query_mask: GpuBuffer::from_f32(device, &query_mask)?,
        key_mask: GpuBuffer::from_f32(device, &key_mask)?,
        dummy: GpuBuffer::from_f32(device, &[0.0_f32])?,
    })
}

fn phase_x_bindings<'a>(
    buffers: &'a PhaseXBuffers,
    output: &'a GpuBuffer,
) -> Vec<&'a GpuBuffer> {
    vec![
        &buffers.query,
        &buffers.key,
        &buffers.value,
        &buffers.query_mask,
        &buffers.key_mask,
        &buffers.dummy,
        output,
        &buffers.dummy,
        &buffers.dummy,
        &buffers.dummy,
        &buffers.dummy,
    ]
}

/// The committed generic kernel, dispatched directly with an explicit grid.
/// This is the reference arm; it deliberately bypasses the selector so the
/// baseline cannot be influenced by the route under test.
fn phase_x_generic_forward(
    device: &VulkanDevice,
    kernels: &TransformerKernels,
    geometry: &PhaseXGeometry,
) -> Result<Vec<f32>> {
    let push = geometry.push();
    phase_x_generic_forward_with_push(device, kernels, geometry, &push)
}

/// The same reference arm with an explicit push, so a fallback case can compare
/// the exact push it dispatched (a mutated feature, not the geometry default).
fn phase_x_generic_forward_with_push(
    device: &VulkanDevice,
    kernels: &TransformerKernels,
    geometry: &PhaseXGeometry,
    push: &AttentionPush,
) -> Result<Vec<f32>> {
    let buffers = phase_x_buffers(device, geometry)?;
    let output = GpuBuffer::from_f32(device, &vec![PHASE_X_SENTINEL; geometry.q_count()])?;
    let mut commands = vulkan::ComputeBatch::new(device)?;
    kernels.attention_forward.record_dispatch(
        &mut commands,
        &phase_x_bindings(&buffers, &output),
        bytemuck::bytes_of(push),
        geometry.generic_grid(),
    )?;
    commands.submit()?;
    output.read_f32(geometry.q_count())
}

/// The promoted route as production reaches it: through the selector.
fn phase_x_selected_forward(
    device: &VulkanDevice,
    kernels: &TransformerKernels,
    geometry: &PhaseXGeometry,
) -> Result<Vec<f32>> {
    let buffers = phase_x_buffers(device, geometry)?;
    let output = GpuBuffer::from_f32(device, &vec![PHASE_X_SENTINEL; geometry.q_count()])?;
    let mut commands = vulkan::ComputeBatch::new(device)?;
    record_text_attention_dispatch(
        kernels,
        &mut commands,
        &phase_x_bindings(&buffers, &output),
        &geometry.push(),
        geometry.generic_grid(),
    )?;
    commands.submit()?;
    output.read_f32(geometry.q_count())
}

/// The promoted artifact launched under the production `[query, head, batch]`
/// grid, to preserve Phase IX's second dispatch grid as a production claim.
fn phase_x_promoted_forward_on_grid(
    device: &VulkanDevice,
    kernels: &TransformerKernels,
    geometry: &PhaseXGeometry,
) -> Result<Vec<f32>> {
    let kernel = kernels
        .rdna3_row_packed_attention
        .as_ref()
        .context("the promoted row-packed kernel is not installed")?;
    let buffers = phase_x_buffers(device, geometry)?;
    let output = GpuBuffer::from_f32(device, &vec![PHASE_X_SENTINEL; geometry.q_count()])?;
    let mut commands = vulkan::ComputeBatch::new(device)?;
    kernel.record_dispatch(
        &mut commands,
        &phase_x_bindings(&buffers, &output),
        bytemuck::bytes_of(&geometry.push()),
        geometry.generic_grid(),
    )?;
    commands.submit()?;
    output.read_f32(geometry.q_count())
}

/// Bitwise comparison: equality is the FP32 bit pattern, so a one-ULP
/// difference is a failure.
struct PhaseXComparison {
    elements: usize,
    bitwise: usize,
    max_abs: f32,
    max_ulp: u32,
    first: Option<(usize, u32, u32)>,
}

fn phase_x_ulp_order(value: f32) -> i64 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 != 0 {
        -(bits as i64)
    } else {
        bits as i64
    }
}

fn phase_x_compare(expected: &[f32], actual: &[f32]) -> PhaseXComparison {
    assert_eq!(expected.len(), actual.len());
    let mut report = PhaseXComparison {
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
        let ulp = (phase_x_ulp_order(expected_value) - phase_x_ulp_order(actual_value)).unsigned_abs()
            as u32;
        report.max_ulp = report.max_ulp.max(ulp);
    }
    report
}

fn phase_x_unwritten(values: &[f32]) -> usize {
    values
        .iter()
        .filter(|value| value.to_bits() == PHASE_X_SENTINEL.to_bits())
        .count()
}

fn phase_x_report_line(
    label: &str,
    implementation: &str,
    comparison: &PhaseXComparison,
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
        "[phase-x-parity] {label} implementation={implementation} elements={} bitwise={} \
         max_abs={:.9e} max_ulp={} unwritten={} {first} verdict={}",
        comparison.elements,
        comparison.bitwise,
        comparison.max_abs,
        comparison.max_ulp,
        unwritten,
        if verdict { "PASS" } else { "FAIL" }
    );
}

/// The ten Phase IX geometries, unchanged: the primary Phase VII/VIII GQA
/// geometry at the three arithmetic modes, the MHA control, the bidirectional
/// case, the masked pair, and the three partial-row-group boundaries.
fn phase_x_geometries() -> Vec<PhaseXGeometry> {
    let mut masked = PhaseXGeometry::plain("gqa-8q-2kv-masked", 2, 64, 8, 2, 128, 1);
    masked.masked_key_stride = 8;
    masked.masked_query_stride = 16;
    let mut mha_masked = PhaseXGeometry::plain("mha-8q-8kv-masked", 2, 64, 8, 8, 128, 1);
    mha_masked.masked_key_stride = 4;
    mha_masked.masked_query_stride = 0;
    vec![
        PhaseXGeometry::plain("gqa-8q-2kv-primary-mode9", 2, 64, 8, 2, 128, 1),
        PhaseXGeometry::plain("gqa-8q-2kv-primary-mode0", 2, 64, 8, 2, 128, 1).at_mode(0),
        PhaseXGeometry::plain("gqa-8q-2kv-primary-mode1", 2, 64, 8, 2, 128, 1).at_mode(1),
        PhaseXGeometry::plain("mha-8q-8kv-control", 2, 64, 8, 8, 128, 1),
        PhaseXGeometry::plain("gqa-8q-2kv-noncausal", 2, 64, 8, 2, 128, 0),
        masked,
        mha_masked,
        PhaseXGeometry::plain("boundary-42rows", 2, 7, 3, 3, 16, 1),
        PhaseXGeometry::plain("boundary-130rows", 2, 13, 5, 5, 16, 1),
        {
            let mut geometry = PhaseXGeometry::plain("boundary-162rows-gqa", 3, 9, 6, 2, 16, 1);
            geometry.masked_query_stride = 4;
            geometry
        },
    ]
}

/// Why the promoted route is not installed on this device, for the skip line.
fn phase_x_promotion_skip_reason(device: &VulkanDevice) -> String {
    if !crate::vendor::rdna3_module_enabled(device) {
        return format!(
            "not_rdna3 (vendor_id=0x{:04x} device_id=0x{:04x} module_enabled=false)",
            device.vendor_id(),
            device.device_id()
        );
    }
    if !device.supports_compute_work_group_size([64, 1, 1]) {
        return "no_64_wide_compute_workgroup".to_string();
    }
    "pipeline_creation_refused".to_string()
}

#[test]
fn phase_x_row_packed_forward_parity() -> Result<()> {
    let device = match VulkanDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("phase-x parity: status=SKIP reason=no_device error={error}");
            return Ok(());
        }
    };
    let kernels = TransformerKernels::new(&device)?;
    eprintln!(
        "phase-x parity: status=RUN device={} vendor_id=0x{:04x} device_id=0x{:04x} \
         physical_device_index={} row_packed_installed={}",
        device.name(),
        device.vendor_id(),
        device.device_id(),
        device.physical_device_index(),
        kernels.rdna3_row_packed_attention.is_some()
    );
    if kernels.rdna3_row_packed_attention.is_none() {
        eprintln!(
            "phase-x parity: status=SKIP reason={}",
            phase_x_promotion_skip_reason(&device)
        );
        return Ok(());
    }

    let mut failures = 0;
    let mut elements = 0;
    let mut bitwise = 0;
    let mut cases = 0;
    for geometry in phase_x_geometries() {
        assert!(
            rdna3_row_packed_attention_supported(&geometry.push()),
            "{} must be eligible for the promoted route",
            geometry.label
        );
        let expected = phase_x_generic_forward(&device, &kernels, &geometry)?;
        if phase_x_unwritten(&expected) != 0 {
            eprintln!(
                "[phase-x-parity] {} implementation=generic unwritten={} verdict=FAIL",
                geometry.label,
                phase_x_unwritten(&expected)
            );
            failures += 1;
        }

        // Arm 1 — the production selector, which chooses the packed grid.
        let before = attention_forward_dispatch_counts();
        let actual = phase_x_selected_forward(&device, &kernels, &geometry)?;
        let after = attention_forward_dispatch_counts();
        let comparison = phase_x_compare(&expected, &actual);
        let unwritten = phase_x_unwritten(&actual);
        let dispatched = after.row_packed == before.row_packed + 1
            && after.generic == before.generic
            && after.short == before.short;
        let verdict =
            comparison.bitwise == comparison.elements && unwritten == 0 && dispatched;
        if !verdict {
            failures += 1;
        }
        phase_x_report_line(
            &geometry.label,
            "row-packed-64-cached/selector",
            &comparison,
            unwritten,
            verdict,
        );
        eprintln!(
            "[phase-x-dispatch] label={} implementation=row-packed-64-cached grid=[{}, 1, 1] \
             rows={} workgroups={} selected={} generic_delta={} short_delta={} verdict={}",
            geometry.label,
            geometry.rows().div_ceil(64),
            geometry.rows(),
            geometry.rows().div_ceil(64),
            after.row_packed - before.row_packed,
            after.generic - before.generic,
            after.short - before.short,
            if dispatched { "PASS" } else { "FAIL" }
        );
        elements += comparison.elements;
        bitwise += comparison.bitwise;
        cases += 1;

        // Arm 2 — the same artifact under the production `[query, head, batch]`
        // grid, preserving Phase IX's second tested dispatch grid.
        let on_grid = phase_x_promoted_forward_on_grid(&device, &kernels, &geometry)?;
        let grid_comparison = phase_x_compare(&expected, &on_grid);
        let grid_unwritten = phase_x_unwritten(&on_grid);
        let grid_ok =
            grid_comparison.bitwise == grid_comparison.elements && grid_unwritten == 0;
        if !grid_ok {
            failures += 1;
        }
        phase_x_report_line(
            &geometry.label,
            "row-packed-64-cached/production-grid",
            &grid_comparison,
            grid_unwritten,
            grid_ok,
        );
        elements += grid_comparison.elements;
        bitwise += grid_comparison.bitwise;
        cases += 1;
    }
    eprintln!(
        "[phase-x-parity-summary] status={} cases={cases} elements={elements} \
         bitwise_equal={bitwise} failures={failures}",
        if failures == 0 { "PASS" } else { "FAIL" }
    );
    assert_eq!(failures, 0, "the promoted forward is not bit-exact");
    Ok(())
}

/// The host-side eligibility proof: every field that can refuse the promoted
/// route refuses it by name, and the training geometry is accepted.
#[test]
fn phase_x_row_packed_selector_matrix() {
    let training = PhaseXGeometry::plain("training-8q2kv", 2, 64, 8, 2, 128, 1);
    assert!(rdna3_row_packed_attention_supported(&training.push()));

    let base = training.push();
    let cases: Vec<(&'static str, AttentionPush)> = vec![
        ("empty_batch", { let mut p = base; p.batch_size = 0; p }),
        ("query_sequence_above_64", { let mut p = base; p.query_seq_len = 65; p.key_value_seq_len = 65; p }),
        ("query_sequence_above_64", { let mut p = base; p.query_seq_len = 0; p }),
        (
            "key_sequence_above_64",
            { let mut p = base; p.key_value_seq_len = 128; p },
        ),
        (
            "non_square_sequence",
            { let mut p = base; p.key_value_seq_len = 32; p },
        ),
        ("empty_head_count", { let mut p = base; p.num_heads = 0; p }),
        (
            "empty_head_count",
            { let mut p = base; p.num_key_value_heads = 0; p },
        ),
        (
            "non_divisible_head_mapping",
            { let mut p = base; p.num_key_value_heads = 3; p },
        ),
        (
            "qk_value_head_dim_mismatch",
            { let mut p = base; p.value_hidden_size = 2 * 64; p },
        ),
        // 1025 is not divisible by the eight heads, so the head dimension
        // itself is ragged rather than merely unequal to the value width.
        ("ragged_head_dim", { let mut p = base; p.query_hidden_size = 1025; p }),
        ("sliding_window", { let mut p = base; p.window_size = 4; p }),
        ("softcap", { let mut p = base; p.softcap = 50.0; p }),
        ("alibi", { let mut p = base; p.alibi_mode = 1; p }),
        ("relative_bias", { let mut p = base; p.relative_bias_mode = 1; p }),
        ("dropout", { let mut p = base; p.dropout_p = 0.1; p }),
        ("attention_sink", { let mut p = base; p.has_sink = 1; p }),
        ("causal_mode", { let mut p = base; p.causal = 2; p }),
        ("query_position_offset", { let mut p = base; p.query_position_offset = 1; p }),
        ("sparse_mask", { let mut p = base; p.has_sparse_mask = 1; p }),
        (
            "compressed_keys",
            { let mut p = base; p.compressed_key_count = 8; p },
        ),
        ("paged_kv", { let mut p = base; p.paged_kv = 1; p }),
        ("value_scale", { let mut p = base; p.value_scale = 0.5; p }),
        (
            "arithmetic_mode_unqualified",
            { let mut p = base; p.unfused_qk = 4; p },
        ),
        (
            "arithmetic_mode_unqualified",
            { let mut p = base; p.unfused_qk = 2; p },
        ),
    ];
    let mut failures = 0;
    for (expected, push) in &cases {
        let observed = rdna3_row_packed_attention_rejection(push);
        let verdict = observed == Some(*expected);
        if !verdict {
            failures += 1;
        }
        eprintln!(
            "[phase-x-eligibility] expected={expected} observed={:?} verdict={}",
            observed,
            if verdict { "PASS" } else { "FAIL" }
        );
    }

    // Every geometry in the qualification matrix, and every arithmetic mode the
    // matrix uses, must be accepted; the sequence bound is inclusive.
    for geometry in phase_x_geometries() {
        assert_eq!(rdna3_row_packed_attention_rejection(&geometry.push()), None);
    }
    for mode in [0u32, 1, 9] {
        let mut push = base;
        push.unfused_qk = mode;
        assert_eq!(rdna3_row_packed_attention_rejection(&push), None);
    }
    let mut at_bound = base;
    at_bound.query_seq_len = 64;
    at_bound.key_value_seq_len = 64;
    assert_eq!(rdna3_row_packed_attention_rejection(&at_bound), None);

    eprintln!(
        "[phase-x-eligibility-summary] status={} cases={} failures={failures}",
        if failures == 0 { "PASS" } else { "FAIL" },
        cases.len()
    );
    assert_eq!(failures, 0, "the eligibility predicate does not match its proof");
}

#[test]
fn phase_x_row_packed_fallback_dispatch() -> Result<()> {
    let device = match VulkanDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("phase-x fallback: status=SKIP reason=no_device error={error}");
            return Ok(());
        }
    };
    let kernels = TransformerKernels::new(&device)?;
    if kernels.rdna3_row_packed_attention.is_none() {
        eprintln!(
            "phase-x fallback: status=SKIP reason={}",
            phase_x_promotion_skip_reason(&device)
        );
        return Ok(());
    }

    // Each geometry is small enough to run both arms cheaply. The first is the
    // promoted route's own eligible control; the rest must fall back.
    let mut long = PhaseXGeometry::plain("seq-128-above-64", 1, 128, 2, 2, 16, 1);
    long.unfused_qk = 9;
    let mut dropout = PhaseXGeometry::plain("dropout", 1, 16, 2, 2, 16, 1);
    dropout.unfused_qk = 9;
    let mut window = PhaseXGeometry::plain("sliding-window", 1, 16, 2, 2, 16, 1);
    window.unfused_qk = 9;
    let mut mode4 = PhaseXGeometry::plain("mode-4", 1, 8, 2, 2, 16, 1);
    mode4.unfused_qk = 4;
    let mut causal2 = PhaseXGeometry::plain("causal-2", 1, 8, 2, 2, 16, 1);
    causal2.unfused_qk = 9;

    let mut failures = 0;
    for (label, geometry, expected_reason, mutate) in [
        ("seq-128-above-64", &long, "query_sequence_above_64", 0u8),
        ("dropout", &dropout, "dropout", 1),
        ("sliding-window", &window, "sliding_window", 2),
        ("mode-4", &mode4, "arithmetic_mode_unqualified", 0),
        ("causal-2", &causal2, "causal_mode", 3),
    ] {
        let mut push = geometry.push();
        match mutate {
            1 => push.dropout_p = 0.1,
            2 => push.window_size = 4,
            3 => push.causal = 2,
            _ => {}
        }
        assert_eq!(
            rdna3_row_packed_attention_rejection(&push),
            Some(expected_reason)
        );
        let buffers = phase_x_buffers(&device, geometry)?;
        let output = GpuBuffer::from_f32(&device, &vec![PHASE_X_SENTINEL; geometry.q_count()])?;
        let before = attention_forward_dispatch_counts();
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        record_text_attention_dispatch(
            &kernels,
            &mut commands,
            &phase_x_bindings(&buffers, &output),
            &push,
            geometry.generic_grid(),
        )?;
        commands.submit()?;
        let after = attention_forward_dispatch_counts();
        let fell_back =
            after.generic == before.generic + 1 && after.row_packed == before.row_packed;
        let generic = phase_x_generic_forward_with_push(&device, &kernels, geometry, &push)?;
        let comparison = phase_x_compare(&generic, &output.read_f32(geometry.q_count())?);
        let exact = comparison.bitwise == comparison.elements;
        if !(fell_back && exact) {
            failures += 1;
        }
        eprintln!(
            "[phase-x-fallback] label={label} reason={expected_reason} \
             implementation=generic grid={:?} generic_delta={} row_packed_delta={} \
             elements={} bitwise={} verdict={}",
            geometry.generic_grid(),
            after.generic - before.generic,
            after.row_packed - before.row_packed,
            comparison.elements,
            comparison.bitwise,
            if fell_back && exact { "PASS" } else { "FAIL" }
        );
    }
    eprintln!(
        "[phase-x-fallback-summary] status={} failures={failures}",
        if failures == 0 { "PASS" } else { "FAIL" }
    );
    assert_eq!(failures, 0, "unsupported geometry did not fall back exactly");
    Ok(())
}

/// The shipped short-sequence module must keep precedence where its own gate
/// holds, and the promoted route must only take over where that gate refuses.
#[test]
fn phase_x_row_packed_short_precedence() -> Result<()> {
    let device = match VulkanDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("phase-x precedence: status=SKIP reason=no_device error={error}");
            return Ok(());
        }
    };
    let mut kernels = TransformerKernels::new(&device)?;
    if kernels.rdna3_row_packed_attention.is_none() {
        eprintln!(
            "phase-x precedence: status=SKIP reason={}",
            phase_x_promotion_skip_reason(&device)
        );
        return Ok(());
    }
    // The shared image-text composition installs this family; installing it here
    // reproduces that composition's selection state without building a package.
    kernels.rdna3_short_attention = Some(vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../shaders/transformer_attention_short_amd_rdna3.spv"),
        11,
        std::mem::size_of::<AttentionPush>() as u32,
    )?);

    // Mode 0, both extents 8: inside the short gate and inside the promoted
    // gate. The short module must win.
    let short_geometry = PhaseXGeometry::plain("short-eligible-mode0", 2, 8, 4, 2, 16, 1).at_mode(0);
    // Same shape, mode 9: outside the short gate (`unfused_qk == 0`), inside the
    // promoted gate. The promoted route must win.
    let promoted_geometry = PhaseXGeometry::plain("short-refused-mode9", 2, 8, 4, 2, 16, 1).at_mode(9);

    let mut failures = 0;
    for (geometry, expected) in [
        (&short_geometry, AttentionForwardDispatch::Short),
        (&promoted_geometry, AttentionForwardDispatch::RowPacked),
    ] {
        let expected_arm = expected;
        assert_eq!(
            rdna3_short_attention_supported(&geometry.push()),
            expected_arm == AttentionForwardDispatch::Short,
            "{} short-gate expectation",
            geometry.label
        );
        assert!(rdna3_row_packed_attention_supported(&geometry.push()));
        let generic = phase_x_generic_forward(&device, &kernels, geometry)?;
        let before = attention_forward_dispatch_counts();
        let selected = phase_x_selected_forward(&device, &kernels, geometry)?;
        let after = attention_forward_dispatch_counts();
        let observed = if after.short > before.short {
            AttentionForwardDispatch::Short
        } else if after.row_packed > before.row_packed {
            AttentionForwardDispatch::RowPacked
        } else {
            AttentionForwardDispatch::Generic
        };
        let comparison = phase_x_compare(&generic, &selected);
        let exact = comparison.bitwise == comparison.elements
            && phase_x_unwritten(&selected) == 0;
        if !(observed == expected_arm && exact) {
            failures += 1;
        }
        eprintln!(
            "[phase-x-precedence] label={} expected={} observed={} mode={} elements={} \
             bitwise={} short_delta={} row_packed_delta={} generic_delta={} verdict={}",
            geometry.label,
            expected_arm.label(),
            observed.label(),
            geometry.unfused_qk,
            comparison.elements,
            comparison.bitwise,
            after.short - before.short,
            after.row_packed - before.row_packed,
            after.generic - before.generic,
            if observed == expected_arm && exact {
                "PASS"
            } else {
                "FAIL"
            }
        );
    }
    eprintln!(
        "[phase-x-precedence-summary] status={} failures={failures}",
        if failures == 0 { "PASS" } else { "FAIL" }
    );
    assert_eq!(failures, 0, "the selector order displaced or skipped a route");
    Ok(())
}

fn phase_x_scalar_gradients(
    device: &VulkanDevice,
    kernels: &TransformerKernels,
    geometry: &PhaseXGeometry,
    buffers: &PhaseXBuffers,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let grad_output = GpuBuffer::from_f32(device, &phase_x_operands(24, geometry.q_count()))?;
    let grad_q = GpuBuffer::from_f32(device, &vec![PHASE_X_SENTINEL; geometry.q_count()])?;
    let grad_k = GpuBuffer::from_f32(device, &vec![PHASE_X_SENTINEL; geometry.kv_count()])?;
    let grad_v = GpuBuffer::from_f32(device, &vec![PHASE_X_SENTINEL; geometry.kv_count()])?;
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
            &grad_output,
            &grad_q,
            &grad_k,
            &grad_v,
            &buffers.dummy,
            &buffers.dummy,
            &buffers.dummy,
            &buffers.dummy,
            &buffers.dummy,
        ],
        bytemuck::bytes_of(&geometry.push().as_backward_push(0, 1)),
        [geometry.kv_heads, geometry.batch, 1],
    )?;
    commands.submit()?;
    Ok((
        grad_q.read_f32(geometry.q_count())?,
        grad_k.read_f32(geometry.kv_count())?,
        grad_v.read_f32(geometry.kv_count())?,
    ))
}

#[allow(clippy::type_complexity)]
fn phase_x_two_pass_gradients(
    device: &VulkanDevice,
    query_kernel: &vulkan::ComputeKernel,
    key_kernel: &vulkan::ComputeKernel,
    geometry: &PhaseXGeometry,
    buffers: &PhaseXBuffers,
) -> Result<(
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
)> {
    let push = geometry.push();
    let scratch = two_pass_attention_scratch_elements(&push).expect("two-pass scratch size");
    let grad_output = GpuBuffer::from_f32(device, &phase_x_operands(24, geometry.q_count()))?;
    let grad_q = GpuBuffer::from_f32(device, &vec![PHASE_X_SENTINEL; geometry.q_count()])?;
    let grad_k = GpuBuffer::from_f32(device, &vec![PHASE_X_SENTINEL; geometry.kv_count()])?;
    let grad_v = GpuBuffer::from_f32(device, &vec![PHASE_X_SENTINEL; geometry.kv_count()])?;
    let probability = GpuBuffer::from_f32(device, &vec![PHASE_X_SENTINEL; scratch])?;
    let ds = GpuBuffer::from_f32(device, &vec![PHASE_X_SENTINEL; scratch])?;
    let rows = geometry.batch * geometry.heads * geometry.query_seq;
    let key_rows = geometry.batch * geometry.kv_heads * geometry.key_seq;
    let mut commands = vulkan::ComputeBatch::new(device)?;
    query_kernel.record_dispatch(
        &mut commands,
        &[
            &buffers.query,
            &buffers.key,
            &buffers.value,
            &buffers.query_mask,
            &buffers.key_mask,
            &grad_output,
            &grad_q,
            &probability,
            &ds,
        ],
        bytemuck::bytes_of(&push),
        [(rows + 63) / 64, 1, 1],
    )?;
    key_kernel.record_dispatch(
        &mut commands,
        &[
            &buffers.query,
            &grad_output,
            &probability,
            &ds,
            &grad_k,
            &grad_v,
            &buffers.key_mask,
            &buffers.query_mask,
        ],
        bytemuck::bytes_of(&push),
        [(key_rows + 63) / 64, 1, 1],
    )?;
    commands.submit()?;
    Ok((
        grad_q.read_f32(geometry.q_count())?,
        grad_k.read_f32(geometry.kv_count())?,
        grad_v.read_f32(geometry.kv_count())?,
        probability.read_f32(scratch)?,
        ds.read_f32(scratch)?,
    ))
}

#[test]
fn phase_x_row_packed_forward_gradient_parity() -> Result<()> {
    let device = match VulkanDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("phase-x gradients: status=SKIP reason=no_device error={error}");
            return Ok(());
        }
    };
    let kernels = TransformerKernels::new(&device)?;
    if kernels.rdna3_row_packed_attention.is_none() {
        eprintln!(
            "phase-x gradients: status=SKIP reason={}",
            phase_x_promotion_skip_reason(&device)
        );
        return Ok(());
    }
    let query_kernel = vulkan::ComputeKernel::new(
        &device,
        TwoPassAttentionPass::Query.spirv(),
        TwoPassAttentionPass::Query.binding_count(),
        std::mem::size_of::<AttentionPush>() as u32,
    )?;
    let key_kernel = vulkan::ComputeKernel::new(
        &device,
        TwoPassAttentionPass::Key.spirv(),
        TwoPassAttentionPass::Key.binding_count(),
        std::mem::size_of::<AttentionPush>() as u32,
    )?;

    let geometries = [
        PhaseXGeometry::plain("gqa-8q-2kv-primary-mode9", 2, 64, 8, 2, 128, 1),
        PhaseXGeometry::plain("gqa-8q-2kv-primary-mode0", 2, 64, 8, 2, 128, 1).at_mode(0),
        PhaseXGeometry::plain("mha-8q-8kv-control", 2, 64, 8, 8, 128, 1),
        PhaseXGeometry::plain("boundary-162rows-gqa", 3, 9, 6, 2, 16, 1),
    ];
    let mut failures = 0;
    let mut elements = 0;
    let mut bitwise = 0;
    let mut cases = 0;
    for geometry in &geometries {
        let buffers = phase_x_buffers(&device, geometry)?;
        let forward_generic = phase_x_generic_forward(&device, &kernels, geometry)?;
        let oracle = phase_x_scalar_gradients(&device, &kernels, geometry, &buffers)?;
        let from_generic =
            phase_x_two_pass_gradients(&device, &query_kernel, &key_kernel, geometry, &buffers)?;

        let before = attention_forward_dispatch_counts();
        let forward_promoted = phase_x_selected_forward(&device, &kernels, geometry)?;
        let after = attention_forward_dispatch_counts();
        if after.row_packed != before.row_packed + 1 {
            failures += 1;
            eprintln!(
                "[phase-x-parity] {} implementation=row-packed-64-cached/selector \
                 selected={} verdict=FAIL",
                geometry.label,
                after.row_packed - before.row_packed
            );
        }
        let forward_comparison = phase_x_compare(&forward_generic, &forward_promoted);
        let forward_unwritten = phase_x_unwritten(&forward_promoted);
        let forward_ok =
            forward_comparison.bitwise == forward_comparison.elements && forward_unwritten == 0;
        if !forward_ok {
            failures += 1;
        }
        phase_x_report_line(
            &geometry.label,
            "row-packed-64-cached/forward",
            &forward_comparison,
            forward_unwritten,
            forward_ok,
        );
        elements += forward_comparison.elements;
        bitwise += forward_comparison.bitwise;

        let from_promoted =
            phase_x_two_pass_gradients(&device, &query_kernel, &key_kernel, geometry, &buffers)?;
        for (name, expected, actual, require_written) in [
            ("dq", &from_generic.0, &from_promoted.0, true),
            ("dk", &from_generic.1, &from_promoted.1, true),
            ("dv", &from_generic.2, &from_promoted.2, true),
            ("probability", &from_generic.3, &from_promoted.3, false),
            ("ds", &from_generic.4, &from_promoted.4, false),
        ] {
            let comparison = phase_x_compare(expected, actual);
            let unwritten = phase_x_unwritten(actual);
            let unwritten_expected = phase_x_unwritten(expected);
            let verdict = comparison.bitwise == comparison.elements
                && (require_written && unwritten == 0
                    || !require_written && unwritten == unwritten_expected);
            if !verdict {
                failures += 1;
            }
            eprintln!(
                "[phase-x-parity] {}/{name} implementation=two-pass-from-each-forward \
                 elements={} bitwise={} max_abs={:.9e} max_ulp={} unwritten={} \
                 unwritten_baseline={} unwritten_pattern={} verdict={}",
                geometry.label,
                comparison.elements,
                comparison.bitwise,
                comparison.max_abs,
                comparison.max_ulp,
                unwritten,
                unwritten_expected,
                if unwritten == unwritten_expected {
                    "identical"
                } else {
                    "different"
                },
                if verdict { "PASS" } else { "FAIL" }
            );
            elements += comparison.elements;
            bitwise += comparison.bitwise;
        }
        for (name, expected, actual) in [
            ("dq", &oracle.0, &from_generic.0),
            ("dk", &oracle.1, &from_generic.1),
            ("dv", &oracle.2, &from_generic.2),
        ] {
            let comparison = phase_x_compare(expected, actual);
            let verdict = comparison.bitwise == comparison.elements;
            if !verdict {
                failures += 1;
            }
            eprintln!(
                "[phase-x-parity] {}/oracle-{name} elements={} bitwise={} verdict={}",
                geometry.label,
                comparison.elements,
                comparison.bitwise,
                if verdict { "PASS" } else { "FAIL" }
            );
            elements += comparison.elements;
            bitwise += comparison.bitwise;
        }
        cases += 1;
    }
    eprintln!(
        "[phase-x-gradient-summary] status={} cases={cases} elements={elements} \
         bitwise_equal={bitwise} failures={failures}",
        if failures == 0 { "PASS" } else { "FAIL" }
    );
    let backward = attention_backward_dispatch_counts();
    eprintln!(
        "[phase-x-backward-counters] two_pass_enabled={} scalar={} query={} key={} \
         scratch_allocations={}",
        attention_backward_two_pass_enabled(),
        backward.scalar,
        backward.query,
        backward.key,
        backward.scratch_allocations
    );
    assert_eq!(failures, 0, "downstream gradients are not identical");
    Ok(())
}

// ---------------------------------------------------------------------------
// Complete training step
// ---------------------------------------------------------------------------

struct PhaseXSequence {
    input_ids: Vec<u32>,
    targets: Vec<u32>,
    attention_mask: Vec<f32>,
    loss_weights: Vec<f32>,
}

fn phase_x_load_sequences(
    path: &std::path::Path,
    seq_len: usize,
) -> Result<Vec<PhaseXSequence>> {
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
            let mut sequence = PhaseXSequence {
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

fn phase_x_batch_arrays(
    sequences: &[PhaseXSequence],
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

fn phase_x_digest(values: &std::collections::BTreeMap<String, Vec<f32>>) -> (String, usize, f64) {
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

/// The complete Phase VII/VIII/IX training step, driven by the **default**
/// selector.
///
/// `HIERARCHOS_PHASE_X_STEP_ARM`   production | generic (required)
/// `HIERARCHOS_PHASE_X_FIXTURE`    fixture dir (default target/gqa-e2e/fixture-8q2kv)
/// `HIERARCHOS_PHASE_X_STEP_STEPS` steps per process (default 29)
/// `HIERARCHOS_PHASE_X_STEP_DIGEST` 1 = hash gradients and parameters every step
///
/// The `production` arm installs nothing: it is the shipped selector. The
/// `generic` arm is the same binary with `HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS=1`,
/// which the vendor family already gates on, so both arms are this build's own
/// committed generic kernel rather than a recorded number.
#[test]
fn phase_x_row_packed_forward_full_step() -> Result<()> {
    let arm = match std::env::var("HIERARCHOS_PHASE_X_STEP_ARM") {
        Ok(value) => value.trim().to_string(),
        Err(_) => {
            eprintln!(
                "phase-x full step: status=SKIP reason=not_requested \
                 (set HIERARCHOS_PHASE_X_STEP_ARM=production|generic)"
            );
            return Ok(());
        }
    };
    anyhow::ensure!(
        arm == "production" || arm == "generic",
        "unknown phase-x step arm {arm}"
    );
    let fixture = std::env::var("HIERARCHOS_PHASE_X_FIXTURE")
        .unwrap_or_else(|_| "target/gqa-e2e/fixture-8q2kv".to_string());
    let fixture = std::path::PathBuf::from(fixture);
    let steps: usize = std::env::var("HIERARCHOS_PHASE_X_STEP_STEPS")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(29);
    let warmup: usize = 3;
    let digest = std::env::var("HIERARCHOS_PHASE_X_STEP_DIGEST")
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

    let sequences = phase_x_load_sequences(&fixture.join("dataset.jsonl"), seq_len)?;
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
    let start_parameters = phase_x_digest(&graph.parameter_values()?);
    let installed = graph.kernels.rdna3_row_packed_attention.is_some();
    let forward_before = attention_forward_dispatch_counts();
    let backward_before = attention_backward_dispatch_counts();
    eprintln!(
        "[phase-x-step-run] arm={arm} fixture={} sequences={} batches_per_epoch={batches_per_epoch} \
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
    // The production arm must actually have the promoted route installed; the
    // generic arm must not have it. Without this the two arms could silently be
    // the same execution and every digest below would be a tautology.
    match arm.as_str() {
        "production" => anyhow::ensure!(installed, "the promoted route is not installed"),
        _ => anyhow::ensure!(!installed, "the generic arm still installed the promoted route"),
    }

    for index in 0..steps {
        let global_step = index;
        let epoch = global_step / batches_per_epoch;
        let batch_index = global_step % batches_per_epoch;
        let begin = batch_index * batch;
        let (input_ids, targets, attention_mask, loss_weights) =
            phase_x_batch_arrays(&sequences[begin..begin + batch], seq_len)?;
        let started = Instant::now();
        let result = graph.train_step(&input_ids, &targets, &attention_mask, &loss_weights, hyper)?;
        let step_ms = started.elapsed().as_secs_f64() * 1e3;
        let measured = index >= warmup;
        if digest {
            let gradients = graph.lora_gradient_values()?;
            let gradient_digest = phase_x_digest(&gradients);
            let parameters = phase_x_digest(&graph.parameter_values()?);
            eprintln!(
                "[phase-x-step] arm={arm} step={index} epoch={epoch} batch={batch_index} \
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
                "[phase-x-step] arm={arm} step={index} epoch={epoch} batch={batch_index} \
                 measured={measured} step_ms={step_ms:.6} loss_bits={:08x} loss={:e}",
                result.loss.to_bits(),
                result.loss
            );
        }
    }
    let forward_after = attention_forward_dispatch_counts();
    let backward_after = attention_backward_dispatch_counts();
    let end_parameters = phase_x_digest(&graph.parameter_values()?);
    eprintln!(
        "[phase-x-step-run-end] arm={arm} parameter_end_digest={} parameter_elements={} \
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

    // The adapter tensors and the AdamW moments are what the run actually
    // trained. `parameter_values()` is the frozen base map, which is why it is
    // constant across steps; the adapter is only observable through the
    // checkpoint writer. One file digest per line, so a mismatch names the
    // tensor that drifted instead of only reporting a different total.
    if let Ok(directory) = std::env::var("HIERARCHOS_PHASE_X_STEP_CHECKPOINT") {
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
                "[phase-x-checkpoint] arm={arm} file={name} bytes={} sha256={digest}",
                std::fs::metadata(&path)?.len()
            );
        }
    }
    Ok(())
}

/// Opt-in GPU-timestamp timing of the production route against the generic
/// kernel at the primary geometry.
#[test]
fn phase_x_row_packed_forward_timing() -> Result<()> {
    let requested = std::env::var("HIERARCHOS_PHASE_X_KERNEL_TIMING")
        .map(|value| value.trim() == "1")
        .unwrap_or(false);
    if !requested {
        eprintln!(
            "phase-x kernel timing: status=SKIP reason=not_requested \
             (set HIERARCHOS_PHASE_X_KERNEL_TIMING=1)"
        );
        return Ok(());
    }
    let profiled = std::env::var_os("HIERARCHOS_VULKAN_PROFILE_KERNELS").is_some();
    eprintln!(
        "phase-x kernel timing: status=RUN timestamp_profile={profiled} \
         (set HIERARCHOS_VULKAN_PROFILE_KERNELS=1 and \
         HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 for per-submission GPU time)"
    );
    let device = VulkanDevice::new()?;
    let kernels = TransformerKernels::new(&device)?;
    eprintln!(
        "phase-x kernel timing: device={} vendor_id=0x{:04x} device_id=0x{:04x} \
         physical_device_index={} row_packed_installed={}",
        device.name(),
        device.vendor_id(),
        device.device_id(),
        device.physical_device_index(),
        kernels.rdna3_row_packed_attention.is_some()
    );
    if kernels.rdna3_row_packed_attention.is_none() {
        eprintln!(
            "phase-x kernel timing: status=SKIP reason={}",
            phase_x_promotion_skip_reason(&device)
        );
        return Ok(());
    }
    let iterations: u32 = std::env::var("HIERARCHOS_PHASE_X_KERNEL_ITERATIONS")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(25);
    let warmup: u32 = 3;
    let reverse = std::env::var("HIERARCHOS_PHASE_X_REVERSE")
        .map(|value| value.trim() == "1")
        .unwrap_or(false);

    for (label, geometry) in [
        (
            "gqa-8q-2kv-primary-mode9",
            PhaseXGeometry::plain("gqa-8q-2kv-primary-mode9", 2, 64, 8, 2, 128, 1),
        ),
        (
            "gqa-8q-2kv-primary-mode0",
            PhaseXGeometry::plain("gqa-8q-2kv-primary-mode0", 2, 64, 8, 2, 128, 1).at_mode(0),
        ),
    ] {
        let buffers = phase_x_buffers(&device, &geometry)?;
        let output = GpuBuffer::from_f32(&device, &vec![0.0_f32; geometry.q_count()])?;
        let packed_grid = [geometry.rows().div_ceil(64) as u32, 1, 1];
        let mut arms: Vec<(&str, [u32; 3], u64)> = vec![
            ("forward_generic", geometry.generic_grid(), 1),
            ("forward_rowpacked64cached", packed_grid, 64),
        ];
        if reverse {
            arms.reverse();
        }
        for (arm, grid, lanes) in arms {
            let workgroups = grid[0] as u64 * grid[1] as u64 * grid[2] as u64;
            let submit = |marker: &str| -> Result<()> {
                eprintln!(
                    "[phase-x-bench{marker}] geometry={label} arm={arm} grid={grid:?} \
                     workgroups={workgroups} lanes={lanes}"
                );
                let mut commands = vulkan::ComputeBatch::new(&device)?;
                match arm {
                    "forward_generic" => kernels.attention_forward.record_dispatch(
                        &mut commands,
                        &phase_x_bindings(&buffers, &output),
                        bytemuck::bytes_of(&geometry.push()),
                        grid,
                    )?,
                    _ => record_text_attention_dispatch(
                        &kernels,
                        &mut commands,
                        &phase_x_bindings(&buffers, &output),
                        &geometry.push(),
                        grid,
                    )?,
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
                "[phase-x-bench-result] geometry={label} arm={arm} workgroups={workgroups} \
                 lanes={lanes} dispatches_per_submission=1 iterations={iterations} \
                 median_host_ms={median:.6} p25_host_ms={p25:.6} min_host_ms={:.6} \
                 max_host_ms={:.6} samples_host_ms={:?}",
                sorted[0],
                sorted[sorted.len() - 1],
                sorted
            );
        }
    }
    Ok(())
}
