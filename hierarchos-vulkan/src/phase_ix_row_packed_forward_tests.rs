//! Phase IX — RDNA3 wave64 row-packed attention forward (test builds only).
//!
//! The experiment's entire dispatch path lives here. Nothing in this module is
//! reachable from a production build: it is declared `#[cfg(test)]`, the
//! candidate SPIR-V is embedded with `include_bytes!` from `validation/`, and
//! the production selector, the production eligibility predicates, the generic
//! shader, the shipped short-attention module and every default route are
//! untouched.
//!
//! Three tests:
//!
//! * [`phase_ix_row_packed_forward_parity`] — bitwise forward-output parity of
//!   the candidate against the production generic kernel for the Phase VII
//!   GQA geometry, the MHA control, both causality settings, masked keys,
//!   masked queries and three boundary geometries whose flattened row count is
//!   not a multiple of 64.
//! * [`phase_ix_row_packed_forward_gradient_parity`] — the production two-pass
//!   attention backward (query pass then key pass, unchanged artifacts) is run
//!   from *each* forward result; dQ/dK/dV, the pass-1 probability scratch and
//!   the dS scratch must all be bitwise identical. The production scalar
//!   backward is run as the independent oracle for the same forward tensor.
//! * [`phase_ix_row_packed_forward_timing`] — opt-in GPU-timestamp timing of
//!   the two forward kernels at the primary geometry.
//!
//! Run:
//!
//! ```text
//! cargo test --lib phase_ix_row_packed_forward_parity -- --nocapture --test-threads=1
//! cargo test --lib phase_ix_row_packed_forward_gradient_parity -- --nocapture --test-threads=1
//! HIERARCHOS_PHASE_IX_KERNEL_TIMING=1 HIERARCHOS_VULKAN_PROFILE_KERNELS=1 \
//!   HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 \
//!   cargo test --lib phase_ix_row_packed_forward_timing -- --nocapture --test-threads=1
//! ```

use super::*;

/// Candidate A: row packing with the generic kernel's *streaming* shape and no
/// per-row state. Lives under `validation/`, not `shaders/`.
const ROW_PACKED_STREAM: &[u8] = include_bytes!(
    "../validation/phase_ix_row_packed_forward_20261124/shaders/transformer_attention_forward_rowpacked64_stream_rdna3.spv"
);

/// Candidate B: row packing with the generic kernel's *cached* shape, one
/// per-lane score array bounded at 64 keys — the shipped short kernel's
/// structure with its bound raised from 16.
const ROW_PACKED_CACHED: &[u8] = include_bytes!(
    "../validation/phase_ix_row_packed_forward_20261124/shaders/transformer_attention_forward_rowpacked64_cached_rdna3.spv"
);

/// One experimental candidate and the geometries it is defined for.
struct PhaseIxCandidate {
    label: &'static str,
    spirv: &'static [u8],
    lanes: u64,
    /// Longest sequence the candidate supports. The cached candidate carries a
    /// per-lane `float scores[64]`, so it cannot be dispatched above 64 keys.
    max_seq: u32,
    /// Whether the module derives its row from the linearised 3-D workgroup
    /// index (so the production `[query, head, batch]` grid also works) instead
    /// of reading `gl_GlobalInvocationID.x`.
    grid_agnostic: bool,
    /// Whether the module reproduces every `unfused_qk` arithmetic mode. The
    /// streaming candidate was written for mode 0 only and is reported
    /// out-of-mode rather than silently compared at another mode.
    all_modes: bool,
}

fn phase_ix_candidates() -> [PhaseIxCandidate; 2] {
    [
        PhaseIxCandidate {
            label: "row-packed-64-stream",
            spirv: ROW_PACKED_STREAM,
            lanes: 64,
            max_seq: u32::MAX,
            grid_agnostic: false,
            all_modes: false,
        },
        PhaseIxCandidate {
            label: "row-packed-64-cached",
            spirv: ROW_PACKED_CACHED,
            lanes: 64,
            max_seq: 64,
            grid_agnostic: true,
            all_modes: true,
        },
    ]
}

impl PhaseIxCandidate {
    fn supports(&self, geometry: &PhaseIxGeometry) -> bool {
        geometry.query_seq <= self.max_seq
            && geometry.key_seq <= self.max_seq
            && (self.all_modes || geometry.unfused_qk == 0)
    }
}

/// Value written into every output buffer before a dispatch, so a row the
/// kernel never writes is visible as a surviving sentinel.
const PHASE_IX_SENTINEL: f32 = -98_765.432_1_f32;

/// One forward geometry. `masked_key_stride` / `masked_query_stride` of zero
/// means "every position visible"; `n` means "zero the mask where
/// `(position % n) == n - 1`", which is deterministic and leaves every other
/// position visible.
struct PhaseIxGeometry {
    label: &'static str,
    batch: u32,
    query_seq: u32,
    key_seq: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    causal: u32,
    /// The generic kernel's arithmetic switch word. The production Llama path
    /// sets 9; 0 is the plain mode; 1 is the Gemma materialized mode.
    unfused_qk: u32,
    masked_key_stride: u32,
    masked_query_stride: u32,
}

impl PhaseIxGeometry {
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

    /// The production dispatch: one workgroup per `(query, head, batch)`.
    fn generic_grid(&self) -> [u32; 3] {
        [self.query_seq, self.heads, self.batch]
    }

    /// The candidate dispatch: one workgroup per 64 flattened rows.
    fn packed_grid(&self) -> [u32; 3] {
        [((self.rows() + 63) / 64) as u32, 1, 1]
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

    /// Host-side statement of the plain-path gate this candidate is defined
    /// for. Mirrors `rdna3_short_attention_supported`'s field list.
    fn plain_path(&self) -> bool {
        let push = self.push();
        push.batch_size > 0
            && push.query_seq_len > 0
            && push.key_value_seq_len > 0
            && push.num_heads > 0
            && push.num_key_value_heads > 0
            && push.num_heads % push.num_key_value_heads == 0
            && push.window_size == 0
            && push.softcap == 0.0
            && push.alibi_mode == 0
            && push.relative_bias_mode == 0
            && push.dropout_p == 0.0
            && push.has_sink == 0
            && push.causal <= 1
            && push.query_position_offset == 0
            && push.has_sparse_mask == 0
            && push.compressed_key_count == 0
            && push.paged_kv == 0
            // The candidate reproduces the generic kernel's four arithmetic
            // switches, so the plain modes 0, 1 and 9 are all in contract.
            && matches!(push.unfused_qk, 0 | 1 | 9)
            && push.value_scale == 1.0
    }
}

struct PhaseIxBuffers {
    query: GpuBuffer,
    key: GpuBuffer,
    value: GpuBuffer,
    query_mask: GpuBuffer,
    key_mask: GpuBuffer,
    dummy: GpuBuffer,
}

/// Deterministic operands: identical bytes for every arm of a case.
fn phase_ix_operands(salt: u32, count: usize) -> Vec<f32> {
    let mut state = 0x9e37_79b9_u32 ^ salt.wrapping_mul(0x85eb_ca6b);
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / 16_777_216.0;
            unit * 2.0 - 1.0
        })
        .collect()
}

fn phase_ix_masks(geometry: &PhaseIxGeometry) -> (Vec<f32>, Vec<f32>) {
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

fn phase_ix_buffers(device: &VulkanDevice, geometry: &PhaseIxGeometry) -> Result<PhaseIxBuffers> {
    let (query_mask, key_mask) = phase_ix_masks(geometry);
    Ok(PhaseIxBuffers {
        query: GpuBuffer::from_f32(device, &phase_ix_operands(21, geometry.q_count()))?,
        key: GpuBuffer::from_f32(device, &phase_ix_operands(22, geometry.kv_count()))?,
        value: GpuBuffer::from_f32(device, &phase_ix_operands(23, geometry.kv_count()))?,
        query_mask: GpuBuffer::from_f32(device, &query_mask)?,
        key_mask: GpuBuffer::from_f32(device, &key_mask)?,
        dummy: GpuBuffer::from_f32(device, &[0.0_f32])?,
    })
}

fn phase_ix_forward_bindings<'a>(
    buffers: &'a PhaseIxBuffers,
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

/// One forward dispatch into a sentinel-filled output. Returns the tensor.
fn run_forward(
    device: &VulkanDevice,
    kernel: &vulkan::ComputeKernel,
    geometry: &PhaseIxGeometry,
    buffers: &PhaseIxBuffers,
    grid: [u32; 3],
) -> Result<Vec<f32>> {
    let output = GpuBuffer::from_f32(device, &vec![PHASE_IX_SENTINEL; geometry.q_count()])?;
    let mut commands = vulkan::ComputeBatch::new(device)?;
    kernel.record_dispatch(
        &mut commands,
        &phase_ix_forward_bindings(buffers, &output),
        bytemuck::bytes_of(&geometry.push()),
        grid,
    )?;
    commands.submit()?;
    output.read_f32(geometry.q_count())
}

/// Bitwise comparison: equality is defined on the FP32 bit pattern, so a
/// one-ULP difference is a failure and `max_abs == 0` is the only pass.
struct PhaseIxComparison {
    elements: usize,
    bitwise: usize,
    max_abs: f32,
    max_ulp: u32,
    first: Option<(usize, u32, u32)>,
}

fn phase_ix_ulp_order(value: f32) -> i64 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 != 0 {
        -(bits as i64)
    } else {
        bits as i64
    }
}

fn phase_ix_compare(expected: &[f32], actual: &[f32]) -> PhaseIxComparison {
    assert_eq!(expected.len(), actual.len());
    let mut report = PhaseIxComparison {
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
        let ulp = (phase_ix_ulp_order(expected_value) - phase_ix_ulp_order(actual_value))
            .unsigned_abs() as u32;
        report.max_ulp = report.max_ulp.max(ulp);
    }
    report
}

fn phase_ix_report_line(
    label: &str,
    implementation: &str,
    comparison: &PhaseIxComparison,
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
        "[phase-ix-parity] {label} implementation={implementation} elements={} bitwise={} \
         max_abs={:.9e} max_ulp={} unwritten={} {first} verdict={}",
        comparison.elements,
        comparison.bitwise,
        comparison.max_abs,
        comparison.max_ulp,
        unwritten,
        if verdict { "PASS" } else { "FAIL" }
    );
}

fn phase_ix_unwritten(values: &[f32]) -> usize {
    values
        .iter()
        .filter(|value| value.to_bits() == PHASE_IX_SENTINEL.to_bits())
        .count()
}

fn phase_ix_dispatch_line(
    geometry: &PhaseIxGeometry,
    implementation: &str,
    grid: [u32; 3],
    lanes: u64,
) {
    let workgroups = grid[0] as u64 * grid[1] as u64 * grid[2] as u64;
    eprintln!(
        "[phase-ix-dispatch] label={} implementation={implementation} grid={grid:?} \
         workgroups={workgroups} lanes_per_workgroup={lanes} lane_slots={} \
         useful_rows={} wasted_lanes={}",
        geometry.label,
        workgroups * lanes,
        geometry.rows(),
        workgroups * lanes - geometry.rows()
    );
}

/// The primary Phase VII geometry, the MHA control and the boundary cases.
///
/// Row counts: 1024, 1024, 42, 130, 162, 128, 1024. The first two and the
/// masked pair divide 64 exactly; the three marked `boundary` do not, so the
/// final workgroup is partial (or, at 42 rows, the only workgroup is).
fn phase_ix_geometries() -> Vec<PhaseIxGeometry> {
    let mut masked = PhaseIxGeometry::plain("gqa-8q-2kv-masked", 2, 64, 8, 2, 128, 1);
    masked.masked_key_stride = 8;
    masked.masked_query_stride = 16;
    let mut mha_masked = PhaseIxGeometry::plain("mha-8q-8kv-masked", 2, 64, 8, 8, 128, 1);
    mha_masked.masked_key_stride = 4;
    mha_masked.masked_query_stride = 0;
    vec![
        // Primary: the Phase VII / Phase VIII 8q/2kv end-to-end geometry, at the
        // arithmetic mode the production Llama path actually dispatches.
        PhaseIxGeometry::plain("gqa-8q-2kv-primary-mode9", 2, 64, 8, 2, 128, 1),
        // The same geometry at mode 0 and at the Gemma materialized mode 1, so
        // every arithmetic switch the candidate reproduces is exercised.
        PhaseIxGeometry::plain("gqa-8q-2kv-primary-mode0", 2, 64, 8, 2, 128, 1).at_mode(0),
        PhaseIxGeometry::plain("gqa-8q-2kv-primary-mode1", 2, 64, 8, 2, 128, 1).at_mode(1),
        // Secondary control: identical everything except the head mapping.
        PhaseIxGeometry::plain("mha-8q-8kv-control", 2, 64, 8, 8, 128, 1),
        // Bidirectional (causal = 0) at the primary geometry.
        PhaseIxGeometry::plain("gqa-8q-2kv-noncausal", 2, 64, 8, 2, 128, 0),
        // Mask handling inside the plain path.
        masked,
        mha_masked,
        // Boundary: 42 rows, one partial workgroup, causal.
        PhaseIxGeometry::plain("boundary-42rows", 2, 7, 3, 3, 16, 1),
        // Boundary: 130 rows = 64 + 64 + 2.
        PhaseIxGeometry::plain("boundary-130rows", 2, 13, 5, 5, 16, 1),
        // Boundary: 162 rows = 64 + 64 + 34, GQA, causal, plus a masked query
        // row in the partial group whose output must be zeroed.
        {
            let mut geometry = PhaseIxGeometry::plain("boundary-162rows-gqa", 3, 9, 6, 2, 16, 1);
            geometry.masked_query_stride = 4;
            geometry
        },
    ]
}

#[test]
fn phase_ix_row_packed_forward_parity() -> Result<()> {
    let device = match VulkanDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("phase-ix parity: status=SKIP reason=no_device error={error}");
            return Ok(());
        }
    };
    eprintln!(
        "phase-ix parity: status=RUN device={} vendor_id=0x{:04x} device_id=0x{:04x} \
         physical_device_index={}",
        device.name(),
        device.vendor_id(),
        device.device_id(),
        device.physical_device_index()
    );
    let generic = vulkan::ComputeKernel::new(
        &device,
        ATTENTION_FORWARD_SPV,
        11,
        std::mem::size_of::<AttentionPush>() as u32,
    )?;
    let candidates = phase_ix_candidates()
        .into_iter()
        .map(|candidate| {
            let kernel = vulkan::ComputeKernel::new(
                &device,
                candidate.spirv,
                11,
                std::mem::size_of::<AttentionPush>() as u32,
            )?;
            Ok((candidate, kernel))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut failures = 0;
    let mut elements = 0;
    let mut bitwise = 0;
    let mut cases = 0;
    for geometry in phase_ix_geometries() {
        assert!(
            geometry.plain_path(),
            "{} is not on the candidate's plain path",
            geometry.label
        );
        let buffers = phase_ix_buffers(&device, &geometry)?;
        let generic_grid = geometry.generic_grid();
        let packed_grid = geometry.packed_grid();
        phase_ix_dispatch_line(&geometry, "generic", generic_grid, 1);
        let expected = run_forward(&device, &generic, &geometry, &buffers, generic_grid)?;
        let unwritten_generic = phase_ix_unwritten(&expected);
        if unwritten_generic != 0 {
            eprintln!(
                "[phase-ix-parity] {} implementation=generic unwritten={unwritten_generic} \
                 verdict=FAIL",
                geometry.label
            );
            failures += 1;
        }
        for (candidate, kernel) in &candidates {
            if !candidate.supports(&geometry) {
                eprintln!(
                    "[phase-ix-parity] {} implementation={} verdict=OUT_OF_SCOPE \
                     reason={}",
                    geometry.label,
                    candidate.label,
                    if geometry.query_seq > candidate.max_seq
                        || geometry.key_seq > candidate.max_seq
                    {
                        "sequence_exceeds_candidate_bound"
                    } else {
                        "arithmetic_mode_not_reproduced_by_candidate"
                    }
                );
                continue;
            }
            phase_ix_dispatch_line(&geometry, candidate.label, packed_grid, candidate.lanes);
            let actual = run_forward(&device, kernel, &geometry, &buffers, packed_grid)?;
            let comparison = phase_ix_compare(&expected, &actual);
            let unwritten_candidate = phase_ix_unwritten(&actual);
            let verdict = comparison.bitwise == comparison.elements
                && unwritten_candidate == 0
                && unwritten_generic == 0;
            if !verdict {
                failures += 1;
            }
            phase_ix_report_line(
                &geometry.label,
                candidate.label,
                &comparison,
                unwritten_candidate,
                verdict,
            );
            elements += comparison.elements;
            bitwise += comparison.bitwise;
            cases += 1;

            // A grid-agnostic module must also produce the same tensor when it
            // is launched under the production `[query, head, batch]` grid,
            // where all but the first ceil(rows/64) workgroups return early.
            // This is the grid the production dispatch site supplies, so the
            // claim "the candidate is correct inside the real training step"
            // rests on this comparison and not on the packed grid alone.
            if candidate.grid_agnostic {
                let production_grid = run_forward(
                    &device,
                    kernel,
                    &geometry,
                    &buffers,
                    geometry.generic_grid(),
                )?;
                let grid_comparison = phase_ix_compare(&expected, &production_grid);
                let grid_unwritten = phase_ix_unwritten(&production_grid);
                let grid_ok = grid_comparison.bitwise == grid_comparison.elements
                    && grid_unwritten == 0;
                if !grid_ok {
                    failures += 1;
                }
                phase_ix_report_line(
                    &geometry.label,
                    &format!("{}/production-grid", candidate.label),
                    &grid_comparison,
                    grid_unwritten,
                    grid_ok,
                );
                elements += grid_comparison.elements;
                bitwise += grid_comparison.bitwise;
                cases += 1;
            }
        }
    }
    eprintln!(
        "[phase-ix-parity-summary] status={} cases={cases} elements={elements} \
         bitwise_equal={bitwise} failures={failures}",
        if failures == 0 { "PASS" } else { "FAIL" }
    );
    assert_eq!(failures, 0, "the row-packed forward is not bit-exact");
    Ok(())
}

/// The production scalar backward, as the independent oracle for one forward
/// tensor: `(dQ, dK, dV)`.
fn phase_ix_scalar_gradients(
    device: &VulkanDevice,
    kernels: &TransformerKernels,
    geometry: &PhaseIxGeometry,
    buffers: &PhaseIxBuffers,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let grad_output =
        GpuBuffer::from_f32(device, &phase_ix_operands(24, geometry.q_count()))?;
    let grad_q = GpuBuffer::from_f32(device, &vec![PHASE_IX_SENTINEL; geometry.q_count()])?;
    let grad_k = GpuBuffer::from_f32(device, &vec![PHASE_IX_SENTINEL; geometry.kv_count()])?;
    let grad_v = GpuBuffer::from_f32(device, &vec![PHASE_IX_SENTINEL; geometry.kv_count()])?;
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

/// The production two-pass backward from a forward tensor: pass 1 (query
/// owned) then pass 2 (key owned), unchanged artifacts, one submission each.
#[allow(clippy::type_complexity)]
fn phase_ix_two_pass_gradients(
    device: &VulkanDevice,
    query_kernel: &vulkan::ComputeKernel,
    key_kernel: &vulkan::ComputeKernel,
    geometry: &PhaseIxGeometry,
    buffers: &PhaseIxBuffers,
) -> Result<(
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
)> {
    let push = geometry.push();
    let scratch = two_pass_attention_scratch_elements(&push).expect("two-pass scratch size");
    let grad_output =
        GpuBuffer::from_f32(device, &phase_ix_operands(24, geometry.q_count()))?;
    let grad_q = GpuBuffer::from_f32(device, &vec![PHASE_IX_SENTINEL; geometry.q_count()])?;
    let grad_k = GpuBuffer::from_f32(device, &vec![PHASE_IX_SENTINEL; geometry.kv_count()])?;
    let grad_v = GpuBuffer::from_f32(device, &vec![PHASE_IX_SENTINEL; geometry.kv_count()])?;
    let probability = GpuBuffer::from_f32(device, &vec![PHASE_IX_SENTINEL; scratch])?;
    let ds = GpuBuffer::from_f32(device, &vec![PHASE_IX_SENTINEL; scratch])?;
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
fn phase_ix_row_packed_forward_gradient_parity() -> Result<()> {
    let device = match VulkanDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("phase-ix gradients: status=SKIP reason=no_device error={error}");
            return Ok(());
        }
    };
    let kernels = TransformerKernels::new(&device)?;
    let generic = vulkan::ComputeKernel::new(
        &device,
        ATTENTION_FORWARD_SPV,
        11,
        std::mem::size_of::<AttentionPush>() as u32,
    )?;
    let candidates = phase_ix_candidates()
        .into_iter()
        .map(|candidate| {
            let kernel = vulkan::ComputeKernel::new(
                &device,
                candidate.spirv,
                11,
                std::mem::size_of::<AttentionPush>() as u32,
            )?;
            Ok((candidate, kernel))
        })
        .collect::<Result<Vec<_>>>()?;
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

    // The gradient sweep is the primary geometry, the MHA control and the
    // boundary geometry with the largest partial workgroup.
    let geometries = [
        PhaseIxGeometry::plain("gqa-8q-2kv-primary-mode9", 2, 64, 8, 2, 128, 1),
        PhaseIxGeometry::plain("gqa-8q-2kv-primary-mode0", 2, 64, 8, 2, 128, 1).at_mode(0),
        PhaseIxGeometry::plain("mha-8q-8kv-control", 2, 64, 8, 8, 128, 1),
        PhaseIxGeometry::plain("boundary-162rows-gqa", 3, 9, 6, 2, 16, 1),
    ];
    let mut failures = 0;
    let mut elements = 0;
    let mut bitwise = 0;
    let mut cases = 0;
    for geometry in &geometries {
        let buffers = phase_ix_buffers(&device, geometry)?;
        let forward_generic = run_forward(
            &device,
            &generic,
            geometry,
            &buffers,
            geometry.generic_grid(),
        )?;
        let oracle = phase_ix_scalar_gradients(&device, &kernels, geometry, &buffers)?;
        let from_generic =
            phase_ix_two_pass_gradients(&device, &query_kernel, &key_kernel, geometry, &buffers)?;

        for (candidate, kernel) in &candidates {
            if !candidate.supports(geometry) {
                eprintln!(
                    "[phase-ix-parity] {} implementation={} verdict=OUT_OF_SCOPE \
                     reason={}",
                    geometry.label,
                    candidate.label,
                    if geometry.query_seq > candidate.max_seq
                        || geometry.key_seq > candidate.max_seq
                    {
                        "sequence_exceeds_candidate_bound"
                    } else {
                        "arithmetic_mode_not_reproduced_by_candidate"
                    }
                );
                continue;
            }
            let forward_candidate = run_forward(
                &device,
                kernel,
                geometry,
                &buffers,
                geometry.packed_grid(),
            )?;
            let forward_comparison = phase_ix_compare(&forward_generic, &forward_candidate);
            let forward_unwritten = phase_ix_unwritten(&forward_candidate);
            let forward_ok = forward_comparison.bitwise == forward_comparison.elements
                && forward_unwritten == 0;
            if !forward_ok {
                failures += 1;
            }
            phase_ix_report_line(
                &geometry.label,
                &format!("{}/forward", candidate.label),
                &forward_comparison,
                forward_unwritten,
                forward_ok,
            );
            elements += forward_comparison.elements;
            bitwise += forward_comparison.bitwise;

            let from_candidate = phase_ix_two_pass_gradients(
                &device,
                &query_kernel,
                &key_kernel,
                geometry,
                &buffers,
            )?;

            // The two-pass pair is a function of Q/K/V/masks and grad_output
            // only: it never reads the forward output or any forward
            // intermediate. Both arms therefore feed it the same bytes, and
            // this comparison proves that by measuring it rather than
            // asserting it.
            //
            // `dq`/`dk`/`dv` are assignments: every element must be written.
            // The pass-1 probability and dS scratch is allocated for the whole
            // rectangular row x key extent while causal masking only owns the
            // lower triangle, so its upper triangle is structurally never
            // written. For those two tensors the requirement is that both arms
            // leave the *same* set unwritten and agree bitwise on everything
            // else, which is stronger than comparing only the written subset.
            for (name, expected, actual, require_written) in [
                ("dq", &from_generic.0, &from_candidate.0, true),
                ("dk", &from_generic.1, &from_candidate.1, true),
                ("dv", &from_generic.2, &from_candidate.2, true),
                ("probability", &from_generic.3, &from_candidate.3, false),
                ("ds", &from_generic.4, &from_candidate.4, false),
            ] {
                let comparison = phase_ix_compare(expected, actual);
                let unwritten = phase_ix_unwritten(actual);
                let unwritten_expected = phase_ix_unwritten(expected);
                let verdict = comparison.bitwise == comparison.elements
                    && (require_written && unwritten == 0
                        || !require_written && unwritten == unwritten_expected);
                if !verdict {
                    failures += 1;
                }
                phase_ix_report_line(
                    &format!("{}/{}/{name}", geometry.label, candidate.label),
                    "two-pass-from-each-forward",
                    &comparison,
                    unwritten,
                    verdict,
                );
                eprintln!(
                    "[phase-ix-parity] {}/{}/{name} unwritten_baseline={unwritten_expected} \
                     unwritten_pattern={} verdict={}",
                    geometry.label,
                    candidate.label,
                    if unwritten == unwritten_expected {
                        "identical"
                    } else {
                        "different"
                    },
                    if unwritten == unwritten_expected {
                        "PASS"
                    } else {
                        "FAIL"
                    }
                );
                elements += comparison.elements;
                bitwise += comparison.bitwise;
            }
            cases += 1;
        }
        // Oracle check: the qualified production two-pass pair against the
        // production scalar backward. This is the pre-existing gradient
        // contract, re-measured here so a PASS above is against a verified
        // baseline rather than against another candidate.
        for (name, expected, actual) in [
            ("dq", &oracle.0, &from_generic.0),
            ("dk", &oracle.1, &from_generic.1),
            ("dv", &oracle.2, &from_generic.2),
        ] {
            let comparison = phase_ix_compare(expected, actual);
            let verdict = comparison.bitwise == comparison.elements;
            eprintln!(
                "[phase-ix-parity] {}/oracle-{} elements={} bitwise={} max_abs={:.9e} \
                 max_ulp={} verdict={}",
                geometry.label,
                name,
                comparison.elements,
                comparison.bitwise,
                comparison.max_abs,
                comparison.max_ulp,
                if verdict { "PASS" } else { "FAIL" }
            );
            // The oracle is mode-0 scalar backward; equality with the two-pass
            // pair is the Phase III/VII contract and is expected to hold.
            if !verdict {
                failures += 1;
            }
            elements += comparison.elements;
            bitwise += comparison.bitwise;
        }
        cases += 1;
    }
    eprintln!(
        "[phase-ix-gradient-summary] status={} cases={cases} elements={elements} \
         bitwise_equal={bitwise} failures={failures}",
        if failures == 0 { "PASS" } else { "FAIL" }
    );
    assert_eq!(failures, 0, "downstream gradients are not identical");
    Ok(())
}

/// One dataset row of the Phase VII qualification fixture, split exactly as
/// `hierarchos-vulkan-training-qualification` splits it: `seq_len` tokens per
/// window, next token as target, zero mask and zero loss weight past the end of
/// the window.
struct PhaseIxSequence {
    input_ids: Vec<u32>,
    targets: Vec<u32>,
    attention_mask: Vec<f32>,
    loss_weights: Vec<f32>,
}

fn phase_ix_load_sequences(path: &std::path::Path, seq_len: usize) -> Result<Vec<PhaseIxSequence>> {
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
            let mut sequence = PhaseIxSequence {
                input_ids: vec![0; seq_len],
                targets: vec![u32::MAX; seq_len],
                attention_mask: vec![0.0; seq_len],
                loss_weights: vec![0.0; seq_len],
            };
            sequence.input_ids[..count].copy_from_slice(&tokens[start..start + count]);
            sequence.targets[..count]
                .copy_from_slice(&tokens[start + 1..start + 1 + count]);
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

fn phase_ix_batch_arrays(
    sequences: &[PhaseIxSequence],
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

fn phase_ix_digest(values: &std::collections::BTreeMap<String, Vec<f32>>) -> (String, usize, f64) {
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

/// The complete Phase VII training step with a selectable attention-forward
/// kernel.
///
/// This drives the same public training surface the Phase VII harness drives
/// (`VulkanTransformer::from_hf_package`, the same LoRA config, the same AdamW
/// hyper-parameters, the same fixture dataset and step order), and its only
/// difference is that the attention-forward kernel is swapped before the loop.
/// One arm per process: `HIERARCHOS_PHASE_IX_STEP_ARM` selects it, so the timing
/// runs are separate processes exactly as Phase VII's protocol requires.
///
/// `HIERARCHOS_PHASE_IX_STEP_ARM`      generic | rowpacked-cached (required)
/// `HIERARCHOS_PHASE_IX_FIXTURE`       fixture dir (default target/gqa-e2e/fixture-8q2kv)
/// `HIERARCHOS_PHASE_IX_STEP_STEPS`    total steps per process (default 29)
/// `HIERARCHOS_PHASE_IX_STEP_DIGEST`   1 = hash gradients and parameters every step
///
/// Digests are hashed host-side and are off by default: turning them on inside
/// a timing run would add per-step host work to the measured interval.
#[test]
fn phase_ix_row_packed_forward_full_step() -> Result<()> {
    let arm = match std::env::var("HIERARCHOS_PHASE_IX_STEP_ARM") {
        Ok(value) => value.trim().to_string(),
        Err(_) => {
            eprintln!(
                "phase-ix full step: status=SKIP reason=not_requested \
                 (set HIERARCHOS_PHASE_IX_STEP_ARM=generic|rowpacked-cached)"
            );
            return Ok(());
        }
    };
    let fixture = std::env::var("HIERARCHOS_PHASE_IX_FIXTURE")
        .unwrap_or_else(|_| "target/gqa-e2e/fixture-8q2kv".to_string());
    let fixture = std::path::PathBuf::from(fixture);
    let steps: usize = std::env::var("HIERARCHOS_PHASE_IX_STEP_STEPS")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(29);
    let warmup: usize = 3;
    let digest = std::env::var("HIERARCHOS_PHASE_IX_STEP_DIGEST")
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

    let sequences = phase_ix_load_sequences(&fixture.join("dataset.jsonl"), seq_len)?;
    let batches_per_epoch = sequences.len() / batch;
    anyhow::ensure!(batches_per_epoch > 0, "fixture dataset is smaller than one batch");

    let device = VulkanDevice::new()?;
    let replacement = match arm.as_str() {
        "generic" => None,
        "rowpacked-cached" => Some(vulkan::ComputeKernel::new(
            &device,
            ROW_PACKED_CACHED,
            11,
            std::mem::size_of::<AttentionPush>() as u32,
        )?),
        other => anyhow::bail!("unknown phase-ix step arm {other}"),
    };
    let mut graph = VulkanTransformer::from_hf_package(device, fixture.join("model"), batch, seq_len)?;
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
    if let Some(kernel) = replacement {
        graph.kernels.attention_forward = kernel;
    }
    let start_parameters = phase_ix_digest(&graph.parameter_values()?);
    eprintln!(
        "[phase-ix-step-run] arm={arm} fixture={} sequences={} batches_per_epoch={batches_per_epoch} \
         batch={batch} seq={seq_len} steps={steps} warmup={warmup} lora_rank=8 lora_alpha=16 \
         lr={} seed={seed} digest={digest} device={} parameter_start_digest={} \
         parameter_elements={}",
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
            phase_ix_batch_arrays(&sequences[begin..begin + batch], seq_len)?;
        let started = Instant::now();
        let result = graph.train_step(&input_ids, &targets, &attention_mask, &loss_weights, hyper)?;
        let step_ms = started.elapsed().as_secs_f64() * 1e3;
        let measured = index >= warmup;
        if digest {
            let gradients = graph.lora_gradient_values()?;
            let gradient_digest = phase_ix_digest(&gradients);
            let parameters = phase_ix_digest(&graph.parameter_values()?);
            eprintln!(
                "[phase-ix-step] arm={arm} step={index} epoch={epoch} batch={batch_index} \
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
                "[phase-ix-step] arm={arm} step={index} epoch={epoch} batch={batch_index} \
                 measured={measured} step_ms={step_ms:.6} loss_bits={:08x} loss={:e}",
                result.loss.to_bits(),
                result.loss
            );
        }
    }
    let end_parameters = phase_ix_digest(&graph.parameter_values()?);
    eprintln!(
        "[phase-ix-step-run-end] arm={arm} parameter_end_digest={} parameter_elements={}",
        end_parameters.0, end_parameters.1
    );
    Ok(())
}

/// Opt-in GPU-timestamp timing of the two forward kernels at the primary
/// geometry.
///
/// Each arm is `warmup + iterations` submissions of one dispatch; the marker
/// line before each submission is what the shipped profiler's block is
/// attributed to (`analyze_timing.py`). Host-observed submission latency is
/// recorded too, but the GPU numbers are the ones reported: an integrated GPU's
/// fence wait is a different quantity from its kernel time.
#[test]
fn phase_ix_row_packed_forward_timing() -> Result<()> {
    let requested = std::env::var("HIERARCHOS_PHASE_IX_KERNEL_TIMING")
        .map(|value| value.trim() == "1")
        .unwrap_or(false);
    if !requested {
        eprintln!(
            "phase-ix kernel timing: status=SKIP reason=not_requested \
             (set HIERARCHOS_PHASE_IX_KERNEL_TIMING=1)"
        );
        return Ok(());
    }
    let profiled = std::env::var_os("HIERARCHOS_VULKAN_PROFILE_KERNELS").is_some();
    eprintln!(
        "phase-ix kernel timing: status=RUN timestamp_profile={profiled} \
         (set HIERARCHOS_VULKAN_PROFILE_KERNELS=1 and \
         HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE=1 for per-submission GPU time)"
    );
    let device = VulkanDevice::new()?;
    eprintln!(
        "phase-ix kernel timing: device={} vendor_id=0x{:04x} device_id=0x{:04x} \
         physical_device_index={}",
        device.name(),
        device.vendor_id(),
        device.device_id(),
        device.physical_device_index()
    );
    let generic = vulkan::ComputeKernel::new(
        &device,
        ATTENTION_FORWARD_SPV,
        11,
        std::mem::size_of::<AttentionPush>() as u32,
    )?;
    let candidates = phase_ix_candidates()
        .into_iter()
        .map(|candidate| {
            let kernel = vulkan::ComputeKernel::new(
                &device,
                candidate.spirv,
                11,
                std::mem::size_of::<AttentionPush>() as u32,
            )?;
            Ok((candidate, kernel))
        })
        .collect::<Result<Vec<_>>>()?;
    let iterations: u32 = std::env::var("HIERARCHOS_PHASE_IX_KERNEL_ITERATIONS")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(25);
    let warmup: u32 = 3;
    let reverse = std::env::var("HIERARCHOS_PHASE_IX_REVERSE")
        .map(|value| value.trim() == "1")
        .unwrap_or(false);

    for geometry in [
        PhaseIxGeometry::plain("gqa-8q-2kv-primary-mode9", 2, 64, 8, 2, 128, 1),
        PhaseIxGeometry::plain("gqa-8q-2kv-primary-mode0", 2, 64, 8, 2, 128, 1).at_mode(0),
        PhaseIxGeometry::plain("mha-8q-8kv-control", 2, 64, 8, 8, 128, 1),
    ] {
        let buffers = phase_ix_buffers(&device, &geometry)?;
        let output = GpuBuffer::from_f32(&device, &vec![0.0_f32; geometry.q_count()])?;
        let mut arms: Vec<(String, &vulkan::ComputeKernel, [u32; 3])> = vec![(
            "forward_generic".to_string(),
            &generic,
            geometry.generic_grid(),
        )];
        for (candidate, kernel) in &candidates {
            if candidate.supports(&geometry) {
                arms.push((
                    format!("forward_{}", candidate.label.replace('-', "")),
                    kernel,
                    geometry.packed_grid(),
                ));
            }
        }
        // The production dispatch site supplies `[query, head, batch]`. A
        // grid-agnostic module is correct under it, at the cost of the
        // workgroups that only return. Measuring that arm quantifies exactly
        // what the training step pays for not changing the dispatch.
        for (candidate, kernel) in &candidates {
            if candidate.supports(&geometry) && candidate.grid_agnostic {
                arms.push((
                    format!(
                        "forward_{}_productiongrid",
                        candidate.label.replace('-', "")
                    ),
                    kernel,
                    geometry.generic_grid(),
                ));
            }
        }
        if reverse {
            arms.reverse();
        }
        for (arm, kernel, grid) in &arms {
            let workgroups = grid[0] as u64 * grid[1] as u64 * grid[2] as u64;
            let lanes: u64 = if arm.contains("rowpacked") { 64 } else { 1 };
            let grid = *grid;
            phase_ix_dispatch_line(&geometry, arm, grid, lanes);
            let submit = |marker: &str| -> Result<()> {
                eprintln!(
                    "[phase-ix-bench{marker}] geometry={} arm={arm} grid={grid:?} \
                     workgroups={workgroups} lanes={lanes}",
                    geometry.label
                );
                let mut commands = vulkan::ComputeBatch::new(&device)?;
                kernel.record_dispatch(
                    &mut commands,
                    &phase_ix_forward_bindings(&buffers, &output),
                    bytemuck::bytes_of(&geometry.push()),
                    grid,
                )?;
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
                "[phase-ix-bench-result] geometry={} arm={arm} workgroups={workgroups} \
                 lanes={lanes} dispatches_per_submission=1 iterations={iterations} \
                 median_host_ms={median:.6} p25_host_ms={p25:.6} min_host_ms={:.6} \
                 max_host_ms={:.6} samples_host_ms={:?}",
                geometry.label,
                sorted[0],
                sorted[sorted.len() - 1],
                sorted
            );
        }
    }
    Ok(())
}
