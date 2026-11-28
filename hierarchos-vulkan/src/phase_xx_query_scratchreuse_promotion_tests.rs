//! Phase XX — production promotion of the Phase XIX query scratch-reuse variant.
//!
//! The Phase XIX parity matrix was driven through a harness pair, so it never
//! exercised the production selector. These tests close that gap: every route
//! assertion below runs [`record_two_pass_attention_backward_dispatch`] with a
//! real production pair (`reason == "ready"`), so the only thing that can change
//! the chosen module is the production selector, the device qualification, the
//! new opt-out and the loaders themselves. Nothing here uses
//! `experimental_pair_override`.
//!
//! Production order under test:
//!
//! 1. `query_dqunroll_scratchreuse` when the Phase XVIII selector admits the
//!    geometry, scratch reuse is enabled (default) and the artifact loads;
//! 2. `query_dqunroll` (Phase XVIII) when scratch reuse is disabled or its
//!    pipeline is refused;
//! 3. `query_original` when the whole dQ-unroll path is disabled or its geometry
//!    is not admitted.
//!
//! The one end-to-end comparison is gated behind `HIERARCHOS_PHASE_XX_ROUTE_AB=1`
//! because it trains two complete LoRA arms; it is an integration check, not a
//! new performance campaign.
//!
//! That A/B driver chooses its two arms deterministically: it injects the
//! scratch-reuse switch value through the existing `#[cfg(test)]`
//! [`two_pass_attention_test_hooks::scope_scratch_reuse`] hook and never parses a
//! per-arm setting from the process environment. Environment parsing, default
//! behaviour, opt-out and precedence are covered by the real-process runs in
//! `validation/phase_xx_query_scratchreuse_promotion_20261127/raw/27_env_precedence.log`,
//! not by this paired test.
//!
//! ```text
//! cargo test --lib phase_xx_ -- --nocapture --test-threads=1
//! HIERARCHOS_PHASE_XX_ROUTE_AB=1 \
//!   cargo test --release --lib phase_xx_production_route_ab -- --nocapture --test-threads=1
//! ```

use super::phase_xix_query_scratch_reuse_tests::{
    phase_xix_batch_arrays, phase_xix_digest, phase_xix_load_sequences,
};
use super::*;
use crate::vulkan::{self, GpuBuffer, VulkanDevice};

/// Written into every output buffer before a dispatch so an unwritten element
/// survives as a sentinel.
const PHASE_XX_SENTINEL: f32 = -98_765.432_1_f32;
const PHASE_XX_BATCH: u32 = 2;

const PHASE_XX_SCRATCH_REUSE_NAME: &str =
    "transformer_attention_backward_query_rdna3_dqunroll_scratchreuse";
const PHASE_XX_DQUNROLL_NAME: &str = "transformer_attention_backward_query_rdna3_dqunroll";
const PHASE_XX_ORIGINAL_NAME: &str = "transformer_attention_backward_query_rdna3";

/// Deterministic operands: identical bytes for every arm of a case.
fn phase_xx_operands(salt: u32, count: usize) -> Vec<f32> {
    let mut state = 0x9e37_79b9_u32 ^ salt.wrapping_mul(0x85eb_ca6b);
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / 16_777_216.0;
            unit * 2.0 - 1.0
        })
        .collect()
}

struct PhaseXxCase {
    push: AttentionPush,
    q_count: usize,
    kv_count: usize,
    scratch: usize,
}

impl PhaseXxCase {
    fn new(
        heads: u32,
        kv_heads: u32,
        mode: u32,
        seq: u32,
        head_dim: u32,
        value_head_dim: u32,
        causal: u32,
    ) -> Self {
        let push = AttentionPush {
            batch_size: PHASE_XX_BATCH,
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
        let q_count = (PHASE_XX_BATCH * seq * push.query_hidden_size) as usize;
        let kv_count = (PHASE_XX_BATCH * seq * push.value_hidden_size) as usize;
        let scratch = two_pass_attention_scratch_elements(&push).expect("scratch size");
        Self {
            push,
            q_count,
            kv_count,
            scratch,
        }
    }
}

struct PhaseXxBuffers {
    query: GpuBuffer,
    key: GpuBuffer,
    value: GpuBuffer,
    grad_output: GpuBuffer,
    query_mask: GpuBuffer,
    key_mask: GpuBuffer,
}

impl PhaseXxBuffers {
    fn new(device: &VulkanDevice, case: &PhaseXxCase) -> Result<Self> {
        Ok(Self {
            query: GpuBuffer::from_f32(device, &phase_xx_operands(71, case.q_count))?,
            key: GpuBuffer::from_f32(device, &phase_xx_operands(72, case.kv_count))?,
            value: GpuBuffer::from_f32(device, &phase_xx_operands(73, case.kv_count))?,
            grad_output: GpuBuffer::from_f32(device, &phase_xx_operands(74, case.q_count))?,
            query_mask: GpuBuffer::from_f32(
                device,
                &vec![1.0_f32; (PHASE_XX_BATCH * case.push.query_seq_len) as usize],
            )?,
            key_mask: GpuBuffer::from_f32(
                device,
                &vec![1.0_f32; (PHASE_XX_BATCH * case.push.key_value_seq_len) as usize],
            )?,
        })
    }
}

/// One submission of the production dispatch helper, exactly as a step makes it.
fn phase_xx_route(
    device: &VulkanDevice,
    kernels: &TransformerKernels,
    case: &PhaseXxCase,
    buffers: &PhaseXxBuffers,
    pair: &TwoPassAttentionBackward,
) -> Result<AttentionBackwardDispatchCounts> {
    let scratch = TwoPassAttentionScratch::new(device, case.scratch)?;
    scratch
        .probability
        .write_f32(&vec![PHASE_XX_SENTINEL; case.scratch])?;
    scratch.ds.write_f32(&vec![PHASE_XX_SENTINEL; case.scratch])?;
    let grad_q = GpuBuffer::from_f32(device, &vec![PHASE_XX_SENTINEL; case.q_count])?;
    let grad_k = GpuBuffer::from_f32(device, &vec![PHASE_XX_SENTINEL; case.kv_count])?;
    let grad_v = GpuBuffer::from_f32(device, &vec![PHASE_XX_SENTINEL; case.kv_count])?;

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
    Ok(two_pass_attention_test_hooks::thread_counts())
}

/// The production pair as an ordinary step resolves it: `reason == "ready"`, so
/// the helper lets the production selectors decide every pass.
fn phase_xx_production_pair(device: &VulkanDevice) -> Result<TwoPassAttentionBackward> {
    let pair = TwoPassAttentionBackward::load(device)?;
    anyhow::ensure!(
        pair.reason == "ready" && pair.is_available(),
        "this device did not build the production two-pass pair ({})",
        pair.reason
    );
    Ok(pair)
}

fn phase_xx_signature(device: &VulkanDevice, spirv: &'static [u8]) -> Result<u64> {
    let kernel = vulkan::ComputeKernel::new(
        device,
        spirv,
        9,
        std::mem::size_of::<AttentionPush>() as u32,
    )?;
    Ok(kernel.shader_signature())
}

/// A fresh kernel set plus production pair for one 8q/2kv sequence-64 mode-0
/// causal dispatch.
fn phase_xx_setup() -> Result<(VulkanDevice, TransformerKernels, PhaseXxCase, PhaseXxBuffers, TwoPassAttentionBackward)>
{
    let device = VulkanDevice::new()?;
    let kernels = TransformerKernels::new(&device)?;
    let case = PhaseXxCase::new(8, 2, 0, 64, 128, 128, 1);
    let buffers = PhaseXxBuffers::new(&device, &case)?;
    let pair = phase_xx_production_pair(&device)?;
    Ok((device, kernels, case, buffers, pair))
}

/// 1. The eligible default selects the scratch-reuse production identity, on the
///    production selector, and nothing else is counted.
#[test]
fn phase_xx_scratch_reuse_default_route() -> Result<()> {
    let _two_pass = two_pass_attention_test_hooks::scope(Some(None), None);
    let _dqunroll = two_pass_attention_test_hooks::scope_dqunroll(Some(None), false);
    let _scratch_reuse = two_pass_attention_test_hooks::scope_scratch_reuse(Some(None), false);
    assert!(attention_backward_query_dqunroll_enabled());
    assert!(attention_backward_query_scratch_reuse_enabled());

    let (device, kernels, case, buffers, pair) = phase_xx_setup()?;
    let counts = phase_xx_route(&device, &kernels, &case, &buffers, &pair)?;
    eprintln!(
        "[phase-xx-default] route=query_dqunroll_scratchreuse query_original={} \
         query_dqunroll={} query_dqunroll_scratchreuse={} scalar={} key_channel4={}",
        counts.query, counts.query_dqunroll, counts.query_dqunroll_scratch_reuse, counts.scalar,
        counts.key_channel4
    );
    assert_eq!(counts.query_dqunroll_scratch_reuse, 1);
    assert_eq!(counts.query_dqunroll, 0);
    assert_eq!(counts.query, 0);
    assert_eq!(counts.scalar, 0);

    let resolved = kernels
        .rdna3_scratchreuse_attention_backward_query
        .get()
        .expect("the default must resolve the scratch-reuse pipeline");
    assert_eq!(resolved.reason, "ready");
    let kernel = resolved
        .kernel
        .as_ref()
        .expect("a ready scratch-reuse pipeline must carry a kernel");
    assert_eq!(kernel.shader_name(), Some(PHASE_XX_SCRATCH_REUSE_NAME));
    assert_eq!(
        kernel.shader_signature(),
        phase_xx_signature(&device, ATTENTION_BACKWARD_QUERY_DQUNROLL_SCRATCH_REUSE_SPV)?
    );
    assert_ne!(
        kernel.shader_signature(),
        phase_xx_signature(&device, ATTENTION_BACKWARD_QUERY_DQUNROLL_SPV)?
    );
    Ok(())
}

/// 2. The scratch-reuse opt-out selects the Phase XVIII dQ-unroll identity and
///    does not even build the candidate pipeline.
#[test]
fn phase_xx_scratch_reuse_opt_out_selects_dqunroll() -> Result<()> {
    let _two_pass = two_pass_attention_test_hooks::scope(Some(None), None);
    let _dqunroll = two_pass_attention_test_hooks::scope_dqunroll(Some(None), false);
    let _scratch_reuse = two_pass_attention_test_hooks::scope_scratch_reuse(Some(Some(false)), false);
    assert!(attention_backward_query_dqunroll_enabled());
    assert!(!attention_backward_query_scratch_reuse_enabled());

    let (device, kernels, case, buffers, pair) = phase_xx_setup()?;
    let counts = phase_xx_route(&device, &kernels, &case, &buffers, &pair)?;
    eprintln!(
        "[phase-xx-opt-out] route=query_dqunroll query_original={} query_dqunroll={} \
         query_dqunroll_scratchreuse={} scalar={}",
        counts.query, counts.query_dqunroll, counts.query_dqunroll_scratch_reuse, counts.scalar
    );
    assert_eq!(counts.query_dqunroll, 1);
    assert_eq!(counts.query_dqunroll_scratch_reuse, 0);
    assert_eq!(counts.query, 0);
    assert_eq!(counts.scalar, 0);
    assert!(
        kernels
            .rdna3_scratchreuse_attention_backward_query
            .get()
            .is_none(),
        "the opt-out must not build the scratch-reuse pipeline"
    );
    let resolved = kernels
        .rdna3_dqunroll_attention_backward_query
        .get()
        .expect("the opt-out must resolve the Phase XVIII pipeline");
    assert_eq!(
        resolved
            .kernel
            .as_ref()
            .expect("a ready dQ-unroll pipeline must carry a kernel")
            .shader_name(),
        Some(PHASE_XX_DQUNROLL_NAME)
    );
    Ok(())
}

/// 3. Disabling the overall dQ-unroll path selects the original query fallback,
///    and enabling scratch reuse cannot override it.
#[test]
fn phase_xx_dqunroll_disable_precedence() -> Result<()> {
    let _two_pass = two_pass_attention_test_hooks::scope(Some(None), None);
    let _dqunroll = two_pass_attention_test_hooks::scope_dqunroll(Some(Some(false)), false);
    let _scratch_reuse = two_pass_attention_test_hooks::scope_scratch_reuse(Some(Some(true)), false);
    assert!(!attention_backward_query_dqunroll_enabled());
    assert!(!attention_backward_query_scratch_reuse_enabled());

    let (device, kernels, case, buffers, pair) = phase_xx_setup()?;
    let counts = phase_xx_route(&device, &kernels, &case, &buffers, &pair)?;
    eprintln!(
        "[phase-xx-disabled] route=query_original query_original={} query_dqunroll={} \
         query_dqunroll_scratchreuse={} scalar={}",
        counts.query, counts.query_dqunroll, counts.query_dqunroll_scratch_reuse, counts.scalar
    );
    assert_eq!(counts.query, 1);
    assert_eq!(counts.query_dqunroll, 0);
    assert_eq!(counts.query_dqunroll_scratch_reuse, 0);
    assert_eq!(counts.scalar, 0);
    assert!(
        kernels
            .rdna3_dqunroll_attention_backward_query
            .get()
            .is_none(),
        "the overall opt-out must not build the Phase XVIII pipeline"
    );
    assert!(
        kernels
            .rdna3_scratchreuse_attention_backward_query
            .get()
            .is_none(),
        "the overall opt-out must not build the scratch-reuse pipeline"
    );
    Ok(())
}

/// 4. A refused (here: undecodable) scratch-reuse pipeline falls back to the
///    Phase XVIII dQ-unroll module, not to the scalar or original query shader.
#[test]
fn phase_xx_scratch_reuse_refused_falls_back_to_dqunroll() -> Result<()> {
    let _two_pass = two_pass_attention_test_hooks::scope(Some(None), None);
    let _dqunroll = two_pass_attention_test_hooks::scope_dqunroll(Some(None), false);
    let _scratch_reuse = two_pass_attention_test_hooks::scope_scratch_reuse(Some(None), true);

    let (device, kernels, case, buffers, pair) = phase_xx_setup()?;
    let counts = phase_xx_route(&device, &kernels, &case, &buffers, &pair)?;
    eprintln!(
        "[phase-xx-fallback] route=query_dqunroll reason=scratchreuse_shader_unavailable \
         query_original={} query_dqunroll={} query_dqunroll_scratchreuse={} scalar={}",
        counts.query, counts.query_dqunroll, counts.query_dqunroll_scratch_reuse, counts.scalar
    );
    assert_eq!(counts.query_dqunroll, 1);
    assert_eq!(counts.query_dqunroll_scratch_reuse, 0);
    assert_eq!(counts.query, 0);
    assert_eq!(counts.scalar, 0);

    let refused = kernels
        .rdna3_scratchreuse_attention_backward_query
        .get()
        .expect("the refused pipeline must be recorded, not retried");
    assert_eq!(refused.reason, "scratchreuse_shader_unavailable");
    assert!(refused.kernel.is_none());
    assert!(refused.detail.is_some());
    let fallback = kernels
        .rdna3_dqunroll_attention_backward_query
        .get()
        .expect("the Phase XVIII fallback must resolve");
    assert_eq!(fallback.reason, "ready");
    Ok(())
}

/// 5. Ineligible geometry keeps its existing behaviour: the original query
///    shader, and neither optional query pipeline is even built.
#[test]
fn phase_xx_ineligible_geometry_unchanged() -> Result<()> {
    let _two_pass = two_pass_attention_test_hooks::scope(Some(None), None);
    let _dqunroll = two_pass_attention_test_hooks::scope_dqunroll(Some(None), false);
    let _scratch_reuse = two_pass_attention_test_hooks::scope_scratch_reuse(Some(None), false);

    let device = VulkanDevice::new()?;
    let kernels = TransformerKernels::new(&device)?;
    let pair = phase_xx_production_pair(&device)?;
    // MHA bypasses the grouped gate, so these are geometries the *outer* two-pass
    // production gate admits while the dQ-unroll head-dimension selector refuses
    // them: exactly the shapes that must keep the original query shader.
    for (label, heads, kv_heads, seq, head_dim, value_head_dim, causal) in [
        ("mha_head_dim_64", 8u32, 8u32, 64u32, 64u32, 64u32, 1u32),
        ("mha_head_dim_32", 8, 8, 32, 32, 32, 1),
    ] {
        let case = PhaseXxCase::new(heads, kv_heads, 0, seq, head_dim, value_head_dim, causal);
        assert_eq!(
            two_pass_attention_rejection(&case.push),
            None,
            "{label} must still be admitted by the outer two-pass gate"
        );
        assert!(
            dqunroll_query_attention_rejection(&case.push).is_some(),
            "{label} must not pass the production query selector"
        );
        let buffers = PhaseXxBuffers::new(&device, &case)?;
        let counts = phase_xx_route(&device, &kernels, &case, &buffers, &pair)?;
        eprintln!(
            "[phase-xx-ineligible] case={label} head_dim={head_dim} \
             value_head_dim={value_head_dim} route=query_original query_original={} \
             query_dqunroll={} query_dqunroll_scratchreuse={}",
            counts.query, counts.query_dqunroll, counts.query_dqunroll_scratch_reuse
        );
        assert_eq!(counts.query, 1);
        assert_eq!(counts.query_dqunroll, 0);
        assert_eq!(counts.query_dqunroll_scratch_reuse, 0);
    }
    assert!(
        kernels
            .rdna3_dqunroll_attention_backward_query
            .get()
            .is_none()
    );
    assert!(
        kernels
            .rdna3_scratchreuse_attention_backward_query
            .get()
            .is_none()
    );
    Ok(())
}

/// 6. The Phase XVIII selector and the enclosing grouped-query production gate
///    were not broadened: the same geometries are admitted and refused, with the
///    same reasons, and the scratch-reuse promotion adds no geometry rule.
#[test]
fn phase_xx_production_gates_not_broadened() -> Result<()> {
    // Helper-level dQ-unroll selector: both head dimensions exactly 128.
    let eligible = PhaseXxCase::new(8, 2, 0, 64, 128, 128, 1);
    assert_eq!(dqunroll_query_attention_rejection(&eligible.push), None);
    for (label, case, reason) in [
        (
            "qk_head_dim_64",
            PhaseXxCase::new(8, 2, 0, 64, 64, 128, 1),
            "dqunroll_query_head_dim_not_128",
        ),
        (
            "value_head_dim_64",
            PhaseXxCase::new(8, 2, 0, 64, 128, 64, 1),
            "dqunroll_query_value_head_dim_not_128",
        ),
    ] {
        assert_eq!(
            dqunroll_query_attention_rejection(&case.push),
            Some(reason),
            "{label}"
        );
    }

    // Enclosing grouped-query gate: unchanged allowlist, head dimension,
    // symmetric sequence 64 and causal-only.
    for (label, case, reason) in [
        (
            "grouping_8q4kv_unchanged",
            PhaseXxCase::new(8, 4, 0, 64, 128, 128, 1),
            None,
        ),
        (
            "grouped_sequence_128",
            PhaseXxCase::new(8, 2, 0, 128, 128, 128, 1),
            Some("grouped_query_attention_unqualified_sequence"),
        ),
        (
            "grouped_non_causal",
            PhaseXxCase::new(8, 2, 0, 64, 128, 128, 0),
            Some("grouped_query_attention_unqualified_causal"),
        ),
        (
            "grouped_unqualified_grouping",
            PhaseXxCase::new(4, 4, 0, 64, 128, 128, 1),
            Some("grouped_query_attention_unqualified_grouping"),
        ),
    ] {
        let head_dim = case.push.query_hidden_size / case.push.num_heads;
        assert_eq!(
            grouped_query_attention_rejection(&case.push, head_dim),
            reason,
            "{label}"
        );
    }

    // MHA keeps its own bypass: one query head per KV head, at any causal
    // setting, still reaches the two-pass route at its qualified dimensions.
    let mha = PhaseXxCase::new(8, 8, 0, 128, 128, 128, 0);
    assert_eq!(two_pass_attention_rejection(&mha.push), None);
    let mha_grouped = group_by_kv(&mha.push);
    assert_eq!(
        grouped_query_attention_rejection(&mha_grouped, 128),
        Some("grouped_query_attention_unqualified_sequence"),
        "the grouped gate still refuses this shape when it is applied"
    );
    Ok(())
}

/// A copy of `push` with one query head per KV head removed, used only to show
/// that the grouped gate itself is unchanged when it *is* consulted.
fn group_by_kv(push: &AttentionPush) -> AttentionPush {
    let mut grouped = *push;
    grouped.num_key_value_heads = push.num_heads / 4;
    grouped.value_hidden_size = grouped.num_key_value_heads * 128;
    grouped
}

/// 7. The three query modules report three distinct production identities, so the
///    profiler cannot attribute one route's work to another's counter.
#[test]
fn phase_xx_profiler_identities_distinct() -> Result<()> {
    let Ok(device) = VulkanDevice::new() else {
        eprintln!("phase-xx identities: status=BLOCKED reason=no_vulkan_device");
        return Ok(());
    };
    let mut names = Vec::new();
    let mut signatures = Vec::new();
    for (label, spirv) in [
        ("scratchreuse", ATTENTION_BACKWARD_QUERY_DQUNROLL_SCRATCH_REUSE_SPV),
        ("dqunroll", ATTENTION_BACKWARD_QUERY_DQUNROLL_SPV),
        ("original", ATTENTION_BACKWARD_QUERY_PARALLEL_SPV),
    ] {
        let kernel = vulkan::ComputeKernel::new(
            &device,
            spirv,
            9,
            std::mem::size_of::<AttentionPush>() as u32,
        )?;
        eprintln!(
            "[phase-xx-identity] artifact={label} signature=0x{:016x} name={:?}",
            kernel.shader_signature(),
            kernel.shader_name()
        );
        names.push(kernel.shader_name());
        signatures.push(kernel.shader_signature());
    }
    assert_eq!(
        names,
        vec![
            Some(PHASE_XX_SCRATCH_REUSE_NAME),
            Some(PHASE_XX_DQUNROLL_NAME),
            Some(PHASE_XX_ORIGINAL_NAME),
        ]
    );
    assert_ne!(signatures[0], signatures[1]);
    assert_ne!(signatures[0], signatures[2]);
    assert_ne!(signatures[1], signatures[2]);
    Ok(())
}

struct PhaseXxArm {
    losses: Vec<u32>,
    grad_digests: Vec<String>,
    param_digests: Vec<String>,
    end_digest: String,
    counts: AttentionBackwardDispatchCounts,
}

/// One A/B arm. `scratch_reuse` is a test-hook value, not an environment
/// setting: it is injected through
/// [`two_pass_attention_test_hooks::scope_scratch_reuse`], so the production
/// selector is handed a deterministic arm instead of one parsed from the process
/// environment. Everything else — the dispatch function, loader, device
/// qualification, buffers, dispatch path and counters — is production code.
#[allow(clippy::too_many_arguments)]
fn phase_xx_run_arm(
    fixture: &std::path::Path,
    sequences: &[super::phase_xix_query_scratch_reuse_tests::PhaseXixSequence],
    steps: usize,
    batch: usize,
    seq_len: usize,
    lora: bool,
    lr: f32,
    scratch_reuse: bool,
) -> Result<PhaseXxArm> {
    let hyper = AdamWHyperParams {
        lr,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1.0e-8,
        weight_decay: 0.0,
    };
    let seed = 7u64;
    let batches_per_epoch = sequences.len() / batch;

    // Pin every switch this phase cares about: two-pass on, dQ-unroll default,
    // and this arm's scratch-reuse value. The value is injected through the
    // `#[cfg(test)]` hook, not parsed from the process environment; no harness
    // pair participates either, so the production pair (`reason == "ready"`) is
    // what the selector sees.
    let _two_pass = two_pass_attention_test_hooks::scope(Some(None), None);
    let _dqunroll = two_pass_attention_test_hooks::scope_dqunroll(Some(None), false);
    let _scratch_reuse = two_pass_attention_test_hooks::scope_scratch_reuse(
        Some(if scratch_reuse { None } else { Some(false) }),
        false,
    );

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

    let backward_before = attention_backward_dispatch_counts();
    let mut losses = Vec::with_capacity(steps);
    let mut grad_digests = Vec::with_capacity(steps);
    let mut param_digests = Vec::with_capacity(steps);
    for index in 0..steps {
        let batch_index = index % batches_per_epoch;
        let begin = batch_index * batch;
        let (input_ids, targets, attention_mask, loss_weights) =
            phase_xix_batch_arrays(&sequences[begin..begin + batch], seq_len)?;
        let result = graph.train_step(&input_ids, &targets, &attention_mask, &loss_weights, hyper)?;
        let parameters = phase_xix_digest(&graph.parameter_values()?);
        let gradients = if lora {
            let gradients = phase_xix_digest(&graph.lora_gradient_values()?);
            eprintln!(
                "[phase-xx-route-step] arm={} step={index} loss_bits={:08x} \
                 grad_digest={} param_digest={}",
                if scratch_reuse { "B" } else { "A" },
                result.loss.to_bits(),
                gradients.0,
                parameters.0
            );
            gradients.0
        } else {
            eprintln!(
                "[phase-xx-route-step] arm={} step={index} loss_bits={:08x} param_digest={}",
                if scratch_reuse { "B" } else { "A" },
                result.loss.to_bits(),
                parameters.0
            );
            String::new()
        };
        losses.push(result.loss.to_bits());
        grad_digests.push(gradients);
        param_digests.push(parameters.0);
    }
    let backward_after = attention_backward_dispatch_counts();
    let end_digest = phase_xix_digest(&graph.parameter_values()?).0;
    Ok(PhaseXxArm {
        losses,
        grad_digests,
        param_digests,
        end_digest,
        counts: AttentionBackwardDispatchCounts {
            scalar: backward_after.scalar - backward_before.scalar,
            query: backward_after.query - backward_before.query,
            query_dqunroll: backward_after.query_dqunroll - backward_before.query_dqunroll,
            query_dqunroll_scratch_reuse: backward_after.query_dqunroll_scratch_reuse
                - backward_before.query_dqunroll_scratch_reuse,
            key: backward_after.key - backward_before.key,
            key_channel4: backward_after.key_channel4 - backward_before.key_channel4,
            scratch_allocations: backward_after.scratch_allocations
                - backward_before.scratch_allocations,
            scratch_bytes: backward_after.scratch_bytes - backward_before.scratch_bytes,
        },
    })
}

/// One actual production-route A/B. Both arms run the production dispatch
/// function, loader, device qualification, selector, buffers, dispatch path and
/// counters with a real production pair (`reason == "ready"`); neither uses
/// `experimental_pair_override`. Arm A pins the scratch-reuse switch off and arm
/// B leaves it at its production default, both through the `#[cfg(test)]`
/// `scope_scratch_reuse` hook, so arm selection is deterministic and never parsed
/// from the process environment. Both arms compare the same per-step losses,
/// LoRA gradients and parameter digests bitwise, and the production dispatch
/// counters name the route each arm actually took.
///
/// This is a dispatch-integration check. It is not a new performance campaign:
/// the Phase XIX complete-step measurements remain the performance evidence for
/// this byte-identical module.
#[test]
fn phase_xx_production_route_ab() -> Result<()> {
    let requested = std::env::var("HIERARCHOS_PHASE_XX_ROUTE_AB")
        .map(|value| value.trim() == "1")
        .unwrap_or(false);
    if !requested {
        eprintln!(
            "phase-xx route A/B: status=SKIP reason=not_requested \
             (set HIERARCHOS_PHASE_XX_ROUTE_AB=1)"
        );
        return Ok(());
    }
    let fixture = std::path::PathBuf::from(
        std::env::var("HIERARCHOS_PHASE_XX_FIXTURE")
            .unwrap_or_else(|_| "target/gqa-e2e/fixture-8q2kv".to_string()),
    );
    anyhow::ensure!(
        fixture.join("model").is_dir(),
        "fixture {} does not contain a model directory",
        fixture.display()
    );
    let steps: usize = std::env::var("HIERARCHOS_PHASE_XX_STEPS")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(29);
    let batch = 2usize;
    let seq_len = 64usize;
    let lora = true;
    let lr = 1.0e-2_f32;
    let sequences = phase_xix_load_sequences(&fixture.join("dataset.jsonl"), seq_len)?;
    anyhow::ensure!(sequences.len() >= batch, "fixture is smaller than one batch");

    eprintln!(
        "[phase-xx-route-ab] fixture={} sequences={} batch={batch} seq={seq_len} steps={steps} \
         lora={lora} lr={lr} arm_a=dqunroll arm_b=scratchreuse",
        fixture.display(),
        sequences.len()
    );
    let arm_a = phase_xx_run_arm(&fixture, &sequences, steps, batch, seq_len, lora, lr, false)?;
    let arm_b = phase_xx_run_arm(&fixture, &sequences, steps, batch, seq_len, lora, lr, true)?;

    let mut comparisons = 0usize;
    let mut mismatches = 0usize;
    for step in 0..steps {
        for (label, a, b) in [
            ("loss", format!("{:08x}", arm_a.losses[step]), format!("{:08x}", arm_b.losses[step])),
            (
                "grad",
                arm_a.grad_digests[step].clone(),
                arm_b.grad_digests[step].clone(),
            ),
            (
                "param",
                arm_a.param_digests[step].clone(),
                arm_b.param_digests[step].clone(),
            ),
        ] {
            comparisons += 1;
            if a != b {
                mismatches += 1;
                eprintln!("[phase-xx-route-ab] step={step} tensor={label} a={a} b={b} verdict=FAIL");
            }
        }
    }
    let loss_bits_equal = arm_a.losses == arm_b.losses;

    eprintln!(
        "[phase-xx-route-ab-end] steps={steps} comparisons={comparisons} mismatches={mismatches} \
         loss_bits_equal={loss_bits_equal} \
         arm_a_query_original={} arm_a_query_dqunroll={} arm_a_query_scratchreuse={} \
         arm_a_scalar={} arm_a_key_original={} arm_a_key_channel4={} \
         arm_b_query_original={} arm_b_query_dqunroll={} arm_b_query_scratchreuse={} \
         arm_b_scalar={} arm_b_key_original={} arm_b_key_channel4={} \
         param_end_a={} param_end_b={} verdict={}",
        arm_a.counts.query,
        arm_a.counts.query_dqunroll,
        arm_a.counts.query_dqunroll_scratch_reuse,
        arm_a.counts.scalar,
        arm_a.counts.key,
        arm_a.counts.key_channel4,
        arm_b.counts.query,
        arm_b.counts.query_dqunroll,
        arm_b.counts.query_dqunroll_scratch_reuse,
        arm_b.counts.scalar,
        arm_b.counts.key,
        arm_b.counts.key_channel4,
        arm_a.end_digest,
        arm_b.end_digest,
        if mismatches == 0 && loss_bits_equal { "PASS" } else { "FAIL" },
    );

    let layers = (steps * 2) as u64;
    anyhow::ensure!(
        arm_a.counts.query_dqunroll == layers
            && arm_a.counts.query_dqunroll_scratch_reuse == 0
            && arm_a.counts.query == 0
            && arm_a.counts.scalar == 0,
        "arm A did not run the Phase XVIII dQ-unroll production route: {:?}",
        arm_a.counts
    );
    anyhow::ensure!(
        arm_b.counts.query_dqunroll_scratch_reuse == layers
            && arm_b.counts.query_dqunroll == 0
            && arm_b.counts.query == 0
            && arm_b.counts.scalar == 0,
        "arm B did not run the scratch-reuse production route: {:?}",
        arm_b.counts
    );
    anyhow::ensure!(
        arm_a.counts.key + arm_a.counts.key_channel4 == layers && arm_a.counts.key == 0,
        "arm A pass 2 route changed: {:?}",
        arm_a.counts
    );
    anyhow::ensure!(
        arm_b.counts.key + arm_b.counts.key_channel4 == layers && arm_b.counts.key == 0,
        "arm B pass 2 route changed: {:?}",
        arm_b.counts
    );
    anyhow::ensure!(loss_bits_equal, "the arms diverged in per-step loss bits");
    anyhow::ensure!(mismatches == 0, "the arms diverged in {mismatches} comparisons");
    anyhow::ensure!(
        arm_a.end_digest == arm_b.end_digest,
        "the arms ended with different parameter digests"
    );
    Ok(())
}
