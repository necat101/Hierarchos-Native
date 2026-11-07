use super::*;
use crate::deepseek_v41::DeepseekV41CompressorAdjoints;
use anyhow::Result;

/// The retained tape must differentiate the gain the forward consumed, not
/// whatever the live projection happens to hold when backward runs.
#[test]
fn deepseek_v41_compressor_tape_snapshots_norm_gain() -> Result<()> {
    let device = VulkanDevice::new()?;
    let (hidden, dim, batch, seq) = (8usize, 16usize, 1usize, 4usize);
    let weights = DeepseekV41CompressorWeights {
        hidden_size: hidden,
        head_dim: dim,
        compress_ratio: 1,
        eps: 1e-20,
        kv_weight: (0..hidden * dim)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.03125)
            .collect(),
        gate_weight: None,
        norm_weight: (0..dim).map(|i| 0.5 + i as f32 * 0.0625).collect(),
    };
    let mut module = DeepseekV41Compressor::new(device.clone(), weights, batch, seq)?;
    let input = GpuBuffer::from_f32(
        &device,
        &(0..batch * seq * hidden)
            .map(|i| (i as f32 * 0.125).sin())
            .collect::<Vec<_>>(),
    )?;
    let upstream = GpuBuffer::from_f32(
        &device,
        &(0..batch * seq * dim)
            .map(|i| (i as f32 * 0.0625).cos())
            .collect::<Vec<_>>(),
    )?;
    let adjoints = DeepseekV41CompressorAdjoints::new(device.clone())?;
    let (_latents, tape) = module.prefill_retained(&input, batch, seq)?;
    let reference = tape
        .backward(&module, &adjoints, Some(&upstream), None)?
        .grad_input
        .read_f32(batch * seq * hidden)?;

    let overwritten = vec![17.0f32; dim];
    module
        .projection
        .as_mut()
        .expect("retained compressor projections")
        .norm_weight
        .write_f32(&overwritten)?;

    let snapshot = tape
        .backward(&module, &adjoints, Some(&upstream), None)?
        .grad_input
        .read_f32(batch * seq * hidden)?;
    let drift = reference
        .iter()
        .zip(&snapshot)
        .map(|(&a, &b)| (a as f64 - b as f64).abs())
        .fold(0.0f64, f64::max);
    assert!(
        drift < 2e-7,
        "backward consumed the overwritten gain: worst absolute drift {drift:e}"
    );

    // Sensitivity check: a tape captured after the overwrite must see the
    // changed gain, so this test cannot pass by ignoring the gain entirely.
    let (_latents, mutated_tape) = module.prefill_retained(&input, batch, seq)?;
    let mutated = mutated_tape
        .backward(&module, &adjoints, Some(&upstream), None)?
        .grad_input
        .read_f32(batch * seq * hidden)?;
    let changed = reference
        .iter()
        .zip(&mutated)
        .map(|(&a, &b)| (a as f64 - b as f64).abs())
        .fold(0.0f64, f64::max);
    assert!(
        changed > 1e-4,
        "overwriting the gain did not change backward: worst absolute drift {changed:e}"
    );
    Ok(())
}

/// Decode reaches a completed group only after a pending cached token. That
/// completed tape must keep the gain its forward consumed, and a tape captured
/// after the overwrite must still see the changed gain. Batch 2 also checks
/// that the per-row pool state and upstream rows stay independent. Its nonzero
/// terminal state VJP is chained through the completed and pending cached steps.
#[test]
fn deepseek_v41_compressor_decode_tape_snapshots_norm_gain() -> Result<()> {
    let device = VulkanDevice::new()?;
    for batch in [1usize, 2usize] {
        decode_tape_snapshot_norm_gain(&device, batch)?;
    }
    Ok(())
}

fn decode_tape_snapshot_norm_gain(device: &VulkanDevice, batch: usize) -> Result<()> {
    let (hidden, dim, prefill_seq, ratio) = (8usize, 16usize, 2usize, 2usize);
    let weights = DeepseekV41CompressorWeights {
        hidden_size: hidden,
        head_dim: dim,
        compress_ratio: ratio,
        eps: 1e-20,
        kv_weight: (0..hidden * dim)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.03125)
            .collect(),
        gate_weight: Some(
            (0..hidden * dim)
                .map(|i| ((i % 5) as f32 - 2.0) * 0.0234375)
                .collect(),
        ),
        norm_weight: (0..dim).map(|i| 0.5 + i as f32 * 0.0625).collect(),
    };
    let mut module = DeepseekV41Compressor::new(device.clone(), weights, batch, 4)?;
    let prefill = GpuBuffer::from_f32(
        &device,
        &(0..batch * prefill_seq * hidden)
            .map(|i| (i as f32 * 0.125).sin())
            .collect::<Vec<_>>(),
    )?;
    let first = GpuBuffer::from_f32(
        &device,
        &(0..batch * hidden)
            .map(|i| (i as f32 * 0.5).sin())
            .collect::<Vec<_>>(),
    )?;
    let second = GpuBuffer::from_f32(
        &device,
        &(0..batch * hidden)
            .map(|i| (i as f32 * 0.25).cos())
            .collect::<Vec<_>>(),
    )?;
    let upstream = GpuBuffer::from_f32(
        &device,
        &(0..batch * dim)
            .map(|i| (i as f32 * 0.0625).cos())
            .collect::<Vec<_>>(),
    )?;
    let adjoints = DeepseekV41CompressorAdjoints::new(device.clone())?;

    let _ = module.prefill_retained(&prefill, batch, prefill_seq)?;
    let (pending, pending_tape) = module.decode_retained(&first, batch, prefill_seq)?;
    assert_eq!(pending.groups_per_batch, 0, "batch {batch}: pending step");
    let (completed, tape) = module.decode_retained(&second, batch, prefill_seq + 1)?;
    assert_eq!(
        completed.groups_per_batch, 1,
        "batch {batch}: completed step"
    );
    assert_eq!(tape.groups_per_batch(), 1, "batch {batch}: retained tape");
    let reference = tape
        .backward(&module, &adjoints, Some(&upstream), None)?
        .grad_input
        .read_f32(batch * hidden)?;

    let state_seed = if batch == 2 {
        let state_len = batch * ratio * dim;
        let seed = |phase: usize| {
            (0..state_len)
                .map(|i| ((i + phase) % 11) as f32 * 0.03125 - 0.140625)
                .collect::<Vec<_>>()
        };
        let kv_seed = seed(0);
        let score_seed = seed(4);
        let state = (
            GpuBuffer::from_f32(device, &kv_seed)?,
            GpuBuffer::from_f32(device, &score_seed)?,
        );

        // An incoming terminal state VJP must affect both decoded tokens:
        // position 3 consumes its written slot; the untouched slot passes to
        // the pending position 2. A dropped state argument would fail this.
        let without_state = decode_backward_chain(
            &module,
            &adjoints,
            &tape,
            &pending_tape,
            Some(&upstream),
            None,
            batch,
            hidden,
            dim,
        )?;
        let with_state = decode_backward_chain(
            &module,
            &adjoints,
            &tape,
            &pending_tape,
            Some(&upstream),
            Some((&state.0, &state.1)),
            batch,
            hidden,
            dim,
        )?;
        for token in 0..2 {
            assert!(
                worst_difference(&with_state[token], &without_state[token]) > 1e-4,
                "batch {batch}: terminal state VJP did not reach decoded token {token}"
            );
        }

        // Activate each row independently with no latent loss. This tests row
        // isolation and the state overwrite boundary without relying on gain
        // sensitivity or comparing two equally misrouted executions.
        for active_row in 0..batch {
            let isolate = |values: &[f32]| {
                values
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        if i / (ratio * dim) == active_row {
                            v
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>()
            };
            let isolated = (
                GpuBuffer::from_f32(device, &isolate(&kv_seed))?,
                GpuBuffer::from_f32(device, &isolate(&score_seed))?,
            );
            let chain = decode_backward_chain(
                &module,
                &adjoints,
                &tape,
                &pending_tape,
                None,
                Some((&isolated.0, &isolated.1)),
                batch,
                hidden,
                dim,
            )?;
            for token in 0..2 {
                for row in 0..batch {
                    let values = &chain[token][row * hidden..(row + 1) * hidden];
                    if row == active_row {
                        assert!(
                            values.iter().any(|v| v.abs() > 1e-6),
                            "batch {batch}: state VJP vanished at token {token}, row {row}"
                        );
                    } else {
                        assert!(
                            values.iter().all(|&v| v == 0.0),
                            "batch {batch}: state VJP crossed into token {token}, row {row}"
                        );
                    }
                }
            }
            // Both inherited slots were overwritten by positions 2 and 3.
            // No terminal-state gradient may reach the earlier prefill state.
            assert!(
                chain[4..].iter().flatten().all(|&v| v == 0.0),
                "batch {batch}: overwritten pool-state slot leaked into prefill"
            );
        }
        Some((state, with_state))
    } else {
        None
    };

    let overwritten = vec![17.0f32; dim];
    module
        .projection
        .as_mut()
        .expect("retained compressor projections")
        .norm_weight
        .write_f32(&overwritten)?;

    if let Some(((kv, score), reference_chain)) = &state_seed {
        // The live cache is mutable too. Neither cached-step tape may observe
        // these writes while differentiating its captured forward operands.
        module
            .kv_state
            .write_f32(&vec![23.0; batch * ratio * dim])?;
        module
            .gate_state
            .write_f32(&vec![-13.0; batch * ratio * dim])?;
        let snapshot_chain = decode_backward_chain(
            &module,
            &adjoints,
            &tape,
            &pending_tape,
            Some(&upstream),
            Some((kv, score)),
            batch,
            hidden,
            dim,
        )?;
        for (index, (before, after)) in reference_chain.iter().zip(&snapshot_chain).enumerate() {
            assert_eq!(
                before.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                after.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "batch {batch}: chained backward buffer {index} consumed overwritten live operands"
            );
        }
        eprintln!("batch 2 state chain: both token VJPs and four state VJPs word-exact after live gain/cache overwrite; isolated rows and overwritten slots verified");
    }

    let snapshot = tape
        .backward(&module, &adjoints, Some(&upstream), None)?
        .grad_input
        .read_f32(batch * hidden)?;
    let drift = reference
        .iter()
        .zip(&snapshot)
        .map(|(&a, &b)| (a as f64 - b as f64).abs())
        .fold(0.0f64, f64::max);
    assert!(
        drift < 2e-7,
        "batch {batch}: decode backward consumed the overwritten gain: worst absolute drift {drift:e}"
    );

    // Restart the sequence after the overwrite and reach the same completed
    // group, so the sensitivity check cannot pass by ignoring the gain.
    let _ = module.prefill_retained(&prefill, batch, prefill_seq)?;
    let (_, mutated_pending_tape) = module.decode_retained(&first, batch, prefill_seq)?;
    let (_completed, mutated_tape) = module.decode_retained(&second, batch, prefill_seq + 1)?;
    let mutated = mutated_tape
        .backward(&module, &adjoints, Some(&upstream), None)?
        .grad_input
        .read_f32(batch * hidden)?;
    let changed = reference
        .iter()
        .zip(&mutated)
        .map(|(&a, &b)| (a as f64 - b as f64).abs())
        .fold(0.0f64, f64::max);
    assert!(
        changed > 1e-4,
        "batch {batch}: overwriting the gain did not change decode backward: worst absolute drift {changed:e}"
    );
    if let Some(((kv, score), reference_chain)) = &state_seed {
        let mutated_chain = decode_backward_chain(
            &module,
            &adjoints,
            &mutated_tape,
            &mutated_pending_tape,
            Some(&upstream),
            Some((kv, score)),
            batch,
            hidden,
            dim,
        )?;
        for token in 0..2 {
            assert!(
                worst_difference(&reference_chain[token], &mutated_chain[token]) > 1e-4,
                "batch {batch}: restarted token {token} ignored the changed norm gain"
            );
        }
    }
    Ok(())
}

fn worst_difference(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    assert!(
        a.iter().chain(b).all(|v| v.is_finite()),
        "nonfinite compressor VJP"
    );
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .fold(0.0, f64::max)
}

#[allow(clippy::too_many_arguments)]
fn decode_backward_chain(
    module: &DeepseekV41Compressor,
    adjoints: &DeepseekV41CompressorAdjoints,
    completed: &DeepseekV41CompressorTape,
    pending: &DeepseekV41CompressorTape,
    upstream: Option<&GpuBuffer>,
    state: Option<(&GpuBuffer, &GpuBuffer)>,
    batch: usize,
    hidden: usize,
    dim: usize,
) -> Result<Vec<Vec<f32>>> {
    let current = completed.backward(module, adjoints, upstream, state)?;
    let kv = current
        .grad_state_kv
        .as_ref()
        .expect("ratio-2 KV state VJP");
    let score = current
        .grad_state_score
        .as_ref()
        .expect("ratio-2 score state VJP");
    let previous = pending.backward(module, adjoints, None, Some((kv, score)))?;
    let state_len = batch * 2 * dim;
    let values = vec![
        current.grad_input.read_f32(batch * hidden)?,
        previous.grad_input.read_f32(batch * hidden)?,
        kv.read_f32(state_len)?,
        score.read_f32(state_len)?,
        previous
            .grad_state_kv
            .as_ref()
            .expect("pending KV state VJP")
            .read_f32(state_len)?,
        previous
            .grad_state_score
            .as_ref()
            .expect("pending score state VJP")
            .read_f32(state_len)?,
    ];
    assert!(
        values.iter().flatten().all(|v| v.is_finite()),
        "nonfinite chained compressor VJP"
    );
    Ok(values)
}
