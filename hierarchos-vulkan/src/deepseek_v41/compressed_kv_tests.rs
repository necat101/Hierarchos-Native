use super::*;

fn weights(ratio: usize) -> DeepseekV41CompressorWeights {
    let kv_weight = (0..16 * 8)
        .map(|i| ((i % 9) as f32 - 4.0) * 0.017)
        .collect();
    let gate_weight = (0..16 * 8)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.031)
        .collect();
    DeepseekV41CompressorWeights {
        hidden_size: 8,
        head_dim: 16,
        compress_ratio: ratio,
        eps: 1e-20,
        kv_weight,
        gate_weight: (ratio == 2).then_some(gate_weight),
        norm_weight: (0..16).map(|i| 0.5 + i as f32 * 0.025).collect(),
    }
}

fn inputs(batch: usize, seq: usize) -> Vec<f32> {
    (0..batch * seq * 8)
        .map(|i| ((i * 37 % 23) as f32 - 11.0) * 0.043)
        .collect()
}

fn assert_bits(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label}, value {index}: native={a:.10e}, composed={b:.10e}"
        );
    }
}

#[test]
fn deepseek_v41_compressed_kv_rotates_then_quantizes_without_mutating_latents() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let mut compressor = DeepseekV41Compressor::new(device.clone(), weights(2), 2, 8)?;
    let input = inputs(2, 6);
    let latent = compressor.prefill(&input, 2, 6)?;
    assert_eq!(latent.compress_ratio, 2);
    assert_eq!(latent.groups_per_batch, 3);
    let before = latent.read_f32()?;

    let config = DeepseekV41RotaryConfig::sliding(8, 32, 10000.0);
    let finalizer = DeepseekV41CompressedKvFinalizer::new(device.clone(), config, 2, 16)?;
    assert_eq!(finalizer.ratio(), 2);
    assert_eq!(finalizer.rope_dim(), 8);
    assert_bits(
        "pre-RoPE hand-off",
        &finalizer.pre_rope_latent(&latent).read_f32(96)?,
        &before,
    );

    let finalized = finalizer.finalize_prefill(&latent)?;
    let actual = finalized.read_values()?;
    // The released order: index keys read the latent first; only then does the
    // compressed cache consume the rotated, quantized form.
    let rotary = DeepseekV41Rotary::new(device.clone(), config)?;
    let rotated = rotary.apply_grouped(&before, 6, 16, 3, 0, 2, false)?;
    let quantizer = DeepseekV41ActivationQuantizer::new(device.clone())?;
    let expected = quantizer.quantize(
        &rotated.read_f32(96)?,
        6,
        16,
        COMPRESSED_KV_QUANT_BLOCK,
        DeepseekV41QuantMode::Fp4E2M1,
        DeepseekV41ScaleMode::E4m3,
    )?;
    assert_bits("prefill", &actual, &expected.read_values()?);
    assert_eq!(
        finalized.read_scale_bits()?,
        expected.read_scale_bits()?,
        "prefill scales"
    );
    // Finalization reads the latent through a separate output buffer.
    assert_bits("latent after finalization", &latent.read_f32()?, &before);
    Ok(())
}

#[test]
fn deepseek_v41_compressed_kv_decode_uses_the_completed_group_position() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let mut compressor = DeepseekV41Compressor::new(device.clone(), weights(2), 2, 8)?;
    compressor.prefill(&inputs(2, 6), 2, 6)?;
    let token = inputs(2, 1);
    let pending = compressor.decode_step(&token, 2, 6)?;
    assert_eq!(pending.groups_per_batch, 0);
    let completed = compressor.decode_step(&token, 2, 7)?;
    assert_eq!(completed.groups_per_batch, 1);
    assert_eq!(completed.first_group, 3);
    let before = completed.read_f32()?;

    let config = DeepseekV41RotaryConfig::sliding(8, 32, 10000.0);
    let finalizer = DeepseekV41CompressedKvFinalizer::new(device.clone(), config, 2, 16)?;
    let finalized = finalizer.finalize_decode(&completed, 7)?;
    let actual = finalized.read_values()?;
    let rotary = DeepseekV41Rotary::new(device.clone(), config)?;
    let rotated = rotary.apply_grouped(&before, 2, 16, 1, 6, 2, false)?;
    let quantizer = DeepseekV41ActivationQuantizer::new(device.clone())?;
    let expected = quantizer.quantize(
        &rotated.read_f32(32)?,
        2,
        16,
        COMPRESSED_KV_QUANT_BLOCK,
        DeepseekV41QuantMode::Fp4E2M1,
        DeepseekV41ScaleMode::E4m3,
    )?;
    assert_bits("decode", &actual, &expected.read_values()?);
    Ok(())
}

#[test]
fn deepseek_v41_compressed_kv_rejects_wrong_ratio_and_phase() -> Result<()> {
    let device = crate::VulkanDevice::new()?;
    let mut compressor = DeepseekV41Compressor::new(device.clone(), weights(2), 2, 8)?;
    let latent = compressor.prefill(&inputs(2, 6), 2, 6)?;
    let config = DeepseekV41RotaryConfig::sliding(8, 32, 10000.0);
    let ratio_one = DeepseekV41CompressedKvFinalizer::new(device.clone(), config, 1, 16)?;
    let err = ratio_one
        .finalize_prefill(&latent)
        .err()
        .expect("ratio mismatch must fail")
        .to_string();
    assert!(err.contains("compression ratio mismatch"), "{err}");

    let finalizer = DeepseekV41CompressedKvFinalizer::new(device.clone(), config, 2, 16)?;
    let token = inputs(2, 1);
    let pending = compressor.decode_step(&token, 2, 6)?;
    assert!(finalizer.finalize_prefill(&pending).is_err());
    assert!(finalizer.finalize_decode(&pending, 5).is_err());
    assert!(DeepseekV41CompressedKvFinalizer::new(device.clone(), config, 3, 16).is_err());
    assert!(
        DeepseekV41CompressedKvFinalizer::new(device, config, 2, 12).is_err(),
        "head dimension must quantize in 16-element groups"
    );
    Ok(())
}
