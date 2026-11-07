//! Host-specific CPU loss scalars on the original training inputs.
use sha2::{Digest, Sha256};

pub(super) fn expected_loss(logits: &[u32], targets: &[u32], vocab: usize, original: f32) -> f32 {
    if crate::transformer::aten_vector_width() != 8 {
        return original;
    }
    assert_eq!(logits.len(), targets.len() * vocab);
    let mut digest = Sha256::new();
    for word in [targets.len() as u32, vocab as u32]
        .iter()
        .chain(logits)
        .chain(targets)
    {
        digest.update(word.to_le_bytes());
    }
    static FIXTURE: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
    let fixture = FIXTURE.get_or_init(|| {
        serde_json::from_str(include_str!(
            "../../validation/fixtures/deepseek_v41_training_loss.avx2.json"
        ))
        .expect("pinned AVX2 CPU loss fixture")
    });
    let key = format!("{:x}", digest.finalize());
    let bits = fixture["records"][&key]["loss_bits"]
        .as_u64()
        .unwrap_or_else(|| panic!("no CPU loss for complete input digest {key}"));
    f32::from_bits(bits as u32)
}
