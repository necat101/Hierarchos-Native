use super::*;
use sha2::{Digest, Sha256};

#[test]
fn gpt2_cpu_math_preserves_gen9_and_other_architecture_routes() {
    use VulkanTransformerArchitecture::{Gpt2, Llama};
    assert!(rdna3_gpt2_cpu_math(Gpt2, 0x1002, 0x15bf));
    assert!(!rdna3_gpt2_cpu_math(Gpt2, 0x8086, 0x1916));
    assert!(!rdna3_gpt2_cpu_math(Gpt2, 0x8086, 0x5916));
    assert!(!rdna3_gpt2_cpu_math(Gpt2, 0x1002, 0x163f));
    assert!(!rdna3_gpt2_cpu_math(Llama, 0x1002, 0x15bf));
    // The Welford norm and correctly rounded reciprocal follow the GPT-2
    // architecture on every adapter; the tanh GELU and k-block GEMM stay on
    // their measured RDNA3 gate.
    assert!(gpt2_welford_norm(Gpt2));
    assert!(!gpt2_welford_norm(Llama));
    assert!(gpt2_attention_reciprocal(Gpt2));
    assert!(!gpt2_attention_reciprocal(Llama));
}

#[derive(Deserialize)]
struct ParityFixture {
    comparison: String,
    absolute_limit: f64,
    rtol: u32,
    files: BTreeMap<String, String>,
    cases: Vec<ParityCase>,
}

#[derive(Deserialize)]
struct ParityCase {
    name: String,
    batch_size: usize,
    seq_len: usize,
    input_ids: Vec<u32>,
    attention_mask: Vec<f32>,
    logits_bits: Vec<u32>,
    visible: Vec<bool>,
}

#[test]
fn gpt2_wide_frozen_cpu_logits_resident_and_streamed() -> Result<()> {
    let _env = crate::test_support::env_lock();
    // The frozen fixture records an AVX2 oracle (`"cpu": "AVX2"`). Pin the
    // CPU-shaped kernels to that oracle instead of inheriting this host's
    // AVX-512 width; the global 2e-7, rtol=0 gate is unchanged.
    let _width = with_aten_vector_width(8);
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("validation/fixtures/gpt2_rdna3_parity");
    let fixture: ParityFixture = serde_json::from_slice(&fs::read(root.join("fixture.json"))?)?;
    assert_eq!(fixture.comparison, "strict_less_than");
    assert_eq!(fixture.absolute_limit, 2e-7);
    assert_eq!(fixture.rtol, 0);
    assert_eq!(fixture.cases.len(), 2);
    for (name, expected) in &fixture.files {
        assert_eq!(
            format!("{:x}", Sha256::digest(fs::read(root.join(name))?)),
            *expected
        );
    }
    let device = VulkanDevice::new()?;
    let config = VulkanTransformerConfig::from_hf_config(root.join("config.json"))?;
    assert_eq!((config.hidden_size, config.vocab_size), (64, 320));
    for case in fixture.cases {
        let mut resident = VulkanTransformer::from_hf_package_with_config(
            device.clone(),
            &root,
            config.clone(),
            case.batch_size,
            case.seq_len,
        )?;
        let mut streamed = VulkanTransformer::from_hf_package_with_streamed_parameters(
            device.clone(),
            &root,
            config.clone(),
            case.batch_size,
            case.seq_len,
        )?;
        assert_eq!(resident.streamed_linear_count(), 0);
        assert!(streamed.streamed_linear_count() > 0);
        let a = resident.forward_logits(&case.input_ids, &case.attention_mask)?;
        let b = streamed.forward_logits(&case.input_ids, &case.attention_mask)?;
        assert_eq!(a.len(), case.logits_bits.len());
        assert_eq!(a.len(), case.visible.len());
        assert!(case
            .logits_bits
            .iter()
            .all(|x| f32::from_bits(*x).is_finite()));
        assert!(a.iter().chain(&b).all(|x| x.is_finite()));
        assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        let mut worst = 0.0f64;
        let mut count = 0;
        for ((value, expected), visible) in a.iter().zip(&case.logits_bits).zip(&case.visible) {
            if *visible {
                worst = worst.max((*value as f64 - f32::from_bits(*expected) as f64).abs());
                count += 1;
            }
        }
        assert!(count > 0);
        assert!(
            worst < fixture.absolute_limit,
            "{} drift {worst:e}",
            case.name
        );
        eprintln!(
            "GPT-2 frozen CPU {}: {count} visible logits per mode, worst {worst:e}",
            case.name
        );
    }
    Ok(())
}
