use super::*;
use crate::deepseek_v41::{
    DeepseekV41EngramResidency, DeepseekV41ExpertResidency, DeepseekV41ProjectionResidency,
};

fn packed_fixture() -> Result<Fixture> {
    let source: Value = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_bf16_model.json"
    ))?;
    Ok(serde_json::from_value(source["cases"][0].clone())?)
}

fn load_vocabulary(
    device: &VulkanDevice,
    f: &Fixture,
    package: &Package,
    precision: DeepseekV41WeightOutput,
    policy: DeepseekV41VocabularyResidency,
) -> Result<DeepseekV41CausalLM> {
    let mut reader = package.reader()?;
    DeepseekV41CausalLM::from_checkpoint_with_vocabulary_residency(
        device.clone(),
        &mut reader,
        runtime(f),
        Some(f.tokens.clone()),
        precision,
        DeepseekV41ExpertResidency::Resident,
        DeepseekV41EngramResidency::Resident,
        DeepseekV41ProjectionResidency::Resident,
        policy,
    )
}

#[test]
fn deepseek_v41_streamed_vocabulary_preflights_both_headers() -> Result<()> {
    let f = packed_fixture()?;
    let package = Package::new(&f, true)?;
    let path = package.root.join("model.safetensors.index.json");
    let mut index: Value = serde_json::from_slice(&fs::read(&path)?)?;
    assert!(index["weight_map"]
        .as_object_mut()
        .unwrap()
        .remove("head.weight")
        .is_some());
    fs::write(path, serde_json::to_vec(&index)?)?;
    let error = package
        .reader()?
        .validate_vocabulary_metadata(true)
        .err()
        .unwrap();
    assert!(format!("{error:#}").contains("head.weight"), "{error:#}");

    // Malformed temporary packages only; frozen expectations remain untouched.
    let mut f = packed_fixture()?;
    f.tensors.get_mut("head.weight").unwrap().shape.swap(0, 1);
    let package = Package::new(&f, false)?;
    let error = package
        .reader()?
        .validate_vocabulary_metadata(true)
        .err()
        .unwrap();
    assert!(
        format!("{error:#}").contains("head.weight shape"),
        "{error:#}"
    );

    let mut f = packed_fixture()?;
    f.tensors.get_mut("head.weight").unwrap().dtype.clear();
    let package = Package::new(&f, false)?;
    package.reader()?.validate_vocabulary_metadata(false)?;
    let error = package
        .reader()?
        .validate_vocabulary_metadata(true)
        .err()
        .unwrap();
    assert!(
        format!("{error:#}").contains("head.weight must use BF16"),
        "{error:#}"
    );
    Ok(())
}

#[test]
fn deepseek_v41_streamed_vocabulary_payload_errors_invalidate_sequence() -> Result<()> {
    for half in [false, true] {
        for name in ["embed.weight", "head.weight"] {
            let mut f = if half { packed_fixture()? } else { fixture()? };
            f.tensors.get_mut(name).unwrap().values[0] = f32::from_bits(0x7fc00000);
            let package = Package::new(&f, half)?;
            let device = VulkanDevice::new()?;
            let precision = if half {
                DeepseekV41WeightOutput::Bf16
            } else {
                DeepseekV41WeightOutput::Fp32
            };
            let error = load_vocabulary(
                &device,
                &f,
                &package,
                precision,
                DeepseekV41VocabularyResidency::Resident,
            )
            .err()
            .unwrap();
            assert!(format!("{error:#}").contains(name), "{error:#}");
            let mut streamed = load_vocabulary(
                &device,
                &f,
                &package,
                precision,
                DeepseekV41VocabularyResidency::Streamed,
            )?;
            assert_eq!(streamed.resident_vocabulary_tensor_count(), 0);
            let error = run(&mut streamed, &f.steps[0]).err().unwrap();
            assert!(
                format!("{error:#}").contains("non-finite") && format!("{error:#}").contains(name),
                "{error:#}"
            );
            assert_eq!(streamed.next_position(), None);
            let error = streamed
                .decode(&vec![0; f.batch_size], 1, None)
                .err()
                .unwrap();
            assert!(
                format!("{error:#}").contains("restart with prefill"),
                "{error:#}"
            );
            assert_eq!(streamed.resident_vocabulary_tensor_count(), 0);
        }
    }
    Ok(())
}

#[test]
fn deepseek_v41_streamed_vocabulary_buffers_are_released_after_each_operation() -> Result<()> {
    for half in [false, true] {
        let f = if half { packed_fixture()? } else { fixture()? };
        let package = Package::new(&f, half)?;
        let device = VulkanDevice::new()?;
        let precision = if half {
            DeepseekV41WeightOutput::Bf16
        } else {
            DeepseekV41WeightOutput::Fp32
        };
        let resident = load_vocabulary(
            &device,
            &f,
            &package,
            precision,
            DeepseekV41VocabularyResidency::Resident,
        )?;
        let streamed = load_vocabulary(
            &device,
            &f,
            &package,
            precision,
            DeepseekV41VocabularyResidency::Streamed,
        )?;
        assert_eq!(resident.resident_vocabulary_tensor_count(), 2);
        assert_eq!(streamed.resident_vocabulary_tensor_count(), 0);
        let rows = f.batch_size;
        let hidden = f.config["text_config"]["hidden_size"].as_u64().unwrap() as usize;
        let ids = GpuBuffer::from_u32(&device, &f.steps[0].ids[..rows])?;
        let x = GpuBuffer::from_f32(&device, &f.tensors["embed.weight"].values[..rows * hidden])?;
        let baseline = device.memory_stats()?.live_buffer_bytes;
        for _ in 0..3 {
            let a = resident.embed(&ids, rows)?;
            let b = streamed.embed(&ids, rows)?;
            let count =
                rows * hidden * f.config["text_config"]["hc_mult"].as_u64().unwrap() as usize;
            assert_eq!(a.read_u32(count)?, b.read_u32(count)?);
            drop((a, b));
            assert_eq!(
                device.memory_stats()?.live_buffer_bytes,
                baseline,
                "embedding staging retained"
            );
            let a = resident.project_logits(&x, rows)?;
            let b = streamed.project_logits(&x, rows)?;
            let count = rows * f.config["text_config"]["vocab_size"].as_u64().unwrap() as usize;
            assert_eq!(a.read_u32(count)?, b.read_u32(count)?);
            drop((a, b));
            assert_eq!(
                device.memory_stats()?.live_buffer_bytes,
                baseline,
                "head staging retained"
            );
            assert_eq!(streamed.resident_vocabulary_tensor_count(), 0);
        }
    }
    Ok(())
}
