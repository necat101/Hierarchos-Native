use super::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

fn bits(v: &Value) -> Vec<u32> {
    serde_json::from_value(v.clone()).unwrap()
}

#[test]
fn deepseek_v41_schedule_width_attention_contractions_match_cpu_bits() -> Result<()> {
    let source = include_bytes!("../../validation/fixtures/deepseek_v41_schedule_attention.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(source)),
        "c04ac2bfa81aebe9c98219222c48548acbf5504ab7bb14a66a7082a32358a725"
    );
    let f: Value = serde_json::from_slice(source)?;
    assert_eq!(
        f["parent_fixture_sha256"],
        "70d2218429c2d4b8623a8ebbdd6c659eec22f7f64f9eeecd3d95c5d7877fdcf0"
    );
    let device = VulkanDevice::new()?;
    let s = &f["sparse"];
    let capacity = s["kv_shape"][1].as_u64().unwrap() as usize;
    let slots = s["indices_shape"][2].as_u64().unwrap() as usize;
    let q = GpuBuffer::from_u32(&device, &bits(&s["q_bits"]))?;
    let kv = GpuBuffer::from_u32(&device, &bits(&s["kv_bits"]))?;
    let sink = GpuBuffer::from_u32(&device, &bits(&s["sink_bits"]))?;
    let indices: Vec<i32> = serde_json::from_value(s["indices"].clone())?;
    let idx = GpuBuffer::from_u32(
        &device,
        &indices.iter().map(|&i| i as u32).collect::<Vec<_>>(),
    )?;
    let empty = GpuBuffer::zeros_u32(&device, 1)?;
    let weights = GpuBuffer::zeros_f32(&device, 3 * 64 * slots)?;
    let output = GpuBuffer::zeros_f32(&device, 3 * 64 * 512)?;
    let wk = vulkan::ComputeKernel::new(
        &device,
        ATTENTION_WEIGHTS,
        7,
        std::mem::size_of::<WeightPush>() as u32,
    )?;
    let vk = vulkan::ComputeKernel::new(
        &device,
        ATTENTION_VALUES,
        6,
        std::mem::size_of::<ValuePush>() as u32,
    )?;
    let wp = WeightPush {
        rows: 3,
        batch: 1,
        seq: 3,
        heads: 64,
        dim: 512,
        window_capacity: capacity as u32,
        window_topk: slots as u32,
        compressed_topk: 0,
        compressed_capacity: 0,
        scale: 512f32.powf(-0.5),
    };
    let vp = ValuePush {
        rows: 3,
        seq: 3,
        heads: 64,
        dim: 512,
        window_capacity: capacity as u32,
        window_topk: slots as u32,
        compressed_topk: 0,
        compressed_capacity: 0,
    };
    let mut commands = vulkan::ComputeBatch::new(&device)?;
    wk.record_dispatch(
        &mut commands,
        &[&q, &kv, &empty, &idx, &empty, &sink, &weights],
        bytemuck::bytes_of(&wp),
        [1, 64, 3],
    )?;
    vk.record_dispatch(
        &mut commands,
        &[&kv, &empty, &idx, &empty, &weights, &output],
        bytemuck::bytes_of(&vp),
        [8, 64, 3],
    )?;
    commands.submit()?;
    let cast = DeepseekV41MixedLinearKernel::new(device.clone())?;
    let actual_weights = weights.read_u32(3 * 64 * slots)?;
    let raw = output.read_u32(3 * 64 * 512)?;
    let rounded = cast
        .cast_buffer(&output, raw.len(), DeepseekV41WeightOutput::Bf16)?
        .read_u32(raw.len())?;
    let count = |a: &[u32], b: &[u32]| {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).filter(|(x, y)| x != y).count()
    };
    let mut failures = vec![];
    for (name, actual, key) in [
        ("weights", &actual_weights, "weight_bits"),
        ("raw", &raw, "raw_bits"),
        ("rounded", &rounded, "rounded_bits"),
    ] {
        let different = count(actual, &bits(&s[key]));
        eprintln!("scheduled sparse {name}: {different} mismatches");
        if different > 0 {
            failures.push(format!("sparse {name}: {different}"));
        }
    }
    let g = &f["grouped"];
    let mut w = vec![0u32; 8 * 1024 * 4096];
    for row in g["weight_rows"].as_array().unwrap() {
        let group = row["group"].as_u64().unwrap() as usize;
        let r = row["row"].as_u64().unwrap() as usize;
        let start = (group * 1024 + r) * 4096;
        w[start..start + 4096].copy_from_slice(&bits(&row["bits"]));
    }
    let w = GpuBuffer::from_u32(&device, &w)?;
    let kernel = vulkan::ComputeKernel::new(
        &device,
        GROUPED_OUTPUT,
        3,
        std::mem::size_of::<GroupedPush>() as u32,
    )?;
    let mut grouped = vec![];
    for step in g["steps"].as_array().unwrap() {
        let rows = step["rows"].as_u64().unwrap() as usize;
        let x = GpuBuffer::from_u32(&device, &bits(&step["input_bits"]))?;
        let out = GpuBuffer::zeros_f32(&device, rows * 8 * 1024)?;
        let push = GroupedPush {
            rows: rows as u32,
            groups: 8,
            group_width: 4096,
            output_rank: 1024,
            bf16: 1,
        };
        let mut commands = vulkan::ComputeBatch::new(&device)?;
        kernel.record_dispatch(
            &mut commands,
            &[&x, &w, &out],
            bytemuck::bytes_of(&push),
            [16, 8, rows as u32],
        )?;
        commands.submit()?;
        let actual = cast
            .cast_buffer(&out, rows * 8 * 1024, DeepseekV41WeightOutput::Bf16)?
            .read_u32(rows * 8 * 1024)?;
        let different = count(&actual, &bits(&step["expected_bits"]));
        eprintln!("scheduled grouped rows {rows}: {different} mismatches");
        if different > 0 {
            failures.push(format!("grouped rows {rows}: {different}"));
        }
        grouped.push(serde_json::json!({"rows":rows,"bits":actual}));
    }
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../target/deepseek-v41-reference/schedule-attention-native-components.json");
    std::fs::write(
        path,
        serde_json::to_vec(
            &serde_json::json!({"weights":actual_weights,"raw":raw,"rounded":rounded,"grouped":grouped}),
        )?,
    )?;
    assert!(failures.is_empty(), "{failures:?}");
    Ok(())
}
