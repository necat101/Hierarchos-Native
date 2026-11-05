use super::*;
use crate::deepseek_v41::DeepseekV41MultimodalTrainer;
#[path = "native_vision_backward_tests.rs"]
mod backward_primitives;
#[path = "native_vision_state_tests.rs"]
mod native_state;

#[test]
fn deepseek_v41_native_avx2_terminal_matches_frozen_words() -> Result<()> {
    let _math = crate::deepseek_v41::cpu_math::avx2();
    let bytes = include_bytes!(
        "../../validation/fixtures/vision_avx2_row_tail/deepseek_v41_native_terminal.json"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "3191595dbd4be397e1fea0f2476634e676afd92fb087b7f0e7a00a3558866cfd"
    );
    let corpus: Value = serde_json::from_slice(bytes)?;
    let cases = corpus["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 6);
    let device = VulkanDevice::new()?;
    let terminal = crate::deepseek_v41::DeepseekV41TerminalAdjoints::new(device.clone())?;
    let kernel = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/transformer_deepseek_v41_head_input_grad_avx2.spv"),
        3,
        16,
    )?;
    let mut count = 0;
    for (index, c) in cases.iter().enumerate() {
        let vector = |name: &str| {
            c[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let rows = c["rows"].as_u64().unwrap() as usize;
        let targets = c["targets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| {
                let t = x.as_i64().unwrap();
                if t < 0 {
                    u32::MAX
                } else {
                    t as u32
                }
            })
            .collect::<Vec<_>>();
        let logits = GpuBuffer::from_f32(&device, &vector("logits"))?;
        let (loss, gradient) = terminal.masked_cross_entropy(&logits, &targets, 16)?;
        assert_eq!(
            loss.to_bits(),
            (c["loss"].as_f64().unwrap() as f32).to_bits(),
            "case {index} loss"
        );
        assert_eq!(
            gradient.read_u32(rows * 16)?,
            bits(&vector("gradient")),
            "case {index} logit VJP"
        );
        let weight = GpuBuffer::from_f32(&device, &vector("head_weight"))?;
        let head = GpuBuffer::zeros_f32(&device, rows * 32)?;
        let mut commands = device.compute_batch()?;
        kernel.record_dispatch(
            &mut commands,
            &[&gradient, &weight, &head],
            bytemuck::cast_slice(&[rows as u32, 32, 16, 0]),
            [2, 1, 1],
        )?;
        commands.submit()?;
        assert_eq!(
            head.read_u32(rows * 32)?,
            bits(&vector("head_gradient")),
            "case {index} head VJP"
        );
        count += rows * 48;
    }
    assert_eq!(count, 2448);
    Ok(())
}

#[test]
fn deepseek_v41_native_avx2_unary_boundaries_match_frozen_words() -> Result<()> {
    let _math = crate::deepseek_v41::cpu_math::avx2();
    let bytes = include_bytes!(
        "../../validation/fixtures/vision_avx2_row_tail/deepseek_v41_unary_avx2.json"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "ad1a798fbe1dda879509e2bb41936858cbdf4d9cef94cc1e38354fa6c50032cc"
    );
    let corpus: Value = serde_json::from_slice(bytes)?;
    let cases = corpus["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 44);
    let device = VulkanDevice::new()?;
    let kernel = vulkan::ComputeKernel::new(
        &device,
        crate::deepseek_v41::gen9::shader(
            &device,
            include_bytes!("../../shaders/transformer_deepseek_v41_unary_test.spv"),
        ),
        3,
        8,
    )?;
    let dummy = GpuBuffer::zeros_f32(&device, 1)?;
    let mut count = 0;
    for (index, case) in cases.iter().enumerate() {
        let input: Vec<f32> = case["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let expected: Vec<u32> = case["expected_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let x = GpuBuffer::from_f32(&device, &input)?;
        let output = GpuBuffer::zeros_f32(&device, expected.len())?;
        let mut commands = device.compute_batch()?;
        kernel.record_dispatch(
            &mut commands,
            &[&x, &output, &dummy],
            bytemuck::cast_slice(&[input.len() as u32, case["op"].as_u64().unwrap() as u32]),
            [input.len().div_ceil(64) as u32, 1, 1],
        )?;
        commands.submit()?;
        assert_eq!(output.read_u32(expected.len())?, expected, "case={index}");
        count += expected.len();
    }
    assert_eq!(count, 1396);
    let bytes = include_bytes!(
        "../../validation/fixtures/vision_avx2_row_tail/deepseek_v41_sqrt_avx2.json"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "e6ce075bc6240376c918953441ec8f2a2fabf3d699aec431063571dd557495d0"
    );
    let corpus: Value = serde_json::from_slice(bytes)?;
    let input: Vec<f32> = corpus["input"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    let expected: Vec<u32> = corpus["expected_bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(input.len(), 4096);
    let x = GpuBuffer::from_f32(&device, &input)?;
    let output = GpuBuffer::zeros_f32(&device, input.len())?;
    let mut commands = device.compute_batch()?;
    kernel.record_dispatch(
        &mut commands,
        &[&x, &output, &dummy],
        bytemuck::cast_slice(&[4096u32, 5]),
        [64, 1, 1],
    )?;
    commands.submit()?;
    assert_eq!(output.read_u32(4096)?, expected, "Intel AVX2 sqrt");
    Ok(())
}

#[test]
fn deepseek_v41_native_avx2_projection_panels_match_frozen_words() -> Result<()> {
    let _math = crate::deepseek_v41::cpu_math::avx2();
    let device = VulkanDevice::new()?;
    let mut count = 0;
    for (grouped, bytes, hash, shader, expected_cases) in [
        (
            false,
            include_bytes!(
                "../../validation/fixtures/vision_avx2_row_tail/deepseek_v41_language_linear.json"
            )
            .as_slice(),
            "062d0c3cad26665f7a750b215f6114778111e1eeb02e71c476a6b5a85a72667d",
            include_bytes!("../../shaders/transformer_deepseek_v41_linear.spv").as_slice(),
            154,
        ),
        (
            true,
            include_bytes!(
                "../../validation/fixtures/vision_avx2_row_tail/deepseek_v41_grouped_avx2.json"
            )
            .as_slice(),
            "c3d1106cf549751e78a890447ab352e64f5915ae374cc32c2994183d68e6b119",
            include_bytes!("../../shaders/transformer_deepseek_v41_grouped_output.spv").as_slice(),
            44,
        ),
    ] {
        assert_eq!(format!("{:x}", Sha256::digest(bytes)), hash);
        let corpus: Value = serde_json::from_slice(bytes)?;
        let cases = corpus["cases"].as_array().unwrap();
        assert_eq!(cases.len(), expected_cases);
        let kernel = vulkan::ComputeKernel::new(
            &device,
            crate::deepseek_v41::gen9::shader(&device, shader),
            3,
            if grouped { 20 } else { 16 },
        )?;
        for (index, case) in cases.iter().enumerate() {
            let number = |name: &str| case[name].as_u64().unwrap() as u32;
            let vector = |name: &str| {
                case[name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap() as f32)
                    .collect::<Vec<_>>()
            };
            let expected: Vec<u32> = case["expected_bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect();
            let x = GpuBuffer::from_f32(&device, &vector("input"))?;
            let weight = GpuBuffer::from_f32(&device, &vector("weight"))?;
            let output = GpuBuffer::zeros_f32(&device, expected.len())?;
            let rows = number("rows");
            let (push, grid) = if grouped {
                let groups = number("groups");
                let rank = number("output_rank");
                (
                    vec![rows, groups, number("group_width"), rank, 0],
                    [rank.div_ceil(64), groups, rows],
                )
            } else {
                let width = number("output_dim");
                (
                    vec![rows, number("input_dim"), width, 0],
                    [width.div_ceil(16), rows.div_ceil(16), 1],
                )
            };
            let mut commands = device.compute_batch()?;
            kernel.record_dispatch(
                &mut commands,
                &[&x, &weight, &output],
                bytemuck::cast_slice(&push),
                grid,
            )?;
            commands.submit()?;
            assert_eq!(
                output.read_u32(expected.len())?,
                expected,
                "grouped={grouped}, case={index}"
            );
            count += expected.len();
        }
    }
    assert_eq!(count, 36_344);
    Ok(())
}

#[test]
fn deepseek_v41_native_avx2_narrow_projections_match_frozen_cpu_tolerance() -> Result<()> {
    let _math = crate::deepseek_v41::cpu_math::avx2();
    let bytes = include_bytes!(
        "../../validation/fixtures/vision_avx2_row_tail/deepseek_v41_narrow_avx2.json"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "06a6bb787d7faf97a2a172135ebb18169be8c4f5cec7a7d2df300d6fa8ac316a"
    );
    let corpus: Value = serde_json::from_slice(bytes)?;
    let cases = corpus["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 200);
    let device = VulkanDevice::new()?;
    let kernel = vulkan::ComputeKernel::new(
        &device,
        crate::deepseek_v41::gen9::shader(
            &device,
            include_bytes!("../../shaders/transformer_deepseek_v41_linear.spv"),
        ),
        3,
        16,
    )?;
    let mut total = Tally::default();
    let mut failures = Vec::new();
    let mut count = 0;
    for (index, case) in cases.iter().enumerate() {
        let vector = |name: &str| {
            case[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let number = |name: &str| case[name].as_u64().unwrap() as u32;
        let expected: Vec<u32> = case["expected_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let x = GpuBuffer::from_f32(&device, &vector("input"))?;
        let w = GpuBuffer::from_f32(&device, &vector("weight"))?;
        let output = GpuBuffer::zeros_f32(&device, expected.len())?;
        let mut commands = device.compute_batch()?;
        kernel.record_dispatch(
            &mut commands,
            &[&x, &w, &output],
            bytemuck::cast_slice(&[number("rows"), number("input_dim"), number("output_dim"), 0]),
            [1, 1, 1],
        )?;
        commands.submit()?;
        assert_eq!(
            output.read_u32(expected.len())?,
            expected,
            "narrow projection words {index}"
        );
        total.add(full_gate(
            &format!("narrow-{index}"),
            &output.read_f32(expected.len())?,
            &expected,
            &mut failures,
        ));
        count += expected.len();
    }
    assert_eq!(count, 2850);
    eprintln!("V4.1 narrow AVX2 projections: {}", total.json());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

fn floats(v: &Value) -> Vec<f32> {
    v["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}
fn check(
    label: &str,
    actual: &[f32],
    expected: &Value,
    failures: &mut Vec<String>,
    total: &mut Tally,
) {
    let expected = floats(expected);
    assert!(!expected.is_empty());
    total.add(full_gate(label, actual, &bits(&expected), failures));
}

/// Isolate inference arithmetic from optimizer propagation. This diagnostic
/// loads CPU states into separate fresh models; the joint gate below always
/// continues its own native gradients and updates without state pinning.
#[test]
fn deepseek_v41_native_cpu_state_forward_diagnostic() -> Result<()> {
    let bytes = include_bytes!("../../validation/fixtures/deepseek_v41_native_pipeline_avx2.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "9f5592989820c1567caf3fd4ca6b5354f7bb8e78dfb6b38f1ebb43b9b4cc403b"
    );
    let cpu: Value = serde_json::from_slice(bytes)?;
    let device = VulkanDevice::new()?;
    let mut tally = Tally::default();
    let mut failures = Vec::new();
    let mut values = 0;
    for case in cpu["cases"].as_array().unwrap() {
        for state in 0..3 {
            let mut f = fixture()?;
            f.config = case["config"].clone();
            f.batch_size = 1;
            let image = if state == 0 {
                &case["image_parameters"]
            } else {
                &case["steps"][state - 1]["updated_image_parameters"]
            };
            for (name, t) in image.as_object().unwrap() {
                f.tensors.insert(
                    name.clone(),
                    Tensor {
                        shape: t["shape"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|x| x.as_u64().unwrap() as usize)
                            .collect(),
                        values: floats(t),
                        dtype: String::new(),
                    },
                );
            }
            for layer in 0..6 {
                f.tensors.insert(
                    format!("layers.{layer}.ffn.gate.bias_vl"),
                    Tensor {
                        shape: vec![3],
                        values: case["gate_bias_vl"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|x| x.as_f64().unwrap() as f32)
                            .collect(),
                        dtype: String::new(),
                    },
                );
            }
            let package = Package::new(&f, false)?;
            fs::write(
                package.root.join("engram_tokens.json"),
                serde_json::to_vec(&f.tokens)?,
            )?;
            let mut reader = package.reader()?;
            let mut adapter =
                DeepseekV41LoraAdapter::initialize(&mut reader, &[], 2, 2., false, 1793)?;
            if state > 0 {
                let previous = &case["steps"][state - 1]["updated_adapters"];
                adapter = adapter.with_factors(|target| {
                    previous
                        .get(target)
                        .map(|t| (floats(&t["a"]), floats(&t["b"])))
                })?;
            }
            let mut trainer = DeepseekV41MultimodalTrainer::new(
                device.clone(),
                &package.root,
                &adapter,
                serde_json::from_value(case["hyper"].clone())?,
                DeepseekV41ModelTrainingConfig::default(),
                case["image"]["n_vit_h"].as_u64().unwrap() as usize,
                case["image"]["n_vit_w"].as_u64().unwrap() as usize,
            )?;
            let batch = DeepseekV41TrainingBatch {
                token_ids: case["token_ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u32)
                    .collect(),
                targets: case["targets"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| {
                        if x.as_i64().unwrap() < 0 {
                            u32::MAX
                        } else {
                            x.as_u64().unwrap() as u32
                        }
                    })
                    .collect(),
                seq_len: case["seq_len"].as_u64().unwrap() as usize,
                engram_mask: None,
            };
            let types = case["token_types"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect::<Vec<_>>();
            let output = trainer
                .forward_logits(
                    &batch,
                    &floats(&case["patches"]),
                    &types,
                    case["image_start"].as_u64().unwrap() as usize,
                )?
                .read_all()?;
            check(
                &format!(
                    "CPU-state-forward/{}/{state}",
                    case["name"].as_str().unwrap()
                ),
                &output,
                &case["steps"][state]["logits"],
                &mut failures,
                &mut tally,
            );
            values += output.len();
        }
    }
    assert_eq!(values, 816);
    assert_eq!(tally.comparisons, 6);
    eprintln!("Native CPU-state forward diagnostic: {}", tally.json());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn deepseek_v41_native_multimodal_joint_two_adamw_steps_match_frozen_cpu() -> Result<()> {
    let bytes = include_bytes!("../../validation/fixtures/deepseek_v41_native_pipeline_avx2.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "9f5592989820c1567caf3fd4ca6b5354f7bb8e78dfb6b38f1ebb43b9b4cc403b"
    );
    let cpu: Value = serde_json::from_slice(bytes)?;
    assert_eq!(cpu["tolerance"].as_f64(), Some(2e-7));
    assert_eq!(cpu["rtol"].as_u64(), Some(0));
    assert_eq!(cpu["comparison"], "strict_less_than");
    let cases = cpu["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    let device = VulkanDevice::new()?;
    let mut failures = Vec::new();
    let mut total = Tally::default();
    let mut updates = 0;
    let mut logit_values = 0;
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let mut f = fixture()?;
        f.config = case["config"].clone();
        f.batch_size = case["batch_size"].as_u64().unwrap() as usize;
        for (name, t) in case["image_parameters"].as_object().unwrap() {
            f.tensors.insert(
                name.clone(),
                Tensor {
                    shape: t["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as usize)
                        .collect(),
                    values: floats(t),
                    dtype: String::new(),
                },
            );
        }
        for layer in 0..f.config["text_config"]["num_hidden_layers"]
            .as_u64()
            .unwrap()
        {
            f.tensors.insert(
                format!("layers.{layer}.ffn.gate.bias_vl"),
                Tensor {
                    shape: vec![3],
                    values: case["gate_bias_vl"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap() as f32)
                        .collect(),
                    dtype: String::new(),
                },
            );
        }
        let package = Package::new(&f, false)?;
        fs::write(
            package.root.join("engram_tokens.json"),
            serde_json::to_vec(&f.tokens)?,
        )?;
        let mut reader = package.reader()?;
        let adapter = DeepseekV41LoraAdapter::initialize(&mut reader, &[], 2, 2., false, 1793)?;
        let specifications: BTreeMap<String, TrainingModule> =
            serde_json::from_value(case["adapters"].clone())?;
        assert!(!specifications.is_empty());
        for (name, m) in &specifications {
            let native = adapter
                .module_for_tensor(&format!("{name}.weight"))
                .unwrap();
            assert_eq!(bits(&native.a), m.a_bits);
            assert_eq!(bits(&native.b), m.b_bits);
        }
        let hyper: DeepseekV41AdamWConfig = serde_json::from_value(case["hyper"].clone())?;
        let options = DeepseekV41ModelTrainingConfig {
            batch_size: f.batch_size,
            ..Default::default()
        };
        let mut trainer = DeepseekV41MultimodalTrainer::new(
            device.clone(),
            &package.root,
            &adapter,
            hyper,
            options,
            case["image"]["n_vit_h"].as_u64().unwrap() as usize,
            case["image"]["n_vit_w"].as_u64().unwrap() as usize,
        )?;
        let batch = DeepseekV41TrainingBatch {
            token_ids: case["token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect(),
            targets: case["targets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| {
                    if v.as_i64().unwrap() == -100 {
                        u32::MAX
                    } else {
                        v.as_u64().unwrap() as u32
                    }
                })
                .collect(),
            seq_len: case["seq_len"].as_u64().unwrap() as usize,
            engram_mask: None,
        };
        let types = case["token_types"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect::<Vec<_>>();
        let patches = floats(&case["patches"]);
        let start = case["image_start"].as_u64().unwrap() as usize;
        let steps = case["steps"].as_array().unwrap();
        assert_eq!(steps.len(), 3);
        for (step, expected) in steps.iter().enumerate() {
            let actual = trainer
                .forward_logits(&batch, &patches, &types, start)?
                .read_all()?;
            logit_values += actual.len();
            check(
                &format!("{name}/{step}/inference-logits"),
                &actual,
                &expected["logits"],
                &mut failures,
                &mut total,
            );
            if step < 2 {
                if std::env::var_os("HIERARCHOS_V41_NATIVE_DIAGNOSTICS").is_some() {
                    crate::deepseek_v41::training_trace::begin();
                }
                let gradients = trainer.gradients(&batch, &patches, &types, start)?;
                if let Some(dir) = std::env::var_os("HIERARCHOS_V41_NATIVE_DIAGNOSTICS") {
                    fs::create_dir_all(&dir)?;
                    fs::write(
                        Path::new(&dir).join(format!(
                            "rows{}-step{step}-backward.json",
                            batch.token_ids.len()
                        )),
                        serde_json::to_vec(&crate::deepseek_v41::training_trace::finish())?,
                    )?;
                }
                assert_eq!(gradients.text.logits.len(), 1);
                let actual = gradients.text.logits[0].read_all()?;
                logit_values += actual.len();
                check(
                    &format!("{name}/{step}/training-logits"),
                    &actual,
                    &expected["logits"],
                    &mut failures,
                    &mut total,
                );
                total.add(full_gate(
                    &format!("{name}/{step}/loss"),
                    &[gradients.text.loss],
                    &[(expected["loss"].as_f64().unwrap() as f32).to_bits()],
                    &mut failures,
                ));
                check(
                    &format!("{name}/{step}/patch-gradient"),
                    &trainer.image().patch_gradient()?,
                    &expected["patch_gradient"],
                    &mut failures,
                    &mut total,
                );
                check(
                    &format!("{name}/{step}/input-gradient"),
                    &gradients
                        .input_gradient
                        .read_f32(floats(&expected["stream_gradient"]).len())?,
                    &expected["stream_gradient"],
                    &mut failures,
                    &mut total,
                );
                check(
                    &format!("{name}/{step}/injected-stream"),
                    &trainer.image().injected_stream()?,
                    &expected["injected_stream"],
                    &mut failures,
                    &mut total,
                );
                let native = trainer.image().gradients()?;
                let reference = expected["image_gradients"].as_object().unwrap();
                assert_eq!(native.len(), 26);
                assert_eq!(native.len(), reference.len());
                for (key, t) in native {
                    check(
                        &format!("{name}/{step}/image-gradient/{key}"),
                        &t.values,
                        &reference[&key],
                        &mut failures,
                        &mut total,
                    );
                }
                let native = gradients
                    .text
                    .adapters
                    .iter()
                    .map(|g| (g.target.as_str(), g))
                    .collect::<BTreeMap<_, _>>();
                let reference = expected["adapter_gradients"].as_object().unwrap();
                assert_eq!(
                    native.len(),
                    reference.values().filter(|v| !v.is_null()).count()
                );
                assert!(!native.is_empty());
                for (key, t) in reference {
                    if t.is_null() {
                        assert!(!native.contains_key(key.as_str()));
                        continue;
                    }
                    let g = native[key.as_str()];
                    check(
                        &format!("{name}/{step}/adapter-gradient/{key}/a"),
                        &g.grad_a,
                        &t["a"],
                        &mut failures,
                        &mut total,
                    );
                    check(
                        &format!("{name}/{step}/adapter-gradient/{key}/b"),
                        &g.grad_b,
                        &t["b"],
                        &mut failures,
                        &mut total,
                    );
                }
                trainer.apply_gradients(&gradients)?;
                updates += 1;
                assert_eq!(trainer.optimizer_step(), (step + 1) as u64);
                assert_eq!(trainer.image().step(), (step + 1) as u32);
                let native = trainer.image().parameters()?;
                let reference = expected["updated_image_parameters"].as_object().unwrap();
                assert_eq!(native.len(), reference.len());
                for (key, t) in native {
                    check(
                        &format!("{name}/{step}/image-updated/{key}"),
                        &t.values,
                        &reference[&key],
                        &mut failures,
                        &mut total,
                    );
                }
                let native = trainer.adapter()?;
                let reference = expected["updated_adapters"].as_object().unwrap();
                assert_eq!(reference.len(), specifications.len());
                for (key, t) in reference {
                    let g = native.module_for_tensor(&format!("{key}.weight")).unwrap();
                    check(
                        &format!("{name}/{step}/adapter-updated/{key}/a"),
                        &g.a,
                        &t["a"],
                        &mut failures,
                        &mut total,
                    );
                    check(
                        &format!("{name}/{step}/adapter-updated/{key}/b"),
                        &g.b,
                        &t["b"],
                        &mut failures,
                        &mut total,
                    );
                }
            }
        }
    }
    assert_eq!(updates, 4);
    assert_eq!(logit_values, 1360);
    assert!(total.comparisons > 1000);
    let result = json!({"pass":failures.is_empty(),"expected_cases":2,"completed_cases":2,"optimizer_updates":updates,"logit_values":logit_values,"all_comparisons":total.json(),"failures":failures,"scope":"joint native V4.1 image/aligner/delimiter and frozen-base LoRA training; export/reload not yet checked"});
    let output = std::env::var_os("HIERARCHOS_V41_JOINT_QUALIFICATION")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../target/multimodal-gen9/v41-native-joint-current/qualification.json")
        });
    fs::create_dir_all(output.parent().unwrap())?;
    fs::write(output, serde_json::to_vec_pretty(&result)?)?;
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}
