use super::*;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    config: DeepseekV41VisionConfig,
    image: Image,
    patches: DeepseekV41VisionTensor,
    upstream: DeepseekV41VisionTensor,
    parameters: Tensors,
    hyper: Hyper,
    steps: Vec<Step>,
}
#[derive(Deserialize)]
struct Image {
    n_vit_h: usize,
    n_vit_w: usize,
}
#[derive(Deserialize)]
struct Hyper {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
}
#[derive(Deserialize)]
struct Step {
    features: DeepseekV41VisionTensor,
    projected: DeepseekV41VisionTensor,
    patch_gradient: Option<DeepseekV41VisionTensor>,
    gradients: Option<Tensors>,
    updated_parameters: Option<Tensors>,
}
fn compare(name: &str, actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len(), "{name} length");
    assert!(!actual.is_empty());
    let worst = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| {
            if a.is_finite() && e.is_finite() {
                (a - e).abs()
            } else {
                f32::INFINITY
            }
        })
        .fold(0., f32::max);
    println!("{name}: {worst:e}");
    worst
}

#[test]
fn deepseek_v41_native_vision_final_rms_adjoint_matches_frozen_words() -> Result<()> {
    let bytes =
        include_bytes!("../../validation/fixtures/vision_avx2_row_tail/deepseek_v41_rms.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "a994f828919639f776fb5069cbc68c636d04e93b76ca79eda56966c68400ff89"
    );
    let document: serde_json::Value = serde_json::from_slice(bytes)?;
    let device = VulkanDevice::new()?;
    let kernel = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_deepseek_v41_rms_input_grad.spv"),
        5,
        8,
    )?;
    let cases = document["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 8);
    let mut words = 0;
    for (index, c) in cases.iter().enumerate() {
        let floats = |name: &str| {
            c[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let rows = c["rows"].as_u64().unwrap() as usize;
        let n = rows * 16;
        let buffers = [
            floats("dy"),
            floats("x"),
            floats("weight"),
            floats("rstd"),
            vec![0.; n],
        ]
        .into_iter()
        .map(|v| GpuBuffer::from_f32(&device, &v))
        .collect::<Result<Vec<_>>>()?;
        let mut b = vulkan::ComputeBatch::new(&device)?;
        kernel.record_dispatch(
            &mut b,
            &buffers.iter().collect::<Vec<_>>(),
            bytemuck::cast_slice(&[rows as u32, 16]),
            [1, 1, 1],
        )?;
        b.submit()?;
        let actual = buffers[4].read_f32(n)?;
        let expected = floats("dx");
        assert_eq!(actual.len(), expected.len());
        assert_eq!(
            actual
                .iter()
                .zip(expected)
                .filter(|(a, e)| a.to_bits() != e.to_bits())
                .count(),
            0,
            "native final RMS case {index}"
        );
        words += n;
    }
    assert_eq!(words, 640);
    println!("V4.1 final vision RMS: {words} frozen CPU words exact");
    Ok(())
}

#[test]
fn deepseek_v41_native_vision_aligner_linears_match_frozen_words() -> Result<()> {
    let bytes = include_bytes!(
        "../../validation/fixtures/vision_avx2_row_tail/deepseek_v41_aligner_linear.json"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "f4cbcf68a9940213982bcf42a92e64ce109cfbe815fd94fb5add97e81afe8d2c"
    );
    let document: serde_json::Value = serde_json::from_slice(bytes)?;
    let device = VulkanDevice::new()?;
    let gemv = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_deepseek_v41_gemv.spv"),
        4,
        16,
    )?;
    let gemm = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_linear_cpu_avx2.spv"),
        4,
        16,
    )?;
    let backward = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_deepseek_v41_aligner_input_grad.spv"),
        3,
        12,
    )?;
    let cases = document["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 16);
    let mut words = 0;
    for (index, c) in cases.iter().enumerate() {
        let floats = |name: &str| {
            c[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let rows = c["rows"].as_u64().unwrap() as usize;
        let input = c["input_dim"].as_u64().unwrap() as usize;
        let output = 32;
        let x = GpuBuffer::from_f32(&device, &floats("x"))?;
        let w = GpuBuffer::from_f32(&device, &floats("weight"))?;
        let bias = GpuBuffer::from_f32(&device, &floats("bias"))?;
        let dy = GpuBuffer::from_f32(&device, &floats("dy"))?;
        let y = GpuBuffer::zeros_f32(&device, rows * output)?;
        let dx = GpuBuffer::zeros_f32(&device, rows * input)?;
        let mut b = vulkan::ComputeBatch::new(&device)?;
        if rows == 1 {
            gemv.record_dispatch(
                &mut b,
                &[&x, &w, &bias, &y],
                bytemuck::cast_slice(&[input as u32, 32, 0, 1]),
                [1, 1, 1],
            )?;
        } else {
            gemm.record_dispatch(
                &mut b,
                &[&x, &w, &bias, &y],
                bytemuck::cast_slice(&[rows as u32, input as u32, 32, 1]),
                [2, 1, 1],
            )?;
        }
        backward.record_dispatch(
            &mut b,
            &[&dy, &w, &dx],
            bytemuck::cast_slice(&[rows as u32, input as u32, 32]),
            [div_ceil_u32(input, 16), 1, 1],
        )?;
        b.submit()?;
        for (name, buffer, n) in [("output", &y, rows * output), ("dx", &dx, rows * input)] {
            let actual = buffer.read_f32(n)?;
            let expected = floats(name);
            assert_eq!(actual.len(), expected.len());
            assert_eq!(
                actual
                    .iter()
                    .zip(&expected)
                    .filter(|(a, e)| a.to_bits() != e.to_bits())
                    .count(),
                0,
                "aligner case{index} {name}"
            );
            words += n;
        }
    }
    assert_eq!(words, 1920);
    println!("V4.1 native aligner Linear: {words} independent frozen words exact");
    Ok(())
}

#[test]
fn deepseek_v41_native_vision_sdpa_matches_independent_random_cpu() -> Result<()> {
    let bytes =
        include_bytes!("../../validation/fixtures/vision_avx2_row_tail/deepseek_v41_sdpa.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "2c3eb0bb951cfd1a9a3f8023248d019aa39b56e06f9edb5681ae5da93a8211ae"
    );
    let document: serde_json::Value = serde_json::from_slice(bytes)?;
    let device = VulkanDevice::new()?;
    let forward = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_deepseek_v41_attention.spv"),
        4,
        16,
    )?;
    let backward = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/vision_deepseek_v41_attention_backward.spv"),
        7,
        16,
    )?;
    let cases = document["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 8);
    let mut words = 0;
    for (index, case) in cases.iter().enumerate() {
        let floats = |name: &str| {
            case[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let rows = case["rows"].as_u64().unwrap() as usize;
        let n = rows * 16;
        let q = GpuBuffer::from_f32(&device, &floats("q"))?;
        let k = GpuBuffer::from_f32(&device, &floats("k"))?;
        let v = GpuBuffer::from_f32(&device, &floats("v"))?;
        let dy = GpuBuffer::from_f32(&device, &floats("dy"))?;
        let output = GpuBuffer::zeros_f32(&device, n)?;
        let dq = GpuBuffer::zeros_f32(&device, n)?;
        let dk = GpuBuffer::zeros_f32(&device, n)?;
        let dv = GpuBuffer::zeros_f32(&device, n)?;
        let push = [rows as u32, 16, 2, ((1. / 8f64).sqrt() as f32).to_bits()];
        let mut b = vulkan::ComputeBatch::new(&device)?;
        forward.record_dispatch(
            &mut b,
            &[&q, &k, &v, &output],
            bytemuck::cast_slice(&push),
            [rows as u32, 2, 1],
        )?;
        backward.record_dispatch(
            &mut b,
            &[&q, &k, &v, &dy, &dq, &dk, &dv],
            bytemuck::cast_slice(&push),
            [2, 1, 1],
        )?;
        b.submit()?;
        for (name, buffer) in [("output", &output), ("dq", &dq), ("dk", &dk), ("dv", &dv)] {
            let expected = floats(name);
            let actual = buffer.read_f32(n)?;
            assert!(
                compare(
                    &format!("V4.1 random SDPA {index} {name}"),
                    &actual,
                    &expected
                ) < 2e-7
            );
            words += actual.len();
        }
    }
    assert_eq!(words, 2560);
    Ok(())
}
#[test]
fn deepseek_v41_native_vision_two_adamw_steps_match_frozen_cpu() -> Result<()> {
    let bytes = include_bytes!("../../validation/fixtures/deepseek_v41_native_vision_avx2.json");
    let hash = format!("{:x}", Sha256::digest(bytes));
    assert_eq!(
        hash,
        "f6bb7f095bb829f1124fd528ed79d36d12e8fc2fe041a89734c9a2ab4cdfc439"
    );
    // The producer, all released source hashes, and the immutable fixture are
    // recorded together; this gate never imports a CPU model or rewrites it.
    println!("V4.1 native vision frozen fixture {hash}");
    let fixture: Fixture = serde_json::from_slice(bytes)?;
    assert_eq!(fixture.cases.len(), 2);
    let device = VulkanDevice::new()?;
    let mut errors = Vec::new();
    let mut tensors = 0;
    for case in fixture.cases {
        let mut net = VulkanDeepseekV41Vision::from_tensors(
            device.clone(),
            case.config,
            case.image.n_vit_h,
            case.image.n_vit_w,
            &case.parameters,
        )?;
        let gradient = GpuBuffer::from_f32(&device, &case.upstream.values)?;
        assert_eq!(case.steps.len(), 3);
        for (step, reference) in case.steps.into_iter().enumerate() {
            net.forward(&case.patches.values)?;
            let stages = net.diagnostic_stages()?;
            let output = serde_json::json!({"stages":stages,"parameters":net.parameters()?});
            let debug_dir = std::env::var_os("HIERARCHOS_V41_VISION_DIAGNOSTICS");
            if let Some(dir) = debug_dir.as_ref() {
                fs::create_dir_all(dir)?;
                fs::write(
                    Path::new(dir).join(format!("{}-step{step}-forward.json", case.name)),
                    serde_json::to_vec(&output)?,
                )?;
            }
            for (name, actual, expected) in [
                ("features", &stages["features"], &reference.features.values),
                (
                    "projected",
                    &stages["projected"],
                    &reference.projected.values,
                ),
            ] {
                let label = format!("{} step{step} {name}", case.name);
                if compare(&label, actual, expected) >= 2e-7 {
                    errors.push(label);
                }
            }
            if step < 2 {
                net.backward(&gradient)?;
                let gradients = net.gradients()?;
                if let Some(dir) = debug_dir.as_ref() {
                    fs::write(
                        Path::new(dir).join(format!("{}-step{step}-backward.json", case.name)),
                        serde_json::to_vec(
                            &serde_json::json!({"stages":net.diagnostic_stages()?,"gradients":gradients}),
                        )?,
                    )?;
                }
                let label = format!("{} step{step} patch gradient", case.name);
                if compare(
                    &label,
                    &net.patch_gradient().read_f32(case.patches.values.len())?,
                    &reference.patch_gradient.unwrap().values,
                ) >= 2e-7
                {
                    errors.push(label);
                }
                let expected = reference.gradients.unwrap();
                assert_eq!(gradients.len(), expected.len());
                for (name, t) in &gradients {
                    let label = format!("{} step{step} gradient {name}", case.name);
                    if compare(&label, &t.values, &expected[name].values) >= 2e-7 {
                        errors.push(label);
                    }
                    tensors += 1;
                }
                let h = &case.hyper;
                net.apply_gradients(crate::CpuAdamWHyperParams {
                    lr: h.learning_rate,
                    beta1: h.beta1,
                    beta2: h.beta2,
                    eps: h.eps,
                    weight_decay: h.weight_decay,
                })?;
                let actual = net.parameters()?;
                let expected = reference.updated_parameters.unwrap();
                assert_eq!(actual.len(), expected.len());
                for (name, t) in actual {
                    let label = format!("{} step{step} updated {name}", case.name);
                    if compare(&label, &t.values, &expected[&name].values) >= 2e-7 {
                        errors.push(label);
                    }
                    tensors += 1;
                }
            }
        }
    }
    assert_eq!(
        tensors, 184,
        "must compare every gradient and parameter after both updates"
    );
    assert!(
        errors.is_empty(),
        "V4.1 native vision failed strict <2e-7: {errors:?}"
    );
    Ok(())
}
