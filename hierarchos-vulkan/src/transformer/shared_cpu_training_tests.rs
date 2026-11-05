//! Independent frozen CPU contracts for the composed image training path.
use super::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

fn corpus(bytes: &[u8], sha256: &str) -> Result<Value> {
    assert_eq!(format!("{:x}", Sha256::digest(bytes)), sha256);
    let value: Value = serde_json::from_slice(bytes)?;
    assert_eq!(value["cpu"], "AVX2");
    assert_eq!(value["threads"], 1);
    Ok(value)
}

fn training_corpus() -> Result<Value> {
    corpus(
        include_bytes!("../../validation/fixtures/vision_avx2_row_tail/shared_training.json"),
        "b099d6e1124fa1284990dce01925fe3ef2970cf3cfec739deb53aaa0ae3b2d53",
    )
}

fn floats(value: &Value) -> Vec<f32> {
    serde_json::from_value(value.clone()).expect("frozen FP32 array")
}

fn words(value: &Value) -> Vec<u32> {
    serde_json::from_value(value.clone()).expect("frozen FP32 words")
}

fn replay(
    device: &VulkanDevice,
    shader: &[u8],
    inputs: Vec<Vec<f32>>,
    push: &[u32],
    dispatch: [u32; 3],
    outputs: &[(usize, Vec<u32>)],
    absolute_limit: Option<f32>,
) -> Result<usize> {
    let buffers = inputs
        .iter()
        .map(|values| GpuBuffer::from_f32(device, values))
        .collect::<Result<Vec<_>>>()?;
    let kernel =
        vulkan::ComputeKernel::new(device, shader, buffers.len(), (push.len() * 4) as u32)?;
    let mut commands = vulkan::ComputeBatch::new(device)?;
    kernel.record_dispatch(
        &mut commands,
        &buffers.iter().collect::<Vec<_>>(),
        bytemuck::cast_slice(push),
        dispatch,
    )?;
    commands.submit()?;
    let mut count = 0;
    for (binding, expected) in outputs {
        let actual = buffers[*binding].read_f32(expected.len())?;
        for (index, (&a, &bits)) in actual.iter().zip(expected).enumerate() {
            let b = f32::from_bits(bits);
            if let Some(limit) = absolute_limit {
                assert!(
                    a.is_finite() && (a - b).abs() < limit,
                    "binding {binding} word {index}: {a:e} != {b:e}"
                );
            } else {
                assert_eq!(
                    a.to_bits(),
                    bits,
                    "binding {binding} word {index}: {a:e} != {b:e}"
                );
            }
        }
        count += actual.len();
    }
    Ok(count)
}

#[test]
fn shared_cpu_flash_sdpa_forward_and_adjoint_match_independent_words() -> Result<()> {
    let data = corpus(
        include_bytes!("../../validation/fixtures/vision_avx2_row_tail/shared_text_sdpa.json"),
        "fdb71103a5548df29146148deaccf2821c51c0c51045b3e169c0e392b36fa63e",
    )?;
    let device = VulkanDevice::new()?;
    let cases = data["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 8);
    for c in cases {
        let (b, s, h, kh, d, window) = (
            c["batch"].as_u64().unwrap() as u32,
            c["seq"].as_u64().unwrap() as u32,
            c["heads"].as_u64().unwrap() as u32,
            c["kv_heads"].as_u64().unwrap() as u32,
            c["dim"].as_u64().unwrap() as u32,
            c["window"].as_u64().unwrap() as u32,
        );
        let n = (b * s * h * d) as usize;
        let k = (b * s * kh * d) as usize;
        let scale = (c["scale"].as_f64().unwrap() as f32).to_bits();
        let mut push = vec![
            b,
            s,
            s,
            h * d,
            kh * d,
            h,
            kh,
            window,
            scale,
            1f32.to_bits(),
            0,
            0,
            8f32.to_bits(),
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        replay(
            &device,
            include_bytes!("../../shaders/image_text_sdpa_cpu_avx2.spv"),
            vec![
                floats(&c["q"]),
                floats(&c["k"]),
                floats(&c["v"]),
                floats(&c["query_mask"]),
                floats(&c["key_mask"]),
                vec![0.],
                vec![0.; n],
                vec![0.],
                vec![0.],
                vec![0.],
                vec![0.],
            ],
            &push,
            [s, h, b],
            &[(6, words(&c["expected_forward_bits"]))],
            None,
        )?;
        push.extend([0, 1]);
        replay(
            &device,
            include_bytes!("../../shaders/image_text_sdpa_cpu_avx2_backward.spv"),
            vec![
                floats(&c["q"]),
                floats(&c["k"]),
                floats(&c["v"]),
                floats(&c["query_mask"]),
                floats(&c["key_mask"]),
                vec![0.],
                floats(&c["dy"]),
                vec![0.; n],
                vec![0.; k],
                vec![0.; k],
                vec![0.],
                vec![0.],
                vec![0.],
                vec![0.],
                vec![0.],
            ],
            &push,
            [kh, b, 1],
            &[
                (7, words(&c["expected_dq_bits"])),
                (8, words(&c["expected_dk_bits"])),
                (9, words(&c["expected_dv_bits"])),
            ],
            None,
        )?;
    }
    Ok(())
}

#[test]
fn shared_cpu_wide_norms_match_independent_words() -> Result<()> {
    let data = corpus(
        include_bytes!("../../validation/fixtures/vision_avx2_row_tail/shared_text_norm.json"),
        "b9cf177e825d0c2ad3830b6f68f33f526650a6f4a1b2fd6b0f777c800277286a",
    )?;
    let device = VulkanDevice::new()?;
    for c in data["cases"].as_array().unwrap() {
        let (x, w, b, dy, mean, rstd) = (
            floats(&c["x"]),
            floats(&c["weight"]),
            floats(&c["bias"]),
            floats(&c["dy"]),
            floats(&c["mean"]),
            floats(&c["rstd"]),
        );
        if c["kind"] == "layer_norm" {
            replay(
                &device,
                include_bytes!("../../shaders/vision_layer_norm_welford_forward.spv"),
                vec![
                    x.clone(),
                    w.clone(),
                    b,
                    vec![0.; 1024],
                    vec![0.; 16],
                    vec![0.; 16],
                ],
                &[16, 64, (c["eps"].as_f64().unwrap() as f32).to_bits()],
                [1, 1, 1],
                &[(3, words(&c["expected_forward_bits"]))],
                None,
            )?;
            replay(
                &device,
                include_bytes!("../../shaders/vision_layer_norm_input_grad_cpu_avx2.spv"),
                vec![
                    dy.clone(),
                    x.clone(),
                    w,
                    mean.clone(),
                    rstd.clone(),
                    vec![0.; 1024],
                ],
                &[16, 64],
                [1, 1, 1],
                &[(5, words(&c["expected_dx_bits"]))],
                None,
            )?;
            replay(
                &device,
                include_bytes!("../../shaders/vision_layer_norm_param_grad_cpu_avx2.spv"),
                vec![dy, x, mean, rstd, vec![0.; 64], vec![0.; 64]],
                &[16, 64],
                [1, 1, 1],
                &[
                    (4, words(&c["expected_dw_bits"])),
                    (5, words(&c["expected_db_bits"])),
                ],
                None,
            )?;
        } else {
            replay(
                &device,
                include_bytes!("../../shaders/image_text_rms_wide_cpu_avx2.spv"),
                vec![
                    x.clone(),
                    w.clone(),
                    vec![0.; 1024],
                    vec![0.; 16],
                    vec![0.; 16],
                ],
                &[
                    16,
                    64,
                    1,
                    (c["eps"].as_f64().unwrap() as f32).to_bits(),
                    0,
                    0,
                    1,
                    3,
                ],
                [1, 1, 1],
                &[(2, words(&c["expected_forward_bits"]))],
                None,
            )?;
            replay(
                &device,
                include_bytes!("../../shaders/image_text_rms_wide_cpu_avx2_backward.spv"),
                vec![dy.clone(), x.clone(), w, rstd.clone(), mean, vec![0.; 1024]],
                &[16, 64, 1, 0, 0, 2, 0],
                [1, 1, 1],
                &[(5, words(&c["expected_dx_bits"]))],
                None,
            )?;
            replay(
                &device,
                include_bytes!("../../shaders/transformer_rms_norm_param_grad.spv"),
                vec![dy, x, rstd, vec![0.; 64]],
                &[16, 64, 1],
                [1, 1, 1],
                &[(3, words(&c["expected_dw_bits"]))],
                None,
            )?;
        }
    }
    Ok(())
}

#[test]
fn shared_cpu_clip_adjoint_and_bias_match_independent_words() -> Result<()> {
    let data = training_corpus()?;
    let device = VulkanDevice::new()?;
    for c in data["attention"].as_array().unwrap() {
        let scale = (c["scale"].as_f64().unwrap() as f32).to_bits();
        replay(
            &device,
            include_bytes!("../../shaders/seq5_attention_cpu_avx2.spv"),
            vec![
                floats(&c["q"]),
                floats(&c["k"]),
                floats(&c["v"]),
                vec![0.; 160],
            ],
            &[2, 5, 16, 2, scale],
            [5, 2, 2],
            &[(3, words(&c["expected_forward_bits"]))],
            None,
        )?;
        replay(
            &device,
            include_bytes!("../../shaders/seq5_attention_cpu_avx2_backward.spv"),
            vec![
                floats(&c["q"]),
                floats(&c["k"]),
                floats(&c["v"]),
                vec![1.; 10],
                vec![1.; 10],
                vec![0.],
                floats(&c["dy"]),
                vec![0.; 160],
                vec![0.; 160],
                vec![0.; 160],
                vec![0.],
                vec![0.],
                vec![0.],
                vec![0.],
                vec![0.],
            ],
            &[
                2,
                5,
                5,
                16,
                16,
                2,
                2,
                0,
                scale,
                1f32.to_bits(),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
            [2, 2, 1],
            &[
                (7, words(&c["expected_dq_bits"])),
                (8, words(&c["expected_dk_bits"])),
                (9, words(&c["expected_dv_bits"])),
            ],
            None,
        )?;
    }
    for c in data["bias"].as_array().unwrap() {
        let rows = c["rows"].as_u64().unwrap() as u32;
        let dim = c["dim"].as_u64().unwrap() as u32;
        replay(
            &device,
            include_bytes!("../../shaders/vision_bias_grad_cpu_avx2.spv"),
            vec![floats(&c["dy"]), vec![0.; dim as usize]],
            &[rows, dim],
            [dim.div_ceil(256), 1, 1],
            &[(1, words(&c["expected_bits"]))],
            None,
        )?;
    }
    Ok(())
}

#[test]
fn shared_cpu_wide_residual_and_weight_adjoints_match_independent_words() -> Result<()> {
    let data = training_corpus()?;
    let device = VulkanDevice::new()?;
    for c in data["residual_norm"].as_array().unwrap() {
        replay(
            &device,
            include_bytes!("../../shaders/image_text_rms_wide_cpu_avx2_residual.spv"),
            vec![
                floats(&c["dy"]),
                floats(&c["x"]),
                floats(&c["weight"]),
                floats(&c["rstd"]),
                floats(&c["mean"]),
                floats(&c["residual"]),
                vec![0.; 1024],
                vec![0.; 1024],
            ],
            &[16, 64, 1, 0, 0, 2, 0],
            [1, 1, 1],
            &[(7, words(&c["expected_dx_bits"]))],
            None,
        )?;
    }
    for c in data["weight_gradient"].as_array().unwrap() {
        let output = c["output_dim"].as_u64().unwrap() as u32;
        replay(
            &device,
            include_bytes!("../../shaders/vision_linear_weight_grad_wide_cpu_avx2.spv"),
            vec![
                floats(&c["x"]),
                floats(&c["dy"]),
                vec![0.; 64 * output as usize],
            ],
            &[16, 64, output],
            [4, output.div_ceil(16), 1],
            &[(2, words(&c["expected_bits"]))],
            None,
        )?;
    }
    Ok(())
}

#[test]
fn shared_cpu_activation_adjoints_preserve_cpu_graph_order() -> Result<()> {
    let data = training_corpus()?;
    let device = VulkanDevice::new()?;
    for c in data["unary"].as_array().unwrap() {
        let x = floats(&c["x"]);
        let dy = floats(&c["dy"]);
        let n = x.len();
        let (shader, inputs, limit): (&[u8], _, _) = if c["name"] == "silu" {
            (
                include_bytes!("../../shaders/silu_backward_cpu_avx2.spv"),
                vec![dy, x, vec![0.; n]],
                None,
            )
        } else {
            // SLEEF-compatible tanh has a bounded error at rare wide-range
            // polynomial boundaries; keep that independent diagnostic explicit.
            (
                include_bytes!("../../shaders/image_text_new_gelu_cpu_avx2_backward.spv"),
                vec![x, dy, vec![0.; n]],
                Some(1e-8),
            )
        };
        replay(
            &device,
            shader,
            inputs,
            &[n as u32],
            [(n as u32).div_ceil(256), 1, 1],
            &[(2, words(&c["expected_backward_bits"]))],
            limit,
        )?;
    }
    Ok(())
}

#[test]
fn shared_cpu_rope_forward_and_adjoint_match_independent_words() -> Result<()> {
    let data = training_corpus()?;
    let cases = data["rope"].as_array().unwrap();
    let k = &cases[0];
    let q = &cases[1];
    assert_eq!(k["heads"], 2);
    assert_eq!(q["heads"], 4);
    let device = VulkanDevice::new()?;
    let push = [
        16,
        8,
        64,
        4,
        2,
        16,
        0,
        10000f32.to_bits(),
        0,
        7,
        1f32.to_bits(),
        0,
        0,
        0,
        1f32.to_bits(),
        0,
        0,
        0,
        0,
        0,
        1,
    ];
    for backward in [false, true] {
        let (shader, field, expected): (&[u8], _, _) = if backward {
            (
                include_bytes!("../../shaders/image_text_rope_cpu_avx2_backward.spv"),
                "dy",
                "expected_backward_bits",
            )
        } else {
            (
                include_bytes!("../../shaders/image_text_rope_cpu_avx2_forward.spv"),
                "x",
                "expected_forward_bits",
            )
        };
        replay(
            &device,
            shader,
            vec![
                floats(&q[field]),
                floats(&k[field]),
                vec![0.; 1024],
                vec![0.; 512],
                floats(&q["frequency"]),
            ],
            &push,
            [4, 1, 1],
            &[(2, words(&q[expected])), (3, words(&k[expected]))],
            None,
        )?;
    }
    Ok(())
}

#[test]
fn shared_cpu_cosine_matches_frozen_active_vml_dispatch() -> Result<()> {
    let data = training_corpus()?;
    let case = &data["cosine"];
    let input = floats(&case["x"]);
    assert_eq!(input.len(), 8320);
    let n = input.len() as u32;
    replay(
        &VulkanDevice::new()?,
        include_bytes!("../../shaders/image_text_vml_cos.spv"),
        vec![input, vec![0.; n as usize]],
        &[n],
        [n.div_ceil(256), 1, 1],
        &[(1, words(&case["expected_bits"]))],
        None,
    )?;
    Ok(())
}

#[test]
fn shared_cpu_patch_projection_matches_independent_onednn_words() -> Result<()> {
    let data = corpus(
        include_bytes!("../../validation/fixtures/vision_avx2_row_tail/shared_patch.json"),
        "5a6914e6c0a16c9d60f040f6e294203e3fb484029a566b2fca1faabd3cc0698e",
    )?;
    let device = VulkanDevice::new()?;
    for c in data["cases"].as_array().unwrap() {
        replay(
            &device,
            include_bytes!("../../shaders/vision_patch_linear_cpu_avx2.spv"),
            vec![floats(&c["input"]), floats(&c["weight"]), vec![0.; 128]],
            &[8, 48, 16, 3],
            [1, 1, 1],
            &[(2, words(&c["expected_bits"]))],
            None,
        )?;
    }
    Ok(())
}
