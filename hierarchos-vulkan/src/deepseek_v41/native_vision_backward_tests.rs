use super::*;
fn words(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}
fn vector(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

#[test]
fn deepseek_v41_native_avx2_lora_matmuls_match_frozen_words() -> Result<()> {
    let _math = crate::deepseek_v41::cpu_math::avx2();
    let bytes = include_bytes!(
        "../../validation/fixtures/vision_avx2_row_tail/deepseek_v41_lora_matmul_avx2.json"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "8d001be791b31b1a2fa34752c358f229a3e5bed31e11ea7424fa9d32165fb25a"
    );
    let cpu: Value = serde_json::from_slice(bytes)?;
    let device = VulkanDevice::new()?;
    let weight = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/transformer_deepseek_v41_weight_grad_avx2.spv"),
        3,
        12,
    )?;
    let input = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/transformer_deepseek_v41_input_grad_avx2.spv"),
        3,
        12,
    )?;
    let narrow = vulkan::ComputeKernel::new(
        &device,
        crate::deepseek_v41::gen9::shader(
            &device,
            include_bytes!("../../shaders/transformer_deepseek_v41_linear.spv"),
        ),
        3,
        16,
    )?;
    let mut count = 0;
    for (section, expected_cases) in [
        ("weight_gradients", 160),
        ("odd_inputs", 80),
        ("narrow", 132),
    ] {
        let cases = cpu[section].as_array().unwrap();
        assert_eq!(cases.len(), expected_cases);
        for (i, c) in cases.iter().enumerate() {
            let m = c["rows"].as_u64().unwrap() as u32;
            let expected = words(&c["expected_bits"]);
            let x = GpuBuffer::from_f32(&device, &vector(&c["input"]))?;
            let second = GpuBuffer::from_f32(
                &device,
                &vector(
                    &c[if section == "weight_gradients" {
                        "upstream"
                    } else {
                        "weight"
                    }],
                ),
            )?;
            let output = GpuBuffer::zeros_f32(&device, expected.len())?;
            let (kernel, push, dispatch) = match section {
                "weight_gradients" => {
                    let n = c["input_dim"].as_u64().unwrap() as u32;
                    let k = c["output_dim"].as_u64().unwrap() as u32;
                    (&weight, vec![m, n, k], [n.div_ceil(16), k.div_ceil(16), 1])
                }
                "odd_inputs" => {
                    let n = c["columns"].as_u64().unwrap() as u32;
                    let k = c["reduction"].as_u64().unwrap() as u32;
                    (&input, vec![m, n, k], [n.div_ceil(16), m.div_ceil(16), 1])
                }
                _ => {
                    let k = c["input_dim"].as_u64().unwrap() as u32;
                    let n = c["output_dim"].as_u64().unwrap() as u32;
                    (&narrow, vec![m, k, n, 0], [1, 1, 1])
                }
            };
            let mut commands = device.compute_batch()?;
            kernel.record_dispatch(
                &mut commands,
                &[&x, &second, &output],
                bytemuck::cast_slice(&push),
                dispatch,
            )?;
            commands.submit()?;
            assert_eq!(
                output.read_u32(expected.len())?,
                expected,
                "native LoRA {section}/{i}"
            );
            count += expected.len();
        }
    }
    assert_eq!(count, 18806);
    eprintln!("Native AVX2 LoRA arithmetic: 372 cases, {count} exact words");
    Ok(())
}

#[test]
fn deepseek_v41_native_avx2_compressor_adjoint_matches_frozen_words() -> Result<()> {
    let _math = crate::deepseek_v41::cpu_math::avx2();
    let bytes = include_bytes!(
        "../../validation/fixtures/vision_avx2_row_tail/deepseek_v41_compressor_avx2.json"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "6968719fd8bc8e7daab0ef55178ca60c9a04aa2b4817ca26588e551cd2db769a"
    );
    let corpus: Value = serde_json::from_slice(bytes)?;
    let cases = corpus["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 12);
    let operand_bytes = include_bytes!(
        "../../validation/fixtures/vision_avx2_row_tail/deepseek_v41_compressor_operands_avx2.json"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(operand_bytes)),
        "1e1b0015b08f77650b489cdf2a26bd95ad1547f6497dce4edbbbffd5c227a85f"
    );
    let operands: Value = serde_json::from_slice(operand_bytes)?;
    let operand_cases = operands["cases"].as_array().unwrap();
    assert_eq!(operand_cases.len(), cases.len());
    let device = VulkanDevice::new()?;
    let adjoints = crate::deepseek_v41::DeepseekV41CompressorAdjoints::new(device.clone())?;
    let mut count = 0;
    for (i, c) in cases.iter().enumerate() {
        let rows = c["rows"].as_u64().unwrap() as usize;
        assert_eq!(operand_cases[i]["rows"], c["rows"]);
        assert_eq!(operand_cases[i]["seed"], c["seed"]);
        // Freeze the forward operands separately, so this input-VJP regression
        // measures the repaired norm/projection reverse chain. The joint gate
        // still exercises the uninterrupted native forward and both updates.
        let projected = GpuBuffer::from_u32(&device, &words(&operand_cases[i]["projected_bits"]))?;
        let weight = GpuBuffer::from_f32(&device, &vector(&c["weight"]))?;
        let norm = GpuBuffer::from_f32(&device, &vector(&c["norm_weight"]))?;
        let dy = GpuBuffer::from_f32(&device, &vector(&c["upstream"]))?;
        let grad = adjoints.prefill(
            &crate::deepseek_v41::DeepseekV41CompressorPrefill {
                batch: 1,
                seq: rows,
                start_pos: 0,
                hidden: 32,
                dim: 32,
                ratio: 1,
                eps: c["eps"].as_f64().unwrap() as f32,
            },
            &crate::deepseek_v41::DeepseekV41CompressorOperands {
                pooled: &projected,
                projected_kv: &projected,
                projected_gate: None,
                kv_weight: &weight,
                gate_weight: None,
                norm_weight: &norm,
            },
            &dy,
            None,
        )?;
        assert_eq!(
            grad.grad_input.read_u32(rows * 32)?,
            words(&c["input_bits"]),
            "compressor input adjoint {i}"
        );
        assert!(grad.grad_state_kv.is_none() && grad.grad_state_score.is_none());
        count += rows * 32;
    }
    assert_eq!(count, 2112);
    eprintln!(
        "Native AVX2 ratio1 compressor: 12 input-adjoint cases, {count} exact gradient words"
    );
    Ok(())
}

#[test]
fn deepseek_v41_native_avx2_sparse_adjoints_match_frozen_words() -> Result<()> {
    let _math = crate::deepseek_v41::cpu_math::avx2();
    let bytes = include_bytes!(
        "../../validation/fixtures/vision_avx2_row_tail/deepseek_v41_sparse_backward_avx2.json"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "5a75bfb5fb6fb62510f30125f59cf8c3ca88c604fccd6d45cb661963b7f74bfb"
    );
    let corpus: Value = serde_json::from_slice(bytes)?;
    let cases = corpus["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 8);
    let device = VulkanDevice::new()?;
    let adjoints = crate::deepseek_v41::DeepseekV41AttentionAdjoints::new(device.clone())?;
    let weights_kernel = vulkan::ComputeKernel::new(
        &device,
        crate::deepseek_v41::gen9::shader(
            &device,
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_weights.spv"),
        ),
        7,
        40,
    )?;
    let values_kernel = vulkan::ComputeKernel::new(
        &device,
        crate::deepseek_v41::gen9::shader(
            &device,
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_values.spv"),
        ),
        6,
        32,
    )?;
    let mut count = 0;
    for (i, c) in cases.iter().enumerate() {
        let m = c["rows"].as_u64().unwrap() as usize;
        let k = c["compressed_topk"].as_u64().unwrap() as usize;
        let capacity = c["compressed_capacity"].as_u64().unwrap() as usize;
        let total = 4 + k;
        let q = GpuBuffer::from_f32(&device, &vector(&c["query"]))?;
        let window = GpuBuffer::from_f32(&device, &vector(&c["window"]))?;
        let comp = if k > 0 {
            GpuBuffer::from_f32(&device, &vector(&c["compressed"]))?
        } else {
            GpuBuffer::zeros_f32(&device, 1)?
        };
        let indices = c["indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as u32)
            .collect::<Vec<_>>();
        let wi = indices
            .chunks_exact(total)
            .flat_map(|row| row[..4].iter().copied())
            .collect::<Vec<_>>();
        let ci = indices
            .chunks_exact(total)
            .flat_map(|row| row[4..].iter().copied())
            .collect::<Vec<_>>();
        let wi = GpuBuffer::from_u32(&device, &wi)?;
        let ci = if k > 0 {
            GpuBuffer::from_u32(&device, &ci)?
        } else {
            GpuBuffer::zeros_f32(&device, 1)?
        };
        let sink = GpuBuffer::from_f32(&device, &vector(&c["sink"]))?;
        let upstream = GpuBuffer::from_f32(&device, &vector(&c["upstream"]))?;
        let weights = GpuBuffer::zeros_f32(&device, m * 2 * total)?;
        let output = GpuBuffer::zeros_f32(&device, m * 64)?;
        let mut commands = device.compute_batch()?;
        weights_kernel.record_dispatch(
            &mut commands,
            &[&q, &window, &comp, &wi, &ci, &sink, &weights],
            bytemuck::cast_slice(&[
                m as u32,
                1,
                m as u32,
                2,
                32,
                m as u32,
                4,
                k as u32,
                capacity as u32,
                (32f64.powf(-0.5) as f32).to_bits(),
            ]),
            [1, 2, m as u32],
        )?;
        values_kernel.record_dispatch(
            &mut commands,
            &[&window, &comp, &wi, &ci, &weights, &output],
            bytemuck::cast_slice(&[
                m as u32,
                m as u32,
                2,
                32,
                m as u32,
                4,
                k as u32,
                capacity as u32,
            ]),
            [1, 2, m as u32],
        )?;
        commands.submit()?;
        assert_eq!(
            weights.read_u32(m * 2 * total)?,
            bits(&vector(&c["weights"])),
            "sparse weights {i}"
        );
        assert_eq!(
            output.read_u32(m * 64)?,
            bits(&vector(&c["output"])),
            "sparse values {i}"
        );
        let result = adjoints.sparse(
            &q,
            &window,
            if k > 0 { Some(&comp) } else { None },
            &wi,
            &ci,
            &sink,
            &upstream,
            2,
            32,
            crate::deepseek_v41::DeepseekV41SparseAttentionParams {
                batch: 1,
                seq: m,
                window_rows: m,
                window_topk: 4,
                compressed_topk: k,
                compressed_capacity: capacity,
            },
        )?;
        for (name, buffer) in [
            ("query_gradient", &result.grad_query),
            ("window_gradient", &result.grad_window),
            ("dot_gradient", &result.grad_dot),
        ] {
            let expected = vector(&c[name]);
            let actual = buffer.read_u32(expected.len())?;
            let reference = bits(&expected);
            for (j, (&a, &b)) in actual.iter().zip(&reference).enumerate() {
                // An unused masked probability derivative can carry either
                // signed zero. Every active edge still compares raw words.
                let masked = name == "dot_gradient"
                    && (indices[(j / (2 * total)) * total + j % total] as i32) < 0;
                if masked {
                    assert_eq!(a & 0x7fffffff, 0);
                    assert_eq!(b & 0x7fffffff, 0);
                } else {
                    assert_eq!(a, b, "sparse case {i}/{name}/{j}");
                }
            }
            count += expected.len();
        }
        if let Some(comp) = result.grad_compressed {
            let expected = vector(&c["compressed_gradient"]);
            assert_eq!(
                comp.read_u32(expected.len())?,
                bits(&expected),
                "sparse compressed {i}"
            );
            count += expected.len();
        }
    }
    assert!(count > 7000);
    eprintln!(
        "Native AVX2 sparse attention: eight forward/adjoint cases; {count} exact gradient words"
    );
    Ok(())
}

#[test]
fn deepseek_v41_native_avx2_grouped_adjoints_match_frozen_words() -> Result<()> {
    let bytes = include_bytes!(
        "../../validation/fixtures/vision_avx2_row_tail/deepseek_v41_grouped_backward_avx2.json"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "755cfb5adb50e727f94a256727a32c0fad223975bb494d6f81f6081fbf888779"
    );
    let corpus: Value = serde_json::from_slice(bytes)?;
    let cases = corpus["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 72);
    let device = VulkanDevice::new()?;
    let kernel = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/transformer_deepseek_v41_grouped_input_grad_avx2.spv"),
        3,
        16,
    )?;
    let mut count = 0;
    for (i, c) in cases.iter().enumerate() {
        let m = c["rows"].as_u64().unwrap() as u32;
        let g = c["groups"].as_u64().unwrap() as u32;
        let w = c["group_width"].as_u64().unwrap() as u32;
        let r = c["output_rank"].as_u64().unwrap() as u32;
        let dy = GpuBuffer::from_f32(&device, &vector(&c["upstream"]))?;
        let weight = GpuBuffer::from_f32(&device, &vector(&c["weight"]))?;
        let expected = words(&c["expected_bits"]);
        let dx = GpuBuffer::zeros_f32(&device, expected.len())?;
        let mut commands = device.compute_batch()?;
        kernel.record_dispatch(
            &mut commands,
            &[&dy, &weight, &dx],
            bytemuck::cast_slice(&[m, g, w, r]),
            [(m * g * w).div_ceil(64), 1, 1],
        )?;
        commands.submit()?;
        assert_eq!(
            dx.read_u32(expected.len())?,
            expected,
            "grouped VJP case {i}"
        );
        count += expected.len();
    }
    assert_eq!(count, 69888);
    Ok(())
}

#[test]
fn deepseek_v41_native_avx2_hc_adjoints_match_released_cpu() -> Result<()> {
    let _math = crate::deepseek_v41::cpu_math::avx2();
    let bytes = include_bytes!(
        "../../validation/fixtures/vision_avx2_row_tail/deepseek_v41_hc_backward_avx2.json"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "c332c3e7ec76ebc997f0f4d5a8631376e8b7acc0c95c9bffbf7ae70245609d06"
    );
    let corpus: Value = serde_json::from_slice(bytes)?;
    let cases = corpus["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 20);
    let device = VulkanDevice::new()?;
    let split = vulkan::ComputeKernel::new(
        &device,
        crate::deepseek_v41::cpu_math::selected(
            &device,
            include_bytes!("../../shaders/transformer_deepseek_v41_hc_split_grad.spv"),
        ),
        8,
        28,
    )?;
    let mix = vulkan::ComputeKernel::new(
        &device,
        crate::deepseek_v41::gen9::shader(
            &device,
            include_bytes!("../../shaders/transformer_deepseek_v41_hc_mixes_input_grad.spv"),
        ),
        5,
        20,
    )?;
    let mut failures = Vec::new();
    let mut tally = Tally::default();
    for (i, c) in cases.iter().enumerate() {
        let m = c["rows"].as_u64().unwrap() as usize;
        let h = c["hc"].as_u64().unwrap() as usize;
        let k = h * (h + 2);
        let raw = GpuBuffer::from_f32(&device, &vector(&c["raw"]))?;
        let x = GpuBuffer::from_f32(&device, &vector(&c["input"]))?;
        let scale = GpuBuffer::from_f32(&device, &vector(&c["scale"]))?;
        let bias = GpuBuffer::from_f32(&device, &vector(&c["bias"]))?;
        let gp = GpuBuffer::from_f32(&device, &vector(&c["grad_pre"]))?;
        let gpost = GpuBuffer::from_f32(&device, &vector(&c["grad_post"]))?;
        let gc = GpuBuffer::from_f32(&device, &vector(&c["grad_comb"]))?;
        let dm = GpuBuffer::zeros_f32(&device, m * k)?;
        let mut commands = device.compute_batch()?;
        split.record_dispatch(
            &mut commands,
            &[&raw, &x, &scale, &bias, &gp, &gpost, &gc, &dm],
            bytemuck::cast_slice(&[
                m as u32,
                h as u32,
                32,
                k as u32,
                3,
                1e-6f32.to_bits(),
                1e-20f32.to_bits(),
            ]),
            [m as u32, 1, 1],
        )?;
        commands.submit()?;
        assert_eq!(
            dm.read_u32(m * k)?,
            words(&c["mix_bits"]),
            "HC split case {i}"
        );
        let w = GpuBuffer::from_f32(&device, &vector(&c["weight"]))?;
        let dx = GpuBuffer::zeros_f32(&device, m * h * 32)?;
        let mut commands = device.compute_batch()?;
        mix.record_dispatch(
            &mut commands,
            &[&dm, &raw, &x, &w, &dx],
            bytemuck::cast_slice(&[m as u32, h as u32, 32, k as u32, 1e-20f32.to_bits()]),
            [m as u32, 1, 1],
        )?;
        commands.submit()?;
        let expected = words(&c["input_bits"]);
        if h == 2 {
            assert_eq!(dx.read_u32(m * h * 32)?, expected, "HC2 input case {i}");
        }
        tally.add(full_gate(
            &format!("HC{i}-input"),
            &dx.read_f32(m * h * 32)?,
            &expected,
            &mut failures,
        ));
    }
    eprintln!("Native AVX2 released HC adjoints: {}", tally.json());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn deepseek_v41_native_avx2_input_adjoints_match_frozen_words() -> Result<()> {
    let bytes = include_bytes!(
        "../../validation/fixtures/vision_avx2_row_tail/deepseek_v41_backward_avx2.json"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "a67d71118eb14abba4ce5dcf9905a85a48c45112ce3023ab2b548e8d050100aa"
    );
    let corpus: Value = serde_json::from_slice(bytes)?;
    let cases = corpus["matmul"].as_array().unwrap();
    assert_eq!(cases.len(), 160);
    let device = VulkanDevice::new()?;
    let kernel = vulkan::ComputeKernel::new(
        &device,
        include_bytes!("../../shaders/transformer_deepseek_v41_input_grad_avx2.spv"),
        3,
        12,
    )?;
    let mut count = 0;
    for (i, c) in cases.iter().enumerate() {
        let m = c["rows"].as_u64().unwrap() as u32;
        let k = c["reduction"].as_u64().unwrap() as u32;
        let n = c["columns"].as_u64().unwrap() as u32;
        let a = GpuBuffer::from_f32(&device, &vector(&c["dy"]))?;
        let w = GpuBuffer::from_f32(&device, &vector(&c["weight"]))?;
        let expected = words(&c["expected_bits"]);
        let output = GpuBuffer::zeros_f32(&device, expected.len())?;
        let mut commands = device.compute_batch()?;
        kernel.record_dispatch(
            &mut commands,
            &[&a, &w, &output],
            bytemuck::cast_slice(&[m, n, k]),
            [n.div_ceil(16), m.div_ceil(16), 1],
        )?;
        commands.submit()?;
        assert_eq!(
            output.read_u32(expected.len())?,
            expected,
            "input VJP case {i}: M={m} N={n} K={k}"
        );
        count += expected.len();
    }
    assert_eq!(count, 12768);
    Ok(())
}

#[test]
fn deepseek_v41_native_avx2_swiglu_adjoints_match_frozen_cpu() -> Result<()> {
    let _math = crate::deepseek_v41::cpu_math::avx2();
    let bytes = include_bytes!(
        "../../validation/fixtures/vision_avx2_row_tail/deepseek_v41_backward_avx2.json"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "a67d71118eb14abba4ce5dcf9905a85a48c45112ce3023ab2b548e8d050100aa"
    );
    let corpus: Value = serde_json::from_slice(bytes)?;
    let cases = corpus["swiglu"].as_array().unwrap();
    assert_eq!(cases.len(), 48);
    let device = VulkanDevice::new()?;
    let shader = crate::deepseek_v41::cpu_math::selected(
        &device,
        include_bytes!("../../shaders/transformer_deepseek_v41_mixed_swiglu_grad.spv"),
    );
    let kernel = vulkan::ComputeKernel::new(&device, shader, 8, 20)?;
    let mut failures = Vec::new();
    let mut tally = Tally::default();
    for (i, c) in cases.iter().enumerate() {
        let m = c["rows"].as_u64().unwrap() as usize;
        let n = c["inter"].as_u64().unwrap() as usize;
        let shared = c["shared"].as_bool().unwrap();
        let g = GpuBuffer::from_f32(&device, &vector(&c["gate"]))?;
        let u = GpuBuffer::from_f32(&device, &vector(&c["up"]))?;
        let r = GpuBuffer::from_f32(&device, &vector(&c["route"]))?;
        let dy = GpuBuffer::from_f32(&device, &vector(&c["dy"]))?;
        let counts = GpuBuffer::from_u32(&device, &[m as u32])?;
        let dg = GpuBuffer::zeros_f32(&device, m * n)?;
        let du = GpuBuffer::zeros_f32(&device, m * n)?;
        let dr = GpuBuffer::zeros_f32(&device, m)?;
        let mut commands = device.compute_batch()?;
        kernel.record_dispatch(
            &mut commands,
            &[&g, &u, &r, &counts, &dy, &dg, &du, &dr],
            bytemuck::cast_slice(&[
                m as u32,
                n as u32,
                if shared { u32::MAX } else { 0 },
                0,
                1.5f32.to_bits(),
            ]),
            [m.div_ceil(64) as u32, 1, 1],
        )?;
        commands.submit()?;
        for (name, output, len) in [("dg", &dg, m * n), ("du", &du, m * n), ("dr", &dr, m)] {
            tally.add(full_gate(
                &format!("SwiGLU-{i}-{name}"),
                &output.read_f32(len)?,
                &words(&c[format!("{name}_bits")]),
                &mut failures,
            ));
        }
    }
    assert_eq!(tally.comparisons, 144);
    eprintln!("Native AVX2 SwiGLU adjoints: {}", tally.json());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn deepseek_v41_native_avx2_gate_adjoints_match_released_cpu() -> Result<()> {
    let _math = crate::deepseek_v41::cpu_math::avx2();
    let bytes = include_bytes!(
        "../../validation/fixtures/vision_avx2_row_tail/deepseek_v41_gate_backward_avx2.json"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "4dac62d5b3a015dbf9124b2cb0987d4713d875d5142300f965dbd432e647d70b"
    );
    let corpus: Value = serde_json::from_slice(bytes)?;
    let cases = corpus["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 20);
    let device = VulkanDevice::new()?;
    let shader = crate::deepseek_v41::gen9::shader(
        &device,
        include_bytes!("../../shaders/transformer_deepseek_v41_moe_gate_grad.spv"),
    );
    let kernel = vulkan::ComputeKernel::new(&device, shader, 6, 32)?;
    let seed = GpuBuffer::zeros_f32(&device, 1)?;
    let mut failures = Vec::new();
    let mut tally = Tally::default();
    let mut count = 0;
    for (i, c) in cases.iter().enumerate() {
        let m = c["rows"].as_u64().unwrap() as u32;
        let input = GpuBuffer::from_f32(&device, &vector(&c["input"]))?;
        let weight = GpuBuffer::from_f32(&device, &vector(&c["weight"]))?;
        let indices = GpuBuffer::from_u32(&device, &words(&c["indices"]))?;
        let dy = GpuBuffer::from_f32(&device, &vector(&c["upstream"]))?;
        let output = GpuBuffer::zeros_f32(&device, m as usize * 32)?;
        let mut commands = device.compute_batch()?;
        kernel.record_dispatch(
            &mut commands,
            &[&input, &weight, &indices, &dy, &output, &seed],
            bytemuck::cast_slice(&[m, 32, 3, 2, 2, 1f32.to_bits(), 1, 1.5f32.to_bits()]),
            [m, 1, 1],
        )?;
        commands.submit()?;
        let expected = words(&c["expected_bits"]);
        assert_eq!(
            output.read_u32(expected.len())?,
            expected,
            "released Gate words {i}"
        );
        tally.add(full_gate(
            &format!("native-Gate-{i}"),
            &output.read_f32(expected.len())?,
            &expected,
            &mut failures,
        ));
        count += expected.len();
    }
    assert_eq!(count, 2944);
    assert_eq!(tally.comparisons, 20);
    eprintln!("Native AVX2 Gate adjoints: {}", tally.json());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}
