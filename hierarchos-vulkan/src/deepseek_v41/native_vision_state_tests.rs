use super::*;

fn setup(
    device: VulkanDevice,
    case: &Value,
) -> Result<(
    Package,
    DeepseekV41MultimodalTrainer,
    DeepseekV41TrainingBatch,
    Vec<i32>,
    Vec<f32>,
    usize,
)> {
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
                    .map(|x| x.as_u64().unwrap() as usize)
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
    let adapter =
        DeepseekV41LoraAdapter::initialize(&mut package.reader()?, &[], 2, 2., false, 1793)?;
    // Benchmark-only residency switch: the throughput run compares the default
    // streamed experts against resident experts on the same binary. Unset
    // everywhere else, so qualification keeps the production configuration.
    let resident_experts = std::env::var_os("HIERARCHOS_MULTIMODAL_BENCH_RESIDENT_EXPERTS")
        .is_some_and(|value| value != "0");
    let trainer = DeepseekV41MultimodalTrainer::new(
        device,
        &package.root,
        &adapter,
        serde_json::from_value(case["hyper"].clone())?,
        DeepseekV41ModelTrainingConfig {
            batch_size: f.batch_size,
            stream_experts: !resident_experts,
            ..Default::default()
        },
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
        .collect();
    Ok((
        package,
        trainer,
        batch,
        types,
        floats(&case["patches"]),
        case["image_start"].as_u64().unwrap() as usize,
    ))
}

/// Explicit performance run: measures training without qualification readbacks,
/// export I/O or timestamp-query instrumentation in the timed region.
#[test]
#[ignore = "explicit multimodal throughput benchmark; requires frozen shared fixtures"]
fn multimodal_training_throughput_benchmark() -> Result<()> {
    use std::time::Instant;
    let destination = PathBuf::from(
        std::env::var_os("HIERARCHOS_MULTIMODAL_BENCH_OUTPUT")
            .context("HIERARCHOS_MULTIMODAL_BENCH_OUTPUT is required")?,
    );
    let fixtures = PathBuf::from(
        std::env::var_os("HIERARCHOS_MULTIMODAL_BENCH_FIXTURES")
            .context("HIERARCHOS_MULTIMODAL_BENCH_FIXTURES is required")?,
    );
    assert!(
        std::env::var_os("HIERARCHOS_VULKAN_PROFILE_KERNELS").is_none(),
        "timestamp profiling must be disabled for throughput measurements"
    );
    let iterations = std::env::var("HIERARCHOS_MULTIMODAL_BENCH_STEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(7);
    assert!((3..=100).contains(&iterations));
    let warmup = 2;
    let device = VulkanDevice::new()?;
    let cpu: Value = serde_json::from_slice(include_bytes!(
        "../../validation/fixtures/deepseek_v41_native_pipeline_avx2.json"
    ))?;
    let mut records = Vec::new();
    for case in cpu["cases"].as_array().unwrap() {
        let (_package, mut trainer, batch, types, patches, start) = setup(device.clone(), case)?;
        let mut samples = Vec::new();
        for index in 0..warmup + iterations {
            let before = device.submission_count();
            let begin = Instant::now();
            let gradients = trainer.gradients(&batch, &patches, &types, start)?;
            let gradients_ms = begin.elapsed().as_secs_f64() * 1000.;
            let update = Instant::now();
            trainer.apply_gradients(&gradients)?;
            let update_ms = update.elapsed().as_secs_f64() * 1000.;
            assert!(gradients.text.loss.is_finite());
            if index >= warmup {
                samples.push(json!({"gradients_ms":gradients_ms,"update_ms":update_ms,
                    "total_ms":gradients_ms+update_ms,"submissions":device.submission_count()-before}));
            }
        }
        let logits = trainer
            .forward_logits(&batch, &patches, &types, start)?
            .read_all()?;
        let mut digest = Sha256::new();
        for (name, tensor) in trainer.image().parameters()? {
            digest.update(name.as_bytes());
            for value in tensor.values {
                digest.update(value.to_bits().to_le_bytes());
            }
        }
        let memory = device.memory_stats()?;
        records.push(
            json!({"case":case["name"],"kind":"native_v41","samples":samples,
            "final_logits_bits":bits(&logits),"image_digest":format!("{:x}",digest.finalize()),
            "adapter_digest":trainer.adapter()?.snapshot_digest()?,
            "memory":{"live_buffer_bytes":memory.live_buffer_bytes,
                "peak_live_buffer_bytes":memory.peak_live_buffer_bytes}}),
        );
    }
    for (name, folder) in [
        ("llama", "shared-image-cpu-v3"),
        ("gpt2", "shared-image-cpu-v3"),
        ("t5", "seq2seq-image-cpu-v1"),
    ] {
        let directory = fixtures.join(folder).join(name).join("unmasked");
        let f: Value = serde_json::from_slice(&fs::read(directory.join("input.json"))?)?;
        let u32s = |key: &str| {
            f[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect::<Vec<_>>()
        };
        let fp32s = |key: &str| {
            f[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let ids = u32s("input_ids");
        let mask = fp32s("attention_mask");
        let targets = u32s("targets");
        let weights = fp32s("loss_weights");
        let decoder = f
            .get("decoder_input_ids")
            .filter(|x| !x.is_null())
            .map(|_| (u32s("decoder_input_ids"), fp32s("decoder_attention_mask")));
        let mut graph = crate::VulkanImageText::from_hf_package(
            device.clone(),
            directory.join("model"),
            f["batch_size"].as_u64().unwrap() as usize,
            f["seq_len"].as_u64().unwrap() as usize,
            f["image_count"].as_u64().unwrap() as usize,
        )?;
        let rgb = f["rgb"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u8)
            .collect::<Vec<_>>();
        let pixels = graph.preprocess_rgb8(&rgb)?;
        let hyper = crate::CpuAdamWHyperParams {
            lr: f["lr"].as_f64().unwrap(),
            beta1: f["beta1"].as_f64().unwrap(),
            beta2: f["beta2"].as_f64().unwrap(),
            eps: f["eps"].as_f64().unwrap(),
            weight_decay: f["weight_decay"].as_f64().unwrap(),
        };
        let mut samples = Vec::new();
        for index in 0..warmup + iterations {
            let before = device.submission_count();
            let begin = Instant::now();
            let result = if let Some((d, m)) = &decoder {
                graph.train_seq2seq_with_optimizer(
                    &ids, &mask, d, m, &pixels, &targets, &weights, hyper,
                )?
            } else {
                graph.train_step_with_optimizer(&ids, &mask, &pixels, &targets, &weights, hyper)?
            };
            let ms = begin.elapsed().as_secs_f64() * 1000.;
            assert_eq!(result.text.step as usize, index + 1);
            if index >= warmup {
                samples.push(json!({"total_ms":ms,"submissions":device.submission_count()-before}));
            }
        }
        let output = if let Some((d, m)) = &decoder {
            graph.forward_seq2seq(&ids, &mask, d, m, &pixels)?
        } else {
            graph.forward(&ids, &mask, &pixels)?
        };
        records.push(json!({"case":name,"kind":"shared_image","samples":samples,"final_logits_bits":bits(&output.logits)}));
    }
    // Per-call-site submission accounting for the whole run. Every timed
    // iteration performs the same work, so dividing by the step count gives the
    // per-step cost of each site.
    if crate::vulkan::submission_report::enabled() {
        let steps = (warmup + iterations) as u64;
        let mut report = crate::vulkan::submission_report::take();
        report.sort_by_key(|&(_, _, count)| std::cmp::Reverse(count));
        for (file, line, count) in report {
            eprintln!(
                "hierarchos_submission_report file={file} line={line} total={count} per_step={:.2}",
                count as f64 / steps as f64
            );
        }
    }
    fs::write(
        destination,
        serde_json::to_vec_pretty(
            &json!({"device":device.name(),"warmup":warmup,"iterations":iterations,"records":records}),
        )?,
    )?;
    Ok(())
}

fn equal_weights(a: &DeepseekV41MultimodalTrainer, b: &DeepseekV41MultimodalTrainer) -> Result<()> {
    assert_eq!(
        a.adapter()?.snapshot_digest()?,
        b.adapter()?.snapshot_digest()?
    );
    let ap = a.image().parameters()?;
    let bp = b.image().parameters()?;
    assert_eq!(ap.len(), 26);
    assert_eq!(ap.keys().collect::<Vec<_>>(), bp.keys().collect::<Vec<_>>());
    for (name, p) in ap {
        assert_eq!(
            bits(&p.values),
            bits(&bp[&name].values),
            "image state {name}"
        );
    }
    Ok(())
}

#[test]
fn deepseek_v41_native_joint_checkpoint_resume_export_and_freshness() -> Result<()> {
    let bytes = include_bytes!("../../validation/fixtures/deepseek_v41_native_pipeline_avx2.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "9f5592989820c1567caf3fd4ca6b5354f7bb8e78dfb6b38f1ebb43b9b4cc403b"
    );
    let cpu: Value = serde_json::from_slice(bytes)?;
    let device = VulkanDevice::new()?;
    let mut cases = 0;
    for case in cpu["cases"].as_array().unwrap() {
        let (package, mut original, batch, types, patches, start) = setup(device.clone(), case)?;
        let mut owned = OwnedCheckpoints {
            root: package.root.clone(),
            paths: Vec::new(),
        };
        let checkpoint = package.root.join("joint-step1");
        let roundtrip = package.root.join("joint-step1-reloaded");
        let export = package.root.join("native-export");
        owned
            .paths
            .extend([checkpoint.clone(), roundtrip.clone(), export.clone()]);
        original.train_step(&batch, &patches, &types, start)?;
        original.save_checkpoint(&checkpoint)?;
        assert!(original.save_checkpoint(&checkpoint).is_err());
        let mut resumed =
            DeepseekV41MultimodalTrainer::resume(device.clone(), &package.root, &checkpoint)?;
        assert_eq!(resumed.optimizer_step(), 1);
        assert_eq!(resumed.image().step(), 1);
        equal_weights(&original, &resumed)?;
        resumed.save_checkpoint(&roundtrip)?;
        for name in [
            "native_image.safetensors",
            "language/optimizer_state.safetensors",
        ] {
            assert_eq!(
                fs::read(checkpoint.join(name))?,
                fs::read(roundtrip.join(name))?,
                "exact optimizer checkpoint {name}"
            );
        }
        let stale = original.gradients(&batch, &patches, &types, start)?;
        original.forward_logits(&batch, &patches, &types, start)?;
        assert!(original.apply_gradients(&stale).is_err());
        let old = original.gradients(&batch, &patches, &types, start)?;
        let current = original.gradients(&batch, &patches, &types, start)?;
        assert!(original.apply_gradients(&old).is_err());
        resumed.gradients(&batch, &patches, &types, start)?;
        assert!(resumed.apply_gradients(&current).is_err());
        original.apply_gradients(&current)?;
        assert!(original.apply_gradients(&current).is_err());
        resumed.train_step(&batch, &patches, &types, start)?;
        assert_eq!(original.optimizer_step(), 2);
        assert_eq!(resumed.optimizer_step(), 2);
        equal_weights(&original, &resumed)?;
        let expected = original
            .forward_logits(&batch, &patches, &types, start)?
            .read_all()?;
        assert_eq!(
            bits(&expected),
            bits(
                &resumed
                    .forward_logits(&batch, &patches, &types, start)?
                    .read_all()?
            )
        );
        original.export_native(&export)?;
        assert!(original.export_native(&export).is_err());
        assert!(!export.join("language/optimizer_state.safetensors").exists());
        let mut reloaded = DeepseekV41MultimodalTrainer::from_native_export(
            device.clone(),
            &package.root,
            &export,
        )?;
        assert_eq!(reloaded.optimizer_step(), 0);
        equal_weights(&original, &reloaded)?;
        assert_eq!(
            bits(&expected),
            bits(
                &reloaded
                    .forward_logits(&batch, &patches, &types, start)?
                    .read_all()?
            )
        );
        let alternate: Vec<f32> = patches.iter().map(|v| -*v).collect();
        let changed = reloaded
            .forward_logits(&batch, &alternate, &types, start)?
            .read_all()?;
        assert!(
            changed
                .iter()
                .zip(&expected)
                .any(|(a, b)| (*a - *b).abs() > 2e-7),
            "native image must affect real language logits"
        );
        let image_path = export.join("native_image.safetensors");
        let mut corrupt = fs::read(&image_path)?;
        *corrupt.last_mut().unwrap() ^= 1;
        fs::write(&image_path, corrupt)?;
        assert!(DeepseekV41MultimodalTrainer::from_native_export(
            device.clone(),
            &package.root,
            &export
        )
        .is_err());
        let foreign = Package::new(&fixture()?, false)?;
        assert!(
            DeepseekV41MultimodalTrainer::resume(device.clone(), &foreign.root, &checkpoint)
                .is_err()
        );
        cases += 1;
    }
    assert_eq!(cases, 2);
    eprintln!("Native V4.1 joint resume/export: two cases, exact weights/moments/continued updates/reloaded logits; stale and foreign gradients rejected");
    Ok(())
}
