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

/// Expand a released batch-1 case to a real native batch. The released fixture
/// has one complete image span in sample zero; additional samples are text-only
/// rows with the processor's TEXT=-1 type. The trainer receives the full
/// `batch * seq_len` token, target and type geometry, and the model runtime is
/// constructed with the same batch size. This is not a repeated batch-1 call.
fn with_batch(case: &Value, batch: usize) -> Result<Value> {
    if !(1..=4).contains(&batch) {
        bail!("native benchmark batch must be 1..=4, got {batch}");
    }
    let mut case = case.clone();
    if batch == 1 {
        return Ok(case);
    }
    let seq_len = case["seq_len"]
        .as_u64()
        .context("benchmark case has no seq_len")? as usize;
    let mut ids = case["token_ids"]
        .as_array()
        .context("benchmark case has no token_ids")?
        .iter()
        .map(|v| Ok(v.as_u64().context("benchmark token id")? as u32))
        .collect::<Result<Vec<_>>>()?;
    let mut targets = case["targets"]
        .as_array()
        .context("benchmark case has no targets")?
        .iter()
        .map(|v| {
            if v.as_i64() == Some(-100) {
                Ok(u32::MAX)
            } else {
                Ok(v.as_u64().context("benchmark target")? as u32)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let mut types = case["token_types"]
        .as_array()
        .context("benchmark case has no token_types")?
        .iter()
        .map(|v| Ok(v.as_i64().context("benchmark token type")? as i32))
        .collect::<Result<Vec<_>>>()?;
    if ids.len() != seq_len || targets.len() != seq_len || types.len() != seq_len {
        bail!("batch-1 benchmark case geometry is not one sequence");
    }
    let text_row = types
        .iter()
        .position(|&kind| kind == -1)
        .context("batch expansion requires a text-only token in the released case")?;
    let filler_id = ids[text_row];
    let filler_target = targets
        .iter()
        .copied()
        .find(|&target| target != u32::MAX)
        .unwrap_or(filler_id);
    for _ in 1..batch {
        ids.extend(std::iter::repeat_n(filler_id, seq_len));
        targets.extend(std::iter::repeat_n(filler_target, seq_len));
        types.extend(std::iter::repeat_n(-1, seq_len));
    }
    case["batch_size"] = json!(batch);
    case["token_ids"] = json!(ids);
    case["targets"] = json!(targets);
    case["token_types"] = json!(types);
    Ok(case)
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
    let profile_requested = std::env::var_os("HIERARCHOS_VULKAN_PROFILE_KERNELS").is_some();
    let profile_allowed = std::env::var_os("HIERARCHOS_MULTIMODAL_BENCH_ALLOW_PROFILE").is_some();
    assert!(
        !profile_requested || profile_allowed,
        "timestamp profiling must be explicitly allowed for throughput measurements"
    );
    let case_filter = std::env::var("HIERARCHOS_MULTIMODAL_BENCH_CASE").ok();
    let telemetry = std::env::var_os("HIERARCHOS_MULTIMODAL_BENCH_TELEMETRY").is_some();
    let iterations = std::env::var("HIERARCHOS_MULTIMODAL_BENCH_STEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(7);
    assert!((3..=100).contains(&iterations));
    let bench_batch = std::env::var("HIERARCHOS_MULTIMODAL_BENCH_BATCH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    assert!((1..=4).contains(&bench_batch));
    let warmup = 2;
    let device = VulkanDevice::new()?;
    let cpu: Value = serde_json::from_slice(include_bytes!(
        "../../validation/fixtures/deepseek_v41_native_pipeline_avx2.json"
    ))?;
    let mut records = Vec::new();
    for case in cpu["cases"].as_array().unwrap() {
        let case = with_batch(case, bench_batch)?;
        if case_filter
            .as_deref()
            .is_some_and(|requested| case["name"] != requested)
        {
            continue;
        }
        let (_package, mut trainer, batch, types, patches, start) = setup(device.clone(), &case)?;
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
            json!({"case":case["name"],"kind":"native_v41","batch":bench_batch,"samples":samples,
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
        if case_filter.as_deref().is_some_and(|requested| name != requested) {
            continue;
        }
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
        let telemetry_start = telemetry.then(|| benchmark_telemetry(&device)).transpose()?;
        for index in 0..warmup + iterations {
            let before = device.submission_count();
            let aten_calls_before = crate::transformer::aten_vector_width_call_count();
            let telemetry_before = telemetry.then(|| benchmark_telemetry(&device)).transpose()?;
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
                let mut sample = json!({"total_ms":ms,"submissions":device.submission_count()-before});
                if telemetry {
                    sample["aten_width_calls"] = json!(
                        crate::transformer::aten_vector_width_call_count()
                            .saturating_sub(aten_calls_before)
                    );
                }
                if let Some(before) = telemetry_before {
                    sample["telemetry"] = benchmark_telemetry_delta(
                        &before,
                        &benchmark_telemetry(&device)?,
                    );
                }
                samples.push(sample);
            }
        }
        let output = if let Some((d, m)) = &decoder {
            graph.forward_seq2seq(&ids, &mask, d, m, &pixels)?
        } else {
            graph.forward(&ids, &mask, &pixels)?
        };
        let final_telemetry = match telemetry_start.as_ref() {
            Some(start) => benchmark_telemetry_delta(start, &benchmark_telemetry(&device)?),
            None => Value::Null,
        };
        records.push(json!({"case":name,"kind":"shared_image","samples":samples,
            "final_logits_bits":bits(&output.logits),"telemetry_total":final_telemetry}));
    }
    // Per-call-site submission accounting for the whole run. Every timed
    // iteration performs the same work, so dividing by the step count gives the
    // per-step cost of each site.
    if crate::vulkan::submission_report::enabled() {
        // Every recorded case performs `warmup + iterations` timed steps, so the
        // run total must be divided by the number of case-steps. Dividing by the
        // per-case step count alone inflated every per-step figure by the number
        // of cases (five), which is what made an earlier profile report more
        // retirements per step than the step's own native submission count.
        let steps = ((warmup + iterations) * records.len()) as u64;
        let mut report = crate::vulkan::submission_report::take();
        report.sort_by_key(|&(_, _, count)| std::cmp::Reverse(count));
        for (file, line, count) in report {
            eprintln!(
                "hierarchos_submission_report file={file} line={line} total={count} per_step={:.2}",
                count as f64 / steps as f64
            );
        }
        let mut retirements = crate::vulkan::retirement_report::take();
        retirements.sort_by_key(|&(_, _, count, _, _, _)| std::cmp::Reverse(count));
        for (file, line, count, dispatches, elapsed, host) in retirements {
            eprintln!(
                "hierarchos_retirement_report file={file} line={line} total={count} per_step={:.2} dispatches={dispatches} total_ms={:.3} ms_per_step={:.3} host_ms={:.3} host_ms_per_step={:.3} wait_ms_per_step={:.3}",
                count as f64 / steps as f64,
                elapsed.as_secs_f64() * 1000.0,
                elapsed.as_secs_f64() * 1000.0 / steps as f64,
                host.as_secs_f64() * 1000.0,
                host.as_secs_f64() * 1000.0 / steps as f64,
                elapsed.saturating_sub(host).as_secs_f64() * 1000.0 / steps as f64,
            );
        }
    }
    fs::write(
        destination,
        serde_json::to_vec_pretty(
            &json!({"device":device.name(),"batch":bench_batch,"warmup":warmup,"iterations":iterations,"records":records}),
        )?,
    )?;
    Ok(())
}

type BenchmarkTelemetry = (
    crate::VulkanMemoryStats,
    crate::VulkanSubmissionArenaStats,
    crate::VulkanPendingClearStats,
);

fn benchmark_telemetry(device: &crate::VulkanDevice) -> Result<BenchmarkTelemetry> {
    Ok((
        device.memory_stats()?,
        device.submission_arena_stats()?,
        device.pending_clear_stats()?,
    ))
}

fn benchmark_telemetry_delta(before: &BenchmarkTelemetry, after: &BenchmarkTelemetry) -> Value {
    let (memory0, arena0, clears0) = before;
    let (memory1, arena1, clears1) = after;
    json!({
        "buffer_allocations": memory1.buffer_allocations_total.saturating_sub(memory0.buffer_allocations_total),
        "buffer_frees": memory1.buffer_frees_total.saturating_sub(memory0.buffer_frees_total),
        "live_buffer_bytes": memory1.live_buffer_bytes,
        "peak_live_buffer_bytes": memory1.peak_live_buffer_bytes,
        "submissions_reaped": arena1.timeline_reaped_submissions.saturating_sub(arena0.timeline_reaped_submissions),
        "retirement_latency_ns": arena1.timeline_retirement_latency_ns_total.saturating_sub(arena0.timeline_retirement_latency_ns_total),
        "retirement_samples": arena1.timeline_retirement_latency_samples.saturating_sub(arena0.timeline_retirement_latency_samples),
        "buffer_reuse": arena1.buffer_allocation_reused.saturating_sub(arena0.buffer_allocation_reused),
        "scratch_allocations": arena1.scratch_slab_allocated.saturating_sub(arena0.scratch_slab_allocated),
        "command_buffers_allocated": arena1.command_buffer_allocated.saturating_sub(arena0.command_buffer_allocated),
        "command_buffers_reused": arena1.command_buffer_reused.saturating_sub(arena0.command_buffer_reused),
        "descriptor_pools_allocated": arena1.descriptor_pool_allocated.saturating_sub(arena0.descriptor_pool_allocated),
        "descriptor_pools_reused": arena1.descriptor_pool_reused.saturating_sub(arena0.descriptor_pool_reused),
        "pending_clear_enqueued": clears1.enqueued.saturating_sub(clears0.enqueued),
        "pending_clear_scans": clears1.scans.saturating_sub(clears0.scans),
        "pending_clear_scanned_entries": clears1.scanned_entries.saturating_sub(clears0.scanned_entries),
        "pending_clear_pruned": clears1.pruned.saturating_sub(clears0.pruned),
        "pending_clear_recorded": clears1.recorded.saturating_sub(clears0.recorded),
        "pending_clear_entries": clears1.pending_entries,
        "pending_clear_max_entries": clears1.max_entries.max(clears0.max_entries),
        "kernel_timestamp_samples": arena1.kernel_timestamp_profile_samples.saturating_sub(arena0.kernel_timestamp_profile_samples),
        "kernel_timestamp_dispatches": arena1.kernel_timestamp_profile_dispatches.saturating_sub(arena0.kernel_timestamp_profile_dispatches),
        "kernel_gpu_ns": arena1.kernel_timestamp_profile_gpu_ns_total.saturating_sub(arena0.kernel_timestamp_profile_gpu_ns_total),
    })
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

/// Cleanup is deliberately narrower than the package root: only paths owned by
/// this test invocation are removed, and an unrelated directory in the same
/// temporary root survives. This keeps interrupted state harmless without
/// turning cleanup into a broad `%TEMP%` sweep.
#[test]
fn native_checkpoint_cleanup_is_scoped_to_test_owned_paths() -> Result<()> {
    let package = Package::new(&fixture()?, false)?;
    let root = package.root.clone();
    let checkpoint = root.join("joint-step1");
    let unrelated = root.join("unrelated-owner");
    fs::create_dir_all(&checkpoint)?;
    fs::create_dir_all(&unrelated)?;
    fs::write(checkpoint.join("state.bin"), b"checkpoint")?;
    fs::write(unrelated.join("state.bin"), b"keep")?;

    drop(OwnedCheckpoints {
        root: root.clone(),
        paths: vec![checkpoint.clone()],
    });

    assert!(!checkpoint.exists());
    assert!(unrelated.is_dir());
    assert_eq!(fs::read(unrelated.join("state.bin"))?, b"keep");
    fs::remove_dir_all(&root)?;
    Ok(())
}
