//! Architecture checks for Hierarchos only; no transformer implementation is used.
//! Requires a Vulkan device. Run with --test-threads=1.
use crate::{
    read_model_ltm_running_state, AdamWHyperParams, HierarchosLabeledSequenceObjective,
    HierarchosLanguageLossSummary, HierarchosRawTokenLabeledSequenceInput,
    HierarchosTapeMemoryPolicy, HierarchosTokenTapeReadbackPolicy, HierarchosTokenTapeTrainResult,
    HierarchosTokenTapeUpdateMode, HierarchosTrainingGraph, RwkvParameterSnapshot, VulkanDevice,
};
use anyhow::Result;
use hierarchos_inference::{
    initialize_model_package, HierarchosModel, NativeBootstrapConfig, RosaState,
};
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

fn objective() -> HierarchosLabeledSequenceObjective {
    HierarchosLabeledSequenceObjective {
        z_loss_weight: 0.0,
        ponder_loss_weight: 0.0,
        commitment_loss_weight: 0.0,
        max_ce_loss_for_backward: 0.0,
        max_ponder_cost_for_backward: 0.0,
        max_commitment_cost_for_backward: 0.0,
    }
}

#[test]
fn network_tcp_gradient_merge_matches_existing_transport_and_reload_logits() -> Result<()> {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(network_tcp_gradient_merge_impl)?
        .join()
        .map_err(|_| anyhow::anyhow!("network gradient parity worker panicked"))?
}

fn network_tcp_gradient_merge_impl() -> Result<()> {
    use crate::{
        distributed::{NetworkGradientSource, NetworkSendChannel},
        HierarchosFullModelReplicaState,
    };
    use std::net::TcpListener;
    let path = fixture()?;
    let device = VulkanDevice::new()?;
    let mut reference = graph(&device, &path, 1)?;
    let mut primary = graph(&device, &path, 1)?;
    let mut replica = graph(&device, &path, 1)?;
    for (g, ids, labels, mode) in [
        (
            &mut reference,
            [1, 2],
            [2i64, 3],
            HierarchosTokenTapeUpdateMode::BeginAccumulation,
        ),
        (
            &mut primary,
            [1, 2],
            [2i64, 3],
            HierarchosTokenTapeUpdateMode::BeginAccumulation,
        ),
        (
            &mut replica,
            [4, 5],
            [5i64, 6],
            HierarchosTokenTapeUpdateMode::BeginAccumulation,
        ),
    ] {
        run(g, 1, &ids, &labels, None, None, objective(), mode)?;
    }
    let source = replica.full_model_pending_gradient_transport_source()?;
    reference.accumulate_full_model_pending_gradients_streamed_from_source(&source, 257)?;
    let gradients = replica.full_model_pending_gradient_snapshots()?;
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?;
    let sender = std::thread::spawn(move || -> Result<()> {
        let mut sink = NetworkSendChannel::connect("tcp", addr, false, 257, 2)?;
        for tensor in gradients {
            for chunk in tensor.values.chunks(257) {
                sink.send_chunk(chunk)?;
            }
        }
        sink.flush()
    });
    let (mut receiver, _control) =
        NetworkGradientSource::connect_with_bound_listener("tcp", listener, 257, 2)?;
    let statistics = primary.accumulate_full_model_pending_gradients_streamed_from_remote(
        &mut receiver,
        &source.metadata,
        257,
    )?;
    assert_eq!(
        statistics.backend,
        crate::VulkanGradientTransportBackend::NetworkTcp
    );
    sender.join().unwrap()?;
    reference.finish_full_model_accumulation_with_gradient_clipping(hyper(), 0.0)?;
    primary.finish_full_model_accumulation_with_gradient_clipping(hyper(), 0.0)?;
    let expected = reference.full_model_replica_state()?.wire_encode();
    let actual = primary.full_model_replica_state()?.wire_encode();
    assert!(
        expected == actual,
        "parameters, Adam moments/clocks and metadata must match the existing transport exactly"
    );

    // An unaligned network payload must round-trip without a bytemuck panic.
    let mut unaligned = vec![0u8];
    unaligned.extend_from_slice(&actual.1);
    assert!(
        HierarchosFullModelReplicaState::wire_decode(&actual.0, &unaligned[1..])?.wire_encode()
            == actual,
        "unaligned replica wire round-trip must preserve each slot's moments"
    );
    let mut broken: serde_json::Value = serde_json::from_str(&actual.0)?;
    broken["optimizer_slot_steps"].as_array_mut().unwrap().pop();
    assert!(HierarchosFullModelReplicaState::wire_decode(&broken.to_string(), &actual.1).is_err());
    assert!(HierarchosFullModelReplicaState::wire_decode(
        &actual.0,
        &actual.1[..actual.1.len() - 4]
    )
    .is_err());

    let mut all_logits = Vec::new();
    for (name, g) in [("reference", &reference), ("network", &primary)] {
        let checkpoint = path.join(name);
        g.export_training_checkpoint_package(&path, &checkpoint)?;
        let inference = HierarchosModel::load(&checkpoint)?;
        let mut state = inference.new_state();
        let mut bits = Vec::new();
        for token in [1, 2, 4, 5] {
            bits.extend(
                inference
                    .step(token, &mut state)?
                    .iter()
                    .map(|v| v.to_bits()),
            );
        }
        all_logits.push(bits);
    }
    assert_eq!(
        all_logits[0], all_logits[1],
        "post-training recurrent logits"
    );
    fs::remove_dir_all(path)?;
    Ok(())
}

fn hyper() -> AdamWHyperParams {
    AdamWHyperParams {
        lr: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    }
}

fn fixture() -> Result<PathBuf> {
    let path = std::env::temp_dir().join(format!(
        "hierarchos-causal-lm-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    let mut c = NativeBootstrapConfig::for_vocab(16);
    c.context_dim = 32;
    c.h_hidden = 32;
    c.l_hidden = 32;
    c.persistent_dim = 8;
    c.ltm_slots = 8;
    c.ltm_key_dim = 8;
    c.ltm_val_dim = 8;
    c.ltm_topk = 2;
    c.h_stride = 2;
    c.max_h_steps = 3;
    c.max_l_steps = 2;
    c.rwkv_head_size = 32;
    c.token_adapter_rank = 8;
    c.rosa_max_context = 8;
    c.memory_gate_warmup_steps = 0;
    initialize_model_package(&path, &c)?;
    // Isolate the causal objective from the independently scheduled LTM writer objective.
    for name in ["hierarchos_config.json", "hierarchos_rust_config.json"] {
        let file = path.join(name);
        let mut config: serde_json::Value = serde_json::from_slice(&fs::read(&file)?)?;
        config["ltm_value_alignment_weight"] = serde_json::json!(0.0);
        fs::write(file, serde_json::to_vec_pretty(&config)?)?;
    }
    Ok(path)
}

fn graph(device: &VulkanDevice, path: &PathBuf, batch: usize) -> Result<HierarchosTrainingGraph> {
    HierarchosTrainingGraph::from_model_package_with_token_frontend(
        device.clone(),
        path,
        batch,
        3,
        2,
        batch,
        batch,
    )
}

fn run(
    g: &mut HierarchosTrainingGraph,
    batch: usize,
    ids: &[u32],
    labels: &[i64],
    mask: Option<&[f32]>,
    weights: Option<&[f32]>,
    obj: HierarchosLabeledSequenceObjective,
    mode: HierarchosTokenTapeUpdateMode,
) -> Result<HierarchosTokenTapeTrainResult> {
    let tokens = ids.len() / batch;
    let contexts = vec![0.0; batch * 32];
    let mut tape = g.create_zero_token_tape(batch, tokens)?;
    g.train_raw_token_labeled_sequence_with_update_mode(
        &mut tape,
        &HierarchosRawTokenLabeledSequenceInput {
            tokens,
            input_ids: ids,
            labels,
            attention_mask: mask,
            loss_weights: weights,
            initial_previous_context: &contexts,
            initial_target_context: &contexts,
            global_pos_offset: 0,
            reset_rosa_at_start: true,
            pytorch_tbptt_chunk_size: None,
        },
        obj,
        hyper(),
        mode,
    )
}

fn close(a: &[f32], b: &[f32], tolerance: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label} shape");
    let error = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            assert!(x.is_finite() && y.is_finite(), "{label}: nonfinite");
            (x - y).abs()
        })
        .fold(0.0f32, f32::max);
    assert!(
        error <= tolerance,
        "{label}: max_abs={error}, tolerance={tolerance}"
    );
}

fn close_registry(a: &[RwkvParameterSnapshot], b: &[RwkvParameterSnapshot], tolerance: f32) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.name, b.name);
        close(&a.values, &b.values, tolerance, &a.name);
    }
}

#[test]
fn hierarchos_causal_lm_architecture_contract() -> Result<()> {
    with_trainer_stack(architecture_contract)
}

// Match the native trainer's explicit Windows host stack for the large graph.
fn with_trainer_stack(check: fn() -> Result<()>) -> Result<()> {
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(check)?
        .join()
        .unwrap()
}

fn architecture_contract() -> Result<()> {
    let path = fixture()?;
    let device = VulkanDevice::new()?;
    eprintln!("Hierarchos causal-LM device: {}", device.name());
    let ids = [1, 2, 1, 2];
    let labels = [1, 2, 1, 2];
    let mut g = graph(&device, &path, 1)?;
    let baseline = run(
        &mut g,
        1,
        &ids,
        &labels,
        None,
        None,
        objective(),
        HierarchosTokenTapeUpdateMode::BeginAccumulation,
    )?;
    let gradients = g.full_model_pending_gradient_snapshots()?;
    assert!(gradients
        .iter()
        .all(|p| p.values.iter().all(|v| v.is_finite())));
    for prefix in [
        "lm_head",
        "in_proj",
        "h_rnn",
        "l_rnn",
        "h_to_context",
        "l_to_out",
    ] {
        assert!(
            gradients
                .iter()
                .filter(|p| p.name.starts_with(prefix))
                .any(|p| p.values.iter().any(|v| v.abs() > 1e-10)),
            "no CE gradient into {prefix}"
        );
    }
    assert_eq!(
        gradients
            .iter()
            .filter(|p| p.name == "lm_head.weight")
            .count(),
        1
    );
    eprintln!(
        "PASS finite gradients and tied parameter registry ({} tensors)",
        gradients.len()
    );

    // Independent scalar Rust inference supplies a next-token CE oracle.
    let model = HierarchosModel::load(&path)?;
    let mut state = model.new_state();
    let logits = model.prefill(&ids, &mut state)?;
    let expected = logits
        .iter()
        .take(ids.len() - 1)
        .enumerate()
        .map(|(t, row)| {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            max + row.iter().map(|v| (v - max).exp()).sum::<f32>().ln()
                - row[labels[t + 1] as usize]
        })
        .collect::<Vec<_>>();
    close(
        &baseline.losses[..3],
        &expected,
        2e-5,
        "CPU/Vulkan shifted CE",
    );
    let state = model.snapshot_runtime_state(&state)?;
    close(
        &baseline.final_h_packed_state,
        &state.h_state.values,
        2e-5,
        "CPU/Vulkan H state",
    );
    close(
        &baseline.final_l_packed_state,
        &state.l_state.values,
        2e-5,
        "CPU/Vulkan L state",
    );
    eprintln!("PASS independent CPU/Vulkan causal loss and recurrent state");

    // The first label is never a target, and z-loss uses the same causal positions.
    g = graph(&device, &path, 1)?;
    let z_weight = 0.01;
    let z = run(
        &mut g,
        1,
        &ids,
        &[-100, 2, 1, 2],
        None,
        None,
        HierarchosLabeledSequenceObjective {
            z_loss_weight: z_weight,
            ..objective()
        },
        HierarchosTokenTapeUpdateMode::BeginAccumulation,
    )?;
    let expected_z = logits
        .iter()
        .take(3)
        .zip(&expected)
        .map(|(row, ce)| {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let log_z = max + row.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
            ce + z_weight * log_z * log_z
        })
        .collect::<Vec<_>>();
    close(
        &z.losses[..3],
        &expected_z,
        2e-5,
        "CPU/Vulkan CE plus z-loss",
    );
    close(
        &z.final_h_packed_state,
        &baseline.final_h_packed_state,
        0.0,
        "labels do not enter H forward",
    );
    close(
        &z.final_l_packed_state,
        &baseline.final_l_packed_state,
        0.0,
        "labels do not enter L forward",
    );
    eprintln!("PASS z-loss oracle and label/forward separation");

    // An unseen suffix must not affect earlier predictions or gradients when its labels are ignored.
    g = graph(&device, &path, 1)?;
    let short = run(
        &mut g,
        1,
        &ids[..3],
        &labels[..3],
        None,
        None,
        objective(),
        HierarchosTokenTapeUpdateMode::BeginAccumulation,
    )?;
    let short_grads = g.full_model_pending_gradient_snapshots()?;
    g = graph(&device, &path, 1)?;
    let suffix = run(
        &mut g,
        1,
        &[1, 2, 1, 9, 8],
        &[1, 2, 1, -100, -100],
        None,
        None,
        objective(),
        HierarchosTokenTapeUpdateMode::BeginAccumulation,
    )?;
    close(
        &short.losses[..2],
        &suffix.losses[..2],
        0.0,
        "future suffix causality",
    );
    close_registry(
        &short_grads,
        &g.full_model_pending_gradient_snapshots()?,
        2e-7,
    );
    eprintln!("PASS future-token isolation and ignored suffix gradients");

    // Padding must freeze H/L/context and leave every parameter gradient unchanged.
    g = graph(&device, &path, 1)?;
    let padded = run(
        &mut g,
        1,
        &[1, 2, 1, 9, 8],
        &[1, 2, 1, -100, -100],
        Some(&[1., 1., 1., 0., 0.]),
        Some(&[1., 1., 1., 0., 0.]),
        objective(),
        HierarchosTokenTapeUpdateMode::BeginAccumulation,
    )?;
    close(
        &short.final_h_packed_state,
        &padded.final_h_packed_state,
        0.0,
        "padded H",
    );
    close(
        &short.final_l_packed_state,
        &padded.final_l_packed_state,
        0.0,
        "padded L",
    );
    close(
        short.final_previous_context.as_ref().unwrap(),
        padded.final_previous_context.as_ref().unwrap(),
        0.0,
        "padded previous context",
    );
    close(
        short.final_target_context.as_ref().unwrap(),
        padded.final_target_context.as_ref().unwrap(),
        0.0,
        "padded target context",
    );
    close_registry(
        &short_grads,
        &g.full_model_pending_gradient_snapshots()?,
        2e-7,
    );
    eprintln!("PASS padding state identity and gradient isolation");
    assert_eq!(
        g.snapshot_portable_frontend_state(1)?.1,
        vec![vec![1, 2, 1]],
        "padding must not enter persistent ROSA history"
    );

    // Weight mass is normalized once per complete optimizer window, including unequal microbatches.
    let mut joint = graph(&device, &path, 2)?;
    run(
        &mut joint,
        2,
        &[1, 2, 1, 3, 4, 5],
        &[1, 2, 1, 3, -100, 5],
        None,
        Some(&[1., 0.5, 2., 1., 1., 3.]),
        objective(),
        HierarchosTokenTapeUpdateMode::Step,
    )?;
    let mut split = graph(&device, &path, 1)?;
    run(
        &mut split,
        1,
        &[1, 2, 1],
        &[1, 2, 1],
        None,
        Some(&[1., 0.5, 2.]),
        objective(),
        HierarchosTokenTapeUpdateMode::BeginAccumulation,
    )?;
    run(
        &mut split,
        1,
        &[3, 4, 5],
        &[3, -100, 5],
        None,
        Some(&[1., 1., 3.]),
        objective(),
        HierarchosTokenTapeUpdateMode::FinishAccumulation,
    )?;
    close_registry(
        &joint.full_model_parameter_snapshots()?,
        &split.full_model_parameter_snapshots()?,
        2e-6,
    );
    eprintln!("PASS unequal weighted microbatch optimizer equivalence");

    let before = split.full_model_parameter_snapshots()?;
    assert!(run(
        &mut split,
        1,
        &ids,
        &[-100; 4],
        None,
        None,
        objective(),
        HierarchosTokenTapeUpdateMode::Step
    )
    .is_err());
    close_registry(&before, &split.full_model_parameter_snapshots()?, 0.0);
    eprintln!("PASS empty supervision rejected without a parameter update");

    // A real full-model optimization trajectory must improve next-token CE,
    // and its exported weights must remain usable by the inference runtime.
    drop(g);
    drop(joint);
    drop(split);
    let mut learner = graph(&device, &path, 1)?;
    let mut trajectory = Vec::new();
    for _ in 0..12 {
        let result = run(
            &mut learner,
            1,
            &ids,
            &labels,
            None,
            None,
            objective(),
            HierarchosTokenTapeUpdateMode::Step,
        )?;
        trajectory.push(result.losses[..3].iter().sum::<f32>() / 3.0);
    }
    assert!(
        trajectory[11] < trajectory[0],
        "CE did not improve: {trajectory:?}"
    );
    let trained = path.join("trained");
    fs::create_dir(&trained)?;
    fs::copy(
        path.join("hierarchos_rust_config.json"),
        trained.join("hierarchos_rust_config.json"),
    )?;
    learner.save_full_model_checkpoint(
        path.join("model.safetensors"),
        trained.join("model.safetensors"),
    )?;
    let model = HierarchosModel::load(&trained)?;
    let mut state = model.new_state();
    let logits = model.prefill(&ids, &mut state)?;
    let ce = logits
        .iter()
        .take(3)
        .enumerate()
        .map(|(t, row)| {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            max + row.iter().map(|v| (v - max).exp()).sum::<f32>().ln()
                - row[labels[t + 1] as usize]
        })
        .sum::<f32>()
        / 3.0;
    assert!(ce.is_finite() && ce < trajectory[0]);
    eprintln!(
        "PASS 12-step native learning and export: CE {} -> {ce}",
        trajectory[0]
    );
    Ok(())
}

#[test]
fn hierarchos_causal_lm_dense_sparse_objective_equivalence() -> Result<()> {
    with_trainer_stack(dense_sparse_objective_equivalence)
}

fn dense_sparse_objective_equivalence() -> Result<()> {
    let path = fixture()?;
    let device = VulkanDevice::new()?;
    let ids = [1, 2, 1, 2, 0];
    let labels = [-100, 2, 1, 2, -100];
    let mask = [1., 1., 1., 1., 0.];
    let weights = [0., 0.5, 2., 1., 0.];
    let contexts = [0.; 32];
    let obj = HierarchosLabeledSequenceObjective {
        z_loss_weight: 0.01,
        ponder_loss_weight: 0.02,
        commitment_loss_weight: 0.03,
        ..objective()
    };
    for chunk_size in [None, Some(2)] {
        let input = HierarchosRawTokenLabeledSequenceInput {
            tokens: ids.len(),
            input_ids: &ids,
            labels: &labels,
            attention_mask: Some(&mask),
            loss_weights: Some(&weights),
            initial_previous_context: &contexts,
            initial_target_context: &contexts,
            global_pos_offset: 0,
            reset_rosa_at_start: true,
            pytorch_tbptt_chunk_size: chunk_size,
        };
        let mut dense = graph(&device, &path, 1)?;
        let mut tape = dense.create_zero_token_tape(1, ids.len())?;
        let expected = dense.train_raw_token_labeled_sequence_with_update_mode(
            &mut tape,
            &input,
            obj,
            hyper(),
            HierarchosTokenTapeUpdateMode::BeginAccumulation,
        )?;
        let gradients = dense.full_model_pending_gradient_snapshots()?;
        for stride in [1, 2] {
            let mut planned = graph(&device, &path, 1)?;
            let actual = planned
                .train_zero_state_raw_token_labeled_sequences_with_plan_and_update_mode(
                    1,
                    &[HierarchosRawTokenLabeledSequenceInput { ..input }],
                    obj,
                    hyper(),
                    HierarchosTokenTapeUpdateMode::BeginAccumulation,
                    1,
                    stride,
                    HierarchosTapeMemoryPolicy::default(),
                    HierarchosTokenTapeReadbackPolicy::Full,
                )?;
            close(
                &expected.losses,
                &actual.sequences[0].losses,
                2e-7,
                "dense/planned losses",
            );
            close(
                &expected.final_h_packed_state,
                &actual.sequences[0].final_h_packed_state,
                2e-7,
                "dense/planned H",
            );
            close(
                &expected.final_l_packed_state,
                &actual.sequences[0].final_l_packed_state,
                2e-7,
                "dense/planned L",
            );
            close_registry(
                &gradients,
                &planned.full_model_pending_gradient_snapshots()?,
                2e-7,
            );
            assert_eq!(
                planned.snapshot_portable_frontend_state(1)?.1,
                vec![vec![1, 2, 1, 2]]
            );
            eprintln!("PASS full objective dense/planned stride={stride}, TBPTT={chunk_size:?}");
        }
    }
    Ok(())
}

#[test]
fn hierarchos_causal_lm_weighted_language_metric_matches_cpu() -> Result<()> {
    with_trainer_stack(weighted_language_metric_matches_cpu)
}

fn weighted_language_metric_matches_cpu() -> Result<()> {
    let path = fixture()?;
    let device = VulkanDevice::new()?;
    let model = HierarchosModel::load(&path)?;
    let batch = 2;
    let tokens = 5;
    let label_cols = tokens + 1;
    let ids = [1, 2, 1, 2, 3, 5, 6, 5, 0, 0];
    let mask = [1., 1., 1., 1., 1., 1., 1., 1., 0., 0.];
    let labels = [-100, -100, 1, 2, 3, 4, -100, -100, 5, -100, -100, -100];
    // Ignored prompt labels deliberately retain positive weights. They must
    // contribute neither numerator nor denominator; padded labels have zero.
    let weights = [11., 7., 0.5, 2., 0.25, 1.5, 13., 17., 3.25, 0., 0., 0.];
    let contexts = vec![0.; batch * 32];
    let obj = HierarchosLabeledSequenceObjective {
        z_loss_weight: 0.01,
        ponder_loss_weight: 0.02,
        commitment_loss_weight: 0.03,
        // The metric describes CE plus z-loss before this backward-only cap,
        // and excludes the independently normalized auxiliary objectives.
        max_ce_loss_for_backward: 0.1,
        ..objective()
    };
    let input = HierarchosRawTokenLabeledSequenceInput {
        tokens,
        input_ids: &ids,
        labels: &labels,
        attention_mask: Some(&mask),
        loss_weights: Some(&weights),
        initial_previous_context: &contexts,
        initial_target_context: &contexts,
        global_pos_offset: 0,
        reset_rosa_at_start: true,
        pytorch_tbptt_chunk_size: Some(2),
    };

    let mut expected_sum = 0.0f64;
    let mut expected_mass = 0.0f64;
    for row in 0..batch {
        let real_tokens = mask[row * tokens..(row + 1) * tokens]
            .iter()
            .filter(|&&active| active != 0.)
            .count();
        let mut state = model.new_state();
        let logits = model.prefill(&ids[row * tokens..row * tokens + real_tokens], &mut state)?;
        for (token_index, logits) in logits.iter().enumerate() {
            let label_index = row * label_cols + token_index + 1;
            let target = labels[label_index];
            if target == -100 || weights[label_index] == 0. {
                continue;
            }
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let log_z = max
                + logits
                    .iter()
                    .map(|value| (value - max).exp())
                    .sum::<f32>()
                    .ln();
            let language_loss = log_z - logits[target as usize] + obj.z_loss_weight * log_z * log_z;
            expected_sum += f64::from(language_loss) * f64::from(weights[label_index]);
            expected_mass += f64::from(weights[label_index]);
        }
    }
    assert_eq!(expected_mass, 7.5);
    let expected_mean = expected_sum / expected_mass;
    assert!(expected_mean > f64::from(obj.max_ce_loss_for_backward));

    let mut dense = graph(&device, &path, batch)?;
    let mut tape = dense.create_zero_token_tape(batch, tokens)?;
    let result = dense.train_raw_token_labeled_sequence_with_update_mode(
        &mut tape,
        &input,
        obj,
        hyper(),
        HierarchosTokenTapeUpdateMode::BeginAccumulation,
    )?;
    assert_eq!(result.language_loss.supervision_mass, expected_mass);
    assert!(
        (result.language_loss.weighted_sum - expected_sum).abs() < 2e-4,
        "CPU/Vulkan weighted language numerator: expected {expected_sum}, got {:?}",
        result.language_loss,
    );
    assert!((result.language_loss.mean().unwrap() - expected_mean).abs() < 2e-5);
    let diagnostic_mean = result
        .losses
        .iter()
        .map(|&loss| f64::from(loss))
        .sum::<f64>()
        / result.losses.len() as f64;
    assert!(
        (diagnostic_mean - expected_mean).abs() > 1e-4,
        "the fixture must distinguish the weighted language metric from its all-row diagnostic mean",
    );

    let mut merged = HierarchosLanguageLossSummary::default();
    let mut independent_means = Vec::new();
    for row in 0..batch {
        let mut independent = graph(&device, &path, 1)?;
        let actual = run(
            &mut independent,
            1,
            &ids[row * tokens..(row + 1) * tokens],
            &labels[row * label_cols..(row + 1) * label_cols],
            Some(&mask[row * tokens..(row + 1) * tokens]),
            Some(&weights[row * label_cols..(row + 1) * label_cols]),
            obj,
            HierarchosTokenTapeUpdateMode::BeginAccumulation,
        )?;
        independent_means.push(actual.language_loss.mean().unwrap());
        merged.merge(actual.language_loss);
    }
    assert_eq!(merged.supervision_mass, expected_mass);
    assert!((merged.weighted_sum - result.language_loss.weighted_sum).abs() < 2e-6);
    assert!((merged.mean().unwrap() - expected_mean).abs() < 2e-5);
    let mean_of_means = independent_means.iter().sum::<f64>() / batch as f64;
    assert!(
        (merged.mean().unwrap() - mean_of_means).abs() > 1e-6,
        "unequal supervision masses must distinguish merged metrics from a mean of means",
    );

    for stride in [1, 3] {
        let mut planned = graph(&device, &path, batch)?;
        let actual = planned
            .train_zero_state_raw_token_labeled_sequences_with_plan_and_update_mode(
                batch,
                &[HierarchosRawTokenLabeledSequenceInput { ..input }],
                obj,
                hyper(),
                HierarchosTokenTapeUpdateMode::BeginAccumulation,
                1,
                stride,
                HierarchosTapeMemoryPolicy::default(),
                HierarchosTokenTapeReadbackPolicy::LossOnly,
            )?;
        let actual = &actual.sequences[0];
        close(
            &actual.losses,
            &result.losses,
            2e-7,
            "weighted metric dense/planned diagnostics",
        );
        assert_eq!(actual.language_loss.supervision_mass, expected_mass);
        assert!(
            (actual.language_loss.weighted_sum - result.language_loss.weighted_sum).abs() < 2e-6,
            "weighted metric dense/planned numerator differs at stride {stride}",
        );
        assert!((actual.language_loss.mean().unwrap() - expected_mean).abs() < 2e-5);
    }
    eprintln!(
        "PASS weighted CPU/Vulkan language metric: sum={expected_sum}, mass={expected_mass}, \
         mean={expected_mean}, diagnostic mean={diagnostic_mean}"
    );
    Ok(())
}

#[test]
fn hierarchos_causal_lm_continuation_matches_cpu_across_manager_and_rosa_boundaries() -> Result<()>
{
    with_trainer_stack(continuation_matches_cpu_across_manager_and_rosa_boundaries)
}

fn continuation_matches_cpu_across_manager_and_rosa_boundaries() -> Result<()> {
    let path = fixture()?;
    let device = VulkanDevice::new()?;
    let model = HierarchosModel::load(&path)?;
    let prefix = [1, 2, 1, 2, 1, 2, 1];
    let ids = [2, 1, 2, 1, 2, 3];
    // One extra label supervises the last input token at a TBPTT handoff.
    let labels = [-100, 1, 2, 1, 2, 3, 4];
    let mut cpu_state = model.new_state();
    model.prefill(&prefix, &mut cpu_state)?;
    let initial = model.snapshot_runtime_state(&cpu_state)?;
    for (name, values) in [
        ("H", &initial.h_state.values),
        ("L", &initial.l_state.values),
        ("previous context", &initial.prev_context),
        ("target context", &initial.target_context),
    ] {
        assert!(
            values.iter().any(|value| value.abs() > 1e-8),
            "zero {name} seed"
        );
    }
    assert_ne!(initial.prev_context, initial.target_context);
    assert_eq!(initial.position % 2, 1, "resume between manager updates");
    assert_eq!(
        initial.rosa.tokens.len(),
        7,
        "resume immediately before ROSA rollover"
    );

    let logits = model.prefill(&ids, &mut cpu_state)?;
    let expected = logits
        .iter()
        .enumerate()
        .map(|(t, row)| {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            max + row
                .iter()
                .map(|value| (value - max).exp())
                .sum::<f32>()
                .ln()
                - row[labels[t + 1] as usize]
        })
        .collect::<Vec<_>>();
    let final_cpu = model.snapshot_runtime_state(&cpu_state)?;
    let mut rosa = RosaState::from_snapshot(&initial.rosa)?;
    let expected_predictions = ids
        .iter()
        .map(|&id| vec![rosa.predict_and_push(id, 8).map_or(-1, i64::from)])
        .collect::<Vec<_>>();
    assert_eq!(expected_predictions[0], vec![1]);
    assert_eq!(expected_predictions[1], vec![-1], "ROSA resets at its cap");

    for chunk_size in [None, Some(2)] {
        let input = HierarchosRawTokenLabeledSequenceInput {
            tokens: ids.len(),
            input_ids: &ids,
            labels: &labels,
            attention_mask: None,
            loss_weights: None,
            initial_previous_context: &initial.prev_context,
            initial_target_context: &initial.target_context,
            global_pos_offset: initial.position as u64,
            reset_rosa_at_start: false,
            pytorch_tbptt_chunk_size: chunk_size,
        };
        let mut dense = graph(&device, &path, 1)?;
        let mut carriers = dense.snapshot_portable_running_carriers(
            1,
            &initial.h_state.values,
            &initial.l_state.values,
            &initial.prev_context,
            &initial.target_context,
            read_model_ltm_running_state(&path.join("model.safetensors"))?,
        )?;
        carriers.rosa_token_histories = vec![initial.rosa.tokens.clone()];
        carriers.drift_state.values.clone_from(&initial.final_drift);
        dense.restore_portable_running_carriers(1, &carriers)?;
        let mut tape = dense.create_token_tape(
            1,
            ids.len(),
            &initial.h_state.values,
            &initial.l_state.values,
        )?;
        let result = dense.train_raw_token_labeled_sequence_with_update_mode(
            &mut tape,
            &input,
            objective(),
            hyper(),
            HierarchosTokenTapeUpdateMode::BeginAccumulation,
        )?;
        close(
            &result.losses,
            &expected,
            2e-5,
            "continued CPU/Vulkan shifted CE",
        );
        close(
            &result.final_h_packed_state,
            &final_cpu.h_state.values,
            2e-5,
            "continued CPU/Vulkan H state",
        );
        close(
            &result.final_l_packed_state,
            &final_cpu.l_state.values,
            2e-5,
            "continued CPU/Vulkan L state",
        );
        close(
            result.final_previous_context.as_ref().unwrap(),
            &final_cpu.prev_context,
            2e-5,
            "continued CPU/Vulkan previous context",
        );
        close(
            result.final_target_context.as_ref().unwrap(),
            &final_cpu.target_context,
            2e-5,
            "continued CPU/Vulkan target context",
        );
        assert_eq!(tape.rosa_prediction_checkpoints()?, expected_predictions);
        assert_eq!(
            dense.snapshot_portable_frontend_state(1)?.1,
            vec![final_cpu.rosa.tokens.clone()],
        );
        // Planned tapes represent independent samples. Their API rejects a
        // persistent ROSA continuation before changing parameters or carriers.
        let mut planned = graph(&device, &path, 1)?;
        planned.restore_portable_running_carriers(1, &carriers)?;
        let before_parameters = planned.full_model_parameter_snapshots()?;
        let before_frontend = planned.snapshot_portable_frontend_state(1)?;
        let error = planned
            .train_raw_token_labeled_sequences_with_plan_and_update_mode_and_readback_policy(
                1,
                &[&initial.h_state.values],
                &[&initial.l_state.values],
                &[input],
                objective(),
                hyper(),
                HierarchosTokenTapeUpdateMode::BeginAccumulation,
                1,
                3,
                HierarchosTapeMemoryPolicy::default(),
                HierarchosTokenTapeReadbackPolicy::Full,
            )
            .err()
            .expect("independent planned sequences must reject ROSA continuation");
        assert!(error.to_string().contains("must reset ROSA at token zero"));
        close_registry(
            &before_parameters,
            &planned.full_model_parameter_snapshots()?,
            0.,
        );
        assert_eq!(
            planned.snapshot_portable_frontend_state(1)?,
            before_frontend
        );
        eprintln!("PASS CPU/Vulkan continuation, manager/ROSA rollover, planned continuation rejection, TBPTT={chunk_size:?}");
    }
    Ok(())
}

/// Fixture geometry for the vendor parity test: every width is 128 and the
/// causal step runs eight rows per timestep, which clears the Intel matmul
/// geometry gate (`rows >= 8`, `input >= 32`, `output >= 128`) for the
/// channel-mix and projection dispatches. The architecture-contract fixture is
/// deliberately tiny (width 32, one row), so every dispatch there stays on the
/// portable module.
fn vendor_fixture() -> Result<PathBuf> {
    let path = std::env::temp_dir().join(format!(
        "hierarchos-causal-lm-vendor-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    let mut c = NativeBootstrapConfig::for_vocab(128);
    c.context_dim = 128;
    c.h_hidden = 128;
    c.l_hidden = 128;
    c.persistent_dim = 32;
    c.ltm_slots = 8;
    c.ltm_key_dim = 32;
    c.ltm_val_dim = 32;
    c.ltm_topk = 2;
    c.h_stride = 2;
    c.max_h_steps = 3;
    c.max_l_steps = 2;
    c.rwkv_head_size = 64;
    c.token_adapter_rank = 32;
    c.rosa_max_context = 8;
    c.memory_gate_warmup_steps = 0;
    initialize_model_package(&path, &c)?;
    // Isolate the causal objective from the independently scheduled LTM writer objective.
    for name in ["hierarchos_config.json", "hierarchos_rust_config.json"] {
        let file = path.join(name);
        let mut config: serde_json::Value = serde_json::from_slice(&fs::read(&file)?)?;
        config["ltm_value_alignment_weight"] = serde_json::json!(0.0);
        fs::write(&file, serde_json::to_vec_pretty(&config)?)?;
    }
    Ok(path)
}

/// Sets a process environment variable and restores the previous value on
/// drop, so a failing assertion cannot leak the override into the suite.
struct EnvVarGuard {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(key);
        std::env::set_var(key, value);
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(value) => std::env::set_var(self.key, value),
            None => std::env::remove_var(self.key),
        }
    }
}

fn run_vendor_step(
    g: &mut HierarchosTrainingGraph,
    batch: usize,
    context_dim: usize,
    ids: &[u32],
    labels: &[i64],
) -> Result<HierarchosTokenTapeTrainResult> {
    let tokens = ids.len() / batch;
    let contexts = vec![0.0; batch * context_dim];
    let mut tape = g.create_zero_token_tape(batch, tokens)?;
    g.train_raw_token_labeled_sequence_with_update_mode(
        &mut tape,
        &HierarchosRawTokenLabeledSequenceInput {
            tokens,
            input_ids: ids,
            labels,
            attention_mask: None,
            loss_weights: None,
            initial_previous_context: &contexts,
            initial_target_context: &contexts,
            global_pos_offset: 0,
            reset_rosa_at_start: true,
            pytorch_tbptt_chunk_size: None,
        },
        objective(),
        hyper(),
        HierarchosTokenTapeUpdateMode::BeginAccumulation,
    )
}

#[test]
fn hierarchos_causal_lm_vendor_kernels_match_portable() -> Result<()> {
    with_trainer_stack(vendor_kernels_match_portable)
}

/// Compare vendor and portable modules within the same explicitly composed
/// graph. Shipping producer routes have separate stable trajectory gates: a
/// vendor module's availability does not qualify an AMD fusion replacement.
/// Dispatch counters prove all four vendor modules actually ran.
fn vendor_kernels_match_portable() -> Result<()> {
    // Module arithmetic qualification includes measured performance losers.
    // Shipping geometry decisions are exercised by the ordinary trajectory gates.
    let _modules = EnvVarGuard::set(
        crate::vendor::HIERARCHOS_VULKAN_QUALIFY_VENDOR_MODULES_ENV, "1",
    );
    let _time_mix = EnvVarGuard::set(
        crate::rwkv_time_mix_core::HIERARCHOS_VULKAN_DISABLE_TIME_MIX_FORWARD_FUSION_ENV, "1",
    );
    let _channel_mix = EnvVarGuard::set(
        crate::rwkv_channel_mix::HIERARCHOS_VULKAN_DISABLE_CHANNEL_MIX_FORWARD_FUSION_ENV, "1",
    );
    let batch = 8usize;
    let context_dim = 128usize;
    let path = vendor_fixture()?;
    let device = VulkanDevice::new()?;
    let vendor_label = crate::vendor::effective_vendor(&device).label();
    let probe = crate::vendor::VendorMatmulKernel::new(
        &device,
        crate::vendor::VendorKernelFamily::LinearForward,
    )?;
    if !probe.variant_available() {
        eprintln!("SKIP Hierarchos causal-LM vendor parity: no tuned variant on this adapter");
        return Ok(());
    }
    // Each recurrent timestep dispatches the channel mix as
    // (rows = batch, k = width = context_dim, n = 4 * width).
    assert!(
        probe.selects_variant_for(batch as u32, context_dim as u32, (4 * context_dim) as u32),
        "vendor parity fixture geometry must reach the tuned module"
    );

    let ids = [
        1u32, 2, 1, 2, 3, 1, 2, 3, 1, 2, 1, 3, 2, 1, 3, 2, 1, 2, 3, 1, 2, 1, 3, 2, 1, 2, 3, 1, 2,
        1, 3, 2,
    ];
    let labels = ids.map(|id| id as i64);
    crate::vendor::reset_variant_dispatch_count();
    let mut vendor = graph(&device, &path, batch)?;
    let vendor_result = run_vendor_step(&mut vendor, batch, context_dim, &ids, &labels)?;
    let vendor_dispatches = crate::vendor::variant_dispatch_count();
    let linear_forward_dispatches =
        crate::vendor::variant_dispatch_count_for(crate::vendor::VendorKernelFamily::LinearForward);
    let linear_bias_forward_dispatches = crate::vendor::variant_dispatch_count_for(
        crate::vendor::VendorKernelFamily::LinearBiasForward,
    );
    let linear_residual_dispatches = crate::vendor::variant_dispatch_count_for(
        crate::vendor::VendorKernelFamily::LinearResidualForward,
    );
    let linear3_dispatches = crate::vendor::variant_dispatch_count_for(
        crate::vendor::VendorKernelFamily::Linear3Forward,
    );
    let vendor_gradients = vendor.full_model_pending_gradient_snapshots()?;
    assert!(
        vendor_dispatches > 0,
        "vendor-enabled Hierarchos step did not dispatch a tuned module"
    );
    // Both producer fusions are explicitly disabled above. All four matmul
    // families must therefore execute, regardless of which producer graph this
    // adapter's shipping route selects. Matching numbers without those actual
    // dispatches would not prove vendor-module parity.
    assert!(
        linear_forward_dispatches > 0 && linear_bias_forward_dispatches > 0,
        "vendor-enabled Hierarchos step did not dispatch the linear-forward/\
         linear-bias-forward vendor modules"
    );
    for family in crate::vendor::VendorKernelFamily::ALL {
        eprintln!(
            "{vendor_label} dispatch histogram: {:<24} {}",
            family.label(),
            crate::vendor::variant_dispatch_count_for(family)
        );
    }
    assert_eq!(
        linear_residual_dispatches, 60,
        "the fixture's channel-mix route should compose through linear-residual-forward at \
         this geometry; 60 dispatches over its 20 layer-timesteps is what that composition \
         costs, and a regression back to the fused producer shows up here"
    );
    assert_eq!(
        linear3_dispatches, 20,
        "the geometry-aware promotion should dispatch linear3-forward once per layer-timestep"
    );

    let guard = EnvVarGuard::set(crate::HIERARCHOS_VULKAN_DISABLE_VENDOR_KERNELS_ENV, "1");
    crate::vendor::reset_variant_dispatch_count();
    let mut portable = graph(&device, &path, batch)?;
    let portable_result = run_vendor_step(&mut portable, batch, context_dim, &ids, &labels)?;
    let portable_dispatches = crate::vendor::variant_dispatch_count();
    let portable_gradients = portable.full_model_pending_gradient_snapshots()?;
    drop(guard);
    assert_eq!(
        portable_dispatches, 0,
        "portable Hierarchos step must not dispatch a vendor module"
    );

    close(
        &vendor_result.losses,
        &portable_result.losses,
        2e-7,
        "vendor/portable causal losses",
    );
    close(
        &vendor_result.final_h_packed_state,
        &portable_result.final_h_packed_state,
        2e-7,
        "vendor/portable H state",
    );
    close(
        &vendor_result.final_l_packed_state,
        &portable_result.final_l_packed_state,
        2e-7,
        "vendor/portable L state",
    );
    close_registry(&vendor_gradients, &portable_gradients, 2e-7);
    eprintln!(
        "PASS Hierarchos causal-LM vendor parity: {vendor_dispatches} {vendor_label} dispatches \
         (linear-forward={linear_forward_dispatches}, \
         linear-bias-forward={linear_bias_forward_dispatches}), portable run {portable_dispatches}"
    );
    Ok(())
}
