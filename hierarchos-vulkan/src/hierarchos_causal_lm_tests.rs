//! Architecture checks for Hierarchos only; no transformer implementation is used.
//! Requires a Vulkan device. Run with --test-threads=1.
use crate::{
    AdamWHyperParams, HierarchosLabeledSequenceObjective, HierarchosRawTokenLabeledSequenceInput,
    HierarchosTapeMemoryPolicy, HierarchosTokenTapeReadbackPolicy, HierarchosTokenTapeTrainResult,
    HierarchosTokenTapeUpdateMode, HierarchosTrainingGraph, RwkvParameterSnapshot, VulkanDevice,
};
use anyhow::Result;
use hierarchos_inference::{initialize_model_package, HierarchosModel, NativeBootstrapConfig};
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
