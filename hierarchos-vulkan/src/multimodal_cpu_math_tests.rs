//! Opt-in diagnostics against immutable projected-feature fixtures.
use super::*;

#[test]
#[ignore = "set HIERARCHOS_MM_DIAGNOSTIC_INPUT and OUTPUT to frozen fixture paths"]
fn multimodal_cpu_math_trace() -> Result<()> {
    let input = PathBuf::from(std::env::var("HIERARCHOS_MM_DIAGNOSTIC_INPUT")?);
    let output = PathBuf::from(std::env::var("HIERARCHOS_MM_DIAGNOSTIC_OUTPUT")?);
    let mode = std::env::var("HIERARCHOS_MM_DIAGNOSTIC_MODE").unwrap_or_default();
    let fixture: serde_json::Value = serde_json::from_slice(&fs::read(input.join("input.json"))?)?;
    let floats = |name: &str| -> Vec<f32> {
        fixture[name]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    };
    let ids = |name: &str| -> Vec<u32> {
        fixture[name]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    };
    let device = VulkanDevice::new()?;
    let batch = fixture["batch_size"].as_u64().unwrap() as usize;
    let seq = fixture["seq_len"].as_u64().unwrap() as usize;
    let mut graph =
        VulkanTransformer::from_hf_package(device.clone(), input.join("model"), batch, seq)?;
    if mode.contains("welford") {
        graph.kernels.layer_norm_forward =
            vulkan::ComputeKernel::new(&device, LAYER_NORM_WELFORD_SMALL_FORWARD_SPV, 6, 12)?;
        graph.kernels.layer_norm_rows_per_workgroup = 8;
    }
    if mode.contains("reciprocal") {
        graph.kernels.attention_forward = vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../shaders/transformer_attention_forward_cpu_reciprocal.spv"),
            11,
            std::mem::size_of::<AttentionPush>() as u32,
        )?;
        graph.kernels.attention_backward = vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../shaders/transformer_attention_backward_cpu_reciprocal.spv"),
            15,
            std::mem::size_of::<AttentionBackwardPush>() as u32,
        )?;
    }
    if mode.contains("tanh") {
        graph.kernels.gelu_tanh_forward = vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../shaders/transformer_gelu_tanh_ha_forward.spv"),
            2,
            4,
        )?;
        graph.kernels.gelu_tanh_backward = vulkan::ComputeKernel::new(
            &device,
            include_bytes!("../shaders/transformer_gelu_tanh_ha_backward.spv"),
            3,
            4,
        )?;
    }
    let dim = graph.config.effective_embedding_size();
    let rows: Vec<usize> = ids("modality_rows")
        .into_iter()
        .map(|x| x as usize)
        .collect();
    let features = floats("features");
    let mut projector = crate::LinearProjectionTrainer::new(
        device.clone(),
        fixture["feature_dim"].as_u64().unwrap() as usize,
        dim,
        rows.len(),
        &floats("projector_weight"),
        Some(&floats("projector_bias")),
        0.0,
    )?;
    let hyper = crate::CpuAdamWHyperParams {
        lr: fixture["lr"].as_f64().unwrap(),
        beta1: fixture["beta1"].as_f64().unwrap(),
        beta2: fixture["beta2"].as_f64().unwrap(),
        eps: fixture["eps"].as_f64().unwrap(),
        weight_decay: fixture["weight_decay"].as_f64().unwrap(),
    };
    if mode.contains("projection") && !mode.contains("frozenprojection") {
        projector.set_cpu_reference_math(true)?;
    }
    graph.set_cpu_adamw_policy(Some(hyper))?;
    let mut steps = Vec::new();
    let cpu: serde_json::Value = serde_json::from_slice(&fs::read(input.join("cpu.json"))?)?;
    for step_index in 0..2 {
        let projected = if mode.contains("frozenprojection") {
            // Attribution only: this ignored diagnostic is never a gate runner.
            cpu["steps"][step_index]["projected"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        } else {
            projector.forward(&features)?
        };
        let modality = ModalityEmbeddingRows {
            rows: &rows,
            values: &projected,
        };
        let logits = graph.forward_logits_with_modality_embeddings(
            &ids("input_ids"),
            &floats("attention_mask"),
            modality,
        )?;
        let result = graph.train_step_with_modality_embeddings(
            &ids("input_ids"),
            &ids("targets"),
            &floats("attention_mask"),
            &floats("loss_weights"),
            hyper.as_f32(),
            modality,
        )?;
        let mut buffers = BTreeMap::new();
        let n = batch * seq * graph.config.hidden_size;
        buffers.insert("input".to_owned(), graph.input_hidden.read_f32(n)?);
        buffers.insert(
            "input.grad".to_owned(),
            graph.grad_input_hidden.read_f32(n)?,
        );
        buffers.insert("norm".to_owned(), graph.final_norm_output.read_f32(n)?);
        buffers.insert(
            "norm.grad".to_owned(),
            graph.grad_final_norm_output.read_f32(n)?,
        );
        for (i, layer) in graph.layers.iter().enumerate() {
            for (name, buffer) in [
                ("ln1", &layer.tape.ln1),
                ("ln2", &layer.tape.ln2),
                ("attention_projection", &layer.tape.attention_projection),
                ("residual1", &layer.tape.residual1),
                ("mlp_output", &layer.tape.mlp_output),
                ("output", &layer.tape.output),
                ("ln1.grad", &layer.tape.grad_ln1),
                ("ln2.grad", &layer.tape.grad_ln2),
                (
                    "output.grad",
                    if i + 1 == graph.layers.len() {
                        &graph.grad_final_norm_input
                    } else {
                        graph.layers[i + 1].backward_input_gradient()
                    },
                ),
            ] {
                buffers.insert(format!("{i}.{name}"), buffer.read_f32(n)?);
            }
            for (name, buffer) in [
                ("ln1.mean", &layer.tape.ln1_mean),
                ("ln1.rstd", &layer.tape.ln1_rstd),
                ("ln2.mean", &layer.tape.ln2_mean),
                ("ln2.rstd", &layer.tape.ln2_rstd),
            ] {
                buffers.insert(format!("{i}.{name}"), buffer.read_f32(batch * seq)?);
            }
        }
        projector.train_step_with_optimizer(&features, &result.modality_embedding_grad, hyper)?;
        steps.push(
            serde_json::json!({"buffers":buffers,"logits":logits,"projected":projected,
            "modality_embedding_grad":result.modality_embedding_grad}),
        );
    }
    let projected = projector.forward(&features)?;
    let final_logits = graph.forward_logits_with_modality_embeddings(
        &ids("input_ids"),
        &floats("attention_mask"),
        ModalityEmbeddingRows {
            rows: &rows,
            values: &projected,
        },
    )?;
    fs::write(
        output,
        serde_json::to_vec(
            &serde_json::json!({"mode":mode,"steps":steps,"final_logits":final_logits}),
        )?,
    )?;
    Ok(())
}
