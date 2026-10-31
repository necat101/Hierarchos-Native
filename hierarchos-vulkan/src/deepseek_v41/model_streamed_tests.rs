use super::*;
use crate::deepseek_v41::{
    DeepseekV41Engram, DeepseekV41EngramLayout, DeepseekV41EngramResidency,
    DeepseekV41ExpertResidency, DeepseekV41ProjectionResidency,
};

fn packed_fixture() -> Result<Fixture> {
    let source: Value = serde_json::from_str(include_str!(
        "../../validation/fixtures/deepseek_v41_bf16_model.json"
    ))?;
    Ok(serde_json::from_value(source["cases"][0].clone())?)
}

#[test]
fn deepseek_v41_streamed_checkpoint_checks_late_expert_metadata() -> Result<()> {
    let f = packed_fixture()?;
    let package = Package::new(&f, true)?;
    let name = "layers.5.ffn.experts.2.w3.scale";
    let path = package.root.join("model.safetensors.index.json");
    let mut index: Value = serde_json::from_slice(&fs::read(&path)?)?;
    assert!(index["weight_map"]
        .as_object_mut()
        .unwrap()
        .remove(name)
        .is_some());
    fs::write(&path, serde_json::to_vec(&index)?)?;
    let mut reader = package.reader()?;
    let error = reader.load_streamed_moe(5, false).err().unwrap();
    assert!(format!("{error:#}").contains(name), "{error:#}");
    assert!(reader.load_streamed_moe(6, false).is_err());
    assert!(reader.load_mixed_expert(0, 3).is_err());

    let dense = Package::new(&fixture()?, false)?;
    dense.reader()?.load_streamed_moe(0, false)?;
    Ok(())
}

#[test]
fn deepseek_v41_streamed_payload_is_lazy_and_errors_are_propagated() -> Result<()> {
    let mut f = packed_fixture()?;
    // Negative I/O case only: never change any accepted full-model fixture.
    f.packed_tensors
        .get_mut("layers.0.ffn.experts.0.w1.weight")
        .unwrap()
        .scales[0] = 255;
    let package = Package::new(&f, true)?;
    let mut reader = package.reader()?;
    let mut model = DeepseekV41CausalLM::from_checkpoint_with_expert_residency(
        VulkanDevice::new()?,
        &mut reader,
        runtime(&f),
        Some(f.tokens.clone()),
        DeepseekV41WeightOutput::Bf16,
        DeepseekV41ExpertResidency::Streamed,
    )?;
    // Payload validation must not have happened during construction. The
    // source owns its metadata after this external reader is dropped.
    drop(reader);
    assert_eq!(model.resident_routed_expert_count(), 0);
    let error = run(&mut model, &f.steps[0]).err().unwrap();
    assert!(format!("{error:#}").contains("NaN code 255"), "{error:#}");
    assert_eq!(model.resident_routed_expert_count(), 0);
    Ok(())
}

#[test]
fn deepseek_v41_streamed_expert_staging_is_released_between_calls() -> Result<()> {
    for stream_shared in [false, true] {
        staging_is_released(stream_shared)?;
    }
    Ok(())
}

fn staging_is_released(stream_shared: bool) -> Result<()> {
    let f = packed_fixture()?;
    let package = Package::new(&f, true)?;
    let mut reader = package.reader()?;
    let device = VulkanDevice::new()?;
    let config = crate::deepseek_v41::DeepseekV41MoeConfig::from_text(&reader.config.text)?;
    let moe = crate::deepseek_v41::DeepseekV41MixedMoe::new_streamed(
        device.clone(),
        config,
        reader.load_streamed_moe(0, stream_shared)?,
        DeepseekV41WeightOutput::Bf16,
    )?;
    drop(reader);
    let rows = 4;
    let x = GpuBuffer::from_f32(
        &device,
        &f.tensors["embed.weight"].values[..rows * config.hidden_size],
    )?;
    let baseline = device.memory_stats()?.live_buffer_bytes;
    for _ in 0..3 {
        let output = moe.apply(&x, rows)?;
        assert!(output
            .read_f32(rows * config.hidden_size)?
            .iter()
            .all(|v| v.is_finite()));
        assert_eq!(moe.resident_routed_expert_count(), 0);
        assert_eq!(
            moe.resident_shared_expert_count(),
            usize::from(!stream_shared)
        );
        drop(output);
        // Only the test trace deliberately holds per-call route buffers.
        drop(moe.route_trace.replace(None));
        assert_eq!(
            device.memory_stats()?.live_buffer_bytes,
            baseline,
            "streamed expert or per-call GPU staging was retained"
        );
    }
    Ok(())
}

#[test]
fn deepseek_v41_streamed_all_checks_shared_metadata_and_storage() -> Result<()> {
    let f = packed_fixture()?;
    let package = Package::new(&f, true)?;
    let name = "layers.5.ffn.shared_experts.w3.scale";
    let path = package.root.join("model.safetensors.index.json");
    let mut index: Value = serde_json::from_slice(&fs::read(&path)?)?;
    assert!(index["weight_map"]
        .as_object_mut()
        .unwrap()
        .remove(name)
        .is_some());
    fs::write(&path, serde_json::to_vec(&index)?)?;
    let mut reader = package.reader()?;
    let error = reader.load_streamed_moe(5, true).err().unwrap();
    assert!(format!("{error:#}").contains(name), "{error:#}");
    assert!(reader.load_mixed_shared_expert(6).is_err());

    // Invalid temporary package only; acceptance fixtures remain unchanged.
    let mut f = packed_fixture()?;
    let name = "layers.0.ffn.shared_experts.w2.weight";
    let packed = f.packed_tensors.remove(name).unwrap();
    f.tensors.insert(
        name.to_owned(),
        Tensor {
            shape: vec![packed.rows, packed.cols],
            values: vec![0.0; packed.rows * packed.cols],
            dtype: String::new(),
        },
    );
    let dense_shared = Package::new(&f, true)?;
    let error = dense_shared
        .reader()?
        .load_streamed_moe(0, true)
        .err()
        .unwrap();
    assert!(
        format!("{error:#}").contains("does not match BF16 stream"),
        "{error:#}"
    );
    Ok(())
}

#[test]
fn deepseek_v41_streamed_engram_tables_match_resident_logits() -> Result<()> {
    let f = packed_fixture()?;
    let package = Package::new(&f, true)?;
    let device = VulkanDevice::new()?;
    let base = device.memory_stats()?.live_buffer_bytes;
    let mut resident = DeepseekV41CausalLM::from_checkpoint_with_residency(
        device.clone(),
        &mut package.reader()?,
        runtime(&f),
        Some(f.tokens.clone()),
        DeepseekV41WeightOutput::Bf16,
        DeepseekV41ExpertResidency::StreamedAllExperts,
        DeepseekV41EngramResidency::Resident,
        DeepseekV41ProjectionResidency::Resident,
    )?;
    let resident_bytes = device.memory_stats()?.live_buffer_bytes - base;
    assert_eq!(resident.resident_engram_table_count(), 2);
    let base = device.memory_stats()?.live_buffer_bytes;
    let mut streamed = DeepseekV41CausalLM::from_checkpoint_with_residency(
        device.clone(),
        &mut package.reader()?,
        runtime(&f),
        Some(f.tokens.clone()),
        DeepseekV41WeightOutput::Bf16,
        DeepseekV41ExpertResidency::StreamedAllExperts,
        DeepseekV41EngramResidency::StreamedTables,
        DeepseekV41ProjectionResidency::Resident,
    )?;
    let streamed_bytes = device.memory_stats()?.live_buffer_bytes - base;
    assert_eq!(streamed.resident_engram_table_count(), 0);
    assert!(streamed_bytes < resident_bytes);
    let mut logits = 0;
    for step in &f.steps {
        let resident_logits = run(&mut resident, step)?.read_all()?;
        let streamed_logits = run(&mut streamed, step)?.read_all()?;
        assert_eq!(bits(&resident_logits), bits(&step.logits), "{}", step.name);
        assert_eq!(
            bits(&streamed_logits),
            bits(&resident_logits),
            "{}",
            step.name
        );
        logits += streamed_logits.len();
    }
    eprintln!(
        "V4.1 streamed Engram tables: {logits} logits bit-exact, {resident_bytes} -> {streamed_bytes} bytes"
    );
    Ok(())
}

#[test]
fn deepseek_v41_streamed_engram_payload_errors_invalidate_sequence() -> Result<()> {
    let mut f = packed_fixture()?;
    f.packed_tensors
        .get_mut("layers.0.engram.embed.weight")
        .unwrap()
        .scales[0] = 255;
    let package = Package::new(&f, true)?;
    let mut reader = package.reader()?;
    let mut model = DeepseekV41CausalLM::from_checkpoint_with_residency(
        VulkanDevice::new()?,
        &mut reader,
        runtime(&f),
        Some(f.tokens.clone()),
        DeepseekV41WeightOutput::Bf16,
        DeepseekV41ExpertResidency::StreamedAllExperts,
        DeepseekV41EngramResidency::StreamedTables,
        DeepseekV41ProjectionResidency::Resident,
    )?;
    // Table payload validation must not have happened during construction; the
    // source owns its metadata after this external reader is dropped.
    drop(reader);
    assert_eq!(model.resident_engram_table_count(), 0);
    let error = run(&mut model, &f.steps[0]).err().unwrap();
    assert!(format!("{error:#}").contains("NaN code 255"), "{error:#}");
    assert_eq!(model.next_position(), None);
    let error = model.decode(&vec![0; f.batch_size], 1, None).err().unwrap();
    assert!(
        format!("{error:#}").contains("restart with prefill"),
        "{error:#}"
    );
    Ok(())
}

#[test]
fn deepseek_v41_streamed_engram_table_metadata_is_deferred() -> Result<()> {
    let f = packed_fixture()?;
    let package = Package::new(&f, true)?;
    let path = package.root.join("model.safetensors.index.json");
    let mut index: Value = serde_json::from_slice(&fs::read(&path)?)?;
    let name = "layers.4.engram.embed.scale";
    assert!(index["weight_map"]
        .as_object_mut()
        .unwrap()
        .remove(name)
        .is_some());
    fs::write(&path, serde_json::to_vec(&index)?)?;
    // Resident construction reads every table eagerly and fails here.
    let error = DeepseekV41CausalLM::from_checkpoint_with_residency(
        VulkanDevice::new()?,
        &mut package.reader()?,
        runtime(&f),
        Some(f.tokens.clone()),
        DeepseekV41WeightOutput::Bf16,
        DeepseekV41ExpertResidency::StreamedAllExperts,
        DeepseekV41EngramResidency::Resident,
        DeepseekV41ProjectionResidency::Resident,
    )
    .err()
    .unwrap();
    assert!(format!("{error:#}").contains(name), "{error:#}");
    // Streamed construction reads projection/norm tensors only.
    let mut model = DeepseekV41CausalLM::from_checkpoint_with_residency(
        VulkanDevice::new()?,
        &mut package.reader()?,
        runtime(&f),
        Some(f.tokens.clone()),
        DeepseekV41WeightOutput::Bf16,
        DeepseekV41ExpertResidency::StreamedAllExperts,
        DeepseekV41EngramResidency::StreamedTables,
        DeepseekV41ProjectionResidency::Resident,
    )?;
    assert_eq!(model.resident_engram_table_count(), 0);
    let error = run(&mut model, &f.steps[0]).err().unwrap();
    assert!(format!("{error:#}").contains(name), "{error:#}");
    Ok(())
}

#[test]
fn deepseek_v41_streamed_engram_table_staging_is_released_between_lookups() -> Result<()> {
    let f = packed_fixture()?;
    let package = Package::new(&f, true)?;
    let device = VulkanDevice::new()?;
    let mut reader = package.reader()?;
    let text = reader.config.text.clone();
    let layout = DeepseekV41EngramLayout::from_text(&text)?.context("Engram schedule")?;
    let (projection, weights) = reader.load_mixed_engram_projection(0)?;
    let layer = DeepseekV41Engram::new_mixed_streamed(
        device.clone(),
        &layout,
        0,
        text.hc_mult,
        text.hidden_size,
        text.rms_norm_eps as f32,
        weights,
        projection,
        DeepseekV41WeightOutput::Bf16,
        reader,
        0,
    )?;
    assert_eq!(layer.resident_table_count(), 0);
    let rows = 4;
    let hashes = GpuBuffer::from_u32(&device, &vec![0u32; rows * layout.n_hash_cols()])?;
    let stream = GpuBuffer::zeros_f32(&device, rows * text.hc_mult * text.hidden_size)?;
    let baseline = device.memory_stats()?.live_buffer_bytes;
    // The recycler may hold one released lookup buffer until its timeline
    // retires; that fixed reservation must not grow or include table state.
    let lookup_bytes =
        rows * (layout.n_hash_cols() * layout.head_dim + (text.hc_mult + 2) * text.hidden_size) * 4;
    let mut after_first = None;
    for iteration in 0..3 {
        let output = layer.apply(&stream, &hashes, None, rows)?;
        assert!(output
            .read_f32(rows * text.hc_mult * text.hidden_size)?
            .iter()
            .all(|value| value.is_finite()));
        drop(output);
        assert_eq!(layer.resident_table_count(), 0);
        let live = device.memory_stats()?.live_buffer_bytes;
        eprintln!("V4.1 streamed Engram lookup {iteration}: {live} vs {baseline}");
        let retained = *after_first.get_or_insert(live - baseline);
        assert_eq!(
            live - baseline,
            retained,
            "streamed Engram table or per-lookup GPU staging was retained"
        );
        assert!(retained < lookup_bytes, "staging grew past one lookup");
    }
    Ok(())
}

#[test]
fn deepseek_v41_streamed_all_shared_payload_failure_invalidates_sequence() -> Result<()> {
    let mut f = packed_fixture()?;
    f.packed_tensors
        .get_mut("layers.0.ffn.shared_experts.w1.weight")
        .unwrap()
        .scales[0] = 255;
    let package = Package::new(&f, true)?;
    let mut reader = package.reader()?;
    // The old routed-only policy still eagerly validates shared payloads.
    assert!(reader.load_streamed_moe(0, false).is_err());
    let mut model = DeepseekV41CausalLM::from_checkpoint_with_expert_residency(
        VulkanDevice::new()?,
        &mut reader,
        runtime(&f),
        Some(f.tokens.clone()),
        DeepseekV41WeightOutput::Bf16,
        DeepseekV41ExpertResidency::StreamedAllExperts,
    )?;
    drop(reader);
    assert_eq!(model.resident_routed_expert_count(), 0);
    assert_eq!(model.resident_shared_expert_count(), 0);
    let error = run(&mut model, &f.steps[0]).err().unwrap();
    assert!(format!("{error:#}").contains("NaN code 255"), "{error:#}");
    assert_eq!(model.next_position(), None);
    let error = model.decode(&vec![0; f.batch_size], 1, None).err().unwrap();
    assert!(
        format!("{error:#}").contains("restart with prefill"),
        "{error:#}"
    );
    assert_eq!(model.resident_shared_expert_count(), 0);
    Ok(())
}

#[test]
fn deepseek_v41_streamed_layer_projections_match_resident_logits() -> Result<()> {
    let f = packed_fixture()?;
    let package = Package::new(&f, true)?;
    let device = VulkanDevice::new()?;
    let base = device.memory_stats()?.live_buffer_bytes;
    let mut resident = DeepseekV41CausalLM::from_checkpoint_with_residency(
        device.clone(),
        &mut package.reader()?,
        runtime(&f),
        Some(f.tokens.clone()),
        DeepseekV41WeightOutput::Bf16,
        DeepseekV41ExpertResidency::StreamedAllExperts,
        DeepseekV41EngramResidency::Resident,
        DeepseekV41ProjectionResidency::Resident,
    )?;
    let resident_bytes = device.memory_stats()?.live_buffer_bytes - base;
    let layers = f.config["text_config"]["num_hidden_layers"]
        .as_u64()
        .unwrap() as usize;
    assert_eq!(resident.resident_projection_layer_count(), layers);
    let base = device.memory_stats()?.live_buffer_bytes;
    let mut streamed = DeepseekV41CausalLM::from_checkpoint_with_residency(
        device.clone(),
        &mut package.reader()?,
        runtime(&f),
        Some(f.tokens.clone()),
        DeepseekV41WeightOutput::Bf16,
        DeepseekV41ExpertResidency::StreamedAllExperts,
        DeepseekV41EngramResidency::Resident,
        DeepseekV41ProjectionResidency::Streamed,
    )?;
    let streamed_bytes = device.memory_stats()?.live_buffer_bytes - base;
    assert_eq!(streamed.resident_projection_layer_count(), 0);
    assert!(streamed_bytes < resident_bytes);
    let mut logits = 0;
    for step in &f.steps {
        let resident_logits = run(&mut resident, step)?.read_all()?;
        let streamed_logits = run(&mut streamed, step)?.read_all()?;
        assert_eq!(bits(&resident_logits), bits(&step.logits), "{}", step.name);
        assert_eq!(
            bits(&streamed_logits),
            bits(&resident_logits),
            "{}",
            step.name
        );
        logits += streamed_logits.len();
        assert_eq!(streamed.resident_projection_layer_count(), 0);
    }
    eprintln!(
        "V4.1 streamed layer projections: {logits} logits bit-exact, {resident_bytes} -> {streamed_bytes} bytes"
    );
    Ok(())
}

#[test]
fn deepseek_v41_streamed_projection_payload_errors_invalidate_sequence() -> Result<()> {
    let mut f = packed_fixture()?;
    let name = "layers.5.attn.wq_a.weight";
    f.packed_tensors.get_mut(name).unwrap().scales[0] = 255;
    let package = Package::new(&f, true)?;
    let mut reader = package.reader()?;
    let mut model = DeepseekV41CausalLM::from_checkpoint_with_residency(
        VulkanDevice::new()?,
        &mut reader,
        runtime(&f),
        Some(f.tokens.clone()),
        DeepseekV41WeightOutput::Bf16,
        DeepseekV41ExpertResidency::StreamedAllExperts,
        DeepseekV41EngramResidency::Resident,
        DeepseekV41ProjectionResidency::Streamed,
    )?;
    // Projection payload validation is deferred to the layer's execution; the
    // streamed source owns its metadata after this external reader is dropped.
    drop(reader);
    assert_eq!(model.resident_projection_layer_count(), 0);
    let error = run(&mut model, &f.steps[0]).err().unwrap();
    assert!(format!("{error:#}").contains("NaN code 255"), "{error:#}");
    assert_eq!(model.next_position(), None);
    assert_eq!(model.resident_projection_layer_count(), 0);
    let error = model.decode(&vec![0; f.batch_size], 1, None).err().unwrap();
    assert!(
        format!("{error:#}").contains("restart with prefill"),
        "{error:#}"
    );
    Ok(())
}

#[test]
fn deepseek_v41_streamed_projection_partial_upload_is_released() -> Result<()> {
    let mut f = packed_fixture()?;
    // Layer 3 owns an indexer, so attention and compressor payloads upload
    // before this scale fails and must not stay pinned after the error.
    f.packed_tensors
        .get_mut("layers.3.attn.indexer.wq_b.weight")
        .unwrap()
        .scales[0] = 255;
    let package = Package::new(&f, true)?;
    let mut reader = package.reader()?;
    let mut model = DeepseekV41CausalLM::from_checkpoint_with_residency(
        VulkanDevice::new()?,
        &mut reader,
        runtime(&f),
        Some(f.tokens.clone()),
        DeepseekV41WeightOutput::Bf16,
        DeepseekV41ExpertResidency::StreamedAllExperts,
        DeepseekV41EngramResidency::Resident,
        DeepseekV41ProjectionResidency::Streamed,
    )?;
    drop(reader);
    assert_eq!(model.resident_projection_layer_count(), 0);
    let error = run(&mut model, &f.steps[0]).err().unwrap();
    assert!(format!("{error:#}").contains("NaN code 255"), "{error:#}");
    assert_eq!(model.next_position(), None);
    assert_eq!(model.resident_projection_layer_count(), 0);
    Ok(())
}
