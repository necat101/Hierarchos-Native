//! Observation-only warm inference/PEFT benchmark for frozen-base streaming.
use anyhow::{bail, Result};
use hierarchos_vulkan::{AdamWHyperParams, VulkanDevice, VulkanTransformer,
    VulkanTransformerConfig, VulkanTransformerLoraConfig};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, time::Instant};

#[derive(Deserialize)]
struct Fixture {
    batch_size: usize,
    seq_len: usize,
    input_ids: Vec<u32>,
    attention_mask: Vec<f32>,
}

fn main() -> Result<()> {
    std::thread::Builder::new().stack_size(32 * 1024 * 1024).spawn(run)?.join()
        .map_err(|_| anyhow::anyhow!("streaming benchmark worker panicked"))?
}

fn run() -> Result<()> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.len() != 5 { bail!("expected MODEL FIXTURE WARMUP ITERATIONS inference|peft|train"); }
    let model = PathBuf::from(&args[0]);
    let fixture: Fixture = serde_json::from_slice(&fs::read(PathBuf::from(&args[1]))?)?;
    let warmup: usize = args[2].to_string_lossy().parse()?;
    let iterations: usize = args[3].to_string_lossy().parse()?;
    let mode = args[4].to_string_lossy();
    if warmup == 0 || iterations < 2 || !matches!(&*mode, "inference" | "peft" | "train") {
        bail!("positive warmup, >=2 iterations and inference|peft|train required");
    }
    let device = VulkanDevice::new()?;
    let config = VulkanTransformerConfig::from_hf_config(model.join("config.json"))?;
    let started = Instant::now();
    let mut graph = VulkanTransformer::from_hf_package_with_streamed_parameters(
        device.clone(), &model, config, fixture.batch_size, fixture.seq_len)?;
    if mode != "inference" {
        let config = VulkanTransformerLoraConfig::causal_lm(2, 4, vec!["all-linear".into()], false)?;
        graph.enable_lora(config, 187)?;
    }
    let construction_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut targets = fixture.input_ids.clone();
    targets.rotate_left(1);
    let hyper = AdamWHyperParams { lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.01 };
    let mut digest = Sha256::new();
    let mut expected = None::<Vec<u32>>;
    let mut observe = |graph: &mut VulkanTransformer| -> Result<()> {
        if mode == "train" {
            let metrics = graph.train_step(&fixture.input_ids, &targets, &fixture.attention_mask,
                &fixture.attention_mask, hyper)?;
            digest.update(metrics.loss.to_bits().to_le_bytes());
        } else {
            let logits = graph.forward_logits(&fixture.input_ids, &fixture.attention_mask)?;
            let bits = logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            if let Some(expected) = expected.as_ref() {
                if expected != &bits { bail!("repeated forward changed a logit bit"); }
            } else { expected = Some(bits); }
            for value in logits { digest.update(value.to_bits().to_le_bytes()); }
        }
        Ok(())
    };
    for _ in 0..warmup { observe(&mut graph)?; }
    let before_io = graph.streamed_linear_package_io()?;
    let before_submissions = device.submission_count();
    let before_memory = device.memory_stats()?;
    let before_arena = device.submission_arena_stats()?;
    let mut samples = Vec::new();
    for _ in 0..iterations {
        let started = Instant::now();
        observe(&mut graph)?;
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    drop(observe);
    let after_io = graph.streamed_linear_package_io()?;
    let after_memory = device.memory_stats()?;
    let after_arena = device.submission_arena_stats()?;
    let submissions = device.submission_count() - before_submissions;
    let final_logits = graph.forward_logits(&fixture.input_ids, &fixture.attention_mask)?;
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    println!("{}", serde_json::json!({
        "device": device.name(), "mode": mode, "construction_ms": construction_ms,
        "warmup": warmup, "iterations": iterations, "samples_ms": samples,
        "median_ms": sorted[sorted.len()/2], "sequence_sha256": format!("{:x}", digest.finalize()),
        "logits": final_logits, "submissions": submissions,
        "memory_before": {"bytes": before_memory.live_buffer_bytes, "buffers": before_memory.live_buffer_count,
            "peak": before_memory.peak_live_buffer_bytes, "largest": before_memory.largest_buffer_bytes},
        "memory_after": {"bytes": after_memory.live_buffer_bytes, "buffers": after_memory.live_buffer_count,
            "peak": after_memory.peak_live_buffer_bytes, "largest": after_memory.largest_buffer_bytes},
        "arena_before": {"scratch_leases": before_arena.scratch_lease_allocated,
            "scratch_reused": before_arena.scratch_lease_reused, "commands": before_arena.command_buffer_allocated},
        "arena_after": {"scratch_leases": after_arena.scratch_lease_allocated,
            "scratch_reused": after_arena.scratch_lease_reused, "commands": after_arena.command_buffer_allocated},
        "io_before": before_io, "io_after": after_io,
    }));
    Ok(())
}
