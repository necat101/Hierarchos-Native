//! Per-graph interleaved A/B training and exact optimizer isolation probe.
use anyhow::{ensure, Context, Result};
use hierarchos_vulkan::{AdamWHyperParams, VulkanDevice, VulkanTransformer};
use serde::Deserialize;
use std::{collections::BTreeMap, fs, path::{Path, PathBuf}};

#[derive(Deserialize)]
struct Fixture {
    batch_size: usize, seq_len: usize,
    input_ids: Vec<u32>, targets: Vec<u32>, attention_mask: Vec<f32>, loss_weights: Vec<f32>,
    learning_rate: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32,
}
fn train(g: &mut VulkanTransformer, f: &Fixture, h: AdamWHyperParams) -> Result<()> {
    g.train_step(&f.input_ids, &f.targets, &f.attention_mask, &f.loss_weights, h)?;
    Ok(())
}
fn equal_payload(a: &Path, b: &Path) -> Result<()> {
    for name in ["adapter_model.safetensors", "hierarchos_adapter_optimizer.safetensors"] {
        let left = fs::read(a.join(name))?;
        let right = fs::read(b.join(name))?;
        let left = safetensors::SafeTensors::deserialize(&left)?;
        let right = safetensors::SafeTensors::deserialize(&right)?;
        ensure!(left.len() == right.len(), "{name}: tensor count differs");
        for key in left.names() {
            let l = left.tensor(key)?;
            let r = right.tensor(key)?;
            ensure!(l.shape() == r.shape() && l.dtype() == r.dtype() && l.data() == r.data(),
                "{name}: interleaved optimizer isolation differs at {key}");
        }
    }
    Ok(())
}
fn safetensor_files(root: &Path) -> Result<BTreeMap<PathBuf, PathBuf>> {
    let mut files = BTreeMap::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|value| value.to_str()) == Some("safetensors") {
            files.insert(PathBuf::from(entry.file_name()), path);
        }
    }
    Ok(files)
}
fn equal_model_payload(a: &Path, b: &Path) -> Result<()> {
    let left_files = safetensor_files(a)?;
    let right_files = safetensor_files(b)?;
    ensure!(
        left_files.keys().eq(right_files.keys()),
        "disabled base safetensor file set differs from a fresh graph"
    );
    for (relative, left_path) in left_files {
        let right_path = right_files
            .get(&relative)
            .with_context(|| format!("missing disabled-base safetensor {}", relative.display()))?;
        let left_bytes = fs::read(&left_path)?;
        let right_bytes = fs::read(right_path)?;
        let left = safetensors::SafeTensors::deserialize(&left_bytes)?;
        let right = safetensors::SafeTensors::deserialize(&right_bytes)?;
        ensure!(
            left.len() == right.len(),
            "{}: disabled base tensor count differs",
            relative.display()
        );
        for key in left.names() {
            let l = left.tensor(key)?;
            let r = right.tensor(key)?;
            ensure!(
                l.shape() == r.shape() && l.dtype() == r.dtype() && l.data() == r.data(),
                "{}: disabled base tensor differs at {key}",
                relative.display()
            );
        }
    }
    Ok(())
}
fn run() -> Result<()> {
    let args = std::env::args_os().skip(1).map(PathBuf::from).collect::<Vec<_>>();
    ensure!(args.len() == 5, "usage: peft_bank_parity BASE FIXTURE ADAPTER_A ADAPTER_B OUTPUT");
    let (base, fixture, a, b, out) = (&args[0], &args[1], &args[2], &args[3], &args[4]);
    let f: Fixture = serde_json::from_slice(&fs::read(fixture)?)?;
    let h = AdamWHyperParams { lr:f.learning_rate, beta1:f.beta1, beta2:f.beta2, eps:f.eps, weight_decay:f.weight_decay };
    fs::create_dir_all(out)?;
    let device = VulkanDevice::new()?;
    let make = || VulkanTransformer::from_hf_package(device.clone(), base, f.batch_size, f.seq_len);
    let mut g = make()?;
    let base_logits = g.forward_logits(&f.input_ids, &f.attention_mask)?;
    g.load_lora_adapter_named_for_training("adapter_a", a)?;
    g.load_lora_adapter_named_for_training("adapter_b", b)?;
    g.set_adapter("adapter_a")?;
    train(&mut g, &f, h)?;
    let grad_a = g.lora_gradient_values()?;
    let logits_a = g.forward_logits(&f.input_ids, &f.attention_mask)?;
    g.save_lora_training_checkpoint(out.join("a1"), h)?;
    g.set_adapter("adapter_b")?;
    train(&mut g, &f, h)?;
    let grad_b = g.lora_gradient_values()?;
    let logits_b = g.forward_logits(&f.input_ids, &f.attention_mask)?;
    g.save_lora_training_checkpoint(out.join("b1"), h)?;
    g.set_adapter("adapter_a")?;
    ensure!(g.lora_gradient_values()? == grad_a, "B training contaminated A gradients");
    ensure!(g.forward_logits(&f.input_ids, &f.attention_mask)? == logits_a, "B training contaminated A output");
    g.save_lora_training_checkpoint(out.join("a1-restored"), h)?;
    equal_payload(&out.join("a1"), &out.join("a1-restored"))?;
    train(&mut g, &f, h)?;
    g.save_lora_training_checkpoint(out.join("a2"), h)?;
    g.set_adapter("adapter_b")?;
    ensure!(g.lora_gradient_values()? == grad_b, "A training contaminated B gradients");
    ensure!(g.forward_logits(&f.input_ids, &f.attention_mask)? == logits_b, "A training contaminated B output");
    g.save_lora_training_checkpoint(out.join("b1-restored"), h)?;
    equal_payload(&out.join("b1"), &out.join("b1-restored"))?;
    train(&mut g, &f, h)?;
    g.save_lora_training_checkpoint(out.join("b2"), h)?;
    g.disable_adapter()?;
    let disabled_logits = g.forward_logits(&f.input_ids, &f.attention_mask)?;
    let (worst_index, disabled_max_abs) = disabled_logits
        .iter()
        .zip(&base_logits)
        .enumerate()
        .map(|(index, (disabled, base))| (index, (disabled - base).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap_or((0, 0.0));
    ensure!(
        disabled_max_abs <= 2.0e-7,
        "disabled output exceeds strict PEFT tolerance: max_abs={disabled_max_abs:e} index={worst_index}"
    );
    let fresh = make()?;
    let fresh_base = out.join("fresh-base");
    let base_after = out.join("base-after");
    fresh.export_hf_package(base, &fresh_base)?;
    g.export_hf_package(base, &base_after)?;
    equal_model_payload(&fresh_base, &base_after)?;

    // Reload BOTH adapter-local checkpoints into one fresh graph, switch and
    // resume. Compare weights and every AdamW moment with uninterrupted A/B.
    let mut resumed = make()?;
    for (name, checkpoint) in [("adapter_a", "a1"), ("adapter_b", "b1")] {
        resumed.load_lora_adapter_named_for_training(name, out.join(checkpoint))?;
        resumed.load_lora_optimizer_state(out.join(checkpoint))?;
    }
    for (name, label, adapter) in [("adapter_a", "a", a), ("adapter_b", "b", b)] {
        resumed.set_adapter(name)?;
        train(&mut resumed, &f, h)?;
        let resumed_path = out.join(format!("{label}-resumed"));
        resumed.save_lora_training_checkpoint(&resumed_path, h)?;
        equal_payload(&out.join(format!("{label}2")), &resumed_path)?;
        let mut independent = make()?;
        independent.load_lora_adapter_named_for_training(name, adapter)?;
        train(&mut independent, &f, h)?;
        train(&mut independent, &f, h)?;
        let independent_path = out.join(format!("{label}-independent"));
        independent.save_lora_training_checkpoint(&independent_path, h)?;
        equal_payload(&out.join(format!("{label}2")), &independent_path)?;
    }
    fs::write(out.join("report.json"), "{\"pass\":true,\"weights_gradients_moments_isolation\":true,\"reload_switch_resume\":true,\"disabled_base_exact\":true}\n")?;
    Ok(())
}
fn main() -> Result<()> {
    std::thread::Builder::new().stack_size(16 * 1024 * 1024).spawn(run)?
        .join().map_err(|_| anyhow::anyhow!("PEFT bank probe panicked"))?
        .context("native PEFT bank validation")
}
