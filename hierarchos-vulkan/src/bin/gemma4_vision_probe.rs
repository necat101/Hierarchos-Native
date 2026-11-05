//! Raw RGB image -> Gemma4 vision tower -> multimodal embedder -> text logits,
//! full backward and export for the qualification gate.
use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{CpuAdamWHyperParams, VulkanDevice, VulkanGemma4Vision};
use serde::Deserialize;
use std::{fs, path::PathBuf};

#[derive(Deserialize)]
struct Fixture {
    batch_size: usize,
    seq_len: usize,
    image_count: usize,
    input_ids: Vec<u32>,
    attention_mask: Vec<f32>,
    targets: Vec<u32>,
    loss_weights: Vec<f32>,
    rgb: Vec<u8>,
    alternate_rgb: Vec<u8>,
    steps: usize,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
}

fn main() -> Result<()> {
    std::thread::Builder::new()
        .name("gemma4-vision".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(run)?
        .join()
        .map_err(|_| anyhow::anyhow!("Gemma4 vision probe panicked"))?
}

fn run() -> Result<()> {
    let mut model = None;
    let mut fixture = None;
    let mut output = None;
    let mut stages = None;
    let mut pixel_domain = "rescaled_rgb8".to_string();
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        let value = PathBuf::from(args.next().context("missing argument value")?);
        match arg.to_str() {
            Some("--model") => model = Some(value),
            Some("--fixture") => fixture = Some(value),
            Some("--output") => output = Some(value),
            Some("--stages") => stages = Some(value),
            Some("--pixel-domain") => {
                pixel_domain = value
                    .to_str()
                    .context("pixel domain must be UTF-8")?
                    .to_string()
            }
            _ => bail!("unknown argument {arg:?}"),
        }
    }
    let model = model.context("--model required")?;
    let output = output.context("--output required")?;
    if !matches!(pixel_domain.as_str(), "rescaled_rgb8" | "raw_patch_values") {
        bail!("unsupported pixel input domain");
    }
    let f: Fixture = serde_json::from_slice(&fs::read(fixture.context("--fixture required")?)?)?;
    if f.steps != 2 {
        bail!("qualification requires exactly two composed updates");
    }
    let side = ((f.rgb.len() / 3) as f64).sqrt() as usize;
    if side * side * 3 != f.rgb.len() {
        bail!("raw RGB must be a square image");
    }
    if f.alternate_rgb.len() != f.rgb.len() {
        bail!("alternate RGB must match the image geometry");
    }
    let device = VulkanDevice::new()?;
    let mut graph = VulkanGemma4Vision::from_hf_package(
        device.clone(),
        &model,
        f.batch_size,
        f.seq_len,
        f.image_count,
        side,
    )?;
    // The historical frozen case deliberately exercises 0..255 model values.
    // Keep that input domain explicit; ordinary RGB8 uses the real processor.
    let raw_pixels = |rgb: &[u8]| -> Vec<f32> {
        let area = side * side;
        (0..3)
            .flat_map(|channel| (0..area).map(move |i| rgb[i * 3 + channel] as f32))
            .collect()
    };
    let pixel_values = if pixel_domain == "raw_patch_values" {
        raw_pixels(&f.rgb)
    } else {
        graph.preprocess_rgb8(&f.rgb)?
    };
    let alternate_pixels = if pixel_domain == "raw_patch_values" {
        raw_pixels(&f.alternate_rgb)
    } else {
        graph.preprocess_rgb8(&f.alternate_rgb)?
    };
    if let Some(path) = stages {
        let mut dump = graph.stage_dump(&pixel_values)?;
        dump.as_object_mut().context("stage map")?.insert(
            "text".into(),
            graph.text_stage_dump(&f.input_ids, &f.attention_mask, &pixel_values)?,
        );
        fs::write(path, serde_json::to_vec_pretty(&dump)?)?;
        return Ok(());
    }
    if std::env::var_os("G4_CONFIG").is_some() {
        println!("{}", graph.text_config_summary());
        return Ok(());
    }
    if std::env::var_os("G4_TEXT_ONLY").is_some() {
        let logits = graph.text_logits(&f.input_ids, &f.attention_mask)?;
        println!("{}", serde_json::to_string(&logits)?);
        return Ok(());
    }
    let alternate = graph.forward(&f.input_ids, &f.attention_mask, &alternate_pixels)?;
    fs::create_dir(&output).context("output must be a new directory")?;
    let hyper = CpuAdamWHyperParams {
        lr: f.lr,
        beta1: f.beta1,
        beta2: f.beta2,
        eps: f.eps,
        weight_decay: f.weight_decay,
    };
    let mut steps = Vec::new();
    for _ in 0..f.steps {
        let forward = graph.forward(&f.input_ids, &f.attention_mask, &pixel_values)?;
        let result = graph.train_step_with_optimizer(
            &f.input_ids,
            &f.attention_mask,
            &pixel_values,
            &f.targets,
            &f.loss_weights,
            hyper,
        )?;
        graph.export_hf_package(&model, output.join(format!("step-{}", result.text.step)))?;
        steps.push(serde_json::json!({"forward":forward,"training":result}));
    }
    let final_output = graph.forward(&f.input_ids, &f.attention_mask, &pixel_values)?;
    let pixels = graph.patches()?;
    let reloaded = VulkanGemma4Vision::from_hf_package(
        device.clone(),
        output.join("step-2"),
        f.batch_size,
        f.seq_len,
        f.image_count,
        side,
    )?;
    let reload_pixels = if pixel_domain == "raw_patch_values" {
        raw_pixels(&f.rgb)
    } else {
        reloaded.preprocess_rgb8(&f.rgb)?
    };
    let reload_output = reloaded.forward(&f.input_ids, &f.attention_mask, &reload_pixels)?;
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema":1,"scope":"raw_rgb_gemma4_full_parameter_training",
            "pixel_input_domain":pixel_domain,
            "device":device.name(),"vendor_id":device.vendor_id(),"device_id":device.device_id(),
            "pixels":pixels,"positions":graph.positions(),"alternate":alternate,
            "steps":steps,"final":final_output,"reload":reload_output,
        }))?,
    )?;
    Ok(())
}
