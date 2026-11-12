//! Raw RGB image -> CLIP -> projector -> text logits, full backward and export.
use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{
    CpuAdamWHyperParams, LlavaForward, LlavaTrainResult, VulkanDevice, VulkanImageText,
};
use serde::Deserialize;
use std::{fs, path::PathBuf};

#[derive(Deserialize)]
struct Fixture {
    batch_size: usize,
    seq_len: usize,
    image_count: usize,
    input_ids: Vec<u32>,
    attention_mask: Vec<f32>,
    decoder_input_ids: Option<Vec<u32>>,
    decoder_attention_mask: Option<Vec<f32>>,
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

impl Fixture {
    fn decoder_inputs(&self) -> Result<Option<(&[u32], &[f32])>> {
        match (&self.decoder_input_ids, &self.decoder_attention_mask) {
            (None, None) => Ok(None),
            (Some(ids), Some(mask)) => Ok(Some((ids, mask))),
            _ => bail!("decoder ids and mask must be supplied together"),
        }
    }

    fn forward(&self, graph: &VulkanImageText, pixels: &[f32]) -> Result<LlavaForward> {
        if let Some((ids, mask)) = self.decoder_inputs()? {
            graph.forward_seq2seq(&self.input_ids, &self.attention_mask, ids, mask, pixels)
        } else {
            graph.forward(&self.input_ids, &self.attention_mask, pixels)
        }
    }

    fn train(
        &self,
        graph: &mut VulkanImageText,
        pixels: &[f32],
        hyper: CpuAdamWHyperParams,
    ) -> Result<LlavaTrainResult> {
        if let Some((ids, mask)) = self.decoder_inputs()? {
            graph.train_seq2seq_with_optimizer(
                &self.input_ids,
                &self.attention_mask,
                ids,
                mask,
                pixels,
                &self.targets,
                &self.loss_weights,
                hyper,
            )
        } else {
            graph.train_step_with_optimizer(
                &self.input_ids,
                &self.attention_mask,
                pixels,
                &self.targets,
                &self.loss_weights,
                hyper,
            )
        }
    }
}
fn main() -> Result<()> {
    std::thread::Builder::new()
        .name("image-text-pipeline".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(run)?
        .join()
        .map_err(|_| anyhow::anyhow!("image-text probe panicked"))?
}
fn run() -> Result<()> {
    let mut model = None;
    let mut fixture = None;
    let mut output = None;
    let mut stages = None;
    let mut gradients = None;
    let mut loss_scale = None;
    let mut loss_scaler_state = None;
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        let value = PathBuf::from(args.next().context("missing argument value")?);
        match arg.to_str() {
            Some("--model") => model = Some(value),
            Some("--fixture") => fixture = Some(value),
            Some("--output") => output = Some(value),
            Some("--stages") => stages = Some(value),
            Some("--gradients") => gradients = Some(value),
            Some("--loss-scale") => loss_scale = Some(value.to_string_lossy().parse::<f32>()?),
            Some("--loss-scaler-state") => loss_scaler_state = Some(value),
            _ => bail!("unknown argument {arg:?}"),
        }
    }
    let model = model.context("--model required")?;
    let output = output.context("--output required")?;
    let f: Fixture = serde_json::from_slice(&fs::read(fixture.context("--fixture required")?)?)?;
    if f.steps != 2 {
        bail!("qualification requires exactly two composed updates");
    }
    f.decoder_inputs()?;
    let device = VulkanDevice::new()?;
    let phases_before = hierarchos_vulkan::wmma_linear::linear_wmma_phase_counts();
    let accumulation_before = hierarchos_vulkan::wmma_linear::linear_wmma_precision_counts();
    let reduction_before=hierarchos_vulkan::wmma_linear::full_fp16_reduction_fallbacks();
    let counts_before = hierarchos_vulkan::wmma_linear::linear_wmma_dispatch_counts();
    let mut graph = VulkanImageText::from_hf_package(
        device.clone(),
        &model,
        f.batch_size,
        f.seq_len,
        f.image_count,
    )?;
    if let Some(scale)=loss_scale {
        graph.set_loss_scaler(Some(hierarchos_vulkan::TransformerLossScaleState{scale,..Default::default()}))?;
    }
    if let Some(path)=loss_scaler_state {
        graph.set_loss_scaler(Some(serde_json::from_slice(&fs::read(path)?)?))?;
    }
    let pixels = graph.preprocess_rgb8(&f.rgb)?;
    if let Some(path) = gradients {
        let value = graph.debug_train_stages(
            &f.input_ids,
            &f.attention_mask,
            &pixels,
            &f.targets,
            &f.loss_weights,
            CpuAdamWHyperParams {
                lr: f.lr,
                beta1: f.beta1,
                beta2: f.beta2,
                eps: f.eps,
                weight_decay: f.weight_decay,
            },
        )?;
        fs::write(path, serde_json::to_vec(&value)?)?;
        return Ok(());
    }
    if let Some(path) = stages {
        fs::write(
            path,
            serde_json::to_vec(&graph.debug_text_stages(
                &f.input_ids,
                &f.attention_mask,
                &pixels,
            )?)?,
        )?;
        return Ok(());
    }
    let alternate_pixels = graph.preprocess_rgb8(&f.alternate_rgb)?;
    let started = std::time::Instant::now();
    let alternate = f.forward(&graph, &alternate_pixels)?;
    fs::create_dir(&output).context("output must be a new directory")?;
    let hyper = CpuAdamWHyperParams {
        lr: f.lr,
        beta1: f.beta1,
        beta2: f.beta2,
        eps: f.eps,
        weight_decay: f.weight_decay,
    };
    let mut steps = Vec::new();
    for attempt in 0..f.steps {
        let forward = f.forward(&graph, &pixels)?;
        let result = f.train(&mut graph, &pixels, hyper)?;
        graph.export_hf_package(&model, output.join(format!("step-{}", attempt+1)))?;
        steps.push(serde_json::json!({"attempt":attempt+1,"forward":forward,"training":result}));
    }
    let final_output = f.forward(&graph, &pixels)?;
    let reloaded = VulkanImageText::from_hf_package(
        device.clone(),
        output.join("step-2"),
        f.batch_size,
        f.seq_len,
        f.image_count,
    )?;
    let reload_pixels = reloaded.preprocess_rgb8(&f.rgb)?;
    let reload_output = f.forward(&reloaded, &reload_pixels)?;
    let phases = hierarchos_vulkan::wmma_linear::linear_wmma_phase_counts();
    let accumulation = hierarchos_vulkan::wmma_linear::linear_wmma_precision_counts();
    let counts = hierarchos_vulkan::wmma_linear::linear_wmma_dispatch_counts();
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema":1,"scope":"shared_clip_full_parameter_image_text_training",
            "architecture":graph.config().architecture.model_type(),
            "device":device.name(),"vendor_id":device.vendor_id(),"device_id":device.device_id(),
            "elapsed_seconds":started.elapsed().as_secs_f64(),
            "precision": {
                "requested": hierarchos_vulkan::wmma::WmmaMode::from_env()?.label(),
                "mixed_fp32_accumulation":accumulation[0]-accumulation_before[0],
                "full_fp16_accumulation":accumulation[1]-accumulation_before[1],
                "full_fp16_max_reduction":hierarchos_vulkan::wmma_linear::FULL_FP16_MAX_REDUCTION,
                "full_fp16_reduction_fallbacks":hierarchos_vulkan::wmma_linear::full_fp16_reduction_fallbacks()-reduction_before,
                "master_weights":"fp32","optimizer_state":"fp32",
                "automatic_loss_scaling":graph.loss_scaler_state()?.is_some(),
                "loss_scaler":graph.loss_scaler_state()?,
                "cooperative_dispatches": counts.cooperative - counts_before.cooperative,
                "forward": phases[0] - phases_before[0],
                "weight_grad": phases[1] - phases_before[1],
                "input_grad": phases[2] - phases_before[2],
                "shape_fallbacks": counts.shape_fallbacks - counts_before.shape_fallbacks,
            },
            "pixels":pixels,"alternate":alternate,"steps":steps,"final":final_output,"reload":reload_output,
        }))?,
    )?;
    Ok(())
}
