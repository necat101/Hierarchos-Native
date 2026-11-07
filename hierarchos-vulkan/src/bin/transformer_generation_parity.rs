use std::{fs, path::PathBuf};

use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{
    VulkanDevice, VulkanTransformer, VulkanTransformerConfig, VulkanTransformerGenerationConfig,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct Fixture {
    input_ids: Vec<u32>,
    max_new_tokens: usize,
}

#[derive(Debug, Serialize)]
struct Report {
    device: String,
    vendor_id: u32,
    device_id: u32,
    architecture: String,
    streamed_linears: usize,
    source_first_parameters: bool,
    use_cache: bool,
    prompt_length: usize,
    sequence: Vec<u32>,
    generated_ids: Vec<u32>,
    logits: Vec<Vec<f32>>,
    logits_bits: Vec<Vec<u32>>,
}

fn main() -> Result<()> {
    std::thread::Builder::new()
        .name("hierarchos-transformer-generation-parity".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(run)?
        .join()
        .map_err(|_| anyhow::anyhow!("transformer generation parity worker thread panicked"))?
}

fn run() -> Result<()> {
    let mut model = None;
    let mut fixture = None;
    let mut output = None;
    let mut device_index = None;
    let mut use_cache = true;
    let mut stream_parameters = false;
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--model" => model = Some(PathBuf::from(args.next().context("missing --model value")?)),
            "--fixture" => {
                fixture = Some(PathBuf::from(
                    args.next().context("missing --fixture value")?,
                ))
            }
            "--output" => {
                output = Some(PathBuf::from(
                    args.next().context("missing --output value")?,
                ))
            }
            "--device-index" => {
                device_index = Some(
                    args.next()
                        .context("missing --device-index value")?
                        .to_string_lossy()
                        .parse::<usize>()
                        .context("invalid --device-index")?,
                )
            }
            "--use-cache" => {
                use_cache = args
                    .next()
                    .context("missing --use-cache value")?
                    .to_string_lossy()
                    .parse::<bool>()
                    .context("invalid --use-cache")?
            }
            "--stream-parameters" => stream_parameters = true,
            "-h" | "--help" => {
                eprintln!("usage: transformer_generation_parity --model DIR --fixture fixture.json --output report.json [--use-cache true|false] [--stream-parameters] [--device-index N]");
                return Ok(());
            }
            other => bail!("unknown argument {other:?}"),
        }
    }

    let model = model.context("--model is required")?;
    let fixture_path = fixture.context("--fixture is required")?;
    let output_path = output.context("--output is required")?;
    let fixture: Fixture = serde_json::from_slice(&fs::read(&fixture_path)?)?;
    if fixture.input_ids.is_empty() || fixture.max_new_tokens == 0 {
        bail!("generation fixture requires nonempty input_ids and max_new_tokens > 0");
    }
    let sequence_capacity = fixture
        .input_ids
        .len()
        .checked_add(fixture.max_new_tokens)
        .context("generation sequence length overflow")?;
    let device = match device_index {
        Some(index) => VulkanDevice::new_with_index(index)?,
        None => VulkanDevice::new()?,
    };
    let graph = if stream_parameters {
        let config = VulkanTransformerConfig::from_hf_config(model.join("config.json"))?;
        VulkanTransformer::from_hf_package_with_streamed_parameters(
            device.clone(),
            &model,
            config,
            1,
            sequence_capacity,
        )?
    } else {
        VulkanTransformer::from_hf_package(device.clone(), &model, 1, sequence_capacity)?
    };
    let config = VulkanTransformerGenerationConfig {
        max_new_tokens: Some(fixture.max_new_tokens),
        max_length: None,
        do_sample: false,
        use_cache,
        output_logits: true,
        return_dict_in_generate: true,
        eos_token_ids: Vec::new(),
        ..Default::default()
    };
    let generated = graph.generate_output(&fixture.input_ids, &config)?;
    let mut sequences = generated.sequences.into_iter();
    let sequence = sequences
        .next()
        .context("generation returned no sequence")?;
    if sequences.next().is_some() {
        bail!("generation parity fixture expected exactly one sequence");
    }
    let generated_ids = sequence.generated_ids().to_vec();
    let logits = sequence
        .logits
        .clone()
        .context("generation did not return raw per-step logits")?;
    if logits.len() != fixture.max_new_tokens {
        bail!(
            "generation returned {} logit steps, expected {}",
            logits.len(),
            fixture.max_new_tokens
        );
    }
    let report = Report {
        device: graph.device_name().to_owned(),
        vendor_id: device.vendor_id(),
        device_id: device.device_id(),
        architecture: graph.config().architecture.model_type().to_owned(),
        streamed_linears: graph.streamed_linear_count(),
        source_first_parameters: graph.parameters_streamed_during_loading(),
        use_cache,
        prompt_length: sequence.prompt_length,
        sequence: sequence.sequence,
        generated_ids,
        logits_bits: logits
            .iter()
            .map(|step| step.iter().map(|value| value.to_bits()).collect())
            .collect(),
        logits,
    };
    fs::write(&output_path, serde_json::to_vec(&report)?)?;
    Ok(())
}
