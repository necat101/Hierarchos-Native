use std::{fs, path::PathBuf};

use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{VulkanDevice, VulkanSeq2SeqTransformer, VulkanTransformer};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct Fixture {
    batch_size: usize,
    seq_len: usize,
    input_ids: Vec<u32>,
    attention_mask: Vec<f32>,
    #[serde(default)]
    token_type_ids: Option<Vec<u32>>,
    #[serde(default)]
    encoder_input_ids: Option<Vec<u32>>,
    #[serde(default)]
    encoder_attention_mask: Option<Vec<f32>>,
}

#[derive(Debug, Serialize)]
struct Report {
    device: String,
    architecture: String,
    batch_size: usize,
    seq_len: usize,
    vocab_size: usize,
    logits: Vec<f32>,
}

fn main() -> Result<()> {
    // Keep validation behavior aligned with the other Vulkan harnesses on
    // Windows. The unified Transformer graph has a large constructor frame in
    // debug builds, while the executable main thread has a comparatively small
    // fixed stack. Run the harness on an explicitly sized host stack so parity
    // failures reflect model/backend behavior rather than a launcher artifact.
    std::thread::Builder::new()
        .name("hierarchos-transformer-logits".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(run)?
        .join()
        .map_err(|_| anyhow::anyhow!("transformer logits worker thread panicked"))?
}

fn run() -> Result<()> {
    let mut model = None;
    let mut fixture = None;
    let mut output = None;
    let mut export_model = None;
    let mut device_index = None;
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
            "--export-model" => {
                export_model = Some(PathBuf::from(
                    args.next().context("missing --export-model value")?,
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
            "-h" | "--help" => {
                eprintln!(
                    "usage: transformer_logits --model DIR --fixture fixture.json [--output report.json] [--export-model DIR] [--device-index N]"
                );
                return Ok(());
            }
            other => bail!("unknown argument {other:?}"),
        }
    }

    let model = model.context("--model is required")?;
    let fixture_path = fixture.context("--fixture is required")?;
    let fixture: Fixture = serde_json::from_slice(
        &fs::read(&fixture_path).with_context(|| format!("reading {}", fixture_path.display()))?,
    )
    .with_context(|| format!("decoding {}", fixture_path.display()))?;
    let rows = fixture
        .batch_size
        .checked_mul(fixture.seq_len)
        .context("fixture row count overflow")?;
    if fixture.input_ids.len() != rows || fixture.attention_mask.len() != rows {
        bail!(
            "fixture input_ids and attention_mask must contain batch_size*seq_len={rows} entries"
        );
    }
    if let Some(token_type_ids) = fixture.token_type_ids.as_ref() {
        if token_type_ids.len() != rows {
            bail!("fixture token_type_ids must contain batch_size*seq_len={rows} entries");
        }
    }

    let device = match device_index {
        Some(index) => VulkanDevice::new_with_index(index)?,
        None => VulkanDevice::new()?,
    };
    if let Some(encoder_ids) = fixture.encoder_input_ids.as_deref() {
        if fixture.batch_size == 0 || fixture.token_type_ids.is_some() {
            bail!("seq2seq logit fixtures require a positive batch_size and no token_type_ids");
        }
        if encoder_ids.is_empty()
            || !encoder_ids.len().is_multiple_of(fixture.batch_size)
            || fixture
                .encoder_attention_mask
                .as_ref()
                .is_some_and(|mask| mask.len() != encoder_ids.len())
        {
            bail!(
                "seq2seq fixture requires nonempty encoder_input_ids and a matching encoder mask"
            );
        }
    } else if fixture.encoder_attention_mask.is_some() {
        bail!("encoder_attention_mask requires encoder_input_ids");
    }
    let report = if let Some(encoder_ids) = fixture.encoder_input_ids.as_deref() {
        let graph = VulkanSeq2SeqTransformer::from_hf_package_with_batch_size(
            device,
            &model,
            fixture.batch_size,
            encoder_ids.len() / fixture.batch_size,
            fixture.seq_len,
        )?;
        let logits = graph.forward_logits(
            encoder_ids,
            fixture.encoder_attention_mask.as_deref(),
            &fixture.input_ids,
            &fixture.attention_mask,
        )?;
        Report {
            device: graph.decoder().device_name().to_owned(),
            architecture: graph
                .decoder()
                .config()
                .architecture
                .model_type()
                .to_owned(),
            batch_size: fixture.batch_size,
            seq_len: fixture.seq_len,
            vocab_size: graph.decoder().config().vocab_size,
            logits,
        }
    } else {
        eprintln!("transformer_logits: constructing graph");
        let graph = VulkanTransformer::from_hf_package(
            device,
            &model,
            fixture.batch_size,
            fixture.seq_len,
        )?;
        eprintln!("transformer_logits: graph constructed; running forward");
        let logits = match fixture.token_type_ids.as_deref() {
            Some(token_type_ids) => graph.forward_logits_with_token_type_ids(
                &fixture.input_ids,
                Some(token_type_ids),
                &fixture.attention_mask,
            )?,
            None => graph.forward_logits(&fixture.input_ids, &fixture.attention_mask)?,
        };
        if let Some(export_dir) = export_model.as_deref() {
            graph.export_hf_package(&model, export_dir)?;
        }
        eprintln!("transformer_logits: forward complete");
        Report {
            device: graph.device_name().to_owned(),
            architecture: graph.config().architecture.model_type().to_owned(),
            batch_size: fixture.batch_size,
            seq_len: fixture.seq_len,
            vocab_size: graph.config().vocab_size,
            logits,
        }
    };
    let encoded = serde_json::to_vec(&report)?;
    if let Some(output) = output {
        fs::write(&output, &encoded).with_context(|| format!("writing {}", output.display()))?;
    } else {
        println!("{}", String::from_utf8(encoded)?);
    }
    Ok(())
}
