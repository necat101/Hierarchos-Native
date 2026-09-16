use std::{
    fs::{self, File},
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{bail, Context, Result};
use hf_hub::HFClientSync;
use hierarchos_vulkan::{
    AdamWHyperParams, VulkanDevice, VulkanTransformer, VulkanTransformerArchitecture,
    VulkanTransformerConfig, VulkanTransformerLoraConfig,
};
use serde::Serialize;
use serde_json::Value;
use tokenizers::Tokenizer;

#[derive(Debug)]
struct Args {
    model: Option<PathBuf>,
    hf_model: Option<String>,
    hf_model_revision: Option<String>,
    dataset: Option<PathBuf>,
    hf_dataset: Option<String>,
    hf_dataset_file: Option<String>,
    hf_dataset_revision: Option<String>,
    output: PathBuf,
    text_field: String,
    batch_size: usize,
    seq_len: usize,
    epochs: usize,
    max_steps: Option<usize>,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    device_index: Option<usize>,
    seed: u64,
    log_steps: usize,
    pad_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    lora_rank: Option<usize>,
    lora_alpha: Option<usize>,
    lora_targets: Vec<String>,
    lora_adapter: Option<PathBuf>,
}

#[derive(Clone, Debug)]
struct Sequence {
    input_ids: Vec<u32>,
    targets: Vec<u32>,
    attention_mask: Vec<f32>,
    loss_weights: Vec<f32>,
    encoder_context: Option<Arc<EncoderContext>>,
}

#[derive(Debug)]
struct EncoderContext {
    hidden_states: Vec<f32>,
    attention_mask: Vec<f32>,
    seq_len: usize,
}

#[derive(Debug, Serialize)]
struct TrainingReport {
    backend: &'static str,
    device: String,
    architecture: String,
    model_source: String,
    dataset_source: String,
    batch_size: usize,
    seq_len: usize,
    encoder_seq_len: Option<usize>,
    epochs_requested: usize,
    optimizer_steps: usize,
    training_sequences: usize,
    supervised_tokens: usize,
    final_loss: f32,
    peft: bool,
    output: String,
}

fn required_arg(
    args: &mut impl Iterator<Item = std::ffi::OsString>,
    flag: &str,
) -> Result<std::ffi::OsString> {
    args.next()
        .with_context(|| format!("missing value for {flag}"))
}

fn parse_num<T: std::str::FromStr>(raw: std::ffi::OsString, flag: &str) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    raw.to_string_lossy()
        .parse::<T>()
        .map_err(|error| anyhow::anyhow!("invalid value for {flag}: {error}"))
}

fn usage() -> &'static str {
    "hierarchos-vulkan-transformer-train \
  (--model MODEL_DIR | --hf-model OWNER/REPO) \
  (--dataset DATA.jsonl|DATA.json|DATA.txt | --hf-dataset OWNER/REPO --hf-dataset-file FILE) \
  --output OUTPUT_DIR [options]\n\n\
Native data/tokenizer options:\n\
  --hf-model-revision REV       Hugging Face model revision (default: main)\n\
  --hf-dataset-revision REV     Hugging Face dataset revision (default: main)\n\
  --text-field FIELD            JSON field containing training text (default: text; dotted paths supported)\n\
  --pad-token-id N              Override padding token id\n\
  --eos-token-id N              Override end-of-sequence token id\n\n\
Training options:\n\
  --batch-size N                Fixed Vulkan graph batch size (default: 1)\n\
  --seq-len N                   Fixed Vulkan graph sequence length (default: 128)\n\
  --epochs N                    Number of deterministic dataset passes (default: 1)\n\
  --max-steps N                 Optional optimizer-step cap\n\
  --learning-rate F             AdamW learning rate (default: 5e-5)\n\
  --beta1 F                     AdamW beta1 (default: 0.9)\n\
  --beta2 F                     AdamW beta2 (default: 0.999)\n\
  --eps F                       AdamW epsilon (default: 1e-8)\n\
  --weight-decay F              AdamW weight decay (default: 0.0)\n\
  --device-index N              Vulkan physical-device index\n\
  --log-steps N                 Emit progress JSON every N updates (default: 10)\n\
  --seed N                      LoRA initialization + Transformer dropout seed (default: 42)\n\n\
PEFT options:\n\
  --lora-rank N                 Enable native Vulkan LoRA training\n\
  --lora-alpha N                LoRA alpha (default: 2 * rank)\n\
  --lora-target NAME            Repeatable target module suffix; architecture default if omitted\n\
  --lora-adapter DIR            Resume weights from an existing PEFT/peft-rs adapter directory\n\n\
Dataset rows may be raw text, JSON objects with --text-field, or pretokenized JSON objects\n\
with input_ids and optional labels/targets + attention_mask. labels=-100 are masked.\n\
Cross-attention decoder rows may additionally provide encoder_hidden_states as either a flat\n\
[encoder_sequence * hidden] array or nested [encoder_sequence, hidden] arrays, plus an optional\n\
encoder_attention_mask. Encoder geometry must be fixed across the dataset; an omitted encoder\n\
mask defaults to all-visible positions. This path trains the decoder and exposes the encoder-state\n\
gradient natively without PyTorch.\n\
Raw text and unlabeled input_ids use next-token labels and therefore require a causal model.\n\
Bidirectional masked-LM models require explicit aligned labels/targets (use -100 to ignore rows).\n\
The trainer contains no PyTorch dependency: tokenization/I/O are Rust host work; model forward,\n\
architecture-appropriate attention, loss, backward, LoRA, and AdamW math execute through Vulkan."
}

fn parse_args() -> Result<Args> {
    let mut parsed = Args {
        model: None,
        hf_model: None,
        hf_model_revision: None,
        dataset: None,
        hf_dataset: None,
        hf_dataset_file: None,
        hf_dataset_revision: None,
        output: PathBuf::new(),
        text_field: "text".to_owned(),
        batch_size: 1,
        seq_len: 128,
        epochs: 1,
        max_steps: None,
        learning_rate: 5.0e-5,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1.0e-8,
        weight_decay: 0.0,
        device_index: None,
        seed: 42,
        log_steps: 10,
        pad_token_id: None,
        eos_token_id: None,
        lora_rank: None,
        lora_alpha: None,
        lora_targets: Vec::new(),
        lora_adapter: None,
    };
    let mut args = std::env::args_os().skip(1);
    while let Some(raw) = args.next() {
        match raw.to_string_lossy().as_ref() {
            "--model" => parsed.model = Some(PathBuf::from(required_arg(&mut args, "--model")?)),
            "--hf-model" => {
                parsed.hf_model = Some(
                    required_arg(&mut args, "--hf-model")?
                        .to_string_lossy()
                        .into_owned(),
                )
            }
            "--hf-model-revision" => {
                parsed.hf_model_revision = Some(
                    required_arg(&mut args, "--hf-model-revision")?
                        .to_string_lossy()
                        .into_owned(),
                )
            }
            "--dataset" => {
                parsed.dataset = Some(PathBuf::from(required_arg(&mut args, "--dataset")?))
            }
            "--hf-dataset" => {
                parsed.hf_dataset = Some(
                    required_arg(&mut args, "--hf-dataset")?
                        .to_string_lossy()
                        .into_owned(),
                )
            }
            "--hf-dataset-file" => {
                parsed.hf_dataset_file = Some(
                    required_arg(&mut args, "--hf-dataset-file")?
                        .to_string_lossy()
                        .into_owned(),
                )
            }
            "--hf-dataset-revision" => {
                parsed.hf_dataset_revision = Some(
                    required_arg(&mut args, "--hf-dataset-revision")?
                        .to_string_lossy()
                        .into_owned(),
                )
            }
            "--output" => parsed.output = PathBuf::from(required_arg(&mut args, "--output")?),
            "--text-field" => {
                parsed.text_field = required_arg(&mut args, "--text-field")?
                    .to_string_lossy()
                    .into_owned()
            }
            "--batch-size" => {
                parsed.batch_size =
                    parse_num(required_arg(&mut args, "--batch-size")?, "--batch-size")?
            }
            "--seq-len" => {
                parsed.seq_len = parse_num(required_arg(&mut args, "--seq-len")?, "--seq-len")?
            }
            "--epochs" => {
                parsed.epochs = parse_num(required_arg(&mut args, "--epochs")?, "--epochs")?
            }
            "--max-steps" => {
                parsed.max_steps = Some(parse_num(
                    required_arg(&mut args, "--max-steps")?,
                    "--max-steps",
                )?)
            }
            "--learning-rate" => {
                parsed.learning_rate = parse_num(
                    required_arg(&mut args, "--learning-rate")?,
                    "--learning-rate",
                )?
            }
            "--beta1" => parsed.beta1 = parse_num(required_arg(&mut args, "--beta1")?, "--beta1")?,
            "--beta2" => parsed.beta2 = parse_num(required_arg(&mut args, "--beta2")?, "--beta2")?,
            "--eps" => parsed.eps = parse_num(required_arg(&mut args, "--eps")?, "--eps")?,
            "--weight-decay" => {
                parsed.weight_decay =
                    parse_num(required_arg(&mut args, "--weight-decay")?, "--weight-decay")?
            }
            "--device-index" => {
                parsed.device_index = Some(parse_num(
                    required_arg(&mut args, "--device-index")?,
                    "--device-index",
                )?)
            }
            "--seed" => parsed.seed = parse_num(required_arg(&mut args, "--seed")?, "--seed")?,
            "--log-steps" => {
                parsed.log_steps =
                    parse_num(required_arg(&mut args, "--log-steps")?, "--log-steps")?
            }
            "--pad-token-id" => {
                parsed.pad_token_id = Some(parse_num(
                    required_arg(&mut args, "--pad-token-id")?,
                    "--pad-token-id",
                )?)
            }
            "--eos-token-id" => {
                parsed.eos_token_id = Some(parse_num(
                    required_arg(&mut args, "--eos-token-id")?,
                    "--eos-token-id",
                )?)
            }
            "--lora-rank" => {
                parsed.lora_rank = Some(parse_num(
                    required_arg(&mut args, "--lora-rank")?,
                    "--lora-rank",
                )?)
            }
            "--lora-alpha" => {
                parsed.lora_alpha = Some(parse_num(
                    required_arg(&mut args, "--lora-alpha")?,
                    "--lora-alpha",
                )?)
            }
            "--lora-target" => parsed.lora_targets.push(
                required_arg(&mut args, "--lora-target")?
                    .to_string_lossy()
                    .into_owned(),
            ),
            "--lora-adapter" => {
                parsed.lora_adapter =
                    Some(PathBuf::from(required_arg(&mut args, "--lora-adapter")?))
            }
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other => bail!("unknown argument {other:?}\n\n{}", usage()),
        }
    }
    validate_args(&parsed)?;
    Ok(parsed)
}

fn validate_args(args: &Args) -> Result<()> {
    if args.model.is_some() == args.hf_model.is_some() {
        bail!("select exactly one model source: --model or --hf-model");
    }
    if args.dataset.is_some() == args.hf_dataset.is_some() {
        bail!("select exactly one dataset source: --dataset or --hf-dataset");
    }
    if args.hf_dataset.is_some() && args.hf_dataset_file.is_none() {
        bail!("--hf-dataset requires --hf-dataset-file");
    }
    if args.output.as_os_str().is_empty() {
        bail!("--output is required");
    }
    if args.batch_size == 0 || args.seq_len == 0 || args.epochs == 0 {
        bail!("--batch-size, --seq-len, and --epochs must be positive");
    }
    if args.max_steps == Some(0) || args.log_steps == 0 {
        bail!("--max-steps (when supplied) and --log-steps must be positive");
    }
    if args.lora_adapter.is_some() && args.lora_rank.is_some() {
        bail!("--lora-adapter and --lora-rank are mutually exclusive");
    }
    if !args.learning_rate.is_finite() || args.learning_rate < 0.0 {
        bail!("--learning-rate must be finite and non-negative");
    }
    if !(0.0..1.0).contains(&args.beta1) || !(0.0..1.0).contains(&args.beta2) {
        bail!("--beta1 and --beta2 must be in [0, 1)");
    }
    if !args.eps.is_finite() || args.eps < 0.0 {
        bail!("--eps must be finite and non-negative");
    }
    if !args.weight_decay.is_finite() || args.weight_decay < 0.0 {
        bail!("--weight-decay must be finite and non-negative");
    }
    Ok(())
}

fn split_hf_id(id: &str) -> Result<(&str, &str)> {
    if let Some((owner, name)) = id.split_once('/') {
        if owner.is_empty() || name.is_empty() || name.contains('/') {
            bail!("invalid Hugging Face repository id {id:?}; expected OWNER/REPO");
        }
        Ok((owner, name))
    } else if id.is_empty() {
        bail!("Hugging Face repository id must not be empty")
    } else {
        // hf-hub supports short-form model ids such as `gpt2` by leaving owner empty.
        Ok(("", id))
    }
}

fn download_hf_model(id: &str, revision: Option<&str>) -> Result<PathBuf> {
    let client = HFClientSync::new().context("initializing Hugging Face Hub client")?;
    let (owner, name) = split_hf_id(id)?;
    let repo = client.model(owner, name);
    repo.snapshot_download()
        .maybe_revision(revision.map(str::to_owned))
        .allow_patterns(vec![
            "config.json".to_owned(),
            "generation_config.json".to_owned(),
            "tokenizer.json".to_owned(),
            "tokenizer_config.json".to_owned(),
            "special_tokens_map.json".to_owned(),
            "added_tokens.json".to_owned(),
            "vocab.json".to_owned(),
            "merges.txt".to_owned(),
            "tokenizer.model".to_owned(),
            "model.safetensors.index.json".to_owned(),
            "model*.safetensors".to_owned(),
        ])
        .send()
        .with_context(|| format!("downloading Hugging Face model {id}"))
}

fn download_hf_dataset_file(id: &str, filename: &str, revision: Option<&str>) -> Result<PathBuf> {
    let client = HFClientSync::new().context("initializing Hugging Face Hub client")?;
    let (owner, name) = split_hf_id(id)?;
    client
        .dataset(owner, name)
        .download_file()
        .filename(filename.to_owned())
        .maybe_revision(revision.map(str::to_owned))
        .send()
        .with_context(|| format!("downloading Hugging Face dataset file {id}:{filename}"))
}

fn load_tokenizer(model_dir: &Path) -> Result<Option<Tokenizer>> {
    let path = model_dir.join("tokenizer.json");
    if !path.is_file() {
        return Ok(None);
    }
    Tokenizer::from_file(&path)
        .map(Some)
        .map_err(|error| anyhow::anyhow!("loading {}: {error}", path.display()))
}

fn special_token_text(value: &Value) -> Option<&str> {
    match value {
        Value::String(text) => Some(text),
        Value::Object(map) => map.get("content").and_then(Value::as_str),
        _ => None,
    }
}

fn infer_special_token_id(model_dir: &Path, tokenizer: &Tokenizer, name: &str) -> Option<u32> {
    let config = fs::read(model_dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
    if let Some(config) = config.as_ref() {
        if let Some(id) = config
            .get(format!("{name}_token_id"))
            .and_then(Value::as_u64)
            .and_then(|id| u32::try_from(id).ok())
        {
            return Some(id);
        }
        if let Some(text) = config
            .get(format!("{name}_token"))
            .and_then(special_token_text)
        {
            if let Some(id) = tokenizer.token_to_id(text) {
                return Some(id);
            }
        }
    }
    let fallbacks: &[&str] = match name {
        "eos" => &["<|endoftext|>", "</s>", "<eos>"],
        "pad" => &["<|pad|>", "<pad>"],
        _ => &[],
    };
    fallbacks
        .iter()
        .find_map(|token| tokenizer.token_to_id(token))
}

fn json_field<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = value;
    for component in path.split('.') {
        current = current.get(component)?;
    }
    Some(current)
}

fn parse_u32_array(value: &Value, field: &str) -> Result<Vec<u32>> {
    let values = value
        .as_array()
        .with_context(|| format!("{field} must be an integer array"))?;
    values
        .iter()
        .map(|value| {
            let raw = value
                .as_u64()
                .with_context(|| format!("{field} contains a non-u32 value"))?;
            u32::try_from(raw).with_context(|| format!("{field} contains a value larger than u32"))
        })
        .collect()
}

fn parse_labels(value: &Value, field: &str) -> Result<Vec<u32>> {
    let values = value
        .as_array()
        .with_context(|| format!("{field} must be an integer array"))?;
    values
        .iter()
        .map(|value| {
            let raw = value
                .as_i64()
                .with_context(|| format!("{field} contains a non-integer value"))?;
            if raw == -100 {
                Ok(u32::MAX)
            } else if raw < 0 {
                bail!("{field} contains unsupported negative label {raw}; only -100 is masked")
            } else {
                u32::try_from(raw)
                    .with_context(|| format!("{field} contains a value larger than u32"))
            }
        })
        .collect()
}

fn parse_mask(value: &Value, field: &str) -> Result<Vec<f32>> {
    value
        .as_array()
        .with_context(|| format!("{field} must be a numeric array"))?
        .iter()
        .map(|value| {
            let raw = value
                .as_f64()
                .with_context(|| format!("{field} contains a non-numeric value"))?;
            let raw = raw as f32;
            if !raw.is_finite() || raw < 0.0 {
                bail!("{field} values must be finite and non-negative");
            }
            Ok(raw)
        })
        .collect()
}

fn flatten_f32_tensor(value: &Value, field: &str, output: &mut Vec<f32>) -> Result<()> {
    if let Some(values) = value.as_array() {
        for value in values {
            flatten_f32_tensor(value, field, output)?;
        }
        return Ok(());
    }
    let raw = value
        .as_f64()
        .with_context(|| format!("{field} must contain only numeric values"))? as f32;
    if !raw.is_finite() {
        bail!("{field} values must be finite");
    }
    output.push(raw);
    Ok(())
}

fn parse_encoder_context(
    object: &serde_json::Map<String, Value>,
    hidden_size: usize,
) -> Result<Option<Arc<EncoderContext>>> {
    let Some(hidden_value) = object.get("encoder_hidden_states") else {
        if object.contains_key("encoder_attention_mask") {
            bail!("encoder_attention_mask was supplied without encoder_hidden_states");
        }
        return Ok(None);
    };
    if hidden_size == 0 {
        bail!("Transformer hidden size must be positive for encoder hidden states");
    }
    let mut hidden_states = Vec::new();
    flatten_f32_tensor(hidden_value, "encoder_hidden_states", &mut hidden_states)?;
    if hidden_states.is_empty() || hidden_states.len() % hidden_size != 0 {
        bail!(
            "encoder_hidden_states must contain a positive multiple of model hidden size {hidden_size}, got {} values",
            hidden_states.len()
        );
    }
    let seq_len = hidden_states.len() / hidden_size;
    let attention_mask = object
        .get("encoder_attention_mask")
        .map(|value| parse_mask(value, "encoder_attention_mask"))
        .transpose()?
        .unwrap_or_else(|| vec![1.0; seq_len]);
    if attention_mask.len() != seq_len {
        bail!(
            "encoder_attention_mask must contain exactly {seq_len} values, got {}",
            attention_mask.len()
        );
    }
    Ok(Some(Arc::new(EncoderContext {
        hidden_states,
        attention_mask,
        seq_len,
    })))
}

fn push_aligned_chunks(
    output: &mut Vec<Sequence>,
    ids: &[u32],
    labels: &[u32],
    mask: Option<&[f32]>,
    weights: Option<&[f32]>,
    seq_len: usize,
    pad_token_id: u32,
) -> Result<()> {
    if ids.len() != labels.len() {
        bail!("pretokenized input_ids and labels/targets must have identical lengths");
    }
    if let Some(mask) = mask {
        if mask.len() != ids.len() {
            bail!("pretokenized attention_mask must match input_ids length");
        }
    }
    if let Some(weights) = weights {
        if weights.len() != ids.len() {
            bail!("pretokenized loss_weights must match input_ids length");
        }
    }
    for start in (0..ids.len()).step_by(seq_len) {
        let count = (ids.len() - start).min(seq_len);
        let mut sequence = Sequence {
            input_ids: vec![pad_token_id; seq_len],
            targets: vec![u32::MAX; seq_len],
            attention_mask: vec![0.0; seq_len],
            loss_weights: vec![0.0; seq_len],
            encoder_context: None,
        };
        sequence.input_ids[..count].copy_from_slice(&ids[start..start + count]);
        sequence.targets[..count].copy_from_slice(&labels[start..start + count]);
        for offset in 0..count {
            let active = mask.map_or(1.0, |mask| mask[start + offset]);
            sequence.attention_mask[offset] = active;
            if active > 0.0 && sequence.targets[offset] != u32::MAX {
                sequence.loss_weights[offset] =
                    weights.map_or(1.0, |weights| weights[start + offset]);
            }
        }
        if sequence.loss_weights.iter().any(|weight| *weight > 0.0) {
            output.push(sequence);
        }
    }
    Ok(())
}

fn push_shifted_chunks(
    output: &mut Vec<Sequence>,
    tokens: &[u32],
    seq_len: usize,
    pad_token_id: u32,
) {
    if tokens.len() < 2 {
        return;
    }
    let predict_count = tokens.len() - 1;
    for start in (0..predict_count).step_by(seq_len) {
        let count = (predict_count - start).min(seq_len);
        let mut sequence = Sequence {
            input_ids: vec![pad_token_id; seq_len],
            targets: vec![u32::MAX; seq_len],
            attention_mask: vec![0.0; seq_len],
            loss_weights: vec![0.0; seq_len],
            encoder_context: None,
        };
        sequence.input_ids[..count].copy_from_slice(&tokens[start..start + count]);
        sequence.targets[..count].copy_from_slice(&tokens[start + 1..start + 1 + count]);
        sequence.attention_mask[..count].fill(1.0);
        sequence.loss_weights[..count].fill(1.0);
        output.push(sequence);
    }
}

fn encode_text(text: &str, tokenizer: &Tokenizer, eos_token_id: Option<u32>) -> Result<Vec<u32>> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|error| anyhow::anyhow!("tokenizing dataset text: {error}"))?;
    let mut ids = encoding.get_ids().to_vec();
    if let Some(eos) = eos_token_id {
        if ids.last().copied() != Some(eos) {
            ids.push(eos);
        }
    }
    Ok(ids)
}

fn row_to_sequences(
    row: &Value,
    tokenizer: Option<&Tokenizer>,
    text_field: &str,
    seq_len: usize,
    pad_token_id: u32,
    eos_token_id: Option<u32>,
    causal_attention: bool,
    hidden_size: usize,
    output: &mut Vec<Sequence>,
) -> Result<()> {
    let encoder_context = row
        .as_object()
        .map(|object| parse_encoder_context(object, hidden_size))
        .transpose()?
        .flatten();
    let output_start = output.len();
    if let Some(object) = row.as_object() {
        if let Some(input_ids) = object.get("input_ids") {
            let ids = parse_u32_array(input_ids, "input_ids")?;
            let labels = object
                .get("labels")
                .or_else(|| object.get("targets"))
                .map(|value| parse_labels(value, "labels/targets"))
                .transpose()?;
            if let Some(labels) = labels {
                let mask = object
                    .get("attention_mask")
                    .map(|value| parse_mask(value, "attention_mask"))
                    .transpose()?;
                let weights = object
                    .get("loss_weights")
                    .map(|value| parse_mask(value, "loss_weights"))
                    .transpose()?;
                push_aligned_chunks(
                    output,
                    &ids,
                    &labels,
                    mask.as_deref(),
                    weights.as_deref(),
                    seq_len,
                    pad_token_id,
                )?;
                for sequence in &mut output[output_start..] {
                    sequence.encoder_context = encoder_context.clone();
                }
                return Ok(());
            }
            if !causal_attention {
                bail!(
                    "bidirectional Transformer training requires pretokenized input_ids with aligned labels/targets; prepare masked-language-model labels with -100 for unmasked tokens"
                );
            }
            push_shifted_chunks(output, &ids, seq_len, pad_token_id);
            for sequence in &mut output[output_start..] {
                sequence.encoder_context = encoder_context.clone();
            }
            return Ok(());
        }
    }
    if !causal_attention {
        bail!(
            "bidirectional Transformer training requires pretokenized input_ids with aligned labels/targets; raw text would imply an incorrect next-token objective"
        );
    }
    let text = match row {
        Value::String(text) => text.as_str(),
        _ => json_field(row, text_field)
            .and_then(Value::as_str)
            .with_context(|| {
                format!("dataset row has neither input_ids nor string field {text_field:?}")
            })?,
    };
    let tokenizer = tokenizer.context(
        "raw-text training requires tokenizer.json in the model package; use a model with a fast tokenizer or provide pretokenized input_ids",
    )?;
    let ids = encode_text(text, tokenizer, eos_token_id)?;
    push_shifted_chunks(output, &ids, seq_len, pad_token_id);
    for sequence in &mut output[output_start..] {
        sequence.encoder_context = encoder_context.clone();
    }
    Ok(())
}

fn load_json_values(path: &Path) -> Result<Vec<Value>> {
    let bytes = fs::read(path).with_context(|| format!("reading dataset {}", path.display()))?;
    let value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("decoding JSON dataset {}", path.display()))?;
    match value {
        Value::Array(rows) => Ok(rows),
        other => Ok(vec![other]),
    }
}

fn load_jsonl_values(path: &Path) -> Result<Vec<Value>> {
    let file = File::open(path).with_context(|| format!("opening dataset {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line =
            line.with_context(|| format!("reading {} line {}", path.display(), index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        rows.push(
            serde_json::from_str(&line)
                .with_context(|| format!("decoding {} line {}", path.display(), index + 1))?,
        );
    }
    Ok(rows)
}

fn load_text_values(path: &Path) -> Result<Vec<Value>> {
    let file = File::open(path).with_context(|| format!("opening dataset {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    for line in reader.lines() {
        let line = line.with_context(|| format!("reading dataset {}", path.display()))?;
        if !line.trim().is_empty() {
            rows.push(Value::String(line));
        }
    }
    Ok(rows)
}

fn load_dataset(
    path: &Path,
    tokenizer: Option<&Tokenizer>,
    text_field: &str,
    seq_len: usize,
    pad_token_id: u32,
    eos_token_id: Option<u32>,
    causal_attention: bool,
    hidden_size: usize,
) -> Result<(Vec<Sequence>, Option<usize>)> {
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let rows = if filename.ends_with(".jsonl") || filename.ends_with(".ndjson") {
        load_jsonl_values(path)?
    } else if filename.ends_with(".json") {
        load_json_values(path)?
    } else if filename.ends_with(".txt") || filename.ends_with(".text") {
        load_text_values(path)?
    } else {
        bail!(
            "unsupported dataset file {}; native Transformer trainer currently accepts .jsonl/.ndjson, .json, and .txt. For Hub datasets choose a JSON/text export or pretokenized JSONL file",
            path.display()
        );
    };
    let mut sequences = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        row_to_sequences(
            row,
            tokenizer,
            text_field,
            seq_len,
            pad_token_id,
            eos_token_id,
            causal_attention,
            hidden_size,
            &mut sequences,
        )
        .with_context(|| format!("preparing dataset row {}", index + 1))?;
    }
    if sequences.is_empty() {
        bail!("dataset produced no trainable sequences after tokenization/chunking");
    }
    let encoder_seq_len = sequences.iter().find_map(|sequence| {
        sequence
            .encoder_context
            .as_ref()
            .map(|context| context.seq_len)
    });
    if let Some(encoder_seq_len) = encoder_seq_len {
        for sequence in &sequences {
            let context = sequence.encoder_context.as_ref().context(
                "dataset mixes rows with encoder_hidden_states and rows without encoder context",
            )?;
            if context.seq_len != encoder_seq_len {
                bail!(
                    "encoder sequence length must be fixed across the Vulkan training dataset: expected {encoder_seq_len}, got {}",
                    context.seq_len
                );
            }
        }
    }
    Ok((sequences, encoder_seq_len))
}

fn default_lora_targets(architecture: VulkanTransformerArchitecture) -> Vec<String> {
    vec![architecture.default_lora_target_module().to_owned()]
}

fn batch_arrays(
    sequences: &[Sequence],
    seq_len: usize,
) -> Result<(
    Vec<u32>,
    Vec<u32>,
    Vec<f32>,
    Vec<f32>,
    Option<Vec<f32>>,
    Option<Vec<f32>>,
)> {
    let rows = sequences
        .len()
        .checked_mul(seq_len)
        .context("batch row count overflow")?;
    let mut input_ids = Vec::with_capacity(rows);
    let mut targets = Vec::with_capacity(rows);
    let mut attention_mask = Vec::with_capacity(rows);
    let mut loss_weights = Vec::with_capacity(rows);
    let has_encoder_context = sequences
        .first()
        .and_then(|sequence| sequence.encoder_context.as_ref())
        .is_some();
    let mut encoder_hidden_states = has_encoder_context.then(Vec::new);
    let mut encoder_attention_mask = has_encoder_context.then(Vec::new);
    for sequence in sequences {
        input_ids.extend_from_slice(&sequence.input_ids);
        targets.extend_from_slice(&sequence.targets);
        attention_mask.extend_from_slice(&sequence.attention_mask);
        loss_weights.extend_from_slice(&sequence.loss_weights);
        match (
            has_encoder_context,
            sequence.encoder_context.as_ref(),
            encoder_hidden_states.as_mut(),
            encoder_attention_mask.as_mut(),
        ) {
            (true, Some(context), Some(hidden), Some(mask)) => {
                hidden.extend_from_slice(&context.hidden_states);
                mask.extend_from_slice(&context.attention_mask);
            }
            (false, None, None, None) => {}
            _ => bail!("training batch mixes rows with and without encoder context"),
        }
    }
    let normalization = loss_weights.iter().copied().sum::<f32>();
    if !normalization.is_finite() || normalization <= 0.0 {
        bail!("batch has no finite positive supervised loss weight");
    }
    for weight in &mut loss_weights {
        *weight /= normalization;
    }
    Ok((
        input_ids,
        targets,
        attention_mask,
        loss_weights,
        encoder_hidden_states,
        encoder_attention_mask,
    ))
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let (model_dir, model_source) = if let Some(path) = args.model.as_ref() {
        (path.clone(), path.display().to_string())
    } else {
        let id = args
            .hf_model
            .as_deref()
            .context("missing Hugging Face model id")?;
        (
            download_hf_model(id, args.hf_model_revision.as_deref())?,
            format!("hf://{id}"),
        )
    };
    let (dataset_path, dataset_source) = if let Some(path) = args.dataset.as_ref() {
        (path.clone(), path.display().to_string())
    } else {
        let id = args
            .hf_dataset
            .as_deref()
            .context("missing Hugging Face dataset id")?;
        let file = args
            .hf_dataset_file
            .as_deref()
            .context("missing Hugging Face dataset filename")?;
        (
            download_hf_dataset_file(id, file, args.hf_dataset_revision.as_deref())?,
            format!("hf://datasets/{id}/{file}"),
        )
    };

    let model_config = VulkanTransformerConfig::from_hf_config(model_dir.join("config.json"))?;
    let causal_attention = model_config.architecture.uses_causal_attention();
    let tokenizer = load_tokenizer(&model_dir)?;
    let eos_token_id = args.eos_token_id.or_else(|| {
        tokenizer
            .as_ref()
            .and_then(|tokenizer| infer_special_token_id(&model_dir, tokenizer, "eos"))
    });
    let pad_token_id = args
        .pad_token_id
        .or_else(|| {
            tokenizer
                .as_ref()
                .and_then(|tokenizer| infer_special_token_id(&model_dir, tokenizer, "pad"))
        })
        .or(eos_token_id)
        .unwrap_or(0);
    let (sequences, encoder_seq_len) = load_dataset(
        &dataset_path,
        tokenizer.as_ref(),
        &args.text_field,
        args.seq_len,
        pad_token_id,
        eos_token_id,
        causal_attention,
        model_config.hidden_size,
    )?;
    if sequences.len() < args.batch_size {
        bail!(
            "dataset produced {} fixed-length sequences, fewer than --batch-size {}",
            sequences.len(),
            args.batch_size
        );
    }

    let device = match args.device_index {
        Some(index) => VulkanDevice::new_with_index(index)?,
        None => VulkanDevice::new()?,
    };
    let mut graph = if let Some(encoder_seq_len) = encoder_seq_len {
        VulkanTransformer::from_hf_package_with_encoder_seq_len(
            device,
            &model_dir,
            args.batch_size,
            args.seq_len,
            encoder_seq_len,
        )?
    } else {
        VulkanTransformer::from_hf_package(device, &model_dir, args.batch_size, args.seq_len)?
    };
    if graph.config().architecture != model_config.architecture {
        bail!(
            "model config changed architecture while constructing the Vulkan graph: parsed {:?}, constructed {:?}",
            model_config.architecture,
            graph.config().architecture
        );
    }
    match (graph.encoder_seq_len(), encoder_seq_len) {
        (Some(_), None) => bail!(
            "{} checkpoint contains cross-attention blocks; training rows must provide encoder_hidden_states (and optionally encoder_attention_mask) so cross-attention cannot be silently skipped",
            graph.config().architecture.model_type()
        ),
        (None, Some(_)) => bail!(
            "dataset provides encoder_hidden_states, but {} has no cross-attention encoder input",
            graph.config().architecture.model_type()
        ),
        _ => {}
    }
    graph.set_dropout_seed((args.seed as u32) ^ ((args.seed >> 32) as u32));
    if pad_token_id as usize >= graph.config().vocab_size {
        bail!(
            "padding token id {pad_token_id} is outside model vocabulary size {}",
            graph.config().vocab_size
        );
    }
    if let Some(eos) = eos_token_id {
        if eos as usize >= graph.config().vocab_size {
            bail!(
                "EOS token id {eos} is outside model vocabulary size {}",
                graph.config().vocab_size
            );
        }
    }

    if let Some(adapter) = args.lora_adapter.as_deref() {
        graph.load_lora_adapter(adapter)?;
    } else if let Some(rank) = args.lora_rank {
        let alpha = args.lora_alpha.unwrap_or(rank.saturating_mul(2));
        let targets = if args.lora_targets.is_empty() {
            default_lora_targets(graph.config().architecture)
        } else {
            args.lora_targets.clone()
        };
        let fan_in_fan_out = matches!(
            graph.config().architecture,
            VulkanTransformerArchitecture::OpenAiGpt
                | VulkanTransformerArchitecture::Gpt2
                | VulkanTransformerArchitecture::GptSw3
        );
        graph.enable_lora(
            VulkanTransformerLoraConfig::causal_lm(rank, alpha, targets, fan_in_fan_out)?,
            args.seed,
        )?;
    }
    let peft = graph.lora_config().is_some();
    let hyper = AdamWHyperParams {
        lr: args.learning_rate,
        beta1: args.beta1,
        beta2: args.beta2,
        eps: args.eps,
        weight_decay: args.weight_decay,
    };

    let batches_per_epoch = sequences.len() / args.batch_size;
    if batches_per_epoch == 0 {
        bail!("dataset does not contain a complete training batch");
    }
    let mut optimizer_steps = 0usize;
    let mut supervised_tokens = 0usize;
    let mut final_loss = None;
    'epochs: for epoch in 0..args.epochs {
        for batch_index in 0..batches_per_epoch {
            if args.max_steps.is_some_and(|limit| optimizer_steps >= limit) {
                break 'epochs;
            }
            let begin = batch_index * args.batch_size;
            let end = begin + args.batch_size;
            let (
                input_ids,
                targets,
                attention_mask,
                loss_weights,
                encoder_hidden_states,
                encoder_attention_mask,
            ) = batch_arrays(&sequences[begin..end], args.seq_len)?;
            let result = if let (Some(encoder_hidden_states), Some(encoder_attention_mask)) = (
                encoder_hidden_states.as_deref(),
                encoder_attention_mask.as_deref(),
            ) {
                graph.train_step_with_encoder_hidden_states(
                    &input_ids,
                    &targets,
                    &attention_mask,
                    &loss_weights,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    hyper,
                )?
            } else {
                graph.train_step(&input_ids, &targets, &attention_mask, &loss_weights, hyper)?
            };
            optimizer_steps += 1;
            supervised_tokens += result.supervised_tokens;
            final_loss = Some(result.loss);
            if optimizer_steps == 1 || optimizer_steps % args.log_steps == 0 {
                println!(
                    "{}",
                    serde_json::json!({
                        "event": "train_step",
                        "epoch": epoch + 1,
                        "epoch_batch": batch_index + 1,
                        "step": result.step,
                        "loss": result.loss,
                        "supervised_tokens": result.supervised_tokens,
                        "device": graph.device_name(),
                        "peft": peft,
                    })
                );
            }
        }
    }
    let final_loss = final_loss.context("training produced no optimizer steps")?;

    if peft {
        graph.export_lora_adapter(&args.output)?;
    } else {
        graph.export_hf_package(&model_dir, &args.output)?;
    }
    let report = TrainingReport {
        backend: "rust-native-vulkan-transformer",
        device: graph.device_name().to_owned(),
        architecture: graph.config().architecture.model_type().to_owned(),
        model_source,
        dataset_source,
        batch_size: args.batch_size,
        seq_len: args.seq_len,
        encoder_seq_len,
        epochs_requested: args.epochs,
        optimizer_steps,
        training_sequences: sequences.len(),
        supervised_tokens,
        final_loss,
        peft,
        output: args.output.display().to_string(),
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shifted_chunks_mask_padding_and_preserve_next_token_targets() {
        let mut output = Vec::new();
        push_shifted_chunks(&mut output, &[10, 11, 12, 13], 4, 0);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].input_ids, vec![10, 11, 12, 0]);
        assert_eq!(output[0].targets, vec![11, 12, 13, u32::MAX]);
        assert_eq!(output[0].attention_mask, vec![1.0, 1.0, 1.0, 0.0]);
        assert_eq!(output[0].loss_weights, vec![1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn aligned_chunks_keep_hf_minus_100_masking() -> Result<()> {
        let mut output = Vec::new();
        push_aligned_chunks(
            &mut output,
            &[1, 2, 3],
            &[u32::MAX, 2, 3],
            Some(&[1.0, 1.0, 1.0]),
            None,
            4,
            0,
        )?;
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].targets, vec![u32::MAX, 2, 3, u32::MAX]);
        assert_eq!(output[0].loss_weights, vec![0.0, 1.0, 1.0, 0.0]);
        Ok(())
    }

    #[test]
    fn aligned_chunks_preserve_explicit_loss_weights() -> Result<()> {
        let mut output = Vec::new();
        push_aligned_chunks(
            &mut output,
            &[1, 2, 3],
            &[2, 3, u32::MAX],
            Some(&[1.0, 1.0, 1.0]),
            Some(&[0.25, 2.0, 9.0]),
            4,
            0,
        )?;
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].loss_weights, vec![0.25, 2.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn batch_loss_weights_are_normalized_for_mean_cross_entropy() -> Result<()> {
        let sequence = Sequence {
            input_ids: vec![1, 2],
            targets: vec![2, 3],
            attention_mask: vec![1.0, 1.0],
            loss_weights: vec![1.0, 1.0],
            encoder_context: None,
        };
        let (_, _, _, weights, _, _) = batch_arrays(&[sequence.clone(), sequence], 2)?;
        assert_eq!(weights, vec![0.25; 4]);
        Ok(())
    }

    #[test]
    fn bidirectional_rows_require_explicit_aligned_labels() {
        let row = serde_json::json!({"input_ids": [10, 11, 12]});
        let mut output = Vec::new();
        let error = row_to_sequences(&row, None, "text", 4, 0, None, false, 4, &mut output)
            .expect_err("unlabeled bidirectional rows must not become next-token training data");
        assert!(error.to_string().contains("aligned labels/targets"));
        assert!(output.is_empty());
    }

    #[test]
    fn bidirectional_rows_accept_masked_lm_labels() -> Result<()> {
        let row = serde_json::json!({
            "input_ids": [10, 11, 12],
            "labels": [-100, 21, -100],
            "attention_mask": [1, 1, 1]
        });
        let mut output = Vec::new();
        row_to_sequences(&row, None, "text", 4, 0, None, false, 4, &mut output)?;
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].input_ids, vec![10, 11, 12, 0]);
        assert_eq!(output[0].targets, vec![u32::MAX, 21, u32::MAX, u32::MAX]);
        assert_eq!(output[0].loss_weights, vec![0.0, 1.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn encoder_context_accepts_nested_hidden_states_and_defaults_mask() -> Result<()> {
        let row = serde_json::json!({
            "input_ids": [10, 11],
            "labels": [11, 12],
            "encoder_hidden_states": [[0.1, 0.2], [0.3, 0.4]]
        });
        let mut output = Vec::new();
        row_to_sequences(&row, None, "text", 2, 0, None, true, 2, &mut output)?;
        let context = output[0]
            .encoder_context
            .as_ref()
            .expect("encoder context should be attached");
        assert_eq!(context.seq_len, 2);
        assert_eq!(context.hidden_states, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(context.attention_mask, vec![1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn encoder_context_rejects_hidden_width_mismatch() {
        let row = serde_json::json!({
            "input_ids": [10, 11],
            "labels": [11, 12],
            "encoder_hidden_states": [0.1, 0.2, 0.3]
        });
        let mut output = Vec::new();
        let error = row_to_sequences(&row, None, "text", 2, 0, None, true, 2, &mut output)
            .expect_err("encoder hidden width mismatch should fail");
        assert!(error.to_string().contains("positive multiple"));
    }
}
